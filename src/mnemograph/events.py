"""Append-only event store backed by SQLite.

The event log is the source of truth. State is derived by replaying events.
SQLite prevents agents from reading raw data via cat/Read.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import MemoryEvent

logger = logging.getLogger(__name__)


class EventStore:
    """Append-only event log backed by SQLite."""

    def __init__(self, db_path: Path):
        """Initialize event store.

        Args:
            db_path: Path to mnemograph.db (shared with vectors)
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
        return self._conn

    def _row_to_event(self, row: sqlite3.Row) -> "MemoryEvent":
        """Convert database row to MemoryEvent."""
        from .models import MemoryEvent

        return MemoryEvent(
            id=row["id"],
            ts=row["ts"],
            op=row["op"],
            session_id=row["session_id"],
            source=row["source"],
            data=json.loads(row["data"]),
        )

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()

        # Create schema version table for migration detection
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Check/set schema version
        version = conn.execute("SELECT version FROM schema_version").fetchone()
        if version is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        elif version[0] < 1:
            # Future: Add migration logic here
            logger.warning(f"Schema version {version[0]} detected, may need migration")

        # Create events table
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                op TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_op ON events(op);
        """)
        conn.commit()

    def append(self, event: MemoryEvent, durable: bool = True) -> MemoryEvent:
        """Append event to log.

        Args:
            event: Event to append
            durable: If True, commit immediately. Set False for batch operations.
        """

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO events (id, ts, op, session_id, source, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.ts.isoformat() if hasattr(event.ts, 'isoformat') else str(event.ts),
                event.op,
                event.session_id,
                event.source,
                json.dumps(event.data),
            ),
        )
        if durable:
            conn.commit()
        return event

    def append_batch(
        self, events: list[MemoryEvent], commit: bool = True
    ) -> list[MemoryEvent]:
        """Append multiple events with single commit at end.

        Args:
            events: Events to append
            commit: If True, commit immediately. Set False when called within
                   an external transaction (e.g., compaction).
        """
        if not events:
            return events

        conn = self._get_conn()
        for event in events:
            conn.execute(
                """
                INSERT INTO events (id, ts, op, session_id, source, data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.ts.isoformat() if hasattr(event.ts, 'isoformat') else str(event.ts),
                    event.op,
                    event.session_id,
                    event.source,
                    json.dumps(event.data),
                ),
            )
        if commit:
            conn.commit()
        return events

    def read_all(self, tolerant: bool = True) -> list[MemoryEvent]:
        """Read all events from log.

        Args:
            tolerant: If True, skip malformed rows with warnings.
                      If False, raise on first error (strict mode).

        Returns:
            List of valid events.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id, ts, op, session_id, source, data FROM events ORDER BY ts, id"
        )

        events = []
        errors = []

        for row in cursor:
            try:
                event = self._row_to_event(row)
                events.append(event)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if tolerant:
                    errors.append((row["id"], str(e)))
                    logger.warning(f"Skipping malformed event {row['id']}: {e}")
                else:
                    raise ValueError(f"Malformed event {row['id']}: {e}") from e

        if errors:
            logger.warning(
                f"Loaded {len(events)} events, skipped {len(errors)} malformed rows"
            )

        return events

    def read_since(self, since_id: str | None = None) -> list[MemoryEvent]:
        """Read events after a given event ID (for incremental replay)."""
        if since_id is None:
            return self.read_all()

        conn = self._get_conn()

        # Get the timestamp of the since_id event
        cursor = conn.execute("SELECT ts FROM events WHERE id = ?", (since_id,))
        row = cursor.fetchone()
        if row is None:
            return self.read_all()  # ID not found, return all

        since_ts = row["ts"]

        # Get all events after that timestamp (or same ts but later id)
        cursor = conn.execute(
            """
            SELECT id, ts, op, session_id, source, data FROM events
            WHERE (ts > ?) OR (ts = ? AND id > ?)
            ORDER BY ts, id
            """,
            (since_ts, since_ts, since_id),
        )

        return [self._row_to_event(row) for row in cursor]

    def read_by_session(self, session_id: str) -> list[MemoryEvent]:
        """Read all events for a specific session."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, ts, op, session_id, source, data FROM events
            WHERE session_id = ?
            ORDER BY ts, id
            """,
            (session_id,),
        )

        return [self._row_to_event(row) for row in cursor]

    def count(self) -> int:
        """Count events."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection.

        Forces a WAL checkpoint before closing to ensure all changes
        are written to the main database file.
        """
        if self._conn is not None:
            # Force checkpoint to merge WAL into main db
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            self._conn = None

    def get_connection(self) -> sqlite3.Connection:
        """Get the database connection for sharing with other components."""
        return self._get_conn()

    def clear(self, commit: bool = True) -> None:
        """Clear all events from the store.

        Args:
            commit: If True, commit immediately. Set False when called within
                   an external transaction (e.g., compaction).

        Used for compaction and testing.
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM events")
        if commit:
            conn.commit()
