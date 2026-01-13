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
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
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
        from .models import MemoryEvent as ME  # Avoid circular import

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

    def append_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events with single commit at end."""
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
        from .models import MemoryEvent

        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id, ts, op, session_id, source, data FROM events ORDER BY ts, id"
        )

        events = []
        errors = []

        for row in cursor:
            try:
                event = MemoryEvent(
                    id=row["id"],
                    ts=row["ts"],
                    op=row["op"],
                    session_id=row["session_id"],
                    source=row["source"],
                    data=json.loads(row["data"]),
                )
                events.append(event)
            except Exception as e:
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

        from .models import MemoryEvent

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

        events = []
        for row in cursor:
            event = MemoryEvent(
                id=row["id"],
                ts=row["ts"],
                op=row["op"],
                session_id=row["session_id"],
                source=row["source"],
                data=json.loads(row["data"]),
            )
            events.append(event)

        return events

    def read_by_session(self, session_id: str) -> list[MemoryEvent]:
        """Read all events for a specific session."""
        from .models import MemoryEvent

        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT id, ts, op, session_id, source, data FROM events
            WHERE session_id = ?
            ORDER BY ts, id
            """,
            (session_id,),
        )

        events = []
        for row in cursor:
            event = MemoryEvent(
                id=row["id"],
                ts=row["ts"],
                op=row["op"],
                session_id=row["session_id"],
                source=row["source"],
                data=json.loads(row["data"]),
            )
            events.append(event)

        return events

    def count(self) -> int:
        """Count events."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def get_connection(self) -> sqlite3.Connection:
        """Get the database connection for sharing with other components."""
        return self._get_conn()

    def clear(self) -> None:
        """Clear all events from the store.

        Used for compaction and testing.
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM events")
        conn.commit()
