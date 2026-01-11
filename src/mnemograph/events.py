"""Append-only event store.

The event log is the source of truth. State is derived by replaying events.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from .models import MemoryEvent

logger = logging.getLogger(__name__)

# Platform-specific locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False  # Windows fallback


class EventStore:
    """Append-only event log backed by a JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock_path = path.with_suffix(".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _write_lock(self):
        """Acquire exclusive write lock (Unix only, no-op on Windows)."""
        if not HAS_FCNTL:
            # Windows: skip locking with warning on first use
            yield
            return

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._lock_path, "w") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
            except BlockingIOError:
                raise RuntimeError(
                    "Another process is writing to the event log. "
                    "Wait for it to complete or check for stale locks."
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def append(self, event: MemoryEvent, durable: bool = True) -> MemoryEvent:
        """Append event to log with durability guarantees.

        Args:
            event: Event to append
            durable: If True, fsync after write. Set False for batch operations.
        """
        with self._write_lock():
            # Serialize once, write atomically
            line = event.model_dump_json() + "\n"

            with open(self.path, "a") as f:
                f.write(line)
                if durable:
                    f.flush()
                    os.fsync(f.fileno())

        return event

    def append_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events with single fsync at end."""
        if not events:
            return events

        with self._write_lock():
            with open(self.path, "a") as f:
                for event in events:
                    f.write(event.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

        return events

    def read_all(self, tolerant: bool = True) -> list[MemoryEvent]:
        """Read all events from log.

        Args:
            tolerant: If True, skip malformed lines with warnings.
                      If False, raise on first error (strict mode).

        Returns:
            List of valid events. Malformed lines are skipped in tolerant mode.
        """
        if not self.path.exists():
            return []

        events = []
        errors = []

        with open(self.path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip blank lines

                try:
                    events.append(MemoryEvent.model_validate_json(line))
                except Exception as e:
                    if tolerant:
                        errors.append((line_num, str(e), line[:100]))
                        logger.warning(
                            f"Skipping malformed event at line {line_num}: {e}"
                        )
                    else:
                        raise ValueError(
                            f"Malformed event at line {line_num}: {e}"
                        ) from e

        if errors:
            logger.warning(
                f"Loaded {len(events)} events, skipped {len(errors)} malformed lines"
            )

        return events

    def read_since(self, since_id: str | None = None) -> list[MemoryEvent]:
        """Read events after a given event ID (for incremental replay)."""
        events = self.read_all()
        if since_id is None:
            return events
        for i, event in enumerate(events):
            if event.id == since_id:
                return events[i + 1:]
        return events  # ID not found, return all

    def count(self) -> int:
        """Count events without loading all into memory."""
        if not self.path.exists():
            return 0
        with open(self.path, "r") as f:
            return sum(1 for line in f if line.strip())
