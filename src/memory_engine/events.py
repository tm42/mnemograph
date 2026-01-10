"""Append-only event store.

The event log is the source of truth. State is derived by replaying events.
"""

from pathlib import Path

from .models import MemoryEvent


class EventStore:
    """Append-only event log backed by a JSONL file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: MemoryEvent) -> MemoryEvent:
        """Append event to log. Returns the event."""
        with open(self.path, "a") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    def read_all(self) -> list[MemoryEvent]:
        """Read all events from log."""
        if not self.path.exists():
            return []
        events = []
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    events.append(MemoryEvent.model_validate_json(line))
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
