"""Tests for the event store."""

import tempfile
from pathlib import Path

from mnemograph.events import EventStore
from mnemograph.models import MemoryEvent


def test_append_and_read():
    """Test appending and reading events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        event = MemoryEvent(
            op="create_entity",
            session_id="test-session",
            source="cc",
            data={"id": "e1", "name": "test", "type": "concept"},
        )
        store.append(event)

        events = store.read_all()
        assert len(events) == 1
        assert events[0].id == event.id
        assert events[0].op == "create_entity"
        assert events[0].data["name"] == "test"


def test_read_empty():
    """Test reading from non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")
        events = store.read_all()
        assert events == []


def test_read_since():
    """Test reading events since a given ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        events = []
        for i in range(5):
            event = MemoryEvent(
                op="create_entity",
                session_id="test",
                data={"id": f"e{i}", "name": f"entity{i}"},
            )
            store.append(event)
            events.append(event)

        # Read since the 3rd event (index 2)
        since_events = store.read_since(events[2].id)
        assert len(since_events) == 2
        assert since_events[0].data["id"] == "e3"
        assert since_events[1].data["id"] == "e4"


def test_count():
    """Test counting events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        assert store.count() == 0

        for i in range(3):
            store.append(
                MemoryEvent(op="create_entity", session_id="test", data={"id": f"e{i}"})
            )

        assert store.count() == 3
