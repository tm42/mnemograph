"""Tests for the SQLite event store."""

import tempfile
from pathlib import Path

from mnemograph.events import EventStore
from mnemograph.models import MemoryEvent


def test_append_and_read():
    """Test appending and reading events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")

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
    """Test reading from fresh database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events = store.read_all()
        assert events == []


def test_read_since():
    """Test reading events since a given ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")

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
        store = EventStore(Path(tmpdir) / "mnemograph.db")

        assert store.count() == 0

        for i in range(3):
            store.append(
                MemoryEvent(op="create_entity", session_id="test", data={"id": f"e{i}"})
            )

        assert store.count() == 3


def test_read_by_session():
    """Test reading events filtered by session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")

        # Create events from different sessions
        store.append(MemoryEvent(op="create_entity", session_id="session1", data={"id": "e1"}))
        store.append(MemoryEvent(op="create_entity", session_id="session2", data={"id": "e2"}))
        store.append(MemoryEvent(op="create_entity", session_id="session1", data={"id": "e3"}))

        # Read only session1 events
        session1_events = store.read_by_session("session1")
        assert len(session1_events) == 2
        assert all(e.session_id == "session1" for e in session1_events)


def test_append_creates_parent_dirs():
    """Verify directory creation works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "dirs" / "mnemograph.db"
        store = EventStore(nested_path)

        event = MemoryEvent(op="create_entity", session_id="test", data={})
        store.append(event)

        assert nested_path.exists()
        assert store.count() == 1


def test_append_batch_writes_all_events():
    """Verify batch writes complete atomically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")

        events = [
            MemoryEvent(op="create_entity", session_id="test", data={"id": f"e{i}"})
            for i in range(5)
        ]
        store.append_batch(events)

        loaded = store.read_all()
        assert len(loaded) == 5
        for i, event in enumerate(loaded):
            assert event.data["id"] == f"e{i}"


def test_append_batch_empty_list():
    """Verify batch with empty list is a no-op."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        result = store.append_batch([])
        assert result == []
        assert store.count() == 0


def test_clear():
    """Test clearing all events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")

        # Add some events
        for i in range(3):
            store.append(
                MemoryEvent(op="create_entity", session_id="test", data={"id": f"e{i}"})
            )
        assert store.count() == 3

        # Clear
        store.clear()
        assert store.count() == 0


def test_get_connection_returns_same_connection():
    """Verify get_connection returns the same connection instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")

        conn1 = store.get_connection()
        conn2 = store.get_connection()
        assert conn1 is conn2


def test_close_and_reopen():
    """Verify data persists after close and reopen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "mnemograph.db"

        # Write and close
        store1 = EventStore(db_path)
        store1.append(MemoryEvent(op="create_entity", session_id="test", data={"id": "e1"}))
        store1.close()

        # Reopen and read
        store2 = EventStore(db_path)
        events = store2.read_all()
        assert len(events) == 1
        assert events[0].data["id"] == "e1"
