"""Tests for the SQLite event store."""

import tempfile
from datetime import datetime, timedelta, timezone
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


# ─────────────────────────────────────────────────────────────────────────────
# D17: read_between() SQL Range Query Tests
# ─────────────────────────────────────────────────────────────────────────────


def _make_timed_events(store, count=10, base_ts=None):
    """Create events with known timestamps spaced 1 hour apart."""
    if base_ts is None:
        base_ts = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    events = []
    for i in range(count):
        ts = base_ts + timedelta(hours=i)
        ev = MemoryEvent(
            ts=ts,
            op="create_entity",
            session_id="test",
            data={"id": f"e{i}", "name": f"entity-{i}"},
        )
        store.append(ev)
        events.append(ev)
    return events, base_ts


def test_read_between_returns_correct_range():
    """read_between should return only events in the given time window."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events, base = _make_timed_events(store, count=10)

        # Query hours 3-6 (inclusive)
        start = base + timedelta(hours=3)
        end = base + timedelta(hours=6)
        result = store.read_between(start, end)

        result_ids = [e.data["id"] for e in result]
        assert result_ids == ["e3", "e4", "e5", "e6"]


def test_read_between_inclusive_boundaries():
    """Events at exact start and end timestamps should be included."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events, base = _make_timed_events(store, count=5)

        # Query with exact boundary timestamps
        start = events[1].ts  # exact ts of e1
        end = events[3].ts  # exact ts of e3
        result = store.read_between(start, end)

        result_ids = [e.data["id"] for e in result]
        assert "e1" in result_ids
        assert "e3" in result_ids
        assert len(result) == 3  # e1, e2, e3


def test_read_between_empty_range():
    """Querying a range with no events should return empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events, base = _make_timed_events(store, count=5)

        # Query a gap between events (e.g., 30 minutes after e0, before e1)
        start = base + timedelta(minutes=30)
        end = base + timedelta(minutes=45)
        result = store.read_between(start, end)

        assert result == []


def test_read_between_empty_store():
    """Querying an empty store should return empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 31, tzinfo=timezone.utc)
        result = store.read_between(start, end)
        assert result == []


def test_read_between_matches_read_all_filter():
    """read_between results should match manual read_all + filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events, base = _make_timed_events(store, count=10)

        start = base + timedelta(hours=2)
        end = base + timedelta(hours=7)

        # SQL-based
        sql_result = store.read_between(start, end)

        # Python filter-based (old behavior)
        all_events = store.read_all()
        py_result = [e for e in all_events if start <= e.ts <= end]

        assert len(sql_result) == len(py_result)
        assert [e.id for e in sql_result] == [e.id for e in py_result]


def test_read_between_ordered_by_ts_then_id():
    """Results should be ordered by timestamp, then by ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events, base = _make_timed_events(store, count=5)

        start = base
        end = base + timedelta(hours=4)
        result = store.read_between(start, end)

        # Verify ordering
        for i in range(len(result) - 1):
            assert result[i].ts <= result[i + 1].ts


def test_events_between_integration():
    """Integration test: TimeTraveler.events_between delegates to read_between."""
    from mnemograph.time_travel import TimeTraveler

    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "mnemograph.db")
        events, base = _make_timed_events(store, count=5)

        traveler = TimeTraveler(lambda: store)

        start = base + timedelta(hours=1)
        end = base + timedelta(hours=3)
        result = traveler.events_between(start, end)

        assert len(result) == 3  # e1, e2, e3
        assert result[0].data["id"] == "e1"
        assert result[2].data["id"] == "e3"
