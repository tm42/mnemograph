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


# ─────────────────────────────────────────────────────────────────────────────
# Priority 1: Tolerant Reader Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_read_all_skips_malformed_line():
    """Verify valid events still load when there's a malformed line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        # Write a valid event
        event1 = MemoryEvent(
            op="create_entity", session_id="test", data={"id": "e1"}
        )
        store.append(event1)

        # Manually write a malformed line
        with open(store.path, "a") as f:
            f.write("this is not valid json\n")

        # Write another valid event
        event2 = MemoryEvent(
            op="create_entity", session_id="test", data={"id": "e2"}
        )
        store.append(event2)

        # Should load both valid events, skip malformed
        events = store.read_all(tolerant=True)
        assert len(events) == 2
        assert events[0].id == event1.id
        assert events[1].id == event2.id


def test_read_all_strict_mode_raises():
    """Verify strict mode raises on malformed line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        # Write a valid event
        store.append(MemoryEvent(op="create_entity", session_id="test", data={}))

        # Write malformed line
        with open(store.path, "a") as f:
            f.write("not json\n")

        # Strict mode should raise
        import pytest

        with pytest.raises(ValueError, match="Malformed event at line 2"):
            store.read_all(tolerant=False)


def test_read_all_blank_lines_ignored():
    """Verify blank lines don't cause issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        event = MemoryEvent(op="create_entity", session_id="test", data={})
        store.append(event)

        # Add blank lines
        with open(store.path, "a") as f:
            f.write("\n\n\n")

        store.append(MemoryEvent(op="create_entity", session_id="test", data={}))

        events = store.read_all()
        assert len(events) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Priority 2 & 3: Durable Append and Process Lock Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_append_creates_parent_dirs():
    """Verify directory creation works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "dirs" / "events.jsonl"
        store = EventStore(nested_path)

        event = MemoryEvent(op="create_entity", session_id="test", data={})
        store.append(event)

        assert nested_path.exists()
        assert store.count() == 1


def test_append_batch_writes_all_events():
    """Verify batch writes complete atomically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

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
        store = EventStore(Path(tmpdir) / "events.jsonl")
        result = store.append_batch([])
        assert result == []
        assert store.count() == 0


def test_concurrent_append_blocked():
    """Verify concurrent writes are blocked (Unix only)."""
    import sys

    if sys.platform == "win32":
        import pytest

        pytest.skip("File locking not available on Windows")

    import threading
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        # Hold the lock for a bit
        lock_held = threading.Event()
        lock_released = threading.Event()
        blocked = []

        def hold_lock():
            with store._write_lock():
                lock_held.set()
                time.sleep(0.2)
            lock_released.set()

        def try_write():
            lock_held.wait()  # Wait for first thread to acquire lock
            try:
                store.append(MemoryEvent(op="create_entity", session_id="t2", data={}))
            except RuntimeError as e:
                blocked.append(str(e))

        t1 = threading.Thread(target=hold_lock)
        t2 = threading.Thread(target=try_write)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # t2 should have been blocked
        assert len(blocked) == 1
        assert "Another process" in blocked[0]


def test_lock_released_after_exception():
    """Verify lock is released even when exception occurs."""
    import sys

    if sys.platform == "win32":
        import pytest

        pytest.skip("File locking not available on Windows")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir) / "events.jsonl")

        # Force an exception inside the lock
        try:
            with store._write_lock():
                raise ValueError("test error")
        except ValueError:
            pass

        # Lock should be released, so this should work
        with store._write_lock():
            pass  # Should not block
