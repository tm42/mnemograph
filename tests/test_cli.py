"""Tests for CLI commands."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_engine.cli import cmd_log, cmd_revert, cmd_status, cmd_sessions, parse_since, main
from memory_engine.engine import MemoryEngine
from memory_engine.events import EventStore
from memory_engine.state import materialize


class Args:
    """Simple namespace for test args."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_parse_since_hours():
    """Test parsing '1 hour ago'."""
    now = datetime.now(timezone.utc)
    result = parse_since("1 hour ago")
    assert (now - result).total_seconds() < 3700  # ~1 hour with some slack


def test_parse_since_days():
    """Test parsing '2 days ago'."""
    now = datetime.now(timezone.utc)
    result = parse_since("2 days ago")
    diff = now - result
    assert 1.9 < diff.days + diff.seconds / 86400 < 2.1


def test_parse_since_today():
    """Test parsing 'today'."""
    result = parse_since("today")
    now = datetime.now(timezone.utc)
    assert result.date() == now.date()
    assert result.hour == 0 and result.minute == 0


def test_cmd_status_empty():
    """Test status command with empty memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        args = Args(memory_path=tmpdir)
        result = cmd_status(args)
        assert result == 0  # Returns 0 even with empty memory (just shows zeros)


def test_cmd_status_with_data():
    """Test status command with some data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "Entity A", "entityType": "concept", "observations": ["Obs 1"]},
            {"name": "Entity B", "entityType": "decision", "observations": ["Obs 2"]},
        ])

        args = Args(memory_path=tmpdir)
        result = cmd_status(args)
        assert result == 0


def test_cmd_log_empty():
    """Test log command with no events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        # Create empty events file
        (Path(tmpdir) / "events.jsonl").touch()

        args = Args(
            memory_path=tmpdir,
            since=None,
            session=None,
            op=None,
            limit=None,
            asc=False,
            json=False,
        )
        result = cmd_log(args)
        assert result == 0


def test_cmd_log_with_data():
    """Test log command with events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "Entity A", "entityType": "concept", "observations": ["Obs 1"]},
        ])

        args = Args(
            memory_path=tmpdir,
            since=None,
            session=None,
            op=None,
            limit=None,
            asc=False,
            json=False,
        )
        result = cmd_log(args)
        assert result == 0


def test_cmd_log_filter_session():
    """Test log command filtering by session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine1 = MemoryEngine(Path(tmpdir), "session-1")
        engine1.create_entities([{"name": "Entity A", "entityType": "concept"}])

        engine2 = MemoryEngine(Path(tmpdir), "session-2")
        engine2.create_entities([{"name": "Entity B", "entityType": "concept"}])

        args = Args(
            memory_path=tmpdir,
            since=None,
            session="session-1",
            op=None,
            limit=None,
            asc=False,
            json=False,
        )
        result = cmd_log(args)
        assert result == 0


def test_cmd_log_filter_op():
    """Test log command filtering by operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([{"name": "Entity A", "entityType": "concept"}])
        engine.create_relations([{"from": "Entity A", "to": "Entity A", "relationType": "self"}])

        args = Args(
            memory_path=tmpdir,
            since=None,
            session=None,
            op="create_entity",
            limit=None,
            asc=False,
            json=False,
        )
        result = cmd_log(args)
        assert result == 0


def test_cmd_sessions():
    """Test sessions command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine1 = MemoryEngine(Path(tmpdir), "session-1")
        engine1.create_entities([{"name": "Entity A", "entityType": "concept"}])

        engine2 = MemoryEngine(Path(tmpdir), "session-2")
        engine2.create_entities([{"name": "Entity B", "entityType": "concept"}])

        args = Args(memory_path=tmpdir)
        result = cmd_sessions(args)
        assert result == 0


def test_cmd_revert_session():
    """Test reverting all events from a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "Entity A", "entityType": "concept", "observations": ["Obs 1"]},
        ])

        # Verify entity exists
        event_store = EventStore(Path(tmpdir) / "events.jsonl")
        events_before = event_store.read_all()
        state_before = materialize(events_before)
        assert len(state_before.entities) == 1

        # Revert the session
        args = Args(
            memory_path=tmpdir,
            session="test-session",
            event_ids=[],
            yes=True,  # Skip confirmation
        )
        result = cmd_revert(args)
        assert result == 0

        # Verify entity is gone after replay
        events_after = event_store.read_all()
        state_after = materialize(events_after)
        assert len(state_after.entities) == 0


def test_cmd_revert_specific_event():
    """Test reverting a specific event by ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "Entity A", "entityType": "concept"},
            {"name": "Entity B", "entityType": "concept"},
        ])

        event_store = EventStore(Path(tmpdir) / "events.jsonl")
        events = event_store.read_all()
        # Get the first create_entity event
        first_event = events[0]

        # Revert just that event (use full ID to avoid prefix collision
        # since ULIDs created in same millisecond share prefix)
        args = Args(
            memory_path=tmpdir,
            session=None,
            event_ids=[first_event.id],  # Use full ID
            yes=True,
        )
        result = cmd_revert(args)
        assert result == 0

        # Verify only first entity is gone
        events_after = event_store.read_all()
        state_after = materialize(events_after)
        assert len(state_after.entities) == 1
        assert "Entity B" in [e.name for e in state_after.entities.values()]


def test_main_entry_point():
    """Test main CLI entry point."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test")
        engine.create_entities([{"name": "Test", "entityType": "concept"}])

        result = main(["--memory-path", tmpdir, "status"])
        assert result == 0
