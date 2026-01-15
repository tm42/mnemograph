"""Tests for CLI commands."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from click.testing import CliRunner

from mnemograph.cli import cli, parse_since
from mnemograph.engine import MemoryEngine
from mnemograph.events import EventStore
from mnemograph.state import materialize


runner = CliRunner()


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


def test_status_empty():
    """Test status command with empty memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, ["--memory-path", tmpdir, "status"])
        assert result.exit_code == 0


def test_status_with_data():
    """Test status command with some data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept", "observations": ["Obs 1"]},
            {"name": "Redis Cache System", "entityType": "decision", "observations": ["Obs 2"]},
        ])
        engine.event_store.close()

        result = runner.invoke(cli, ["--memory-path", tmpdir, "status"])
        assert result.exit_code == 0
        assert "2" in result.output or "Entities" in result.output


def test_log_empty():
    """Test log command with no events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli, ["--memory-path", tmpdir, "log"])
        assert result.exit_code == 0


def test_log_with_data():
    """Test log command with events."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept", "observations": ["Obs 1"]},
        ])
        engine.event_store.close()

        result = runner.invoke(cli, ["--memory-path", tmpdir, "log"])
        assert result.exit_code == 0
        assert "create_entity" in result.output or "PostgreSQL Database" in result.output


def test_log_filter_session():
    """Test log command filtering by session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine1 = MemoryEngine(Path(tmpdir), "session-1")
        engine1.create_entities([{"name": "PostgreSQL Database", "entityType": "concept"}])
        engine1.event_store.close()

        engine2 = MemoryEngine(Path(tmpdir), "session-2")
        engine2.create_entities([{"name": "Redis Cache System", "entityType": "concept"}])
        engine2.event_store.close()

        result = runner.invoke(cli, ["--memory-path", tmpdir, "log", "--session", "session-1"])
        assert result.exit_code == 0


def test_log_filter_op():
    """Test log command filtering by operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([{"name": "PostgreSQL Database", "entityType": "concept"}])
        engine.create_relations([{"from": "PostgreSQL Database", "to": "PostgreSQL Database", "relationType": "self"}])
        engine.event_store.close()

        result = runner.invoke(cli, ["--memory-path", tmpdir, "log", "--op", "create_entity"])
        assert result.exit_code == 0


def test_sessions():
    """Test sessions command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine1 = MemoryEngine(Path(tmpdir), "session-1")
        engine1.create_entities([{"name": "PostgreSQL Database", "entityType": "concept"}])
        engine1.event_store.close()

        engine2 = MemoryEngine(Path(tmpdir), "session-2")
        engine2.create_entities([{"name": "Redis Cache System", "entityType": "concept"}])
        engine2.event_store.close()

        result = runner.invoke(cli, ["--memory-path", tmpdir, "sessions"])
        assert result.exit_code == 0


def test_revert_session():
    """Test reverting all events from a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept", "observations": ["Obs 1"]},
        ])
        engine.event_store.close()

        # Verify entity exists
        event_store = EventStore(Path(tmpdir) / "mnemograph.db")
        events_before = event_store.read_all()
        state_before = materialize(events_before)
        assert len(state_before.entities) == 1
        event_store.close()

        # Revert the session (--yes to skip confirmation)
        result = runner.invoke(cli, [
            "--memory-path", tmpdir,
            "vcs", "revert",
            "--session", "test-session",
            "--yes"
        ])
        assert result.exit_code == 0

        # Verify entity is gone after replay
        event_store2 = EventStore(Path(tmpdir) / "mnemograph.db")
        events_after = event_store2.read_all()
        state_after = materialize(events_after)
        assert len(state_after.entities) == 0
        event_store2.close()


def test_revert_specific_event():
    """Test reverting a specific event by ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")
        # Use distinct names to avoid semantic similarity blocking
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept"},
            {"name": "Redis Cache System", "entityType": "concept"},
        ])
        engine.event_store.close()

        event_store = EventStore(Path(tmpdir) / "mnemograph.db")
        events = event_store.read_all()
        # Get the first create_entity event
        first_event = events[0]
        event_store.close()

        # Revert just that event (use full ID to avoid prefix collision)
        result = runner.invoke(cli, [
            "--memory-path", tmpdir,
            "vcs", "revert",
            "--event", first_event.id,
            "--yes"
        ])
        assert result.exit_code == 0

        # Verify only first entity is gone
        event_store2 = EventStore(Path(tmpdir) / "mnemograph.db")
        events_after = event_store2.read_all()
        state_after = materialize(events_after)
        assert len(state_after.entities) == 1
        assert "Redis Cache System" in [e.name for e in state_after.entities.values()]
        event_store2.close()


def test_main_entry_point():
    """Test CLI entry point via status command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test")
        engine.create_entities([{"name": "Test", "entityType": "concept"}])
        engine.event_store.close()

        result = runner.invoke(cli, ["--memory-path", tmpdir, "status"])
        assert result.exit_code == 0
