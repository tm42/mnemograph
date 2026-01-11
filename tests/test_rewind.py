"""Tests for Event Rewind feature.

Tests materialize_at(), time parsing, and diff functions.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mnemograph.models import MemoryEvent
from mnemograph.state import materialize, materialize_at
from mnemograph.timeutil import parse_time_reference, format_relative_time


# --- Time Parsing Tests ---


class TestParseTimeReference:
    """Tests for parse_time_reference()."""

    def test_iso_date(self):
        """Should parse ISO date format."""
        result = parse_time_reference("2025-01-15")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_iso_datetime(self):
        """Should parse ISO datetime format."""
        result = parse_time_reference("2025-01-15T14:30:00")
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30

    def test_days_ago(self):
        """Should parse 'N days ago'."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = parse_time_reference("7 days ago", now=now)
        expected = now - timedelta(days=7)
        assert result == expected

    def test_weeks_ago(self):
        """Should parse 'N weeks ago'."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = parse_time_reference("2 weeks ago", now=now)
        expected = now - timedelta(weeks=2)
        assert result == expected

    def test_months_ago(self):
        """Should parse 'N months ago'."""
        now = datetime(2025, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = parse_time_reference("1 month ago", now=now)
        assert result.month == 2
        assert result.day == 15

    def test_yesterday(self):
        """Should parse 'yesterday'."""
        now = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = parse_time_reference("yesterday", now=now)
        assert result.day == 14
        assert result.hour == 0  # Start of day

    def test_last_week(self):
        """Should parse 'last week'."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = parse_time_reference("last week", now=now)
        expected = now - timedelta(weeks=1)
        assert result == expected

    def test_today(self):
        """Should parse 'today'."""
        now = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = parse_time_reference("today", now=now)
        assert result.day == 15
        assert result.hour == 0

    def test_invalid_raises_value_error(self):
        """Should raise ValueError for unparseable input."""
        with pytest.raises(ValueError):
            parse_time_reference("not a valid time")


class TestFormatRelativeTime:
    """Tests for format_relative_time()."""

    def test_seconds_ago(self):
        """Should format seconds ago."""
        now = datetime(2025, 1, 15, 12, 0, 30, tzinfo=timezone.utc)
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = format_relative_time(dt, now=now)
        assert "30 seconds ago" in result

    def test_minutes_ago(self):
        """Should format minutes ago."""
        now = datetime(2025, 1, 15, 12, 5, 0, tzinfo=timezone.utc)
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = format_relative_time(dt, now=now)
        assert "5 minutes ago" in result

    def test_hours_ago(self):
        """Should format hours ago."""
        now = datetime(2025, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = format_relative_time(dt, now=now)
        assert "3 hours ago" in result

    def test_days_ago(self):
        """Should format days ago."""
        now = datetime(2025, 1, 18, 12, 0, 0, tzinfo=timezone.utc)
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = format_relative_time(dt, now=now)
        assert "3 days ago" in result


# --- Materialize At Tests ---


def make_event(id: str, op: str, data: dict, ts: datetime) -> MemoryEvent:
    """Helper to create test events."""
    return MemoryEvent(
        id=id,
        ts=ts,
        op=op,
        session_id="test",
        data=data,
    )


def make_entity_data(id: str, name: str, ts: datetime) -> dict:
    """Helper to create entity data."""
    return {
        "id": id,
        "name": name,
        "type": "concept",
        "observations": [],
        "created_at": ts.isoformat(),
        "updated_at": ts.isoformat(),
        "created_by": "test",
        "access_count": 0,
    }


class TestMaterializeAt:
    """Tests for materialize_at()."""

    def test_empty_at_beginning(self):
        """State before any events should be empty."""
        ts_base = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts_event = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts_event), ts_event),
        ]

        # Query before the event
        state = materialize_at(events, ts_base)
        assert len(state.entities) == 0

    def test_includes_events_up_to_timestamp(self):
        """Should include events at or before timestamp."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "First", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "Second", ts2), ts2),
            make_event("ev3", "create_entity", make_entity_data("e3", "Third", ts3), ts3),
        ]

        # Query at ts2 should include e1 and e2
        state = materialize_at(events, ts2)
        assert len(state.entities) == 2
        assert "e1" in state.entities
        assert "e2" in state.entities
        assert "e3" not in state.entities

    def test_excludes_events_after_timestamp(self):
        """Should not include events after timestamp."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        query_ts = datetime(2025, 1, 12, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "First", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "Second", ts2), ts2),
        ]

        state = materialize_at(events, query_ts)
        assert len(state.entities) == 1
        assert "e1" in state.entities
        assert "e2" not in state.entities

    def test_handles_exact_timestamp_match(self):
        """Event at exact timestamp should be included."""
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts),
        ]

        state = materialize_at(events, ts)
        assert "e1" in state.entities

    def test_delete_then_query_before(self):
        """Should show entity when querying before deletion."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        query_ts = datetime(2025, 1, 12, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts1), ts1),
            make_event("ev2", "delete_entity", {"id": "e1"}, ts2),
        ]

        # Current state should not have the entity
        current = materialize(events)
        assert "e1" not in current.entities

        # But querying before deletion should show it
        past = materialize_at(events, query_ts)
        assert "e1" in past.entities

    def test_observation_history(self):
        """Should show observation state at different times."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts1), ts1),
            make_event("ev2", "add_observation", {
                "entity_id": "e1",
                "observation": {"id": "o1", "text": "First obs", "ts": ts2.isoformat(), "source": "test"},
            }, ts2),
            make_event("ev3", "add_observation", {
                "entity_id": "e1",
                "observation": {"id": "o2", "text": "Second obs", "ts": ts3.isoformat(), "source": "test"},
            }, ts3),
        ]

        # At ts1 - no observations
        state1 = materialize_at(events, ts1)
        assert len(state1.entities["e1"].observations) == 0

        # At ts2 - one observation
        state2 = materialize_at(events, ts2)
        assert len(state2.entities["e1"].observations) == 1

        # At ts3 - two observations
        state3 = materialize_at(events, ts3)
        assert len(state3.entities["e1"].observations) == 2


# --- Integration Tests ---


class TestRewindIntegration:
    """Integration tests for rewind functionality."""

    def test_full_lifecycle_rewind(self):
        """Test rewinding through a complete entity lifecycle."""
        ts_create = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts_update = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts_relate = datetime(2025, 1, 18, 12, 0, 0, tzinfo=timezone.utc)
        ts_delete = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Alpha", ts_create), ts_create),
            make_event("ev2", "create_entity", make_entity_data("e2", "Beta", ts_create), ts_create),
            make_event("ev3", "add_observation", {
                "entity_id": "e1",
                "observation": {"id": "o1", "text": "Updated", "ts": ts_update.isoformat(), "source": "test"},
            }, ts_update),
            make_event("ev4", "create_relation", {
                "id": "r1",
                "from_entity": "e1",
                "to_entity": "e2",
                "type": "relates_to",
                "created_at": ts_relate.isoformat(),
                "created_by": "test",
            }, ts_relate),
            make_event("ev5", "delete_entity", {"id": "e1"}, ts_delete),
        ]

        # Before creation
        state = materialize_at(events, ts_create - timedelta(days=1))
        assert len(state.entities) == 0
        assert len(state.relations) == 0

        # After creation
        state = materialize_at(events, ts_create)
        assert len(state.entities) == 2
        assert len(state.relations) == 0

        # After update
        state = materialize_at(events, ts_update)
        assert len(state.entities["e1"].observations) == 1

        # After relation
        state = materialize_at(events, ts_relate)
        assert len(state.relations) == 1

        # After delete
        state = materialize_at(events, ts_delete)
        assert len(state.entities) == 1  # Only e2 remains
        assert len(state.relations) == 0  # Cascade deleted


# --- Recovery Operations Tests ---

import tempfile
from pathlib import Path
from mnemograph.engine import MemoryEngine


class TestReload:
    """Tests for reload() functionality."""

    def test_reload_syncs_with_disk(self):
        """Reload should re-read events from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test")

            # Create an entity
            engine.create_entities([
                {"name": "Test", "entityType": "concept"},
            ])
            assert len(engine.state.entities) == 1

            # Simulate external modification by creating new engine
            # (which reads same file)
            engine2 = MemoryEngine(Path(tmpdir), "test2")
            engine2.create_entities([
                {"name": "Added", "entityType": "concept"},
            ])

            # Original engine doesn't see it yet
            assert len(engine.state.entities) == 1

            # After reload, it should
            result = engine.reload()
            assert result["status"] == "reloaded"
            assert result["entities"] == 2

    def test_reload_returns_counts(self):
        """Reload should return entity/relation counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test")

            engine.create_entities([
                {"name": "A", "entityType": "concept"},
                {"name": "B", "entityType": "concept"},
            ])
            engine.create_relations([
                {"from": "A", "to": "B", "relationType": "links_to"},
            ])

            result = engine.reload()
            assert result["entities"] == 2
            assert result["relations"] == 1
            assert result["events_processed"] >= 3

    def test_reload_invalidates_vector_index(self):
        """Reload should invalidate vector index so it rebuilds on next access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test")

            # Create entity and trigger vector index load
            engine.create_entities([
                {"name": "Original", "entityType": "concept", "observations": ["first version"]},
            ])
            # Access vector_index to lazy-load it
            _ = engine.vector_index
            assert engine._vector_index is not None

            # Reload should invalidate the index
            engine.reload()
            assert engine._vector_index is None

            # Next access should rebuild
            _ = engine.vector_index
            assert engine._vector_index is not None


class TestRestoreStateAt:
    """Tests for restore_state_at() functionality."""

    def test_restore_to_past(self):
        """Should restore graph to past state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test")

            # Create initial entity
            engine.create_entities([
                {"name": "Initial", "entityType": "concept"},
            ])

            # Wait a tiny bit, then create more
            import time
            time.sleep(0.01)
            ts_before = datetime.now(timezone.utc)
            time.sleep(0.01)

            engine.create_entities([
                {"name": "Later", "entityType": "concept"},
            ])

            assert len(engine.state.entities) == 2

            # Restore to before "Later" was created
            result = engine.restore_state_at(ts_before.isoformat())

            assert result["status"] == "restored"
            assert result["entities"] == 1

            # Verify only "Initial" exists
            names = [e.name for e in engine.state.entities.values()]
            assert "Initial" in names
            assert "Later" not in names

    def test_restore_with_reason(self):
        """Reason should be recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test")

            engine.create_entities([
                {"name": "Test", "entityType": "concept"},
            ])

            import time
            time.sleep(0.01)

            result = engine.restore_state_at(
                datetime.now(timezone.utc).isoformat(),
                reason="Testing restore"
            )

            assert result["status"] == "restored"
            # Check that clear_graph event has the reason
            events = engine.event_store.read_all()
            clear_events = [e for e in events if e.op == "clear_graph"]
            assert len(clear_events) >= 1
            assert "Testing restore" in clear_events[-1].data.get("reason", "")

    def test_restore_empty_state_error(self):
        """Should error if no entities at timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test")

            # Create entity
            engine.create_entities([
                {"name": "Test", "entityType": "concept"},
            ])

            # Try to restore to time before any events
            very_old = datetime(2020, 1, 1, tzinfo=timezone.utc)
            result = engine.restore_state_at(very_old.isoformat())

            assert result["status"] == "error"
            assert "No entities found" in result["error"]


class TestCompactEvent:
    """Tests for compact event handling in materialize."""

    def test_compact_clears_state(self):
        """Compact event should clear state before subsequent creates."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 15, 12, 0, 1, tzinfo=timezone.utc)

        events = [
            # Original entity
            make_event("ev1", "create_entity", make_entity_data("e1", "Original", ts1), ts1),
            # Compact event (clears state)
            make_event("ev2", "compact", {
                "reason": "test compact",
                "deleted_entities": ["Original"],
            }, ts2),
            # New entity after compact
            make_event("ev3", "create_entity", make_entity_data("e2", "New", ts3), ts3),
        ]

        state = materialize(events)

        # Should only have the new entity
        assert len(state.entities) == 1
        assert "e2" in state.entities
        assert "e1" not in state.entities
