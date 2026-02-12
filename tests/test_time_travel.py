"""Tests for time_travel.py TimeTraveler class.

Direct unit tests for the TimeTraveler class, separate from
engine-level integration tests in test_rewind.py.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from mnemograph.time_travel import TimeTraveler
from mnemograph.models import MemoryEvent
from mnemograph.state import GraphState


# --- Test Fixtures ---


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
    }


def make_relation_data(id: str, from_id: str, to_id: str, ts: datetime) -> dict:
    """Helper to create relation data."""
    return {
        "id": id,
        "from_entity": from_id,
        "to_entity": to_id,
        "type": "relates_to",
        "created_at": ts.isoformat(),
        "created_by": "test",
    }


class MockEventStore:
    """Mock event store for testing."""

    def __init__(self, events: list[MemoryEvent]):
        self._events = events

    def read_all(self) -> list[MemoryEvent]:
        return self._events


# --- state_at() Tests ---


class TestStateAt:
    """Tests for TimeTraveler.state_at()."""

    def test_with_string_timestamp(self):
        """Should parse string timestamp and return state."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "First", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "Second", ts2), ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Query before second entity
        state = traveler.state_at("2025-01-12T00:00:00")
        assert len(state.entities) == 1
        assert "e1" in state.entities

    def test_with_datetime_object(self):
        """Should accept datetime object directly."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "First", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "Second", ts2), ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Query with datetime object
        query_ts = datetime(2025, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
        state = traveler.state_at(query_ts)
        assert len(state.entities) == 1

    def test_with_naive_datetime(self):
        """Should treat naive datetime as UTC."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts1), ts1),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Naive datetime (no timezone)
        naive_ts = datetime(2025, 1, 15, 0, 0, 0)
        state = traveler.state_at(naive_ts)
        assert len(state.entities) == 1


# --- events_between() Tests ---


class TestEventsBetween:
    """Tests for TimeTraveler.events_between()."""

    def test_with_string_timestamps(self):
        """Should filter events between string timestamps."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"name": "First"}, ts1),
            make_event("ev2", "create_entity", {"name": "Second"}, ts2),
            make_event("ev3", "create_entity", {"name": "Third"}, ts3),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Get events between Jan 12 and Jan 18
        result = traveler.events_between("2025-01-12", "2025-01-18")
        assert len(result) == 1
        assert result[0].id == "ev2"

    def test_with_datetime_objects(self):
        """Should filter events between datetime objects."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"name": "First"}, ts1),
            make_event("ev2", "create_entity", {"name": "Second"}, ts2),
            make_event("ev3", "create_entity", {"name": "Third"}, ts3),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Get events between datetime objects
        start = datetime(2025, 1, 12, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 18, 0, 0, 0, tzinfo=timezone.utc)
        result = traveler.events_between(start, end)
        assert len(result) == 1

    def test_end_defaults_to_now(self):
        """Should use current time if end not specified."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"name": "Test"}, ts1),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Get events from Jan 1 to now
        result = traveler.events_between("2025-01-01")
        # Event at Jan 10 should be included
        assert len(result) == 1

    def test_boundary_inclusion(self):
        """Events at exact boundaries should be included."""
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"name": "Test"}, ts),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Query with exact boundary
        result = traveler.events_between(ts, ts)
        assert len(result) == 1


# --- diff_between() Tests ---


class TestDiffBetween:
    """Tests for TimeTraveler.diff_between()."""

    def test_entity_added(self):
        """Should detect added entities."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "First", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "Second", ts2), ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between("2025-01-12", "2025-01-18")

        assert len(diff["entities"]["added"]) == 1
        assert diff["entities"]["added"][0]["name"] == "Second"
        assert len(diff["entities"]["removed"]) == 0

    def test_entity_removed(self):
        """Should detect removed entities."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts1), ts1),
            make_event("ev2", "delete_entity", {"id": "e1"}, ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between("2025-01-12", "2025-01-18")

        assert len(diff["entities"]["removed"]) == 1
        assert diff["entities"]["removed"][0]["name"] == "Test"

    def test_entity_modified_name(self):
        """Should detect name changes."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Create entity then recreate with different name (simulating rename)
        entity_data_v1 = make_entity_data("e1", "OldName", ts1)
        entity_data_v2 = make_entity_data("e1", "NewName", ts2)

        events = [
            make_event("ev1", "create_entity", entity_data_v1, ts1),
            make_event("ev2", "delete_entity", {"id": "e1"}, ts2),
            make_event("ev3", "create_entity", entity_data_v2, ts2 + timedelta(seconds=1)),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        # Between ts1 and after ts2
        diff = traveler.diff_between(ts1, ts2 + timedelta(seconds=2))

        # Entity was removed then recreated with new name
        # The modified list checks for same ID with different attributes
        assert len(diff["entities"]["modified"]) == 1
        assert "name" in diff["entities"]["modified"][0]["changes"]

    def test_entity_modified_type(self):
        """Should detect type changes."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        entity_data_v1 = make_entity_data("e1", "Test", ts1)
        entity_data_v1["type"] = "concept"

        entity_data_v2 = make_entity_data("e1", "Test", ts2)
        entity_data_v2["type"] = "decision"

        events = [
            make_event("ev1", "create_entity", entity_data_v1, ts1),
            make_event("ev2", "delete_entity", {"id": "e1"}, ts2),
            make_event("ev3", "create_entity", entity_data_v2, ts2 + timedelta(seconds=1)),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between(ts1, ts2 + timedelta(seconds=2))

        assert len(diff["entities"]["modified"]) == 1
        assert "type" in diff["entities"]["modified"][0]["changes"]

    def test_observation_changes(self):
        """Should detect observation additions."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts1), ts1),
            make_event("ev2", "add_observation", {
                "entity_id": "e1",
                "observation": {"id": "o1", "text": "New obs", "ts": ts2.isoformat(), "source": "test"},
            }, ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between(ts1, ts2 + timedelta(seconds=1))

        # Entity should show as modified with observation changes
        assert len(diff["entities"]["modified"]) == 1
        assert "observations" in diff["entities"]["modified"][0]["changes"]
        assert diff["entities"]["modified"][0]["changes"]["observations"]["added"] == 1

    def test_relation_added(self):
        """Should detect added relations."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "A", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "B", ts1), ts1),
            make_event("ev3", "create_relation", make_relation_data("r1", "e1", "e2", ts2), ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between("2025-01-12", "2025-01-18")

        assert len(diff["relations"]["added"]) == 1
        assert diff["relations"]["added"][0]["type"] == "relates_to"

    def test_relation_removed(self):
        """Should detect removed relations."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "A", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "B", ts1), ts1),
            make_event("ev3", "create_relation", make_relation_data("r1", "e1", "e2", ts1), ts1),
            # delete_relation uses from_entity, to_entity, type (not id)
            make_event("ev4", "delete_relation", {
                "from_entity": "e1",
                "to_entity": "e2",
                "type": "relates_to",
            }, ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between("2025-01-12", "2025-01-18")

        assert len(diff["relations"]["removed"]) == 1

    def test_event_count(self):
        """Should count events in range."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "First", ts1), ts1),
            make_event("ev2", "create_entity", make_entity_data("e2", "Second", ts2), ts2),
            make_event("ev3", "create_entity", make_entity_data("e3", "Third", ts3), ts3),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between("2025-01-12", "2025-01-18")

        # Only ev2 is between Jan 12 and Jan 18 (exclusive of start)
        assert diff["event_count"] == 1

    def test_timestamps_in_result(self):
        """Should include resolved timestamps in result."""
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        events = [make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts)]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        diff = traveler.diff_between("2025-01-10", "2025-01-20")

        assert "start_timestamp" in diff
        assert "end_timestamp" in diff


# --- get_entity_history() Tests ---


class TestGetEntityHistory:
    """Tests for TimeTraveler.get_entity_history()."""

    def test_returns_relevant_events(self):
        """Should return events affecting the entity."""
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"id": "e1", "name": "Test"}, ts1),
            make_event("ev2", "add_observation", {"entity_id": "e1", "text": "obs"}, ts2),
            make_event("ev3", "create_entity", {"id": "e2", "name": "Other"}, ts2),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        history = traveler.get_entity_history("e1")

        # Should include ev1 and ev2 but not ev3
        assert len(history) == 2
        assert history[0]["operation"] == "create_entity"
        assert history[1]["operation"] == "add_observation"

    def test_includes_timestamp_and_session(self):
        """History entries should have timestamp and session info."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"id": "e1", "name": "Test"}, ts),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        history = traveler.get_entity_history("e1")

        assert len(history) == 1
        assert "timestamp" in history[0]
        assert "session_id" in history[0]
        assert "data" in history[0]

    def test_empty_for_unknown_entity(self):
        """Should return empty list for unknown entity."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("ev1", "create_entity", {"id": "e1", "name": "Test"}, ts),
        ]

        store = MockEventStore(events)
        traveler = TimeTraveler(lambda: store)

        history = traveler.get_entity_history("nonexistent")

        assert history == []


# --- restore_state_at() Tests ---


class TestRestoreStateAt:
    """Tests for TimeTraveler.restore_state_at()."""

    def test_error_without_emit_callback(self):
        """Should return error if emit callback not configured."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = [make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts)]

        store = MockEventStore(events)
        # No emit callback
        traveler = TimeTraveler(lambda: store)

        result = traveler.restore_state_at("2025-01-15")

        assert result["status"] == "error"
        assert "no emit callback" in result["error"].lower()

    def test_error_for_empty_past_state(self):
        """Should error if no entities at target timestamp."""
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        events = [make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts)]

        store = MockEventStore(events)
        emit_mock = Mock()
        traveler = TimeTraveler(lambda: store, emit_mock)

        # Try to restore to before any entities existed
        result = traveler.restore_state_at("2025-01-01")

        assert result["status"] == "error"
        assert "No entities found" in result["error"]
        # Should not have called emit
        emit_mock.assert_not_called()

    def test_emits_restore_to_event(self):
        """Should emit single restore_to marker event."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = [make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts)]

        store = MockEventStore(events)
        emit_mock = Mock()
        traveler = TimeTraveler(lambda: store, emit_mock)

        result = traveler.restore_state_at("2025-01-15")

        assert result["status"] == "restored"
        # Should call emit once for restore_to
        assert emit_mock.call_count == 1
        # Should be restore_to event
        assert emit_mock.call_args_list[0][0][0] == "restore_to"
        # Should include timestamp
        assert "timestamp" in emit_mock.call_args_list[0][0][1]

    def test_includes_reason_in_restore_to(self):
        """Reason should be included in restore_to event."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = [make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts)]

        store = MockEventStore(events)
        emit_mock = Mock()
        traveler = TimeTraveler(lambda: store, emit_mock)

        traveler.restore_state_at("2025-01-15", reason="Testing restore")

        # Check restore_to was called with reason
        restore_call = emit_mock.call_args_list[0]
        assert restore_call[0][0] == "restore_to"
        assert "Testing restore" in restore_call[0][1]["reason"]

    def test_handles_relations_in_restore(self):
        """Should restore state with relations using single restore_to event."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "A", ts), ts),
            make_event("ev2", "create_entity", make_entity_data("e2", "B", ts), ts),
            make_event("ev3", "create_relation", make_relation_data("r1", "e1", "e2", ts), ts),
        ]

        store = MockEventStore(events)
        emit_mock = Mock()
        traveler = TimeTraveler(lambda: store, emit_mock)

        result = traveler.restore_state_at("2025-01-15")

        assert result["status"] == "restored"
        # Should emit single restore_to event
        assert emit_mock.call_count == 1
        assert emit_mock.call_args_list[0][0][0] == "restore_to"
        # Counts should reflect restored state (2 entities, 1 relation)
        assert result["entities"] == 2
        assert result["relations"] == 1

    def test_returns_counts(self):
        """Should return entity and relation counts."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = [
            make_event("ev1", "create_entity", make_entity_data("e1", "A", ts), ts),
            make_event("ev2", "create_entity", make_entity_data("e2", "B", ts), ts),
            make_event("ev3", "create_relation", make_relation_data("r1", "e1", "e2", ts), ts),
        ]

        store = MockEventStore(events)
        emit_mock = Mock()
        traveler = TimeTraveler(lambda: store, emit_mock)

        result = traveler.restore_state_at("2025-01-15")

        assert result["entities"] == 2
        assert result["relations"] == 1

    def test_uses_state_getter_for_counts(self):
        """Should use state_getter if provided for accurate counts."""
        ts = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = [make_event("ev1", "create_entity", make_entity_data("e1", "Test", ts), ts)]

        store = MockEventStore(events)
        emit_mock = Mock()
        traveler = TimeTraveler(lambda: store, emit_mock)

        # Mock state getter that returns different counts
        mock_state = Mock()
        mock_state.entities = {"e1": None, "e2": None, "e3": None}  # 3 entities
        mock_state.relations = []
        state_getter = Mock(return_value=mock_state)

        result = traveler.restore_state_at("2025-01-15", state_getter=state_getter)

        assert result["entities"] == 3
        state_getter.assert_called_once()


# --- resolve_restores() Tests ---


class TestResolveRestores:
    """Tests for state.resolve_restores() function."""

    def test_nested_restores(self):
        """Should handle nested restore_to events recursively."""
        from mnemograph.state import resolve_restores

        # Timeline: T1 < T2 < T3 < T4 < T5
        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 12, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 14, 12, 0, 0, tzinfo=timezone.utc)
        ts4 = datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc)
        ts5 = datetime(2025, 1, 18, 12, 0, 0, tzinfo=timezone.utc)

        # Event sequence:
        # - e1 at T1
        # - e2 at T2
        # - restore_to(T1) at T3 - restores to T1, so only e1 should be visible
        # - e3 at T4 (after first restore)
        # - restore_to(T3) at T5 - restores to T3, which itself had a restore
        # Expected: resolve to [e1] (the state at T1, as restored at T3)

        events = [
            make_event("e1", "create_entity", make_entity_data("ent1", "First", ts1), ts1),
            make_event("e2", "create_entity", make_entity_data("ent2", "Second", ts2), ts2),
            make_event("restore1", "restore_to", {"timestamp": ts1.isoformat()}, ts3),
            make_event("e3", "create_entity", make_entity_data("ent3", "Third", ts4), ts4),
            make_event("restore2", "restore_to", {"timestamp": ts3.isoformat()}, ts5),
        ]

        resolved = resolve_restores(events)

        # After resolving nested restores, should have only e1
        # (the state at T1, which is what the graph looked like at T3 after the first restore)
        assert len(resolved) == 1
        assert resolved[0].id == "e1"

    def test_resolve_filters_out_restore_to_markers(self):
        """restore_to markers should not appear in resolved events."""
        from mnemograph.state import resolve_restores

        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("e1", "create_entity", make_entity_data("ent1", "Test", ts1), ts1),
            make_event("restore1", "restore_to", {"timestamp": ts1.isoformat()}, ts2),
        ]

        resolved = resolve_restores(events)

        # Should have e1 but NOT the restore_to marker
        assert len(resolved) == 1
        assert resolved[0].op == "create_entity"
        assert all(e.op != "restore_to" for e in resolved)

    def test_malformed_restore_to_filtered_out(self):
        """Malformed restore_to events (no timestamp) should be filtered."""
        from mnemograph.state import resolve_restores

        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("e1", "create_entity", make_entity_data("ent1", "Test", ts1), ts1),
            make_event("bad_restore", "restore_to", {}, ts2),  # No timestamp
        ]

        resolved = resolve_restores(events)

        # Should have e1, malformed restore_to filtered out
        assert len(resolved) == 1
        assert resolved[0].id == "e1"

    def test_invalid_timestamp_filtered_out(self):
        """restore_to with invalid timestamp should be filtered."""
        from mnemograph.state import resolve_restores

        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("e1", "create_entity", make_entity_data("ent1", "Test", ts1), ts1),
            make_event("bad_restore", "restore_to", {"timestamp": "invalid-timestamp"}, ts2),
        ]

        resolved = resolve_restores(events)

        # Should have e1, invalid restore_to filtered out
        assert len(resolved) == 1
        assert resolved[0].id == "e1"

    def test_future_timestamp_restore(self):
        """restore_to with future timestamp should include all events before it."""
        from mnemograph.state import resolve_restores

        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts_future = datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            make_event("e1", "create_entity", make_entity_data("ent1", "First", ts1), ts1),
            make_event("e2", "create_entity", make_entity_data("ent2", "Second", ts2), ts2),
            make_event("restore1", "restore_to", {"timestamp": ts_future.isoformat()}, ts3),
        ]

        resolved = resolve_restores(events)

        # Future restore should include all events before the restore_to
        # (acts as no-op since T_ref is after all events)
        assert len(resolved) == 2
        assert resolved[0].id == "e1"
        assert resolved[1].id == "e2"

    def test_multiple_sequential_restores(self):
        """Multiple restore_to events in sequence should be resolved correctly."""
        from mnemograph.state import resolve_restores

        ts1 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 12, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 14, 12, 0, 0, tzinfo=timezone.utc)
        ts4 = datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc)

        # Three sequential restores, last one wins
        events = [
            make_event("e1", "create_entity", make_entity_data("ent1", "First", ts1), ts1),
            make_event("restore1", "restore_to", {"timestamp": ts1.isoformat()}, ts2),
            make_event("restore2", "restore_to", {"timestamp": ts2.isoformat()}, ts3),
            make_event("restore3", "restore_to", {"timestamp": ts1.isoformat()}, ts4),
        ]

        resolved = resolve_restores(events)

        # Last restore_to(T1) should win - only e1 should be included
        assert len(resolved) == 1
        assert resolved[0].id == "e1"
