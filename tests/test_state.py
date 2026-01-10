"""Tests for state materialization."""

from datetime import datetime, timezone

from mnemograph.models import MemoryEvent
from mnemograph.state import materialize


def test_materialize_empty():
    """Test materializing empty event list."""
    state = materialize([])
    assert state.entities == {}
    assert state.relations == []
    assert state.last_event_id is None


def test_materialize_create_entity():
    """Test creating an entity."""
    events = [
        MemoryEvent(
            id="ev1",
            op="create_entity",
            session_id="test",
            data={
                "id": "e1",
                "name": "Test Entity",
                "type": "concept",
                "observations": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "created_by": "test",
                "access_count": 0,
                "last_accessed": None,
            },
        )
    ]
    state = materialize(events)

    assert "e1" in state.entities
    assert state.entities["e1"].name == "Test Entity"
    assert state.entities["e1"].type == "concept"
    assert state.last_event_id == "ev1"


def test_materialize_delete_entity():
    """Test deleting an entity."""
    events = [
        MemoryEvent(
            id="ev1",
            op="create_entity",
            session_id="test",
            data={
                "id": "e1",
                "name": "Test",
                "type": "entity",
                "observations": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "created_by": "test",
                "access_count": 0,
            },
        ),
        MemoryEvent(
            id="ev2",
            op="delete_entity",
            session_id="test",
            data={"id": "e1"},
        ),
    ]
    state = materialize(events)

    assert "e1" not in state.entities
    assert state.last_event_id == "ev2"


def test_materialize_create_relation():
    """Test creating a relation."""
    events = [
        MemoryEvent(
            id="ev1",
            op="create_relation",
            session_id="test",
            data={
                "id": "r1",
                "from_entity": "e1",
                "to_entity": "e2",
                "type": "implements",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_by": "test",
            },
        )
    ]
    state = materialize(events)

    assert len(state.relations) == 1
    assert state.relations[0].from_entity == "e1"
    assert state.relations[0].to_entity == "e2"
    assert state.relations[0].type == "implements"


def test_materialize_cascade_delete():
    """Test that deleting an entity cascades to relations."""
    now = datetime.now(timezone.utc).isoformat()
    events = [
        MemoryEvent(
            id="ev1",
            op="create_entity",
            session_id="test",
            data={
                "id": "e1", "name": "A", "type": "entity", "observations": [],
                "created_at": now, "updated_at": now, "created_by": "test", "access_count": 0,
            },
        ),
        MemoryEvent(
            id="ev2",
            op="create_entity",
            session_id="test",
            data={
                "id": "e2", "name": "B", "type": "entity", "observations": [],
                "created_at": now, "updated_at": now, "created_by": "test", "access_count": 0,
            },
        ),
        MemoryEvent(
            id="ev3",
            op="create_relation",
            session_id="test",
            data={
                "id": "r1", "from_entity": "e1", "to_entity": "e2", "type": "relates_to",
                "created_at": now, "created_by": "test",
            },
        ),
        MemoryEvent(
            id="ev4",
            op="delete_entity",
            session_id="test",
            data={"id": "e1"},
        ),
    ]
    state = materialize(events)

    assert "e1" not in state.entities
    assert "e2" in state.entities
    assert len(state.relations) == 0  # Relation should be cascade deleted


def test_materialize_add_observation():
    """Test adding an observation."""
    now = datetime.now(timezone.utc).isoformat()
    events = [
        MemoryEvent(
            id="ev1",
            op="create_entity",
            session_id="test",
            data={
                "id": "e1", "name": "Test", "type": "entity", "observations": [],
                "created_at": now, "updated_at": now, "created_by": "test", "access_count": 0,
            },
        ),
        MemoryEvent(
            id="ev2",
            op="add_observation",
            session_id="test",
            data={
                "entity_id": "e1",
                "observation": {"id": "o1", "text": "This is a test", "ts": now, "source": "test"},
            },
        ),
    ]
    state = materialize(events)

    assert len(state.entities["e1"].observations) == 1
    assert state.entities["e1"].observations[0].text == "This is a test"
