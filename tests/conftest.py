"""Shared test fixtures and helpers for mnemograph tests."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemograph.engine import MemoryEngine
from mnemograph.models import MemoryEvent


# --- Fixtures ---


@pytest.fixture
def temp_memory_dir():
    """Provide a temporary directory for memory storage.

    Yields a Path to a temporary directory that's cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def engine(temp_memory_dir):
    """Provide a fresh MemoryEngine instance.

    Creates a new engine for each test with an empty memory store.
    """
    return MemoryEngine(temp_memory_dir, "test-session")


@pytest.fixture
def populated_engine(engine):
    """Provide a MemoryEngine with sample entities and relations.

    Pre-populates the engine with test data for testing retrieval and
    graph operations.
    """
    # Create sample entities (use force to avoid duplicate-check blocking
    # "Event Sourcing" vs "Use Event Sourcing")
    engine.create_entities_force([
        {"name": "Python", "entityType": "concept", "observations": ["Programming language"]},
        {"name": "Event Sourcing", "entityType": "concept", "observations": ["Architecture pattern"]},
        {"name": "Use Event Sourcing", "entityType": "decision", "observations": ["Chosen for audit trail"]},
    ])

    # Get entity IDs (they're generated, so we need to fetch them)
    entities = engine.get_recent_entities(limit=10)
    entity_map = {e.name: e.id for e in entities}

    # Create relations between entities
    if "Python" in entity_map and "Event Sourcing" in entity_map:
        engine.create_relations([
            {
                "from": entity_map["Python"],
                "to": entity_map["Use Event Sourcing"],
                "relationType": "relates_to",
            }
        ])

    return engine


# --- Helper Functions (not fixtures) ---


def make_event(id: str, op: str, data: dict, ts: datetime) -> MemoryEvent:
    """Helper to create test events.

    Args:
        id: Event ID (typically a ULID string)
        op: Operation type (e.g., "create_entity", "create_relation")
        data: Event data payload
        ts: Timestamp when the event occurred

    Returns:
        A MemoryEvent instance for testing.
    """
    return MemoryEvent(
        id=id,
        ts=ts,
        op=op,
        session_id="test",
        data=data,
    )


def make_entity_data(id: str, name: str, ts: datetime) -> dict:
    """Helper to create entity data for test events.

    Args:
        id: Entity ID (typically a ULID string)
        name: Human-readable entity name
        ts: Timestamp when the entity was created

    Returns:
        A dictionary with entity data for event creation.
    """
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
    """Helper to create relation data for test events.

    Args:
        id: Relation ID (typically a ULID string)
        from_id: Source entity ID
        to_id: Target entity ID
        ts: Timestamp when the relation was created

    Returns:
        A dictionary with relation data for event creation.
    """
    return {
        "id": id,
        "from_entity": from_id,
        "to_entity": to_id,
        "type": "relates_to",
        "created_at": ts.isoformat(),
        "created_by": "test",
    }
