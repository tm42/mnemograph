"""Tests for the memory engine (Phase 2 features)."""

import tempfile
from pathlib import Path

from mnemograph.engine import MemoryEngine


def test_create_and_search():
    """Test creating entities and searching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Python", "entityType": "concept", "observations": ["A programming language"]},
            {"name": "JavaScript", "entityType": "concept", "observations": ["Another language"]},
        ])

        results = engine.search_nodes("python")
        assert len(results["entities"]) == 1
        assert results["entities"][0]["name"] == "Python"


def test_access_tracking():
    """Test that search updates access counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Test Entity", "entityType": "concept", "observations": ["test"]},
        ])

        # Initial access count is 0
        entity_id = list(engine.state.entities.keys())[0]
        assert engine.state.entities[entity_id].access_count == 0

        # Search should increment access count
        engine.search_nodes("test")
        assert engine.state.entities[entity_id].access_count == 1

        # Search again
        engine.search_nodes("test")
        assert engine.state.entities[entity_id].access_count == 2


def test_get_recent_entities():
    """Test getting recently updated entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create entities in order
        engine.create_entities([{"name": "First", "entityType": "entity"}])
        engine.create_entities([{"name": "Second", "entityType": "entity"}])
        engine.create_entities([{"name": "Third", "entityType": "entity"}])

        recent = engine.get_recent_entities(limit=2)
        assert len(recent) == 2
        assert recent[0].name == "Third"  # Most recent first
        assert recent[1].name == "Second"


def test_get_hot_entities():
    """Test getting frequently accessed entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Popular", "entityType": "concept", "observations": ["hot topic"]},
            {"name": "Unpopular", "entityType": "concept", "observations": ["cold topic"]},
        ])

        # Access "Popular" multiple times
        engine.search_nodes("hot")
        engine.search_nodes("hot")
        engine.search_nodes("hot")

        # Access "Unpopular" once
        engine.search_nodes("cold")

        hot = engine.get_hot_entities(limit=2)
        assert hot[0].name == "Popular"
        assert hot[0].access_count == 3
        assert hot[1].name == "Unpopular"
        assert hot[1].access_count == 1


def test_get_entities_by_type():
    """Test filtering entities by type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Decision A", "entityType": "decision"},
            {"name": "Concept A", "entityType": "concept"},
            {"name": "Decision B", "entityType": "decision"},
        ])

        decisions = engine.get_entities_by_type("decision")
        assert len(decisions) == 2
        assert all(e.type == "decision" for e in decisions)


def test_get_entity_neighbors():
    """Test getting entity neighbors via relations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "A", "entityType": "entity"},
            {"name": "B", "entityType": "entity"},
            {"name": "C", "entityType": "entity"},
        ])

        engine.create_relations([
            {"from": "A", "to": "B", "relationType": "connects_to"},
            {"from": "C", "to": "A", "relationType": "depends_on"},
        ])

        a_id = engine._resolve_entity("A")
        result = engine.get_entity_neighbors(a_id)

        assert result["entity"]["name"] == "A"
        assert len(result["neighbors"]) == 2  # B and C
        assert len(result["outgoing"]) == 1  # A -> B
        assert len(result["incoming"]) == 1  # C -> A


def test_open_nodes_with_neighbors():
    """Test that open_nodes returns neighbors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Center", "entityType": "entity"},
            {"name": "Left", "entityType": "entity"},
            {"name": "Right", "entityType": "entity"},
            {"name": "Unrelated", "entityType": "entity"},
        ])

        engine.create_relations([
            {"from": "Center", "to": "Left", "relationType": "links"},
            {"from": "Right", "to": "Center", "relationType": "links"},
        ])

        result = engine.open_nodes(["Center"])

        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Center"
        assert len(result["relations"]) == 2
        assert len(result["neighbors"]) == 2  # Left and Right
        neighbor_names = {n["name"] for n in result["neighbors"]}
        assert neighbor_names == {"Left", "Right"}
