"""Tests for vector search (Phase 3)."""

import tempfile
from pathlib import Path

import pytest

from mnemograph.engine import MemoryEngine


@pytest.fixture
def engine_with_data():
    """Create an engine with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create diverse entities for semantic search testing
        engine.create_entities([
            {
                "name": "Python Programming",
                "entityType": "concept",
                "observations": [
                    "A high-level programming language",
                    "Known for readable syntax",
                    "Used for web development, data science, AI",
                ]
            },
            {
                "name": "Machine Learning",
                "entityType": "concept",
                "observations": [
                    "Subset of artificial intelligence",
                    "Algorithms that learn from data",
                    "Used for predictions and pattern recognition",
                ]
            },
            {
                "name": "Database Design",
                "entityType": "concept",
                "observations": [
                    "Organizing data in tables and relations",
                    "SQL for querying relational databases",
                    "Schema design and normalization",
                ]
            },
            {
                "name": "Use SQLite for storage",
                "entityType": "decision",
                "observations": [
                    "Chose SQLite over PostgreSQL",
                    "Single file, no server needed",
                    "Good enough for prototype",
                ]
            },
        ])

        yield engine


def test_semantic_search_finds_related(engine_with_data):
    """Test that semantic search finds conceptually related entities."""
    # Search for AI-related content
    results = engine_with_data.search_semantic("artificial intelligence and neural networks")

    # Should find Machine Learning (most related to AI)
    entity_names = [e["name"] for e in results["entities"]]
    assert "Machine Learning" in entity_names

    # Machine Learning should rank higher than unrelated entities
    ml_index = entity_names.index("Machine Learning")
    assert ml_index < len(entity_names) - 1  # Not last


def test_semantic_search_differs_from_keyword(engine_with_data):
    """Test that semantic search finds things keyword search wouldn't."""
    # "neural networks" doesn't appear in any entity text
    # but semantic search should still find ML
    results = engine_with_data.search_semantic("neural networks deep learning")

    entity_names = [e["name"] for e in results["entities"]]
    # Should still find ML even without exact keyword match
    assert "Machine Learning" in entity_names


def test_semantic_search_type_filter(engine_with_data):
    """Test filtering by entity type."""
    results = engine_with_data.search_semantic(
        "data storage solutions",
        type_filter="decision"
    )

    # Should only return decisions
    assert all(e["type"] == "decision" for e in results["entities"])
    assert len(results["entities"]) > 0


def test_semantic_search_scores(engine_with_data):
    """Test that results include similarity scores."""
    results = engine_with_data.search_semantic("programming languages")

    assert len(results["entities"]) > 0
    # Each entity should have a _score
    for entity in results["entities"]:
        assert "_score" in entity
        assert 0 < entity["_score"] <= 1


def test_semantic_search_limit(engine_with_data):
    """Test result limit."""
    results = engine_with_data.search_semantic("technology", limit=2)
    assert len(results["entities"]) <= 2


def test_vector_index_lazy_loading():
    """Test that vector index is lazy-loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Vector index should not be loaded yet
        assert engine._vector_index is None

        # Create an entity
        engine.create_entities([{"name": "Test", "entityType": "entity"}])

        # Still not loaded (no semantic operations)
        assert engine._vector_index is None

        # Now trigger semantic search
        engine.search_semantic("test query")

        # Now it should be loaded
        assert engine._vector_index is not None


# ─────────────────────────────────────────────────────────────────────────────
# Vector Index Freshness Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_vector_index_updated_on_add_observations(engine_with_data):
    """Test that vector index is re-indexed when observations are added."""
    # First, do a semantic search to trigger index loading
    initial_results = engine_with_data.search_semantic("cooking recipes food")
    initial_names = [e["name"] for e in initial_results["entities"]]

    # Python Programming should not be highly ranked for "cooking"
    # Now add a cooking-related observation to Python Programming
    engine_with_data.add_observations([{
        "entityName": "Python Programming",
        "contents": ["Great for building recipe recommendation systems and cooking apps"]
    }])

    # Search again - now Python Programming should rank higher for cooking
    updated_results = engine_with_data.search_semantic("cooking recipes food")
    updated_names = [e["name"] for e in updated_results["entities"]]

    # Python should now be in results (if it wasn't) or ranked higher
    assert "Python Programming" in updated_names


def test_vector_index_updated_on_delete_observations(engine_with_data):
    """Test that vector index is re-indexed when observations are deleted."""
    # Trigger index loading
    engine_with_data.search_semantic("test")

    # Get initial hash for Python Programming
    python_id = engine_with_data._resolve_entity("Python Programming")
    initial_meta = engine_with_data._vector_index.conn.execute(
        "SELECT text_hash FROM entity_meta WHERE entity_id = ?", (python_id,)
    ).fetchone()
    initial_hash = initial_meta[0] if initial_meta else None

    # Delete an observation
    engine_with_data.delete_observations([{
        "entityName": "Python Programming",
        "observations": ["Known for readable syntax"]
    }])

    # Hash should have changed (re-indexed with different text)
    updated_meta = engine_with_data._vector_index.conn.execute(
        "SELECT text_hash FROM entity_meta WHERE entity_id = ?", (python_id,)
    ).fetchone()
    updated_hash = updated_meta[0] if updated_meta else None

    assert updated_hash != initial_hash


def test_vector_index_cleaned_on_delete_entity(engine_with_data):
    """Test that deleted entities are removed from vector index."""
    # Trigger index loading and verify entity is indexed
    engine_with_data.search_semantic("database")

    db_id = engine_with_data._resolve_entity("Database Design")
    assert db_id is not None

    # Verify entity is in vector index
    initial_count = engine_with_data._vector_index.conn.execute(
        "SELECT COUNT(*) FROM entity_vectors WHERE entity_id = ?", (db_id,)
    ).fetchone()[0]
    assert initial_count == 1

    # Delete the entity
    engine_with_data.delete_entities(["Database Design"])

    # Verify entity is removed from vector index
    final_count = engine_with_data._vector_index.conn.execute(
        "SELECT COUNT(*) FROM entity_vectors WHERE entity_id = ?", (db_id,)
    ).fetchone()[0]
    assert final_count == 0

    # Verify metadata is also cleaned
    meta_count = engine_with_data._vector_index.conn.execute(
        "SELECT COUNT(*) FROM entity_meta WHERE entity_id = ?", (db_id,)
    ).fetchone()[0]
    assert meta_count == 0


def test_semantic_search_always_available_via_property():
    """Test that semantic search works even without prior vector access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create entities
        engine.create_entities([
            {"name": "FastAPI", "entityType": "concept", "observations": ["Python web framework"]}
        ])

        # Vector index not yet loaded
        assert engine._vector_index is None

        # search_structure should still try semantic search (via lazy-loaded property)
        # This tests that we removed the `if self._vector_index is not None:` guard
        results = engine.search_structure("web frameworks python")

        # Vector index should now be loaded
        assert engine._vector_index is not None

        # And we should get results
        assert results["matched_count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# GraphState Index Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_graphstate_index_consistency_after_materialize():
    """Test that indices are consistent after materialization."""
    from mnemograph.state import GraphState, materialize
    from mnemograph.models import MemoryEvent

    events = [
        MemoryEvent(op="create_entity", session_id="test", data={
            "id": "e1", "name": "Entity One", "type": "concept", "observations": []
        }),
        MemoryEvent(op="create_entity", session_id="test", data={
            "id": "e2", "name": "Entity Two", "type": "concept", "observations": []
        }),
        MemoryEvent(op="create_relation", session_id="test", data={
            "id": "r1", "from_entity": "e1", "to_entity": "e2", "type": "relates_to"
        }),
    ]

    state = materialize(events)
    errors = state.check_index_consistency()
    assert errors == [], f"Index consistency errors: {errors}"


def test_graphstate_index_consistency_after_apply_event():
    """Test that indices remain consistent after incremental updates."""
    from mnemograph.state import GraphState, materialize, apply_event
    from mnemograph.models import MemoryEvent

    # Start with some entities
    events = [
        MemoryEvent(op="create_entity", session_id="test", data={
            "id": "e1", "name": "Entity One", "type": "concept", "observations": []
        }),
        MemoryEvent(op="create_entity", session_id="test", data={
            "id": "e2", "name": "Entity Two", "type": "concept", "observations": []
        }),
    ]
    state = materialize(events)

    # Apply incremental updates
    apply_event(state, MemoryEvent(op="create_relation", session_id="test", data={
        "id": "r1", "from_entity": "e1", "to_entity": "e2", "type": "relates_to"
    }))

    errors = state.check_index_consistency()
    assert errors == [], f"Index consistency errors after add relation: {errors}"

    # Delete the relation
    apply_event(state, MemoryEvent(op="delete_relation", session_id="test", data={
        "from_entity": "e1", "to_entity": "e2", "type": "relates_to"
    }))

    errors = state.check_index_consistency()
    assert errors == [], f"Index consistency errors after delete relation: {errors}"

    # Delete an entity
    apply_event(state, MemoryEvent(op="delete_entity", session_id="test", data={
        "id": "e1"
    }))

    errors = state.check_index_consistency()
    assert errors == [], f"Index consistency errors after delete entity: {errors}"
