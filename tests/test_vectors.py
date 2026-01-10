"""Tests for vector search (Phase 3)."""

import tempfile
from pathlib import Path

import pytest

from memory_engine.engine import MemoryEngine


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
