"""Tests for tiered retrieval (Phase 4)."""

import tempfile
from pathlib import Path

from mnemograph.engine import MemoryEngine


def test_shallow_context_empty():
    """Test shallow context with no entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.recall(depth="shallow")

        assert result["depth"] == "shallow"
        assert "No entities stored yet" in result["content"]
        assert result["entity_count"] == 0


def test_shallow_context_with_entities():
    """Test shallow context returns summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Entity A", "entityType": "concept", "observations": ["Obs A"]},
            {"name": "Entity B", "entityType": "decision", "observations": ["Obs B"]},
            {"name": "Entity C", "entityType": "concept", "observations": ["Obs C"]},
        ])

        result = engine.recall(depth="shallow")

        assert result["depth"] == "shallow"
        assert "3 entities" in result["content"]
        assert "Memory Summary" in result["content"]
        assert "concept: 2" in result["content"]
        assert "decision: 1" in result["content"]


def test_medium_context_with_focus():
    """Test medium context focuses on specific entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Center", "entityType": "concept", "observations": ["The center node"]},
            {"name": "Left", "entityType": "concept", "observations": ["Connected to center"]},
            {"name": "Right", "entityType": "concept", "observations": ["Also connected"]},
            {"name": "Isolated", "entityType": "concept", "observations": ["Not connected"]},
        ])

        engine.create_relations([
            {"from": "Center", "to": "Left", "relationType": "links_to"},
            {"from": "Center", "to": "Right", "relationType": "links_to"},
        ])

        result = engine.recall(depth="medium", focus=["Center"])

        assert result["depth"] == "medium"
        assert "Center" in result["content"]
        # Should include neighbors
        assert "Left" in result["content"] or "Right" in result["content"]
        # May or may not include isolated depending on implementation
        assert result["entity_count"] >= 1


def test_medium_context_with_query():
    """Test medium context uses semantic search when query provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Machine Learning", "entityType": "concept",
             "observations": ["AI and neural networks", "Deep learning models"]},
            {"name": "Cooking Recipes", "entityType": "concept",
             "observations": ["Italian pasta", "French cuisine"]},
        ])

        result = engine.recall(depth="medium", query="artificial intelligence")

        assert result["depth"] == "medium"
        # Should find ML via semantic search
        assert "Machine Learning" in result["content"]


def test_deep_context_multi_hop():
    """Test deep context traverses multiple hops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create a chain: A -> B -> C -> D
        engine.create_entities([
            {"name": "A", "entityType": "entity", "observations": ["Start"]},
            {"name": "B", "entityType": "entity", "observations": ["Hop 1"]},
            {"name": "C", "entityType": "entity", "observations": ["Hop 2"]},
            {"name": "D", "entityType": "entity", "observations": ["Hop 3"]},
        ])

        engine.create_relations([
            {"from": "A", "to": "B", "relationType": "connects"},
            {"from": "B", "to": "C", "relationType": "connects"},
            {"from": "C", "to": "D", "relationType": "connects"},
        ])

        result = engine.recall(depth="deep", focus=["A"])

        assert result["depth"] == "deep"
        # Should traverse multiple hops and find more entities
        assert result["entity_count"] >= 3  # A + at least 2 hops


def test_context_token_budget():
    """Test that token budget is respected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create many entities
        for i in range(20):
            engine.create_entities([
                {"name": f"Entity {i}", "entityType": "concept",
                 "observations": [f"Long observation text for entity {i} " * 10]}
            ])

        # Request with small token budget
        result = engine.recall(depth="shallow", max_tokens=200)

        # Content should be truncated
        assert result["tokens_estimate"] <= 250  # Allow some slack


def test_invalid_depth():
    """Test that invalid depth raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        try:
            engine.recall(depth="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid depth" in str(e)
