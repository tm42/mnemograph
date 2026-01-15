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
    """Test shallow context returns summary (graph format)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept", "observations": ["Obs A"]},
            {"name": "Redis Cache System", "entityType": "decision", "observations": ["Obs B"]},
            {"name": "MongoDB Storage", "entityType": "concept", "observations": ["Obs C"]},
        ])

        result = engine.recall(depth="shallow", format="graph")

        assert result["depth"] == "shallow"
        assert result["format"] == "graph"
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


def test_shallow_context_frequently_accessed():
    """Test shallow context shows frequently accessed entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Popular", "entityType": "concept", "observations": ["Popular entity"]},
            {"name": "Other", "entityType": "concept", "observations": ["Regular entity"]},
        ])

        # Manually increment access count to test the display
        for entity in engine.state.entities.values():
            if entity.name == "Popular":
                entity.access_count = 5
                break

        result = engine.recall(depth="shallow", format="graph")

        # Should show frequently accessed section
        assert "Frequently Accessed" in result["content"]
        assert "Popular" in result["content"]
        assert "5 accesses" in result["content"]


def test_deep_context_no_focus_uses_recent():
    """Test deep context uses recent entities when no focus provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Recent1", "entityType": "concept", "observations": ["Recent entity 1"]},
            {"name": "Recent2", "entityType": "concept", "observations": ["Recent entity 2"]},
        ])

        # Call without focus - should use recent entities as seeds
        result = engine.recall(depth="deep")

        assert result["depth"] == "deep"
        assert result["entity_count"] >= 1
        # Should find recent entities
        assert "Recent1" in result["content"] or "Recent2" in result["content"]


def test_format_entities_empty():
    """Test _format_entities_with_relations with no entities."""
    from mnemograph.retrieval import _format_entities_with_relations
    from mnemograph.state import GraphState

    state = GraphState()
    result = _format_entities_with_relations(state, [], set(), 2000)

    assert result == "No matching entities found."


def test_format_entities_many_observations():
    """Test _format_entities_with_relations truncates observations."""
    from mnemograph.retrieval import _format_entities_with_relations
    from mnemograph.state import GraphState
    from mnemograph.models import Entity, Observation
    from datetime import datetime, timezone

    state = GraphState()

    # Create entity with many observations
    now = datetime.now(timezone.utc)
    observations = [
        Observation(id=f"o{i}", text=f"Observation {i}", ts=now, source="test")
        for i in range(10)
    ]
    entity = Entity(
        id="e1",
        name="TestEntity",
        type="concept",
        observations=observations,
        created_at=now,
        updated_at=now,
        created_by="test",
        access_count=0,
    )
    state.entities["e1"] = entity

    # Non-verbose should show truncation notice
    result = _format_entities_with_relations(state, [entity], {"e1"}, 2000, verbose=False)

    assert "7 more observations" in result  # 10 total, shows 3, so 7 more


def test_format_entities_truncation():
    """Test _format_entities_with_relations truncates when over token budget."""
    from mnemograph.retrieval import _format_entities_with_relations
    from mnemograph.state import GraphState
    from mnemograph.models import Entity, Observation
    from datetime import datetime, timezone

    state = GraphState()
    now = datetime.now(timezone.utc)

    # Create many entities with observations
    entities = []
    for i in range(20):
        observations = [
            Observation(id=f"o{i}_{j}", text=f"Long observation text " * 20, ts=now, source="test")
            for j in range(5)
        ]
        entity = Entity(
            id=f"e{i}",
            name=f"Entity{i}",
            type="concept",
            observations=observations,
            created_at=now,
            updated_at=now,
            created_by="test",
            access_count=0,
        )
        state.entities[f"e{i}"] = entity
        entities.append(entity)

    # Request with very small token budget
    result = _format_entities_with_relations(state, entities, {f"e{i}" for i in range(20)}, 100)

    # Should be truncated
    assert "truncated" in result.lower()


def test_medium_context_no_focus_no_query():
    """Test medium context with neither focus nor query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Entity1", "entityType": "concept", "observations": ["First entity"]},
            {"name": "Entity2", "entityType": "concept", "observations": ["Second entity"]},
        ])

        # Call with neither focus nor query - relies on weighted_bfs with empty seeds
        result = engine.recall(depth="medium")

        assert result["depth"] == "medium"
        # Should return something (implementation-dependent)


def test_format_with_relations():
    """Test _format_entities_with_relations shows relations."""
    from mnemograph.retrieval import _format_entities_with_relations
    from mnemograph.state import GraphState
    from mnemograph.models import Entity, Relation
    from datetime import datetime, timezone

    state = GraphState()
    now = datetime.now(timezone.utc)

    # Create two entities with a relation
    e1 = Entity(id="e1", name="Source", type="concept", observations=[],
                created_at=now, updated_at=now, created_by="test", access_count=0)
    e2 = Entity(id="e2", name="Target", type="concept", observations=[],
                created_at=now, updated_at=now, created_by="test", access_count=0)
    state.entities["e1"] = e1
    state.entities["e2"] = e2

    rel = Relation(id="r1", from_entity="e1", to_entity="e2", type="links_to",
                   created_at=now, created_by="test")
    state.relations.append(rel)

    result = _format_entities_with_relations(state, [e1, e2], {"e1", "e2"}, 2000)

    # Should show relation arrows
    assert "→" in result or "←" in result
    assert "links_to" in result
