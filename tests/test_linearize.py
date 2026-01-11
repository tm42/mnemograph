"""Tests for prose linearization (format='prose' in recall)."""

import tempfile
from pathlib import Path

from mnemograph.engine import MemoryEngine
from mnemograph.linearize import (
    linearize_to_prose,
    linearize_shallow_summary,
    _extract_gotchas,
)
from mnemograph.models import Entity, Observation
from mnemograph.state import GraphState


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for linearize module functions
# ─────────────────────────────────────────────────────────────────────────────


def test_linearize_shallow_summary_empty():
    """Test shallow summary with empty graph."""
    state = GraphState()
    result = linearize_shallow_summary(state)
    assert "empty" in result.lower() or "no entities" in result.lower()


def test_linearize_shallow_summary_with_entities():
    """Test shallow summary includes stats and recent entities."""
    state = GraphState()

    # Add entities directly to state
    e1 = Entity(id="e1", name="FastAPI", type="project", observations=[])
    e2 = Entity(id="e2", name="Python", type="concept", observations=[])
    state.entities["e1"] = e1
    state.entities["e2"] = e2

    result = linearize_shallow_summary(state)

    assert "2 entities" in result
    assert "FastAPI" in result or "Python" in result


def test_extract_gotchas_with_prefixes():
    """Test gotcha extraction finds observations with warning prefixes."""
    obs1 = Observation(id="o1", text="Gotcha: This can fail silently", source="test")
    obs2 = Observation(id="o2", text="Warning: Don't use in production", source="test")
    obs3 = Observation(id="o3", text="Regular observation", source="test")
    obs4 = Observation(id="o4", text="Note: Important detail", source="test")

    entity = Entity(
        id="e1",
        name="TestEntity",
        type="concept",
        observations=[obs1, obs2, obs3, obs4]
    )

    gotchas = _extract_gotchas([entity])

    assert len(gotchas) == 3  # Gotcha, Warning, Note (not regular)
    assert "This can fail silently" in gotchas
    assert "Don't use in production" in gotchas
    assert "Important detail" in gotchas


def test_linearize_to_prose_groups_by_type():
    """Test that prose output groups entities by type."""
    state = GraphState()

    project = Entity(id="p1", name="MyProject", type="project", observations=[
        Observation(id="o1", text="A Python web service", source="test")
    ])
    decision = Entity(id="d1", name="Decision: Use Redis", type="decision", observations=[
        Observation(id="o2", text="Chosen for caching", source="test")
    ])
    concept = Entity(id="c1", name="Caching", type="concept", observations=[
        Observation(id="o3", text="Improves performance", source="test")
    ])

    state.entities = {"p1": project, "d1": decision, "c1": concept}
    entities = [project, decision, concept]
    entity_ids = {"p1", "d1", "c1"}

    result = linearize_to_prose(state, entities, entity_ids, depth="medium")

    # Check that all entity types appear
    assert "MyProject" in result
    assert "Decision:" in result or "Decisions:" in result
    assert "Caching" in result


def test_linearize_to_prose_depth_affects_verbosity():
    """Test that depth parameter changes output verbosity."""
    state = GraphState()

    entity = Entity(id="e1", name="TestEntity", type="concept", observations=[
        Observation(id=f"o{i}", text=f"Observation {i}", source="test")
        for i in range(10)
    ])

    state.entities = {"e1": entity}
    entities = [entity]
    entity_ids = {"e1"}

    shallow = linearize_to_prose(state, entities, entity_ids, depth="shallow")
    deep = linearize_to_prose(state, entities, entity_ids, depth="deep")

    # Deep should be longer than shallow
    assert len(deep) > len(shallow)


def test_linearize_gotchas_section():
    """Test that gotchas are extracted into their own section."""
    state = GraphState()

    entity = Entity(id="e1", name="API Client", type="concept", observations=[
        Observation(id="o1", text="HTTP client wrapper", source="test"),
        Observation(id="o2", text="Gotcha: Timeout is 30s by default", source="test"),
        Observation(id="o3", text="Warning: Rate limited to 100/min", source="test"),
    ])

    state.entities = {"e1": entity}
    entities = [entity]
    entity_ids = {"e1"}

    result = linearize_to_prose(state, entities, entity_ids, depth="medium")

    assert "Gotchas" in result
    assert "Timeout is 30s by default" in result
    assert "Rate limited to 100/min" in result


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests with MemoryEngine
# ─────────────────────────────────────────────────────────────────────────────


def test_recall_prose_format_shallow():
    """Test recall with format='prose' at shallow depth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Python", "entityType": "concept", "observations": ["A programming language"]},
            {"name": "FastAPI", "entityType": "project", "observations": ["Web framework"]},
        ])

        result = engine.recall(depth="shallow", format="prose")

        assert result["format"] == "prose"
        assert result["depth"] == "shallow"
        assert "2 entities" in result["content"]


def test_recall_prose_format_medium():
    """Test recall with format='prose' at medium depth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Redis", "entityType": "concept", "observations": [
                "In-memory data store",
                "Gotcha: Single-threaded by design"
            ]},
            {"name": "Decision: Use Redis", "entityType": "decision", "observations": [
                "Chosen for session caching"
            ]},
        ])

        result = engine.recall(depth="medium", query="caching", format="prose")

        assert result["format"] == "prose"
        assert result["depth"] == "medium"
        # Should include gotcha section
        assert "Gotcha" in result["content"] or "gotcha" in result["content"].lower()


def test_recall_graph_format_unchanged():
    """Test that format='graph' returns the original markdown structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "TestEntity", "entityType": "concept", "observations": ["Test"]},
        ])

        result = engine.recall(depth="shallow", format="graph")

        assert result["format"] == "graph"
        # Graph format uses the old retrieval format
        assert "Memory Summary" in result["content"]


def test_recall_default_format_is_prose():
    """Test that default format is prose."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Test", "entityType": "concept", "observations": ["Test obs"]},
        ])

        result = engine.recall(depth="shallow")

        # Default should be prose
        assert result["format"] == "prose"


def test_recall_prose_with_relations():
    """Test that prose format includes relation information for projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "MyApp", "entityType": "project", "observations": ["Main application"]},
            {"name": "PostgreSQL", "entityType": "concept", "observations": ["SQL database"]},
            {"name": "Redis", "entityType": "concept", "observations": ["Cache layer"]},
        ])

        engine.create_relations([
            {"from": "MyApp", "to": "PostgreSQL", "relationType": "uses"},
            {"from": "MyApp", "to": "Redis", "relationType": "uses"},
        ])

        result = engine.recall(depth="medium", focus=["MyApp"], format="prose")

        # Should show relations in prose
        assert "Uses:" in result["content"] or "uses" in result["content"].lower()


def test_recall_prose_empty_graph():
    """Test prose format with empty graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.recall(depth="shallow", format="prose")

        assert result["format"] == "prose"
        assert "empty" in result["content"].lower() or "no entities" in result["content"].lower()
