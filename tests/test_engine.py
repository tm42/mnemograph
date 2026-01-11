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


# --- Universal Agent Tools Tests ---


def test_get_primer():
    """Test get_primer returns expected structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Project A", "entityType": "project"},
            {"name": "Concept B", "entityType": "concept"},
            {"name": "Learning C", "entityType": "learning"},
        ])

        engine.create_relations([
            {"from": "Project A", "to": "Concept B", "relationType": "uses"},
        ])

        result = engine.get_primer()

        assert result["status"]["entity_count"] == 3
        assert result["status"]["relation_count"] == 1
        assert result["status"]["types"]["project"] == 1
        assert result["status"]["types"]["concept"] == 1
        assert result["status"]["types"]["learning"] == 1
        assert len(result["recent_activity"]) <= 5
        assert "tools" in result
        assert "quick_start" in result


def test_session_start():
    """Test session_start returns context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Test Project", "entityType": "project"},
            {"name": "Some Concept", "entityType": "concept"},
        ])

        result = engine.session_start()

        assert result["session_id"] == "test-session"
        assert result["memory_summary"]["entity_count"] == 2
        assert "context" in result
        assert "tip" in result


def test_session_start_with_project_hint():
    """Test session_start with project hint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "My Project", "entityType": "project"},
        ])

        result = engine.session_start(project_hint="My Project")

        assert result["project"] == "My Project"


def test_session_end():
    """Test session_end basic flow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.session_end()

        assert result["status"] == "session_ended"
        assert result["summary_stored"] is False


def test_session_end_with_summary_to_project():
    """Test session_end stores summary on project entity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "My Project", "entityType": "project"},
        ])

        result = engine.session_end(summary="Completed feature X")

        assert result["status"] == "session_ended"
        assert result["summary_stored"] is True

        # Reload state and check observation was added
        engine.state = engine._load_state()
        project = next(e for e in engine.state.entities.values() if e.name == "My Project")
        assert any("Completed feature X" in obs.text for obs in project.observations)


def test_session_end_with_summary_creates_learning():
    """Test session_end creates learning entity when no project exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # No project entity exists
        result = engine.session_end(summary="Learned something important")

        assert result["status"] == "session_ended"
        assert result["summary_stored"] is True

        # Reload state and check learning was created
        engine.state = engine._load_state()
        learnings = [e for e in engine.state.entities.values() if e.type == "learning"]
        assert len(learnings) == 1
        assert "Learned something important" in learnings[0].observations[0].text


# --- Graph Coherence Tools Tests ---


def test_find_similar_exact_match_excluded():
    """Test find_similar excludes exact matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "React", "entityType": "concept"},
        ])

        # Searching for "React" should not find "React" itself
        result = engine.find_similar("React")
        assert len(result) == 0


def test_find_similar_finds_variants():
    """Test find_similar finds name variants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "React", "entityType": "concept"},
            {"name": "ReactJS", "entityType": "concept"},
            {"name": "React Native", "entityType": "concept"},
            {"name": "Vue", "entityType": "concept"},  # Unrelated
        ])

        result = engine.find_similar("React", threshold=0.5)

        # Should find ReactJS and React Native, but not Vue
        names = [r["name"] for r in result]
        assert "ReactJS" in names
        assert "React Native" in names
        assert "Vue" not in names


def test_find_orphans_detects_unconnected():
    """Test find_orphans finds entities with no relations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Connected A", "entityType": "concept"},
            {"name": "Connected B", "entityType": "concept"},
            {"name": "Orphan C", "entityType": "concept"},
        ])

        engine.create_relations([
            {"from": "Connected A", "to": "Connected B", "relationType": "uses"},
        ])

        orphans = engine.find_orphans()

        assert len(orphans) == 1
        assert orphans[0]["name"] == "Orphan C"


def test_find_orphans_empty_graph():
    """Test find_orphans on empty graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        orphans = engine.find_orphans()
        assert len(orphans) == 0


def test_merge_entities_basic():
    """Test merge_entities combines two entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "ReactJS", "entityType": "concept", "observations": ["A JS framework"]},
            {"name": "React", "entityType": "concept", "observations": ["Popular UI library"]},
            {"name": "Frontend", "entityType": "concept"},
        ])

        engine.create_relations([
            {"from": "ReactJS", "to": "Frontend", "relationType": "part_of"},
        ])

        result = engine.merge_entities("ReactJS", "React")

        assert result["status"] == "merged"
        assert result["observations_merged"] == 1
        assert result["relations_redirected"] == 1
        assert result["source_deleted"] is True

        # Verify ReactJS is gone
        engine.state = engine._load_state()
        assert all(e.name != "ReactJS" for e in engine.state.entities.values())

        # Verify React has the merged observation
        react = next(e for e in engine.state.entities.values() if e.name == "React")
        obs_texts = [o.text for o in react.observations]
        assert any("A JS framework" in t for t in obs_texts)

        # Verify relation was redirected
        assert any(
            r.type == "part_of"
            for r in engine.state.relations
            if engine.state.entities.get(r.from_entity, None)
            and engine.state.entities[r.from_entity].name == "React"
        )


def test_merge_entities_source_not_found():
    """Test merge_entities handles missing source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "React", "entityType": "concept"},
        ])

        result = engine.merge_entities("NonExistent", "React")
        assert "error" in result


def test_get_graph_health_empty():
    """Test get_graph_health on empty graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.get_graph_health()

        assert result["summary"]["total_entities"] == 0
        assert result["summary"]["orphan_count"] == 0
        assert "Graph looks healthy" in result["recommendations"][0]


def test_get_graph_health_detects_issues():
    """Test get_graph_health detects various issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create orphan
        engine.create_entities([
            {"name": "Orphan", "entityType": "concept"},
        ])

        # Create potential duplicates
        engine.create_entities([
            {"name": "React", "entityType": "concept"},
            {"name": "ReactJS", "entityType": "concept"},
        ])

        result = engine.get_graph_health()

        assert result["summary"]["total_entities"] == 3
        # All are orphans since no relations
        assert result["summary"]["orphan_count"] == 3
        # Should detect React/ReactJS as potential duplicates
        assert result["summary"]["duplicate_groups"] >= 1


def test_suggest_relations_finds_similar():
    """Test suggest_relations suggests based on similarity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "FastAPI", "entityType": "concept", "observations": ["Python web framework"]},
            {"name": "Flask", "entityType": "concept", "observations": ["Python web framework"]},
            {"name": "Django", "entityType": "concept", "observations": ["Python web framework"]},
            {"name": "Unrelated", "entityType": "concept", "observations": ["Something else"]},
        ])

        # Force vector index to initialize
        _ = engine.vector_index

        suggestions = engine.suggest_relations("FastAPI", limit=3)

        # Should suggest Flask and Django as similar
        targets = [s["target"] for s in suggestions if "error" not in s]
        # At least one of the similar frameworks should be suggested
        assert len(targets) > 0


def test_suggest_relations_entity_not_found():
    """Test suggest_relations handles missing entity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.suggest_relations("NonExistent")
        assert len(result) == 1
        assert "error" in result[0]


def test_count_connected_components():
    """Test _count_connected_components counts clusters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create two clusters
        engine.create_entities([
            {"name": "A1", "entityType": "concept"},
            {"name": "A2", "entityType": "concept"},
            {"name": "B1", "entityType": "concept"},
            {"name": "B2", "entityType": "concept"},
            {"name": "Orphan", "entityType": "concept"},
        ])

        engine.create_relations([
            {"from": "A1", "to": "A2", "relationType": "uses"},
            {"from": "B1", "to": "B2", "relationType": "uses"},
        ])

        # 3 components: A cluster, B cluster, Orphan
        count = engine._count_connected_components()
        assert count == 3
