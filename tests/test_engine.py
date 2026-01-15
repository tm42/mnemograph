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

        results = engine.search_graph("python")
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
        engine.search_graph("test")
        assert engine.state.entities[entity_id].access_count == 1

        # Search again
        engine.search_graph("test")
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
            {"name": "Hot Topic", "entityType": "concept", "observations": ["hot stuff"]},
            {"name": "Cold Topic", "entityType": "concept", "observations": ["cold stuff"]},
        ])

        # Access "Hot Topic" multiple times
        engine.search_graph("hot")
        engine.search_graph("hot")
        engine.search_graph("hot")

        # Access "Cold Topic" once
        engine.search_graph("cold")

        hot = engine.get_hot_entities(limit=2)
        assert hot[0].name == "Hot Topic"
        assert hot[0].access_count == 3
        assert hot[1].name == "Cold Topic"
        assert hot[1].access_count == 1


def test_get_entities_by_type():
    """Test filtering entities by type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Use distinct names to avoid semantic similarity blocking
        engine.create_entities([
            {"name": "Decision: Use PostgreSQL", "entityType": "decision"},
            {"name": "Concept: Event Sourcing", "entityType": "concept"},
            {"name": "Decision: Use Redis Cache", "entityType": "decision"},
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
    """Test session_start returns context and quick_start guide."""
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
        assert "quick_start" in result
        assert "recall" in result["quick_start"]  # Verify guide mentions key tools
        assert "remember" in result["quick_start"]


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

        # Use create_entities_force since we're intentionally creating similar entities
        engine.create_entities_force([
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

        # Use distinct names to avoid semantic similarity blocking
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept"},
            {"name": "Redis Cache System", "entityType": "concept"},
            {"name": "Orphan Concept", "entityType": "concept"},
        ])

        engine.create_relations([
            {"from": "PostgreSQL Database", "to": "Redis Cache System", "relationType": "uses"},
        ])

        orphans = engine.find_orphans()

        assert len(orphans) == 1
        assert orphans[0]["name"] == "Orphan Concept"


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

        # Use force since we're intentionally creating similar entities to merge
        engine.create_entities_force([
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

        # Create potential duplicates (use force to bypass auto-check)
        # Use names that are similar enough to trigger duplicate detection (>0.8)
        # "Test Entity" and "Test Entity v2" have high substring overlap
        engine.create_entities_force([
            {"name": "Test Entity", "entityType": "concept"},
            {"name": "Test Entity v2", "entityType": "concept"},  # High overlap
        ])

        result = engine.get_graph_health()

        assert result["summary"]["total_entities"] == 3
        # All are orphans since no relations
        assert result["summary"]["orphan_count"] == 3
        # Should detect Test Entity/Test Entity v2 as potential duplicates
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


# --- Clear Graph Tests ---


def test_clear_graph_basic():
    """Test clear_graph removes all entities and relations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create some data
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept"},
            {"name": "Redis Cache System", "entityType": "concept"},
            {"name": "MongoDB Storage", "entityType": "project"},
        ])

        engine.create_relations([
            {"from": "PostgreSQL Database", "to": "Redis Cache System", "relationType": "uses"},
        ])

        assert len(engine.state.entities) == 3
        assert len(engine.state.relations) == 1

        # Clear the graph
        result = engine.clear_graph(reason="Testing clear")

        assert result["status"] == "cleared"
        assert result["entities_cleared"] == 3
        assert result["relations_cleared"] == 1
        assert result["reason"] == "Testing clear"

        # Verify state is empty
        assert len(engine.state.entities) == 0
        assert len(engine.state.relations) == 0


def test_clear_graph_event_sourced():
    """Test clear_graph is event-sourced (can be rewound)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create some data
        engine.create_entities([
            {"name": "PostgreSQL Database", "entityType": "concept"},
        ])

        # Get timestamp before clear
        from datetime import datetime, timezone
        import time
        time.sleep(0.01)  # Ensure timestamp difference
        before_clear = datetime.now(timezone.utc)
        time.sleep(0.01)

        # Clear the graph
        engine.clear_graph()

        assert len(engine.state.entities) == 0

        # Rewind to before clear
        state_before = engine.state_at(before_clear)

        assert len(state_before.entities) == 1
        assert any(e.name == "PostgreSQL Database" for e in state_before.entities.values())


def test_clear_graph_clears_indices():
    """Test clear_graph also clears O(1) indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        engine.create_entities([
            {"name": "Nginx Webserver", "entityType": "concept"},
            {"name": "Apache Kafka", "entityType": "concept"},
        ])

        engine.create_relations([
            {"from": "Nginx Webserver", "to": "Apache Kafka", "relationType": "uses"},
        ])

        # Verify indices have data
        assert len(engine.state._name_to_id) == 2
        assert len(engine.state._connected_entities) == 2

        # Clear
        engine.clear_graph()

        # Verify indices are cleared
        assert len(engine.state._name_to_id) == 0
        assert len(engine.state._outgoing) == 0
        assert len(engine.state._incoming) == 0
        assert len(engine.state._connected_entities) == 0


# --- Auto-Duplicate Check Tests ---


def test_create_entities_blocks_duplicate():
    """Test create_entities blocks when similar entity exists (>80% similarity)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create initial entity
        engine.create_entities([
            {"name": "Test Entity", "entityType": "concept"},
        ])

        # Try to create similar entity - should be blocked (81% similarity)
        results = engine.create_entities([
            {"name": "Test Entity v2", "entityType": "concept"},
        ])

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["status"] == "duplicate_warning"
        assert "Test Entity" in results[0]["warning"]
        assert results[0]["name"] == "Test Entity v2"

        # Verify entity was NOT created
        assert len(engine.state.entities) == 1


def test_create_entities_allows_distinct():
    """Test create_entities allows sufficiently different names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create initial entity
        engine.create_entities([
            {"name": "React", "entityType": "concept"},
        ])

        # Try to create clearly different entity - should succeed
        results = engine.create_entities([
            {"name": "Vue", "entityType": "concept"},
        ])

        assert len(results) == 1
        # Should be an Entity, not a warning dict
        from mnemograph.models import Entity
        assert isinstance(results[0], Entity)
        assert results[0].name == "Vue"

        # Verify both entities exist
        assert len(engine.state.entities) == 2


def test_create_entities_force_bypasses_check():
    """Test create_entities_force bypasses duplicate check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create initial entity
        engine.create_entities([
            {"name": "React", "entityType": "concept"},
        ])

        # Use force to create similar entity
        results = engine.create_entities_force([
            {"name": "ReactJS", "entityType": "concept", "observations": ["Alternative name"]},
        ])

        assert len(results) == 1
        from mnemograph.models import Entity
        assert isinstance(results[0], Entity)
        assert results[0].name == "ReactJS"

        # Verify both entities exist
        assert len(engine.state.entities) == 2


def test_create_entities_mixed_results():
    """Test create_entities can return mix of created and blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create initial entity
        engine.create_entities([
            {"name": "Test Entity", "entityType": "concept"},
        ])

        # Try to create one similar (>80%), one distinct
        results = engine.create_entities([
            {"name": "Test Entity v2", "entityType": "concept"},  # Should be blocked (81%)
            {"name": "Angular", "entityType": "concept"},  # Should succeed
        ])

        assert len(results) == 2

        # First should be warning
        assert isinstance(results[0], dict)
        assert results[0]["status"] == "duplicate_warning"

        # Second should be Entity
        from mnemograph.models import Entity
        assert isinstance(results[1], Entity)
        assert results[1].name == "Angular"

        # Verify correct entities exist (Test Entity and Angular, not Test Entity v2)
        assert len(engine.state.entities) == 2
        names = {e.name for e in engine.state.entities.values()}
        assert names == {"Test Entity", "Angular"}


# --- Remember (High-Level Knowledge Creation) Tests ---


def test_remember_creates_entity_with_observations():
    """Test remember creates entity with observations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.remember(
            name="FastAPI",
            entity_type="concept",
            observations=["Async Python web framework", "Uses Pydantic for validation"],
        )

        assert result["status"] == "created"
        assert result["entity"]["name"] == "FastAPI"
        assert result["entity"]["type"] == "concept"
        assert len(result["entity"]["observations"]) == 2

        # Verify entity exists in state
        assert len(engine.state.entities) == 1
        entity = list(engine.state.entities.values())[0]
        assert entity.name == "FastAPI"


def test_remember_creates_entity_with_relations():
    """Test remember creates entity and relations atomically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # First create target entities
        engine.create_entities([
            {"name": "Python", "entityType": "concept"},
            {"name": "Pydantic", "entityType": "concept"},
        ])

        # Now remember with relations
        result = engine.remember(
            name="FastAPI",
            entity_type="concept",
            observations=["Async Python web framework"],
            relations=[
                {"to": "Python", "type": "uses"},
                {"to": "Pydantic", "type": "depends_on"},
            ],
        )

        assert result["status"] == "created"
        assert result["relations_created"] == 2
        assert "relations_failed" not in result

        # Verify relations in state
        assert len(engine.state.relations) == 2


def test_remember_handles_missing_relation_target():
    """Test remember reports failed relations when target doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.remember(
            name="FastAPI",
            entity_type="concept",
            relations=[
                {"to": "NonExistent", "type": "uses"},
            ],
        )

        # Entity should still be created
        assert result["status"] == "created"
        assert result["relations_created"] == 0
        assert len(result["relations_failed"]) == 1
        assert "NonExistent" in result["relations_failed"][0]["error"]


def test_remember_blocks_duplicate():
    """Test remember blocks when similar entity exists (>80% similarity)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create initial entity
        engine.create_entities([
            {"name": "Test Entity", "entityType": "concept"},
        ])

        # Try to remember similar entity (81% similarity)
        result = engine.remember(
            name="Test Entity v2",
            entity_type="concept",
            observations=["Similar concept"],
        )

        assert result["status"] == "duplicate_warning"
        assert "Test Entity" in result["warning"]
        assert "suggestion" in result

        # Verify entity was NOT created
        assert len(engine.state.entities) == 1


def test_remember_force_bypasses_duplicate_check():
    """Test remember with force=True bypasses duplicate check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create initial entity
        engine.create_entities([
            {"name": "React", "entityType": "concept"},
        ])

        # Force remember similar entity
        result = engine.remember(
            name="ReactJS",
            entity_type="concept",
            observations=["Alternative name for React"],
            force=True,
        )

        assert result["status"] == "created"
        assert result["entity"]["name"] == "ReactJS"

        # Verify both entities exist
        assert len(engine.state.entities) == 2


def test_remember_minimal_call():
    """Test remember with only required arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        result = engine.remember(
            name="SimpleConcept",
            entity_type="concept",
        )

        assert result["status"] == "created"
        assert result["entity"]["name"] == "SimpleConcept"
        assert result["relations_created"] == 0
        assert len(result["entity"]["observations"]) == 0


def test_remember_full_workflow():
    """Test remember with full entity + observations + relations workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MemoryEngine(Path(tmpdir), "test-session")

        # Create a project
        engine.remember(
            name="MyProject",
            entity_type="project",
            observations=["A web application"],
        )

        # Create concepts the project uses
        engine.remember(name="Python", entity_type="concept")
        engine.remember(name="PostgreSQL", entity_type="concept")

        # Create a decision with relations
        result = engine.remember(
            name="Decision: Use FastAPI",
            entity_type="decision",
            observations=[
                "Chose FastAPI for its async support",
                "Better performance than Flask for our use case",
            ],
            relations=[
                {"to": "MyProject", "type": "decided_for"},
                {"to": "Python", "type": "uses"},
            ],
        )

        assert result["status"] == "created"
        assert result["relations_created"] == 2

        # Verify the decision is connected
        decision_id = result["entity"]["id"]
        neighbors = engine.get_entity_neighbors(decision_id)
        assert len(neighbors["neighbors"]) == 2
