"""Tests for D4: QueryService extraction from engine.py.

Tests QueryService directly (with mocks) and verifies the
MemoryEngine delegation layer works correctly.
"""

import inspect
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from mnemograph.models import Entity, Observation, Relation
from mnemograph.query import QueryService
from mnemograph.state import GraphState


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(entities=None, relations=None) -> GraphState:
    """Build a GraphState with indexed lookups."""
    entities = entities or {}
    relations = relations or []
    state = GraphState(entities=entities, relations=relations)
    state._rebuild_indices()
    return state


def _make_entity(eid, name, etype="concept", observations=None):
    """Shorthand for creating an Entity."""
    obs = [
        Observation(id=f"o-{eid}-{i}", text=text, ts=datetime(2025, 1, 1, tzinfo=timezone.utc), source="test")
        for i, text in enumerate(observations or [])
    ]
    return Entity(
        id=eid, name=name, type=etype, observations=obs,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _make_relation(rid, from_id, to_id, rtype="relates_to"):
    """Shorthand for creating a Relation."""
    return Relation(
        id=rid, from_entity=from_id, to_entity=to_id, type=rtype,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_state():
    """A state with 3 entities and 2 relations."""
    entities = {
        "e1": _make_entity("e1", "Python", observations=["Programming language"]),
        "e2": _make_entity("e2", "FastAPI", observations=["Async web framework"]),
        "e3": _make_entity("e3", "PostgreSQL", "concept", ["Relational database"]),
    }
    relations = [
        _make_relation("r1", "e1", "e2", "uses"),
        _make_relation("r2", "e2", "e3", "depends_on"),
    ]
    return _make_state(entities, relations)


@pytest.fixture
def query_service(sample_state):
    """QueryService wired to sample_state with mock vector index."""
    mock_vi = Mock()
    mock_vi.search.return_value = []

    mock_similarity = Mock()
    mock_similarity.find_similar.return_value = []

    mock_tt = Mock()

    def resolve(name_or_id):
        if name_or_id in sample_state.entities:
            return name_or_id
        return sample_state.get_entity_id_by_name(name_or_id)

    return QueryService(
        get_state=lambda: sample_state,
        get_active_state=lambda: sample_state,
        get_vector_index=lambda: mock_vi,
        similarity=lambda: mock_similarity,
        time_traveler=lambda: mock_tt,
        resolve_entity=resolve,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Direct QueryService Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQueryServiceDirect:
    """Test QueryService methods without MemoryEngine."""

    def test_read_graph_returns_all(self, query_service, sample_state):
        result = query_service.read_graph()
        assert len(result["entities"]) == 3
        assert len(result["relations"]) == 2

    def test_search_graph_by_name(self, query_service):
        result = query_service.search_graph("Python")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Python"

    def test_search_graph_by_observation(self, query_service):
        result = query_service.search_graph("database")
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "PostgreSQL"

    def test_search_graph_no_match(self, query_service):
        result = query_service.search_graph("nonexistent")
        assert len(result["entities"]) == 0

    def test_open_nodes_returns_entity_and_neighbors(self, query_service):
        result = query_service.open_nodes(["FastAPI"])
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "FastAPI"
        # FastAPI connects to Python and PostgreSQL
        assert len(result["neighbors"]) == 2
        neighbor_names = {n["name"] for n in result["neighbors"]}
        assert neighbor_names == {"Python", "PostgreSQL"}

    def test_open_nodes_unknown_entity(self, query_service):
        result = query_service.open_nodes(["NonExistent"])
        assert len(result["entities"]) == 0

    def test_get_recent_entities(self, query_service):
        result = query_service.get_recent_entities(limit=2)
        assert len(result) == 2

    def test_get_entities_by_type(self, query_service):
        result = query_service.get_entities_by_type("concept")
        assert len(result) == 3

    def test_get_entity_neighbors(self, query_service):
        result = query_service.get_entity_neighbors("e2")
        assert result["entity"]["name"] == "FastAPI"
        assert len(result["outgoing"]) == 1  # FastAPI -> PostgreSQL
        assert len(result["incoming"]) == 1  # Python -> FastAPI

    def test_get_entity_neighbors_not_found(self, query_service):
        result = query_service.get_entity_neighbors("nonexistent")
        assert result["entity"] is None
        assert result["neighbors"] == []

    def test_find_orphans(self, sample_state):
        """An entity with no relations should be detected as orphan."""
        entities = {
            "e1": _make_entity("e1", "Connected"),
            "e2": _make_entity("e2", "Orphan"),
        }
        relations = [_make_relation("r1", "e1", "e1", "self_ref")]
        state = _make_state(entities, relations)

        qs = QueryService(
            get_state=lambda: state,
            get_active_state=lambda: state,
            get_vector_index=Mock(side_effect=RuntimeError("no vectors")),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in state.entities else None,
        )
        orphans = qs.find_orphans()
        assert len(orphans) == 1
        assert orphans[0]["name"] == "Orphan"

    def test_get_structure(self, query_service):
        result = query_service.get_structure()
        assert result["entity_count"] == 3
        assert result["relation_count"] == 2

    def test_get_structure_specific_ids(self, query_service):
        result = query_service.get_structure(entity_ids=["e1"], include_neighbors=False)
        assert result["entity_count"] == 1

    def test_get_graph_health_empty(self):
        """Health check on empty graph should not error."""
        empty_state = _make_state()
        qs = QueryService(
            get_state=lambda: empty_state,
            get_active_state=lambda: empty_state,
            get_vector_index=Mock(side_effect=RuntimeError("no vectors")),
            similarity=lambda: Mock(find_similar=Mock(return_value=[])),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: None,
        )
        result = qs.get_graph_health()
        assert result["summary"]["total_entities"] == 0
        assert "healthy" in result["recommendations"][0].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Recall Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRecall:
    """Test recall() at various depths."""

    def test_shallow_returns_summary(self, query_service):
        result = query_service.recall(depth="shallow")
        assert result["depth"] == "shallow"
        assert result["entity_count"] == 3
        assert isinstance(result["content"], str)

    def test_shallow_graph_format(self, query_service):
        result = query_service.recall(depth="shallow", format="graph")
        assert result["format"] == "graph"
        assert "content" in result

    def test_medium_returns_results(self, query_service):
        result = query_service.recall(depth="medium")
        assert result["depth"] == "medium"

    def test_deep_returns_results(self, query_service):
        result = query_service.recall(depth="deep", focus=["Python"])
        assert result["depth"] == "deep"

    def test_invalid_depth_raises(self, query_service):
        with pytest.raises(ValueError, match="Invalid depth"):
            query_service.recall(depth="ultra")


# ─────────────────────────────────────────────────────────────────────────────
# Co-access Callback Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCoAccessCallback:
    """Verify on_co_access fires during recall."""

    def test_co_access_fires_on_medium(self, sample_state):
        callback = Mock()
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(search=Mock(return_value=[])),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else None,
            on_co_access=callback,
        )
        qs.recall(depth="medium")
        callback.assert_called_once()

    def test_co_access_fires_on_deep(self, sample_state):
        callback = Mock()
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(search=Mock(return_value=[])),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else None,
            on_co_access=callback,
        )
        qs.recall(depth="deep", focus=["Python"])
        callback.assert_called_once()

    def test_co_access_not_fired_on_shallow(self, sample_state):
        callback = Mock()
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(search=Mock(return_value=[])),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else None,
            on_co_access=callback,
        )
        qs.recall(depth="shallow")
        callback.assert_not_called()

    def test_no_callback_configured(self, sample_state):
        """Should not error when on_co_access is None."""
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(search=Mock(return_value=[])),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else None,
            on_co_access=None,
        )
        # Should not raise
        qs.recall(depth="medium")


# ─────────────────────────────────────────────────────────────────────────────
# Degraded Mode (None vector_index)
# ─────────────────────────────────────────────────────────────────────────────


class TestDegradedMode:
    """Test QueryService with unavailable vector index."""

    def test_search_structure_without_vectors(self, sample_state):
        """search_structure should fall back to keyword search when vectors fail."""
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=Mock(side_effect=RuntimeError("model not loaded")),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else None,
        )
        result = qs.search_structure("Python")
        assert result["matched_count"] >= 1

    def test_recall_medium_without_vectors(self, sample_state):
        """recall at medium depth should work without vectors."""
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=Mock(side_effect=RuntimeError("model not loaded")),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else None,
        )
        result = qs.recall(depth="medium")
        assert result["depth"] == "medium"

    def test_suggest_relations_without_vectors(self, sample_state):
        """suggest_relations should fall back gracefully."""
        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=Mock(side_effect=RuntimeError("model not loaded")),
            similarity=lambda: Mock(),
            time_traveler=lambda: Mock(),
            resolve_entity=lambda x: x if x in sample_state.entities else sample_state.get_entity_id_by_name(x),
        )
        result = qs.suggest_relations("Python")
        # Should not error, may return empty or co-occurrence suggestions
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# Delegation Layer — All Engine Delegates Match QueryService
# ─────────────────────────────────────────────────────────────────────────────


class TestDelegationCompleteness:
    """Verify all delegated methods on MemoryEngine match QueryService."""

    def test_all_query_public_methods_are_delegated(self):
        """Every public method on QueryService should have a delegate on MemoryEngine."""
        from mnemograph.engine import MemoryEngine

        qs_public = {
            name for name, _ in inspect.getmembers(QueryService, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

        engine_public = {
            name for name, _ in inspect.getmembers(MemoryEngine, predicate=inspect.isfunction)
            if not name.startswith("_")
        }

        missing = qs_public - engine_public
        assert missing == set(), f"QueryService methods missing from MemoryEngine: {missing}"

    def test_engine_recall_delegates_to_query(self):
        """Engine.recall() should produce the same result as going through QueryService."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from mnemograph.engine import MemoryEngine

            engine = MemoryEngine(Path(tmpdir), "test-session")
            engine.create_entities([
                {"name": "TestConcept", "entityType": "concept",
                 "observations": ["A test entity"]},
            ])

            # Call through engine
            engine_result = engine.recall(depth="shallow")
            # Call through QueryService directly
            qs_result = engine._query.recall(depth="shallow")

            assert engine_result["depth"] == qs_result["depth"]
            assert engine_result["entity_count"] == qs_result["entity_count"]

    def test_engine_search_graph_delegates(self):
        """Engine.search_graph() should delegate to QueryService."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from mnemograph.engine import MemoryEngine

            engine = MemoryEngine(Path(tmpdir), "test-session")
            engine.create_entities([
                {"name": "Alpha", "entityType": "concept"},
            ])

            result = engine.search_graph("Alpha")
            assert len(result["entities"]) == 1
            assert result["entities"][0]["name"] == "Alpha"


# ─────────────────────────────────────────────────────────────────────────────
# Time Travel Delegates
# ─────────────────────────────────────────────────────────────────────────────


class TestTimeTravelDelegates:
    """Test that time travel methods delegate correctly."""

    def test_state_at_delegates(self, sample_state):
        mock_tt = Mock()
        mock_tt.state_at.return_value = sample_state

        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(),
            similarity=lambda: Mock(),
            time_traveler=lambda: mock_tt,
            resolve_entity=lambda x: x,
        )
        result = qs.state_at("2025-01-15")
        mock_tt.state_at.assert_called_once_with("2025-01-15")

    def test_events_between_delegates(self, sample_state):
        mock_tt = Mock()
        mock_tt.events_between.return_value = []

        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(),
            similarity=lambda: Mock(),
            time_traveler=lambda: mock_tt,
            resolve_entity=lambda x: x,
        )
        qs.events_between("2025-01-10", "2025-01-20")
        mock_tt.events_between.assert_called_once_with("2025-01-10", "2025-01-20")

    def test_get_entity_history_resolves_name(self, sample_state):
        mock_tt = Mock()
        mock_tt.get_entity_history.return_value = []

        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(),
            similarity=lambda: Mock(),
            time_traveler=lambda: mock_tt,
            resolve_entity=lambda x: "e1" if x == "Python" else None,
        )
        qs.get_entity_history("Python")
        mock_tt.get_entity_history.assert_called_once_with("e1")

    def test_get_entity_history_unknown_name(self, sample_state):
        mock_tt = Mock()

        qs = QueryService(
            get_state=lambda: sample_state,
            get_active_state=lambda: sample_state,
            get_vector_index=lambda: Mock(),
            similarity=lambda: Mock(),
            time_traveler=lambda: mock_tt,
            resolve_entity=lambda x: None,
        )
        result = qs.get_entity_history("NonExistent")
        assert result == []
        mock_tt.get_entity_history.assert_not_called()
