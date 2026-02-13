"""Tests for Edge Weights feature.

Tests recency score, co-access score, explicit weight, and weighted traversal.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from mnemograph.models import Entity, Observation, Relation
from mnemograph.state import GraphState, materialize
from mnemograph.weights import (
    compute_recency_score,
    compute_co_access_score,
    update_co_access_scores,
    weighted_bfs,
    get_strongest_connections,
)


# --- Recency Score Tests ---


class TestRecencyScore:
    """Tests for compute_recency_score()."""

    def test_fresh_relation_has_score_1(self):
        """Just-accessed relation should have recency 1.0."""
        now = datetime.now(timezone.utc)
        score = compute_recency_score(now, reference_time=now)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_30_day_old_relation_has_score_half(self):
        """Relation accessed 30 days ago should have ~0.5."""
        now = datetime.now(timezone.utc)
        last_accessed = now - timedelta(days=30)
        score = compute_recency_score(last_accessed, half_life_days=30.0, reference_time=now)
        assert score == pytest.approx(0.5, abs=0.02)

    def test_60_day_old_relation_has_score_quarter(self):
        """Relation accessed 60 days ago should have ~0.25."""
        now = datetime.now(timezone.utc)
        last_accessed = now - timedelta(days=60)
        score = compute_recency_score(last_accessed, half_life_days=30.0, reference_time=now)
        assert score == pytest.approx(0.25, abs=0.02)

    def test_very_old_relation_approaches_zero(self):
        """180+ day old relation should be near 0."""
        now = datetime.now(timezone.utc)
        last_accessed = now - timedelta(days=180)
        score = compute_recency_score(last_accessed, half_life_days=30.0, reference_time=now)
        assert score < 0.02

    def test_custom_half_life(self):
        """Different half-life values should decay differently."""
        now = datetime.now(timezone.utc)
        last_accessed = now - timedelta(days=7)

        # 7-day half-life: after 7 days, should be ~0.5
        score_7day = compute_recency_score(last_accessed, half_life_days=7.0, reference_time=now)
        assert score_7day == pytest.approx(0.5, abs=0.02)

        # 30-day half-life: after 7 days, should be higher (~0.85)
        score_30day = compute_recency_score(last_accessed, half_life_days=30.0, reference_time=now)
        assert score_30day > 0.8

    def test_future_date_returns_1(self):
        """Future date should be treated as fresh (score 1.0)."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=10)
        score = compute_recency_score(future, reference_time=now)
        assert score == 1.0

    def test_score_bounded_0_to_1(self):
        """Score should always be between 0.0 and 1.0."""
        now = datetime.now(timezone.utc)

        # Very old
        very_old = now - timedelta(days=10000)
        score = compute_recency_score(very_old, reference_time=now)
        assert 0.0 <= score <= 1.0

        # Just now
        score = compute_recency_score(now, reference_time=now)
        assert 0.0 <= score <= 1.0


# --- Co-Access Score Tests ---


class TestCoAccessScore:
    """Tests for update_co_access_scores()."""

    def setup_method(self):
        """Create test state with entities and relations."""
        self.state = GraphState()

        # Add entities
        e1 = Entity(id="e1", name="Alpha", type="concept")
        e2 = Entity(id="e2", name="Beta", type="concept")
        e3 = Entity(id="e3", name="Gamma", type="concept")
        e4 = Entity(id="e4", name="Delta", type="concept")

        self.state.entities = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}

        # Add relations
        r1 = Relation(id="r1", from_entity="e1", to_entity="e2", type="relates_to")
        r2 = Relation(id="r2", from_entity="e2", to_entity="e3", type="relates_to")
        r3 = Relation(id="r3", from_entity="e3", to_entity="e4", type="relates_to")

        self.state.relations = [r1, r2, r3]

    def test_co_accessed_relations_strengthen(self):
        """Relations between co-retrieved entities should increase density."""
        # Retrieve e1 and e2 together
        retrieved = {"e1", "e2"}
        updated = update_co_access_scores(self.state, retrieved)

        assert "r1" in updated
        assert self.state.relations[0].co_access_density == pytest.approx(1.0, abs=0.01)
        # Score should be > 0 (sigmoid of density=1.0)
        assert self.state.relations[0].co_access_score > 0

    def test_non_co_accessed_unchanged(self):
        """Relations not involved shouldn't change."""
        # Retrieve only e1
        retrieved = {"e1"}
        update_co_access_scores(self.state, retrieved)

        # r1 connects e1 and e2, but only e1 retrieved
        assert self.state.relations[0].co_access_density == 0.0

    def test_score_bounded_0_to_1(self):
        """Score should always be between 0 and 1, even with high density."""
        retrieved = {"e1", "e2"}

        # Apply co-access many times to build up density
        for _ in range(100):
            update_co_access_scores(self.state, retrieved)

        assert self.state.relations[0].co_access_score <= 1.0
        assert self.state.relations[0].co_access_score >= 0.0

    def test_access_count_increments(self):
        """access_count should track co-access events."""
        retrieved = {"e1", "e2"}

        update_co_access_scores(self.state, retrieved)
        assert self.state.relations[0].access_count == 1

        update_co_access_scores(self.state, retrieved)
        assert self.state.relations[0].access_count == 2

    def test_last_accessed_updated(self):
        """last_accessed should be updated on co-access."""
        old_time = self.state.relations[0].last_accessed
        retrieved = {"e1", "e2"}

        update_co_access_scores(self.state, retrieved)

        new_time = self.state.relations[0].last_accessed
        assert new_time >= old_time

    def test_multiple_relations_updated(self):
        """Multiple relations can be updated in one call."""
        # Retrieve e1, e2, e3 together - should update r1 and r2
        retrieved = {"e1", "e2", "e3"}
        updated = update_co_access_scores(self.state, retrieved)

        assert len(updated) == 2
        assert "r1" in updated
        assert "r2" in updated


# --- Combined Weight Tests ---


class TestCombinedWeight:
    """Tests for the combined weight property."""

    def test_weight_formula(self):
        """Combined weight should be 0.4*recency + 0.3*co_access + 0.3*explicit."""
        now = datetime.now(timezone.utc)
        r = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            explicit_weight=1.0,
            co_access_density=10.0,  # High density → co_access_score ≈ 1.0
            last_accessed=now,
        )

        # All components near max: 0.4*1.0 + 0.3*~1.0 + 0.3*1.0 ≈ 1.0
        assert r.weight == pytest.approx(1.0, abs=0.03)

    def test_all_components_contribute(self):
        """Each component should affect final weight."""
        now = datetime.now(timezone.utc)

        # All zeros (except explicit default)
        r_base = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            explicit_weight=0.0,
            co_access_density=0.0,
            last_accessed=now - timedelta(days=365),  # Very old
        )
        base_weight = r_base.weight

        # Increase explicit
        r_explicit = Relation(
            id="r2",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            explicit_weight=1.0,
            co_access_density=0.0,
            last_accessed=now - timedelta(days=365),
        )
        assert r_explicit.weight > base_weight

        # Increase co_access (high density → high score)
        r_coaccess = Relation(
            id="r3",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            explicit_weight=0.0,
            co_access_density=10.0,  # High density → score ≈ 1.0
            last_accessed=now - timedelta(days=365),
        )
        assert r_coaccess.weight > base_weight

    def test_recency_property_matches_formula(self):
        """recency_score property should use exponential decay."""
        now = datetime.now(timezone.utc)
        r = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            last_accessed=now - timedelta(days=30),
        )

        # Should be approximately 0.5 with 30-day half-life
        assert r.recency_score == pytest.approx(0.5, abs=0.02)


# --- Weighted BFS Tests ---


class TestWeightedBFS:
    """Tests for weighted_bfs()."""

    def setup_method(self):
        """Create test state with entities and weighted relations."""
        self.state = GraphState()
        now = datetime.now(timezone.utc)

        # Add entities: A -> B -> C, A -> D -> E
        # Make D -> E path have higher weight than B -> C
        entities = [
            Entity(id="A", name="A", type="concept"),
            Entity(id="B", name="B", type="concept"),
            Entity(id="C", name="C", type="concept"),
            Entity(id="D", name="D", type="concept"),
            Entity(id="E", name="E", type="concept"),
        ]
        self.state.entities = {e.id: e for e in entities}

        # Relations with different weights
        self.state.relations = [
            # A -> B (low weight)
            Relation(
                id="r_ab", from_entity="A", to_entity="B", type="connects",
                explicit_weight=0.2, co_access_density=0.0, last_accessed=now,
            ),
            # B -> C (low weight)
            Relation(
                id="r_bc", from_entity="B", to_entity="C", type="connects",
                explicit_weight=0.2, co_access_density=0.0, last_accessed=now,
            ),
            # A -> D (high weight)
            Relation(
                id="r_ad", from_entity="A", to_entity="D", type="connects",
                explicit_weight=0.9, co_access_density=5.0, last_accessed=now,
            ),
            # D -> E (high weight)
            Relation(
                id="r_de", from_entity="D", to_entity="E", type="connects",
                explicit_weight=0.9, co_access_density=5.0, last_accessed=now,
            ),
        ]

        # Rebuild indices after manually setting up the state
        self.state._rebuild_indices()

    def test_prefers_high_weight_paths(self):
        """Should visit high-weight edges before low-weight."""
        result = weighted_bfs(["A"], self.state, max_depth=3, max_nodes=5)

        # Should start with A
        assert result[0] == "A"

        # D should be visited before B (higher weight edge from A)
        d_idx = result.index("D")
        b_idx = result.index("B")
        assert d_idx < b_idx, f"D (idx={d_idx}) should be before B (idx={b_idx})"

    def test_respects_max_depth(self):
        """Should not traverse beyond max_depth."""
        result = weighted_bfs(["A"], self.state, max_depth=1, max_nodes=10)

        # At depth 1, should only reach direct neighbors of A
        assert "A" in result
        assert "B" in result or "D" in result
        # C and E require depth 2
        assert "C" not in result
        assert "E" not in result

    def test_respects_max_nodes(self):
        """Should stop after max_nodes."""
        result = weighted_bfs(["A"], self.state, max_depth=5, max_nodes=3)

        assert len(result) <= 3

    def test_respects_min_weight(self):
        """Should skip edges below min_weight."""
        # With high min_weight, low-weight edges won't be traversed
        result = weighted_bfs(["A"], self.state, max_depth=3, max_nodes=10, min_weight=0.5)

        # Only high-weight path A -> D -> E should be traversed
        assert "A" in result
        assert "D" in result
        assert "E" in result
        # B and C are behind low-weight edges
        assert "B" not in result
        assert "C" not in result

    def test_handles_missing_start_entities(self):
        """Should gracefully handle non-existent start entities."""
        result = weighted_bfs(["nonexistent"], self.state, max_depth=3)
        assert result == []

    def test_handles_empty_state(self):
        """Should handle empty graph state."""
        empty_state = GraphState()
        result = weighted_bfs(["A"], empty_state, max_depth=3)
        assert result == []


# --- Get Strongest Connections Tests ---


class TestGetStrongestConnections:
    """Tests for get_strongest_connections()."""

    def setup_method(self):
        """Create test state with entities and weighted relations."""
        self.state = GraphState()
        now = datetime.now(timezone.utc)

        # Central entity with multiple connections of varying weights
        entities = [
            Entity(id="center", name="Center", type="concept"),
            Entity(id="strong1", name="Strong1", type="concept"),
            Entity(id="strong2", name="Strong2", type="concept"),
            Entity(id="weak1", name="Weak1", type="concept"),
            Entity(id="weak2", name="Weak2", type="concept"),
        ]
        self.state.entities = {e.id: e for e in entities}

        self.state.relations = [
            Relation(
                id="r1", from_entity="center", to_entity="strong1", type="connects",
                explicit_weight=0.9, co_access_density=7.0, last_accessed=now,
            ),
            Relation(
                id="r2", from_entity="center", to_entity="strong2", type="connects",
                explicit_weight=0.8, co_access_density=6.0, last_accessed=now,
            ),
            Relation(
                id="r3", from_entity="center", to_entity="weak1", type="connects",
                explicit_weight=0.1, co_access_density=0.0, last_accessed=now - timedelta(days=60),
            ),
            Relation(
                id="r4", from_entity="weak2", to_entity="center", type="connects",  # Incoming
                explicit_weight=0.2, co_access_density=1.0, last_accessed=now - timedelta(days=30),
            ),
        ]

        # Rebuild indices after manually setting up the state
        self.state._rebuild_indices()

    def test_returns_sorted_by_weight(self):
        """Should return connections sorted by weight descending."""
        result = get_strongest_connections(self.state, "center", limit=10)

        weights = [r.weight for r, _ in result]
        assert weights == sorted(weights, reverse=True)

    def test_respects_limit(self):
        """Should respect the limit parameter."""
        result = get_strongest_connections(self.state, "center", limit=2)
        assert len(result) == 2

    def test_includes_both_directions(self):
        """Should include both outgoing and incoming relations."""
        result = get_strongest_connections(self.state, "center", limit=10)

        # r4 is incoming (weak2 -> center)
        relation_ids = [r.id for r, _ in result]
        assert "r4" in relation_ids

    def test_returns_correct_neighbor(self):
        """Should return the correct neighbor entity ID."""
        result = get_strongest_connections(self.state, "center", limit=10)

        # For r1 (center -> strong1), neighbor should be strong1
        for rel, neighbor in result:
            if rel.id == "r1":
                assert neighbor == "strong1"
            elif rel.id == "r4":
                # For incoming relation, neighbor is the source
                assert neighbor == "weak2"

    def test_handles_entity_with_no_relations(self):
        """Should return empty list for entity with no relations."""
        # Add isolated entity
        self.state.entities["isolated"] = Entity(id="isolated", name="Isolated", type="concept")

        result = get_strongest_connections(self.state, "isolated", limit=10)
        assert result == []


# --- Explicit Weight Event Tests ---


class TestExplicitWeightEvent:
    """Tests for explicit weight updates via events."""

    def test_update_weight_event_replayed(self):
        """update_weight events should be replayed correctly."""
        from mnemograph.models import MemoryEvent

        now = datetime.now(timezone.utc)

        events = [
            # Create entities
            MemoryEvent(
                id="ev1", ts=now, op="create_entity", session_id="test", source="cc",
                data={
                    "id": "e1", "name": "Entity1", "type": "concept",
                    "observations": [], "created_at": now.isoformat(),
                    "updated_at": now.isoformat(), "created_by": "test",
                },
            ),
            MemoryEvent(
                id="ev2", ts=now, op="create_entity", session_id="test", source="cc",
                data={
                    "id": "e2", "name": "Entity2", "type": "concept",
                    "observations": [], "created_at": now.isoformat(),
                    "updated_at": now.isoformat(), "created_by": "test",
                },
            ),
            # Create relation
            MemoryEvent(
                id="ev3", ts=now, op="create_relation", session_id="test", source="cc",
                data={
                    "id": "r1", "from_entity": "e1", "to_entity": "e2",
                    "type": "relates_to", "created_at": now.isoformat(), "created_by": "test",
                },
            ),
            # Update weight
            MemoryEvent(
                id="ev4", ts=now + timedelta(seconds=1), op="update_weight",
                session_id="test", source="cc",
                data={
                    "relation_id": "r1", "weight_type": "explicit",
                    "old_value": 0.5, "new_value": 0.9,
                },
            ),
        ]

        state = materialize(events)

        # Check relation has updated weight
        assert len(state.relations) == 1
        assert state.relations[0].explicit_weight == 0.9

    def test_multiple_weight_updates(self):
        """Multiple weight updates should apply in order."""
        from mnemograph.models import MemoryEvent

        now = datetime.now(timezone.utc)

        events = [
            MemoryEvent(
                id="ev1", ts=now, op="create_entity", session_id="test", source="cc",
                data={
                    "id": "e1", "name": "E1", "type": "concept",
                    "observations": [], "created_at": now.isoformat(),
                    "updated_at": now.isoformat(), "created_by": "test",
                },
            ),
            MemoryEvent(
                id="ev2", ts=now, op="create_entity", session_id="test", source="cc",
                data={
                    "id": "e2", "name": "E2", "type": "concept",
                    "observations": [], "created_at": now.isoformat(),
                    "updated_at": now.isoformat(), "created_by": "test",
                },
            ),
            MemoryEvent(
                id="ev3", ts=now, op="create_relation", session_id="test", source="cc",
                data={
                    "id": "r1", "from_entity": "e1", "to_entity": "e2",
                    "type": "relates_to", "created_at": now.isoformat(), "created_by": "test",
                },
            ),
            # First update
            MemoryEvent(
                id="ev4", ts=now + timedelta(seconds=1), op="update_weight",
                session_id="test", source="cc",
                data={
                    "relation_id": "r1", "weight_type": "explicit",
                    "old_value": 0.5, "new_value": 0.9,
                },
            ),
            # Second update
            MemoryEvent(
                id="ev5", ts=now + timedelta(seconds=2), op="update_weight",
                session_id="test", source="cc",
                data={
                    "relation_id": "r1", "weight_type": "explicit",
                    "old_value": 0.9, "new_value": 0.3,
                },
            ),
        ]

        state = materialize(events)

        # Should have final value
        assert state.relations[0].explicit_weight == 0.3


# --- D7: Default Weight Tests ---


class TestDefaultWeight:
    """Tests for D7: Relation with default explicit_weight=0.3 should have low
    combined weight when recency=0 and co_access=0."""

    def test_default_weight_low_when_stale(self):
        """New relation with no usage should have low weight when very old."""
        now = datetime.now(timezone.utc)
        r = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            explicit_weight=0.3,  # default
            co_access_density=0.0,  # density=0 → co_access_score ≈ 0.018
            last_accessed=now - timedelta(days=365),  # very old → recency ≈ 0
        )
        # weight = 0.4 * ~0 + 0.3 * ~0.018 + 0.3 * 0.3 ≈ 0.095
        assert r.weight < 0.15, (
            f"Stale relation with default weight should be < 0.15, got {r.weight}"
        )

    def test_default_explicit_weight_is_0_3(self):
        """Verify the default explicit_weight is 0.3 (D7 change from 0.5)."""
        r = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
        )
        assert r.explicit_weight == 0.3


# --- D6: Leaky Integrator / Sigmoid Tests ---


class TestComputeCoAccessScore:
    """Tests for compute_co_access_score() sigmoid function."""

    def test_sigmoid_zero_density_near_zero(self):
        """density=0 should produce score near 0 (below midpoint)."""
        score = compute_co_access_score(0.0)
        assert score < 0.05, f"sigmoid(0) should be near 0, got {score}"

    def test_sigmoid_midpoint_is_half(self):
        """density=midpoint (5.0) should produce score ≈ 0.5."""
        from mnemograph.constants import CO_ACCESS_SIGMOID_MIDPOINT
        score = compute_co_access_score(CO_ACCESS_SIGMOID_MIDPOINT)
        assert score == pytest.approx(0.5, abs=0.01), (
            f"sigmoid(midpoint) should be 0.5, got {score}"
        )

    def test_sigmoid_high_density_near_one(self):
        """density=10 should produce score near 1.0 (above midpoint)."""
        score = compute_co_access_score(10.0)
        assert score > 0.95, f"sigmoid(10) should be near 1.0, got {score}"

    def test_sigmoid_monotonically_increasing(self):
        """Higher density should always produce higher score."""
        scores = [compute_co_access_score(d) for d in [0, 1, 2, 5, 8, 10, 20]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Score should increase: {scores[i]} < {scores[i+1]} at densities"
            )


class TestLeakyIntegrator:
    """Tests for the leaky integrator decay + pulse model in update_co_access_scores."""

    def setup_method(self):
        """Create test state with two entities and a relation."""
        self.state = GraphState()

        e1 = Entity(id="e1", name="Alpha", type="concept")
        e2 = Entity(id="e2", name="Beta", type="concept")
        self.state.entities = {"e1": e1, "e2": e2}

        r1 = Relation(id="r1", from_entity="e1", to_entity="e2", type="relates_to")
        self.state.relations = [r1]

    def test_density_increases_with_co_access(self):
        """Co-access should increase density by the increment."""
        update_co_access_scores(self.state, {"e1", "e2"})
        assert self.state.relations[0].co_access_density == pytest.approx(1.0, abs=0.01)

    def test_density_accumulates(self):
        """Multiple rapid co-accesses should accumulate density."""
        update_co_access_scores(self.state, {"e1", "e2"})
        update_co_access_scores(self.state, {"e1", "e2"})
        # Two quick pulses → density ≈ 2.0 (minimal decay between)
        assert self.state.relations[0].co_access_density > 1.5

    def test_density_decays_over_time(self):
        """Density should decay when time passes between co-accesses."""
        # First co-access
        update_co_access_scores(self.state, {"e1", "e2"})
        density_after_first = self.state.relations[0].co_access_density

        # Simulate 30 days passing by backdating last_co_access
        self.state.relations[0].last_co_access = (
            datetime.now(timezone.utc) - timedelta(days=30)
        )

        # Second co-access — old density decays to ~0.5, then +1.0 pulse
        update_co_access_scores(self.state, {"e1", "e2"})
        density_after_second = self.state.relations[0].co_access_density

        # Should be roughly 0.5 * density_after_first + 1.0 ≈ 1.5
        assert density_after_second == pytest.approx(
            density_after_first * 0.5 + 1.0, abs=0.1
        )

    def test_last_co_access_tracked(self):
        """last_co_access should be set after update."""
        assert self.state.relations[0].last_co_access is None
        update_co_access_scores(self.state, {"e1", "e2"})
        assert self.state.relations[0].last_co_access is not None

    def test_combined_weight_with_leaky_integrator(self):
        """Relation.weight should produce sensible values with new co-access model."""
        now = datetime.now(timezone.utc)

        # Fresh relation with some co-access density
        r = Relation(
            id="r1",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            explicit_weight=0.5,
            co_access_density=5.0,  # midpoint → score ≈ 0.5
            last_accessed=now,
        )

        # weight = 0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 0.5 = 0.4 + 0.15 + 0.15 = 0.7
        assert r.weight == pytest.approx(0.7, abs=0.05)

        # Zero density, fresh access, default explicit
        r_zero = Relation(
            id="r2",
            from_entity="e1",
            to_entity="e2",
            type="relates_to",
            co_access_density=0.0,
            last_accessed=now,
        )
        # co_access_score(0) ≈ 0.018, recency=1.0, explicit=0.3
        # weight ≈ 0.4 * 1.0 + 0.3 * 0.018 + 0.3 * 0.3 ≈ 0.495
        assert 0.3 < r_zero.weight < 0.6


class TestCacheMigration:
    """Tests for old-format cache migration in _load_co_access_cache."""

    def test_old_format_cache_migrated(self):
        """Old-format cache (co_access_score) should be migrated to density."""
        import tempfile
        import json
        from pathlib import Path
        from mnemograph.engine import MemoryEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test-session")

            # Create entities + relation
            engine.create_entities_force([
                {"name": "Alpha", "entityType": "concept"},
                {"name": "Beta", "entityType": "concept"},
            ])
            engine.create_relations([
                {"from": "Alpha", "to": "Beta", "relationType": "links_to"},
            ])

            rel_id = engine.state.relations[0].id

            # Write old-format cache
            old_cache = {rel_id: {"co_access_score": 0.5}}
            cache_path = Path(tmpdir) / "co_access_cache.json"
            cache_path.write_text(json.dumps(old_cache))

            # Reload engine to trigger cache loading
            engine2 = MemoryEngine(Path(tmpdir), "test-session")

            # Find the relation
            rel = next(r for r in engine2.state.relations if r.id == rel_id)

            # Density should be set (inverse sigmoid of 0.5 = midpoint = 5.0)
            assert rel.co_access_density > 0, (
                "Old cache format should be migrated to density"
            )
            # Score 0.5 → density ≈ midpoint (5.0)
            assert rel.co_access_density == pytest.approx(5.0, abs=0.5)

    def test_new_format_cache_loaded(self):
        """New-format cache (co_access_density) should be loaded directly."""
        import tempfile
        import json
        from pathlib import Path
        from mnemograph.engine import MemoryEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryEngine(Path(tmpdir), "test-session")

            engine.create_entities_force([
                {"name": "Alpha", "entityType": "concept"},
                {"name": "Beta", "entityType": "concept"},
            ])
            engine.create_relations([
                {"from": "Alpha", "to": "Beta", "relationType": "links_to"},
            ])

            rel_id = engine.state.relations[0].id

            # Write new-format cache
            new_cache = {
                rel_id: {
                    "co_access_density": 3.7,
                    "last_co_access": "2026-01-15T12:00:00+00:00",
                }
            }
            cache_path = Path(tmpdir) / "co_access_cache.json"
            cache_path.write_text(json.dumps(new_cache))

            # Reload
            engine2 = MemoryEngine(Path(tmpdir), "test-session")
            rel = next(r for r in engine2.state.relations if r.id == rel_id)

            assert rel.co_access_density == pytest.approx(3.7, abs=0.01)
            assert rel.last_co_access is not None
