"""Tests for the branch system data model, validation, and BranchManager."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from mnemograph.models import (
    Branch,
    Entity,
    Relation,
    make_main_branch,
    validate_branch_name,
    BRANCH_TYPES,
)
from mnemograph.branches import BranchManager
from mnemograph.state import GraphState


# ─────────────────────────────────────────────────────────────────────────────
# Branch Model Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBranchModel:
    """Tests for the Branch dataclass."""

    def test_create_branch_with_defaults(self):
        """Branch should have sensible defaults."""
        branch = Branch(name="project/test")
        assert branch.name == "project/test"
        assert branch.description == ""
        assert branch.entity_ids == set()
        assert branch.relation_ids == set()
        assert branch.parent == "main"
        assert branch.is_active is True
        assert branch.auto_include_depth == 1

    def test_create_branch_with_all_fields(self):
        """Branch should accept all fields."""
        branch = Branch(
            name="feature/auth",
            description="Authentication feature",
            entity_ids={"e1", "e2"},
            relation_ids={"r1"},
            parent="project/backend",
            is_active=True,
            auto_include_depth=2,
        )
        assert branch.name == "feature/auth"
        assert branch.description == "Authentication feature"
        assert branch.entity_ids == {"e1", "e2"}
        assert branch.relation_ids == {"r1"}
        assert branch.parent == "project/backend"
        assert branch.auto_include_depth == 2

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization should be lossless."""
        original = Branch(
            name="project/test",
            description="Test project",
            entity_ids={"e1", "e2", "e3"},
            relation_ids={"r1", "r2"},
            parent="main",
            is_active=True,
            auto_include_depth=2,
            created_from_commit="abc1234",
        )

        data = original.to_dict()
        restored = Branch.from_dict(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.entity_ids == original.entity_ids
        assert restored.relation_ids == original.relation_ids
        assert restored.parent == original.parent
        assert restored.is_active == original.is_active
        assert restored.auto_include_depth == original.auto_include_depth
        assert restored.created_from_commit == original.created_from_commit

    def test_to_dict_sorts_ids(self):
        """to_dict should sort IDs for deterministic output."""
        branch = Branch(
            name="project/test",
            entity_ids={"z", "a", "m"},
            relation_ids={"r3", "r1", "r2"},
        )
        data = branch.to_dict()
        assert data["entity_ids"] == ["a", "m", "z"]
        assert data["relation_ids"] == ["r1", "r2", "r3"]

    def test_includes_entity_on_regular_branch(self):
        """includes_entity should check entity_ids set."""
        branch = Branch(name="project/test", entity_ids={"e1", "e2"})
        assert branch.includes_entity("e1") is True
        assert branch.includes_entity("e2") is True
        assert branch.includes_entity("e3") is False

    def test_includes_relation_on_regular_branch(self):
        """includes_relation should check relation_ids set."""
        branch = Branch(name="project/test", relation_ids={"r1", "r2"})
        assert branch.includes_relation("r1") is True
        assert branch.includes_relation("r2") is True
        assert branch.includes_relation("r3") is False

    def test_main_branch_includes_everything(self):
        """Main branch should include all entities and relations."""
        main = make_main_branch()
        assert main.includes_entity("any-entity-id") is True
        assert main.includes_relation("any-relation-id") is True

    def test_main_branch_has_empty_sets(self):
        """Main branch should have empty sets (special case)."""
        main = make_main_branch()
        assert main.entity_ids == set()
        assert main.relation_ids == set()
        assert main.name == "main"
        assert main.parent is None

    def test_from_dict_handles_missing_fields(self):
        """from_dict should handle minimal data."""
        data = {"name": "project/minimal"}
        branch = Branch.from_dict(data)
        assert branch.name == "project/minimal"
        assert branch.description == ""
        assert branch.entity_ids == set()
        assert branch.parent == "main"

    def test_from_dict_handles_string_datetime(self):
        """from_dict should parse ISO datetime strings."""
        data = {
            "name": "project/test",
            "created_at": "2025-01-15T10:30:00+00:00",
        }
        branch = Branch.from_dict(data)
        assert branch.created_at.year == 2025
        assert branch.created_at.month == 1
        assert branch.created_at.day == 15


# ─────────────────────────────────────────────────────────────────────────────
# Branch Name Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBranchNameValidation:
    """Tests for validate_branch_name function."""

    def test_valid_project_name(self):
        """project/auth-service should be valid."""
        valid, msg = validate_branch_name("project/auth-service")
        assert valid is True
        assert msg is None

    def test_valid_feature_name(self):
        """feature/jwt-refresh should be valid."""
        valid, msg = validate_branch_name("feature/jwt-refresh")
        assert valid is True
        assert msg is None

    def test_valid_domain_name(self):
        """domain/security-patterns should be valid."""
        valid, msg = validate_branch_name("domain/security-patterns")
        assert valid is True
        assert msg is None

    def test_valid_spike_name(self):
        """spike/rust-rewrite should be valid."""
        valid, msg = validate_branch_name("spike/rust-rewrite")
        assert valid is True
        assert msg is None

    def test_valid_archive_name(self):
        """archive/old-project-2025-01 should be valid."""
        valid, msg = validate_branch_name("archive/old-project-2025-01")
        assert valid is True
        assert msg is None

    def test_reserved_name_main(self):
        """'main' should be rejected."""
        valid, msg = validate_branch_name("main")
        assert valid is False
        assert "reserved" in msg.lower()

    def test_reserved_name_head(self):
        """'HEAD' should be rejected."""
        valid, msg = validate_branch_name("HEAD")
        assert valid is False
        assert "reserved" in msg.lower()

    def test_reserved_name_empty(self):
        """Empty string should be rejected."""
        valid, msg = validate_branch_name("")
        assert valid is False
        assert "reserved" in msg.lower()

    def test_invalid_uppercase(self):
        """Uppercase letters should be rejected."""
        valid, msg = validate_branch_name("project/AuthService")
        assert valid is False
        assert "lowercase" in msg.lower()

    def test_invalid_special_characters(self):
        """Special characters should be rejected."""
        valid, msg = validate_branch_name("project/auth_service")
        assert valid is False
        assert "lowercase" in msg.lower() or "invalid" in msg.lower()

    def test_invalid_spaces(self):
        """Spaces should be rejected."""
        valid, msg = validate_branch_name("project/auth service")
        assert valid is False

    def test_unknown_type_prefix(self):
        """Unknown type like 'foo/bar' should be rejected."""
        valid, msg = validate_branch_name("foo/bar")
        assert valid is False
        assert "type" in msg.lower()

    def test_no_prefix_non_strict_warns(self):
        """No prefix should warn but allow in non-strict mode."""
        valid, msg = validate_branch_name("myproject")
        assert valid is True
        assert msg is not None
        assert "prefix" in msg.lower()

    def test_no_prefix_strict_rejects(self):
        """No prefix should reject in strict mode."""
        valid, msg = validate_branch_name("myproject", strict=True)
        assert valid is False
        assert "prefix" in msg.lower()

    def test_single_char_name_rejected(self):
        """Single character after slash should be rejected (need 2+ chars)."""
        valid, msg = validate_branch_name("project/a")
        assert valid is False

    def test_name_ending_with_hyphen_rejected(self):
        """Name ending with hyphen should be rejected."""
        valid, msg = validate_branch_name("project/test-")
        assert valid is False

    def test_name_starting_with_hyphen_rejected(self):
        """Name starting with hyphen should be rejected."""
        valid, msg = validate_branch_name("project/-test")
        assert valid is False

    def test_multiple_slashes_rejected(self):
        """Multiple slashes should be rejected."""
        valid, msg = validate_branch_name("project/sub/path")
        assert valid is False

    def test_all_branch_types_valid(self):
        """All defined branch types should be valid prefixes."""
        for branch_type in BRANCH_TYPES:
            valid, msg = validate_branch_name(f"{branch_type}/test-name")
            assert valid is True, f"Failed for type: {branch_type}"
            assert msg is None

    def test_numeric_name_valid(self):
        """Names with numbers should be valid."""
        valid, msg = validate_branch_name("project/v2-auth")
        assert valid is True
        assert msg is None

    def test_all_numeric_name_valid(self):
        """All-numeric names should be valid."""
        valid, msg = validate_branch_name("project/123")
        assert valid is True
        assert msg is None


# ─────────────────────────────────────────────────────────────────────────────
# BranchManager Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_state():
    """Create a sample GraphState for testing."""
    entities = {
        "e1": Entity(id="e1", name="Entity One", type="concept"),
        "e2": Entity(id="e2", name="Entity Two", type="concept"),
        "e3": Entity(id="e3", name="Entity Three", type="concept"),
        "e4": Entity(id="e4", name="Entity Four", type="concept"),
        "e5": Entity(id="e5", name="Entity Five", type="concept"),
    }
    relations = [
        Relation(id="r1", from_entity="e1", to_entity="e2", type="relates_to"),
        Relation(id="r2", from_entity="e2", to_entity="e3", type="relates_to"),
        Relation(id="r3", from_entity="e3", to_entity="e4", type="relates_to"),
        Relation(id="r4", from_entity="e1", to_entity="e5", type="relates_to"),
    ]
    state = GraphState(entities=entities, relations=relations)
    state._rebuild_indices()  # Populate indices for O(1) lookups
    return state


@pytest.fixture
def branch_manager(tmp_path, sample_state):
    """Create a BranchManager with sample state."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return BranchManager(memory_dir, lambda: sample_state)


class TestBranchManagerCRUD:
    """Tests for basic BranchManager CRUD operations."""

    def test_exists_main_always_true(self, branch_manager):
        """Main branch should always exist."""
        assert branch_manager.exists("main") is True

    def test_exists_nonexistent_false(self, branch_manager):
        """Nonexistent branch should return False."""
        assert branch_manager.exists("project/nonexistent") is False

    def test_get_main_returns_main_branch(self, branch_manager):
        """Getting main should return the main branch."""
        main = branch_manager.get("main")
        assert main.name == "main"
        assert main.parent is None

    def test_get_nonexistent_raises(self, branch_manager):
        """Getting nonexistent branch should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            branch_manager.get("project/nonexistent")

    def test_list_includes_main(self, branch_manager):
        """List should always include main."""
        branches = branch_manager.list()
        names = [b.name for b in branches]
        assert "main" in names

    def test_delete_main_raises(self, branch_manager):
        """Deleting main should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot delete main"):
            branch_manager.delete("main")

    def test_delete_current_branch_raises(self, branch_manager):
        """Deleting current branch should raise ValueError."""
        # Create and checkout a branch
        branch_manager.create("project/test", ["e1"])
        branch_manager.checkout("project/test")

        with pytest.raises(ValueError, match="current branch"):
            branch_manager.delete("project/test")

    def test_delete_removes_branch(self, branch_manager):
        """Deleting a branch should remove it."""
        branch_manager.create("project/test", ["e1"])
        assert branch_manager.exists("project/test") is True

        branch_manager.delete("project/test")
        assert branch_manager.exists("project/test") is False


class TestBranchManagerCreate:
    """Tests for branch creation."""

    def test_create_with_seeds(self, branch_manager):
        """Should create branch with seed entities."""
        branch = branch_manager.create("project/test", ["e1"])
        assert branch.name == "project/test"
        assert "e1" in branch.entity_ids

    def test_create_expands_neighbors(self, branch_manager):
        """Should include N-hop neighbors of seeds."""
        # e1 -> e2 -> e3 -> e4, e1 -> e5
        # With depth=2, starting from e1: should get e1, e2, e5, e3
        branch = branch_manager.create("project/test", ["e1"], depth=2)
        assert "e1" in branch.entity_ids
        assert "e2" in branch.entity_ids
        assert "e5" in branch.entity_ids
        assert "e3" in branch.entity_ids
        # e4 is 3 hops away, should not be included
        assert "e4" not in branch.entity_ids

    def test_create_includes_relations(self, branch_manager):
        """Should include relations between branch entities."""
        branch = branch_manager.create("project/test", ["e1"], depth=1)
        # Should have e1, e2, e5 and relations r1, r4
        assert "r1" in branch.relation_ids  # e1 -> e2
        assert "r4" in branch.relation_ids  # e1 -> e5

    def test_create_by_entity_name(self, branch_manager):
        """Should resolve entity names to IDs."""
        branch = branch_manager.create("project/test", ["Entity One"], depth=0)
        assert "e1" in branch.entity_ids

    def test_create_duplicate_name_raises(self, branch_manager):
        """Should raise for existing branch name."""
        branch_manager.create("project/test", ["e1"])
        with pytest.raises(ValueError, match="already exists"):
            branch_manager.create("project/test", ["e2"])

    def test_create_invalid_name_raises(self, branch_manager):
        """Should raise for invalid branch name."""
        with pytest.raises(ValueError):
            branch_manager.create("Invalid Name!", ["e1"])

    def test_create_no_valid_seeds_raises(self, branch_manager):
        """Should raise if no seeds resolve to entities."""
        with pytest.raises(ValueError, match="No valid seed"):
            branch_manager.create("project/test", ["nonexistent"])

    def test_create_persists_to_disk(self, branch_manager):
        """Created branch should persist to disk."""
        branch_manager.create("project/test", ["e1"])
        path = branch_manager._branch_path("project/test")
        assert path.exists()


class TestBranchManagerCheckout:
    """Tests for checkout operations."""

    def test_current_branch_default_main(self, branch_manager):
        """Default current branch should be main."""
        assert branch_manager.current_branch_name() == "main"

    def test_checkout_switches_current(self, branch_manager):
        """Checkout should update current branch."""
        branch_manager.create("project/test", ["e1"])
        branch_manager.checkout("project/test")
        assert branch_manager.current_branch_name() == "project/test"

    def test_checkout_returns_branch(self, branch_manager):
        """Checkout should return the checked-out branch."""
        branch_manager.create("project/test", ["e1"])
        branch = branch_manager.checkout("project/test")
        assert branch.name == "project/test"

    def test_checkout_nonexistent_raises(self, branch_manager):
        """Checkout nonexistent branch should raise."""
        with pytest.raises(ValueError, match="not found"):
            branch_manager.checkout("project/nonexistent")

    def test_checkout_main(self, branch_manager):
        """Should be able to checkout main."""
        branch_manager.create("project/test", ["e1"])
        branch_manager.checkout("project/test")
        branch_manager.checkout("main")
        assert branch_manager.current_branch_name() == "main"


class TestBranchManagerModification:
    """Tests for branch modification operations."""

    def test_add_entities(self, branch_manager):
        """Should add entity IDs to branch."""
        branch_manager.create("project/test", ["e1"], depth=0)
        branch = branch_manager.add_entities("project/test", ["e3"])
        assert "e3" in branch.entity_ids

    def test_add_entities_with_relations(self, branch_manager):
        """Should add connecting relations."""
        branch_manager.create("project/test", ["e1"], depth=0)
        branch = branch_manager.add_entities("project/test", ["e2"])
        # r1 connects e1 -> e2
        assert "r1" in branch.relation_ids

    def test_remove_entities(self, branch_manager):
        """Should remove entity IDs from branch."""
        branch_manager.create("project/test", ["e1", "e2"], depth=0)
        branch = branch_manager.remove_entities("project/test", ["e2"])
        assert "e2" not in branch.entity_ids

    def test_remove_cascades_relations(self, branch_manager):
        """Should remove orphaned relations."""
        branch_manager.create("project/test", ["e1"], depth=1)
        # Should have e1, e2, e5 and relations r1, r4
        branch = branch_manager.remove_entities("project/test", ["e2"])
        # r1 (e1->e2) should be removed since e2 is gone
        assert "r1" not in branch.relation_ids

    def test_modify_main_raises(self, branch_manager):
        """Should raise when modifying main."""
        with pytest.raises(ValueError, match="Cannot modify main"):
            branch_manager.add_entities("main", ["e1"])


class TestBranchManagerMerge:
    """Tests for merge operations."""

    def test_merge_into_main_noop(self, branch_manager):
        """Merging into main should return main unchanged."""
        branch_manager.create("project/test", ["e1"])
        result = branch_manager.merge("project/test", "main")
        assert result.name == "main"

    def test_merge_into_branch(self, branch_manager):
        """Should expand target filter."""
        branch_manager.create("project/source", ["e1"], depth=0)
        branch_manager.create("project/target", ["e3"], depth=0)

        result = branch_manager.merge("project/source", "project/target")
        assert "e1" in result.entity_ids
        assert "e3" in result.entity_ids

    def test_cherry_pick(self, branch_manager):
        """Should copy specific entities."""
        branch_manager.create("project/source", ["e1", "e2"], depth=0)
        branch_manager.create("project/target", ["e3"], depth=0)
        branch_manager.checkout("project/target")

        result = branch_manager.cherry_pick("project/source", ["e1"])
        assert "e1" in result.entity_ids
        assert "e2" not in result.entity_ids


class TestBranchManagerArchive:
    """Tests for archive operations."""

    def test_archive_renames_branch(self, branch_manager):
        """Should move to archive/ prefix."""
        branch_manager.create("project/test", ["e1"])
        archived = branch_manager.archive("project/test")
        assert archived.name.startswith("archive/")

    def test_archive_adds_timestamp(self, branch_manager):
        """Should add date to name."""
        branch_manager.create("project/test", ["e1"])
        archived = branch_manager.archive("project/test")
        # Name should contain year-month
        import re
        assert re.search(r"\d{4}-\d{2}", archived.name)

    def test_archive_marks_inactive(self, branch_manager):
        """Should set is_active=False."""
        branch_manager.create("project/test", ["e1"])
        archived = branch_manager.archive("project/test")
        assert archived.is_active is False

    def test_archive_main_raises(self, branch_manager):
        """Should raise for main."""
        with pytest.raises(ValueError, match="Cannot archive main"):
            branch_manager.archive("main")

    def test_archive_already_archived_raises(self, branch_manager):
        """Should raise for already archived branch."""
        branch_manager.create("project/test", ["e1"])
        archived = branch_manager.archive("project/test")
        with pytest.raises(ValueError, match="already archived"):
            branch_manager.archive(archived.name)

    def test_unarchive_restores(self, branch_manager):
        """Should restore to active branch."""
        branch_manager.create("project/test", ["e1"])
        archived = branch_manager.archive("project/test")
        restored = branch_manager.unarchive(archived.name)
        assert restored.is_active is True
        assert not restored.name.startswith("archive/")

    def test_checkout_archived_raises(self, branch_manager):
        """Should raise when checking out archived branch."""
        branch_manager.create("project/test", ["e1"])
        archived = branch_manager.archive("project/test")
        with pytest.raises(ValueError, match="archived"):
            branch_manager.checkout(archived.name)


class TestBranchManagerDiff:
    """Tests for diff operations."""

    def test_diff_shows_only_in_a(self, branch_manager):
        """Should list entities unique to first branch."""
        branch_manager.create("project/branch-a", ["e1"], depth=0)
        branch_manager.create("project/branch-b", ["e2"], depth=0)

        diff = branch_manager.diff("project/branch-a", "project/branch-b")
        assert "e1" in diff["only_in_a"]["entities"]
        assert "e2" not in diff["only_in_a"]["entities"]

    def test_diff_shows_only_in_b(self, branch_manager):
        """Should list entities unique to second branch."""
        branch_manager.create("project/branch-a", ["e1"], depth=0)
        branch_manager.create("project/branch-b", ["e2"], depth=0)

        diff = branch_manager.diff("project/branch-a", "project/branch-b")
        assert "e2" in diff["only_in_b"]["entities"]
        assert "e1" not in diff["only_in_b"]["entities"]

    def test_diff_shows_common(self, branch_manager):
        """Should list shared entities."""
        branch_manager.create("project/branch-a", ["e1", "e2"], depth=0)
        branch_manager.create("project/branch-b", ["e2", "e3"], depth=0)

        diff = branch_manager.diff("project/branch-a", "project/branch-b")
        assert "e2" in diff["in_both"]["entities"]

    def test_diff_with_main(self, branch_manager):
        """Should handle main's special empty-means-all."""
        branch_manager.create("project/test", ["e1"], depth=0)

        diff = branch_manager.diff("project/test", "main")
        # Main has all entities, so only_in_b should have e2, e3, e4, e5
        assert "e2" in diff["only_in_b"]["entities"]
        assert "e1" in diff["in_both"]["entities"]


class TestBranchManagerFilteredState:
    """Tests for filtered state retrieval."""

    def test_filtered_state_respects_branch(self, branch_manager):
        """get_filtered_state should only return branch entities."""
        branch_manager.create("project/test", ["e1"], depth=0)

        state = branch_manager.get_filtered_state("project/test")
        assert "e1" in state.entities
        assert "e2" not in state.entities

    def test_main_sees_everything(self, branch_manager):
        """Main branch should see all entities."""
        state = branch_manager.get_filtered_state("main")
        assert len(state.entities) == 5  # All entities

    def test_filtered_state_uses_current_branch(self, branch_manager):
        """Should use current branch when none specified."""
        branch_manager.create("project/test", ["e1"], depth=0)
        branch_manager.checkout("project/test")

        state = branch_manager.get_filtered_state()
        assert "e1" in state.entities
        assert "e2" not in state.entities


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Include Integration Tests (using MemoryEngine)
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineAutoInclude:
    """Tests for auto-include behavior in MemoryEngine."""

    @pytest.fixture
    def engine_with_branch(self):
        """Create engine with a non-main branch checked out."""
        from mnemograph.engine import MemoryEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            # Create some seed entities on main
            engine.create_entities([
                {"name": "Python", "entityType": "concept"},
                {"name": "FastAPI", "entityType": "concept"},
            ])

            # Create a branch with one seed entity
            engine.branch_manager.create(
                "project/test",
                seed_entities=["Python"],
                depth=0,
            )
            engine.branch_manager.checkout("project/test")

            yield engine

    def test_create_entity_auto_includes_on_branch(self, engine_with_branch):
        """Creating entity on branch should auto-include it."""
        engine = engine_with_branch

        # Create new entity while on branch
        result = engine.create_entities([
            {"name": "NewConcept", "entityType": "concept"}
        ])
        assert len(result) == 1
        new_entity = result[0]

        # Check that entity is in branch
        branch = engine.branch_manager.get("project/test")
        assert new_entity.id in branch.entity_ids

    def test_create_relation_auto_includes_on_branch(self, engine_with_branch):
        """Creating relation on branch should auto-include it."""
        engine = engine_with_branch

        # Create relation (Python is in branch, FastAPI is not but relation still created)
        result = engine.create_relations([
            {"from": "Python", "to": "FastAPI", "relationType": "related_to"}
        ])
        assert len(result["created"]) == 1
        rel_id = result["created"][0]["id"]

        # Check that relation is in branch
        branch = engine.branch_manager.get("project/test")
        assert rel_id in branch.relation_ids

    def test_remember_auto_includes_entity_and_relations(self, engine_with_branch):
        """remember() should auto-include entity and relations on branch."""
        engine = engine_with_branch

        # Use remember to create entity with relation
        result = engine.remember(
            name="Django",
            entity_type="concept",
            observations=["A Python web framework"],
            relations=[{"to": "Python", "type": "uses"}],
            force=True,  # Bypass duplicate check
        )

        assert result["status"] == "created"
        entity_id = result["entity"]["id"]

        # Check that entity is in branch
        branch = engine.branch_manager.get("project/test")
        assert entity_id in branch.entity_ids

        # Check that relation is in branch (there should be one)
        assert result["relations_created"] == 1
        # The relation ID was auto-included
        assert len(branch.relation_ids) > 0

    def test_no_auto_include_on_main(self):
        """Creating on main should not modify branch files."""
        from mnemograph.engine import MemoryEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            # Create entity on main (default)
            engine.create_entities([
                {"name": "TestEntity", "entityType": "concept"}
            ])

            # No branch files should exist (except _current)
            branches_dir = memory_dir / "branches"
            branch_files = list(branches_dir.glob("**/*.json"))
            assert len(branch_files) == 0

    def test_retrieval_respects_branch_filtering(self, engine_with_branch):
        """Retrieval methods should respect branch filtering."""
        engine = engine_with_branch

        # Python is in branch, FastAPI is not
        graph = engine.read_graph()

        entity_names = [e["name"] for e in graph["entities"]]
        assert "Python" in entity_names
        assert "FastAPI" not in entity_names

    def test_search_respects_branch_filtering(self, engine_with_branch):
        """search_graph should only search within branch."""
        engine = engine_with_branch

        # Search for "Fast" - should not find FastAPI (not in branch)
        result = engine.search_graph("Fast")
        entity_names = [e["name"] for e in result["entities"]]
        assert "FastAPI" not in entity_names

        # Search for "Python" - should find it (in branch)
        result = engine.search_graph("Python")
        entity_names = [e["name"] for e in result["entities"]]
        assert "Python" in entity_names

    def test_open_nodes_respects_branch_filtering(self, engine_with_branch):
        """open_nodes should only return branch-visible entities."""
        engine = engine_with_branch

        # Try to open FastAPI (not in branch)
        result = engine.open_nodes(["FastAPI"])
        assert len(result["entities"]) == 0  # Not visible

        # Open Python (in branch)
        result = engine.open_nodes(["Python"])
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Python"
