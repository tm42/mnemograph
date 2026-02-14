"""Tests for D5: Constants consolidation into constants.py.

Verifies that all constants are accessible from the constants module
and that no duplicates remain in engine.py or similarity.py.
"""

import ast
import importlib
from pathlib import Path

import mnemograph.constants as constants


class TestConstantsAccessible:
    """Verify every constant is importable from the constants module."""

    def test_default_similarity_threshold(self):
        assert constants.DEFAULT_SIMILARITY_THRESHOLD == 0.7

    def test_duplicate_auto_block_threshold(self):
        assert constants.DUPLICATE_AUTO_BLOCK_THRESHOLD == 0.8

    def test_duplicate_detection_threshold(self):
        assert constants.DUPLICATE_DETECTION_THRESHOLD == 0.8

    def test_time_constants(self):
        assert constants.SECONDS_PER_DAY == 86400
        assert constants.SECONDS_PER_WEEK == 604800

    def test_weight_coefficients_sum_to_one(self):
        total = (
            constants.WEIGHT_RECENCY_COEFF
            + constants.WEIGHT_CO_ACCESS_COEFF
            + constants.WEIGHT_EXPLICIT_COEFF
        )
        assert abs(total - 1.0) < 1e-9

    def test_retrieval_limits(self):
        assert constants.DEFAULT_RECENT_LIMIT == 5
        assert constants.DEFAULT_SHALLOW_ENTITIES == 5
        assert constants.DEFAULT_MEDIUM_MAX_NODES == 30
        assert constants.DEFAULT_DEEP_SEEDS == 10

    def test_token_budgets(self):
        assert constants.SHALLOW_CONTEXT_TOKENS == 500
        assert constants.MEDIUM_CONTEXT_TOKENS == 2000
        assert constants.DEEP_CONTEXT_TOKENS == 5000

    def test_query_limits(self):
        assert constants.DEFAULT_QUERY_LIMIT == 10
        assert constants.DEFAULT_SUGGESTION_LIMIT == 5
        assert constants.DEFAULT_VECTOR_SEARCH_LIMIT == 20

    def test_health_limits(self):
        assert constants.OVERLOADED_ENTITY_OBSERVATION_LIMIT == 15
        assert constants.HEALTH_REPORT_ITEM_LIMIT == 10
        assert constants.MAX_OBSERVATION_TOKENS == 2000

    def test_affix_match_bonus(self):
        assert constants.AFFIX_MATCH_BONUS == 0.2

    def test_suggestion_confidence_scores(self):
        assert constants.CO_OCCURRENCE_CONFIDENCE == 0.7
        assert constants.SHARED_RELATION_CONFIDENCE == 0.65


class TestNoDuplicateDefinitions:
    """Verify constants are NOT defined in both engine.py and constants.py."""

    def _get_module_level_assignments(self, filepath: Path) -> set[str]:
        """Parse a Python file and return top-level assignment names."""
        source = filepath.read_text()
        tree = ast.parse(source)
        names = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        names.add(target.id)
        return names

    def test_engine_has_no_constant_definitions(self):
        """engine.py should import constants, not define them."""
        engine_path = Path(__file__).parent.parent / "src" / "mnemograph" / "engine.py"
        assignments = self._get_module_level_assignments(engine_path)

        # These constants should NOT be defined in engine.py anymore
        migrated = {
            "DEFAULT_QUERY_LIMIT",
            "DEFAULT_SIMILARITY_THRESHOLD",
            "DUPLICATE_AUTO_BLOCK_THRESHOLD",
            "SHALLOW_CONTEXT_TOKENS",
            "AFFIX_MATCH_BONUS",
        }
        overlap = assignments & migrated
        assert overlap == set(), f"Constants still defined in engine.py: {overlap}"

    def test_similarity_has_no_constant_definitions(self):
        """similarity.py should import constants, not define them."""
        sim_path = Path(__file__).parent.parent / "src" / "mnemograph" / "similarity.py"
        assignments = self._get_module_level_assignments(sim_path)

        migrated = {"DEFAULT_SIMILARITY_THRESHOLD", "AFFIX_MATCH_BONUS"}
        overlap = assignments & migrated
        assert overlap == set(), f"Constants still defined in similarity.py: {overlap}"

    def test_engine_imports_from_constants(self):
        """engine.py should import constants from the constants module."""
        engine_path = Path(__file__).parent.parent / "src" / "mnemograph" / "engine.py"
        source = engine_path.read_text()
        assert "from .constants import" in source

    def test_similarity_imports_from_constants(self):
        """similarity.py should import constants from the constants module."""
        sim_path = Path(__file__).parent.parent / "src" / "mnemograph" / "similarity.py"
        source = sim_path.read_text()
        assert "from .constants import" in source
