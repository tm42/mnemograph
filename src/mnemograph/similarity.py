"""Similarity detection for entity deduplication.

Provides text-based and embedding-based similarity scoring
to detect potential duplicate entities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .state import GraphState
    from .vectors import VectorIndex

logger = logging.getLogger(__name__)

# Similarity tuning constants
DEFAULT_SIMILARITY_THRESHOLD = 0.7
AFFIX_MATCH_BONUS = 0.2  # Applied as bonus * 0.1 = max 0.02 boost


class SimilarityChecker:
    """Checks for similar entities to detect potential duplicates.

    Uses combination of:
    - Substring containment (React in ReactJS)
    - Jaccard similarity on tokens
    - Prefix/suffix matching bonus
    - Semantic similarity from embeddings (optional)
    """

    def __init__(self, vector_index_getter: Callable[[], "VectorIndex"] | None = None):
        """Initialize checker.

        Args:
            vector_index_getter: Callable that returns VectorIndex (lazy loading)
        """
        self._get_vector_index = vector_index_getter

    def find_similar(
        self,
        name: str,
        state: GraphState,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """Find entities with similar names (potential duplicates).

        Args:
            name: Entity name to check
            state: Current graph state
            threshold: Similarity threshold 0-1 (default 0.7)

        Returns:
            List of similar entities with similarity scores, sorted by score
        """
        results = []
        name_lower = name.lower().strip()
        name_tokens = set(name_lower.split())

        for eid, entity in state.entities.items():
            entity_lower = entity.name.lower()

            # Skip exact matches
            if entity_lower == name_lower:
                continue

            similarity = self.combined_similarity(
                name, entity.name, name_lower, name_tokens
            )

            if similarity >= threshold:
                results.append({
                    "id": eid,
                    "name": entity.name,
                    "type": entity.type,
                    "similarity": round(similarity, 2),
                    "observation_count": len(entity.observations),
                })

        return sorted(results, key=lambda x: x["similarity"], reverse=True)

    def combined_similarity(
        self,
        name1: str,
        name2: str,
        name1_lower: str | None = None,
        name1_tokens: set[str] | None = None,
    ) -> float:
        """Calculate combined similarity between two names.

        Combines multiple signals:
        - Substring containment (strong for "React" in "ReactJS")
        - Jaccard similarity on tokens (good for multi-word names)
        - Prefix/suffix matching bonus
        - Embedding similarity (if vector_index available)

        Args:
            name1: First name
            name2: Second name to compare
            name1_lower: Pre-computed lowercase (optimization)
            name1_tokens: Pre-computed tokens (optimization)

        Returns:
            Similarity score 0.0-1.0
        """
        # Pre-compute if not provided
        if name1_lower is None:
            name1_lower = name1.lower().strip()
        if name1_tokens is None:
            name1_tokens = set(name1_lower.split())

        name2_lower = name2.lower()
        name2_tokens = set(name2_lower.split())

        # 1. Substring containment
        substring_score = self._substring_similarity(name1_lower, name2_lower)

        # 2. Jaccard similarity on tokens
        jaccard = self._jaccard_similarity(name1_tokens, name2_tokens)

        # 3. Prefix/suffix matching bonus
        affix_bonus = self._affix_bonus(name1_lower, name2_lower)

        # 4. Embedding similarity (if available)
        embedding_sim = 0.0
        if self._get_vector_index is not None:
            embedding_sim = self._embedding_similarity(name1, name2)

        # Combined score - max of approaches + capped bonus
        base_similarity = max(substring_score, jaccard, embedding_sim)
        return min(base_similarity + affix_bonus * 0.1, 1.0)

    def _substring_similarity(self, s1: str, s2: str) -> float:
        """Check for substring containment.

        Strong signal for "React" in "ReactJS" type matches.
        """
        if s1 in s2 or s2 in s1:
            shorter = min(len(s1), len(s2))
            longer = max(len(s1), len(s2))
            return shorter / longer if longer > 0 else 0.0
        return 0.0

    def _jaccard_similarity(self, tokens1: set[str], tokens2: set[str]) -> float:
        """Jaccard similarity on token sets.

        Good for multi-word names like "React Native" vs "React".
        """
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0

    def _affix_bonus(self, s1: str, s2: str) -> float:
        """Bonus for prefix/suffix matches."""
        prefix_match = s1.startswith(s2) or s2.startswith(s1)
        suffix_match = s1.endswith(s2) or s2.endswith(s1)
        return AFFIX_MATCH_BONUS if (prefix_match or suffix_match) else 0.0

    def _embedding_similarity(self, name1: str, name2: str) -> float:
        """Get embedding-based similarity score between two names.

        Uses vector index to compute cosine similarity between embeddings.
        Only triggers model loading when this method is actually called.

        Args:
            name1: First name to compare
            name2: Second name to compare

        Returns:
            Cosine similarity score 0.0-1.0
        """
        if self._get_vector_index is None:
            return 0.0

        try:
            vector_index = self._get_vector_index()
            return vector_index.text_similarity(name1, name2)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Embedding similarity lookup failed: {e}")

        return 0.0
