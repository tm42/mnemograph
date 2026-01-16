"""Edge weight computation and weighted traversal.

Provides:
- compute_recency_score(): Time-based decay scoring
- update_co_access_scores(): Learning from retrieval patterns
- weighted_bfs(): Priority-queue traversal favoring strong connections
"""

import heapq
import math
from datetime import datetime, timezone

from .models import Relation
from .state import GraphState


def compute_recency_score(
    last_accessed: datetime,
    half_life_days: float = 30.0,
    reference_time: datetime | None = None,
) -> float:
    """Compute recency score with exponential decay.

    Args:
        last_accessed: When the relation was last accessed
        half_life_days: Time for score to decay to 0.5 (default: 30 days)
        reference_time: Calculate relative to this time (default: now)

    Returns:
        Score between 0.0 and 1.0

    Examples:
        >>> compute_recency_score(now)  # Just accessed
        1.0
        >>> compute_recency_score(now - timedelta(days=30))  # 30 days ago
        0.5
        >>> compute_recency_score(now - timedelta(days=60))  # 60 days ago
        0.25
    """
    from .constants import SECONDS_PER_DAY, LN_2

    reference = reference_time or datetime.now(timezone.utc)
    days_since = (reference - last_accessed).total_seconds() / SECONDS_PER_DAY

    if days_since < 0:
        return 1.0  # Future date, treat as fresh

    # Exponential decay: score = e^(-λt) where λ = ln(2) / half_life
    decay_rate = LN_2 / half_life_days
    score = math.exp(-decay_rate * days_since)

    return max(0.0, min(1.0, score))


def update_co_access_scores(
    state: GraphState,
    retrieved_entity_ids: set[str],
    increment: float = 0.1,
) -> list[str]:
    """Strengthen relations where both entities were retrieved together.

    Call this after each retrieval operation to learn from usage patterns.
    Uses diminishing returns (asymptotic to 1.0).

    Args:
        state: Current graph state (modified in place)
        retrieved_entity_ids: Set of entity IDs returned in this retrieval
        increment: Amount to increase co_access_score (default: 0.1)

    Returns:
        List of relation IDs that were updated
    """
    updated = []
    now = datetime.now(timezone.utc)

    for rel in state.relations:
        if (rel.from_entity in retrieved_entity_ids and
            rel.to_entity in retrieved_entity_ids):
            # Increment with diminishing returns (asymptotic to 1.0)
            rel.co_access_score = min(1.0, rel.co_access_score + increment)
            rel.access_count += 1
            rel.last_accessed = now
            updated.append(rel.id)

    return updated


def weighted_bfs(
    start_entity_ids: list[str],
    state: GraphState,
    max_depth: int = 3,
    max_nodes: int = 50,
    min_weight: float = 0.1,
) -> list[str]:
    """Breadth-first traversal prioritizing high-weight edges.

    Uses a priority queue to visit stronger connections first, even if
    they're farther away in terms of hops.

    Args:
        start_entity_ids: Seed entities to start from
        state: Current graph state
        max_depth: Maximum hops from seeds (default: 3)
        max_nodes: Maximum entities to return (default: 50)
        min_weight: Ignore edges below this weight (default: 0.1)

    Returns:
        Entity IDs in priority order (highest relevance first)
    """
    visited: set[str] = set()
    result: list[str] = []

    # Priority queue: (priority, depth, entity_id)
    # Lower priority value = visited sooner (heapq is min-heap)
    # Start seeds with priority 0
    heap: list[tuple[float, int, str]] = [
        (0.0, 0, eid)
        for eid in start_entity_ids
        if eid in state.entities
    ]
    heapq.heapify(heap)

    while heap and len(result) < max_nodes:
        priority, depth, entity_id = heapq.heappop(heap)

        if entity_id in visited:
            continue

        visited.add(entity_id)
        result.append(entity_id)

        if depth >= max_depth:
            continue

        # Add neighbors, prioritized by edge weight
        for relation in state.get_relations_for(entity_id):
            neighbor = relation.other_entity(entity_id)

            if neighbor in visited:
                continue
            if neighbor not in state.entities:
                continue
            if relation.weight < min_weight:
                continue

            # Priority increases (gets worse) as we go deeper
            # and decreases (gets better) with higher edge weights
            # Formula: new_priority = parent_priority + (1 - edge_weight)
            new_priority = priority + (1.0 - relation.weight)
            heapq.heappush(heap, (new_priority, depth + 1, neighbor))

    return result


def get_strongest_connections(
    state: GraphState,
    entity_id: str,
    limit: int = 10,
) -> list[tuple[Relation, str]]:
    """Get an entity's strongest connections by weight.

    Args:
        state: Current graph state
        entity_id: Entity to get connections for
        limit: Maximum number of connections to return

    Returns:
        List of (relation, connected_entity_id) tuples sorted by weight desc
    """
    relations = state.get_relations_for(entity_id)
    connections = [
        (rel, rel.other_entity(entity_id))
        for rel in relations
    ]
    # Sort by weight descending
    connections.sort(key=lambda x: x[0].weight, reverse=True)
    return connections[:limit]
