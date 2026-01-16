"""Tiered context retrieval for memory queries.

Three depth levels:
- shallow: summary stats, recent entities, hot topics (~500 tokens)
- medium: vector search results + weighted 1-hop neighbors (~2000 tokens)
- deep: weighted multi-hop subgraph from focus entities (~5000 tokens)

Uses edge weights to prioritize stronger connections during traversal.
"""

from dataclasses import dataclass

from .models import Entity
from .state import GraphState
from .weights import weighted_bfs, update_co_access_scores


@dataclass
class ContextResult:
    """Result of a context retrieval operation."""
    depth: str  # 'shallow' | 'medium' | 'deep'
    tokens_estimate: int
    content: str
    entity_count: int
    relation_count: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


def get_shallow_context(state: GraphState, max_tokens: int = 500) -> ContextResult:
    """Summary stats, recent entities, hot topics.

    Cheap and fast — good for "what do I know about?" queries.
    """
    entity_count = len(state.entities)
    relation_count = len(state.relations)

    if entity_count == 0:
        content = "## Memory Summary\nNo entities stored yet."
        return ContextResult(
            depth="shallow",
            tokens_estimate=estimate_tokens(content),
            content=content,
            entity_count=0,
            relation_count=0,
        )

    # Recent entities (by updated_at)
    recent = sorted(
        state.entities.values(),
        key=lambda e: e.updated_at,
        reverse=True,
    )[:5]

    # Most accessed
    hot = sorted(
        state.entities.values(),
        key=lambda e: e.access_count,
        reverse=True,
    )[:5]

    # Entity type breakdown
    type_counts: dict[str, int] = {}
    for e in state.entities.values():
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    lines = [
        "## Memory Summary",
        f"- {entity_count} entities, {relation_count} relations",
        "",
        "### Entity Types",
    ]
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {etype}: {count}")

    lines.extend([
        "",
        "### Recently Updated",
    ])
    for e in recent:
        obs_preview = e.observations[0].text[:50] + "..." if e.observations else ""
        lines.append(f"- **{e.name}** ({e.type}): {obs_preview}")

    if any(e.access_count > 0 for e in hot):
        lines.extend([
            "",
            "### Frequently Accessed",
        ])
        for e in hot:
            if e.access_count > 0:
                lines.append(f"- **{e.name}**: {e.access_count} accesses")

    content = "\n".join(lines)

    # Truncate if over budget
    while estimate_tokens(content) > max_tokens and len(lines) > 5:
        lines.pop()
        content = "\n".join(lines)

    return ContextResult(
        depth="shallow",
        tokens_estimate=estimate_tokens(content),
        content=content,
        entity_count=min(10, entity_count),
        relation_count=0,
    )


def get_medium_context(
    state: GraphState,
    vector_search_results: list[tuple[str, float]] | None = None,
    focus: list[str] | None = None,
    max_tokens: int = 2000,
    min_weight: float = 0.1,
) -> ContextResult:
    """Vector search results + weighted 1-hop neighbors.

    Uses edge weights to prioritize stronger connections.
    Good for targeted queries about specific topics.
    """
    # Collect seed entity IDs
    seed_ids: list[str] = []

    # Start with focused entities
    if focus:
        for name in focus:
            entity = _find_entity_by_name(state, name)
            if entity:
                seed_ids.append(entity.id)

    # Add vector search results
    if vector_search_results:
        for entity_id, _score in vector_search_results:
            if entity_id in state.entities and entity_id not in seed_ids:
                seed_ids.append(entity_id)

    # Use weighted BFS with depth=1 for 1-hop neighbors
    entity_ids = weighted_bfs(
        start_entity_ids=seed_ids,
        state=state,
        max_depth=1,
        max_nodes=30,  # Limit for medium context
        min_weight=min_weight,
    )

    # Build entity list
    seen_ids: set[str] = set(entity_ids)
    entities: list[Entity] = [
        state.entities[eid] for eid in entity_ids
        if eid in state.entities
    ]

    # Update co-access scores for learning
    update_co_access_scores(state, seen_ids)

    # Format output
    content = _format_entities_with_relations(state, entities, seen_ids, max_tokens)

    return ContextResult(
        depth="medium",
        tokens_estimate=estimate_tokens(content),
        content=content,
        entity_count=len(entities),
        relation_count=len([r for r in state.relations
                          if r.from_entity in seen_ids or r.to_entity in seen_ids]),
    )


def get_deep_context(
    state: GraphState,
    focus: list[str] | None = None,
    max_tokens: int = 5000,
    max_hops: int = 3,
    max_nodes: int = 50,
    min_weight: float = 0.1,
) -> ContextResult:
    """Weighted multi-hop subgraph extraction.

    Uses weighted_bfs to prioritize stronger connections.
    Comprehensive context for complex queries.
    """
    # Resolve focus names to entity IDs
    seed_ids: list[str] = []
    if focus:
        for name in focus:
            entity = _find_entity_by_name(state, name)
            if entity:
                seed_ids.append(entity.id)

    if not seed_ids:
        # No focus = use recent entities as seeds
        recent = sorted(
            state.entities.values(),
            key=lambda e: e.updated_at,
            reverse=True,
        )[:10]
        seed_ids = [e.id for e in recent]

    # Use weighted BFS for traversal
    entity_ids = weighted_bfs(
        start_entity_ids=seed_ids,
        state=state,
        max_depth=max_hops,
        max_nodes=max_nodes,
        min_weight=min_weight,
    )

    # Build entity list and seen set
    seen_ids: set[str] = set(entity_ids)
    entities: list[Entity] = [
        state.entities[eid] for eid in entity_ids
        if eid in state.entities
    ]

    # Update co-access scores for learning
    update_co_access_scores(state, seen_ids)

    # Format with full details
    content = _format_entities_with_relations(state, entities, seen_ids, max_tokens, verbose=True)

    return ContextResult(
        depth="deep",
        tokens_estimate=estimate_tokens(content),
        content=content,
        entity_count=len(entities),
        relation_count=len([r for r in state.relations
                          if r.from_entity in seen_ids and r.to_entity in seen_ids]),
    )


def _find_entity_by_name(state: GraphState, name: str) -> Entity | None:
    """Find entity by name using O(1) index lookup."""
    entity_id = state.get_entity_id_by_name(name)
    if entity_id:
        return state.entities.get(entity_id)
    return None


def _format_entities_with_relations(
    state: GraphState,
    entities: list[Entity],
    entity_ids: set[str],
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """Format entities with their relations."""
    if not entities:
        return "No matching entities found."

    lines: list[str] = []

    for entity in entities:
        lines.append(f"### {entity.name} ({entity.type})")

        # Observations
        obs_limit = 5 if verbose else 3
        for obs in entity.observations[:obs_limit]:
            text = obs.text if verbose else obs.text[:100]
            lines.append(f"  - {text}")
        if len(entity.observations) > obs_limit:
            lines.append(f"  - ... and {len(entity.observations) - obs_limit} more observations")

        # Relations (only within the entity set)
        for r in state.relations:
            if r.from_entity == entity.id and r.to_entity in entity_ids:
                target = state.entities.get(r.to_entity)
                if target:
                    lines.append(f"  → {r.type} → **{target.name}**")
            elif r.to_entity == entity.id and r.from_entity in entity_ids:
                source = state.entities.get(r.from_entity)
                if source:
                    lines.append(f"  ← {r.type} ← **{source.name}**")

        lines.append("")

    content = "\n".join(lines)

    # Truncate if over budget
    while estimate_tokens(content) > max_tokens and len(lines) > 10:
        # Remove from the end, keeping structure
        lines = lines[:-5]
        content = "\n".join(lines) + "\n\n... (truncated)"

    return content
