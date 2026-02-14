"""Read-only query operations on the knowledge graph.

Extracted from engine.py to separate read-only queries from mutations/lifecycle.
MemoryEngine delegates all query methods to QueryService via thin wrappers.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Callable

from .constants import (
    CO_OCCURRENCE_CONFIDENCE,
    DEFAULT_QUERY_LIMIT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SUGGESTION_LIMIT,
    DEFAULT_VECTOR_SEARCH_LIMIT,
    DEEP_CONTEXT_TOKENS,
    DUPLICATE_DETECTION_THRESHOLD,
    HEALTH_REPORT_ITEM_LIMIT,
    MEDIUM_CONTEXT_TOKENS,
    OVERLOADED_ENTITY_OBSERVATION_LIMIT,
    PRUNING_CANDIDATE_THRESHOLD,
    SHALLOW_CONTEXT_TOKENS,
    SHARED_RELATION_CONFIDENCE,
    SUGGEST_RELATION_CONFIDENCE_THRESHOLD,
    WEAK_RELATION_THRESHOLD,
)
from .models import Entity, MemoryEvent, Relation
from .similarity import SimilarityChecker
from .state import GraphState
from .time_travel import TimeTraveler

logger = logging.getLogger(__name__)


class QueryService:
    """Read-only query operations on the knowledge graph.

    Uses callable accessors to always read current state (not stale copies).
    Does not import from engine.py to avoid circular imports.
    """

    def __init__(
        self,
        get_state: Callable[[], GraphState],
        get_active_state: Callable[[], GraphState],
        get_vector_index: Callable,
        similarity: Callable[[], SimilarityChecker],
        time_traveler: Callable[[], TimeTraveler],
        resolve_entity: Callable[[str], str | None],
        on_co_access: Callable[[], None] | None = None,
    ):
        self._get_state = get_state
        self._get_active_state = get_active_state
        self._get_vector_index = get_vector_index
        self._get_similarity = similarity
        self._get_time_traveler = time_traveler
        self._resolve_entity = resolve_entity
        self._on_co_access = on_co_access

    # --- Query operations ---

    def read_graph(self) -> dict:
        """Return full graph state for current branch.

        On main branch, returns everything. On other branches, returns
        only entities and relations belonging to that branch.
        """
        active_state = self._get_active_state()
        return {
            "entities": [e.model_dump(mode="json") for e in active_state.entities.values()],
            "relations": [r.model_dump(mode="json") for r in active_state.relations],
        }

    def get_structure(
        self,
        entity_ids: list[str] | None = None,
        include_neighbors: bool = True,
    ) -> dict:
        """Get graph structure (names, types, relations) without full observations.

        This is the lightweight alternative to search_graph/read_graph for initial
        exploration. Returns ~10x fewer tokens than full data.

        Uses branch-filtered state on non-main branches.

        Args:
            entity_ids: Specific entity IDs to include (None = all)
            include_neighbors: If True, include 1-hop neighbors of specified entities

        Returns:
            Dict with 'entities' (summaries) and 'relations' (summaries)
        """
        active_state = self._get_active_state()

        if entity_ids is None:
            # All entities in active state
            target_ids = set(active_state.entities.keys())
        else:
            # Filter to entities visible in active state
            target_ids = {eid for eid in entity_ids if eid in active_state.entities}

            if include_neighbors:
                # Add 1-hop neighbors (within active state)
                for eid in list(target_ids):
                    for rel in active_state.get_outgoing_relations(eid):
                        if rel.to_entity in active_state.entities:
                            target_ids.add(rel.to_entity)
                    for rel in active_state.get_incoming_relations(eid):
                        if rel.from_entity in active_state.entities:
                            target_ids.add(rel.from_entity)

        # Build lightweight entity summaries
        entities = []
        for eid in target_ids:
            if eid in active_state.entities:
                entity = active_state.entities[eid]
                entities.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "observation_count": len(entity.observations),
                })

        # Build relation summaries
        relations = []
        for rel in active_state.relations:
            if rel.from_entity in target_ids or rel.to_entity in target_ids:
                from_entity = active_state.entities.get(rel.from_entity)
                to_entity = active_state.entities.get(rel.to_entity)
                relations.append({
                    "from": from_entity.name if from_entity else rel.from_entity,
                    "to": to_entity.name if to_entity else rel.to_entity,
                    "type": rel.type,
                    "weight": round(rel.weight, 2),
                })

        return {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "entities": entities,
            "relations": relations,
        }

    def search_structure(self, query: str, limit: int = 20) -> dict:
        """Search for entities and return structure-only results.

        Combines text and semantic search, returns lightweight summaries.
        Use open_nodes() to get full data for specific entities.

        Uses branch-filtered state on non-main branches.

        Args:
            query: Search query
            limit: Max entities to return

        Returns:
            Structure with matched entities, their relations, and token estimate
        """
        active_state = self._get_active_state()
        query_lower = query.lower()
        matched_ids: list[str] = []
        scores: dict[str, float] = {}

        # 1. Text search (exact matches score higher) - within active state
        for entity in active_state.entities.values():
            score = 0.0

            # Name match (highest)
            if query_lower in entity.name.lower():
                score = 1.0 if query_lower == entity.name.lower() else 0.9
            # Type match
            elif query_lower in entity.type.lower():
                score = 0.5
            # Observation match
            else:
                for obs in entity.observations:
                    if query_lower in obs.text.lower():
                        score = 0.6
                        break

            if score > 0:
                matched_ids.append(entity.id)
                scores[entity.id] = score

        # 2. Semantic search (adds more candidates) - filter to active state
        # Uses lazy-loaded vector_index which gracefully returns [] if unavailable
        try:
            vector_index = self._get_vector_index()
            vector_results = vector_index.search(query, limit=limit)
            for eid, vscore in vector_results:
                # Only include if entity is in active state (branch-filtered)
                if eid in active_state.entities:
                    if eid not in scores:
                        matched_ids.append(eid)
                        scores[eid] = vscore * 0.8  # Weight semantic slightly lower
                    else:
                        # Boost score if found by both methods
                        scores[eid] = min(1.0, scores[eid] + vscore * 0.2)
        except Exception as e:
            logger.debug(f"Semantic search failed (continuing with keyword results): {e}")

        # Sort by score and limit
        matched_ids.sort(key=lambda x: scores.get(x, 0), reverse=True)
        matched_ids = matched_ids[:limit]

        # Get structure for matched entities
        structure = self.get_structure(matched_ids, include_neighbors=True)

        # Estimate tokens for full data
        total_obs = sum(
            len(active_state.entities[eid].observations)
            for eid in matched_ids
            if eid in active_state.entities
        )
        estimated_full_tokens = total_obs * 50  # ~50 tokens per observation avg

        return {
            "query": query,
            "matched_count": len(matched_ids),
            "structure": structure,
            "estimated_full_tokens": estimated_full_tokens,
            "hint": f"Use open_nodes([...]) to get full data for specific entities. "
                   f"Full data for all matches would be ~{estimated_full_tokens} tokens.",
        }

    def search_graph(self, query: str) -> dict:
        """Search entities by text across names, types, observations.

        Uses branch-filtered state on non-main branches.
        """
        active_state = self._get_active_state()
        query_lower = query.lower()
        matching_entities = []
        matching_entity_ids = set()

        for entity in active_state.entities.values():
            matched = False
            if query_lower in entity.name.lower():
                matched = True
            elif query_lower in entity.type.lower():
                matched = True
            else:
                for obs in entity.observations:
                    if query_lower in obs.text.lower():
                        matched = True
                        break
            if matched:
                matching_entities.append(entity)
                matching_entity_ids.add(entity.id)

        # Include relations between matching entities (within active state)
        matching_relations = [
            r for r in active_state.relations
            if r.from_entity in matching_entity_ids or r.to_entity in matching_entity_ids
        ]

        return {
            "entities": [e.model_dump(mode="json") for e in matching_entities],
            "relations": [r.model_dump(mode="json") for r in matching_relations],
        }

    def open_nodes(self, names: list[str]) -> dict:
        """Get specific entities by name, their relations, and neighbors.

        Uses branch-filtered state on non-main branches. Only returns
        entities/relations visible on the current branch.
        """
        active_state = self._get_active_state()
        entities = []
        entity_ids = set()

        for name in names:
            entity_id = self._resolve_entity(name)
            # Only include if entity is in active state (branch-filtered)
            if entity_id and entity_id in active_state.entities:
                entities.append(active_state.entities[entity_id])
                entity_ids.add(entity_id)

        # Include relations involving these entities (within active state)
        relations = [
            r for r in active_state.relations
            if r.from_entity in entity_ids or r.to_entity in entity_ids
        ]

        # Find neighbor IDs (entities connected via relations, within active state)
        neighbor_ids = set()
        for r in relations:
            if r.from_entity not in entity_ids and r.from_entity in active_state.entities:
                neighbor_ids.add(r.from_entity)
            if r.to_entity not in entity_ids and r.to_entity in active_state.entities:
                neighbor_ids.add(r.to_entity)

        neighbors = [
            active_state.entities[nid] for nid in neighbor_ids
            if nid in active_state.entities
        ]

        return {
            "entities": [e.model_dump(mode="json") for e in entities],
            "relations": [r.model_dump(mode="json") for r in relations],
            "neighbors": [n.model_dump(mode="json") for n in neighbors],
        }

    # --- Richer queries ---

    def get_recent_entities(self, limit: int = DEFAULT_QUERY_LIMIT) -> list[Entity]:
        """Get most recently updated entities.

        Uses branch-filtered state on non-main branches.
        """
        active_state = self._get_active_state()
        return sorted(
            active_state.entities.values(),
            key=lambda e: e.updated_at,
            reverse=True,
        )[:limit]

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Filter entities by type.

        Uses branch-filtered state on non-main branches.
        """
        active_state = self._get_active_state()
        return [e for e in active_state.entities.values() if e.type == entity_type]

    def get_entity_neighbors(self, entity_id: str, include_entity: bool = True) -> dict:
        """Get entity and its directly connected neighbors via relations.

        Uses branch-filtered state on non-main branches.
        """
        active_state = self._get_active_state()
        entity = active_state.entities.get(entity_id)
        if not entity:
            return {"entity": None, "neighbors": [], "outgoing": [], "incoming": []}

        outgoing = [r for r in active_state.relations if r.from_entity == entity_id]
        incoming = [r for r in active_state.relations if r.to_entity == entity_id]

        neighbor_ids = set(
            [r.to_entity for r in outgoing] + [r.from_entity for r in incoming]
        )
        neighbors = [active_state.entities[nid] for nid in neighbor_ids if nid in active_state.entities]

        return {
            "entity": entity.model_dump(mode="json") if include_entity else None,
            "neighbors": [n.model_dump(mode="json") for n in neighbors],
            "outgoing": [r.model_dump(mode="json") for r in outgoing],
            "incoming": [r.model_dump(mode="json") for r in incoming],
        }

    # --- Vector search ---

    def search_semantic(
        self,
        query: str,
        limit: int = DEFAULT_QUERY_LIMIT,
        type_filter: str | None = None
    ) -> dict:
        """Semantic search using embeddings.

        Uses branch-filtered state on non-main branches. Vector search
        returns all candidates, but results are filtered to current branch.
        """
        active_state = self._get_active_state()
        vector_index = self._get_vector_index()
        results = vector_index.search(query, limit, type_filter)

        entities = []
        entity_ids = []
        for entity_id, score in results:
            # Only include if entity is in active state (branch-filtered)
            if entity_id in active_state.entities:
                entity = active_state.entities[entity_id]
                entity_dict = entity.model_dump(mode="json")
                entity_dict["_score"] = score  # Attach similarity score
                entities.append(entity_dict)
                entity_ids.append(entity_id)

        # Include relations between matched entities (within active state)
        entity_id_set = set(entity_ids)
        relations = [
            r.model_dump(mode="json") for r in active_state.relations
            if r.from_entity in entity_id_set or r.to_entity in entity_id_set
        ]

        return {
            "entities": entities,
            "relations": relations,
        }

    # --- Tiered retrieval ---

    def recall(
        self,
        depth: str = "shallow",
        query: str | None = None,
        focus: list[str] | None = None,
        max_tokens: int | None = None,
        format: str = "prose",  # noqa: A002 - shadowing builtin is intentional for API
    ) -> dict:
        """Get memory context at varying depth levels.

        This is the PRIMARY retrieval tool. Returns structure-first for large results,
        with hints to use open_nodes() for full data on specific entities.

        Args:
            depth: 'shallow' (summary), 'medium' (search + neighbors), 'deep' (multi-hop)
            query: Search query for medium/deep depth (uses semantic search)
            focus: Entity names to focus on
            max_tokens: Token budget (defaults vary by depth)
            format: 'prose' (human-readable, default) or 'graph' (JSON structure)

        Returns:
            Dict with depth, tokens_estimate, content, entity_count, relation_count
            For medium/deep: may include 'structure_only' flag if results too large
            For format='prose': content is human-readable prose
            For format='graph': content is JSON-serializable structure
        """
        from .retrieval import get_shallow_context, get_medium_context, get_deep_context
        from .linearize import linearize_to_prose, linearize_shallow_summary

        # Use branch-filtered state for all retrieval
        active_state = self._get_active_state()

        if depth == "shallow":
            tokens = max_tokens or SHALLOW_CONTEXT_TOKENS

            if format == "prose":
                # Use prose linearization for shallow
                content = linearize_shallow_summary(active_state)
                return {
                    "depth": "shallow",
                    "format": "prose",
                    "tokens_estimate": len(content) // 4,
                    "content": content,
                    "entity_count": len(active_state.entities),
                    "relation_count": len(active_state.relations),
                }
            else:
                # Graph format — return the old markdown format
                result = get_shallow_context(active_state, tokens)
                return {
                    "depth": result.depth,
                    "format": "graph",
                    "tokens_estimate": result.tokens_estimate,
                    "content": result.content,
                    "entity_count": result.entity_count,
                    "relation_count": result.relation_count,
                }

        elif depth == "medium":
            tokens = max_tokens or MEDIUM_CONTEXT_TOKENS

            # First, do a structure search to estimate size
            # Skip early return when focus is provided — the caller has already
            # narrowed what they want, so the broad estimate is irrelevant.
            if query and not focus:
                structure_result = self.search_structure(query, limit=20)
                estimated_tokens = structure_result["estimated_full_tokens"]

                # If results would be too large, return structure-only
                if estimated_tokens > tokens:
                    if self._on_co_access:
                        self._on_co_access()
                    return {
                        "depth": "medium",
                        "format": format,
                        "structure_only": True,
                        "reason": f"Full results would be ~{estimated_tokens} tokens (budget: {tokens})",
                        "matched_count": structure_result["matched_count"],
                        "content": self._format_structure(structure_result["structure"]),
                        "tokens_estimate": structure_result["structure"]["entity_count"] * 15,
                        "entity_count": structure_result["structure"]["entity_count"],
                        "relation_count": structure_result["structure"]["relation_count"],
                        "hint": "Use open_nodes(['entity1', 'entity2']) to get full data for specific entities",
                    }

            # Normal path: get vector search results
            vector_results = None
            if query:
                vector_index = self._get_vector_index()
                vector_results = vector_index.search(query, limit=DEFAULT_QUERY_LIMIT)
            result = get_medium_context(active_state, vector_results, focus, tokens)
            if self._on_co_access:
                self._on_co_access()

            # Collect entity IDs from result for prose linearization
            matched_entity_ids = result.entity_ids

        elif depth == "deep":
            tokens = max_tokens or DEEP_CONTEXT_TOKENS

            # For deep, estimate first based on focus entities
            if focus:
                focus_ids = [self._resolve_entity(f) for f in focus]
                focus_ids = [fid for fid in focus_ids if fid]
                structure = self.get_structure(focus_ids, include_neighbors=True)
                total_obs = sum(
                    len(active_state.entities[eid].observations)
                    for eid in focus_ids
                    if eid in active_state.entities
                )
                estimated_tokens = total_obs * 50

                if estimated_tokens > tokens:
                    if self._on_co_access:
                        self._on_co_access()
                    return {
                        "depth": "deep",
                        "format": format,
                        "structure_only": True,
                        "reason": f"Full results would be ~{estimated_tokens} tokens (budget: {tokens})",
                        "content": self._format_structure(structure),
                        "tokens_estimate": structure["entity_count"] * 15,
                        "entity_count": structure["entity_count"],
                        "relation_count": structure["relation_count"],
                        "hint": "Use open_nodes(['entity1', 'entity2']) to get full data for specific entities",
                    }

            result = get_deep_context(active_state, focus, tokens)
            if self._on_co_access:
                self._on_co_access()

            # Collect entity IDs from result for prose linearization
            matched_entity_ids = result.entity_ids

        else:
            raise ValueError(f"Invalid depth: {depth}. Use 'shallow', 'medium', or 'deep'.")

        # Format output based on requested format
        if format == "prose":
            # Get entities and linearize to prose
            entities = [
                active_state.entities[eid]
                for eid in matched_entity_ids
                if eid in active_state.entities
            ]
            content = linearize_to_prose(active_state, entities, matched_entity_ids, depth)
            return {
                "depth": result.depth,
                "format": "prose",
                "tokens_estimate": len(content) // 4,
                "content": content,
                "entity_count": result.entity_count,
                "relation_count": result.relation_count,
            }
        else:
            # Graph format — return the existing markdown
            return {
                "depth": result.depth,
                "format": "graph",
                "tokens_estimate": result.tokens_estimate,
                "content": result.content,
                "entity_count": result.entity_count,
                "relation_count": result.relation_count,
            }

    def _format_structure(self, structure: dict) -> str:
        """Format structure-only results as readable markdown."""
        lines = [
            f"## Graph Structure ({structure['entity_count']} entities, {structure['relation_count']} relations)",
            "",
            "### Entities",
        ]

        # Group entities by type
        by_type: dict[str, list[dict]] = {}
        for e in structure["entities"]:
            etype = e["type"]
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(e)

        for etype, entities in sorted(by_type.items()):
            lines.append(f"\n**{etype}** ({len(entities)})")
            for e in entities[:10]:  # Limit per type
                obs_hint = f" [{e['observation_count']} obs]" if e["observation_count"] > 0 else ""
                lines.append(f"- {e['name']}{obs_hint}")
            if len(entities) > 10:
                lines.append(f"  ... and {len(entities) - 10} more")

        # Relations summary
        if structure["relations"]:
            lines.append("\n### Key Relations")
            # Show strongest relations
            sorted_rels = sorted(structure["relations"], key=lambda r: r["weight"], reverse=True)
            for rel in sorted_rels[:15]:
                lines.append(f"- {rel['from']} --{rel['type']}--> {rel['to']}")
            if len(sorted_rels) > 15:
                lines.append(f"  ... and {len(sorted_rels) - 15} more relations")

        return "\n".join(lines)

    # --- Time travel delegates ---

    def state_at(self, timestamp: str | datetime) -> GraphState:
        """Get graph state at a specific point in time.

        Delegates to TimeTraveler.
        """
        return self._get_time_traveler().state_at(timestamp)

    def events_between(
        self,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> list[MemoryEvent]:
        """Get events in a time range. Delegates to TimeTraveler."""
        return self._get_time_traveler().events_between(start, end)

    def diff_between(
        self,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> dict:
        """Compute what changed between two points in time. Delegates to TimeTraveler."""
        return self._get_time_traveler().diff_between(start, end)

    def get_entity_history(self, entity_name: str) -> list[dict]:
        """Get the history of changes to an entity.

        Returns list of events that affected this entity, oldest first.
        """
        entity_id = self._resolve_entity(entity_name)
        if not entity_id:
            return []
        return self._get_time_traveler().get_entity_history(entity_id)

    # --- Edge Weights (read-only) ---

    def get_relation_weight(self, relation_id: str) -> dict | None:
        """Get weight breakdown for a relation.

        Args:
            relation_id: ID of the relation (full ID or prefix)

        Returns:
            Weight breakdown dict, or None if relation not found
        """
        state = self._get_state()
        # Support prefix matching
        relation = None
        for r in state.relations:
            if r.id == relation_id or r.id.startswith(relation_id):
                relation = r
                break

        if not relation:
            return None

        # Get entity names for context
        from_entity = state.entities.get(relation.from_entity)
        to_entity = state.entities.get(relation.to_entity)

        return {
            "relation_id": relation.id,
            "from_entity": from_entity.name if from_entity else relation.from_entity,
            "to_entity": to_entity.name if to_entity else relation.to_entity,
            "relation_type": relation.type,
            "combined_weight": relation.weight,
            "components": {
                "recency": relation.recency_score,
                "co_access": relation.co_access_score,
                "explicit": relation.explicit_weight,
            },
            "metadata": {
                "created_at": relation.created_at.isoformat(),
            },
        }

    def get_strongest_connections(
        self,
        entity_name: str,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[dict]:
        """Get an entity's strongest connections by weight.

        Args:
            entity_name: Name or ID of the entity
            limit: Maximum number of connections to return

        Returns:
            List of connection info dicts sorted by weight (strongest first)
        """
        from .weights import get_strongest_connections as _get_strongest

        state = self._get_state()
        entity_id = self._resolve_entity(entity_name)
        if not entity_id:
            return []

        connections = _get_strongest(state, entity_id, limit)

        result = []
        for relation, neighbor_id in connections:
            neighbor = state.entities.get(neighbor_id)
            result.append({
                "relation_id": relation.id,
                "connected_to": neighbor.name if neighbor else neighbor_id,
                "relation_type": relation.type,
                "weight": relation.weight,
                "components": {
                    "recency": relation.recency_score,
                    "co_access": relation.co_access_score,
                    "explicit": relation.explicit_weight,
                },
            })

        return result

    def get_weak_relations(
        self,
        max_weight: float = PRUNING_CANDIDATE_THRESHOLD,
        limit: int = DEFAULT_VECTOR_SEARCH_LIMIT,
    ) -> list[dict]:
        """Get relations below a weight threshold (candidates for pruning).

        Args:
            max_weight: Only include relations with weight <= this value
            limit: Maximum number to return

        Returns:
            List of weak relations sorted by weight ascending
        """
        state = self._get_state()
        weak = [r for r in state.relations if r.weight <= max_weight]
        weak.sort(key=lambda r: r.weight)

        result = []
        for r in weak[:limit]:
            from_entity = state.entities.get(r.from_entity)
            to_entity = state.entities.get(r.to_entity)
            result.append({
                "relation_id": r.id,
                "from_entity": from_entity.name if from_entity else r.from_entity,
                "to_entity": to_entity.name if to_entity else r.to_entity,
                "relation_type": r.type,
                "weight": r.weight,
            })

        return result

    # --- Graph Coherence Tools (read-only) ---

    def find_similar(self, name: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[dict]:
        """Find entities with similar names (potential duplicates).

        Delegates to SimilarityChecker for consistent similarity scoring.

        Args:
            name: Entity name to check
            threshold: Similarity threshold 0-1 (default 0.7)

        Returns:
            List of similar entities with similarity scores, sorted by score
        """
        return self._get_similarity().find_similar(name, self._get_state(), threshold)

    def find_orphans(self) -> list[dict]:
        """Find entities with no relations (likely incomplete).

        Uses the _connected_entities index for O(n) iteration without nested loops.

        Returns:
            List of orphan entities with metadata
        """
        state = self._get_state()
        orphans = []
        for eid, entity in state.entities.items():
            if state.is_orphan(eid):
                orphans.append({
                    "id": eid,
                    "name": entity.name,
                    "type": entity.type,
                    "observation_count": len(entity.observations),
                    "created_at": entity.created_at.isoformat(),
                })

        # Sort by creation date (oldest first - most likely to be stale)
        return sorted(orphans, key=lambda x: x["created_at"])

    def get_graph_health(self, full: bool = False) -> dict:
        """Assess overall health of the knowledge graph.

        Args:
            full: If True, include expensive duplicate detection (O(n*k)). Default False for fast O(n) checks.

        Returns:
            Health report with issues and recommendations
        """
        state = self._get_state()
        # Collect issues
        orphans = self.find_orphans()
        duplicates = self._find_duplicate_groups() if full else []
        overloaded = self._find_overloaded_entities()
        weak_relations = self._find_weak_relations()
        cluster_count = self._count_connected_components()

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            orphans, duplicates, overloaded, weak_relations, cluster_count
        )

        return {
            "summary": {
                "total_entities": len(state.entities),
                "total_relations": len(state.relations),
                "orphan_count": len(orphans),
                "duplicate_groups": len(duplicates),
                "overloaded_count": len(overloaded),
                "weak_relation_count": len(weak_relations),
                "cluster_count": cluster_count,
            },
            "issues": {
                "orphans": orphans[:HEALTH_REPORT_ITEM_LIMIT],
                "potential_duplicates": duplicates[:HEALTH_REPORT_ITEM_LIMIT],
                "overloaded_entities": overloaded[:HEALTH_REPORT_ITEM_LIMIT],
                "weak_relations": weak_relations[:HEALTH_REPORT_ITEM_LIMIT],
            },
            "recommendations": recommendations,
        }

    def _get_similar_candidates(self, entity_name: str, limit: int = 10) -> list[str]:
        """Get candidate entity IDs for similarity comparison using vector search.

        Args:
            entity_name: Name of entity to find candidates for
            limit: Max candidates to return (default 10)

        Returns:
            List of entity IDs that are semantically similar (top-k from vector search)
        """
        try:
            # Use vector search to narrow candidates from O(n) to O(k)
            vector_index = self._get_vector_index()
            vector_results = vector_index.search(entity_name, limit=limit)
            return [eid for eid, _score in vector_results]
        except (RuntimeError, ValueError, TypeError, AttributeError):
            # Vector index unavailable - fall back to all entities
            return list(self._get_state().entities.keys())

    def _find_duplicate_groups(self, threshold: float = DUPLICATE_DETECTION_THRESHOLD) -> list[dict]:
        """Find groups of similar entities that may be duplicates.

        Uses vector search to narrow candidates from O(n²) to O(n*k) where k≈10.
        Falls back to brute-force if vector index unavailable.
        """
        state = self._get_state()
        similarity = self._get_similarity()
        duplicates = []
        seen_ids: set[str] = set()

        for eid, entity in state.entities.items():
            if eid in seen_ids:
                continue

            # Get top-k semantically similar candidates via vector search
            candidate_ids = self._get_similar_candidates(entity.name, limit=10)

            # Compute full combined_similarity only for candidates
            similar = []
            entity_lower = entity.name.lower().strip()
            entity_tokens = set(entity_lower.split())

            for candidate_id in candidate_ids:
                if candidate_id == eid or candidate_id in seen_ids:
                    continue
                if candidate_id not in state.entities:
                    continue

                candidate = state.entities[candidate_id]
                sim_score = similarity.combined_similarity(
                    entity.name, candidate.name, entity_lower, entity_tokens
                )

                if sim_score >= threshold:
                    similar.append({
                        "id": candidate_id,
                        "name": candidate.name,
                        "type": candidate.type,
                        "similarity": round(sim_score, 2),
                        "observation_count": len(candidate.observations),
                    })

            if similar:
                # Sort by similarity score descending
                similar.sort(key=lambda x: x["similarity"], reverse=True)
                duplicates.append({
                    "entity": entity.name,
                    "similar_to": [s["name"] for s in similar],
                })
                seen_ids.add(eid)
                seen_ids.update(s["id"] for s in similar)

        return duplicates

    def _find_overloaded_entities(self, max_observations: int = OVERLOADED_ENTITY_OBSERVATION_LIMIT) -> list[dict]:
        """Find entities with too many observations (may need splitting)."""
        state = self._get_state()
        return [
            {"name": e.name, "observation_count": len(e.observations), "type": e.type}
            for e in state.entities.values()
            if len(e.observations) > max_observations
        ]

    def _find_weak_relations(self, min_weight: float = WEAK_RELATION_THRESHOLD) -> list[dict]:
        """Find relations with low weight (may be noise)."""
        state = self._get_state()
        weak = []
        for rel in state.relations:
            if rel.weight < min_weight:
                from_entity = state.entities.get(rel.from_entity)
                to_entity = state.entities.get(rel.to_entity)
                weak.append({
                    "from": from_entity.name if from_entity else rel.from_entity,
                    "to": to_entity.name if to_entity else rel.to_entity,
                    "type": rel.type,
                    "weight": round(rel.weight, 2),
                })
        return weak

    def _count_connected_components(self) -> int:
        """Count number of disconnected subgraphs.

        Uses relation indices for efficient neighbor lookup.
        Uses deque for O(1) popleft instead of O(n) list.pop(0).
        """
        state = self._get_state()

        if not state.entities:
            return 0

        # BFS to find components using relation indices
        visited: set[str] = set()
        components = 0

        for start in state.entities:
            if start in visited:
                continue

            # BFS from this node using deque for O(1) popleft
            queue = deque([start])
            while queue:
                node = queue.popleft()  # O(1) instead of O(n) for list.pop(0)
                if node in visited:
                    continue
                visited.add(node)

                # Get neighbors from indices
                for rel in state.get_outgoing_relations(node):
                    if rel.to_entity not in visited:
                        queue.append(rel.to_entity)
                for rel in state.get_incoming_relations(node):
                    if rel.from_entity not in visited:
                        queue.append(rel.from_entity)

            components += 1

        return components

    def _generate_health_recommendations(
        self,
        orphans: list,
        duplicates: list,
        overloaded: list,
        weak_relations: list,
        cluster_count: int,
    ) -> list[str]:
        """Generate actionable recommendations based on detected issues."""
        recommendations = []
        if orphans:
            recommendations.append(
                f"Connect {len(orphans)} orphan entities or consider merging/deleting them"
            )
        if duplicates:
            recommendations.append(
                f"Review {len(duplicates)} potential duplicate groups for merging"
            )
        if overloaded:
            recommendations.append(
                f"Consider splitting {len(overloaded)} overloaded entities into sub-concepts"
            )
        if weak_relations:
            recommendations.append(
                f"Review {len(weak_relations)} weak relations — may be noise or need strengthening"
            )
        if cluster_count > 1:
            recommendations.append(
                f"Graph has {cluster_count} disconnected clusters — consider linking them"
            )
        if not recommendations:
            recommendations.append("Graph looks healthy! Keep up the good knowledge hygiene.")
        return recommendations

    def suggest_relations(self, entity: str, limit: int = DEFAULT_SUGGESTION_LIMIT) -> list[dict]:
        """Suggest potential relations for an entity.

        Based on:
        - Semantic similarity to other entities
        - Co-occurrence in observations
        - Common patterns

        Args:
            entity: Entity name/ID
            limit: Max suggestions

        Returns:
            List of suggested relations with confidence
        """
        state = self._get_state()
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            return [{"error": f"Entity not found: {entity}"}]

        entity_obj = state.entities[entity_id]

        # Get existing relations to avoid duplicates
        existing_targets = set()
        for rel in state.relations:
            if rel.from_entity == entity_id:
                existing_targets.add(rel.to_entity)
            elif rel.to_entity == entity_id:
                existing_targets.add(rel.from_entity)

        suggestions = []

        # 1. Semantic similarity via vector search (uses lazy-loaded vector_index)
        try:
            entity_text = f"{entity_obj.name} {' '.join(o.text for o in entity_obj.observations)}"
            vector_index = self._get_vector_index()
            search_results = vector_index.search(entity_text, limit=DEFAULT_VECTOR_SEARCH_LIMIT)

            for result_id, score in search_results:
                if result_id == entity_id or result_id in existing_targets:
                    continue
                if result_id not in state.entities:
                    continue

                other = state.entities[result_id]
                if score > SUGGEST_RELATION_CONFIDENCE_THRESHOLD:
                    rel_type = self._guess_relation_type(entity_obj, other)
                    suggestions.append({
                        "target": other.name,
                        "target_id": result_id,
                        "suggested_relation": rel_type,
                        "confidence": round(score, 2),
                        "reason": "semantic similarity",
                    })
        except Exception as e:
            logger.debug(f"Semantic relation suggestions unavailable: {e}")

        # 2. Co-occurrence in observations (entity name mentioned in other's observations)
        entity_text_lower = " ".join(o.text.lower() for o in entity_obj.observations)

        for oid, other in state.entities.items():
            if oid == entity_id or oid in existing_targets:
                continue

            # Check if other entity's name appears in our observations
            if other.name.lower() in entity_text_lower:
                suggestions.append({
                    "target": other.name,
                    "target_id": oid,
                    "suggested_relation": "mentions",
                    "confidence": CO_OCCURRENCE_CONFIDENCE,
                    "reason": f"'{other.name}' mentioned in observations",
                })

            # Check if our name appears in other's observations
            other_text_lower = " ".join(o.text.lower() for o in other.observations)
            if entity_obj.name.lower() in other_text_lower:
                # Only add if not already suggested
                existing = [s for s in suggestions if s["target_id"] == oid]
                if not existing:
                    suggestions.append({
                        "target": other.name,
                        "target_id": oid,
                        "suggested_relation": "mentioned_by",
                        "confidence": SHARED_RELATION_CONFIDENCE,
                        "reason": f"mentioned in '{other.name}' observations",
                    })

        # Dedupe and sort by confidence
        seen = set()
        unique = []
        for s in sorted(suggestions, key=lambda x: x["confidence"], reverse=True):
            if s["target_id"] not in seen:
                seen.add(s["target_id"])
                unique.append(s)

        return unique[:limit]

    def _guess_relation_type(self, entity, other) -> str:
        """Guess appropriate relation type based on entity types."""
        type_map = {
            ("concept", "project"): "used_by",
            ("project", "concept"): "uses",
            ("decision", "concept"): "decides_on",
            ("concept", "decision"): "decided_by",
            ("pattern", "concept"): "applies_to",
            ("concept", "pattern"): "has_pattern",
            ("learning", "concept"): "about",
            ("question", "concept"): "about",
            ("project", "project"): "related_to",
        }
        return type_map.get((entity.type, other.type), "related_to")
