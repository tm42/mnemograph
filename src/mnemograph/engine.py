"""Memory engine - orchestrates event store, state, and operations."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .events import EventStore
from .models import Entity, MemoryEvent, Observation, Relation
from .state import GraphState, materialize, materialize_at, apply_event
from .timeutil import parse_time_reference

# --- Constants ---
# Default limits for queries
DEFAULT_QUERY_LIMIT = 10
DEFAULT_RECENT_LIMIT = 5
DEFAULT_SUGGESTION_LIMIT = 5
DEFAULT_VECTOR_SEARCH_LIMIT = 20

# Token budgets for tiered retrieval
SHALLOW_CONTEXT_TOKENS = 500
MEDIUM_CONTEXT_TOKENS = 2000
DEEP_CONTEXT_TOKENS = 5000

# Similarity and weight thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DUPLICATE_DETECTION_THRESHOLD = 0.8
WEAK_RELATION_THRESHOLD = 0.2
PRUNING_CANDIDATE_THRESHOLD = 0.1
SUGGEST_RELATION_CONFIDENCE_THRESHOLD = 0.4

# Health check limits
OVERLOADED_ENTITY_OBSERVATION_LIMIT = 15
HEALTH_REPORT_ITEM_LIMIT = 10

# Suggestion confidence scores
CO_OCCURRENCE_CONFIDENCE = 0.7
SHARED_RELATION_CONFIDENCE = 0.65

# Similarity calculation weights
SUBSTRING_SIMILARITY_WEIGHT = 0.9
TOKEN_SIMILARITY_WEIGHT = 0.8
EMBEDDING_SIMILARITY_WEIGHT = 0.8
AFFIX_MATCH_BONUS = 0.2


class MemoryEngine:
    """Main entry point for memory operations.

    Thread-safety: MemoryEngine is designed for single-process use. Running
    multiple instances with the same MEMORY_PATH may cause data corruption,
    particularly in the co-access cache. For multi-process scenarios, use
    separate memory directories or implement external locking.

    Performance: Uses O(1) indices for entity lookups and incremental state
    updates. Entity operations are O(1) regardless of graph size. Relation
    operations are O(1) for creation, O(degree) for neighbor queries.
    """

    def __init__(self, memory_dir: Path, session_id: str):
        self.memory_dir = memory_dir
        self.session_id = session_id
        self.event_store = EventStore(memory_dir / "events.jsonl")
        self.state: GraphState = self._load_state()
        self._vector_index = None  # Lazy-loaded
        self._co_access_cache_path = memory_dir / "co_access_cache.json"
        self._load_co_access_cache()

    def _load_state(self) -> GraphState:
        """Load state from events."""
        events = self.event_store.read_all()
        return materialize(events)

    def _load_co_access_cache(self) -> None:
        """Load cached co-access scores into relations.

        Co-access scores are learned from usage patterns and cached
        separately from the event-sourced state.
        """
        if not self._co_access_cache_path.exists():
            return

        try:
            cache = json.loads(self._co_access_cache_path.read_text())
            for rel in self.state.relations:
                if rel.id in cache:
                    rel.co_access_score = cache[rel.id].get("co_access_score", 0.0)
                    rel.access_count = cache[rel.id].get("access_count", 0)
                    if "last_accessed" in cache[rel.id]:
                        rel.last_accessed = datetime.fromisoformat(cache[rel.id]["last_accessed"])
        except (json.JSONDecodeError, ValueError):
            # Corrupted cache, ignore
            pass

    def save_co_access_cache(self) -> None:
        """Save co-access scores to cache file.

        Call this periodically or on shutdown to persist learned scores.
        """
        cache = {}
        for rel in self.state.relations:
            if rel.co_access_score > 0 or rel.access_count > 0:
                cache[rel.id] = {
                    "co_access_score": rel.co_access_score,
                    "access_count": rel.access_count,
                    "last_accessed": rel.last_accessed.isoformat(),
                }

        self._co_access_cache_path.write_text(json.dumps(cache, indent=2))

    def _emit(self, op: str, data: dict) -> MemoryEvent:
        """Emit an event and update local state incrementally.

        Uses apply_event() for O(1) updates instead of full replay.
        """
        event = MemoryEvent(
            op=op,  # type: ignore (we know op is valid EventOp)
            session_id=self.session_id,
            source="cc",
            data=data,
        )
        self.event_store.append(event)
        # Incremental update - O(1) instead of O(events)
        apply_event(self.state, event)
        return event

    # --- Entity operations ---

    def create_entities(self, entities: list[dict]) -> list[Entity]:
        """Create multiple entities."""
        created = []
        for entity_data in entities:
            # Build observations from input
            observations = [
                Observation(text=obs, source=self.session_id)
                for obs in entity_data.get("observations", [])
            ]

            entity = Entity(
                name=entity_data["name"],
                type=entity_data.get("entityType", entity_data.get("type", "entity")),
                observations=observations,
                created_by=self.session_id,
            )
            self._emit("create_entity", entity.model_dump(mode="json"))
            created.append(entity)
        return created

    def delete_entities(self, names: list[str]) -> int:
        """Delete entities by name. Returns count of deleted."""
        deleted = 0
        for name in names:
            entity_id = self._resolve_entity(name)
            if entity_id:
                self._emit("delete_entity", {"id": entity_id})
                deleted += 1
        return deleted

    # --- Relation operations ---

    def create_relations(self, relations: list[dict]) -> dict:
        """Create multiple relations with detailed result.

        Returns dict with 'created' list, 'failed' list, and 'summary'.
        Failed items include reason for failure.
        """
        created = []
        failed = []

        for rel_data in relations:
            from_name = rel_data["from"]
            to_name = rel_data["to"]
            from_id = self._resolve_entity(from_name)
            to_id = self._resolve_entity(to_name)

            if not from_id:
                failed.append({
                    "from": from_name,
                    "to": to_name,
                    "relationType": rel_data.get("relationType", ""),
                    "reason": f"from_entity not found: {from_name}",
                })
                continue
            if not to_id:
                failed.append({
                    "from": from_name,
                    "to": to_name,
                    "relationType": rel_data.get("relationType", ""),
                    "reason": f"to_entity not found: {to_name}",
                })
                continue

            relation = Relation(
                from_entity=from_id,
                to_entity=to_id,
                type=rel_data["relationType"],
                created_by=self.session_id,
            )
            self._emit("create_relation", relation.model_dump(mode="json"))
            created.append(relation)

        return {
            "created": [r.model_dump(mode="json") for r in created],
            "failed": failed,
            "summary": f"Created {len(created)}, failed {len(failed)}",
        }

    def delete_relations(self, relations: list[dict]) -> int:
        """Delete relations. Returns count of deleted."""
        deleted = 0
        for rel_data in relations:
            from_id = self._resolve_entity(rel_data["from"])
            to_id = self._resolve_entity(rel_data["to"])
            if from_id and to_id:
                self._emit(
                    "delete_relation",
                    {
                        "from_entity": from_id,
                        "to_entity": to_id,
                        "type": rel_data["relationType"],
                    },
                )
                deleted += 1
        return deleted

    # --- Observation operations ---

    def add_observations(self, observations: list[dict]) -> list[dict]:
        """Add observations to existing entities."""
        results = []
        for obs_data in observations:
            entity_id = self._resolve_entity(obs_data["entityName"])
            if entity_id:
                added = []
                for content in obs_data.get("contents", []):
                    obs = Observation(text=content, source=self.session_id)
                    self._emit(
                        "add_observation",
                        {"entity_id": entity_id, "observation": obs.model_dump(mode="json")},
                    )
                    added.append(content)
                results.append({"entityName": obs_data["entityName"], "addedObservations": added})
        return results

    def delete_observations(self, deletions: list[dict]) -> int:
        """Delete specific observations from entities."""
        deleted = 0
        for deletion in deletions:
            entity_id = self._resolve_entity(deletion["entityName"])
            if entity_id and entity_id in self.state.entities:
                entity = self.state.entities[entity_id]
                for obs_text in deletion.get("observations", []):
                    # Find observation by text content
                    for obs in entity.observations:
                        if obs.text == obs_text:
                            self._emit(
                                "delete_observation",
                                {"entity_id": entity_id, "observation_id": obs.id},
                            )
                            deleted += 1
                            break
        return deleted

    # --- Query operations ---

    def read_graph(self) -> dict:
        """Return full graph state."""
        return {
            "entities": [e.model_dump(mode="json") for e in self.state.entities.values()],
            "relations": [r.model_dump(mode="json") for r in self.state.relations],
        }

    def search_nodes(self, query: str) -> dict:
        """Simple text search across names, types, observations."""
        query_lower = query.lower()
        matching_entities = []
        matching_entity_ids = set()

        for entity in self.state.entities.values():
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

        # Track access for matched entities
        self._track_access(list(matching_entity_ids))

        # Include relations between matching entities
        matching_relations = [
            r for r in self.state.relations
            if r.from_entity in matching_entity_ids or r.to_entity in matching_entity_ids
        ]

        return {
            "entities": [e.model_dump(mode="json") for e in matching_entities],
            "relations": [r.model_dump(mode="json") for r in matching_relations],
        }

    def open_nodes(self, names: list[str]) -> dict:
        """Get specific entities by name, their relations, and neighbors."""
        entities = []
        entity_ids = set()

        for name in names:
            entity_id = self._resolve_entity(name)
            if entity_id and entity_id in self.state.entities:
                entities.append(self.state.entities[entity_id])
                entity_ids.add(entity_id)

        # Track access
        self._track_access(list(entity_ids))

        # Include relations involving these entities
        relations = [
            r for r in self.state.relations
            if r.from_entity in entity_ids or r.to_entity in entity_ids
        ]

        # Find neighbor IDs (entities connected via relations)
        neighbor_ids = set()
        for r in relations:
            if r.from_entity not in entity_ids:
                neighbor_ids.add(r.from_entity)
            if r.to_entity not in entity_ids:
                neighbor_ids.add(r.to_entity)

        neighbors = [
            self.state.entities[nid] for nid in neighbor_ids
            if nid in self.state.entities
        ]

        return {
            "entities": [e.model_dump(mode="json") for e in entities],
            "relations": [r.model_dump(mode="json") for r in relations],
            "neighbors": [n.model_dump(mode="json") for n in neighbors],
        }

    # --- Helpers ---

    def _resolve_entity(self, name_or_id: str) -> str | None:
        """Find entity by name or ID. O(1) lookup using indices."""
        # Check if it's an ID
        if name_or_id in self.state.entities:
            return name_or_id
        # Check name index
        return self.state.get_entity_id_by_name(name_or_id)

    def _track_access(self, entity_ids: list[str]) -> None:
        """Update access counts for retrieved entities (in-memory only, not persisted)."""
        now = datetime.now(timezone.utc)
        for entity_id in entity_ids:
            if entity_id in self.state.entities:
                entity = self.state.entities[entity_id]
                entity.access_count += 1
                entity.last_accessed = now

    # --- Phase 2: Richer queries ---

    def get_recent_entities(self, limit: int = DEFAULT_QUERY_LIMIT) -> list[Entity]:
        """Get most recently updated entities."""
        return sorted(
            self.state.entities.values(),
            key=lambda e: e.updated_at,
            reverse=True,
        )[:limit]

    def get_hot_entities(self, limit: int = DEFAULT_QUERY_LIMIT) -> list[Entity]:
        """Get most frequently accessed entities."""
        return sorted(
            self.state.entities.values(),
            key=lambda e: e.access_count,
            reverse=True,
        )[:limit]

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Filter entities by type."""
        return [e for e in self.state.entities.values() if e.type == entity_type]

    def get_entity_neighbors(self, entity_id: str, include_entity: bool = True) -> dict:
        """Get entity and its directly connected neighbors via relations."""
        entity = self.state.entities.get(entity_id)
        if not entity:
            return {"entity": None, "neighbors": [], "outgoing": [], "incoming": []}

        outgoing = [r for r in self.state.relations if r.from_entity == entity_id]
        incoming = [r for r in self.state.relations if r.to_entity == entity_id]

        neighbor_ids = set(
            [r.to_entity for r in outgoing] + [r.from_entity for r in incoming]
        )
        neighbors = [self.state.entities[nid] for nid in neighbor_ids if nid in self.state.entities]

        # Track access
        accessed = [entity_id] + list(neighbor_ids)
        self._track_access(accessed)

        return {
            "entity": entity.model_dump(mode="json") if include_entity else None,
            "neighbors": [n.model_dump(mode="json") for n in neighbors],
            "outgoing": [r.model_dump(mode="json") for r in outgoing],
            "incoming": [r.model_dump(mode="json") for r in incoming],
        }

    # --- Phase 3: Vector search ---

    @property
    def vector_index(self):
        """Lazy-load vector index."""
        if self._vector_index is None:
            from .vectors import VectorIndex
            self._vector_index = VectorIndex(self.memory_dir / "vectors.db")
            # Index all existing entities
            self._vector_index.reindex_all(list(self.state.entities.values()))
        return self._vector_index

    def search_semantic(
        self,
        query: str,
        limit: int = DEFAULT_QUERY_LIMIT,
        type_filter: str | None = None
    ) -> dict:
        """Semantic search using embeddings."""
        results = self.vector_index.search(query, limit, type_filter)

        entities = []
        entity_ids = []
        for entity_id, score in results:
            if entity_id in self.state.entities:
                entity = self.state.entities[entity_id]
                entity_dict = entity.model_dump(mode="json")
                entity_dict["_score"] = score  # Attach similarity score
                entities.append(entity_dict)
                entity_ids.append(entity_id)

        # Track access
        self._track_access(entity_ids)

        # Include relations between matched entities
        entity_id_set = set(entity_ids)
        relations = [
            r.model_dump(mode="json") for r in self.state.relations
            if r.from_entity in entity_id_set or r.to_entity in entity_id_set
        ]

        return {
            "entities": entities,
            "relations": relations,
        }

    def ensure_indexed(self, entity_id: str) -> None:
        """Ensure a specific entity is indexed (call after mutations)."""
        if self._vector_index is not None and entity_id in self.state.entities:
            self._vector_index.index_entity(self.state.entities[entity_id])

    # --- Phase 4: Tiered retrieval ---

    def memory_context(
        self,
        depth: str = "shallow",
        query: str | None = None,
        focus: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Get memory context at varying depth levels.

        Args:
            depth: 'shallow' (summary), 'medium' (search + neighbors), 'deep' (multi-hop)
            query: Search query for medium depth (uses semantic search)
            focus: Entity names to focus on
            max_tokens: Token budget (defaults vary by depth)

        Returns:
            Dict with depth, tokens_estimate, content, entity_count, relation_count
        """
        from .retrieval import get_shallow_context, get_medium_context, get_deep_context

        if depth == "shallow":
            tokens = max_tokens or SHALLOW_CONTEXT_TOKENS
            result = get_shallow_context(self.state, tokens)

        elif depth == "medium":
            tokens = max_tokens or MEDIUM_CONTEXT_TOKENS
            # Get vector search results if query provided
            vector_results = None
            if query:
                vector_results = self.vector_index.search(query, limit=DEFAULT_QUERY_LIMIT)
            result = get_medium_context(self.state, vector_results, focus, tokens)
            # Save co-access learning
            self.save_co_access_cache()

        elif depth == "deep":
            tokens = max_tokens or DEEP_CONTEXT_TOKENS
            result = get_deep_context(self.state, focus, tokens)
            # Save co-access learning
            self.save_co_access_cache()

        else:
            raise ValueError(f"Invalid depth: {depth}. Use 'shallow', 'medium', or 'deep'.")

        return {
            "depth": result.depth,
            "tokens_estimate": result.tokens_estimate,
            "content": result.content,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
        }

    # --- Event Rewind ---

    def state_at(self, timestamp: str | datetime) -> GraphState:
        """Get graph state at a specific point in time.

        Args:
            timestamp: ISO datetime string, relative reference ("7 days ago"),
                       or datetime object (will be treated as UTC if naive)

        Returns:
            GraphState as it existed at that timestamp
        """
        if isinstance(timestamp, str):
            ts = parse_time_reference(timestamp)
        else:
            ts = timestamp
            # Ensure timezone-aware (assume UTC if naive)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

        events = self.event_store.read_all()
        return materialize_at(events, ts)

    def events_between(
        self,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> list[MemoryEvent]:
        """Get events in a time range.

        Args:
            start: Start time (ISO string, relative reference, or datetime)
            end: End time (default: now)

        Returns:
            List of events in the range
        """
        if isinstance(start, str):
            start_ts = parse_time_reference(start)
        else:
            start_ts = start

        if end is None:
            end_ts = datetime.now(timezone.utc)
        elif isinstance(end, str):
            end_ts = parse_time_reference(end)
        else:
            end_ts = end

        events = self.event_store.read_all()
        return [e for e in events if start_ts <= e.ts <= end_ts]

    def diff_between(
        self,
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> dict:
        """Compute what changed between two points in time.

        Args:
            start: Start time
            end: End time (default: now)

        Returns:
            Dict with entities (added/removed/modified) and relations (added/removed)
        """
        if isinstance(start, str):
            start_ts = parse_time_reference(start)
        else:
            start_ts = start

        if end is None:
            end_ts = datetime.now(timezone.utc)
        elif isinstance(end, str):
            end_ts = parse_time_reference(end)
        else:
            end_ts = end

        events = self.event_store.read_all()
        state_start = materialize_at(events, start_ts)
        state_end = materialize_at(events, end_ts)

        return {
            "start_timestamp": start_ts.isoformat(),
            "end_timestamp": end_ts.isoformat(),
            "entities": self._diff_entities(state_start, state_end),
            "relations": self._diff_relations(state_start, state_end),
            "event_count": len([e for e in events if start_ts < e.ts <= end_ts]),
        }

    def _diff_entities(self, old: GraphState, new: GraphState) -> dict:
        """Compute entity differences between two states."""
        old_ids = set(old.entities.keys())
        new_ids = set(new.entities.keys())

        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids
        common_ids = old_ids & new_ids

        # Check for modifications in common entities
        modified = []
        for eid in common_ids:
            old_entity = old.entities[eid]
            new_entity = new.entities[eid]

            changes = self._entity_changes(old_entity, new_entity)
            if changes:
                modified.append({
                    "id": eid,
                    "name": new_entity.name,
                    "changes": changes,
                })

        return {
            "added": [new.entities[eid].to_summary() for eid in added_ids],
            "removed": [old.entities[eid].to_summary() for eid in removed_ids],
            "modified": modified,
        }

    def _entity_changes(self, old: Entity, new: Entity) -> dict:
        """Compute specific changes between two entity versions."""
        changes = {}

        if old.name != new.name:
            changes["name"] = {"old": old.name, "new": new.name}
        if old.type != new.type:
            changes["type"] = {"old": old.type, "new": new.type}

        # Check observations
        old_obs_ids = {o.id for o in old.observations}
        new_obs_ids = {o.id for o in new.observations}

        added_obs = new_obs_ids - old_obs_ids
        removed_obs = old_obs_ids - new_obs_ids

        if added_obs or removed_obs:
            changes["observations"] = {
                "added": len(added_obs),
                "removed": len(removed_obs),
            }

        return changes

    def _diff_relations(self, old: GraphState, new: GraphState) -> dict:
        """Compute relation differences between two states."""
        old_ids = {r.id for r in old.relations}
        new_ids = {r.id for r in new.relations}

        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids

        # Build lookup maps
        new_by_id = {r.id: r for r in new.relations}
        old_by_id = {r.id: r for r in old.relations}

        return {
            "added": [new_by_id[rid].to_summary() for rid in added_ids],
            "removed": [old_by_id[rid].to_summary() for rid in removed_ids],
        }

    def get_entity_history(self, entity_name: str) -> list[dict]:
        """Get the history of changes to an entity.

        Returns list of events that affected this entity, oldest first.
        """
        entity_id = self._resolve_entity(entity_name)
        if not entity_id:
            return []

        events = self.event_store.read_all()

        relevant = [
            e for e in events
            if e.data.get("id") == entity_id
            or e.data.get("entity_id") == entity_id
        ]

        return [
            {
                "timestamp": e.ts.isoformat(),
                "operation": e.op,
                "session_id": e.session_id,
                "data": e.data,
            }
            for e in relevant
        ]

    # --- Edge Weights ---

    def get_relation_weight(self, relation_id: str) -> dict | None:
        """Get weight breakdown for a relation.

        Args:
            relation_id: ID of the relation (full ID or prefix)

        Returns:
            Weight breakdown dict, or None if relation not found
        """
        # Support prefix matching
        relation = None
        for r in self.state.relations:
            if r.id == relation_id or r.id.startswith(relation_id):
                relation = r
                break

        if not relation:
            return None

        # Get entity names for context
        from_entity = self.state.entities.get(relation.from_entity)
        to_entity = self.state.entities.get(relation.to_entity)

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
                "access_count": relation.access_count,
                "last_accessed": relation.last_accessed.isoformat(),
                "created_at": relation.created_at.isoformat(),
            },
        }

    def set_relation_importance(self, relation_id: str, importance: float) -> dict | None:
        """Set explicit weight for a relation.

        Args:
            relation_id: ID of the relation (full ID or prefix)
            importance: Value from 0.0 (unimportant) to 1.0 (critical)

        Returns:
            Updated weight breakdown, or None if relation not found

        Raises:
            ValueError: If importance is not between 0.0 and 1.0
        """
        if not 0.0 <= importance <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")

        # Find the relation
        relation = None
        for r in self.state.relations:
            if r.id == relation_id or r.id.startswith(relation_id):
                relation = r
                break

        if not relation:
            return None

        old_weight = relation.explicit_weight

        # Emit event for replayability
        self._emit("update_weight", {
            "relation_id": relation.id,
            "weight_type": "explicit",
            "old_value": old_weight,
            "new_value": importance,
        })

        return self.get_relation_weight(relation.id)

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

        entity_id = self._resolve_entity(entity_name)
        if not entity_id:
            return []

        connections = _get_strongest(self.state, entity_id, limit)

        result = []
        for relation, neighbor_id in connections:
            neighbor = self.state.entities.get(neighbor_id)
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
        weak = [r for r in self.state.relations if r.weight <= max_weight]
        weak.sort(key=lambda r: r.weight)

        result = []
        for r in weak[:limit]:
            from_entity = self.state.entities.get(r.from_entity)
            to_entity = self.state.entities.get(r.to_entity)
            result.append({
                "relation_id": r.id,
                "from_entity": from_entity.name if from_entity else r.from_entity,
                "to_entity": to_entity.name if to_entity else r.to_entity,
                "relation_type": r.type,
                "weight": r.weight,
            })

        return result

    # --- Universal Agent Tools ---

    def get_primer(self) -> dict:
        """Get oriented with this knowledge graph.

        Returns summary of what's available and how to use the tools.
        Call at session start to understand the knowledge graph context.
        """
        # Get type counts
        type_counts = {}
        for entity in self.state.entities.values():
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

        # Get recent entities
        recent = self.get_recent_entities(limit=DEFAULT_RECENT_LIMIT)
        recent_summary = [
            {"name": e.name, "type": e.type}
            for e in recent
        ]

        return {
            "status": {
                "entity_count": len(self.state.entities),
                "relation_count": len(self.state.relations),
                "types": type_counts,
            },
            "recent_activity": recent_summary,
            "tools": {
                "retrieval": [
                    "memory_context(depth, query) - Get relevant context (shallow/medium/deep)",
                    "search_nodes(query) - Text search across entities",
                    "search_semantic(query) - Meaning-based search with embeddings",
                    "open_nodes(names) - Get specific entities with relations",
                ],
                "creation": [
                    "create_entities(entities) - Create new knowledge nodes",
                    "add_observations(observations) - Add info to existing entities",
                    "create_relations(relations) - Link entities together",
                ],
                "history": [
                    "get_state_at(timestamp) - View graph at any point in time",
                    "diff_timerange(start, end) - See what changed",
                    "get_entity_history(name) - Full changelog for an entity",
                ],
            },
            "quick_start": "Call memory_context(depth='shallow') for a summary, or search_semantic(query='...') to find relevant knowledge.",
        }

    def session_start(self, project_hint: str | None = None) -> dict:
        """Signal session start and get initial context.

        Args:
            project_hint: Optional project name or path for context

        Returns:
            Initial context to prime the session
        """
        # Get shallow context for session priming
        context_result = self.memory_context(depth="shallow")

        # Find project entity if hint provided
        project_entity = None
        if project_hint:
            project_id = self._resolve_entity(project_hint)
            if project_id:
                project_entity = self.state.entities.get(project_id)

        return {
            "session_id": self.session_id,
            "memory_summary": {
                "entity_count": len(self.state.entities),
                "relation_count": len(self.state.relations),
            },
            "context": context_result["content"],
            "project": project_entity.name if project_entity else None,
            "tip": "Use memory_context(depth='medium', query='...') for specific topics.",
        }

    def session_end(self, summary: str | None = None) -> dict:
        """Signal session end, optionally save summary.

        Args:
            summary: Optional session summary to store as observation

        Returns:
            Session end acknowledgement
        """
        stored_summary = False

        if summary:
            # Try to find a project entity to attach the summary to
            project_entities = self.get_entities_by_type("project")

            if project_entities:
                # Attach to most recently accessed project
                project_entities.sort(key=lambda e: e.last_accessed or e.created_at, reverse=True)
                project = project_entities[0]

                # Add observation with session summary
                obs = Observation(
                    text=f"[Session {self.session_id}] {summary}",
                    source=self.session_id,
                )
                self._emit(
                    "add_observation",
                    {"entity_id": project.id, "observation": obs.model_dump(mode="json")},
                )
                stored_summary = True
            else:
                # Create a learning entity for the summary
                learning = Entity(
                    name=f"Session Summary ({self.session_id[:8]})",
                    type="learning",
                    observations=[Observation(text=summary, source=self.session_id)],
                    created_by=self.session_id,
                )
                self._emit("create_entity", learning.model_dump(mode="json"))
                stored_summary = True

        # Save co-access cache
        self.save_co_access_cache()

        return {
            "status": "session_ended",
            "summary_stored": stored_summary,
            "tip": "Key learnings can be stored with create_entities or add_observations anytime.",
        }

    # --- Graph Coherence Tools ---

    def find_similar(self, name: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[dict]:
        """Find entities with similar names (potential duplicates).

        Uses combination of:
        - Substring containment (React in ReactJS)
        - Jaccard similarity on tokens
        - Prefix/suffix matching
        - Semantic similarity from embeddings

        Args:
            name: Entity name to check
            threshold: Similarity threshold 0-1 (default 0.7)

        Returns:
            List of similar entities with similarity scores, sorted by score
        """
        results = []
        name_lower = name.lower().strip()
        name_tokens = set(name_lower.split())

        for eid, entity in self.state.entities.items():
            entity_lower = entity.name.lower()

            # Skip exact matches
            if entity_lower == name_lower:
                continue

            entity_tokens = set(entity_lower.split())

            # 1. Substring containment (strong signal for "React" in "ReactJS")
            substring_score = 0.0
            if name_lower in entity_lower or entity_lower in name_lower:
                # Longer substring relative to total length = higher score
                shorter = min(len(name_lower), len(entity_lower))
                longer = max(len(name_lower), len(entity_lower))
                substring_score = shorter / longer  # e.g., "React"(5) in "ReactJS"(7) = 0.71

            # 2. Jaccard similarity on tokens (for multi-word names)
            intersection = len(name_tokens & entity_tokens)
            union = len(name_tokens | entity_tokens)
            jaccard = intersection / union if union > 0 else 0

            # 3. Prefix/suffix matching bonus
            prefix_match = name_lower.startswith(entity_lower) or entity_lower.startswith(name_lower)
            suffix_match = name_lower.endswith(entity_lower) or entity_lower.endswith(name_lower)
            affix_bonus = AFFIX_MATCH_BONUS if (prefix_match or suffix_match) else 0

            # 4. Embedding similarity (if vector index available)
            embedding_sim = 0.0
            if self._vector_index is not None:
                try:
                    search_results = self.vector_index.search(name, limit=DEFAULT_VECTOR_SEARCH_LIMIT)
                    for result_id, score in search_results:
                        if result_id == eid:
                            embedding_sim = score
                            break
                except Exception:
                    pass

            # Combined score - use max of different approaches
            # This handles different cases well:
            # - "React" vs "ReactJS": substring_score dominates
            # - "React Native" vs "React": jaccard + affix
            # - Completely different names: embedding_sim dominates
            similarity = max(
                substring_score * SUBSTRING_SIMILARITY_WEIGHT + affix_bonus,  # Substring-based
                jaccard * TOKEN_SIMILARITY_WEIGHT + affix_bonus,  # Token-based
                embedding_sim * EMBEDDING_SIMILARITY_WEIGHT,  # Semantic-based
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

    def find_orphans(self) -> list[dict]:
        """Find entities with no relations (likely incomplete).

        Uses the _connected_entities index for O(n) iteration without nested loops.

        Returns:
            List of orphan entities with metadata
        """
        orphans = []
        for eid, entity in self.state.entities.items():
            if self.state.is_orphan(eid):
                orphans.append({
                    "id": eid,
                    "name": entity.name,
                    "type": entity.type,
                    "observation_count": len(entity.observations),
                    "created_at": entity.created_at.isoformat(),
                    "last_accessed": entity.last_accessed.isoformat() if entity.last_accessed else None,
                })

        # Sort by creation date (oldest first - most likely to be stale)
        return sorted(orphans, key=lambda x: x["created_at"])

    def merge_entities(self, source: str, target: str, delete_source: bool = True) -> dict:
        """Merge source entity into target.

        - Source's observations are appended to target (with merge note)
        - Source's relations are redirected to target
        - Source is deleted (unless delete_source=False)

        Args:
            source: Entity name/ID to merge FROM
            target: Entity name/ID to merge INTO
            delete_source: Whether to delete source after (default True)

        Returns:
            Result with merged entity details
        """
        # Validate and resolve merge targets
        validation = self._validate_merge_targets(source, target)
        if "error" in validation:
            return validation

        source_id = validation["source_id"]
        target_id = validation["target_id"]
        source_entity = self.state.entities[source_id]
        target_entity = self.state.entities[target_id]

        # Perform merge operations
        observations_merged = self._merge_observations(source_id, target_id)
        relations_redirected = self._redirect_relations(source_id, target_id)

        if delete_source:
            self._emit("delete_entity", {"id": source_id})

        return {
            "status": "merged",
            "source": source_entity.name,
            "target": target_entity.name,
            "observations_merged": observations_merged,
            "relations_redirected": relations_redirected,
            "source_deleted": delete_source,
        }

    def _validate_merge_targets(self, source: str, target: str) -> dict:
        """Validate that merge source and target exist and are different.

        Returns:
            On success: {"source_id": str, "target_id": str}
            On failure: {"error": str}
        """
        source_id = self._resolve_entity(source)
        target_id = self._resolve_entity(target)

        if not source_id:
            return {"error": f"Source entity not found: {source}"}
        if not target_id:
            return {"error": f"Target entity not found: {target}"}
        if source_id == target_id:
            return {"error": "Source and target are the same entity"}
        return {"source_id": source_id, "target_id": target_id}

    def _merge_observations(self, source_id: str, target_id: str) -> int:
        """Copy observations from source to target with merge note."""
        source_entity = self.state.entities[source_id]
        count = 0
        for obs in source_entity.observations:
            merged_text = f"[Merged from {source_entity.name}] {obs.text}"
            new_obs = Observation(text=merged_text, source=self.session_id)
            self._emit(
                "add_observation",
                {"entity_id": target_id, "observation": new_obs.model_dump(mode="json")},
            )
            count += 1
        return count

    def _redirect_relations(self, source_id: str, target_id: str) -> int:
        """Redirect relations from source to target."""
        redirected = 0
        relations_to_delete = []

        for rel in self.state.relations:
            if rel.from_entity == source_id:
                if rel.to_entity != target_id:  # Avoid self-reference
                    new_rel = Relation(
                        from_entity=target_id,
                        to_entity=rel.to_entity,
                        type=rel.type,
                        created_by=self.session_id,
                    )
                    self._emit("create_relation", new_rel.model_dump(mode="json"))
                    redirected += 1
                relations_to_delete.append(rel)

            elif rel.to_entity == source_id:
                if rel.from_entity != target_id:  # Avoid self-reference
                    new_rel = Relation(
                        from_entity=rel.from_entity,
                        to_entity=target_id,
                        type=rel.type,
                        created_by=self.session_id,
                    )
                    self._emit("create_relation", new_rel.model_dump(mode="json"))
                    redirected += 1
                relations_to_delete.append(rel)

        # Delete old relations
        for rel in relations_to_delete:
            self._emit(
                "delete_relation",
                {"from_entity": rel.from_entity, "to_entity": rel.to_entity, "type": rel.type},
            )

        return redirected

    def get_graph_health(self) -> dict:
        """Assess overall health of the knowledge graph.

        Returns:
            Health report with issues and recommendations
        """
        # Collect issues
        orphans = self.find_orphans()
        duplicates = self._find_duplicate_groups()
        overloaded = self._find_overloaded_entities()
        weak_relations = self._find_weak_relations()
        cluster_count = self._count_connected_components()

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            orphans, duplicates, overloaded, weak_relations, cluster_count
        )

        return {
            "summary": {
                "total_entities": len(self.state.entities),
                "total_relations": len(self.state.relations),
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

    def _find_duplicate_groups(self, threshold: float = DUPLICATE_DETECTION_THRESHOLD) -> list[dict]:
        """Find groups of similar entities that may be duplicates."""
        duplicates = []
        seen_ids: set[str] = set()

        for eid, entity in self.state.entities.items():
            if eid in seen_ids:
                continue

            similar = self.find_similar(entity.name, threshold=threshold)
            if similar:
                duplicates.append({
                    "entity": entity.name,
                    "similar_to": [s["name"] for s in similar],
                })
                seen_ids.add(eid)
                seen_ids.update(s["id"] for s in similar)

        return duplicates

    def _find_overloaded_entities(self, max_observations: int = OVERLOADED_ENTITY_OBSERVATION_LIMIT) -> list[dict]:
        """Find entities with too many observations (may need splitting)."""
        return [
            {"name": e.name, "observation_count": len(e.observations), "type": e.type}
            for e in self.state.entities.values()
            if len(e.observations) > max_observations
        ]

    def _find_weak_relations(self, min_weight: float = WEAK_RELATION_THRESHOLD) -> list[dict]:
        """Find relations with low weight (may be noise)."""
        weak = []
        for rel in self.state.relations:
            if rel.weight < min_weight:
                from_entity = self.state.entities.get(rel.from_entity)
                to_entity = self.state.entities.get(rel.to_entity)
                weak.append({
                    "from": from_entity.name if from_entity else rel.from_entity,
                    "to": to_entity.name if to_entity else rel.to_entity,
                    "type": rel.type,
                    "weight": round(rel.weight, 2),
                })
        return weak

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

    def _count_connected_components(self) -> int:
        """Count number of disconnected subgraphs.

        Uses relation indices for efficient neighbor lookup.
        """
        if not self.state.entities:
            return 0

        # BFS to find components using relation indices
        visited: set[str] = set()
        components = 0

        for start in self.state.entities:
            if start in visited:
                continue

            # BFS from this node
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)

                # Get neighbors from indices
                for rel in self.state.get_outgoing_relations(node):
                    if rel.to_entity not in visited:
                        queue.append(rel.to_entity)
                for rel in self.state.get_incoming_relations(node):
                    if rel.from_entity not in visited:
                        queue.append(rel.from_entity)

            components += 1

        return components

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
        entity_id = self._resolve_entity(entity)
        if not entity_id:
            return [{"error": f"Entity not found: {entity}"}]

        entity_obj = self.state.entities[entity_id]

        # Get existing relations to avoid duplicates
        existing_targets = set()
        for rel in self.state.relations:
            if rel.from_entity == entity_id:
                existing_targets.add(rel.to_entity)
            elif rel.to_entity == entity_id:
                existing_targets.add(rel.from_entity)

        suggestions = []

        # 1. Semantic similarity via vector search
        if self._vector_index is not None:
            try:
                entity_text = f"{entity_obj.name} {' '.join(o.text for o in entity_obj.observations)}"
                search_results = self.vector_index.search(entity_text, limit=DEFAULT_VECTOR_SEARCH_LIMIT)

                for result_id, score in search_results:
                    if result_id == entity_id or result_id in existing_targets:
                        continue
                    if result_id not in self.state.entities:
                        continue

                    other = self.state.entities[result_id]
                    if score > SUGGEST_RELATION_CONFIDENCE_THRESHOLD:
                        rel_type = self._guess_relation_type(entity_obj, other)
                        suggestions.append({
                            "target": other.name,
                            "target_id": result_id,
                            "suggested_relation": rel_type,
                            "confidence": round(score, 2),
                            "reason": "semantic similarity",
                        })
            except Exception:
                pass

        # 2. Co-occurrence in observations (entity name mentioned in other's observations)
        entity_text_lower = " ".join(o.text.lower() for o in entity_obj.observations)

        for oid, other in self.state.entities.items():
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
