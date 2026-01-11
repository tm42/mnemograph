"""Memory engine - orchestrates event store, state, and operations."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .events import EventStore
from .models import Entity, MemoryEvent, Observation, Relation
from .state import GraphState, materialize, materialize_at
from .timeutil import parse_time_reference


class MemoryEngine:
    """Main entry point for memory operations."""

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
        """Emit an event and update local state."""
        event = MemoryEvent(
            op=op,  # type: ignore (we know op is valid EventOp)
            session_id=self.session_id,
            source="cc",
            data=data,
        )
        self.event_store.append(event)
        # Re-materialize (could optimize with incremental update later)
        self.state = materialize(self.event_store.read_all())
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

    def create_relations(self, relations: list[dict]) -> list[Relation]:
        """Create multiple relations."""
        created = []
        for rel_data in relations:
            from_id = self._resolve_entity(rel_data["from"])
            to_id = self._resolve_entity(rel_data["to"])
            if from_id and to_id:
                relation = Relation(
                    from_entity=from_id,
                    to_entity=to_id,
                    type=rel_data["relationType"],
                    created_by=self.session_id,
                )
                self._emit("create_relation", relation.model_dump(mode="json"))
                created.append(relation)
        return created

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
        """Find entity by name or ID."""
        if name_or_id in self.state.entities:
            return name_or_id
        for entity in self.state.entities.values():
            if entity.name == name_or_id:
                return entity.id
        return None

    def _track_access(self, entity_ids: list[str]) -> None:
        """Update access counts for retrieved entities (in-memory only, not persisted)."""
        now = datetime.now(timezone.utc)
        for entity_id in entity_ids:
            if entity_id in self.state.entities:
                entity = self.state.entities[entity_id]
                entity.access_count += 1
                entity.last_accessed = now

    # --- Phase 2: Richer queries ---

    def get_recent_entities(self, limit: int = 10) -> list[Entity]:
        """Get most recently updated entities."""
        return sorted(
            self.state.entities.values(),
            key=lambda e: e.updated_at,
            reverse=True,
        )[:limit]

    def get_hot_entities(self, limit: int = 10) -> list[Entity]:
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
        limit: int = 10,
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
            tokens = max_tokens or 500
            result = get_shallow_context(self.state, tokens)

        elif depth == "medium":
            tokens = max_tokens or 2000
            # Get vector search results if query provided
            vector_results = None
            if query:
                vector_results = self.vector_index.search(query, limit=10)
            result = get_medium_context(self.state, vector_results, focus, tokens)
            # Save co-access learning
            self.save_co_access_cache()

        elif depth == "deep":
            tokens = max_tokens or 5000
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
                       or datetime object

        Returns:
            GraphState as it existed at that timestamp
        """
        if isinstance(timestamp, str):
            ts = parse_time_reference(timestamp)
        else:
            ts = timestamp

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
        limit: int = 10,
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
        max_weight: float = 0.1,
        limit: int = 20,
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
