"""State materialization from events.

Replays the event log to build the current graph state.
Supports point-in-time queries via materialize_at().
"""

from dataclasses import dataclass, field
from datetime import datetime

from .models import Entity, Observation, Relation, MemoryEvent


@dataclass
class GraphState:
    """Materialized state of the knowledge graph.

    Includes indices for O(1) lookups:
    - _name_to_id: entity name -> entity ID
    - _outgoing: entity ID -> list of relations from it
    - _incoming: entity ID -> list of relations to it
    - _connected_entities: set of entity IDs with any relation
    - _relations_by_id: relation ID -> Relation object
    """

    entities: dict[str, Entity] = field(default_factory=dict)  # id -> Entity
    relations: list[Relation] = field(default_factory=list)
    last_event_id: str | None = None

    # Indices for O(1) lookups
    _name_to_id: dict[str, str] = field(default_factory=dict)
    _outgoing: dict[str, list[Relation]] = field(default_factory=dict)
    _incoming: dict[str, list[Relation]] = field(default_factory=dict)
    _connected_entities: set[str] = field(default_factory=set)
    _relations_by_id: dict[str, Relation] = field(default_factory=dict)

    def get_entity_id_by_name(self, name: str) -> str | None:
        """O(1) lookup of entity ID by name."""
        return self._name_to_id.get(name)

    def get_outgoing_relations(self, entity_id: str) -> list[Relation]:
        """O(1) lookup of relations where entity is source."""
        return self._outgoing.get(entity_id, [])

    def get_incoming_relations(self, entity_id: str) -> list[Relation]:
        """O(1) lookup of relations where entity is target."""
        return self._incoming.get(entity_id, [])

    def is_orphan(self, entity_id: str) -> bool:
        """O(1) check if entity has no relations."""
        return entity_id not in self._connected_entities

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        """Get all relations involving an entity (as source or target)."""
        return self.get_outgoing_relations(entity_id) + self.get_incoming_relations(entity_id)

    def get_relation_by_id(self, relation_id: str) -> Relation | None:
        """O(1) lookup of relation by ID."""
        return self._relations_by_id.get(relation_id)

    def _rebuild_indices(self) -> None:
        """Rebuild all indices from entities and relations.

        Called after full materialization or when indices may be stale.
        """
        # Rebuild name index
        self._name_to_id = {e.name: eid for eid, e in self.entities.items()}

        # Rebuild relation indices
        self._outgoing = {}
        self._incoming = {}
        self._connected_entities = set()
        self._relations_by_id = {}

        for rel in self.relations:
            self._outgoing.setdefault(rel.from_entity, []).append(rel)
            self._incoming.setdefault(rel.to_entity, []).append(rel)
            self._connected_entities.add(rel.from_entity)
            self._connected_entities.add(rel.to_entity)
            self._relations_by_id[rel.id] = rel

    def check_index_consistency(self) -> list[str]:
        """Validate that indices match base state. Returns list of errors.

        This is a debug/test utility to detect index drift after incremental updates.
        An empty list means indices are consistent.
        """
        errors: list[str] = []

        # 1. Check _name_to_id matches entities
        expected_name_to_id = {e.name: eid for eid, e in self.entities.items()}
        if self._name_to_id != expected_name_to_id:
            missing_in_index = set(expected_name_to_id.keys()) - set(self._name_to_id.keys())
            extra_in_index = set(self._name_to_id.keys()) - set(expected_name_to_id.keys())
            wrong_mapping = {
                name for name in self._name_to_id
                if name in expected_name_to_id and self._name_to_id[name] != expected_name_to_id[name]
            }
            if missing_in_index:
                errors.append(f"_name_to_id missing entities: {missing_in_index}")
            if extra_in_index:
                errors.append(f"_name_to_id has stale entries: {extra_in_index}")
            if wrong_mapping:
                errors.append(f"_name_to_id has wrong mappings: {wrong_mapping}")

        # 2. Check _outgoing matches relations
        expected_outgoing: dict[str, list[Relation]] = {}
        for rel in self.relations:
            expected_outgoing.setdefault(rel.from_entity, []).append(rel)

        for entity_id in set(expected_outgoing.keys()) | set(self._outgoing.keys()):
            expected = set(id(r) for r in expected_outgoing.get(entity_id, []))
            actual = set(id(r) for r in self._outgoing.get(entity_id, []))
            if expected != actual:
                errors.append(
                    f"_outgoing[{entity_id}] mismatch: expected {len(expected_outgoing.get(entity_id, []))} "
                    f"relations, got {len(self._outgoing.get(entity_id, []))}"
                )

        # 3. Check _incoming matches relations
        expected_incoming: dict[str, list[Relation]] = {}
        for rel in self.relations:
            expected_incoming.setdefault(rel.to_entity, []).append(rel)

        for entity_id in set(expected_incoming.keys()) | set(self._incoming.keys()):
            expected = set(id(r) for r in expected_incoming.get(entity_id, []))
            actual = set(id(r) for r in self._incoming.get(entity_id, []))
            if expected != actual:
                errors.append(
                    f"_incoming[{entity_id}] mismatch: expected {len(expected_incoming.get(entity_id, []))} "
                    f"relations, got {len(self._incoming.get(entity_id, []))}"
                )

        # 4. Check _connected_entities matches relation endpoints
        expected_connected = set()
        for rel in self.relations:
            expected_connected.add(rel.from_entity)
            expected_connected.add(rel.to_entity)

        if self._connected_entities != expected_connected:
            missing = expected_connected - self._connected_entities
            extra = self._connected_entities - expected_connected
            if missing:
                errors.append(f"_connected_entities missing: {missing}")
            if extra:
                errors.append(f"_connected_entities has stale entries: {extra}")

        # 5. Check _relations_by_id matches relations list
        expected_by_id = {r.id: r for r in self.relations}
        if set(self._relations_by_id.keys()) != set(expected_by_id.keys()):
            missing = set(expected_by_id.keys()) - set(self._relations_by_id.keys())
            extra = set(self._relations_by_id.keys()) - set(expected_by_id.keys())
            if missing:
                errors.append(f"_relations_by_id missing: {missing}")
            if extra:
                errors.append(f"_relations_by_id has stale entries: {extra}")
        else:
            # Check same object references
            for rid, rel in expected_by_id.items():
                if self._relations_by_id.get(rid) is not rel:
                    errors.append(f"_relations_by_id[{rid}] points to different object")

        return errors


def materialize(events: list[MemoryEvent]) -> GraphState:
    """Replay events to build current state."""
    state = GraphState()

    for event in events:
        apply_event(state, event)

    # Rebuild indices after full materialization
    state._rebuild_indices()

    return state


def apply_event(state: GraphState, event: MemoryEvent) -> None:
    """Apply a single event to state (mutates in place).

    Used for incremental updates after initial materialization.
    Updates indices automatically.

    Args:
        state: The graph state to update
        event: The event to apply
    """
    op = event.op
    data = event.data

    if op == "create_entity":
        entity = Entity.model_validate(data)
        state.entities[entity.id] = entity
        # Update name index
        state._name_to_id[entity.name] = entity.id

    elif op == "update_entity":
        entity_id = data.get("id")
        if entity_id and entity_id in state.entities:
            entity = state.entities[entity_id]
            old_name = entity.name
            updates = data.get("updates", {})
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            entity.updated_at = event.ts
            # Update name index if name changed
            if "name" in updates and updates["name"] != old_name:
                state._name_to_id.pop(old_name, None)
                state._name_to_id[entity.name] = entity_id

    elif op == "delete_entity":
        entity_id = data.get("id")
        if entity_id and entity_id in state.entities:
            # Remove from name index
            name = state.entities[entity_id].name
            state._name_to_id.pop(name, None)
            state.entities.pop(entity_id, None)

            # Cascade delete relations and update indices
            relations_to_remove = [
                r for r in state.relations
                if r.from_entity == entity_id or r.to_entity == entity_id
            ]
            for rel in relations_to_remove:
                _remove_relation_from_indices(state, rel)
            state.relations = [
                r for r in state.relations
                if r.from_entity != entity_id and r.to_entity != entity_id
            ]

    elif op == "create_relation":
        relation = Relation.model_validate(data)
        state.relations.append(relation)
        # Update relation indices
        state._outgoing.setdefault(relation.from_entity, []).append(relation)
        state._incoming.setdefault(relation.to_entity, []).append(relation)
        state._connected_entities.add(relation.from_entity)
        state._connected_entities.add(relation.to_entity)
        state._relations_by_id[relation.id] = relation

    elif op == "delete_relation":
        # Find and remove matching relation
        to_remove = None
        for r in state.relations:
            if (r.from_entity == data.get("from_entity")
                    and r.to_entity == data.get("to_entity")
                    and r.type == data.get("type")):
                to_remove = r
                break

        if to_remove:
            _remove_relation_from_indices(state, to_remove)
            state.relations = [r for r in state.relations if r is not to_remove]

    elif op == "add_observation":
        entity_id = data.get("entity_id")
        if entity_id and entity_id in state.entities:
            obs_data = data.get("observation", {})
            obs = Observation.model_validate(obs_data)
            state.entities[entity_id].observations.append(obs)
            state.entities[entity_id].updated_at = event.ts

    elif op == "delete_observation":
        entity_id = data.get("entity_id")
        obs_id = data.get("observation_id")
        if entity_id and entity_id in state.entities and obs_id:
            entity = state.entities[entity_id]
            entity.observations = [o for o in entity.observations if o.id != obs_id]
            entity.updated_at = event.ts

    elif op == "update_weight":
        relation_id = data.get("relation_id")
        weight_type = data.get("weight_type", "explicit")
        new_value = data.get("new_value")
        if relation_id is not None and new_value is not None:
            for rel in state.relations:
                if rel.id == relation_id:
                    if weight_type == "explicit":
                        rel.explicit_weight = new_value
                    break

    elif op == "clear_graph":
        # Reset to empty state — clears all entities and relations
        # Note: We clear the data structures but keep the same GraphState object
        state.entities.clear()
        state.relations.clear()
        state._name_to_id.clear()
        state._outgoing.clear()
        state._incoming.clear()
        state._connected_entities.clear()
        state._relations_by_id.clear()

    elif op == "compact":
        # Compact is a marker event — actual state changes come from
        # subsequent create_entity/create_relation events.
        # We just clear current state like clear_graph.
        state.entities.clear()
        state.relations.clear()
        state._name_to_id.clear()
        state._outgoing.clear()
        state._incoming.clear()
        state._connected_entities.clear()
        state._relations_by_id.clear()

    state.last_event_id = event.id


def _remove_relation_from_indices(state: GraphState, rel: Relation) -> None:
    """Remove a relation from the indices.

    Also updates _connected_entities if entity no longer has any relations.
    """
    # Remove from _relations_by_id
    state._relations_by_id.pop(rel.id, None)

    # Remove from outgoing
    if rel.from_entity in state._outgoing:
        state._outgoing[rel.from_entity] = [
            r for r in state._outgoing[rel.from_entity] if r is not rel
        ]
        # Clean up empty list
        if not state._outgoing[rel.from_entity]:
            del state._outgoing[rel.from_entity]

    # Remove from incoming
    if rel.to_entity in state._incoming:
        state._incoming[rel.to_entity] = [
            r for r in state._incoming[rel.to_entity] if r is not rel
        ]
        # Clean up empty list
        if not state._incoming[rel.to_entity]:
            del state._incoming[rel.to_entity]

    # Update connected entities
    if rel.from_entity not in state._outgoing and rel.from_entity not in state._incoming:
        state._connected_entities.discard(rel.from_entity)
    if rel.to_entity not in state._outgoing and rel.to_entity not in state._incoming:
        state._connected_entities.discard(rel.to_entity)


def materialize_at(events: list[MemoryEvent], timestamp: datetime) -> GraphState:
    """Materialize graph state at a specific point in time.

    Filters events to those that occurred at or before the timestamp,
    then materializes. This is the core of event rewind functionality.

    Args:
        events: All events from the event store
        timestamp: Point in time to materialize (timezone-aware)

    Returns:
        Graph state as it existed at that timestamp
    """
    # Filter events to those at or before timestamp
    filtered = [e for e in events if e.ts <= timestamp]
    return materialize(filtered)
