"""State materialization from events.

Replays the event log to build the current graph state.
Supports point-in-time queries via materialize_at().
"""

from dataclasses import dataclass, field
from datetime import datetime

from .models import Entity, Observation, Relation, MemoryEvent


@dataclass
class GraphState:
    """Materialized state of the knowledge graph."""

    entities: dict[str, Entity] = field(default_factory=dict)  # id -> Entity
    relations: list[Relation] = field(default_factory=list)
    last_event_id: str | None = None

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        """Get all relations involving an entity (as source or target)."""
        return [
            r for r in self.relations
            if r.from_entity == entity_id or r.to_entity == entity_id
        ]

    def get_relation_by_id(self, relation_id: str) -> Relation | None:
        """Get a relation by its ID."""
        for r in self.relations:
            if r.id == relation_id:
                return r
        return None


def materialize(events: list[MemoryEvent]) -> GraphState:
    """Replay events to build current state."""
    state = GraphState()

    for event in events:
        op = event.op
        data = event.data

        if op == "create_entity":
            entity = Entity.model_validate(data)
            state.entities[entity.id] = entity

        elif op == "update_entity":
            entity_id = data.get("id")
            if entity_id and entity_id in state.entities:
                entity = state.entities[entity_id]
                updates = data.get("updates", {})
                for key, value in updates.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)
                entity.updated_at = event.ts

        elif op == "delete_entity":
            entity_id = data.get("id")
            if entity_id:
                state.entities.pop(entity_id, None)
                # Cascade delete relations
                state.relations = [
                    r for r in state.relations
                    if r.from_entity != entity_id and r.to_entity != entity_id
                ]

        elif op == "create_relation":
            relation = Relation.model_validate(data)
            state.relations.append(relation)

        elif op == "delete_relation":
            state.relations = [
                r for r in state.relations
                if not (
                    r.from_entity == data.get("from_entity")
                    and r.to_entity == data.get("to_entity")
                    and r.type == data.get("type")
                )
            ]

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

        state.last_event_id = event.id

    return state


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
