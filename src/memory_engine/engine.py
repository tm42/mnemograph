"""Memory engine - orchestrates event store, state, and operations."""

from pathlib import Path

from .events import EventStore
from .models import Entity, MemoryEvent, Observation, Relation
from .state import GraphState, materialize


class MemoryEngine:
    """Main entry point for memory operations."""

    def __init__(self, memory_dir: Path, session_id: str):
        self.memory_dir = memory_dir
        self.session_id = session_id
        self.event_store = EventStore(memory_dir / "events.jsonl")
        self.state: GraphState = self._load_state()

    def _load_state(self) -> GraphState:
        """Load state from events."""
        events = self.event_store.read_all()
        return materialize(events)

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
        """Get specific entities by name and their relations."""
        entities = []
        entity_ids = set()

        for name in names:
            entity_id = self._resolve_entity(name)
            if entity_id and entity_id in self.state.entities:
                entities.append(self.state.entities[entity_id])
                entity_ids.add(entity_id)

        # Include relations involving these entities
        relations = [
            r for r in self.state.relations
            if r.from_entity in entity_ids or r.to_entity in entity_ids
        ]

        return {
            "entities": [e.model_dump(mode="json") for e in entities],
            "relations": [r.model_dump(mode="json") for r in relations],
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
