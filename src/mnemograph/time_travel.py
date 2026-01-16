"""Time travel operations for event rewind and state restoration.

Provides methods to:
- Get graph state at any point in time
- Get events within a time range
- Diff states between two timestamps
- Get entity history
- Restore to past states with audit trail
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from .models import Entity, MemoryEvent
from .state import GraphState, materialize_at
from .timeutil import parse_time_reference

if TYPE_CHECKING:
    from .events import EventStore

logger = logging.getLogger(__name__)


class TimeTraveler:
    """Handles time-based operations on the event-sourced graph.

    All operations are read-only except restore_state_at which
    emits new events to recreate a past state.
    """

    def __init__(
        self,
        event_store_getter: Callable[[], "EventStore"],
        emit_event: Callable[[str, dict], MemoryEvent] | None = None,
    ):
        """Initialize time traveler.

        Args:
            event_store_getter: Callable that returns EventStore (lazy loading)
            emit_event: Callable to emit events (for restore operations)
        """
        self._get_event_store = event_store_getter
        self._emit_event = emit_event

    def _normalize_timestamp(self, ts: str | datetime) -> datetime:
        """Convert string or datetime to timezone-aware datetime."""
        if isinstance(ts, str):
            return parse_time_reference(ts)
        # Ensure timezone-aware (assume UTC if naive)
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    def state_at(self, timestamp: str | datetime) -> GraphState:
        """Get graph state at a specific point in time.

        Args:
            timestamp: ISO datetime string, relative reference ("7 days ago"),
                       or datetime object (will be treated as UTC if naive)

        Returns:
            GraphState as it existed at that timestamp
        """
        ts = self._normalize_timestamp(timestamp)
        events = self._get_event_store().read_all()
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
        start_ts = self._normalize_timestamp(start)
        end_ts = self._normalize_timestamp(end) if end else datetime.now(timezone.utc)

        events = self._get_event_store().read_all()
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
        start_ts = self._normalize_timestamp(start)
        end_ts = self._normalize_timestamp(end) if end else datetime.now(timezone.utc)

        events = self._get_event_store().read_all()
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

    def get_entity_history(
        self,
        entity_id: str,
    ) -> list[dict]:
        """Get the history of changes to an entity.

        Args:
            entity_id: The entity's ID (not name)

        Returns:
            List of events that affected this entity, oldest first.
        """
        events = self._get_event_store().read_all()

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

    def restore_state_at(
        self,
        timestamp: str,
        reason: str = "",
        state_getter: Callable[[], GraphState] | None = None,
    ) -> dict:
        """Restore graph to state at a specific timestamp.

        This method:
        1. Materializes state at that timestamp
        2. Emits clear_graph event
        3. Emits events to recreate that state

        Full audit trail preserved — events show the restore happened.

        Args:
            timestamp: ISO format or relative ("2 hours ago", "yesterday")
            reason: Why restoring (recorded in clear event)
            state_getter: Callable to get current state (for count after restore)

        Returns:
            Status with entity/relation counts after restore
        """
        if self._emit_event is None:
            return {
                "status": "error",
                "error": "Restore not available (no emit callback configured)",
            }

        # Parse timestamp
        ts = parse_time_reference(timestamp)

        # Get past state
        events = self._get_event_store().read_all()
        past_state = materialize_at(events, ts)

        if not past_state.entities:
            return {
                "status": "error",
                "error": f"No entities found at {timestamp}",
                "tip": "Use get_state_at() to preview state before restoring.",
            }

        # Clear current graph (recorded in events)
        clear_reason = f"Restoring to {timestamp}"
        if reason:
            clear_reason += f": {reason}"
        self._emit_event("clear_graph", {"reason": clear_reason})

        # Recreate entities from past state (recorded in events)
        for entity in past_state.entities.values():
            # Emit create_entity with original data
            entity_data = entity.model_dump(mode="json")
            self._emit_event("create_entity", entity_data)

        # Recreate relations from past state
        for relation in past_state.relations:
            relation_data = relation.model_dump(mode="json")
            self._emit_event("create_relation", relation_data)

        # Get current state counts
        entity_count = len(past_state.entities)
        relation_count = len(past_state.relations)
        if state_getter:
            current = state_getter()
            entity_count = len(current.entities)
            relation_count = len(current.relations)

        return {
            "status": "restored",
            "restored_to": timestamp,
            "resolved_timestamp": ts.isoformat(),
            "entities": entity_count,
            "relations": relation_count,
            "note": "Full audit trail preserved in events.",
        }
