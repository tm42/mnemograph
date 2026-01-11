"""Core data models for the memory engine.

Uses Pydantic v2 for validation, ULID for sortable unique IDs.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field
from ulid import ULID


def generate_id() -> str:
    """Generate a ULID (sortable, unique identifier)."""
    return str(ULID())


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class Observation(BaseModel):
    """A single observation/fact about an entity."""

    id: str = Field(default_factory=generate_id)
    text: str
    ts: datetime = Field(default_factory=utc_now)
    source: str  # session ID or "user"
    confidence: float | None = None  # 0-1, optional


EntityType = Literal[
    "concept",   # patterns, ideas, approaches
    "decision",  # choices made with rationale
    "project",   # codebases, systems
    "pattern",   # recurring code patterns
    "question",  # open questions, unknowns
    "learning",  # things discovered together
    "entity",    # generic: people, orgs, files, etc.
]


class Entity(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(default_factory=generate_id)
    name: str
    type: EntityType = "entity"
    observations: list[Observation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    created_by: str = ""  # session ID
    access_count: int = 0
    last_accessed: datetime | None = None

    def to_summary(self) -> dict:
        """Return a compact summary of this entity."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "observation_count": len(self.observations),
            "created_at": self.created_at.isoformat(),
        }


class Relation(BaseModel):
    """An edge in the knowledge graph with weighted connections.

    Weight is computed from three components:
    - recency_score: Decays over time from last access (computed, not stored)
    - co_access_score: Increases when both endpoints retrieved together (cached)
    - explicit_weight: Manually set importance (event-sourced)
    """

    id: str = Field(default_factory=generate_id)
    from_entity: str  # entity ID
    to_entity: str    # entity ID
    type: str         # verb phrase: "implements", "decided_for", etc.
    created_at: datetime = Field(default_factory=utc_now)
    created_by: str = ""  # session ID

    # Weight components
    explicit_weight: float = Field(default=0.5, ge=0.0, le=1.0)  # User/CC set
    co_access_score: float = Field(default=0.0, ge=0.0, le=1.0)  # Learned from usage
    access_count: int = 0  # Times this relation was traversed
    last_accessed: datetime = Field(default_factory=utc_now)

    @property
    def recency_score(self) -> float:
        """Compute recency score based on time since last access.

        Uses exponential decay with 30-day half-life.
        """
        import math
        from datetime import timezone

        now = datetime.now(timezone.utc)
        days_since = (now - self.last_accessed).total_seconds() / 86400
        half_life = 30.0
        decay_rate = 0.693 / half_life  # ln(2) / half_life
        return max(0.0, min(1.0, math.exp(-decay_rate * days_since)))

    @property
    def weight(self) -> float:
        """Combined weight for traversal priority.

        Formula: 0.4 * recency + 0.3 * co_access + 0.3 * explicit
        """
        return (
            0.4 * self.recency_score +
            0.3 * self.co_access_score +
            0.3 * self.explicit_weight
        )

    def to_summary(self) -> dict:
        """Return a compact summary of this relation."""
        return {
            "id": self.id,
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "weight": round(self.weight, 3),
        }

    def other_entity(self, entity_id: str) -> str:
        """Return the entity on the other end of this relation."""
        return self.to_entity if self.from_entity == entity_id else self.from_entity


EventOp = Literal[
    "create_entity",
    "update_entity",
    "delete_entity",
    "create_relation",
    "delete_relation",
    "add_observation",
    "delete_observation",
    "update_weight",  # Explicit weight change on a relation
    "clear_graph",  # Reset graph to empty state
    "compact",  # History compaction — clears state, followed by recreate events
]


class MemoryEvent(BaseModel):
    """An append-only event in the event log."""

    id: str = Field(default_factory=generate_id)
    ts: datetime = Field(default_factory=utc_now)
    op: EventOp
    session_id: str
    source: Literal["cc", "user"] = "cc"
    data: dict  # operation-specific payload
