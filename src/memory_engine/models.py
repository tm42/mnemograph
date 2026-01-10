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


class Relation(BaseModel):
    """An edge in the knowledge graph."""

    id: str = Field(default_factory=generate_id)
    from_entity: str  # entity ID
    to_entity: str    # entity ID
    type: str         # verb phrase: "implements", "decided_for", etc.
    created_at: datetime = Field(default_factory=utc_now)
    created_by: str = ""  # session ID


EventOp = Literal[
    "create_entity",
    "update_entity",
    "delete_entity",
    "create_relation",
    "delete_relation",
    "add_observation",
    "delete_observation",
]


class MemoryEvent(BaseModel):
    """An append-only event in the event log."""

    id: str = Field(default_factory=generate_id)
    ts: datetime = Field(default_factory=utc_now)
    op: EventOp
    session_id: str
    source: Literal["cc", "user"] = "cc"
    data: dict  # operation-specific payload
