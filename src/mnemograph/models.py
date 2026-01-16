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
        from .weights import compute_recency_score

        return compute_recency_score(self.last_accessed)

    @property
    def weight(self) -> float:
        """Combined weight for traversal priority.

        Formula: 0.4 * recency + 0.3 * co_access + 0.3 * explicit
        """
        from .constants import (
            WEIGHT_RECENCY_COEFF,
            WEIGHT_CO_ACCESS_COEFF,
            WEIGHT_EXPLICIT_COEFF,
        )

        return (
            WEIGHT_RECENCY_COEFF * self.recency_score +
            WEIGHT_CO_ACCESS_COEFF * self.co_access_score +
            WEIGHT_EXPLICIT_COEFF * self.explicit_weight
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


# ─────────────────────────────────────────────────────────────────────────────
# Branch System
# ─────────────────────────────────────────────────────────────────────────────

import re

BRANCH_TYPES = ("project", "feature", "domain", "spike", "archive")
BRANCH_NAME_PATTERN = re.compile(
    r"^(project|feature|domain|spike|archive)/[a-z0-9][a-z0-9-]*[a-z0-9]$"
)


def validate_branch_name(name: str, strict: bool = False) -> tuple[bool, str | None]:
    """Validate branch name against naming conventions.

    Args:
        name: Branch name to validate
        strict: If True, require type prefix

    Returns:
        (is_valid, message) — message is error description or suggestion
    """
    # Reserved names
    if name in ("main", "HEAD", ""):
        return False, f"'{name}' is a reserved name"

    # Check for invalid characters
    if not re.match(r"^[a-z0-9/-]+$", name):
        return False, "Use only lowercase letters, numbers, hyphens, and one slash"

    # Has type prefix
    if "/" in name:
        if not BRANCH_NAME_PATTERN.match(name):
            types = "|".join(BRANCH_TYPES)
            return False, f"Invalid format. Use: <type>/<name> where type is {types}"
        return True, None

    # No type prefix
    if strict:
        return False, f"Missing type prefix. Suggested: project/{name}"
    else:
        # Warn but allow
        return True, f"Consider using a type prefix: project/{name}"


class Branch(BaseModel):
    """A filtered view of the knowledge graph.

    Branches are NOT forks of events — they're filters over the shared event log.
    The 'main' branch sees all events; other branches see filtered subsets.
    """

    name: str  # e.g., "project/auth-service"
    description: str = ""

    # Filter: which entities/relations belong to this branch
    entity_ids: set[str] = Field(default_factory=set)
    relation_ids: set[str] = Field(default_factory=set)

    # Lineage
    parent: str | None = "main"  # Parent branch name (None for main)
    created_at: datetime = Field(default_factory=utc_now)
    created_from_commit: str | None = None  # Git commit hash when created

    # State
    is_active: bool = True  # False = archived

    # Auto-include settings
    auto_include_depth: int = 1  # When entity added, include N-hop neighbors

    def includes_entity(self, entity_id: str) -> bool:
        """Check if entity is visible on this branch."""
        if self.name == "main":
            return True  # Main sees everything
        return entity_id in self.entity_ids

    def includes_relation(self, relation_id: str) -> bool:
        """Check if relation is visible on this branch."""
        if self.name == "main":
            return True  # Main sees everything
        return relation_id in self.relation_ids

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "name": self.name,
            "description": self.description,
            "entity_ids": sorted(self.entity_ids),  # Sorted for deterministic output
            "relation_ids": sorted(self.relation_ids),
            "parent": self.parent,
            "created_at": self.created_at.isoformat(),
            "created_from_commit": self.created_from_commit,
            "is_active": self.is_active,
            "auto_include_depth": self.auto_include_depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Branch":
        """Deserialize from JSON."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = utc_now()

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            entity_ids=set(data.get("entity_ids", [])),
            relation_ids=set(data.get("relation_ids", [])),
            parent=data.get("parent", "main"),
            created_at=created_at,
            created_from_commit=data.get("created_from_commit"),
            is_active=data.get("is_active", True),
            auto_include_depth=data.get("auto_include_depth", 1),
        )


def make_main_branch() -> Branch:
    """Create the implicit main branch that sees everything."""
    return Branch(
        name="main",
        description="All entities and relations",
        entity_ids=set(),  # Empty = all (special case for main)
        relation_ids=set(),
        parent=None,
        is_active=True,
    )
