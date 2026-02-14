"""Memory engine - orchestrates event store, state, and operations."""

import json
import logging
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from .events import EventStore
from .models import Entity, MemoryEvent, Observation, Relation
from .state import GraphState, materialize, apply_event
from .branches import BranchManager
from .similarity import SimilarityChecker
from .time_travel import TimeTraveler
from .constants import (
    DEFAULT_QUERY_LIMIT,
    DEFAULT_RECENT_LIMIT,
    DUPLICATE_AUTO_BLOCK_THRESHOLD,
    MAX_OBSERVATION_TOKENS,
)
from .query import QueryService

# --- Bootstrap Guide Data ---
# Seeded into empty global memory to teach agents how to use mnemograph
# Data stored in src/mnemograph/data/guide_seed.json

def _load_guide_seed() -> dict:
    """Load the mnemograph guide seed data from package data.

    Returns the guide seed dict with topic, entities, and relations.
    Falls back to minimal seed if JSON file not found.
    """
    import importlib.resources
    import json

    try:
        data_file = importlib.resources.files("mnemograph.data").joinpath("guide_seed.json")
        with data_file.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, ModuleNotFoundError):
        # Minimal fallback if package data not available
        return {
            "topic": {
                "name": "topic/mnemograph-guide",
                "type": "entity",
                "observations": ["Mnemograph guide (data file not found)"],
            },
            "entities": [],
            "relations": [],
        }


def _check_git_repo(memory_dir: Path) -> bool:
    """Check if memory directory is in a git repository.

    Returns True if in a git repo, False otherwise.
    Emits a warning if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=memory_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            warnings.warn(
                f"Memory directory {memory_dir} is not in a git repository. "
                "Git-based features (rewind, compact safety) will not work. "
                "Consider initializing a git repo with 'git init'.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True
    except FileNotFoundError:
        warnings.warn(
            "git command not found. Git-based features (rewind, compact safety) "
            "will not work.",
            UserWarning,
            stacklevel=3,
        )
        return False


def _get_git_root(memory_dir: Path) -> Path | None:
    """Get the root directory of the git repository containing memory_dir.

    Returns None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=memory_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        return None
    except FileNotFoundError:
        return None


def _validate_memory_repo(memory_dir: Path, git_root: Path | None) -> None:
    """Ensure git root is the memory directory, not a parent repo.

    This prevents accidentally running git commands in a parent repository
    when the memory directory is nested inside another git repo.

    Args:
        memory_dir: The memory directory path
        git_root: The detected git root (from git rev-parse --show-toplevel)

    Raises:
        ValueError: If git_root doesn't contain mnemograph.db, indicating
            the memory directory is nested in another git repository.
    """
    if not git_root:
        return

    db_path = git_root / "mnemograph.db"
    if not db_path.exists():
        raise ValueError(
            f"Git root {git_root} does not contain mnemograph.db. "
            f"Memory directory may be nested in another git repository. "
            f"Initialize a separate git repo in {memory_dir} or "
            f"move the memory directory outside the current repo."
        )

    # Optional: check for marker file
    marker = git_root / ".mnemograph"
    if not marker.exists():
        logger.warning(
            f"Memory repo missing .mnemograph marker. "
            f"Consider running: touch {marker}"
        )


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

        # Single SQLite database for events and vectors
        self._db_path = memory_dir / "mnemograph.db"
        self.event_store = EventStore(self._db_path)

        self.state: GraphState = self._load_state()
        self._vector_index = None  # Lazy-loaded, shares db connection
        self._co_access_cache_path = memory_dir / "co_access_cache.json"
        self._load_co_access_cache()

        # Similarity checker (uses lazy-loaded vector index)
        self._similarity: SimilarityChecker | None = None

        # Time traveler (uses event store)
        self._time_traveler: TimeTraveler | None = None

        # Branch manager - provides filtered views of the graph
        self._branch_manager: BranchManager | None = None

        # Check git repo status (warns if not in git)
        self._in_git_repo = _check_git_repo(memory_dir)
        self._git_root = _get_git_root(memory_dir) if self._in_git_repo else None

        # Validate that git root is the memory directory, not a parent repo
        _validate_memory_repo(memory_dir, self._git_root)

        # Query service — all read-only operations delegated here
        self._query = QueryService(
            get_state=lambda: self.state,
            get_active_state=self.get_active_state,
            get_vector_index=lambda: self.vector_index,
            similarity=lambda: self.similarity,
            time_traveler=lambda: self.time_traveler,
            resolve_entity=self._resolve_entity,
            on_co_access=self._handle_co_access,
        )

    def _handle_co_access(self) -> None:
        """Callback for QueryService to save co-access cache."""
        self.save_co_access_cache()

    @property
    def branch_manager(self) -> BranchManager:
        """Lazy-load branch manager."""
        if self._branch_manager is None:
            self._branch_manager = BranchManager(self.memory_dir, lambda: self.state)
        return self._branch_manager

    @property
    def similarity(self) -> SimilarityChecker:
        """Lazy-load similarity checker."""
        if self._similarity is None:
            # Pass lambda so vector index is only loaded when actually needed
            self._similarity = SimilarityChecker(lambda: self.vector_index)
        return self._similarity

    @property
    def time_traveler(self) -> TimeTraveler:
        """Lazy-load time traveler."""
        if self._time_traveler is None:
            self._time_traveler = TimeTraveler(
                lambda: self.event_store,
                self._emit,
            )
        return self._time_traveler

    def get_active_state(self) -> GraphState:
        """Get state filtered to current branch.

        On main branch, returns full state. On other branches, returns
        filtered state containing only branch entities/relations.
        """
        return self.branch_manager.get_filtered_state()

    def _load_state(self) -> GraphState:
        """Load state from events."""
        events = self.event_store.read_all()
        return materialize(events)

    def _load_co_access_cache(self) -> None:
        """Load cached co-access data into relations.

        Handles both old format (co_access_score) and new format
        (co_access_density + last_co_access). Old format is migrated
        via inverse sigmoid to approximate the equivalent density.
        """
        if not self._co_access_cache_path.exists():
            return

        try:
            import math
            from .constants import CO_ACCESS_SIGMOID_MIDPOINT, CO_ACCESS_SIGMOID_STEEPNESS

            cache = json.loads(self._co_access_cache_path.read_text())
            for rel in self.state.relations:
                if rel.id in cache:
                    entry = cache[rel.id]
                    if "co_access_density" in entry:
                        # New format
                        rel.co_access_density = entry["co_access_density"]
                        if entry.get("last_co_access"):
                            rel.last_co_access = datetime.fromisoformat(
                                entry["last_co_access"]
                            )
                    elif "co_access_score" in entry:
                        # Old format: convert score to density via inverse sigmoid
                        score = entry["co_access_score"]
                        if 0 < score < 1:
                            rel.co_access_density = (
                                CO_ACCESS_SIGMOID_MIDPOINT
                                - math.log(1.0 / score - 1.0) / CO_ACCESS_SIGMOID_STEEPNESS
                            )
                        elif score >= 1:
                            # Saturated — use a high density value
                            rel.co_access_density = CO_ACCESS_SIGMOID_MIDPOINT * 2
                        # score == 0 → density stays at default 0.0
        except (json.JSONDecodeError, ValueError):
            # Corrupted cache, ignore
            pass

    def save_co_access_cache(self) -> None:
        """Save co-access density data to cache file.

        Call this periodically or on shutdown to persist learned scores.
        """
        cache = {}
        for rel in self.state.relations:
            if rel.co_access_density > 0:
                entry: dict = {
                    "co_access_density": rel.co_access_density,
                }
                if rel.last_co_access is not None:
                    entry["last_co_access"] = rel.last_co_access.isoformat()
                cache[rel.id] = entry

        self._co_access_cache_path.write_text(json.dumps(cache, indent=2))

    def _seed_guide(self) -> dict:
        """Seed the mnemograph usage guide into memory.

        Called on first run when memory is empty. Creates a living guide
        that teaches agents how to use mnemograph effectively.

        Returns:
            Summary of what was seeded.
        """
        guide = _load_guide_seed()

        # Create topic entity
        topic_data = guide["topic"]
        topic_entity = Entity(
            name=topic_data["name"],
            type=topic_data["type"],
            observations=[
                Observation(text=obs, source=self.session_id)
                for obs in topic_data["observations"]
            ],
            created_by=self.session_id,
        )
        self._emit("create_entity", topic_entity.model_dump(mode="json"))

        # Create guide entities
        created_entities = [topic_data["name"]]
        for entity_data in guide["entities"]:
            entity = Entity(
                name=entity_data["name"],
                type=entity_data["type"],
                observations=[
                    Observation(text=obs, source=self.session_id)
                    for obs in entity_data["observations"]
                ],
                created_by=self.session_id,
            )
            self._emit("create_entity", entity.model_dump(mode="json"))
            created_entities.append(entity_data["name"])

        # Create relations
        created_relations = []
        for from_name, to_name, rel_type in guide["relations"]:
            from_id = self._resolve_entity(from_name)
            to_id = self._resolve_entity(to_name)
            if from_id and to_id:
                relation = Relation(
                    from_entity=from_id,
                    to_entity=to_id,
                    type=rel_type,
                    created_by=self.session_id,
                )
                self._emit("create_relation", relation.model_dump(mode="json"))
                created_relations.append(f"{from_name} --{rel_type}--> {to_name}")

        return {
            "seeded": True,
            "entities_created": len(created_entities),
            "relations_created": len(created_relations),
            "guide_topic": "topic/mnemograph-guide",
            "explore_with": "recall('mnemograph', depth='medium')",
        }

    def _emit(self, op: str, data: dict) -> MemoryEvent:
        """Append event to SQLite store and apply to in-memory state.

        This is the single mutation point — all graph changes flow through here.
        The event is durably stored in the event log (source of truth) and then
        applied to the in-memory materialized state for immediate availability.

        Uses apply_event() for O(1) updates instead of full replay.
        Special case: restore_to triggers full re-materialize since it
        rewrites history and can't be applied incrementally.
        """
        event = MemoryEvent(
            op=op,  # type: ignore (we know op is valid EventOp)
            session_id=self.session_id,
            source="cc",
            data=data,
        )
        self.event_store.append(event)

        # restore_to requires full re-materialize (can't apply incrementally)
        if op == "restore_to":
            self.state = self._load_state()
        else:
            # Incremental update - O(1) instead of O(events)
            apply_event(self.state, event)

        return event

    # --- Branch auto-include helper ---

    def _auto_include_in_branch(
        self,
        entity_ids: list[str] | None = None,
        relation_ids: list[str] | None = None,
    ) -> None:
        """Auto-include newly created entities/relations in current branch.

        On main branch, this is a no-op (main sees everything).
        On other branches, adds the IDs to the branch's filter sets.
        """
        current_branch = self.branch_manager.current_branch_name()
        if current_branch == "main":
            return  # Main sees everything, no filtering needed

        branch = self.branch_manager.current_branch()
        modified = False

        if entity_ids:
            for eid in entity_ids:
                if eid not in branch.entity_ids:
                    branch.entity_ids.add(eid)
                    modified = True

        if relation_ids:
            for rid in relation_ids:
                if rid not in branch.relation_ids:
                    branch.relation_ids.add(rid)
                    modified = True

        if modified:
            self.branch_manager._save_branch(branch)

    # --- Entity operations ---

    def create_entities(
        self,
        entities: list[dict],
        force: bool = False,
    ) -> list[Entity | dict]:
        """Create multiple entities with auto-duplicate checking.

        On non-main branches, newly created entities are automatically
        included in the current branch.

        Args:
            entities: List of entity dicts to create
            force: If True, bypass duplicate check (like create_entities_force)

        Returns:
            List of created Entity objects OR warning dicts for blocked entities.
            Warning dicts have status="duplicate_warning" and include similar entities.
        """
        results: list[Entity | dict] = []
        created_entity_ids: list[str] = []

        for entity_data in entities:
            name = entity_data["name"]

            # Auto-duplicate check (unless force=True)
            if not force:
                similar = self.find_similar(name, threshold=DUPLICATE_AUTO_BLOCK_THRESHOLD)
                if similar:
                    top = similar[0]
                    # Return warning instead of creating - entity NOT created
                    results.append({
                        "status": "duplicate_warning",
                        "name": name,
                        "warning": f"Similar entity exists: '{top['name']}' ({top['similarity']:.0%} match)",
                        "similar": similar,
                        "suggestion": f"Use add_observations('{top['name']}', ...) instead",
                        "override": "Use create_entities with force=True to create anyway",
                    })
                    continue

            # Build observations from input
            observations = [
                Observation(text=obs, source=self.session_id)
                for obs in entity_data.get("observations", [])
            ]

            entity = Entity(
                name=name,
                type=entity_data.get("entityType", entity_data.get("type", "entity")),
                observations=observations,
                created_by=self.session_id,
            )
            self._emit("create_entity", entity.model_dump(mode="json"))
            results.append(entity)
            created_entity_ids.append(entity.id)
            # Index new entity in vector store if index is loaded
            self.ensure_indexed(entity.id)

        # Auto-include in current branch (no-op on main)
        if created_entity_ids:
            self._auto_include_in_branch(entity_ids=created_entity_ids)

        return results

    def create_entities_force(self, entities: list[dict]) -> list[Entity]:
        """Create entities bypassing duplicate check.

        Use when you're certain the entity is distinct despite similar names.
        """
        results = self.create_entities(entities, force=True)
        # Filter to only Entity objects (no warnings when force=True)
        return [r for r in results if isinstance(r, Entity)]

    def delete_entities(self, names: list[str]) -> int:
        """Delete entities by name. Returns count of deleted."""
        deleted = 0
        deleted_ids: list[str] = []
        for name in names:
            entity_id = self._resolve_entity(name)
            if entity_id:
                self._emit("delete_entity", {"id": entity_id})
                deleted += 1
                deleted_ids.append(entity_id)
        # Remove deleted entities from vector index
        if self._vector_index is not None:
            for entity_id in deleted_ids:
                self._vector_index.remove_entity(entity_id)
        self.branch_manager.prune_stale_ids(self.state)
        return deleted

    # --- Relation operations ---

    def create_relations(self, relations: list[dict]) -> dict:
        """Create multiple relations with detailed result.

        On non-main branches, newly created relations are automatically
        included in the current branch.

        Returns dict with 'created' list, 'failed' list, and 'summary'.
        Failed items include reason for failure.
        """
        created = []
        failed = []
        created_relation_ids: list[str] = []

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
            created_relation_ids.append(relation.id)

        # Auto-include in current branch (no-op on main)
        if created_relation_ids:
            self._auto_include_in_branch(relation_ids=created_relation_ids)

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
        """Add observations to existing entities.

        Includes soft token limit warning when observations approach 2000 tokens.
        """
        results = []
        for obs_data in observations:
            entity_id = self._resolve_entity(obs_data["entityName"])
            if entity_id:
                entity = self.state.entities.get(entity_id)
                added = []

                # Estimate current tokens (~4 chars per token)
                current_tokens = sum(len(o.text) // 4 for o in entity.observations) if entity else 0
                new_tokens = sum(len(t) // 4 for t in obs_data.get("contents", []))

                for content in obs_data.get("contents", []):
                    obs = Observation(text=content, source=self.session_id)
                    self._emit(
                        "add_observation",
                        {"entity_id": entity_id, "observation": obs.model_dump(mode="json")},
                    )
                    added.append(content)

                result = {"entityName": obs_data["entityName"], "addedObservations": added}

                # Add warning if approaching token limit
                total_tokens = current_tokens + new_tokens
                if total_tokens > MAX_OBSERVATION_TOKENS:
                    result["warning"] = (
                        f"Entity approaching token limit ({total_tokens}/{MAX_OBSERVATION_TOKENS})"
                    )
                    result["suggestion"] = (
                        "Consider creating related entities or consolidating observations"
                    )

                results.append(result)
                # Re-index entity after adding observations (text changed)
                self.ensure_indexed(entity_id)
        return results

    def delete_observations(self, deletions: list[dict]) -> int:
        """Delete specific observations from entities."""
        deleted = 0
        entities_modified: set[str] = set()
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
                            entities_modified.add(entity_id)
                            break
        # Re-index modified entities (text changed)
        for entity_id in entities_modified:
            self.ensure_indexed(entity_id)
        return deleted

    # --- Query operations (delegated to QueryService) ---

    def read_graph(self) -> dict:
        return self._query.read_graph()

    def get_structure(self, entity_ids=None, include_neighbors=True) -> dict:
        return self._query.get_structure(entity_ids, include_neighbors)

    def search_structure(self, query, limit=20) -> dict:
        return self._query.search_structure(query, limit)

    def search_graph(self, query) -> dict:
        return self._query.search_graph(query)

    def open_nodes(self, names) -> dict:
        return self._query.open_nodes(names)

    # --- Helpers ---

    def _resolve_entity(self, name_or_id: str) -> str | None:
        """Find entity by name or ID. O(1) lookup using indices."""
        # Check if it's an ID
        if name_or_id in self.state.entities:
            return name_or_id
        # Check name index
        return self.state.get_entity_id_by_name(name_or_id)

    def get_recent_entities(self, limit=DEFAULT_QUERY_LIMIT):
        return self._query.get_recent_entities(limit)

    def get_entities_by_type(self, entity_type):
        return self._query.get_entities_by_type(entity_type)

    def get_entity_neighbors(self, entity_id, include_entity=True):
        return self._query.get_entity_neighbors(entity_id, include_entity)

    # --- Phase 3: Vector search ---

    @property
    def vector_index(self):
        """Lazy-load vector index using shared database connection."""
        if self._vector_index is None:
            from .vectors import VectorIndex
            # Use shared connection from event store
            self._vector_index = VectorIndex(conn=self.event_store.get_connection())
            # Index all existing entities
            self._vector_index.reindex_all(list(self.state.entities.values()))
        return self._vector_index

    def search_semantic(self, query, limit=DEFAULT_QUERY_LIMIT, type_filter=None):
        return self._query.search_semantic(query, limit, type_filter)

    def ensure_indexed(self, entity_id: str) -> None:
        """Ensure a specific entity is indexed (call after mutations)."""
        if self._vector_index is not None and entity_id in self.state.entities:
            self._vector_index.index_entity(self.state.entities[entity_id])

    def recall(self, depth="shallow", query=None, focus=None, max_tokens=None, format="prose"):  # noqa: A002
        return self._query.recall(depth, query, focus, max_tokens, format)

    # --- Time travel (delegated to QueryService) ---

    def state_at(self, timestamp):
        return self._query.state_at(timestamp)

    def events_between(self, start, end=None):
        return self._query.events_between(start, end)

    def diff_between(self, start, end=None):
        return self._query.diff_between(start, end)

    def get_entity_history(self, entity_name):
        return self._query.get_entity_history(entity_name)

    # --- Edge Weights ---

    def get_relation_weight(self, relation_id):
        return self._query.get_relation_weight(relation_id)

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

    def get_strongest_connections(self, entity_name, limit=DEFAULT_QUERY_LIMIT):
        return self._query.get_strongest_connections(entity_name, limit)

    def get_weak_relations(self, max_weight=None, limit=None):
        kwargs = {}
        if max_weight is not None:
            kwargs["max_weight"] = max_weight
        if limit is not None:
            kwargs["limit"] = limit
        return self._query.get_weak_relations(**kwargs)

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

        # Vector search status
        vector_status = None
        if self._vector_index:
            vector_status = {
                "status": self._vector_index.health.status.value,
                "error": self._vector_index.health.error,
                "model": self._vector_index.health.embedding_model,
                "dimension": self._vector_index.health.dimension,
            }

        return {
            "status": {
                "entity_count": len(self.state.entities),
                "relation_count": len(self.state.relations),
                "types": type_counts,
                "vector_search": vector_status,
            },
            "recent_activity": recent_summary,
            "tools": {
                "retrieval": [
                    "recall(depth, query, focus) - PRIMARY: Get relevant context (shallow/medium/deep). Use focus=['EntityName'] for full details.",
                ],
                "creation": [
                    "remember(name, type, observations, relations) - PRIMARY: Store knowledge atomically",
                    "create_entities(entities) - Batch entity creation (auto-blocks duplicates)",
                    "add_observations(observations) - Add facts to existing entities",
                    "create_relations(relations) - Link entities together",
                ],
                "history": [
                    "get_state_at(timestamp) - View graph at any point in time",
                    "diff_timerange(start, end) - See what changed",
                    "get_entity_history(name) - Full changelog for an entity",
                ],
                "maintenance": [
                    "find_similar(name) - Check for duplicates before creating",
                    "merge_entities(source, target) - Consolidate duplicates",
                    "get_graph_health() - Assess graph quality",
                ],
            },
            "quick_start": "Call recall(depth='shallow') for a summary, or remember() to store new knowledge.",
        }

    def session_start(self, project_hint: str | None = None) -> dict:
        """Signal session start and get initial context.

        Args:
            project_hint: Optional project name or path for context

        Returns:
            Initial context to prime the session with quick_start guide.
            If memory is empty, seeds the mnemograph usage guide and
            suggests visualization.
        """
        # Check if this is first run (empty memory)
        first_run = len(self.state.entities) == 0
        bootstrap_result = None

        if first_run:
            # Seed the mnemograph guide
            bootstrap_result = self._seed_guide()

        # Get shallow context for session priming
        context_result = self.recall(depth="shallow")

        # Find project entity if hint provided
        project_entity = None
        if project_hint:
            project_id = self._resolve_entity(project_hint)
            if project_id:
                project_entity = self.state.entities.get(project_id)

        quick_start = """DAILY USE:
  recall(depth, query)             → PRIMARY RETRIEVAL (shallow/medium/deep)
  open_nodes(['name1', 'name2'])   → get FULL data for specific entities
  remember(name, type, obs, rels)  → store new knowledge (entity + relations)
  add_observations(entity, [...])  → add facts to existing entity

WORKFLOW:
  1. recall(depth='medium', query='topic') → returns structure if large
  2. If structure-only, use open_nodes() to expand specific entities

BEFORE CREATING:
  find_similar(name)               → check for duplicates (>80% blocks create)

CLEANUP (run periodically):
  get_graph_health()               → orphans, duplicates, overloaded entities
  find_orphans()                   → entities with no relations
  merge_entities(src, target)      → consolidate duplicates

HISTORY (when needed):
  get_state_at(timestamp)          → view graph at point in time
  diff_timerange(start, end)       → what changed between times
  get_entity_history(entity)       → changelog for one entity

UNDO / RECOVERY:
  reload()                         → sync MCP server with disk (after git ops)
  rewind(steps=1)                  → quick undo via git
  restore_state_at(timestamp)      → restore with full audit trail

WEIGHTS (rarely):
  get_relation_weight(id)          → see weight breakdown
  set_relation_importance(id, 0-1) → boost/demote relation
  get_strongest_connections(entity) → most important links
  get_weak_relations()             → pruning candidates

NAMING CONVENTION:
  Use topic/ prefix for grouping: topic/auth, topic/testing, topic/api-design
  This makes related entities easy to find and visualize together.

IMPORTANT — MEMORY SCOPE:
  If this is a new project, ASK the user: "Should memory be project-local or global?"
  - Project-local (.claude/memory): repo-specific decisions, architecture, patterns
  - Global (~/.claude/memory): cross-project learnings, personal preferences
  Configure via MEMORY_PATH env var or --global CLI flag.

TIP: Start with recall(depth='shallow') to see what's known, then remember() or add_observations()."""

        result = {
            "session_id": self.session_id,
            "memory_summary": {
                "entity_count": len(self.state.entities),
                "relation_count": len(self.state.relations),
            },
            "context": context_result["content"],
            "project": project_entity.name if project_entity else None,
            "quick_start": quick_start,
        }

        # Include bootstrap info on first run
        if bootstrap_result:
            result["first_run"] = True
            result["bootstrap"] = bootstrap_result
            result["onboarding"] = {
                "message": (
                    "Welcome to mnemograph! A usage guide has been seeded into memory. "
                    "Explore it with: recall('mnemograph', depth='medium')"
                ),
                "next_steps": [
                    "Run 'mnemograph graph' in terminal to visualize the knowledge graph",
                    "Ask user: 'Should this project use local or global memory?'",
                    "Start adding project-specific knowledge with remember()",
                ],
                "visualization": "mnemograph graph  # Opens interactive D3.js viewer in browser",
            }

        return result

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
                # Attach to most recently updated project
                project_entities.sort(key=lambda e: e.updated_at, reverse=True)
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

    def clear_graph(
        self, reason: str | None = None, confirm_token: str | None = None
    ) -> dict:
        """Clear all entities and relations from the graph.

        This is event-sourced — you can rewind to before the clear using
        get_state_at() with a timestamp before the clear event.

        Note: Branches are NOT cleared — use branch_manager.prune_stale_ids()
        (called automatically) to clean up stale branch references.

        Args:
            reason: Reason for clearing (required for graphs with >10 entities)
            confirm_token: Confirmation token for large graphs (format: CLEAR_<count>)

        Returns:
            Confirmation with entity/relation counts cleared, or error if blocked
        """
        entity_count = len(self.state.entities)

        # Require reason for non-trivial graphs
        if entity_count > 10 and not reason:
            return {
                "error": "Reason required for clearing graph with >10 entities",
                "entity_count": entity_count,
                "hint": "Pass reason='your reason here'",
            }

        # Require confirmation token for large graphs
        expected_token = f"CLEAR_{entity_count}"
        if entity_count > 10 and confirm_token != expected_token:
            return {
                "error": "Large graph requires confirmation token",
                "entity_count": entity_count,
                "confirm_with": expected_token,
                "hint": f"Pass confirm_token='{expected_token}' to proceed",
            }

        counts = {
            "entities_cleared": entity_count,
            "relations_cleared": len(self.state.relations),
        }

        self._emit("clear_graph", {"reason": reason or ""})
        self.branch_manager.prune_stale_ids(self.state)

        return {
            "status": "cleared",
            **counts,
            "reason": reason if reason else None,
            "tip": "Use get_state_at(timestamp) to view graph before clear, or check 'mnemograph log' for history",
            "undo_options": {
                "quick": "rewind() — uses git, fast, audit trail in git only",
                "audit": "restore_state_at('5 minutes ago') — preserves full audit trail in events",
            },
        }

    # --- Recovery Operations ---

    def reload(self) -> dict:
        """Reload graph state from mnemograph.db on disk.

        Use this after:
        - Git operations (checkout, restore, revert)
        - External edits to mnemograph.db
        - Any time MCP server seems out of sync with disk

        Returns:
            Current state after reload with entity/relation counts
        """
        events = self.event_store.read_all()
        self.state = materialize(events)
        self._load_co_access_cache()

        # Invalidate vector index to force reindex from new state
        if self._vector_index is not None:
            self._vector_index.invalidate()
        self._vector_index = None

        return {
            "status": "reloaded",
            "entities": len(self.state.entities),
            "relations": len(self.state.relations),
            "events_processed": len(events),
        }

    def rewind(self, steps: int = 1, to_commit: str | None = None) -> dict:
        """Rewind graph to a previous state using git.

        This restores mnemograph.db from git history and reloads.
        Fast, but audit trail is only in git (not in events).

        For audit-preserving restore, use restore_state_at() instead.

        Args:
            steps: Go back N commits that touched mnemograph.db (default: 1)
            to_commit: Or specify exact commit hash

        Returns:
            Status with entity/relation counts after rewind

        Raises:
            RuntimeError: If not in a git repository or git operation fails
        """
        if not self._in_git_repo:
            return {
                "status": "error",
                "error": "Not in a git repository. Cannot use git-based rewind.",
                "tip": "Use restore_state_at() for event-based restore instead.",
            }

        events_path = self.event_store.db_path
        relative_path = events_path.relative_to(self._git_root) if self._git_root else events_path

        if to_commit:
            commit = to_commit
        else:
            # Find the Nth commit that touched the database
            result = subprocess.run(
                ["git", "log", "--oneline", "--follow", "-n", str(steps + 1), "--", str(relative_path)],
                cwd=self._git_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return {
                    "status": "error",
                    "error": f"Failed to get git history: {result.stderr}",
                }

            commits = result.stdout.strip().split("\n")
            if len(commits) <= steps:
                return {
                    "status": "error",
                    "error": f"Not enough history: only {len(commits)} commits found, requested {steps} steps back",
                }
            commit = commits[steps].split()[0]  # Get commit hash from "abc123 commit msg"

        # Git checkout the file at that commit
        result = subprocess.run(
            ["git", "checkout", commit, "--", str(relative_path)],
            cwd=self._git_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Git restore failed: {result.stderr}",
            }

        # Reload state from restored file
        reload_result = self.reload()

        return {
            "status": "rewound",
            "restored_to_commit": commit,
            "entities": reload_result["entities"],
            "relations": reload_result["relations"],
            "note": "Audit trail in git only. Use restore_state_at() for event-based restore.",
        }

    def restore_state_at(self, timestamp: str, reason: str = "") -> dict:
        """Restore graph to state at a specific timestamp.

        Delegates to TimeTraveler. Emits events to recreate past state
        with full audit trail preserved.

        Note: Only event-sourced data is restored. Branches, vector embeddings,
        and co-access caches are NOT restored by this operation.
        """
        result = self.time_traveler.restore_state_at(
            timestamp, reason, lambda: self.state
        )
        self.branch_manager.prune_stale_ids(self.state)
        return result

    # --- Graph Coherence Tools (read-only delegated to QueryService) ---

    def find_similar(self, name, threshold=None):
        kwargs = {"name": name}
        if threshold is not None:
            kwargs["threshold"] = threshold
        return self._query.find_similar(**kwargs)

    def find_orphans(self):
        return self._query.find_orphans()

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

    def get_graph_health(self, full=False):
        return self._query.get_graph_health(full)

    def _count_connected_components(self):
        return self._query._count_connected_components()

    def suggest_relations(self, entity, limit=None):
        kwargs = {"entity": entity}
        if limit is not None:
            kwargs["limit"] = limit
        return self._query.suggest_relations(**kwargs)

    # --- High-Level Knowledge Creation ---

    def remember(
        self,
        name: str,
        entity_type: str,
        observations: list[str] | None = None,
        relations: list[dict] | None = None,
        force: bool = False,
    ) -> dict:
        """Store knowledge atomically — entity + observations + relations in one call.

        This is the primary tool for storing new knowledge. It creates an entity
        with observations and relations together, preventing orphan entities.

        On non-main branches, newly created entities and relations are
        automatically included in the current branch.

        Args:
            name: Entity name
            entity_type: One of: concept, decision, project, pattern, question, learning, entity
            observations: List of facts/observations about this entity
            relations: List of relations FROM this entity: [{"to": "Target", "type": "uses"}, ...]
            force: If True, bypass duplicate check

        Returns:
            On success: {"status": "created", "entity": {...}, "relations_created": N}
            On duplicate warning: {"status": "duplicate_warning", "similar": [...], ...}

        Example:
            remember(
                name="FastAPI",
                entity_type="concept",
                observations=["Async Python web framework", "Uses Pydantic for validation"],
                relations=[
                    {"to": "Python", "type": "uses"},
                    {"to": "Pydantic", "type": "depends_on"}
                ]
            )
        """
        observations = observations or []
        relations = relations or []

        # Auto-duplicate check (unless force=True)
        if not force:
            similar = self.find_similar(name, threshold=DUPLICATE_AUTO_BLOCK_THRESHOLD)
            if similar:
                top = similar[0]
                return {
                    "status": "duplicate_warning",
                    "name": name,
                    "warning": f"Similar entity exists: '{top['name']}' ({top['similarity']:.0%} match)",
                    "similar": similar,
                    "suggestion": f"Use add_observations('{top['name']}', ...) to add facts to existing entity",
                    "override": "Use remember(..., force=True) to create anyway",
                }

        # Create the entity
        entity = Entity(
            name=name,
            type=entity_type,
            observations=[
                Observation(text=obs, source=self.session_id)
                for obs in observations
            ],
            created_by=self.session_id,
        )
        self._emit("create_entity", entity.model_dump(mode="json"))

        # Index in vector store if available
        self.ensure_indexed(entity.id)

        # Create relations
        relations_created = 0
        relations_failed = []
        created_relation_ids: list[str] = []

        for rel in relations:
            to_name = rel.get("to")
            rel_type = rel.get("type", "related_to")

            if not to_name:
                relations_failed.append({"error": "Missing 'to' field", "relation": rel})
                continue

            to_id = self._resolve_entity(to_name)
            if not to_id:
                relations_failed.append({
                    "error": f"Target entity not found: {to_name}",
                    "relation": rel,
                    "suggestion": f"Create '{to_name}' first, or use create_relations later",
                })
                continue

            relation = Relation(
                from_entity=entity.id,
                to_entity=to_id,
                type=rel_type,
                created_by=self.session_id,
            )
            self._emit("create_relation", relation.model_dump(mode="json"))
            relations_created += 1
            created_relation_ids.append(relation.id)

        # Auto-include in current branch (no-op on main)
        self._auto_include_in_branch(
            entity_ids=[entity.id],
            relation_ids=created_relation_ids if created_relation_ids else None,
        )

        result = {
            "status": "created",
            "entity": entity.model_dump(mode="json"),
            "relations_created": relations_created,
        }

        if relations_failed:
            result["relations_failed"] = relations_failed

        return result
