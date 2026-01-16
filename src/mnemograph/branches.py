"""Branch management for filtered knowledge graph views.

Branches are filters over the shared event log, not forks.
The 'main' branch sees all events; other branches see filtered subsets.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .models import Branch, make_main_branch, validate_branch_name

if TYPE_CHECKING:
    from .state import GraphState

logger = logging.getLogger(__name__)


class BranchManager:
    """Manages branch lifecycle and filtering.

    Branches are stored as JSON files in .claude/memory/branches/
    The current branch is tracked in .claude/memory/branches/_current
    """

    def __init__(self, memory_dir: Path, get_state_fn: Callable[[], "GraphState"]):
        """Initialize the branch manager.

        Args:
            memory_dir: Path to .claude/memory
            get_state_fn: Function that returns current GraphState
        """
        self.memory_dir = memory_dir
        self.branches_dir = memory_dir / "branches"
        self.branches_dir.mkdir(parents=True, exist_ok=True)
        self._current_file = self.branches_dir / "_current"
        self._get_state = get_state_fn

    # ─────────────────────────────────────────────────────────────────────────
    # Branch CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def exists(self, name: str) -> bool:
        """Check if a branch exists."""
        if name == "main":
            return True
        return self._branch_path(name).exists()

    def get(self, name: str) -> Branch:
        """Load a branch by name.

        Args:
            name: Branch name (e.g., "project/auth-service" or "main")

        Returns:
            Branch object

        Raises:
            ValueError: If branch not found
        """
        if name == "main":
            return make_main_branch()

        path = self._branch_path(name)
        if not path.exists():
            raise ValueError(f"Branch '{name}' not found")

        data = json.loads(path.read_text())
        return Branch.from_dict(data)

    def list(self, include_archived: bool = False) -> list[Branch]:
        """List all branches.

        Args:
            include_archived: Include branches in archive/ prefix

        Returns:
            List of Branch objects, main first
        """
        branches = [make_main_branch()]

        # Walk through all branch files
        for path in self.branches_dir.rglob("*.json"):
            try:
                branch = Branch.from_dict(json.loads(path.read_text()))

                if not include_archived and branch.name.startswith("archive/"):
                    continue
                if not branch.is_active and not include_archived:
                    continue

                branches.append(branch)
            except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
                logger.warning(f"Skipping malformed branch file {path.name}: {e}")
                continue

        return branches

    def list_by_type(self, branch_type: str) -> list[Branch]:
        """List branches of a specific type."""
        return [
            b
            for b in self.list(include_archived=(branch_type == "archive"))
            if b.name.startswith(f"{branch_type}/")
        ]

    def delete(self, name: str) -> None:
        """Delete a branch permanently.

        Args:
            name: Branch name to delete

        Raises:
            ValueError: If trying to delete main or current branch
        """
        if name == "main":
            raise ValueError("Cannot delete main branch")

        if name == self.current_branch_name():
            raise ValueError(
                f"Cannot delete current branch. Checkout a different branch first."
            )

        path = self._branch_path(name)
        if path.exists():
            path.unlink()

            # Clean up empty parent directories
            parent = path.parent
            if parent != self.branches_dir and parent.exists():
                try:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                except OSError:
                    pass  # Directory not empty or other issue

    def _save_branch(self, branch: Branch) -> None:
        """Save branch to disk."""
        path = self._branch_path(branch.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(branch.to_dict(), indent=2))

    def _branch_path(self, name: str) -> Path:
        """Get filesystem path for a branch."""
        # name like "project/auth-service" -> branches/project/auth-service.json
        return self.branches_dir / f"{name}.json"

    # ─────────────────────────────────────────────────────────────────────────
    # Current Branch
    # ─────────────────────────────────────────────────────────────────────────

    def current_branch_name(self) -> str:
        """Get name of currently checked-out branch."""
        if self._current_file.exists():
            return self._current_file.read_text().strip()
        return "main"

    def current_branch(self) -> Branch:
        """Get currently checked-out branch object."""
        return self.get(self.current_branch_name())

    def checkout(self, name: str) -> Branch:
        """Switch to a different branch.

        Args:
            name: Branch name to checkout

        Returns:
            Checked-out branch

        Raises:
            ValueError: If branch not found or is archived
        """
        branch = self.get(name)  # Validates existence

        if not branch.is_active and name != "main":
            raise ValueError(
                f"Branch '{name}' is archived. Use 'branch unarchive' to restore it first."
            )

        self._current_file.write_text(name)
        return branch

    # ─────────────────────────────────────────────────────────────────────────
    # Branch Creation
    # ─────────────────────────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        seed_entities: list[str],
        description: str = "",
        depth: int = 2,
    ) -> Branch:
        """Create a new branch from seed entities.

        Automatically expands to include N-hop neighbors of seeds.

        Args:
            name: Branch name (should follow naming conventions)
            seed_entities: Entity IDs or names to seed the branch
            description: Human-readable description
            depth: How many hops to expand from seeds

        Returns:
            Created Branch object

        Raises:
            ValueError: If branch exists or name invalid
        """
        # Validate name
        valid, message = validate_branch_name(name)
        if not valid:
            raise ValueError(message)

        if self.exists(name):
            raise ValueError(f"Branch '{name}' already exists")

        state = self._get_state()

        # Resolve seed names to IDs if needed
        seed_ids = self._resolve_seeds(seed_entities, state)

        if not seed_ids:
            raise ValueError("No valid seed entities found")

        # Expand from seeds to N-hop subgraph
        entity_ids, relation_ids = self._expand_subgraph(
            seeds=seed_ids, state=state, depth=depth
        )

        branch = Branch(
            name=name,
            description=description,
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            parent=self.current_branch_name(),
            created_from_commit=self._get_current_commit(),
            auto_include_depth=depth,
        )

        self._save_branch(branch)
        return branch

    def _resolve_seeds(
        self, seeds: list[str], state: "GraphState"
    ) -> set[str]:
        """Resolve seed references to entity IDs.

        Seeds can be entity IDs or entity names.
        Uses O(1) name index lookup instead of O(n) entity scan.
        """
        resolved = set()

        for seed in seeds:
            # Check if it's an entity ID
            if seed in state.entities:
                resolved.add(seed)
                continue

            # Check if it's an entity name using O(1) index
            eid = state.get_entity_id_by_name(seed)
            if eid:
                resolved.add(eid)

        return resolved

    def _expand_subgraph(
        self, seeds: set[str], state: "GraphState", depth: int
    ) -> tuple[set[str], set[str]]:
        """Expand from seed entities to include N-hop neighbors.

        Uses BFS to find all entities within `depth` hops of any seed.
        Also includes all relations between included entities.
        Uses O(k) relation index lookups instead of O(n) full scans.

        Returns:
            (entity_ids, relation_ids) for the subgraph
        """
        entity_ids = set()

        # BFS expansion
        frontier = seeds.copy()

        for hop in range(depth + 1):
            next_frontier = set()

            for eid in frontier:
                if eid not in state.entities:
                    continue

                entity_ids.add(eid)

                if hop < depth:  # Don't expand on last hop
                    # Get neighbors using O(k) index lookup
                    for rel in state.get_relations_for(eid):
                        # Determine neighbor (the other end of the relation)
                        neighbor = rel.to_entity if rel.from_entity == eid else rel.from_entity

                        if neighbor not in entity_ids:
                            next_frontier.add(neighbor)

            frontier = next_frontier

        # Find all relations between included entities
        relation_ids = set()
        for rel in state.relations:
            if rel.from_entity in entity_ids and rel.to_entity in entity_ids:
                relation_ids.add(rel.id)

        return entity_ids, relation_ids

    def _get_current_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.memory_dir,
            )
            return result.stdout.strip()[:7] if result.returncode == 0 else None
        except (OSError, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug(f"Could not get current commit (not in git repo?): {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Branch Modification
    # ─────────────────────────────────────────────────────────────────────────

    def add_entities(
        self, branch_name: str, entity_ids: list[str], include_relations: bool = True
    ) -> Branch:
        """Add entities to a branch.

        Args:
            branch_name: Branch to modify
            entity_ids: Entity IDs to add
            include_relations: Also add relations between new and existing entities

        Returns:
            Updated branch
        """
        if branch_name == "main":
            raise ValueError("Cannot modify main branch (it includes everything)")

        branch = self.get(branch_name)
        state = self._get_state()

        for eid in entity_ids:
            if eid not in state.entities:
                continue  # Skip invalid entity IDs
            branch.entity_ids.add(eid)

        if include_relations:
            # Add relations connecting new entities to existing branch entities
            for rel in state.relations:
                if (
                    rel.from_entity in branch.entity_ids
                    and rel.to_entity in branch.entity_ids
                ):
                    branch.relation_ids.add(rel.id)

        self._save_branch(branch)
        return branch

    def remove_entities(
        self, branch_name: str, entity_ids: list[str], cascade_relations: bool = True
    ) -> Branch:
        """Remove entities from a branch.

        Args:
            branch_name: Branch to modify
            entity_ids: Entity IDs to remove
            cascade_relations: Also remove orphaned relations

        Returns:
            Updated branch
        """
        if branch_name == "main":
            raise ValueError("Cannot modify main branch")

        branch = self.get(branch_name)

        for eid in entity_ids:
            branch.entity_ids.discard(eid)

        if cascade_relations:
            state = self._get_state()
            to_remove = set()

            for rid in branch.relation_ids:
                # Find the relation using O(1) index lookup
                rel = state.get_relation_by_id(rid)

                if rel is None:
                    to_remove.add(rid)
                    continue

                # Remove if either endpoint is no longer on branch
                if (
                    rel.from_entity not in branch.entity_ids
                    or rel.to_entity not in branch.entity_ids
                ):
                    to_remove.add(rid)

            branch.relation_ids -= to_remove

        self._save_branch(branch)
        return branch

    # ─────────────────────────────────────────────────────────────────────────
    # Archive Operations
    # ─────────────────────────────────────────────────────────────────────────

    def archive(self, name: str) -> Branch:
        """Archive a branch (move to archive/ prefix).

        Archived branches are hidden from default list and read-only.

        Args:
            name: Branch name to archive

        Returns:
            Archived branch with new name
        """
        from datetime import datetime, timezone

        if name == "main":
            raise ValueError("Cannot archive main branch")

        if name.startswith("archive/"):
            raise ValueError("Branch is already archived")

        branch = self.get(name)

        # Generate archive name with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m")
        base_name = name.split("/")[-1]  # Get name without type prefix
        archive_name = f"archive/{base_name}-{timestamp}"

        # Handle name collision
        counter = 1
        final_name = archive_name
        while self.exists(final_name):
            final_name = f"{archive_name}-{counter}"
            counter += 1

        # Delete old branch file
        old_path = self._branch_path(name)
        if old_path.exists():
            old_path.unlink()

        # If we archived the current branch, switch to main
        if self.current_branch_name() == name:
            self._current_file.write_text("main")

        # Create archived version
        branch.name = final_name
        branch.is_active = False
        self._save_branch(branch)

        return branch

    def unarchive(self, name: str, new_name: str | None = None) -> Branch:
        """Restore an archived branch.

        Args:
            name: Archived branch name (e.g., "archive/auth-service-2025-01")
            new_name: Optional new name (default: project/<base-name>)

        Returns:
            Restored branch
        """
        if not name.startswith("archive/"):
            raise ValueError("Branch is not archived")

        branch = self.get(name)

        # Generate new name if not provided
        if new_name is None:
            # Remove "archive/" prefix and timestamp suffix
            base = name.replace("archive/", "")
            # Try to remove timestamp suffix like "-2025-01" or "-2025-01-1"
            import re

            base = re.sub(r"-\d{4}-\d{2}(-\d+)?$", "", base)
            new_name = f"project/{base}"

        valid, message = validate_branch_name(new_name)
        if not valid:
            raise ValueError(message)

        if self.exists(new_name):
            raise ValueError(f"Branch '{new_name}' already exists")

        # Delete archived version
        old_path = self._branch_path(name)
        if old_path.exists():
            old_path.unlink()

        # Create restored version
        branch.name = new_name
        branch.is_active = True
        self._save_branch(branch)

        return branch

    # ─────────────────────────────────────────────────────────────────────────
    # Merge & Diff Operations
    # ─────────────────────────────────────────────────────────────────────────

    def merge(self, source_branch: str, target_branch: str = "main") -> Branch:
        """Merge source branch's filter into target.

        For merging into main: Nothing to do (main sees everything).
        For other targets: Expand target's filter to include source's entities.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (default: main)

        Returns:
            Updated target branch
        """
        source = self.get(source_branch)

        if target_branch == "main":
            # Main already sees all events, nothing to merge
            return make_main_branch()

        target = self.get(target_branch)
        target.entity_ids.update(source.entity_ids)
        target.relation_ids.update(source.relation_ids)

        self._save_branch(target)
        return target

    def cherry_pick(
        self, source_branch: str, entity_refs: list[str], target_branch: str | None = None
    ) -> Branch:
        """Copy specific entities from source to target branch.

        Args:
            source_branch: Branch to copy from
            entity_refs: Entity IDs or names to copy
            target_branch: Branch to copy to (default: current branch)

        Returns:
            Updated target branch
        """
        target_name = target_branch or self.current_branch_name()

        if target_name == "main":
            raise ValueError("Cannot cherry-pick into main (it includes everything)")

        source = self.get(source_branch)
        state = self._get_state()

        # Resolve entity references
        entity_ids = []
        for ref in entity_refs:
            if ref in source.entity_ids:
                entity_ids.append(ref)
            else:
                # Try to find by name
                for eid in source.entity_ids:
                    entity = state.entities.get(eid)
                    if entity and entity.name.lower() == ref.lower():
                        entity_ids.append(eid)
                        break

        return self.add_entities(target_name, entity_ids)

    def diff(self, branch_a: str, branch_b: str | None = None) -> dict:
        """Compare two branches.

        Args:
            branch_a: First branch
            branch_b: Second branch (default: current branch)

        Returns:
            {
                "only_in_a": {"entities": [...], "relations": [...]},
                "only_in_b": {"entities": [...], "relations": [...]},
                "in_both": {"entities": [...], "relations": [...]}
            }
        """
        b_name = branch_b or self.current_branch_name()

        a = self.get(branch_a)
        b = self.get(b_name)

        state = self._get_state()

        # Handle main specially - it includes everything
        a_entities = (
            a.entity_ids if a.name != "main" else set(state.entities.keys())
        )
        b_entities = (
            b.entity_ids if b.name != "main" else set(state.entities.keys())
        )
        a_relations = (
            a.relation_ids if a.name != "main" else {r.id for r in state.relations}
        )
        b_relations = (
            b.relation_ids if b.name != "main" else {r.id for r in state.relations}
        )

        return {
            "only_in_a": {
                "entities": sorted(a_entities - b_entities),
                "relations": sorted(a_relations - b_relations),
            },
            "only_in_b": {
                "entities": sorted(b_entities - a_entities),
                "relations": sorted(b_relations - a_relations),
            },
            "in_both": {
                "entities": sorted(a_entities & b_entities),
                "relations": sorted(a_relations & b_relations),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Filtered State
    # ─────────────────────────────────────────────────────────────────────────

    def get_filtered_state(self, branch_name: str | None = None) -> "GraphState":
        """Get graph state filtered to a branch's entities.

        Args:
            branch_name: Branch to filter to (default: current branch)

        Returns:
            GraphState containing only branch entities/relations
        """
        from .state import GraphState

        branch = self.get(branch_name) if branch_name else self.current_branch()
        state = self._get_state()

        if branch.name == "main":
            return state  # Main sees everything

        # Filter entities
        filtered_entities = {
            eid: entity
            for eid, entity in state.entities.items()
            if eid in branch.entity_ids
        }

        # Filter relations
        filtered_relations = [
            rel for rel in state.relations if rel.id in branch.relation_ids
        ]

        return GraphState(
            entities=filtered_entities,
            relations=filtered_relations,
            last_event_id=state.last_event_id,
        )
