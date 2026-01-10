"""Git operations wrapper for memory versioning."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from .events import EventStore
from .models import MemoryEvent
from .state import materialize


class MemoryVCS:
    """Version control operations for memory graph."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.events_file = memory_dir / "events.jsonl"
        self.state_file = memory_dir / "state.json"
        self.gitignore_file = memory_dir / ".gitignore"
        self._repo: Optional[Repo] = None

    @property
    def repo(self) -> Repo:
        """Lazy-load git repo."""
        if self._repo is None:
            try:
                self._repo = Repo(self.memory_dir)
            except InvalidGitRepositoryError:
                raise RuntimeError(
                    f"Not a git repository: {self.memory_dir}\n"
                    "Run 'claude-mem init' first."
                )
        return self._repo

    def init(self) -> bool:
        """Initialize git repo if not exists."""
        if (self.memory_dir / ".git").exists():
            return False  # Already initialized

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Create .gitignore
        gitignore_content = """# Derived files (regenerate after checkout)
state.json
vectors.db
*.lock
graphmem.log

# Python
__pycache__/
*.pyc
"""
        self.gitignore_file.write_text(gitignore_content)

        # Create empty events file if not exists
        if not self.events_file.exists():
            self.events_file.touch()

        # Initialize repo
        self._repo = Repo.init(self.memory_dir)
        self.repo.index.add([".gitignore", "events.jsonl"])
        self.repo.index.commit("Initialize memory repository")

        return True

    def status(self) -> dict:
        """Get current status."""
        # Get committed version of events.jsonl
        try:
            committed_content = self.repo.git.show("HEAD:events.jsonl")
        except GitCommandError:
            committed_content = ""

        # Get current version
        current_content = ""
        if self.events_file.exists():
            current_content = self.events_file.read_text()

        # Compare to determine if dirty (strip to handle trailing newline differences)
        events_modified = committed_content.strip() != current_content.strip()

        # Count uncommitted events
        uncommitted_events = []
        if events_modified:
            committed_lines = set(committed_content.strip().split("\n")) if committed_content.strip() else set()
            current_lines = set(current_content.strip().split("\n")) if current_content.strip() else set()

            # New events = current - committed
            new_lines = current_lines - committed_lines
            for line in new_lines:
                if line:
                    try:
                        uncommitted_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        return {
            "branch": self.repo.active_branch.name,
            "commit": self.repo.head.commit.hexsha[:7] if self.repo.head.is_valid() else None,
            "commit_message": self.repo.head.commit.message.strip() if self.repo.head.is_valid() else None,
            "uncommitted_events": uncommitted_events,
            "is_dirty": events_modified,
        }

    def log(self, n: int = 10) -> list[dict]:
        """Get commit history."""
        commits = []
        for commit in self.repo.iter_commits(max_count=n):
            commits.append({
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:7],
                "message": commit.message.strip(),
                "author": str(commit.author),
                "date": datetime.fromtimestamp(commit.committed_date),
                "stats": self._get_commit_stats(commit),
            })
        return commits

    def _get_commit_stats(self, commit) -> dict:
        """Extract graph stats from commit."""
        try:
            events_content = self.repo.git.show(f"{commit.hexsha}:events.jsonl")
            events = [json.loads(line) for line in events_content.strip().split("\n") if line]

            entity_ids = set()
            relation_count = 0

            for event in events:
                op = event.get("op", "")
                if op == "create_entity":
                    entity_ids.add(event["data"]["id"])
                elif op == "delete_entity":
                    entity_ids.discard(event["data"].get("id"))
                elif op == "create_relation":
                    relation_count += 1
                elif op == "delete_relation":
                    relation_count -= 1

            return {"entities": len(entity_ids), "relations": max(0, relation_count)}
        except GitCommandError:
            return {"entities": 0, "relations": 0}

    def show(self, ref: str = "HEAD") -> dict:
        """Get graph state at ref."""
        try:
            events_content = self.repo.git.show(f"{ref}:events.jsonl")
        except GitCommandError:
            events_content = ""

        events = []
        for line in events_content.strip().split("\n"):
            if line:
                events.append(json.loads(line))

        typed_events = [MemoryEvent.model_validate(e) for e in events]
        state = materialize(typed_events)

        return {
            "ref": ref,
            "entities": {eid: e.model_dump(mode="json") for eid, e in state.entities.items()},
            "relations": [r.model_dump(mode="json") for r in state.relations],
        }

    def diff(self, ref_a: str = "HEAD", ref_b: Optional[str] = None) -> dict:
        """Compute semantic diff between two refs."""
        state_a = self.show(ref_a)

        if ref_b is None:
            # Compare with working directory
            event_store = EventStore(self.events_file)
            events = event_store.read_all()
            state = materialize(events)
            state_b = {
                "ref": "working",
                "entities": {eid: e.model_dump(mode="json") for eid, e in state.entities.items()},
                "relations": [r.model_dump(mode="json") for r in state.relations],
            }
        else:
            state_b = self.show(ref_b)

        # Compute entity diff
        entities_a = set(state_a["entities"].keys())
        entities_b = set(state_b["entities"].keys())

        added = entities_b - entities_a
        removed = entities_a - entities_b
        maybe_modified = entities_a & entities_b

        modified = set()
        for eid in maybe_modified:
            if state_a["entities"][eid] != state_b["entities"][eid]:
                modified.add(eid)

        # Compute relation diff (by tuple identity)
        def rel_key(r):
            return (r["from_entity"], r["to_entity"], r["type"])

        rels_a = {rel_key(r): r for r in state_a["relations"]}
        rels_b = {rel_key(r): r for r in state_b["relations"]}

        rels_added = set(rels_b.keys()) - set(rels_a.keys())
        rels_removed = set(rels_a.keys()) - set(rels_b.keys())

        return {
            "from": ref_a,
            "to": ref_b or "working",
            "entities": {
                "added": {eid: state_b["entities"][eid] for eid in added},
                "removed": {eid: state_a["entities"][eid] for eid in removed},
                "modified": {
                    eid: {
                        "before": state_a["entities"][eid],
                        "after": state_b["entities"][eid],
                    }
                    for eid in modified
                },
            },
            "relations": {
                "added": [rels_b[k] for k in rels_added],
                "removed": [rels_a[k] for k in rels_removed],
            },
        }

    def commit(self, message: str, auto_summary: bool = False) -> str:
        """Commit current events."""
        # Check if there are changes to commit
        status = self.status()
        if not status["is_dirty"]:
            raise RuntimeError("Nothing to commit (events.jsonl unchanged)")

        # Generate auto-summary if requested
        if auto_summary:
            diff = self.diff("HEAD", None)
            summary_parts = []
            if diff["entities"]["added"]:
                names = [e["name"] for e in diff["entities"]["added"].values()]
                summary_parts.append(f"+{len(names)} entities: {', '.join(names[:3])}")
            if diff["entities"]["removed"]:
                summary_parts.append(f"-{len(diff['entities']['removed'])} entities")
            if diff["relations"]["added"]:
                summary_parts.append(f"+{len(diff['relations']['added'])} relations")

            if summary_parts:
                message = f"{message}\n\n{'; '.join(summary_parts)}"

        self.repo.index.add(["events.jsonl"])
        commit = self.repo.index.commit(message)

        return commit.hexsha[:7]

    def branch(self, name: str = None, delete: bool = False) -> list[str]:
        """List, create, or delete branches."""
        if name is None:
            return [b.name for b in self.repo.branches]
        elif delete:
            self.repo.delete_head(name)
            return [b.name for b in self.repo.branches]
        else:
            self.repo.create_head(name)
            return [b.name for b in self.repo.branches]

    def checkout(self, ref: str, create: bool = False) -> None:
        """Checkout branch or commit."""
        if create:
            new_branch = self.repo.create_head(ref)
            new_branch.checkout()
        else:
            self.repo.git.checkout(ref)

        self._regenerate_derived()

    def _regenerate_derived(self) -> None:
        """Regenerate state.json and vectors.db from events."""
        from .engine import MemoryEngine

        # Re-initialize engine (will rebuild state and vector index)
        engine = MemoryEngine(self.memory_dir, session_id="vcs")

        # Save state snapshot
        state_dict = {
            "entities": {eid: e.model_dump(mode="json") for eid, e in engine.state.entities.items()},
            "relations": [r.model_dump(mode="json") for r in engine.state.relations],
            "last_event_id": engine.state.last_event_id,
        }
        self.state_file.write_text(json.dumps(state_dict, indent=2, default=str))

    def merge(self, branch: str, abort: bool = False) -> dict:
        """Merge branch into current."""
        if abort:
            self.repo.git.merge("--abort")
            return {"status": "aborted"}

        try:
            self.repo.git.merge(branch)
            self._regenerate_derived()
            return {"status": "success", "merged": branch}
        except GitCommandError as e:
            if "CONFLICT" in str(e):
                return {"status": "conflict", "message": str(e)}
            raise
