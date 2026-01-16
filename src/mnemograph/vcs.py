"""Git operations wrapper for memory versioning.

NOTE: Git integration for SQLite-based storage is limited. The database
is tracked as a binary file. Text-based diff/status operations that
depended on JSONL line-by-line analysis are deprecated.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from .events import EventStore
from .state import materialize


class MemoryVCS:
    """Version control operations for memory graph.

    With SQLite storage, git integration is simplified:
    - The database is tracked as a binary blob
    - Semantic diffs are computed from current state only
    - Historical state requires database snapshots or event replay
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.db_file = memory_dir / "mnemograph.db"
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

        # Create .gitignore - vectors are regenerated from events
        gitignore_content = """# Derived files (regenerate after checkout)
state.json
*.lock
mnemograph.log

# Python
__pycache__/
*.pyc
"""
        self.gitignore_file.write_text(gitignore_content)

        # Create empty database if not exists
        if not self.db_file.exists():
            # Initialize empty event store (creates db)
            event_store = EventStore(self.db_file)
            event_store.close()  # Checkpoint WAL and close connection

        # Initialize repo
        self._repo = Repo.init(self.memory_dir)
        self.repo.index.add([".gitignore", "mnemograph.db"])
        self.repo.index.commit("Initialize memory repository")

        return True

    def status(self) -> dict:
        """Get current status.

        With SQLite, we can't do line-by-line comparison with git.
        Instead, we check if the file has changed using git status.
        """
        # Check git status for database file
        is_dirty = bool(self.repo.is_dirty(path="mnemograph.db"))

        # Get current event count from database
        event_count = 0
        if self.db_file.exists():
            event_store = EventStore(self.db_file)
            event_count = event_store.count()
            event_store.close()

        return {
            "branch": self.repo.active_branch.name,
            "commit": self.repo.head.commit.hexsha[:7] if self.repo.head.is_valid() else None,
            "commit_message": self.repo.head.commit.message.strip() if self.repo.head.is_valid() else None,
            "event_count": event_count,
            "is_dirty": is_dirty,
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
            })
        return commits

    def show(self, ref: str = "HEAD") -> dict:
        """Get graph state at ref.

        .. deprecated:: 0.4.0
            Historical state not available with SQLite storage.
            Use time-travel tools (get_state_at) instead.
        """
        if ref != "HEAD" and ref != "working":
            warnings.warn(
                "vcs.show() for non-HEAD refs is deprecated. Use time-travel tools (get_state_at) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return {
                "ref": ref,
                "error": "Historical state not available with SQLite storage. Use time-travel tools instead.",
                "entities": {},
                "relations": [],
            }

        # Load current state
        event_store = EventStore(self.db_file)
        events = event_store.read_all()
        event_store.close()

        state = materialize(events)

        return {
            "ref": ref,
            "entities": {eid: e.model_dump(mode="json") for eid, e in state.entities.items()},
            "relations": [r.model_dump(mode="json") for r in state.relations],
        }

    def diff(self, ref_a: str = "HEAD", ref_b: Optional[str] = None) -> dict:
        """Compute semantic diff.

        .. deprecated:: 0.4.0
            Historical diff not available with SQLite storage.
            Use time-travel tools (get_state_at, diff_timerange) instead.
        """
        warnings.warn(
            "vcs.diff() is deprecated. Use time-travel tools (diff_timerange) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "from": ref_a,
            "to": ref_b or "working",
            "entities": {
                "added": {},
                "removed": {},
                "modified": {},
            },
            "relations": {
                "added": [],
                "removed": [],
            },
            "note": "Historical diff not available with SQLite storage. Use time-travel tools (get_state_at, diff_timerange) instead.",
        }

    def commit(self, message: str, auto_summary: bool = False) -> str:
        """Commit current database state."""
        # Check if there are changes to commit
        status = self.status()
        if not status["is_dirty"]:
            raise RuntimeError("Nothing to commit (mnemograph.db unchanged)")

        # Generate auto-summary if requested
        if auto_summary:
            event_store = EventStore(self.db_file)
            event_count = event_store.count()
            event_store.close()
            message = f"{message}\n\n{event_count} events in database"

        self.repo.index.add(["mnemograph.db"])
        commit = self.repo.index.commit(message)

        return commit.hexsha[:7]

    def branch(self, name: Optional[str] = None, delete: bool = False) -> list[str]:
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
        """Regenerate state.json from events."""
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
