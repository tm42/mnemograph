"""Mnemograph client for Claude Code hooks.

Wraps the mnemograph engine for use in CC hooks. Handles:
- Memory path detection (project-local vs global)
- Graceful fallback when mnemograph unavailable
- Timeout enforcement
"""

from __future__ import annotations

import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class TimeoutError(Exception):
    """Operation timed out."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timeout enforcement."""
    def handler(_signum, _frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _is_valid_memory_path(path: Path) -> bool:
    """Check if a memory path is valid and usable.

    A path is valid if it exists and contains mnemograph.db (v0.4.0+) or events.jsonl (legacy).
    We don't try to init the engine here since the hook may run without mnemograph installed.
    """
    if not path.exists():
        return False
    # Check for v0.4.0+ SQLite format or legacy JSONL
    return (path / "mnemograph.db").exists() or (path / "events.jsonl").exists()


def detect_memory_path(cwd: str | Path | None = None) -> Path | None:
    """Detect memory path from cwd or environment.

    Priority:
    1. Project-local: ./.claude/memory
    2. MEMORY_PATH env var
    3. Global: ~/.claude/memory

    Returns None if no valid memory directory found.
    Each candidate is validated to ensure the engine can actually load it.
    """
    cwd_path = Path(cwd) if cwd else Path.cwd()

    # 1. Project-local
    local_path = cwd_path / ".claude" / "memory"
    if _is_valid_memory_path(local_path):
        return local_path

    # 2. Environment variable
    env_path = os.environ.get("MEMORY_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if _is_valid_memory_path(p):
            return p

    # 3. Global fallback
    global_path = Path.home() / ".claude" / "memory"
    if _is_valid_memory_path(global_path):
        return global_path

    return None


def detect_project_name(cwd: str | Path | None = None) -> str:
    """Detect project name from cwd."""
    cwd_path = Path(cwd) if cwd else Path.cwd()
    return cwd_path.name


class MnemographClient:
    """Client wrapper for mnemograph engine."""

    def __init__(
        self,
        memory_path: Path | None = None,
        session_id: str = "hook",
        cwd: str | Path | None = None,
    ):
        self.memory_path = memory_path or detect_memory_path(cwd)
        self.session_id = session_id
        self._engine = None

    @property
    def available(self) -> bool:
        """Check if mnemograph is available."""
        if not self.memory_path:
            return False
        try:
            from mnemograph.engine import MemoryEngine  # noqa
            return True
        except ImportError:
            return False

    def _get_engine(self):
        """Lazy-load the engine."""
        if self._engine is None:
            if not self.available:
                raise RuntimeError("Mnemograph not available")
            from mnemograph.engine import MemoryEngine
            self._engine = MemoryEngine(self.memory_path, self.session_id)
        return self._engine

    def recall(
        self,
        depth: str = "shallow",
        query: str | None = None,
        focus: list[str] | None = None,
        max_tokens: int | None = None,
        format: str = "prose",
        timeout_seconds: int = 10,
    ) -> dict[str, Any] | None:
        """Retrieve context from memory.

        Returns None on timeout or error (fail silently).
        """
        if not self.available:
            return None

        try:
            with timeout(timeout_seconds):
                engine = self._get_engine()
                return engine.recall(
                    depth=depth,
                    query=query,
                    focus=focus,
                    max_tokens=max_tokens,
                    format=format,
                )
        except (TimeoutError, Exception):
            return None

    def remember(
        self,
        name: str,
        entity_type: str = "learning",
        observations: list[str] | None = None,
        relations: list[dict] | None = None,
        timeout_seconds: int = 5,
    ) -> bool:
        """Store knowledge in memory.

        Returns True on success, False on timeout/error.
        """
        if not self.available:
            return False

        try:
            with timeout(timeout_seconds):
                engine = self._get_engine()
                engine.remember(
                    name=name,
                    entity_type=entity_type,
                    observations=observations or [],
                    relations=relations or [],
                )
                return True
        except (TimeoutError, Exception):
            return False

    def commit(self, message: str, timeout_seconds: int = 5) -> bool:
        """Commit memory changes (git).

        Returns True on success, False on timeout/error.
        """
        import subprocess

        if not self.available or not self.memory_path:
            return False

        try:
            with timeout(timeout_seconds):
                result = subprocess.run(
                    ["git", "add", "-A"],
                    cwd=self.memory_path,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
                if result.returncode != 0:
                    return False

                result = subprocess.run(
                    ["git", "commit", "-m", message],
                    cwd=self.memory_path,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
                # Return True even if nothing to commit (exit code 1)
                return result.returncode in (0, 1)
        except (TimeoutError, subprocess.TimeoutExpired, Exception):
            return False
