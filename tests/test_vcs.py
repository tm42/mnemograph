"""Tests for VCS operations."""

import json
import tempfile
from pathlib import Path

import pytest

from memory_engine.vcs import MemoryVCS


@pytest.fixture
def temp_memory_dir():
    """Create temporary memory directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir) / ".claude" / "memory"
        memory_dir.mkdir(parents=True)
        yield memory_dir


@pytest.fixture
def initialized_vcs(temp_memory_dir):
    """Create initialized VCS."""
    vcs = MemoryVCS(temp_memory_dir)
    vcs.init()
    return vcs


def test_init_creates_repo(temp_memory_dir):
    """Test init creates git repo and .gitignore."""
    vcs = MemoryVCS(temp_memory_dir)
    assert vcs.init() is True
    assert (temp_memory_dir / ".git").exists()
    assert (temp_memory_dir / ".gitignore").exists()
    assert (temp_memory_dir / "events.jsonl").exists()


def test_init_idempotent(initialized_vcs, temp_memory_dir):
    """Test init returns False when already initialized."""
    vcs = MemoryVCS(temp_memory_dir)
    assert vcs.init() is False


def test_status_clean(initialized_vcs):
    """Test status on clean repo."""
    status = initialized_vcs.status()
    assert status["branch"] in ("main", "master")
    assert status["uncommitted_events"] == []
    assert status["is_dirty"] is False


def test_status_with_uncommitted(initialized_vcs, temp_memory_dir):
    """Test status shows uncommitted events."""
    events_file = temp_memory_dir / "events.jsonl"
    event = {
        "op": "create_entity",
        "data": {"id": "e1", "name": "test", "type": "concept"},
        "session_id": "test",
        "source": "cc",
        "ts": "2025-01-01T00:00:00",
        "id": "ev1"
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    status = initialized_vcs.status()
    assert len(status["uncommitted_events"]) == 1
    assert status["is_dirty"] is True


def test_commit(initialized_vcs, temp_memory_dir):
    """Test commit creates a git commit."""
    events_file = temp_memory_dir / "events.jsonl"
    event = {
        "op": "create_entity",
        "data": {"id": "e1", "name": "test", "type": "concept"},
        "session_id": "test",
        "source": "cc",
        "ts": "2025-01-01T00:00:00",
        "id": "ev1"
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    commit_hash = initialized_vcs.commit("Test commit")
    assert len(commit_hash) == 7

    status = initialized_vcs.status()
    assert status["uncommitted_events"] == []
    assert status["is_dirty"] is False


def test_commit_nothing_to_commit(initialized_vcs):
    """Test commit raises when nothing to commit."""
    with pytest.raises(RuntimeError, match="Nothing to commit"):
        initialized_vcs.commit("Should fail")


def test_log(initialized_vcs, temp_memory_dir):
    """Test log shows commit history."""
    # Add and commit
    events_file = temp_memory_dir / "events.jsonl"
    event = {
        "op": "create_entity",
        "data": {"id": "e1", "name": "test", "type": "concept"},
        "session_id": "test",
        "source": "cc",
        "ts": "2025-01-01T00:00:00",
        "id": "ev1"
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    initialized_vcs.commit("Added test entity")

    commits = initialized_vcs.log(n=5)
    assert len(commits) >= 2  # init + our commit
    assert "Added test entity" in commits[0]["message"]


def test_log_stats(initialized_vcs, temp_memory_dir):
    """Test log shows entity/relation stats."""
    events_file = temp_memory_dir / "events.jsonl"

    # Add entities
    for i in range(3):
        event = {
            "op": "create_entity",
            "data": {"id": f"e{i}", "name": f"Entity {i}", "type": "concept"},
            "session_id": "test",
            "source": "cc",
            "ts": "2025-01-01T00:00:00",
            "id": f"ev{i}"
        }
        with open(events_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    initialized_vcs.commit("Added 3 entities")

    commits = initialized_vcs.log(n=1)
    assert commits[0]["stats"]["entities"] == 3


def test_branch_operations(initialized_vcs):
    """Test branch create, list, delete."""
    # Create branch
    branches = initialized_vcs.branch("feature")
    assert "feature" in branches

    # List branches
    branches = initialized_vcs.branch()
    assert len(branches) >= 2

    # Delete branch
    branches = initialized_vcs.branch("feature", delete=True)
    assert "feature" not in branches


def test_diff_empty(initialized_vcs):
    """Test diff when no changes."""
    diff = initialized_vcs.diff("HEAD", None)
    assert diff["entities"]["added"] == {}
    assert diff["entities"]["removed"] == {}


def test_diff_with_changes(initialized_vcs, temp_memory_dir):
    """Test diff shows added entities."""
    events_file = temp_memory_dir / "events.jsonl"
    event = {
        "op": "create_entity",
        "data": {"id": "e1", "name": "New Entity", "type": "concept", "observations": []},
        "session_id": "test",
        "source": "cc",
        "ts": "2025-01-01T00:00:00",
        "id": "ev1"
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    diff = initialized_vcs.diff("HEAD", None)
    assert len(diff["entities"]["added"]) == 1
    assert "e1" in diff["entities"]["added"]


def test_show(initialized_vcs, temp_memory_dir):
    """Test show returns graph state at ref."""
    events_file = temp_memory_dir / "events.jsonl"
    event = {
        "op": "create_entity",
        "data": {"id": "e1", "name": "Test Entity", "type": "concept", "observations": []},
        "session_id": "test",
        "source": "cc",
        "ts": "2025-01-01T00:00:00",
        "id": "ev1"
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    initialized_vcs.commit("Added entity")

    state = initialized_vcs.show("HEAD")
    assert len(state["entities"]) == 1
    assert "e1" in state["entities"]


def test_commit_auto_summary(initialized_vcs, temp_memory_dir):
    """Test commit with auto-summary."""
    events_file = temp_memory_dir / "events.jsonl"
    event = {
        "op": "create_entity",
        "data": {"id": "e1", "name": "My Feature", "type": "concept", "observations": []},
        "session_id": "test",
        "source": "cc",
        "ts": "2025-01-01T00:00:00",
        "id": "ev1"
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    initialized_vcs.commit("Added feature", auto_summary=True)

    commits = initialized_vcs.log(n=1)
    # Message should contain auto-generated summary
    assert "Added feature" in commits[0]["message"]
    assert "+1 entities" in commits[0]["message"] or "My Feature" in commits[0]["message"]
