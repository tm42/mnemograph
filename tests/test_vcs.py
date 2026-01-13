"""Tests for VCS operations with SQLite storage."""

import tempfile
from pathlib import Path

import pytest

from mnemograph.events import EventStore
from mnemograph.models import MemoryEvent
from mnemograph.vcs import MemoryVCS


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
    assert (temp_memory_dir / "mnemograph.db").exists()


def test_init_idempotent(initialized_vcs, temp_memory_dir):
    """Test init returns False when already initialized."""
    vcs = MemoryVCS(temp_memory_dir)
    assert vcs.init() is False


def test_status_clean(initialized_vcs):
    """Test status on clean repo."""
    status = initialized_vcs.status()
    assert status["branch"] in ("main", "master")
    assert status["is_dirty"] is False


def test_status_with_uncommitted(initialized_vcs, temp_memory_dir):
    """Test status shows dirty when database changed."""
    db_path = temp_memory_dir / "mnemograph.db"
    event_store = EventStore(db_path)
    event_store.append(MemoryEvent(
        op="create_entity",
        session_id="test",
        source="cc",
        data={"id": "e1", "name": "test", "type": "concept"},
    ))
    event_store.close()  # Force WAL checkpoint so git sees changes

    status = initialized_vcs.status()
    assert status["is_dirty"] is True
    assert status["event_count"] == 1


def test_commit(initialized_vcs, temp_memory_dir):
    """Test commit creates a git commit."""
    db_path = temp_memory_dir / "mnemograph.db"
    event_store = EventStore(db_path)
    event_store.append(MemoryEvent(
        op="create_entity",
        session_id="test",
        source="cc",
        data={"id": "e1", "name": "test", "type": "concept"},
    ))
    event_store.close()  # Force WAL checkpoint so git sees changes

    commit_hash = initialized_vcs.commit("Test commit")
    assert len(commit_hash) == 7

    status = initialized_vcs.status()
    assert status["is_dirty"] is False


def test_commit_nothing_to_commit(initialized_vcs):
    """Test commit raises when nothing to commit."""
    with pytest.raises(RuntimeError, match="Nothing to commit"):
        initialized_vcs.commit("Should fail")


def test_log(initialized_vcs, temp_memory_dir):
    """Test log shows commit history."""
    db_path = temp_memory_dir / "mnemograph.db"
    event_store = EventStore(db_path)
    event_store.append(MemoryEvent(
        op="create_entity",
        session_id="test",
        source="cc",
        data={"id": "e1", "name": "test", "type": "concept"},
    ))
    event_store.close()  # Force WAL checkpoint
    initialized_vcs.commit("Added test entity")

    commits = initialized_vcs.log(n=5)
    assert len(commits) >= 2  # init + our commit
    assert "Added test entity" in commits[0]["message"]


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


def test_show_current_state(initialized_vcs, temp_memory_dir):
    """Test show returns current graph state."""
    db_path = temp_memory_dir / "mnemograph.db"
    event_store = EventStore(db_path)
    event_store.append(MemoryEvent(
        op="create_entity",
        session_id="test",
        source="cc",
        data={"id": "e1", "name": "Test Entity", "type": "concept", "observations": []},
    ))

    # With SQLite, show only works for HEAD/working
    state = initialized_vcs.show("HEAD")
    assert len(state["entities"]) == 1
    assert "e1" in state["entities"]


def test_show_historical_unavailable(initialized_vcs):
    """Test show returns error for historical refs with SQLite."""
    state = initialized_vcs.show("HEAD~1")
    assert "error" in state
    assert "not available" in state["error"]


def test_diff_unavailable_with_sqlite(initialized_vcs):
    """Test diff returns note about SQLite limitations."""
    diff = initialized_vcs.diff("HEAD", None)
    assert "note" in diff
    assert "not available" in diff["note"]


def test_commit_auto_summary(initialized_vcs, temp_memory_dir):
    """Test commit with auto-summary includes event count."""
    db_path = temp_memory_dir / "mnemograph.db"
    event_store = EventStore(db_path)
    event_store.append(MemoryEvent(
        op="create_entity",
        session_id="test",
        source="cc",
        data={"id": "e1", "name": "My Feature", "type": "concept"},
    ))
    event_store.close()  # Force WAL checkpoint

    initialized_vcs.commit("Added feature", auto_summary=True)

    commits = initialized_vcs.log(n=1)
    assert "Added feature" in commits[0]["message"]
    assert "1 event" in commits[0]["message"]


# --- Git Safety Guard Tests ---


def test_nested_repo_detection():
    """Test that memory dir nested in another git repo raises error."""
    import subprocess
    from mnemograph.engine import MemoryEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        parent_repo = Path(tmpdir)

        # Initialize git in parent directory
        subprocess.run(["git", "init"], cwd=parent_repo, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=parent_repo,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=parent_repo,
            capture_output=True,
        )

        # Create nested memory directory (no mnemograph.db at git root)
        nested_memory_dir = parent_repo / "subdir" / ".claude" / "memory"
        nested_memory_dir.mkdir(parents=True)

        # Create mnemograph.db in nested dir (but not at git root)
        EventStore(nested_memory_dir / "mnemograph.db")

        # Should raise ValueError because git root doesn't contain mnemograph.db
        with pytest.raises(ValueError, match="nested in another git repository"):
            MemoryEngine(nested_memory_dir, "test-session")


def test_proper_memory_repo_passes_validation():
    """Test that memory dir at git root passes validation."""
    import subprocess
    from mnemograph.engine import MemoryEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir)

        # Initialize git in memory directory
        subprocess.run(["git", "init"], cwd=memory_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=memory_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=memory_dir,
            capture_output=True,
        )

        # Create mnemograph.db at git root (proper setup)
        EventStore(memory_dir / "mnemograph.db")

        # Should not raise
        engine = MemoryEngine(memory_dir, "test-session")
        # Use resolve() to handle macOS /var -> /private/var symlink
        assert engine._git_root.resolve() == memory_dir.resolve()


def test_marker_file_warning(caplog):
    """Test warning when .mnemograph marker is missing."""
    import subprocess
    import logging
    from mnemograph.engine import MemoryEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir)

        # Initialize git with mnemograph.db but no .mnemograph marker
        subprocess.run(["git", "init"], cwd=memory_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=memory_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=memory_dir,
            capture_output=True,
        )
        EventStore(memory_dir / "mnemograph.db")

        # Should log warning about missing marker
        with caplog.at_level(logging.WARNING, logger="mnemograph.engine"):
            engine = MemoryEngine(memory_dir, "test-session")

        assert "missing .mnemograph marker" in caplog.text


def test_marker_file_no_warning_when_present(caplog):
    """Test no warning when .mnemograph marker exists."""
    import subprocess
    import logging
    from mnemograph.engine import MemoryEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        memory_dir = Path(tmpdir)

        # Initialize git with mnemograph.db AND .mnemograph marker
        subprocess.run(["git", "init"], cwd=memory_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=memory_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=memory_dir,
            capture_output=True,
        )
        EventStore(memory_dir / "mnemograph.db")
        (memory_dir / ".mnemograph").write_text("")

        # Should not log warning
        with caplog.at_level(logging.WARNING, logger="mnemograph.engine"):
            engine = MemoryEngine(memory_dir, "test-session")

        assert ".mnemograph marker" not in caplog.text
