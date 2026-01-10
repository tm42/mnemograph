# Claude Memory VCS Implementation Brief

## Context

You're continuing work on `claude-memory`, a graph-based persistent memory system for Claude Code. The core system is already implemented:

- **Event store**: Append-only `events.jsonl` (source of truth)
- **State materialization**: Replay events → graph state
- **MCP server**: Tools for create/search/query entities and relations
- **Vector index**: sqlite-vec for semantic search

**Your task**: Add Git-based version control with a human-friendly CLI.

---

## Goal

Enable humans to:
1. **Visualize** the memory graph and its history
2. **Iterate** on the graph without a CC session running
3. **Commit** versions with meaningful messages
4. **Branch/merge** for experiments
5. **Time-travel** to any previous state

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interactions                             │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ claude-mem  │  │   Git CLI   │  │  Web View   │              │
│  │    CLI      │  │  (native)   │  │ (optional)  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
└─────────┼────────────────┼────────────────┼──────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     .claude/memory/                              │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  events.jsonl  ←── Git tracked (source of truth)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  .gitignore:                                              │   │
│  │    state.json      (derived - regenerate on checkout)     │   │
│  │    vectors.db      (derived - regenerate on checkout)     │   │
│  │    *.lock                                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
claude-memory/
├── src/
│   └── memory_engine/
│       ├── __init__.py
│       ├── models.py          # Entity, Relation, Event, Observation
│       ├── events.py          # EventStore class
│       ├── state.py           # materialize() function
│       ├── engine.py          # MemoryEngine class
│       ├── server.py          # MCP server
│       ├── vectors.py         # VectorIndex (sqlite-vec)
│       └── vcs.py             # NEW: Git operations wrapper
├── cli/
│   ├── __init__.py
│   └── main.py                # NEW: CLI entry point
├── templates/                  # NEW: Output templates
│   ├── graph.dot.jinja        # Graphviz DOT template
│   ├── log.txt.jinja          # Text log format
│   └── diff.txt.jinja         # Diff output format
├── tests/
│   ├── test_events.py
│   ├── test_state.py
│   └── test_vcs.py            # NEW: VCS tests
├── pyproject.toml
└── README.md
```

---

## Dependencies to Add

```toml
# pyproject.toml - add to existing dependencies
[project]
dependencies = [
    # ... existing deps ...
    "click>=8.0",              # CLI framework
    "rich>=13.0",              # Pretty terminal output
    "gitpython>=3.1",          # Git operations
    "jinja2>=3.0",             # Output templates
]

[project.scripts]
claude-mem = "cli.main:cli"
```

---

## CLI Commands Specification

### `claude-mem init`

Initialize git repo in memory directory if not exists.

```bash
$ claude-mem init
Initialized memory repository at .claude/memory/
Created .gitignore
Made initial commit
```

### `claude-mem status`

Show working state: uncommitted events, current branch, graph stats.

```bash
$ claude-mem status
On branch: main
Last commit: a1b2c3d "Added authentication decisions"

Graph state:
  Entities: 47 (3 new, 1 modified)
  Relations: 23 (2 new)
  
Uncommitted events: 6
  + create_entity: "OAuth2 flow" (concept)
  + create_entity: "JWT tokens" (concept)  
  + create_relation: "OAuth2 flow" -> "JWT tokens"
  + add_observation: "OAuth2 flow"
  ...
```

### `claude-mem log`

Show commit history with graph summaries.

```bash
$ claude-mem log
$ claude-mem log --oneline
$ claude-mem log -n 5
$ claude-mem log --since="2 days ago"

# Output:
commit a1b2c3d (HEAD -> main)
Author: Claude Code <cc@local>
Date:   2025-01-10 14:32:00

    Added authentication decisions
    
    Entities: +3, Relations: +2
    New: OAuth2 flow, JWT tokens, session management

commit 9f8e7d6
Author: User <user@local>
Date:   2025-01-10 12:15:00

    Initial project setup
    
    Entities: +12, Relations: +8
```

### `claude-mem show`

Show graph state at any ref.

```bash
$ claude-mem show              # current state
$ claude-mem show HEAD~3       # 3 commits ago
$ claude-mem show main         # branch tip
$ claude-mem show a1b2c3d      # specific commit

# Output formats:
$ claude-mem show --format=summary   # default
$ claude-mem show --format=json      # full JSON dump
$ claude-mem show --format=dot       # Graphviz DOT
$ claude-mem show --format=entities  # list entities only
```

### `claude-mem diff`

Semantic diff between graph states.

```bash
$ claude-mem diff                    # working vs HEAD
$ claude-mem diff HEAD~1             # HEAD vs HEAD~1
$ claude-mem diff main feature       # between branches
$ claude-mem diff a1b2c3d 9f8e7d6    # between commits

# Output:
Comparing a1b2c3d..9f8e7d6

Entities:
  + [concept] OAuth2 flow
      "Handles user authentication via external providers"
  + [concept] JWT tokens
      "Stateless authentication tokens"
  ~ [decision] auth_strategy
      - "Using session-based auth"
      + "Using token-based auth with JWT"
  - [question] how_to_handle_auth
      (deleted)

Relations:
  + OAuth2 flow --implements--> auth_strategy
  + JWT tokens --used_by--> OAuth2 flow
```

### `claude-mem commit`

Commit current events to git.

```bash
$ claude-mem commit -m "Added OAuth2 decisions"
$ claude-mem commit -a -m "message"  # auto-generate summary
$ claude-mem commit --amend          # amend last commit

# Output:
[main a1b2c3d] Added OAuth2 decisions
 events.jsonl | 6 ++++++
 
 Entities: +3, Relations: +2
```

### `claude-mem branch`

Branch management.

```bash
$ claude-mem branch                  # list branches
$ claude-mem branch experiment       # create branch
$ claude-mem branch -d experiment    # delete branch
```

### `claude-mem checkout`

Switch branches or restore state.

```bash
$ claude-mem checkout experiment     # switch branch
$ claude-mem checkout -b new-branch  # create and switch
$ claude-mem checkout HEAD~3         # detached HEAD at commit

# IMPORTANT: After checkout, regenerate derived files
# (state.json, vectors.db)
```

### `claude-mem merge`

Merge branches.

```bash
$ claude-mem merge feature           # merge feature into current
$ claude-mem merge --abort           # abort conflicted merge

# Conflict handling for JSONL:
# - Events are append-only, so conflicts are rare
# - If conflict: show both sides, let user pick
```

### `claude-mem graph`

Visualize the graph.

```bash
$ claude-mem graph                   # open in default viewer
$ claude-mem graph -o graph.png      # save to file
$ claude-mem graph --format=dot      # output DOT source
$ claude-mem graph --focus="OAuth2"  # subgraph around entity
$ claude-mem graph --depth=2         # limit traversal depth
```

### `claude-mem export`

Export graph in various formats.

```bash
$ claude-mem export graph.json       # full JSON
$ claude-mem export graph.md         # Markdown documentation
$ claude-mem export graph.dot        # Graphviz DOT
$ claude-mem export graph.cypher     # Neo4j Cypher statements
```

### `claude-mem edit`

Interactive editing (opens in $EDITOR or TUI).

```bash
$ claude-mem edit entity "OAuth2 flow"     # edit specific entity
$ claude-mem edit --interactive            # TUI for browsing/editing
```

---

## Implementation Guide

### 1. Git Operations Wrapper (`src/memory_engine/vcs.py`)

```python
"""Git operations wrapper for memory versioning."""

from pathlib import Path
from git import Repo, GitCommandError
from git.exc import InvalidGitRepositoryError
import json
from datetime import datetime
from typing import Optional

from .state import materialize
from .events import EventStore


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
        # Check for uncommitted changes
        events_modified = "events.jsonl" in [
            item.a_path for item in self.repo.index.diff(None)
        ]
        
        # Count uncommitted events
        uncommitted_events = []
        if events_modified:
            # Get committed version
            try:
                committed = self.repo.git.show("HEAD:events.jsonl")
                committed_lines = set(committed.strip().split("\n")) if committed.strip() else set()
            except GitCommandError:
                committed_lines = set()
            
            # Get current version
            current_lines = set(self.events_file.read_text().strip().split("\n")) if self.events_file.read_text().strip() else set()
            
            # New events = current - committed
            new_lines = current_lines - committed_lines
            for line in new_lines:
                if line:
                    uncommitted_events.append(json.loads(line))
        
        return {
            "branch": self.repo.active_branch.name,
            "commit": self.repo.head.commit.hexsha[:7] if self.repo.head.is_valid() else None,
            "commit_message": self.repo.head.commit.message.strip() if self.repo.head.is_valid() else None,
            "uncommitted_events": uncommitted_events,
            "is_dirty": self.repo.is_dirty(),
        }
    
    def log(self, n: int = 10, oneline: bool = False) -> list[dict]:
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
        # Get events at this commit
        try:
            events_content = self.repo.git.show(f"{commit.hexsha}:events.jsonl")
            events = [json.loads(line) for line in events_content.strip().split("\n") if line]
            
            # Count by operation type
            stats = {"entities": 0, "relations": 0}
            entity_ids = set()
            relation_count = 0
            
            for event in events:
                if event["op"] == "create_entity":
                    entity_ids.add(event["data"]["id"])
                elif event["op"] == "delete_entity":
                    entity_ids.discard(event["data"]["id"])
                elif event["op"] == "create_relation":
                    relation_count += 1
                elif event["op"] == "delete_relation":
                    relation_count -= 1
            
            stats["entities"] = len(entity_ids)
            stats["relations"] = max(0, relation_count)
            return stats
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
        
        # Use existing materialize function
        from .models import MemoryEvent
        typed_events = [MemoryEvent.model_validate(e) for e in events]
        state = materialize(typed_events)
        
        return {
            "ref": ref,
            "entities": {eid: e.model_dump() for eid, e in state.entities.items()},
            "relations": [r.model_dump() for r in state.relations],
        }
    
    def diff(self, ref_a: str = "HEAD", ref_b: str = None) -> dict:
        """Compute semantic diff between two refs."""
        state_a = self.show(ref_a)
        
        if ref_b is None:
            # Compare with working directory
            event_store = EventStore(self.events_file)
            events = event_store.read_all()
            state = materialize(events)
            state_b = {
                "ref": "working",
                "entities": {eid: e.model_dump() for eid, e in state.entities.items()},
                "relations": [r.model_dump() for r in state.relations],
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
        if not self.repo.is_dirty(path="events.jsonl"):
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
            # List branches
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
            # Create and checkout new branch
            new_branch = self.repo.create_head(ref)
            new_branch.checkout()
        else:
            # Checkout existing ref
            self.repo.git.checkout(ref)
        
        # Regenerate derived files
        self._regenerate_derived()
    
    def _regenerate_derived(self) -> None:
        """Regenerate state.json and vectors.db from events."""
        from .engine import MemoryEngine
        
        # Re-initialize engine (will rebuild state and vector index)
        engine = MemoryEngine(self.memory_dir, session_id="vcs")
        
        # Save state snapshot for quick loading
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
```

### 2. CLI Entry Point (`cli/main.py`)

```python
"""CLI for claude-memory version control."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax
from rich.panel import Panel
import json

from memory_engine.vcs import MemoryVCS

console = Console()


def get_memory_dir() -> Path:
    """Find memory directory (walk up to find .claude/memory)."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        memory_dir = parent / ".claude" / "memory"
        if memory_dir.exists():
            return memory_dir
    # Default to cwd/.claude/memory
    return cwd / ".claude" / "memory"


@click.group()
@click.pass_context
def cli(ctx):
    """Claude Memory - Git-based knowledge graph version control."""
    ctx.ensure_object(dict)
    ctx.obj["memory_dir"] = get_memory_dir()


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize memory repository."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    if vcs.init():
        console.print(f"[green]✓[/green] Initialized memory repository at {ctx.obj['memory_dir']}")
    else:
        console.print(f"[yellow]![/yellow] Repository already initialized at {ctx.obj['memory_dir']}")


@cli.command()
@click.pass_context
def status(ctx):
    """Show working tree status."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    status = vcs.status()
    
    console.print(f"On branch: [cyan]{status['branch']}[/cyan]")
    if status["commit"]:
        console.print(f"Last commit: [yellow]{status['commit']}[/yellow] {status['commit_message']}")
    
    console.print()
    
    if status["uncommitted_events"]:
        console.print(f"[yellow]Uncommitted events: {len(status['uncommitted_events'])}[/yellow]")
        for event in status["uncommitted_events"][:10]:
            op = event["op"]
            if op == "create_entity":
                name = event["data"].get("name", "?")
                etype = event["data"].get("type", "entity")
                console.print(f"  [green]+[/green] {op}: \"{name}\" ({etype})")
            elif op == "create_relation":
                console.print(f"  [green]+[/green] {op}: {event['data'].get('from_entity')} -> {event['data'].get('to_entity')}")
            else:
                console.print(f"  [blue]~[/blue] {op}")
        
        if len(status["uncommitted_events"]) > 10:
            console.print(f"  ... and {len(status['uncommitted_events']) - 10} more")
    else:
        console.print("[green]Nothing to commit, working tree clean[/green]")


@cli.command()
@click.option("-n", "--max-count", default=10, help="Number of commits to show")
@click.option("--oneline", is_flag=True, help="Show one line per commit")
@click.option("--since", help="Show commits since date")
@click.pass_context
def log(ctx, max_count, oneline, since):
    """Show commit logs."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    commits = vcs.log(n=max_count)
    
    for commit in commits:
        if oneline:
            console.print(
                f"[yellow]{commit['short_hash']}[/yellow] "
                f"{commit['message'].split(chr(10))[0]}"
            )
        else:
            console.print(f"[yellow]commit {commit['hash']}[/yellow]")
            console.print(f"Author: {commit['author']}")
            console.print(f"Date:   {commit['date']}")
            console.print()
            for line in commit["message"].split("\n"):
                console.print(f"    {line}")
            console.print()
            stats = commit["stats"]
            console.print(f"    [dim]Entities: {stats['entities']}, Relations: {stats['relations']}[/dim]")
            console.print()


@cli.command()
@click.argument("ref", default="HEAD")
@click.option("--format", "fmt", type=click.Choice(["summary", "json", "dot", "entities"]), default="summary")
@click.pass_context
def show(ctx, ref, fmt):
    """Show graph state at a ref."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    state = vcs.show(ref)
    
    if fmt == "json":
        console.print(Syntax(json.dumps(state, indent=2, default=str), "json"))
    
    elif fmt == "dot":
        dot = generate_dot(state)
        console.print(dot)
    
    elif fmt == "entities":
        for eid, entity in state["entities"].items():
            console.print(f"[cyan]{entity['name']}[/cyan] ({entity['type']})")
            for obs in entity.get("observations", [])[:2]:
                console.print(f"  - {obs['text'][:60]}...")
    
    else:  # summary
        console.print(f"[bold]State at {ref}[/bold]")
        console.print(f"Entities: {len(state['entities'])}")
        console.print(f"Relations: {len(state['relations'])}")
        console.print()
        
        # Group by type
        by_type = {}
        for entity in state["entities"].values():
            t = entity["type"]
            by_type.setdefault(t, []).append(entity)
        
        for etype, entities in sorted(by_type.items()):
            console.print(f"[bold]{etype}[/bold] ({len(entities)})")
            for e in entities[:5]:
                console.print(f"  • {e['name']}")
            if len(entities) > 5:
                console.print(f"  ... and {len(entities) - 5} more")


@cli.command()
@click.argument("ref_a", default="HEAD")
@click.argument("ref_b", default=None, required=False)
@click.pass_context
def diff(ctx, ref_a, ref_b):
    """Show diff between refs."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    diff_result = vcs.diff(ref_a, ref_b)
    
    console.print(f"[bold]Comparing {diff_result['from']}..{diff_result['to']}[/bold]")
    console.print()
    
    # Entities
    if diff_result["entities"]["added"]:
        console.print("[bold green]Added entities:[/bold green]")
        for eid, entity in diff_result["entities"]["added"].items():
            console.print(f"  [green]+[/green] [{entity['type']}] {entity['name']}")
            for obs in entity.get("observations", [])[:1]:
                console.print(f"      \"{obs['text'][:50]}...\"")
    
    if diff_result["entities"]["removed"]:
        console.print("[bold red]Removed entities:[/bold red]")
        for eid, entity in diff_result["entities"]["removed"].items():
            console.print(f"  [red]-[/red] [{entity['type']}] {entity['name']}")
    
    if diff_result["entities"]["modified"]:
        console.print("[bold yellow]Modified entities:[/bold yellow]")
        for eid, changes in diff_result["entities"]["modified"].items():
            console.print(f"  [yellow]~[/yellow] {changes['after']['name']}")
            # Show observation changes
            before_obs = {o["id"]: o for o in changes["before"].get("observations", [])}
            after_obs = {o["id"]: o for o in changes["after"].get("observations", [])}
            
            new_obs = set(after_obs.keys()) - set(before_obs.keys())
            for oid in new_obs:
                console.print(f"      [green]+[/green] \"{after_obs[oid]['text'][:50]}...\"")
    
    # Relations
    if diff_result["relations"]["added"]:
        console.print("[bold green]Added relations:[/bold green]")
        for rel in diff_result["relations"]["added"]:
            console.print(f"  [green]+[/green] {rel['from_entity']} --{rel['type']}--> {rel['to_entity']}")
    
    if diff_result["relations"]["removed"]:
        console.print("[bold red]Removed relations:[/bold red]")
        for rel in diff_result["relations"]["removed"]:
            console.print(f"  [red]-[/red] {rel['from_entity']} --{rel['type']}--> {rel['to_entity']}")


@cli.command()
@click.option("-m", "--message", required=True, help="Commit message")
@click.option("-a", "--auto-summary", is_flag=True, help="Add auto-generated summary")
@click.pass_context
def commit(ctx, message, auto_summary):
    """Commit changes to events."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    try:
        commit_hash = vcs.commit(message, auto_summary=auto_summary)
        console.print(f"[green]✓[/green] [{ctx.obj['memory_dir'].parent.name} {commit_hash}] {message}")
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")


@cli.command()
@click.argument("name", required=False)
@click.option("-d", "--delete", is_flag=True, help="Delete branch")
@click.pass_context
def branch(ctx, name, delete):
    """List, create, or delete branches."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    branches = vcs.branch(name, delete=delete)
    
    if name is None:
        # List branches
        current = vcs.repo.active_branch.name
        for b in branches:
            if b == current:
                console.print(f"[green]* {b}[/green]")
            else:
                console.print(f"  {b}")
    elif delete:
        console.print(f"[green]✓[/green] Deleted branch {name}")
    else:
        console.print(f"[green]✓[/green] Created branch {name}")


@cli.command()
@click.argument("ref")
@click.option("-b", "--create", is_flag=True, help="Create and checkout new branch")
@click.pass_context
def checkout(ctx, ref, create):
    """Checkout branch or commit."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    vcs.checkout(ref, create=create)
    console.print(f"[green]✓[/green] Switched to {'new branch' if create else ''} '{ref}'")
    console.print("[dim]Regenerated derived files (state.json, vectors.db)[/dim]")


@cli.command()
@click.argument("branch")
@click.option("--abort", is_flag=True, help="Abort merge in progress")
@click.pass_context
def merge(ctx, branch, abort):
    """Merge branch into current."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    result = vcs.merge(branch, abort=abort)
    
    if result["status"] == "success":
        console.print(f"[green]✓[/green] Merged '{branch}' into current branch")
    elif result["status"] == "aborted":
        console.print("[yellow]Merge aborted[/yellow]")
    elif result["status"] == "conflict":
        console.print(f"[red]Merge conflict![/red]")
        console.print(result["message"])
        console.print("\nResolve conflicts in events.jsonl, then run 'claude-mem commit'")
        console.print("Or run 'claude-mem merge --abort' to cancel")


@cli.command()
@click.option("-o", "--output", type=click.Path(), help="Output file")
@click.option("--format", "fmt", type=click.Choice(["png", "svg", "dot"]), default="png")
@click.option("--focus", help="Entity name to focus on")
@click.option("--depth", default=3, help="Traversal depth from focus")
@click.pass_context
def graph(ctx, output, fmt, focus, depth):
    """Visualize the graph."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    state = vcs.show("HEAD")
    
    dot = generate_dot(state, focus=focus, depth=depth)
    
    if fmt == "dot" or output is None:
        console.print(dot)
    else:
        # Use graphviz to render
        import subprocess
        
        output = output or f"graph.{fmt}"
        result = subprocess.run(
            ["dot", f"-T{fmt}", "-o", output],
            input=dot,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            console.print(f"[green]✓[/green] Saved graph to {output}")
        else:
            console.print(f"[red]Error:[/red] {result.stderr}")
            console.print("[dim]Make sure graphviz is installed (apt install graphviz)[/dim]")


def generate_dot(state: dict, focus: str = None, depth: int = 3) -> str:
    """Generate Graphviz DOT format."""
    lines = ["digraph memory {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=rounded];")
    lines.append("")
    
    # Color by type
    type_colors = {
        "concept": "#E3F2FD",
        "decision": "#FFF3E0",
        "project": "#E8F5E9",
        "pattern": "#F3E5F5",
        "question": "#FFF8E1",
        "learning": "#E0F7FA",
        "entity": "#FAFAFA",
    }
    
    # If focus, filter to nearby entities
    if focus:
        # Find focus entity
        focus_id = None
        for eid, entity in state["entities"].items():
            if entity["name"].lower() == focus.lower():
                focus_id = eid
                break
        
        if focus_id:
            # BFS to find nearby entities
            nearby = {focus_id}
            frontier = [focus_id]
            for _ in range(depth):
                next_frontier = []
                for eid in frontier:
                    for rel in state["relations"]:
                        if rel["from_entity"] == eid:
                            if rel["to_entity"] not in nearby:
                                nearby.add(rel["to_entity"])
                                next_frontier.append(rel["to_entity"])
                        elif rel["to_entity"] == eid:
                            if rel["from_entity"] not in nearby:
                                nearby.add(rel["from_entity"])
                                next_frontier.append(rel["from_entity"])
                frontier = next_frontier
            
            # Filter state
            state = {
                "entities": {k: v for k, v in state["entities"].items() if k in nearby},
                "relations": [r for r in state["relations"] 
                             if r["from_entity"] in nearby and r["to_entity"] in nearby],
            }
    
    # Add nodes
    for eid, entity in state["entities"].items():
        color = type_colors.get(entity["type"], "#FAFAFA")
        label = entity["name"].replace('"', '\\"')
        # Truncate long labels
        if len(label) > 30:
            label = label[:27] + "..."
        lines.append(f'  "{eid}" [label="{label}\\n({entity["type"]})", fillcolor="{color}", style="filled,rounded"];')
    
    lines.append("")
    
    # Add edges
    for rel in state["relations"]:
        label = rel["type"].replace('"', '\\"')
        lines.append(f'  "{rel["from_entity"]}" -> "{rel["to_entity"]}" [label="{label}"];')
    
    lines.append("}")
    return "\n".join(lines)


@cli.command()
@click.argument("output", type=click.Path())
@click.pass_context
def export(ctx, output):
    """Export graph to file."""
    vcs = MemoryVCS(ctx.obj["memory_dir"])
    state = vcs.show("HEAD")
    
    output_path = Path(output)
    suffix = output_path.suffix.lower()
    
    if suffix == ".json":
        output_path.write_text(json.dumps(state, indent=2, default=str))
    
    elif suffix == ".dot":
        output_path.write_text(generate_dot(state))
    
    elif suffix == ".md":
        md = generate_markdown(state)
        output_path.write_text(md)
    
    elif suffix == ".cypher":
        cypher = generate_cypher(state)
        output_path.write_text(cypher)
    
    else:
        console.print(f"[red]Unknown format:[/red] {suffix}")
        console.print("Supported: .json, .dot, .md, .cypher")
        return
    
    console.print(f"[green]✓[/green] Exported to {output}")


def generate_markdown(state: dict) -> str:
    """Generate Markdown documentation."""
    lines = ["# Knowledge Graph", ""]
    
    # Group by type
    by_type = {}
    for entity in state["entities"].values():
        t = entity["type"]
        by_type.setdefault(t, []).append(entity)
    
    for etype, entities in sorted(by_type.items()):
        lines.append(f"## {etype.title()}s")
        lines.append("")
        for entity in sorted(entities, key=lambda e: e["name"]):
            lines.append(f"### {entity['name']}")
            lines.append("")
            for obs in entity.get("observations", []):
                lines.append(f"- {obs['text']}")
            lines.append("")
    
    # Relations
    if state["relations"]:
        lines.append("## Relationships")
        lines.append("")
        lines.append("| From | Relation | To |")
        lines.append("|------|----------|-----|")
        
        # Need to resolve entity IDs to names
        id_to_name = {eid: e["name"] for eid, e in state["entities"].items()}
        
        for rel in state["relations"]:
            from_name = id_to_name.get(rel["from_entity"], rel["from_entity"])
            to_name = id_to_name.get(rel["to_entity"], rel["to_entity"])
            lines.append(f"| {from_name} | {rel['type']} | {to_name} |")
    
    return "\n".join(lines)


def generate_cypher(state: dict) -> str:
    """Generate Neo4j Cypher statements."""
    lines = ["// Neo4j Cypher import for knowledge graph", ""]
    
    # Create nodes
    lines.append("// Create entities")
    for eid, entity in state["entities"].items():
        obs_text = "; ".join(o["text"] for o in entity.get("observations", []))
        obs_text = obs_text.replace("'", "\\'")[:500]
        name = entity["name"].replace("'", "\\'")
        lines.append(
            f"CREATE (:{entity['type'].title()} {{id: '{eid}', name: '{name}', observations: '{obs_text}'}});"
        )
    
    lines.append("")
    lines.append("// Create relationships")
    for rel in state["relations"]:
        rel_type = rel["type"].upper().replace(" ", "_").replace("-", "_")
        lines.append(
            f"MATCH (a {{id: '{rel['from_entity']}'}}), (b {{id: '{rel['to_entity']}'}}) "
            f"CREATE (a)-[:{rel_type}]->(b);"
        )
    
    return "\n".join(lines)


if __name__ == "__main__":
    cli()
```

---

## Testing Strategy

### Unit Tests (`tests/test_vcs.py`)

```python
"""Tests for VCS operations."""

import pytest
from pathlib import Path
import tempfile
import json

from memory_engine.vcs import MemoryVCS
from memory_engine.events import EventStore
from memory_engine.models import MemoryEvent


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
    vcs = MemoryVCS(temp_memory_dir)
    assert vcs.init() is True
    assert (temp_memory_dir / ".git").exists()
    assert (temp_memory_dir / ".gitignore").exists()


def test_init_idempotent(initialized_vcs):
    assert initialized_vcs.init() is False


def test_status_clean(initialized_vcs):
    status = initialized_vcs.status()
    assert status["branch"] == "main" or status["branch"] == "master"
    assert status["uncommitted_events"] == []
    assert status["is_dirty"] is False


def test_status_with_uncommitted(initialized_vcs, temp_memory_dir):
    # Add an event
    events_file = temp_memory_dir / "events.jsonl"
    event = {"op": "create_entity", "data": {"id": "e1", "name": "test", "type": "concept"}, "session_id": "test", "source": "cc", "ts": "2025-01-01T00:00:00", "id": "ev1"}
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    
    status = initialized_vcs.status()
    assert len(status["uncommitted_events"]) == 1
    assert status["is_dirty"] is True


def test_commit(initialized_vcs, temp_memory_dir):
    # Add an event
    events_file = temp_memory_dir / "events.jsonl"
    event = {"op": "create_entity", "data": {"id": "e1", "name": "test", "type": "concept"}, "session_id": "test", "source": "cc", "ts": "2025-01-01T00:00:00", "id": "ev1"}
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    
    commit_hash = initialized_vcs.commit("Test commit")
    assert len(commit_hash) == 7
    
    status = initialized_vcs.status()
    assert status["uncommitted_events"] == []


def test_branch_operations(initialized_vcs):
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
    diff = initialized_vcs.diff("HEAD", None)
    assert diff["entities"]["added"] == {}
    assert diff["entities"]["removed"] == {}


def test_log(initialized_vcs, temp_memory_dir):
    # Add and commit
    events_file = temp_memory_dir / "events.jsonl"
    event = {"op": "create_entity", "data": {"id": "e1", "name": "test", "type": "concept"}, "session_id": "test", "source": "cc", "ts": "2025-01-01T00:00:00", "id": "ev1"}
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    initialized_vcs.commit("Added test entity")
    
    commits = initialized_vcs.log(n=5)
    assert len(commits) >= 2  # init + our commit
    assert "Added test entity" in commits[0]["message"]
```

---

## Important Implementation Notes

### 1. Event File Locking

When multiple processes might access events.jsonl:

```python
import fcntl

def append_event_safe(events_file: Path, event: dict):
    with open(events_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(event) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

### 2. After Checkout Hook

The engine needs to know when to regenerate state:

```python
# In engine.py, check if state is stale
def _load_state(self) -> GraphState:
    state_file = self.memory_dir / "state.json"
    events_file = self.memory_dir / "events.jsonl"
    
    # If state.json exists and is newer than events.jsonl, use it
    if state_file.exists():
        state_mtime = state_file.stat().st_mtime
        events_mtime = events_file.stat().st_mtime if events_file.exists() else 0
        
        if state_mtime > events_mtime:
            # Load cached state
            data = json.loads(state_file.read_text())
            # ... reconstruct GraphState
            return state
    
    # Otherwise rebuild from events
    events = self.event_store.read_all()
    return materialize(events)
```

### 3. Merge Conflict Strategy

Since events.jsonl is append-only, conflicts are rare. When they occur:

```python
def resolve_merge_conflict(events_file: Path):
    """Resolve by keeping all events and sorting by timestamp."""
    content = events_file.read_text()
    
    # Extract conflict markers
    # <<<<<<< HEAD
    # events from HEAD
    # =======
    # events from branch
    # >>>>>>> branch
    
    # Strategy: Keep both sets of events, sort by timestamp
    all_events = []
    
    # Parse and collect all events from both sides
    for line in content.split("\n"):
        if line.startswith(("<<<", "===", ">>>")):
            continue
        if line.strip():
            all_events.append(json.loads(line))
    
    # Sort by timestamp
    all_events.sort(key=lambda e: e["ts"])
    
    # Remove duplicates (by event ID)
    seen = set()
    unique = []
    for event in all_events:
        if event["id"] not in seen:
            seen.add(event["id"])
            unique.append(event)
    
    # Write back
    with open(events_file, "w") as f:
        for event in unique:
            f.write(json.dumps(event) + "\n")
```

### 4. Entity Name Resolution in Diffs

The diff output needs human-readable names, not IDs:

```python
def format_diff_with_names(diff: dict, state_before: dict, state_after: dict) -> str:
    """Format diff with entity names resolved."""
    id_to_name = {}
    for entities in [state_before["entities"], state_after["entities"]]:
        for eid, entity in entities.items():
            id_to_name[eid] = entity["name"]
    
    # Now use id_to_name when formatting relations
    # ...
```

---

## Checklist

- [ ] Add dependencies to pyproject.toml (click, rich, gitpython, jinja2)
- [ ] Create `src/memory_engine/vcs.py` with MemoryVCS class
- [ ] Create `cli/main.py` with all commands
- [ ] Add entry point to pyproject.toml (`claude-mem = "cli.main:cli"`)
- [ ] Test `claude-mem init` in a fresh directory
- [ ] Test `claude-mem status` shows uncommitted events
- [ ] Test `claude-mem commit -m "message"` works
- [ ] Test `claude-mem log` shows history
- [ ] Test `claude-mem diff` shows changes
- [ ] Test `claude-mem branch` / `checkout` / `merge`
- [ ] Test `claude-mem graph` generates DOT output
- [ ] Add unit tests
- [ ] Update README with CLI usage examples

---

## Example Session

```bash
# Initialize
$ cd my-project
$ claude-mem init
✓ Initialized memory repository at .claude/memory/

# After some CC sessions...
$ claude-mem status
On branch: main
Last commit: a1b2c3d "Initialize memory repository"

Uncommitted events: 12
  + create_entity: "Repository pattern" (concept)
  + create_entity: "SQLite choice" (decision)
  ...

# Review what changed
$ claude-mem diff
Comparing HEAD..working

Added entities:
  + [concept] Repository pattern
      "Separates data access from business logic"
  + [decision] SQLite choice
      "Chose SQLite for simplicity and portability"
...

# Commit
$ claude-mem commit -m "Added architecture decisions from design session"
✓ [my-project a1b2c3d] Added architecture decisions from design session

# View history
$ claude-mem log --oneline
a1b2c3d Added architecture decisions from design session
9f8e7d6 Initialize memory repository

# Visualize
$ claude-mem graph -o architecture.png
✓ Saved graph to architecture.png

# Experiment on a branch
$ claude-mem checkout -b experiment/new-approach
✓ Switched to new branch 'experiment/new-approach'

# ... make changes via CC session ...

$ claude-mem commit -m "Trying microservices approach"
$ claude-mem checkout main
$ claude-mem merge experiment/new-approach
✓ Merged 'experiment/new-approach' into current branch
```

---

Good luck! Focus on getting `init`, `status`, `commit`, and `log` working first. The rest can iterate.
