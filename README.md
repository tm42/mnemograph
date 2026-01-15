# Mnemograph

<!-- mcp-name: io.github.tm42/mnemograph -->

A persistent, event-sourced knowledge graph for AI coding agents. Unlike simple key-value memory, Mnemograph captures **entities**, **relations**, and **observations** — enabling semantic search, tiered context retrieval, and git-based version control of your AI's memory.

**Works with**: Claude Code, opencode, codex CLI, Zed, Continue.dev, and any MCP-compatible agent.

## Why Mnemograph?

AI coding sessions are ephemeral. Mnemograph gives your AI partner persistent memory that:

- **Survives across sessions** — decisions, patterns, learnings persist
- **Supports semantic search** — find relevant context by meaning, not just keywords
- **Provides tiered retrieval** — shallow summaries to deep subgraphs based on need
- **Versions like code** — branch, commit, diff, revert your knowledge graph
- **Enables collaboration** — share memory repos across users or projects

## Memory Scope: Local vs Global

**Before using mnemograph, decide where to store memory:**

| Scope | Path | Use When |
|-------|------|----------|
| **Project-local** | `./.claude/memory` | Knowledge specific to this repo (architecture, decisions, patterns) |
| **Global** | `~/.claude/memory` | Cross-project knowledge (personal learnings, universal patterns, preferences) |
| **Custom** | Any path via `MEMORY_PATH` | Shared team memory, org-wide knowledge bases |

**Important:** Agents should ask the user which scope to use when first setting up mnemograph for a project. This affects where knowledge is stored and whether it's shared across projects.

```bash
# Project-local (default)
MEMORY_PATH=".claude/memory"

# Global (cross-project)
MEMORY_PATH="$HOME/.claude/memory"

# CLI: use --global flag
mnemograph --global status
mnemograph --global graph
```

## Quick Start

### Option 1: Let Claude Code install it

Give Claude Code this repo URL and ask it to set up mnemograph:

```
https://github.com/tm42/mnemograph
```

Or point Claude to the setup instructions directly:

```
Read https://raw.githubusercontent.com/tm42/mnemograph/main/SETUP_CLAUDE_CODE.md and follow them
```

### Option 2: Manual installation

```bash
# Install from PyPI
pip install mnemograph

# Add to Claude Code (global, available in all projects)
claude mcp add --scope user mnemograph \
  -e MEMORY_PATH="$HOME/.claude/memory" \
  -- uvx mnemograph

# Initialize memory directory
mkdir -p ~/.claude/memory
```

### Option 3: Other MCP Clients

Each MCP client has a different configuration format. See [UNIVERSAL_MCP_COMPATIBILITY.md](UNIVERSAL_MCP_COMPATIBILITY.md) for copy-paste configs for:

- **opencode** — `~/.config/opencode/opencode.json`
- **Codex CLI** — `~/.codex/config.yaml`
- **Zed** — `~/.config/zed/settings.json`
- **Continue.dev** — `~/.continue/config.json`

The key environment variable is `MEMORY_PATH` — set it to where you want the knowledge graph stored.

### Option 4: Install from source

```bash
git clone https://github.com/tm42/mnemograph.git
cd mnemograph
uv sync

# Add to Claude Code (or adapt for your MCP client)
claude mcp add --scope user mnemograph \
  -e MEMORY_PATH="$HOME/.claude/memory" \
  -- uv run --directory /path/to/mnemograph mnemograph
```

## Usage

### MCP Tools (used by any agent)

Mnemograph exposes these tools via MCP:

**Core Operations:**

| Tool | Description |
|------|-------------|
| `remember` | **Primary storage**: Store knowledge atomically (entity + observations + relations in one call) |
| `recall` | **Primary retrieval**: Get relevant context with auto token management. Use `focus=['Entity']` for full details. Default output is human-readable prose. |
| `create_entities` | Create entities (auto-blocks duplicates >80% match) |
| `create_relations` | Link entities with typed edges (implements, uses, decided_for, etc.) |
| `add_observations` | Add facts/notes to existing entities |
| `read_graph` | Get the full knowledge graph (warning: may be large) |
| `delete_entities` | Remove entities (cascades to relations) |
| `delete_relations` | Remove specific relations |
| `delete_observations` | Remove specific observations |

**Session Lifecycle:**

| Tool | Description |
|------|-------------|
| `session_start` | Signal session start, get initial context. Returns quick_start guide. |
| `session_end` | Signal session end, optionally save summary |
| `get_primer` | Get oriented with the knowledge graph (call at session start) |

**Branching (Parallel Workstreams):**

| Tool | Description |
|------|-------------|
| `create_branch` | Create a named branch for isolated work (e.g., "feature/auth-refactor") |
| `switch_branch` | Switch to a different branch |
| `list_branches` | List all branches |
| `merge_branch` | Merge a branch into main |
| `delete_branch` | Delete a branch |
| `get_current_branch` | Get the current branch name |

**Graph Maintenance:**

| Tool | Description |
|------|-------------|
| `find_similar` | Find entities with similar names (duplicate detection) |
| `find_orphans` | Find entities with no relations |
| `merge_entities` | Merge duplicate entities (consolidates observations, redirects relations) |
| `get_graph_health` | Assess graph quality: orphans, duplicates, overloaded entities |
| `suggest_relations` | Suggest potential relations based on semantic similarity |
| `create_entities_force` | Create entities bypassing duplicate check |
| `clear_graph` | Clear all entities/relations (event-sourced, can rewind) |

**Time Travel:**

| Tool | Description |
|------|-------------|
| `get_state_at` | View graph state at any point in history |
| `diff_timerange` | Show what changed between two points in time |
| `get_entity_history` | Full changelog for a specific entity |
| `rewind` | Rewind graph to a previous state using git |
| `restore_state_at` | Restore graph to state at timestamp (audit-preserving) |
| `reload` | Reload graph state from disk (after git operations) |

**Edge Weights:**

| Tool | Description |
|------|-------------|
| `get_relation_weight` | Get weight breakdown (recency, co-access, explicit) |
| `set_relation_importance` | Set explicit importance weight (0.0-1.0) |
| `get_strongest_connections` | Find entity's most important connections |
| `get_weak_relations` | Find pruning candidates (low-weight relations) |

### Recall: Prose vs Graph Format

The `recall` tool returns context in **prose format by default** — human-readable text that agents can consume directly without parsing JSON:

```python
# Default: prose format (human-readable)
recall(depth="medium", query="authentication")
# Returns:
# **MyApp** (project)
# A Python web service. Uses OAuth2 for user auth.
# Uses: PostgreSQL, Redis
#
# **Decisions:**
# • Decision: Use JWT — Stateless tokens for API authentication
#
# **Gotchas:**
# • Token expiry is 1 hour by default
# • Refresh tokens stored in Redis

# Optional: graph format (structured JSON)
recall(depth="medium", query="authentication", format="graph")
```

**Depth levels:**
- `shallow` — Quick summary: entity counts, recent activity, gotchas
- `medium` — Semantic search + 1-hop neighbors (~2000 tokens)
- `deep` — Multi-hop traversal from focus entities (~5000 tokens)

**Gotcha extraction:** Observations prefixed with `Gotcha:`, `Warning:`, `Note:`, or `Important:` are automatically extracted into a dedicated section.

### CLI Tools

**`mnemograph`** — Unified CLI for all memory operations:

```bash
# Basic operations
mnemograph status                # Show entity/relation counts, recent events
mnemograph log                   # View event history
mnemograph log --session X       # Filter by session
mnemograph sessions              # List all sessions
mnemograph export                # Export graph as JSON

# VCS commands (git-based version control)
mnemograph vcs init              # Initialize memory as git repo
mnemograph vcs commit -m "msg"   # Commit current state
mnemograph vcs log               # View commit history
mnemograph vcs revert --event ID # Undo specific events (compensating events)
mnemograph vcs revert --session X # Undo entire session

# Graph visualization
mnemograph graph                 # Open interactive graph viewer
mnemograph graph --watch         # Live reload mode (refresh button)

# Time travel
mnemograph show --at "2 days ago"  # View state at a point in time
mnemograph diff "1 week ago"       # Show changes since then
mnemograph history "EntityName"    # Full changelog for an entity
mnemograph rewind -n 1             # Git-based rewind by N commits
mnemograph restore --to "yesterday" # Event-based restore (audit-preserving)

# Graph health and maintenance
mnemograph health                # Show graph health report (orphans, duplicates, etc.)
mnemograph health --fix          # Interactive cleanup mode
mnemograph similar "React"       # Find entities similar to "React" (duplicate check)
mnemograph orphans               # List entities with no relations
mnemograph suggest "FastAPI"     # Suggest relations for an entity
mnemograph clear                 # Clear all entities and relations (with confirmation)

# Global options (come *before* the subcommand)
mnemograph --global status       # Use global memory (~/.claude/memory)
mnemograph --memory-path /path graph  # Custom memory location
```

**Running from anywhere** (without activating the venv):

```bash
# Using uv (recommended)
uv run --directory /path/to/mnemograph mnemograph graph

# Using uvx (if installed from PyPI)
uvx --from mnemograph mnemograph status
```

**Graph Visualization** — Interactive D3.js viewer:

- **Layout algorithms**: Force-directed, Radial (hubs at center), Clustered (by component)
- **Color modes**: By entity type, connected component, or degree centrality
- **Edge weight slider**: Filter connections by strength
- **Live refresh**: `--watch` mode with Refresh button for real-time updates

## Architecture

```
~/.mnemograph/memory/    # or ~/.claude/memory, ~/.opencode/memory, etc.
├── mnemograph.db        # SQLite database (events + vectors)
├── state.json           # Cached materialized state (derived)
└── .git/                # Version history
```

**Event sourcing** means all changes are recorded as immutable events in SQLite. The current state is computed by replaying events. This enables:

- Full history of all changes
- Revert any operation
- Branch/merge knowledge graphs
- Audit trail of what Claude learned and when

**Two-layer versioning:**
- `mnemograph vcs revert` — fine-grained, undo specific events via compensating events
- `mnemograph rewind` / `mnemograph restore` — coarse-grained, git-level or timestamp-based restore

## Branching

Branches let you work on isolated knowledge without affecting the main graph. Perfect for:

- **Exploratory work** — try approaches without polluting shared knowledge
- **Feature-specific context** — "feature/auth-refactor" keeps auth decisions separate
- **Multiple projects** — switch context between different codebases

### Creating and Using Branches

```python
# Create a branch for your feature
create_branch(name="feature/auth-refactor")

# Work normally — all operations happen on this branch
remember(name="OAuth2", entity_type="concept",
         observations=["Implementing OAuth2 flow"])

# Switch back to main to see clean state
switch_branch(name="main")

# Merge when ready
merge_branch(source="feature/auth-refactor", target="main")
```

### How Branching Works

- **Main branch** always exists, contains shared knowledge
- **Feature branches** inherit from main but additions stay isolated
- **Automatic filtering** — `recall`, `search`, etc. only see current branch + main
- **Merge** copies branch entities/relations into target branch
- **Delete** cleans up after merge (or abandons exploratory work)

### Branch Naming Conventions

| Pattern | Use Case |
|---------|----------|
| `feature/xyz` | Feature-specific knowledge |
| `explore/xyz` | Exploratory/experimental work |
| `project/xyz` | Project-specific context |
| `user/name` | Personal workspace |

## Entity Types

| Type | Purpose | Example |
|------|---------|---------|
| `concept` | Ideas, patterns, approaches | "Repository pattern", "Event sourcing" |
| `decision` | Choices with rationale | "Chose SQLite over Postgres for simplicity" |
| `project` | Codebases, systems | "auth-service", "mnemograph" |
| `pattern` | Recurring code patterns | "Error handling with Result type" |
| `question` | Open unknowns | "Should we add real-time sync?" |
| `learning` | Discoveries | "pytest fixtures simplify test setup" |
| `entity` | Generic (people, files, etc.) | "Alice", "config.yaml" |

## Topic Convention

Use **topic entities** as entry points for browsing related knowledge:

```python
# Create topic entry points
create_entities([
    {"name": "topic/projects", "entityType": "entity"},
    {"name": "topic/decisions", "entityType": "entity"},
    {"name": "topic/patterns", "entityType": "entity"},
])

# Link entities to their topics
create_relations([
    {"from": "auth-service", "to": "topic/projects", "relationType": "part_of"},
    {"from": "Decision: Use Redis", "to": "topic/decisions", "relationType": "part_of"},
])
```

**Standard topics:**
- `topic/projects` — Project entities
- `topic/decisions` — Architectural decisions
- `topic/patterns` — Patterns and practices
- `topic/learnings` — Key discoveries
- `topic/questions` — Open questions

This makes it easy to query "what decisions have we made?" by exploring `topic/decisions`.

## Development

```bash
git clone https://github.com/tm42/mnemograph.git
cd mnemograph
uv sync                    # Install dependencies
uv run pytest --cov        # Run tests with coverage (enforces 75% minimum)
uv run ruff check .        # Lint
uv run mnemograph          # Run MCP server directly
```
## Based On

Mnemograph builds on [MCP server-memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) — Anthropic's official memory server

## License

MIT
