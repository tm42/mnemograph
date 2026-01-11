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
mg --global status
mg --global graph
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

| Tool | Description |
|------|-------------|
| `remember` | **Primary storage**: Store knowledge atomically (entity + observations + relations in one call) |
| `recall` | **Primary retrieval**: Get relevant context with auto token management (shallow/medium/deep). Returns structure-only if results too large. |
| `open_nodes` | Get full data for specific entities (after recall) |
| `create_entities` | Create entities (auto-blocks duplicates >80% match) |
| `create_relations` | Link entities with typed edges (implements, uses, decided_for, etc.) |
| `add_observations` | Add facts/notes to existing entities |
| `read_graph` | Get the full knowledge graph (warning: may be large) |
| `delete_entities` | Remove entities (cascades to relations) |
| `delete_relations` | Remove specific relations |
| `delete_observations` | Remove specific observations |
| `find_similar` | Find entities with similar names (duplicate detection) |
| `find_orphans` | Find entities with no relations |
| `merge_entities` | Merge duplicate entities (consolidates observations, redirects relations) |
| `get_graph_health` | Assess graph quality: orphans, duplicates, overloaded entities |
| `suggest_relations` | Suggest potential relations based on semantic similarity |
| `get_state_at` | Time travel: view graph state at any point in history |
| `diff_timerange` | Show what changed between two points in time |
| `get_entity_history` | Full changelog for a specific entity |
| `get_relation_weight` | Get weight breakdown (recency, co-access, explicit) |
| `set_relation_importance` | Set explicit importance weight (0.0-1.0) |
| `get_strongest_connections` | Find entity's most important connections |
| `get_weak_relations` | Find pruning candidates (low-weight relations) |
| `clear_graph` | Clear all entities/relations (event-sourced, can rewind) |
| `create_entities_force` | Create entities bypassing duplicate check |

### CLI Tools

**`mnemograph-cli`** — Event-level operations:

```bash
mnemograph-cli status              # Show entity/relation counts, recent events
mnemograph-cli log                 # View event history
mnemograph-cli log --session X     # Filter by session
mnemograph-cli revert --event ID   # Undo specific events
mnemograph-cli revert --session X  # Undo entire session
mnemograph-cli export              # Export graph as JSON
```

**`mg`** (or `claude-mem`) — Git-based version control:

```bash
mg init                  # Initialize memory as git repo
mg status                # Show uncommitted changes
mg commit -m "message"   # Commit current state
mg log                   # View commit history
mg graph                 # Open interactive graph viewer
mg graph --watch         # Live reload mode (refresh button)
mg --global graph        # Use global memory (~/.mnemograph/memory)
mg --memory-path ~/.opencode/memory graph  # Custom memory location

# Graph health and maintenance
mg health                # Show graph health report (orphans, duplicates, etc.)
mg health --fix          # Interactive cleanup mode
mg similar "React"       # Find entities similar to "React" (duplicate check)
mg orphans               # List entities with no relations
mg suggest "FastAPI"     # Suggest relations for an entity
mg clear                 # Clear all entities and relations (with confirmation)
mg clear -y -m "reason"  # Clear without confirmation, record reason
```

**Note**: Global options (`--global`, `--memory-path`) come *before* the subcommand.

**Running from anywhere** (without activating the venv):

```bash
# Using uv (recommended)
uv run --directory /path/to/mnemograph mg graph

# Using uvx (if installed from PyPI)
uvx mnemograph-cli status
```

**Graph Visualization** — Interactive D3.js viewer:

- **Layout algorithms**: Force-directed, Radial (hubs at center), Clustered (by component)
- **Color modes**: By entity type, connected component, or degree centrality
- **Edge weight slider**: Filter connections by strength
- **Live refresh**: `--watch` mode with Refresh button for real-time updates

## Architecture

```
~/.mnemograph/memory/    # or ~/.claude/memory, ~/.opencode/memory, etc.
├── events.jsonl         # Append-only event log (source of truth)
├── state.json           # Cached materialized state (derived)
├── vectors.db           # Semantic search index (derived)
└── .git/                # Version history
```

**Event sourcing** means all changes are recorded as immutable events. The current state is computed by replaying events. This enables:

- Full history of all changes
- Revert any operation
- Branch/merge knowledge graphs
- Audit trail of what Claude learned and when

**Two-layer versioning:**
- `mnemograph-cli revert` — fine-grained, undo specific events via compensating events
- `claude-mem commit/revert` — coarse-grained, git-level checkpoints

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
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run mnemograph          # Run MCP server directly
```
## Based On

Mnemograph builds on [MCP server-memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) — Anthropic's official memory server

## License

MIT
