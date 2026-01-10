# Mnemograph

<!-- mcp-name: io.github.tm42/mnemograph -->

A persistent, event-sourced knowledge graph for Claude Code. Unlike simple key-value memory, Mnemograph captures **entities**, **relations**, and **observations** — enabling semantic search, tiered context retrieval, and git-based version control of your AI's memory.

## Why Mnemograph?

Claude Code sessions are ephemeral. Mnemograph gives your AI partner persistent memory that:

- **Survives across sessions** — decisions, patterns, learnings persist
- **Supports semantic search** — find relevant context by meaning, not just keywords
- **Provides tiered retrieval** — shallow summaries to deep subgraphs based on need
- **Versions like code** — branch, commit, diff, revert your knowledge graph
- **Enables collaboration** — share memory repos across users or projects

## Installation

```bash
# Install from PyPI
pip install mnemograph

# Or install from source
git clone https://github.com/tm42/mnemograph.git
cd mnemograph
uv sync  # or: pip install -e .

# Initialize memory (creates ~/.claude/memory/)
claude-mem init
```

### Configure Claude Code

Add to your MCP settings (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "mnemograph": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mnemograph", "mnemograph"],
      "env": {
        "MEMORY_PATH": "/Users/YOU/.claude/memory"
      }
    }
  }
}
```

## Usage

### MCP Tools (used by Claude)

Mnemograph exposes these tools to Claude Code:

| Tool | Description |
|------|-------------|
| `create_entities` | Create nodes: concepts, decisions, patterns, projects, questions, learnings |
| `create_relations` | Link entities with typed edges (implements, uses, decided_for, etc.) |
| `add_observations` | Add facts/notes to existing entities |
| `delete_entities` | Remove entities (cascades to relations) |
| `delete_relations` | Remove specific relations |
| `delete_observations` | Remove specific observations |
| `read_graph` | Get the full knowledge graph |
| `search_nodes` | Text search across names and observations |
| `open_nodes` | Get specific entities with their relations |
| `search_semantic` | Vector similarity search (meaning-based) |
| `memory_context` | Tiered retrieval: shallow (summary), medium (search+neighbors), deep (subgraph) |

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

**`claude-mem`** — Git-based version control:

```bash
claude-mem init                  # Initialize memory as git repo
claude-mem status                # Show uncommitted changes
claude-mem commit -m "message"   # Commit current state
claude-mem commit -m "msg" -a    # Commit with auto-summary
claude-mem log                   # View commit history
claude-mem log --oneline         # Compact commit log
```

## Architecture

```
~/.claude/memory/
├── events.jsonl    # Append-only event log (source of truth)
├── state.json      # Cached materialized state (derived)
├── vectors.db      # Semantic search index (derived)
└── .git/           # Version history
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

## Development

```bash
uv sync                    # Install dependencies
uv run pytest              # Run tests (56 tests)
uv run ruff check .        # Lint
uv run mnemograph          # Run MCP server directly
```

## Based On

Mnemograph builds on ideas from:
- [MCP server-memory](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) — Anthropic's official memory server (baseline)
- [Mem0](https://github.com/mem0ai/mem0) — extraction/consolidation patterns
- [Graphiti](https://github.com/getzep/graphiti) — bi-temporal modeling inspiration
- Event sourcing principles — append-only logs, state materialization

## License

MIT
