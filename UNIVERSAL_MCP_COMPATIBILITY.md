# Mnemograph: Universal MCP Compatibility

## Goal

Make mnemograph useful for any MCP-compatible agent, not just Claude Code.

**Target clients**: Claude Code, opencode, codex CLI, Zed, Continue.dev, Cursor, and any future MCP-compatible tools.

## Design Principles

1. **MCP server is agent-agnostic** — works with any MCP client
2. **CLI naming is universal** — `mnemograph` primary command
3. **Agent-specific integrations are separate packages** — e.g., `mnemograph-claude-code` plugin
4. **No dumbed-down exports** — graph structure is the value, don't flatten it

---

## MCP Client Configurations

### Claude Code

```bash
# Global (all projects)
claude mcp add --scope user mnemograph \
  -e MEMORY_PATH="$HOME/.claude/memory" \
  -- uvx mnemograph

# Per-project
claude mcp add mnemograph \
  -e MEMORY_PATH=".claude/memory" \
  -- uvx mnemograph
```

Or in `~/.config/claude-code/settings.json`:
```json
{
  "mcpServers": {
    "mnemograph": {
      "command": "uvx",
      "args": ["mnemograph"],
      "env": {
        "MEMORY_PATH": "~/.claude/memory"
      }
    }
  }
}
```

### opencode

In `~/.config/opencode/opencode.json` or project `opencode.json`:
```json
{
  "mcp": {
    "mnemograph": {
      "type": "local",
      "command": ["uvx", "mnemograph"],
      "environment": {
        "MEMORY_PATH": "/Users/yourname/.opencode/memory"
      },
      "enabled": true
    }
  }
}
```

**Note**: Use `environment` (not `env`) and full paths (not `~`) for `MEMORY_PATH`.

### Codex CLI

```bash
# Add MCP server
codex mcp add mnemograph -- uvx mnemograph

# Or via config file
```

In `~/.codex/config.yaml`:
```yaml
mcp_servers:
  mnemograph:
    command: uvx
    args: [mnemograph]
    env:
      MEMORY_PATH: ~/.codex/memory
```

### Zed Editor

In `~/.config/zed/settings.json`:
```json
{
  "context_servers": {
    "mnemograph": {
      "command": {
        "path": "uvx",
        "args": ["mnemograph"]
      },
      "env": {
        "MEMORY_PATH": "~/.zed/memory"
      }
    }
  }
}
```

### Continue.dev

In `~/.continue/config.json`:
```json
{
  "mcpServers": [
    {
      "name": "mnemograph",
      "command": "uvx",
      "args": ["mnemograph"],
      "env": {
        "MEMORY_PATH": "~/.continue/memory"
      }
    }
  ]
}
```

### Generic MCP Configuration

For any MCP-compatible client, use stdio transport:

```json
{
  "name": "mnemograph",
  "transport": "stdio",
  "command": "uvx",
  "args": ["mnemograph"],
  "env": {
    "MEMORY_PATH": "/path/to/memory"
  }
}
```

**Alternative installation methods**:
```bash
# If uvx isn't available
pip install mnemograph
# Then use: python -m mnemograph

# From source
uv run --directory /path/to/mnemograph mnemograph
```

---

## CLI Naming

### Primary Commands

```bash
# Full name (primary)
mnemograph status
mnemograph recall "auth"
mnemograph branch create project/api

# Custom memory location (global options before subcommand)
mnemograph --memory-path ~/.opencode/memory status
mnemograph --memory-path ~/.opencode/memory graph
```

### Implementation

```toml
# pyproject.toml

[project.scripts]
mnemograph-server = "mnemograph.server:main"  # MCP server entry point
mnemograph = "mnemograph.cli:cli"             # Primary CLI
```

---

## Universal MCP Tools

### Tool Description Guidelines

Tool descriptions should work well across different LLM backends:

1. **Keep descriptions under 200 characters** — some clients truncate
2. **Front-load the key action** — "Search entities by..." not "This tool allows you to search..."
3. **Include return type hints** — "Returns list of matching entities"
4. **Avoid Claude-specific terminology** — "knowledge graph" not "memory system"

**Current tools** (already in mnemograph):
| Tool | Description |
|------|-------------|
| `remember` | Store knowledge atomically (entity + observations + relations) |
| `recall` | Get context at shallow/medium/deep levels |
| `create_entities` | Create nodes in the knowledge graph |
| `open_nodes` | Get full data for specific entities |
| `find_similar` | Check for duplicates before creating |
| `read_graph` | Return full graph state |
| `get_state_at` | Time travel: view graph at any timestamp |
| `diff_timerange` | Show changes between two times |

### `get_primer` — Agent Orientation

Any agent can call this at session start to understand what's available:

```python
@tool
def get_primer() -> dict:
    """
    Get oriented with this project's knowledge graph.
    
    Call at session start to understand:
    - What knowledge is available
    - Current branch context
    - Available tools
    
    Works with any MCP-compatible agent.
    """
    state = engine.get_state()
    branch = engine.branch_manager.current_branch()
    
    return {
        "project": detect_project_name(),
        "status": {
            "entity_count": len(state.entities),
            "relation_count": len(state.relations),
            "current_branch": branch.name,
            "branch_description": branch.description,
        },
        "recent_activity": get_recent_entities(limit=5),
        "tools": {
            "retrieval": [
                "recall(depth, query) - Get relevant context at varying depths",
                "open_nodes(names) - Get full data for specific entities",
                "find_similar(name) - Check for duplicates before creating",
            ],
            "creation": [
                "create_entity(name, type, observations) - Store new knowledge",
                "add_observation(entity, observation) - Add info to existing entity",
                "create_relation(from, to, type) - Link entities",
            ],
            "branches": [
                "branch_list() - List all branches",
                "branch_create(name, seeds) - Create focused branch",
                "branch_checkout(name) - Switch context",
            ],
            "history": [
                "get_state_at(timestamp) - Time travel",
                "diff_timerange(start, end) - See changes",
            ],
        },
        "quick_start": "Call recall(depth='shallow') for quick summary, or recall(depth='medium', query='topic') for specific context.",
    }
```

### `session_start` / `session_end` — Universal Lifecycle

For agents that can run shell commands as hooks:

```python
@tool
def session_start(project_hint: str = None) -> dict:
    """
    Signal session start and get initial context.
    
    Call this when beginning work. Returns context to prime the session.
    
    Args:
        project_hint: Optional project name or path for context
    """
    context = get_project_context(project_hint)
    
    return {
        "context": format_context(context),
        "current_branch": engine.branch_manager.current_branch_name(),
        "tip": "Use recall(depth='medium', query=topic) for specific topics.",
    }

@tool  
def session_end(summary: str = None) -> dict:
    """
    Signal session end, optionally save summary.
    
    Args:
        summary: Optional session summary to store
    """
    if summary:
        # Store as observation on project entity
        project = detect_project_name()
        add_observation(project, f"Session summary: {summary}")
    
    return {
        "status": "session_ended",
        "tip": "Consider storing key learnings with create_entity or add_observation.",
    }
```

### CLI Equivalents

```bash
# For agents that prefer shell over MCP
mnemograph session start
mnemograph session start --project my-project

mnemograph session end
mnemograph session end --summary "Implemented auth refresh"
```

---

## Agent Integration Tiers

### Tier 1: Native MCP (Best Experience)
- Agent calls MCP tools directly
- Full graph traversal
- Real-time queries
- **Examples**: Claude Code, Continue.dev, others with MCP support

### Tier 2: CLI Wrapper
- Agent runs shell commands
- Calls `mnemograph` CLI
- Still full functionality
- **Examples**: Agents with shell access but no MCP

### Tier 3: Manual Integration
- User manually runs `mnemograph session start`
- Pastes context into conversation
- Works but not seamless
- **Examples**: Web-based agents, limited tool access

---

## Agent-Specific Plugins

For agents that support plugins/extensions, we can build deeper integrations:

| Agent | Plugin | Features |
|-------|--------|----------|
| Claude Code | `mnemograph-claude-code` | SessionStart hook, recall detection, Stop prompts |
| Continue.dev | `mnemograph-continue` | Context provider, slash commands |
| Cursor | `mnemograph-cursor` | TBD based on their extension API |

These are **separate packages** that depend on core `mnemograph`.

### Claude Code Plugin (Primary)

See `MNEMOGRAPH_HOOK_PLUGIN_SPEC.md` for full design:
- SessionStart: Auto-inject context
- UserPromptSubmit: Detect recall patterns
- Stop: Prompt to save learnings

---

## What We're NOT Doing

### No Static File Export

We considered:
```bash
mnemograph export-context > .mnemograph-context.md  # NO
```

**Why not**:
- Breaks graph structure (the whole point)
- Static snapshot misses updates
- Encourages treating memory as flat text
- Wait for Phase 5 (synthesis via local model) for intelligent summarization

### No Agent Auto-Detection

Agents should explicitly identify themselves if they want special treatment:
```python
# In MCP handshake or tool call
recall(depth, query, agent_hint="claude-code")
```

Simpler, more predictable.

---

## Compatibility Testing

### Test Matrix

| Client | MCP Version | Transport | Status | Notes |
|--------|-------------|-----------|--------|-------|
| Claude Code | 1.25+ | stdio | ✅ Primary | Full integration |
| opencode | TBD | stdio | 🔄 Testing | Awaiting confirmation |
| codex CLI | TBD | stdio | 🔄 Testing | Awaiting confirmation |
| Zed | TBD | stdio | 🔄 Testing | Context server API |
| Continue.dev | 1.0+ | stdio | 🔄 Testing | Slash commands work |
| Cursor | N/A | - | ⏳ Future | Waiting for MCP support |

### Quick Compatibility Test

Run this after configuring any MCP client:

```bash
# 1. Verify server starts
uvx mnemograph --help

# 2. Test MCP tools via CLI (any client should produce similar output)
# In your agent, try:
#   - recall with depth="shallow"
#   - remember with name, entity_type, observations
#   - find_similar with name to check for duplicates
#   - open_nodes with entity names
```

### Known Differences Between Clients

1. **Tool description parsing**: Some clients truncate long descriptions. Keep tool descriptions under 200 chars.

2. **Environment variable expansion**: Not all clients expand `~`. Use full paths when in doubt:
   ```json
   "MEMORY_PATH": "/Users/username/.agent/memory"
   ```

3. **Startup behavior**: Some clients call `list_tools` lazily. First tool call may have slight delay.

4. **JSON schema validation**: Stricter clients may reject optional parameters. Mnemograph uses sensible defaults.

---

## Implementation Checklist

### Core (in mnemograph package)
- [x] CLI entry points (`mnemograph`) — Done
- [x] MCP client configurations documented — Done
- [x] Universal project descriptions — Done
- [x] Add `get_primer` MCP tool — Done
- [x] Add `session_start` / `session_end` MCP tools — Done
- [x] Add `mnemograph session-start/session-end/primer` CLI commands — Done
- [x] CLI consolidation — merged into unified `mnemograph` CLI with `vcs` subgroup

### Claude Code Plugin (separate package: mnemograph-claude-code)
- [ ] SessionStart hook
- [ ] UserPromptSubmit hook (recall detection)
- [ ] Stop hook (save reminder)
- [ ] Plugin packaging for marketplace

### Future Agent Plugins
- [ ] Continue.dev integration
- [ ] Others based on demand

---

## Summary

```
┌─────────────────────────────────────────────────────────┐
│                     mnemograph (core)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ MCP Server  │  │    CLI      │  │   Git VCS       │ │
│  │ (universal) │  │  mnemograph  │ │   Integration   │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                         │                               │
│         get_primer, session_start/end                   │
│              (agent-agnostic tools)                     │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ mnemograph- │ │ mnemograph- │ │   future    │
   │ claude-code │ │  continue   │ │   plugins   │
   │  (plugin)   │ │  (plugin)   │ │             │
   └─────────────┘ └─────────────┘ └─────────────┘
```

Core is universal. Plugins add agent-specific magic.
