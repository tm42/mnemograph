# Mnemograph Claude Code Plugin

A Claude Code hook plugin that provides persistent memory across sessions using the [mnemograph](https://github.com/tm42/mnemograph) knowledge graph.

## Features

- **SessionStart hook**: Automatically injects relevant memory context at the start of each session
- **Stop hook**: Reminds you to save learnings at session end
- **Auto-commit** (optional): Automatically commits memory changes when session ends
- **memory-store agent**: Quality-enforced knowledge storage with deduplication
- **memory-init agent**: Fast session briefings using Haiku

## Installation

### Prerequisites

1. [mnemograph](https://github.com/tm42/mnemograph) must be installed and configured
2. A memory directory must exist (either project-local `.claude/memory/` or global `~/.claude/memory/`)

### Install the plugin

```bash
# Clone or copy this directory
git clone https://github.com/tm42/mnemograph.git
cd mnemograph/mnemograph-claude-code

# Add to Claude Code plugins
claude plugins add /path/to/mnemograph-claude-code
```

Or symlink to your plugins directory:

```bash
ln -s /path/to/mnemograph/mnemograph-claude-code ~/.claude/plugins/mnemograph
```

## Configuration

Create `.claude/mnemograph-hook.yaml` in your project to configure behavior:

```yaml
# Auto-commit memory changes at session end (default: false)
auto_commit: false
```

## How It Works

### SessionStart Hook

When a Claude Code session starts, this hook:

1. Detects the project from the working directory
2. Checks for mnemograph memory (project-local first, then global)
3. Retrieves a shallow memory context via `recall()`
4. Injects the context into the session

The context appears as:

```xml
<mnemograph-context project="my-project">
[Memory summary, recent entities, gotchas...]
</mnemograph-context>
```

### Stop Hook

When a session ends normally, this hook:

1. Outputs a reminder to save any learnings
2. Optionally auto-commits memory changes (if configured)

## Agents

### memory-store

A dedicated subagent (Haiku) that handles all knowledge storage with quality enforcement:

- **Deduplication**: Checks `find_similar()` before every create
- **Canonical naming**: Enforces "Decision: X", proper casing, no articles
- **Auto-relations**: Calls `suggest_relations()` and creates obvious connections
- **Importance levels**: Supports `low|normal|high` for relation weighting

**Usage via command:**
```
/remember we decided to use JWT for auth
```

**Direct invocation for batch stores:**
```xml
<store-request>
  <item content="chose SQLite for simplicity" type_hint="decision"/>
  <item content="gotcha: WAL doesn't work on network drives" type_hint="learning" related_to="Decision: Use SQLite"/>
</store-request>
```

**Background storage (non-blocking):**
```python
Task(
  subagent_type="memory-store",
  run_in_background=True,
  prompt="<store-request><item content=\"...\"/></store-request>"
)
```

### memory-init

Fast session briefing agent that summarizes the knowledge graph at session start. Outputs a compact XML briefing with entity counts and key highlights.

## Memory Detection

The plugin looks for memory in this order:

1. **Project-local**: `./.claude/memory/` (relative to cwd)
2. **MEMORY_PATH**: Environment variable if set
3. **Global**: `~/.claude/memory/`

## Troubleshooting

**No context injected at session start?**
- Check that mnemograph is installed: `pip show mnemograph`
- Check that a memory directory exists with `events.jsonl`
- The hook fails silently if mnemograph is unavailable

**Timeouts?**
- SessionStart has a 10-second timeout
- Stop hook has a 5-second timeout
- If your memory graph is large, context retrieval may time out

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Test the hooks manually
echo '{"cwd": "/path/to/project"}' | python hooks/session_start.py
echo '{"cwd": "/path/to/project", "stopReason": "end_turn"}' | python hooks/stop_hook.py
```

## License

MIT
