# Mnemograph Claude Code Plugin

A Claude Code hook plugin that provides persistent memory across sessions using the [mnemograph](https://github.com/tm42/mnemograph) knowledge graph.

## Features

- **SessionStart hook**: Automatically injects relevant memory context at the start of each session
- **Stop hook**: Reminds you to save learnings at session end
- **Auto-commit** (optional): Automatically commits memory changes when session ends

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
