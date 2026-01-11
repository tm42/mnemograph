# Mnemograph Setup Instructions for Claude Code

You are setting up Mnemograph, an event-sourced knowledge graph memory system. Follow these steps exactly.

## Step 1: Add MCP Server

Run this command to register mnemograph as a global MCP server:

```bash
claude mcp add --scope user mnemograph \
  -e MEMORY_PATH="$HOME/.claude/memory" \
  -- uvx mnemograph
```

## Step 2: Create Memory Directory

```bash
mkdir -p ~/.claude/memory
```

## Step 3: Verify Installation

After setup, you should have access to these MCP tools:
- `mcp__mnemograph__remember` — store knowledge atomically
- `mcp__mnemograph__recall` — retrieve context at shallow/medium/deep levels
- `mcp__mnemograph__create_entities`
- `mcp__mnemograph__create_relations`
- `mcp__mnemograph__add_observations`
- `mcp__mnemograph__open_nodes`
- `mcp__mnemograph__find_similar` — check for duplicates before creating
- `mcp__mnemograph__read_graph`
- `mcp__mnemograph__delete_entities`
- `mcp__mnemograph__delete_relations`
- `mcp__mnemograph__delete_observations`

## Step 4: Test the Connection

Try creating a test entity:

```
Use mcp__mnemograph__create_entities to create an entity named "Setup Test" of type "learning" with observation "Mnemograph successfully installed"
```

Then recall it:

```
Use mcp__mnemograph__recall with depth "shallow" to see memory summary
```

## How to Use Mnemograph

### Creating Knowledge

When you learn something important during a session, store it:

```
create_entities([{
  "name": "Descriptive Name",
  "entityType": "concept|decision|project|pattern|question|learning|entity",
  "observations": ["Key fact 1", "Key fact 2"]
}])
```

### Linking Knowledge

Connect related entities:

```
create_relations([{
  "from": "Entity A",
  "to": "Entity B",
  "relationType": "implements|uses|decided_for|part_of|etc"
}])
```

### Retrieving Context

At the start of sessions, get relevant context:

```
recall({ "depth": "shallow" })  # Quick summary (~500 tokens)
recall({ "depth": "medium", "query": "topic" })  # Search + neighbors (~2000 tokens)
recall({ "depth": "deep", "focus": ["Entity"] })  # Full subgraph (~5000 tokens)
```

### Before Creating — Check for Duplicates

```
find_similar({ "name": "React" })  # Returns similar entities with similarity scores
```

## Entity Types

| Type | Use For |
|------|---------|
| `concept` | Ideas, patterns, approaches |
| `decision` | Choices made with rationale |
| `project` | Codebases, systems being worked on |
| `pattern` | Recurring code patterns discovered |
| `question` | Open questions, unknowns |
| `learning` | Things discovered during sessions |
| `entity` | Generic: people, files, configs |

## Best Practices

1. **Be specific** — "Use Result type for error handling in auth module" > "Error handling pattern"
2. **Record decisions with rationale** — Future you will want to know WHY
3. **Link related knowledge** — Relations make retrieval much more powerful
4. **Use appropriate types** — Helps with filtering and organization
5. **Update, don't duplicate** — Add observations to existing entities when appropriate

## Troubleshooting

If tools aren't available:
1. Restart Claude Code to reload MCP servers
2. Check `~/.claude.json` has the mnemograph entry
3. Verify uvx is installed: `which uvx` (install uv if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

If memory is empty:
- Memory starts fresh — that's expected!
- Start adding knowledge as you work
