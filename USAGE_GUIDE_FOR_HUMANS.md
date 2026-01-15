# Mnemograph Usage Guide for Humans

A quick reference for reviewing and curating your AI's memory.

## Daily Commands

```bash
# Check what's in memory
mnemograph status

# See recent changes
mnemograph log --since "1 hour ago"
mnemograph log --since yesterday

# Visualize the graph
mnemograph graph              # Quick view (auto-closes)
mnemograph graph --watch      # Live mode with refresh button
```

## Review What the Agent Learned

```bash
# What changed today?
mnemograph log --since today

# What did a specific session add?
mnemograph log --session <session-id>

# See full history of an entity
mnemograph show "entity-name" --history

# Compare states
mnemograph diff "yesterday"
mnemograph diff "last week"
```

## Curate and Clean Up

```bash
# Health check — find problems
mnemograph health

# Interactive cleanup mode
mnemograph health --fix

# Find duplicates
mnemograph similar "entity-name"

# Find orphans (unconnected entities)
mnemograph orphans

# Clear everything (with confirmation)
mnemograph clear
```

## Undo Mistakes

```bash
# Quick undo (git-based)
mnemograph rewind              # Undo last change
mnemograph rewind --steps 3    # Go back 3 commits

# Restore to a point in time (preserves audit trail)
mnemograph restore --to "2 hours ago"
mnemograph restore --to "2025-01-10T14:00"

# Preview before restoring
mnemograph show --at "yesterday"
```

## Memory Scope

```bash
# Project-local memory (default)
mnemograph status

# Global memory (cross-project)
mnemograph --global status
mnemograph --global graph

# Custom location
mnemograph --memory-path ~/shared-memory status
```

## When to Intervene

| Symptom | Action |
|---------|--------|
| Too many similar entities | `mnemograph health --fix` → merge duplicates |
| Orphan entities piling up | Connect them to topics or delete |
| Outdated information | Add observations or delete entity |
| Agent storing noise | Review and provide feedback on what to skip |
| Graph feels cluttered | Visualize with `mnemograph graph`, prune weak relations |

## Visualization Tips

- **Force layout**: Good for seeing overall structure
- **Radial layout**: Puts hubs (connected entities) at center
- **Cluster layout**: Groups by connected components
- **Weight slider**: Filter out weak connections to reduce noise
- **Color by type**: See distribution of concepts/decisions/patterns
- **Color by component**: Spot isolated clusters

## Trust but Verify

The agent proposes knowledge; you curate it.

- Run `mnemograph log` periodically to see what's being stored
- Use `mnemograph health` weekly to catch issues early
- Visualize with `mnemograph graph` to understand structure
- Rewind/restore if something goes wrong

Memory is a collaboration. Your oversight keeps it focused and accurate.
