# Mnemograph Usage Guide for Humans

A quick reference for reviewing and curating your AI's memory.

## Daily Commands

```bash
# Check what's in memory
mg status

# See recent changes
mg log --since "1 hour ago"
mg log --since yesterday

# Visualize the graph
mg graph              # Quick view (auto-closes)
mg graph --watch      # Live mode with refresh button
```

## Review What the Agent Learned

```bash
# What changed today?
mg log --since today

# What did a specific session add?
mg log --session <session-id>

# See full history of an entity
mg show "entity-name" --history

# Compare states
mg diff "yesterday"
mg diff "last week"
```

## Curate and Clean Up

```bash
# Health check — find problems
mg health

# Interactive cleanup mode
mg health --fix

# Find duplicates
mg similar "entity-name"

# Find orphans (unconnected entities)
mg orphans

# Clear everything (with confirmation)
mg clear
```

## Undo Mistakes

```bash
# Quick undo (git-based)
mg rewind              # Undo last change
mg rewind --steps 3    # Go back 3 commits

# Restore to a point in time (preserves audit trail)
mg restore --to "2 hours ago"
mg restore --to "2025-01-10T14:00"

# Preview before restoring
mg show --at "yesterday"
```

## Memory Scope

```bash
# Project-local memory (default)
mg status

# Global memory (cross-project)
mg --global status
mg --global graph

# Custom location
mg --memory-path ~/shared-memory status
```

## When to Intervene

| Symptom | Action |
|---------|--------|
| Too many similar entities | `mg health --fix` → merge duplicates |
| Orphan entities piling up | Connect them to topics or delete |
| Outdated information | Add observations or delete entity |
| Agent storing noise | Review and provide feedback on what to skip |
| Graph feels cluttered | Visualize with `mg graph`, prune weak relations |

## Visualization Tips

- **Force layout**: Good for seeing overall structure
- **Radial layout**: Puts hubs (connected entities) at center
- **Cluster layout**: Groups by connected components
- **Weight slider**: Filter out weak connections to reduce noise
- **Color by type**: See distribution of concepts/decisions/patterns
- **Color by component**: Spot isolated clusters

## Trust but Verify

The agent proposes knowledge; you curate it.

- Run `mg log` periodically to see what's being stored
- Use `mg health` weekly to catch issues early
- Visualize with `mg graph` to understand structure
- Rewind/restore if something goes wrong

Memory is a collaboration. Your oversight keeps it focused and accurate.
