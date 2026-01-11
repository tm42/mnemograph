# Mnemograph: Philosophy for Agents

> Add this to your project's CLAUDE.md (or equivalent) to guide agent behavior with mnemograph.

---

## What Memory Is For

Memory is **durable collaborative knowledge** — not a log of what happened, but a distillation of what matters.

Think of it as a shared notebook between you and the human, where you write down:
- **Decisions** that shaped the codebase (with rationale)
- **Patterns** that work well here
- **Questions** still open
- **Learnings** discovered together

## What Memory Is NOT For

- Chat history or session summaries
- Temporary task state ("currently working on X")
- Obvious facts discoverable from code
- Every file or function you touched

## The Core Question

Before storing anything, ask:

> "Would future-me — in a fresh session, with no context — benefit from knowing this?"

If yes, store it. If not, skip it.

## How to Store Well

1. **Be specific**: "auth/jwt-validation-pattern" not "JWT stuff"
2. **Include rationale**: "Chose X because Y" is 10x more valuable than "Using X"
3. **Keep observations atomic**: One fact per observation, not paragraphs
4. **Connect everything**: Orphan entities won't be found. Link to topics.
5. **Check for duplicates**: `find_similar()` before creating

## Entity Types — When to Use What

| Type | Use When |
|------|----------|
| `decision` | Recording a choice with WHY it was made |
| `pattern` | A recurring approach that works |
| `concept` | An idea or principle (not code-specific) |
| `project` | A codebase or system you're working with |
| `question` | Something unknown, to revisit later |
| `learning` | A discovery — "TIL this works better" |

## Session Rhythm

**Start**:
```
session_start() → read quick_start guide → recall(depth='shallow')
```

**During**:
- Store decisions as you make them (with rationale!)
- Note patterns you discover
- Add observations to existing entities rather than creating duplicates
- When results are large, recall returns structure-only — use `open_nodes(['entity1', 'entity2'])` to expand

**Periodically**:
- `get_graph_health()` to spot issues
- Merge duplicates, connect orphans

## Memory Scope

On first setup for a project, **ask the human**:

> "Should memory be project-local (stays in this repo) or global (shared across projects)?"

- **Project-local** (.claude/memory): Repo-specific architecture, decisions
- **Global** (~/.claude/memory): Personal learnings, universal patterns

## The Human's Role

Memory is a collaboration. The human can:
- Review what you stored (`mg log`)
- Curate and clean up (`mg health --fix`)
- Undo mistakes (`mg rewind`)
- Visualize structure (`mg graph`)

Your job is to **propose knowledge**. Their job is to **curate** it.

## Anti-Patterns to Avoid

- Creating entities for every file you read
- Storing conversation summaries
- Vague names ("notes", "misc", "stuff")
- Orphan entities with no connections
- Duplicate entities without checking

## Quick Reference

```
# Get oriented
session_start()
recall(depth='shallow')

# Search and explore
recall(depth='medium', query=query)            # semantic search + neighbors
open_nodes(['entity1', 'entity2'])             # full data for specific entities

# Store knowledge
remember(name, type, observations, relations)  # atomic creation
add_observations(entity, [...])                # extend existing

# Before creating
find_similar(name)  # check for duplicates (>80% blocks)

# Maintenance
get_graph_health()  # periodic check
merge_entities(source, target)  # consolidate duplicates
```

---

*This guide lives in the memory itself as `topic/mnemograph-guide`. Explore with `recall(depth='medium', query='mnemograph')`.*
