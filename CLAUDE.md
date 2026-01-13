# Claude Code Memory System — Project Brief

> **For**: Claude Code instance working on this project
> **From**: Claude (claude.ai session) + M (the human)
> **Date**: January 2025

## What We're Building

A **graph-based long-term memory system** for Claude Code that:

1. Persists knowledge across sessions (projects, decisions, patterns, learnings)
2. Uses **event sourcing** for full history (branch, revert, review changes)
3. Supports **tiered retrieval** (shallow context → deep subgraph → synthesized)
4. Enables **partnership learning** — not just "user preferences" but collaborative knowledge
5. Runs as an **MCP server** that CC can query naturally

This is NOT another "remember user likes dark mode" system. This is a **shared knowledge graph** between human and Claude that grows over time, capturing architectural decisions, learned patterns, open questions, and project understanding.

---

## Architecture Overview

**Language**: Pure Python with MCP SDK. TypeScript adapter optional for ecosystem compatibility.

**Layers**:
- **MCP Server** (`server.py`): Handles stdio transport, defines tool schemas
- **Memory Engine** (`engine.py`): Orchestrates all modules
- **Event Store** (`events.py`): Append-only JSONL log (source of truth)
- **State** (`state.py`): Materialized graph computed from events
- **Vectors** (`vectors.py`): sqlite-vec embeddings with sentence-transformers
- **Retrieval** (`retrieval.py`): Tiered context (shallow/medium/deep)

**Storage** (`<project>/.claude/memory/`):
- `events.jsonl` — Append-only event log (source of truth)
- `state.json` — Cached materialized state
- `vectors.db` — sqlite-vec embeddings index

---

## Data Model

### Entity (node)
```typescript
interface Entity {
  id: string;              // ULID
  name: string;            // human-readable identifier
  type: EntityType;        // concept | decision | project | pattern | question | learning | entity
  observations: Observation[];
  created_at, updated_at: string;
  created_by: string;      // session ID
  access_count: number;
  last_accessed: string;
}
```

### Relation (edge)
```typescript
interface Relation {
  id: string;
  from: string;            // entity ID
  to: string;              // entity ID
  type: string;            // verb phrase: "implements", "decided_for", etc.
  created_at, created_by: string;
}
```

### MemoryEvent (append-only log entry)
```typescript
interface MemoryEvent {
  id: string;              // ULID
  ts: string;              // ISO timestamp
  op: EventOp;             // create_entity, delete_entity, create_relation, etc.
  session_id: string;
  source: "cc" | "user";
  data: EventData;
}
```

---

## MCP Tools

**Core CRUD**: `create_entities`, `create_relations`, `add_observations`, `delete_entities`, `delete_relations`, `delete_observations`, `read_graph`

**Retrieval**: `recall` (tiered: shallow/medium/deep), `remember` (atomic entity+obs+relations)

**Management**: `find_similar`, `find_orphans`, `merge_entities`, `get_graph_health`, `suggest_relations`

**Time Travel**: `get_state_at`, `diff_timerange`, `get_entity_history`, `rewind`, `restore_state_at`

**Branches**: `branch_list`, `branch_current`, `branch_create`, `branch_checkout`, `branch_add`, `branch_diff`

**Edge Weights**: `get_relation_weight`, `set_relation_importance`, `get_strongest_connections`, `get_weak_relations`

---

## Code Conventions

**Structure**: `src/mnemograph/` with modules for events, state, graph, vectors, retrieval, models, server

**Python**: 3.11+, Pydantic v2, type hints everywhere, ULID for IDs, async for MCP server

**Dependencies**: `mcp`, `pydantic`, `python-ulid`, `sqlite-vec`, `sentence-transformers`

**Error Handling**: Return structured errors, log to stderr (MCP uses stdout)

**Testing**: `python -m pytest` (NOT `pytest` directly), use in-memory event store for tests

---

## Common Pitfalls

1. **Don't over-engineer entity types** — Start simple, add as needed
2. **Don't block on embeddings** — Use lazy/background indexing
3. **Don't return too much context** — Respect max_tokens, truncate intelligently
4. **Don't forget session ID** — Track who created each event
5. **Don't make user wait for startup** — Cache state, replay only new events
6. **Don't forget MCP is stdio** — stdout for protocol only, stderr for logs

---

## Self-Knowledge: How I (Claude) Work Best

- **Architecture**: Push me to start simple. Event sourcing core first.
- **Code generation**: I write better code seeing interfaces first.
- **Testing**: Remind me to write tests for critical paths.
- **Scope creep**: Defer features not in current phase. Use `FUTURE.md`.
- **Getting stuck**: Ask me to state problem, list options, pick one.

---

## Decisions Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| Language | Python primary | Better for data processing, ML |
| Embedding model | Local sentence-transformers | No API deps, runs offline |
| Vector store | sqlite-vec | Single file, no external deps |
| Synthesis | Deferred | Focus on core first |
| Scope | Project-local memory | Simpler per-project isolation |

---

## Implementation Status (v0.3.0)

**Completed**: Event sourcing, entity types, vector index, tiered retrieval, unified CLI (`mg`), time travel, edge weights, graph visualization, `remember()`, first-run onboarding, prose recall format, branching

**Recent**: CLI consolidation — merged `cli.py` (argparse) + `vcs_cli.py` (Click) into unified Click CLI. All commands via `mg`, VCS under `mg vcs` subgroup.

**Deferred**: Sub-agent synthesis, engine decomposition (Phase 2 of cleanup spec)

---

## Development Workflow

### Running Tests
```bash
/Users/tm42/.local/bin/uv run python -m pytest -x -q
```

### Release Workflow
1. Update version in `pyproject.toml` AND `server.json` (both version fields)
2. Commit: `git commit -m "chore: Bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push commits: `git push origin main`
5. Push tag separately: `git push origin vX.Y.Z` (triggers PyPI publish)

---

> See `.specs/IMPLEMENTATION_REFERENCE.md` for historical implementation details and prior art notes.
