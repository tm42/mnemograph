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

## Prior Art — What Exists, What We Learned

### `@modelcontextprotocol/server-memory` (Official Anthropic)

The baseline. Simple knowledge graph with entities + relations + observations.

**What's good (keep this):**
- MCP-compliant protocol (stdio transport)
- Simple primitives: Entity, Relation, Observation
- JSONL storage (human-readable)
- Cascading deletes

**What's missing (our opportunity):**
- No history — writes are destructive
- No semantic search — just substring matching
- No tiered retrieval — returns everything or searches everything
- No metadata — no timestamps, confidence, source tracking
- No synthesis — can't summarize large subgraphs

**Tools it exposes (9 total):**
```
create_entities, create_relations, add_observations
delete_entities, delete_relations, delete_observations
read_graph, search_nodes, open_nodes
```

### Mem0 (Most Mature, $24M raised)

Two-phase pipeline: **Extraction → Update** (ADD/UPDATE/DELETE/NOOP)

**Key insight to borrow:**
- Don't just append — consolidate. When new info comes in, compare to existing memories and decide: add new? update existing? delete outdated? do nothing?
- Their graph variant (Mem0ᵍ) uses Neo4j and outperforms base by ~2%

**Their extraction prompt pattern:**
```
Given the conversation and existing memories, extract salient facts.
For each fact, compare to similar existing memories.
Decide: ADD (new info), UPDATE (refines existing), DELETE (contradicts), NOOP (redundant)
```

### Letta/MemGPT

OS-inspired memory hierarchy: core (in-context) ↔ archival (out-of-context)

**Key insight to borrow:**
- Self-editing memory — the agent has tools to modify its own memory
- Memory blocks with labels (persona, human, project, etc.)
- Conversation search as a retrieval mechanism

### Graphiti (Zep AI)

**Key insight to borrow:**
- **Bi-temporal model** — tracks when event occurred AND when it was recorded
- Edges have validity intervals (t_valid, t_invalid)
- Can reconstruct state at any point in time
- Closest to our event-sourcing vision

### Claude-mem (thedotmack)

A Claude Code plugin specifically. Auto-captures session activity.

**Worth studying:**
- Has "3-layer workflow pattern" for token efficiency
- Web viewer for browsing memories
- "Endless Mode" (biomimetic memory) in beta

---

## Architecture Overview

### Language Split: Python + TypeScript Hybrid

**Why hybrid?**
- **Python** excels at data processing, ML/embeddings, graph operations
- **TypeScript** is the native MCP ecosystem language (official SDK, examples)
- For production, core algorithms might be rewritten in Rust/Go anyway — prototype in what's comfortable

**The split:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TypeScript Layer                                │
│                         (MCP protocol, thin adapter)                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  mcp-server.ts                                                          │ │
│  │  - Handles MCP stdio transport                                          │ │
│  │  - Defines tool schemas                                                 │ │
│  │  - Spawns & communicates with Python engine                             │ │
│  │  - Translates MCP requests → Python calls → MCP responses               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ JSON-RPC over stdin/stdout
                                      │ (or Unix socket / HTTP for perf)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Python Layer                                   │
│                    (memory engine, all the actual logic)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   events.py  │  │   state.py   │  │   graph.py   │  │  vectors.py  │    │
│  │              │  │              │  │              │  │              │    │
│  │ Append-only  │  │ Materialize  │  │  Traversal,  │  │ sqlite-vec,  │    │
│  │ event store  │  │ state from   │  │  subgraph    │  │ embeddings   │    │
│  │              │  │ events       │  │  extraction  │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ retrieval.py │  │  engine.py   │  │   rpc.py     │                       │
│  │              │  │              │  │              │                       │
│  │ Tiered       │  │ Main entry,  │  │ JSON-RPC     │                       │
│  │ context      │  │ orchestrates │  │ server for   │                       │
│  │ logic        │  │ all modules  │  │ TS adapter   │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Storage Layer                                   │
│                                                                              │
│    ~/.claude/memory/<project-hash>/                                          │
│    ├── events.jsonl      # Append-only event log (source of truth)          │
│    ├── state.json        # Cached materialized state (rebuild on mismatch)  │
│    ├── vectors.db        # sqlite-vec embeddings index                      │
│    └── config.yaml       # Project-specific settings                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Protocol (TS ↔ Python)

Simple JSON-RPC over subprocess stdin/stdout:

```typescript
// TypeScript side
const pythonProcess = spawn('python', ['-m', 'mnemograph']);

async function callEngine(method: string, params: any): Promise<any> {
  const request = { jsonrpc: '2.0', id: nextId++, method, params };
  pythonProcess.stdin.write(JSON.stringify(request) + '\n');
  // ... await response on stdout
}

// Example: MCP tool handler
if (request.params.name === 'create_entities') {
  return await callEngine('create_entities', request.params.arguments);
}
```

```python
# Python side (rpc.py)
import sys
import json

def handle_request(request: dict) -> dict:
    method = request['method']
    params = request.get('params', {})
    
    if method == 'create_entities':
        result = engine.create_entities(params['entities'])
        return {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
    # ... etc

# Main loop
for line in sys.stdin:
    request = json.loads(line)
    response = handle_request(request)
    print(json.dumps(response), flush=True)
```

### Alternative: Pure Python (Simpler Start)

There IS a Python MCP SDK (`mcp` package). For fastest prototyping:

```python
# Pure Python option — no TypeScript at all
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("memory")

@server.tool()
async def create_entities(entities: list[dict]) -> dict:
    # Direct implementation, no IPC overhead
    return engine.create_entities(entities)

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)
```

**Recommendation**: Start with pure Python (simpler), add TypeScript wrapper later if needed for ecosystem compatibility.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code Session                          │
│                                                                   │
│  ┌─────────────────┐         MCP (stdio)        ┌─────────────┐ │
│  │  Claude Code    │◄──────────────────────────►│   Memory    │ │
│  │  (main agent)   │   tools: query, ingest,    │   Server    │ │
│  └─────────────────┘   context, manage          │  (Python)   │ │
│                                                  └──────┬──────┘ │
└─────────────────────────────────────────────────────────┼────────┘
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────────────┐
                    │                                     ▼                                     │
                    │  ┌─────────────────────────────────────────────────────────────────────┐ │
                    │  │                     Memory Engine (Python)                           │ │
                    │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │ │
                    │  │  │   Event     │  │   Graph     │  │   Vector    │  │  Retrieval │ │ │
                    │  │  │   Store     │  │   State     │  │   Index     │  │   Tiers    │ │ │
                    │  │  │  (append)   │  │ (computed)  │  │ (sqlite-vec)│  │            │ │ │
                    │  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │ │
                    │  └─────────┼────────────────┼────────────────┼───────────────┼────────┘ │
                    │            │                │                │               │          │
                    │            ▼                ▼                ▼               ▼          │
                    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
                    │  │ events.jsonl │  │  state.json  │  │  vectors.db  │                   │
                    │  │ (append-only)│  │  (cached)    │  │ (sqlite-vec) │                   │
                    │  └──────────────┘  └──────────────┘  └──────────────┘                   │
                    │                                                                         │
                    │            <project>/.claude/memory/                                    │
                    └─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Event (append-only log entry)

```typescript
interface MemoryEvent {
  id: string;              // ulid or uuid
  ts: string;              // ISO timestamp
  op: EventOp;             // operation type
  session_id: string;      // CC session that created this
  source: "cc" | "user";   // who initiated
  data: EventData;         // operation-specific payload
}

type EventOp = 
  | "create_entity" 
  | "update_entity"
  | "delete_entity"
  | "create_relation"
  | "delete_relation"
  | "add_observation"
  | "update_observation"
  | "delete_observation";
```

### Entity (node in the graph)

```typescript
interface Entity {
  id: string;              // stable ID (generated)
  name: string;            // human-readable identifier
  type: EntityType;        // concept | decision | project | pattern | question | learning
  observations: Observation[];
  created_at: string;
  updated_at: string;
  created_by: string;      // session ID
  access_count: number;    // for relevance scoring
  last_accessed: string;
}

interface Observation {
  id: string;
  text: string;
  ts: string;
  source: string;          // session ID or "user"
  confidence?: number;     // 0-1, optional
}

type EntityType = 
  | "concept"    // patterns, ideas, approaches (e.g., "repository pattern")
  | "decision"   // choices made with rationale (e.g., "chose SQLite over Postgres")
  | "project"    // codebases, systems (e.g., "auth-service")
  | "pattern"    // recurring code patterns (e.g., "error handling with Result type")
  | "question"   // open questions, unknowns (e.g., "should we use GraphQL?")
  | "learning"   // things discovered together (e.g., "pytest fixtures are powerful")
  | "entity"     // generic: people, orgs, files, etc.
```

### Relation (edge in the graph)

```typescript
interface Relation {
  id: string;
  from: string;            // entity ID
  to: string;              // entity ID
  type: string;            // verb phrase: "implements", "decided_for", "relates_to", etc.
  created_at: string;
  created_by: string;
  properties?: Record<string, any>;  // optional edge attributes
}
```

---

## MCP Tools to Implement

### Phase 1-2: Core CRUD (event-sourced)

```typescript
// Same interface as official, but event-sourced underneath
create_entities({ entities: Entity[] })
create_relations({ relations: Relation[] })
add_observations({ entityName: string, observations: string[] })
delete_entities({ names: string[] })
delete_relations({ relations: { from, to, type }[] })
delete_observations({ entityName: string, observations: string[] })
read_graph()  // returns full state
search_nodes({ query: string })  // text search
open_nodes({ names: string[] })  // get specific entities + their relations
```

### Phase 3-4: Enhanced retrieval

```typescript
// NEW: Semantic search
search_semantic({ 
  query: string, 
  limit?: number,      // default 10
  types?: EntityType[] // filter by type
})

// NEW: Tiered context retrieval
memory_context({
  depth: "shallow" | "medium" | "deep",
  focus?: string[],    // entity names to prioritize
  max_tokens?: number  // budget for response
})
// shallow: summary stats, recent entities, hot topics
// medium: vector search results + 1-hop neighbors
// deep: multi-hop subgraph from focus entities

// NEW: Smart query (auto-selects depth)
memory_query({
  query: string,
  max_tokens?: number
})
```

### Phase 5: Synthesis

```typescript
// When subgraph is large, synthesize it
memory_synthesize({
  query: string,
  subgraph: Entity[],  // or auto-extracted
  style?: "summary" | "analysis" | "recommendations"
})
```

### Phase 6: History & management

```typescript
// Event log access
memory_log({ since?: string, session?: string, limit?: number })

// Revert operations
memory_revert({ event_ids: string[] })  // undo specific events
memory_revert_session({ session_id: string })  // undo entire session

// Branching (future)
memory_branch({ name: string })
memory_merge({ from: string, into: string })
```

---

## Implementation Phases — Detailed

### Phase 0: Bootstrap (Do This First!)

**Goal**: Get the official server running, understand the protocol.

```bash
# Install official memory server
npm install -g @modelcontextprotocol/server-memory

# Or run directly
npx -y @modelcontextprotocol/server-memory
```

**Add to your MCP config** (`.mcp.json` or Claude Code settings):
```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "/path/to/memory.jsonl"
      }
    }
  }
}
```

**Play with it:**
1. Create some entities and relations
2. Search for them
3. Look at the `memory.jsonl` file — understand the format
4. Note pain points and limitations

**Checkpoint**: You understand the MCP protocol and baseline behavior.

---

### Phase 1: Event-Sourced Core

**Goal**: Replace destructive writes with append-only event log.

**File structure:**
```
<project>/.claude/memory/
├── events.jsonl      # Append-only log (source of truth)
├── state.json        # Cached materialized state
└── config.yaml       # Optional settings
```

**Implementation approach:**

1. **Define the data models (models.py):**

```python
# src/mnemograph/models.py
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
from ulid import ULID

def generate_id() -> str:
    return str(ULID())

class Observation(BaseModel):
    id: str = Field(default_factory=generate_id)
    text: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    source: str  # session ID or "user"
    confidence: float | None = None

class Entity(BaseModel):
    id: str = Field(default_factory=generate_id)
    name: str
    type: Literal["concept", "decision", "project", "pattern", "question", "learning", "entity"]
    observations: list[Observation] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = ""
    access_count: int = 0
    last_accessed: datetime | None = None

class Relation(BaseModel):
    id: str = Field(default_factory=generate_id)
    from_entity: str  # entity ID
    to_entity: str    # entity ID
    type: str         # verb phrase: "implements", "decided_for", etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = ""

class MemoryEvent(BaseModel):
    id: str = Field(default_factory=generate_id)
    ts: datetime = Field(default_factory=datetime.utcnow)
    op: str  # create_entity, delete_entity, etc.
    session_id: str
    source: Literal["cc", "user"]
    data: dict  # operation-specific payload
```

2. **Create the event store module (events.py):**

```python
# src/mnemograph/events.py
from pathlib import Path
import json
from .models import MemoryEvent

class EventStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, event: MemoryEvent) -> MemoryEvent:
        """Append event to log. Returns the event with ID/timestamp filled."""
        with open(self.path, 'a') as f:
            f.write(event.model_dump_json() + '\n')
        return event
    
    def read_all(self) -> list[MemoryEvent]:
        """Read all events from log."""
        if not self.path.exists():
            return []
        events = []
        with open(self.path, 'r') as f:
            for line in f:
                if line.strip():
                    events.append(MemoryEvent.model_validate_json(line))
        return events
    
    def read_since(self, since_id: str | None = None) -> list[MemoryEvent]:
        """Read events after a given event ID (for incremental replay)."""
        events = self.read_all()
        if since_id is None:
            return events
        # Find the index of since_id and return everything after
        for i, event in enumerate(events):
            if event.id == since_id:
                return events[i+1:]
        return events  # ID not found, return all
```

3. **Create the state materializer (state.py):**

```python
# src/mnemograph/state.py
from dataclasses import dataclass, field
from .models import Entity, Relation, MemoryEvent

@dataclass
class GraphState:
    entities: dict[str, Entity] = field(default_factory=dict)  # id -> Entity
    relations: list[Relation] = field(default_factory=list)
    last_event_id: str | None = None

def materialize(events: list[MemoryEvent]) -> GraphState:
    """Replay events to build current state."""
    state = GraphState()
    
    for event in events:
        op = event.op
        data = event.data
        
        if op == "create_entity":
            entity = Entity.model_validate(data)
            state.entities[entity.id] = entity
            
        elif op == "update_entity":
            if data["id"] in state.entities:
                entity = state.entities[data["id"]]
                for key, value in data.get("updates", {}).items():
                    setattr(entity, key, value)
                entity.updated_at = event.ts
                
        elif op == "delete_entity":
            entity_id = data["id"]
            state.entities.pop(entity_id, None)
            # Cascade delete relations
            state.relations = [
                r for r in state.relations 
                if r.from_entity != entity_id and r.to_entity != entity_id
            ]
            
        elif op == "create_relation":
            relation = Relation.model_validate(data)
            state.relations.append(relation)
            
        elif op == "delete_relation":
            state.relations = [
                r for r in state.relations
                if not (r.from_entity == data["from_entity"] and 
                       r.to_entity == data["to_entity"] and 
                       r.type == data["type"])
            ]
            
        elif op == "add_observation":
            entity_id = data["entity_id"]
            if entity_id in state.entities:
                from .models import Observation
                obs = Observation.model_validate(data["observation"])
                state.entities[entity_id].observations.append(obs)
                state.entities[entity_id].updated_at = event.ts
        
        state.last_event_id = event.id
    
    return state
```

4. **Create the main engine (engine.py):**

```python
# src/mnemograph/engine.py
from pathlib import Path
from .events import EventStore
from .state import GraphState, materialize
from .models import Entity, Relation, MemoryEvent, Observation, generate_id

class MemoryEngine:
    def __init__(self, memory_dir: Path, session_id: str):
        self.memory_dir = memory_dir
        self.session_id = session_id
        self.event_store = EventStore(memory_dir / "events.jsonl")
        self.state: GraphState = self._load_state()
    
    def _load_state(self) -> GraphState:
        """Load state from events (with optional caching later)."""
        events = self.event_store.read_all()
        return materialize(events)
    
    def _emit(self, op: str, data: dict) -> MemoryEvent:
        """Emit an event and update local state."""
        event = MemoryEvent(
            op=op,
            session_id=self.session_id,
            source="cc",
            data=data
        )
        self.event_store.append(event)
        # Re-materialize just this event (optimization: incremental)
        self.state = materialize(self.event_store.read_all())
        return event
    
    def create_entities(self, entities: list[dict]) -> list[Entity]:
        """Create multiple entities."""
        created = []
        for entity_data in entities:
            entity = Entity(
                name=entity_data["name"],
                type=entity_data.get("type", "entity"),
                observations=[
                    Observation(text=obs, source=self.session_id)
                    for obs in entity_data.get("observations", [])
                ],
                created_by=self.session_id
            )
            self._emit("create_entity", entity.model_dump(mode="json"))
            created.append(entity)
        return created
    
    def create_relations(self, relations: list[dict]) -> list[Relation]:
        """Create multiple relations."""
        created = []
        for rel_data in relations:
            # Resolve entity names to IDs
            from_id = self._resolve_entity(rel_data["from"])
            to_id = self._resolve_entity(rel_data["to"])
            if from_id and to_id:
                relation = Relation(
                    from_entity=from_id,
                    to_entity=to_id,
                    type=rel_data["relationType"],
                    created_by=self.session_id
                )
                self._emit("create_relation", relation.model_dump(mode="json"))
                created.append(relation)
        return created
    
    def _resolve_entity(self, name_or_id: str) -> str | None:
        """Find entity by name or ID."""
        if name_or_id in self.state.entities:
            return name_or_id
        for entity in self.state.entities.values():
            if entity.name == name_or_id:
                return entity.id
        return None
    
    def search_nodes(self, query: str) -> list[Entity]:
        """Simple text search across names, types, observations."""
        query_lower = query.lower()
        results = []
        for entity in self.state.entities.values():
            if query_lower in entity.name.lower():
                results.append(entity)
                continue
            if query_lower in entity.type.lower():
                results.append(entity)
                continue
            for obs in entity.observations:
                if query_lower in obs.text.lower():
                    results.append(entity)
                    break
        return results
    
    def read_graph(self) -> dict:
        """Return full graph state."""
        return {
            "entities": [e.model_dump(mode="json") for e in self.state.entities.values()],
            "relations": [r.model_dump(mode="json") for r in self.state.relations]
        }
```

5. **Wire into MCP server (server.py):**

```python
# src/mnemograph/server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pathlib import Path
import os

from .engine import MemoryEngine

# Initialize engine
memory_dir = Path(os.environ.get("MEMORY_PATH", ".claude/memory"))
session_id = os.environ.get("SESSION_ID", "default")
engine = MemoryEngine(memory_dir, session_id)

server = Server("claude-memory")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="create_entities",
            description="Create new entities in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string", "enum": ["concept", "decision", "project", "pattern", "question", "learning", "entity"]},
                                "observations": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["name"]
                        }
                    }
                },
                "required": ["entities"]
            }
        ),
        Tool(
            name="search_nodes",
            description="Search for entities by text query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="read_graph",
            description="Read the entire knowledge graph",
            inputSchema={"type": "object", "properties": {}}
        ),
        # ... add other tools
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "create_entities":
        created = engine.create_entities(arguments["entities"])
        return [TextContent(type="text", text=f"Created {len(created)} entities")]
    
    elif name == "search_nodes":
        results = engine.search_nodes(arguments["query"])
        import json
        return [TextContent(type="text", text=json.dumps([e.model_dump(mode="json") for e in results], indent=2))]
    
    elif name == "read_graph":
        import json
        return [TextContent(type="text", text=json.dumps(engine.read_graph(), indent=2))]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Checkpoint**: All writes go to event log. State is computed from events.

---

### Phase 2: Richer Node Model

**Goal**: The richer model is already defined in Phase 1! This phase is about exercising it.

**Already done in models.py:**
- Entity has `type` field (concept, decision, project, etc.)
- Observations are objects with id, ts, source, confidence
- Access tracking fields (access_count, last_accessed)

**Additional work for this phase:**

1. **Add access tracking to queries:**

```python
# In engine.py, add to search_nodes and other retrieval methods:
def _track_access(self, entity_ids: list[str]):
    """Update access counts for retrieved entities."""
    from datetime import datetime
    for entity_id in entity_ids:
        if entity_id in self.state.entities:
            entity = self.state.entities[entity_id]
            entity.access_count += 1
            entity.last_accessed = datetime.utcnow()
            # Note: We're updating in-memory state but NOT emitting events
            # Access tracking is ephemeral / derived, not source of truth
```

2. **Add helper methods for common queries:**

```python
# In engine.py
def get_recent_entities(self, limit: int = 10) -> list[Entity]:
    """Get most recently updated entities."""
    return sorted(
        self.state.entities.values(),
        key=lambda e: e.updated_at,
        reverse=True
    )[:limit]

def get_hot_entities(self, limit: int = 10) -> list[Entity]:
    """Get most frequently accessed entities."""
    return sorted(
        self.state.entities.values(),
        key=lambda e: e.access_count,
        reverse=True
    )[:limit]

def get_entities_by_type(self, entity_type: str) -> list[Entity]:
    """Filter entities by type."""
    return [e for e in self.state.entities.values() if e.type == entity_type]

def get_entity_neighbors(self, entity_id: str) -> dict:
    """Get entities connected to this one via relations."""
    outgoing = [r for r in self.state.relations if r.from_entity == entity_id]
    incoming = [r for r in self.state.relations if r.to_entity == entity_id]
    
    neighbor_ids = set(
        [r.to_entity for r in outgoing] + 
        [r.from_entity for r in incoming]
    )
    
    return {
        "entity": self.state.entities.get(entity_id),
        "outgoing": outgoing,
        "incoming": incoming,
        "neighbors": [self.state.entities[nid] for nid in neighbor_ids if nid in self.state.entities]
    }
```

3. **Add `open_nodes` tool (like official server):**

```python
# In server.py, add tool
Tool(
    name="open_nodes",
    description="Get specific entities by name and their relations",
    inputSchema={
        "type": "object",
        "properties": {
            "names": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["names"]
    }
)

# Handler
elif name == "open_nodes":
    results = []
    for name in arguments["names"]:
        entity_id = engine._resolve_entity(name)
        if entity_id:
            results.append(engine.get_entity_neighbors(entity_id))
    return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]
```

**Checkpoint**: Richer data model ready for semantic search and smart retrieval.

---

### Phase 3: Vector Index

**Goal**: Enable semantic search alongside keyword search.

**Stack:**
- **Embeddings**: `sentence-transformers` with `all-MiniLM-L6-v2` (384 dims, fast)
- **Index**: `sqlite-vec` (single file, no external deps)

**Implementation:**

1. **Create vectors module (vectors.py):**

```python
# src/mnemograph/vectors.py
import sqlite3
from pathlib import Path
from sentence_transformers import SentenceTransformer

class VectorIndex:
    def __init__(self, db_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.dims = self.model.get_sentence_embedding_dimension()
        self._init_db()
    
    def _init_db(self):
        """Initialize sqlite-vec tables."""
        import sqlite_vec
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        
        # Create virtual table for vector search
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_vectors 
            USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.dims}]
            )
        """)
        
        # Metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_meta (
                entity_id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                text_hash TEXT
            )
        """)
        self.conn.commit()
    
    def _entity_to_text(self, entity) -> str:
        """Convert entity to searchable text."""
        obs_text = ". ".join(o.text for o in entity.observations)
        return f"{entity.name} ({entity.type}): {obs_text}"
    
    def index_entity(self, entity):
        """Add or update entity in vector index."""
        text = self._entity_to_text(entity)
        text_hash = hash(text)
        
        # Check if already indexed with same content
        existing = self.conn.execute(
            "SELECT text_hash FROM entity_meta WHERE entity_id = ?",
            (entity.id,)
        ).fetchone()
        
        if existing and existing[0] == str(text_hash):
            return  # Already indexed, no change
        
        # Generate embedding
        embedding = self.model.encode(text).tolist()
        
        # Upsert (delete + insert for sqlite-vec)
        self.conn.execute("DELETE FROM entity_vectors WHERE entity_id = ?", (entity.id,))
        self.conn.execute(
            "INSERT INTO entity_vectors (entity_id, embedding) VALUES (?, ?)",
            (entity.id, embedding)
        )
        
        # Upsert metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO entity_meta (entity_id, name, type, text_hash)
            VALUES (?, ?, ?, ?)
        """, (entity.id, entity.name, entity.type, str(text_hash)))
        
        self.conn.commit()
    
    def search(self, query: str, limit: int = 10, type_filter: str | None = None) -> list[tuple[str, float]]:
        """Search for similar entities. Returns (entity_id, similarity_score) tuples."""
        query_embedding = self.model.encode(query).tolist()
        
        if type_filter:
            results = self.conn.execute("""
                SELECT v.entity_id, v.distance
                FROM entity_vectors v
                JOIN entity_meta m ON v.entity_id = m.entity_id
                WHERE m.type = ?
                AND v.embedding MATCH ?
                ORDER BY v.distance
                LIMIT ?
            """, (type_filter, query_embedding, limit)).fetchall()
        else:
            results = self.conn.execute("""
                SELECT entity_id, distance
                FROM entity_vectors
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """, (query_embedding, limit)).fetchall()
        
        # Convert distance to similarity
        return [(r[0], 1 / (1 + r[1])) for r in results]
```

2. **Integrate into engine (add to engine.py):**

```python
# In MemoryEngine.__init__
self._vector_index: VectorIndex | None = None

@property
def vector_index(self) -> VectorIndex:
    """Lazy-load vector index."""
    if self._vector_index is None:
        from .vectors import VectorIndex
        self._vector_index = VectorIndex(self.memory_dir / "vectors.db")
        self._vector_index.reindex_all(list(self.state.entities.values()))
    return self._vector_index

def search_semantic(self, query: str, limit: int = 10, type_filter: str | None = None) -> list[Entity]:
    """Semantic search using embeddings."""
    results = self.vector_index.search(query, limit, type_filter)
    entities = []
    for entity_id, score in results:
        if entity_id in self.state.entities:
            entity = self.state.entities[entity_id]
            entity._search_score = score  # Attach for visibility
            entities.append(entity)
    return entities
```

3. **Add MCP tool:**

```python
Tool(
    name="search_semantic",
    description="Search memory using semantic similarity (meaning-based)",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
            "type": {"type": "string", "description": "Filter by entity type"}
        },
        "required": ["query"]
    }
)
```

**Checkpoint**: Can find relevant memories even when exact keywords don't match.

---

### Phase 4: Tiered Retrieval

**Goal**: Implement shallow → medium → deep context levels.

```python
# src/mnemograph/retrieval.py
from dataclasses import dataclass
from .models import Entity, Relation
from .state import GraphState

@dataclass
class ContextResult:
    depth: str  # 'shallow' | 'medium' | 'deep'
    tokens_used: int
    content: str

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4

def get_shallow_context(state: GraphState, max_tokens: int = 500) -> ContextResult:
    """Summary stats, recent entities, hot topics."""
    entity_count = len(state.entities)
    relation_count = len(state.relations)
    
    # Recent entities (by updated_at)
    recent = sorted(
        state.entities.values(),
        key=lambda e: e.updated_at,
        reverse=True
    )[:5]
    
    # Most accessed
    hot = sorted(
        state.entities.values(),
        key=lambda e: e.access_count,
        reverse=True
    )[:5]
    
    content = f"""## Memory Summary
- {entity_count} entities, {relation_count} relations

### Recently Updated
{chr(10).join(f'- {e.name} ({e.type})' for e in recent)}

### Frequently Accessed
{chr(10).join(f'- {e.name} ({e.type}): {e.access_count} accesses' for e in hot)}
""".strip()
    
    return ContextResult(depth='shallow', tokens_used=estimate_tokens(content), content=content)


def get_medium_context(
    state: GraphState,
    vector_index,
    query: str | None = None,
    focus: list[str] | None = None,
    max_tokens: int = 2000
) -> ContextResult:
    """Vector search results + 1-hop neighbors."""
    entities = []
    
    if focus:
        # Start from focused entities and expand
        for name in focus:
            entity = _find_entity_by_name(state, name)
            if entity:
                entities.append(entity)
                entities.extend(_get_neighbors(state, entity.id, hops=1))
    elif query:
        # Semantic search
        results = vector_index.search(query, limit=10)
        for entity_id, _ in results:
            if entity_id in state.entities:
                entities.append(state.entities[entity_id])
    
    # Dedupe and format
    seen = set()
    unique = []
    for e in entities:
        if e.id not in seen:
            seen.add(e.id)
            unique.append(e)
    
    content = _format_entities_with_relations(state, unique, max_tokens)
    return ContextResult(depth='medium', tokens_used=estimate_tokens(content), content=content)


def get_deep_context(
    state: GraphState,
    focus: list[str] | None = None,
    max_tokens: int = 5000
) -> ContextResult:
    """Multi-hop subgraph extraction."""
    if not focus:
        # Use all entities if no focus
        entities = list(state.entities.values())
    else:
        entities = []
        for name in focus:
            entity = _find_entity_by_name(state, name)
            if entity:
                entities.extend(_get_neighbors(state, entity.id, hops=3))
    
    content = _format_full_subgraph(state, entities, max_tokens)
    return ContextResult(depth='deep', tokens_used=estimate_tokens(content), content=content)


def _find_entity_by_name(state: GraphState, name: str) -> Entity | None:
    for e in state.entities.values():
        if e.name == name:
            return e
    return None


def _get_neighbors(state: GraphState, entity_id: str, hops: int = 1) -> list[Entity]:
    """BFS to find neighbors within N hops."""
    visited = {entity_id}
    frontier = [entity_id]
    
    for _ in range(hops):
        next_frontier = []
        for eid in frontier:
            # Find connected entities
            for r in state.relations:
                if r.from_entity == eid and r.to_entity not in visited:
                    visited.add(r.to_entity)
                    next_frontier.append(r.to_entity)
                elif r.to_entity == eid and r.from_entity not in visited:
                    visited.add(r.from_entity)
                    next_frontier.append(r.from_entity)
        frontier = next_frontier
    
    return [state.entities[eid] for eid in visited if eid in state.entities]


def _format_entities_with_relations(state: GraphState, entities: list[Entity], max_tokens: int) -> str:
    """Format entities with their relations."""
    lines = []
    entity_ids = {e.id for e in entities}
    
    for entity in entities:
        lines.append(f"### {entity.name} ({entity.type})")
        for obs in entity.observations[:3]:  # Limit observations
            lines.append(f"  - {obs.text}")
        
        # Add relevant relations
        for r in state.relations:
            if r.from_entity == entity.id and r.to_entity in entity_ids:
                target = state.entities.get(r.to_entity)
                if target:
                    lines.append(f"  → {r.type} → {target.name}")
        
        lines.append("")
    
    content = "\n".join(lines)
    # Truncate if over budget
    while estimate_tokens(content) > max_tokens and lines:
        lines.pop()
        content = "\n".join(lines)
    
    return content


def _format_full_subgraph(state: GraphState, entities: list[Entity], max_tokens: int) -> str:
    """Format complete subgraph with all details."""
    # Similar to above but more comprehensive
    return _format_entities_with_relations(state, entities, max_tokens)
```

**Add MCP tool:**

```python
Tool(
    name="memory_context",
    description="Get memory context at varying depth levels",
    inputSchema={
        "type": "object",
        "properties": {
            "depth": {"type": "string", "enum": ["shallow", "medium", "deep"]},
            "query": {"type": "string", "description": "Search query (for medium depth)"},
            "focus": {"type": "array", "items": {"type": "string"}, "description": "Entity names to focus on"},
            "max_tokens": {"type": "integer", "default": 2000}
        },
        "required": ["depth"]
    }
)
```

**Checkpoint**: CC can request appropriate context depth based on task.

---

### Phase 5: Sub-Agent Synthesis (DEFERRED)

**Goal**: When subgraph is large, use a smaller model to synthesize.

**Parked for now.** When needed:
- Use Ollama with local model (Mistral 7B, Phi-3) for fast synthesis
- Or Claude API for complex queries
- Trigger when subgraph > N nodes

**Pseudocode:**
```python
# src/mnemograph/synthesis.py (future)
async def synthesize(query: str, entities: list[Entity], style: str = "summary") -> str:
    prompt = f"""Synthesize this knowledge subgraph to answer: {query}
    
    Entities: {format_entities(entities)}
    
    Produce a concise {style}. ~200 words."""
    
    if USE_LOCAL:
        return await ollama_generate("mistral", prompt)
    else:
        return await claude_api(prompt)
```

**Checkpoint**: Large knowledge dumps become digestible context.

---

### Phase 6: User Review & Git Semantics (Later)

**Goal**: CLI for managing memory like code.

```bash
# View recent changes
claude-memory log --since="1 hour ago"

# Interactive review
claude-memory review
# Shows each pending change, user can accept/reject/edit

# Revert a session
claude-memory revert --session abc123

# Create branch
claude-memory branch experiment/new-approach

# Compact old events into snapshot
claude-memory compact --before="7 days ago"
```

**Checkpoint**: User has full control over what persists.

---

## Code Conventions

### File Structure

```
claude-memory/
├── src/
│   └── mnemograph/         # Python package
│       ├── __init__.py
│       ├── __main__.py        # Entry point (python -m mnemograph)
│       ├── server.py          # MCP server (using mcp package)
│       ├── events.py          # Event store (append-only log)
│       ├── state.py           # State materialization
│       ├── graph.py           # Graph operations (neighbors, subgraph)
│       ├── vectors.py         # sqlite-vec index wrapper
│       ├── retrieval.py       # Tiered retrieval logic
│       ├── models.py          # Pydantic models for Entity, Relation, Event
│       └── tools/             # MCP tool handlers
│           ├── __init__.py
│           ├── crud.py        # create/delete entities/relations
│           ├── search.py      # search_nodes, search_semantic
│           ├── context.py     # memory_context, memory_query
│           └── manage.py      # memory_log, memory_revert
├── ts-adapter/                # Optional TypeScript MCP wrapper (later)
│   ├── src/
│   │   └── index.ts
│   ├── package.json
│   └── tsconfig.json
├── cli/
│   └── memory_cli.py          # CLI tool for user management
├── tests/
│   ├── test_events.py
│   ├── test_state.py
│   └── test_tools.py
├── pyproject.toml             # Python project config (use uv or poetry)
└── README.md
```

### Python Preferences

- **Python 3.11+** (for performance, typing improvements)
- **Pydantic v2** for data models (fast, good validation)
- **Type hints everywhere** — this helps me (Claude) generate better code
- Use `dataclasses` for simple structs, `Pydantic` for validated I/O
- **ULID** for IDs (sortable by time, unique) — use `python-ulid` package
- Async where it makes sense (MCP server), sync for simple operations

### Key Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "mcp>=1.0.0",              # MCP Python SDK
    "pydantic>=2.0",           # Data validation
    "python-ulid>=2.0",        # Sortable unique IDs
    "sqlite-vec>=0.1",         # Vector similarity search
    "sentence-transformers",    # Embeddings (Phase 3)
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]
```

### Error Handling

- MCP tools should return structured errors, not raise exceptions
- Use `Result` pattern or return `{"error": "message"}` 
- Log errors to stderr (MCP uses stdout for protocol)
- Event store writes should be atomic

```python
# Good pattern
async def create_entities(entities: list[Entity]) -> dict:
    try:
        created = engine.create_entities(entities)
        return {"success": True, "created": len(created)}
    except ValidationError as e:
        return {"success": False, "error": str(e)}
```

### Testing

- Unit tests for state materialization (given events → expected state)
- Integration tests for MCP tool flows
- Use in-memory event store for tests (no disk I/O)
- pytest-asyncio for async MCP tests

```python
# test_state.py
def test_materialize_creates_entity():
    events = [
        {"op": "create_entity", "data": {"id": "e1", "name": "test", "type": "concept", "observations": []}}
    ]
    state = materialize(events)
    assert "e1" in state.entities
    assert state.entities["e1"].name == "test"
```

---

## Common Pitfalls — Avoid These

### 1. Don't over-engineer entity types early
Start with just `concept`, `decision`, `entity`. Add types as needed.

### 2. Don't block on embeddings
Embeddings are slow. Don't wait for them in the MCP request path.
- Option A: Background indexing (update index after response)
- Option B: Lazy indexing (index on first search)

### 3. Don't return too much context
Token budgets matter. Always have a max_tokens parameter and respect it.
Truncate intelligently (most recent/relevant first).

### 4. Don't forget about the session ID
Track which session created each event. This enables:
- "What did we learn this session?"
- "Revert everything from session X"
- "Show me decisions from last week"

### 5. Don't make the user wait for startup
State materialization from events could be slow with many events.
- Cache materialized state to disk (`state.json`)
- On startup, load cache, then replay only new events

### 6. Don't forget MCP is stdio
- Print protocol messages to stdout only
- Use stderr for logs/debugging
- Don't use console.log for debugging (breaks protocol)

---

## Self-Knowledge: How I (Claude) Work Best

### On architecture decisions
I tend to want to over-design. Push me to start simple. The event sourcing core is the foundation — get that solid before adding vectors/synthesis.

### On code generation
I write better code when I can see the interfaces first. Define the types, then implement. If you're stuck, ask me to write interfaces/types before implementation.

### On testing
I sometimes skip tests in the interest of speed. Remind me to write tests for critical paths (event store, state materialization).

### On scope creep
I'll suggest cool features. Defer anything not in the current phase. Keep a `FUTURE.md` file for ideas.

### On getting stuck
If I'm going in circles, ask me to:
1. State the specific problem
2. List 2-3 concrete options
3. Pick one and commit

---

## First Session Checklist

- [ ] Read this entire document
- [ ] Set up project structure:
  ```bash
  mkdir claude-memory && cd claude-memory
  uv init  # or: python -m venv .venv && pip install ...
  uv add mcp pydantic python-ulid
  ```
- [ ] Get official memory server running, play with it (Phase 0)
- [ ] Create `src/mnemograph/` package structure
- [ ] Implement `models.py` (Entity, Relation, Event, Observation)
- [ ] Implement `events.py` (EventStore class)
- [ ] Implement `state.py` (materialize function)
- [ ] Implement `engine.py` (MemoryEngine class with create_entities, search_nodes)
- [ ] Implement `server.py` (MCP server with basic tools)
- [ ] Test: create entity → check events.jsonl → restart → verify state rebuilt

**After Phase 1 works**, continue to Phase 2 (richer queries) and Phase 3 (vectors).

---

## Decisions Made (with M)

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Language** | Python (primary) + TypeScript (optional adapter) | Python is better for data processing, ML; TS adapter can come later if needed |
| **Embedding model** | Local `sentence-transformers` (all-MiniLM-L6-v2) | No API deps, fast enough, runs offline |
| **Vector store** | sqlite-vec | Single file, no external deps, good enough for prototype |
| **Synthesis** | Parked for later | Focus on core event-sourced graph first |
| **Scope** | Project-local memory | Simpler, each project gets its own `.claude/memory/` |

---

## Implementation Status (v0.2.0)

**Completed:**
- ✅ Phase 0: Bootstrap (MCP protocol understanding)
- ✅ Phase 1: Event-sourced core (append-only log, state materialization)
- ✅ Phase 2: Richer node model (entity types, observations, access tracking)
- ✅ Phase 3: Vector index (sqlite-vec with sentence-transformers)
- ✅ Phase 4: Tiered retrieval (shallow/medium/deep context)
- ✅ Phase 6: VCS CLI (git-based version control)
- ✅ Phase 7: VCS CLI extensions (event rewind, time travel)
- ✅ Edge weights (recency, co-access, explicit importance)
- ✅ Graph visualization (D3.js viewer with layout algorithms, color modes, live refresh)
- ✅ Smart retrieval with token management (structure-only for large results)
- ✅ `remember()` for atomic knowledge storage
- ✅ First-run onboarding with guide seeding
- ✅ Renamed tools: `memory_context` → `recall`, removed `search_nodes`

**Deferred:**
- Phase 5: Sub-agent synthesis (parked for now)

---

## Development Workflow

### Running Tests

```bash
# From the mnemograph directory — use python -m pytest, NOT pytest directly
/Users/tm42/.local/bin/uv run python -m pytest -x -q

# Or with full path
/Users/tm42/.local/bin/uv run --directory /Users/tm42/_mnemograph python -m pytest -x -q
```

**IMPORTANT:** Direct `pytest` command fails with "No such file" error. Always use `python -m pytest`.

### Release Workflow

GitHub Actions auto-publishes to PyPI and MCP registry when a version tag is pushed.

**Steps to release:**

1. Update version in `pyproject.toml`
2. Update version in `server.json` — **BOTH** `version` fields (top-level AND `packages[0].version`)
3. Commit the version bump
4. Create tag: `git tag vX.Y.Z`
5. Push commits: `git push origin main`
6. **Push tag separately**: `git push origin vX.Y.Z`

**CRITICAL:** Pushing commits alone does NOT trigger the release. You MUST push the tag separately.

```bash
# Example release
git add pyproject.toml server.json
git commit -m "chore: Bump version to 0.2.1"
git tag v0.2.1
git push origin main
git push origin v0.2.1  # This triggers PyPI publish!
```

---

Good luck, future me. Build something great. 🚀
