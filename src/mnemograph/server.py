"""MCP server for the memory engine."""

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .engine import MemoryEngine

# --- Logging setup ---
memory_dir = Path(os.environ.get("MEMORY_PATH", ".claude/memory"))
memory_dir.mkdir(parents=True, exist_ok=True)
log_file = memory_dir / "mnemograph.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("mnemograph")

# --- Initialize engine ---
session_id = os.environ.get("SESSION_ID", "default")
engine = MemoryEngine(memory_dir, session_id)

server = Server("mnemograph")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="create_entities",
            description=(
                "Create new entities in the knowledge graph. "
                "AUTO-BLOCKS if similar entity exists (>85% match) — returns warning with suggestion. "
                "Use create_entities_force to override. "
                "Types: concept, decision, project, pattern, question, learning. "
                "Naming: use canonical names ('Python' not 'python language'), prefix decisions ('Decision: Use Redis')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Canonical entity name (e.g., 'Python', 'Decision: Use Redis')",
                                },
                                "entityType": {
                                    "type": "string",
                                    "enum": [
                                        "concept", "decision", "project",
                                        "pattern", "question", "learning", "entity"
                                    ],
                                    "description": "concept=ideas/tech, decision=choices, project=repos, pattern=solutions, question=unknowns, learning=discoveries",
                                },
                                "observations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Atomic facts (one fact per string). Use prefixes: 'Gotcha: ...', 'Warning: ...', 'Status: ...'",
                                },
                            },
                            "required": ["name", "entityType"],
                        },
                    }
                },
                "required": ["entities"],
            },
        ),
        Tool(
            name="create_entities_force",
            description=(
                "Create entities BYPASSING duplicate check. "
                "Use when you're certain the entity is distinct despite similar names "
                "(e.g., 'React' library vs 'React' conference)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Entity name",
                                },
                                "entityType": {
                                    "type": "string",
                                    "enum": [
                                        "concept", "decision", "project",
                                        "pattern", "question", "learning", "entity"
                                    ],
                                },
                                "observations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["name", "entityType"],
                        },
                    }
                },
                "required": ["entities"],
            },
        ),
        Tool(
            name="remember",
            description=(
                "Store knowledge atomically — entity + observations + relations in ONE call. "
                "PRIMARY TOOL for storing new knowledge. Prevents orphan entities. "
                "AUTO-BLOCKS if similar entity exists (>80% match). Use force=True to override."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name (canonical form: 'FastAPI' not 'fastapi framework')",
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "concept", "decision", "project",
                            "pattern", "question", "learning", "entity"
                        ],
                        "description": "concept=ideas/tech, decision=choices, project=repos, pattern=solutions, question=unknowns, learning=discoveries",
                    },
                    "observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Atomic facts about this entity",
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string", "description": "Target entity name"},
                                "type": {
                                    "type": "string",
                                    "description": "Relation type: uses, implements, depends_on, part_of, etc.",
                                },
                            },
                            "required": ["to", "type"],
                        },
                        "description": "Relations FROM this entity to others",
                    },
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "Bypass duplicate check",
                    },
                },
                "required": ["name", "entity_type"],
            },
        ),
        Tool(
            name="create_relations",
            description=(
                "Create relations (edges) between entities. Every entity should have at least one relation. "
                "Use specific types: uses, implements, part_of, depends_on, alternative_to, decided_by, affects. "
                "Avoid generic 'related_to' when a specific type fits."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string", "description": "Source entity name"},
                                "to": {"type": "string", "description": "Target entity name"},
                                "relationType": {
                                    "type": "string",
                                    "description": "Relation type: uses, implements, part_of, depends_on, enables, alternative_to, decided_by, affects, replaced_by, learned_from",
                                },
                            },
                            "required": ["from", "to", "relationType"],
                        },
                    }
                },
                "required": ["relations"],
            },
        ),
        Tool(
            name="add_observations",
            description=(
                "Add atomic facts to existing entities. One fact per observation — don't dump paragraphs. "
                "Use prefixes: 'Gotcha: ...', 'Warning: ...', 'Status: ...', 'Source: ...'. "
                "For relations, use create_relations instead of 'X is related to Y' observations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": {"type": "string", "description": "Entity to add facts to"},
                                "contents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Atomic facts. Good: '88 keys in standard piano'. Bad: multi-sentence paragraphs.",
                                },
                            },
                            "required": ["entityName", "contents"],
                        },
                    }
                },
                "required": ["observations"],
            },
        ),
        Tool(
            name="delete_entities",
            description="Delete entities by name (cascades to relations)",
            inputSchema={
                "type": "object",
                "properties": {
                    "entityNames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of entities to delete",
                    }
                },
                "required": ["entityNames"],
            },
        ),
        Tool(
            name="delete_relations",
            description="Delete specific relations",
            inputSchema={
                "type": "object",
                "properties": {
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "relationType": {"type": "string"},
                            },
                            "required": ["from", "to", "relationType"],
                        },
                    }
                },
                "required": ["relations"],
            },
        ),
        Tool(
            name="delete_observations",
            description="Delete specific observations from entities",
            inputSchema={
                "type": "object",
                "properties": {
                    "deletions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": {"type": "string"},
                                "observations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Observation texts to delete",
                                },
                            },
                            "required": ["entityName", "observations"],
                        },
                    }
                },
                "required": ["deletions"],
            },
        ),
        Tool(
            name="recall",
            description=(
                "PRIMARY RETRIEVAL TOOL. Get relevant context with automatic token management. "
                "Use focus=['EntityName'] to get full details on specific entities. "
                "shallow=quick summary, medium=semantic search+neighbors, deep=full exploration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "string",
                        "enum": ["shallow", "medium", "deep"],
                        "default": "medium",
                        "description": "shallow=summary (~500 tokens), medium=search+1-hop (~2000), deep=2-hop traverse (~5000)",
                    },
                    "query": {
                        "type": "string",
                        "description": "What you're looking for (used for medium/deep semantic search)",
                    },
                    "focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names to retrieve in full detail (replaces open_nodes). Use this to expand specific entities.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Override default token budget",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["prose", "graph"],
                        "default": "prose",
                        "description": "Output format: prose (human-readable) or graph (JSON structure)",
                    },
                },
                "required": ["depth"],
            },
        ),
        # --- Event Rewind Tools ---
        Tool(
            name="get_state_at",
            description="Get graph state at a specific point in time (event rewind)",
            inputSchema={
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "Time reference: ISO datetime (2025-01-15), relative (7 days ago), or named (yesterday, last week)",
                    },
                },
                "required": ["timestamp"],
            },
        ),
        Tool(
            name="diff_timerange",
            description="Show what changed between two points in time",
            inputSchema={
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start time (ISO, relative, or named)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End time (default: now)",
                    },
                },
                "required": ["start"],
            },
        ),
        Tool(
            name="get_entity_history",
            description="Get the history of all changes to an entity",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to get history for",
                    },
                },
                "required": ["entity_name"],
            },
        ),
        # --- Edge Weight Tools ---
        Tool(
            name="get_relation_weight",
            description="Get weight breakdown for a relation (recency, co-access, explicit)",
            inputSchema={
                "type": "object",
                "properties": {
                    "relation_id": {
                        "type": "string",
                        "description": "Relation ID (full or prefix)",
                    },
                },
                "required": ["relation_id"],
            },
        ),
        Tool(
            name="set_relation_importance",
            description="Set explicit importance weight for a relation",
            inputSchema={
                "type": "object",
                "properties": {
                    "relation_id": {
                        "type": "string",
                        "description": "Relation ID (full or prefix)",
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance from 0.0 (unimportant) to 1.0 (critical)",
                    },
                },
                "required": ["relation_id", "importance"],
            },
        ),
        Tool(
            name="get_strongest_connections",
            description="Get an entity's strongest connections by edge weight",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Entity name to get connections for",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max connections to return",
                    },
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="get_weak_relations",
            description="Get relations below a weight threshold (pruning candidates)",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_weight": {
                        "type": "number",
                        "default": 0.1,
                        "description": "Only include relations with weight <= this value",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max relations to return",
                    },
                },
            },
        ),
        # --- Universal Agent Tools ---
        Tool(
            name="get_primer",
            description="Get oriented with this knowledge graph. Call at session start.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="session_start",
            description="Signal session start and get initial context. Returns quick_start guide with tool usage. IMPORTANT: If memory is empty, ask user whether to use project-local or global memory scope.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_hint": {
                        "type": "string",
                        "description": "Optional project name or path for context",
                    },
                },
            },
        ),
        Tool(
            name="session_end",
            description="Signal session end, optionally save summary",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Optional session summary to store as observation",
                    },
                },
            },
        ),
        # --- Graph Coherence Tools ---
        Tool(
            name="find_similar",
            description=(
                "Find entities with similar names (potential duplicates). "
                "Use before creating to check for existing entities. "
                "Returns similarity scores — consider merging if >0.85."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name to check for similar existing entities",
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Similarity threshold 0-1 (default 0.7)",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="find_orphans",
            description=(
                "Find entities with no relations (likely incomplete). "
                "Orphans should be connected, merged into another entity, or deleted."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="merge_entities",
            description=(
                "Merge source entity into target. Source's observations and relations move to target, then source is deleted. "
                "Use to consolidate duplicates (e.g., merge 'ReactJS' into 'React')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Entity to merge FROM (will be deleted)",
                    },
                    "target": {
                        "type": "string",
                        "description": "Entity to merge INTO (will gain observations/relations)",
                    },
                    "delete_source": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to delete source after merge (default true)",
                    },
                },
                "required": ["source", "target"],
            },
        ),
        Tool(
            name="get_graph_health",
            description=(
                "Assess knowledge graph quality. Returns: orphan count, potential duplicates, overloaded entities, weak relations. "
                "Run periodically to maintain graph hygiene."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="suggest_relations",
            description=(
                "Suggest potential relations for an entity based on semantic similarity and co-occurrence. "
                "Useful for connecting newly created entities or discovering missing links."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to get relation suggestions for",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max suggestions to return",
                    },
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="clear_graph",
            description=(
                "Clear ALL entities and relations from the graph. Use sparingly! "
                "For graphs with >10 entities: requires reason AND confirm_token. "
                "Event-sourced: can rewind to before clear with get_state_at(timestamp)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for clearing (required for graphs >10 entities)",
                    },
                    "confirm_token": {
                        "type": "string",
                        "description": "Confirmation token for large graphs (format: CLEAR_<count>)",
                    },
                },
            },
        ),
        # --- Recovery Tools ---
        Tool(
            name="reload",
            description=(
                "Reload graph state from mnemograph.db on disk. "
                "Use after: git operations (checkout, restore), external edits to mnemograph.db, "
                "or any time MCP server seems out of sync."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="rewind",
            description=(
                "Rewind graph to a previous state using git. Fast undo, audit trail in git only. "
                "For audit-preserving restore, use restore_state_at() instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "default": 1,
                        "description": "Go back N commits that touched mnemograph.db",
                    },
                    "to_commit": {
                        "type": "string",
                        "description": "Or specify exact commit hash",
                    },
                },
            },
        ),
        Tool(
            name="restore_state_at",
            description=(
                "Restore graph to state at a specific timestamp. "
                "Emits clear + recreate events — full audit trail preserved. "
                "Use get_state_at() first to preview what will be restored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "ISO datetime or relative ('2 hours ago', 'yesterday')",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why restoring (recorded in events)",
                    },
                },
                "required": ["timestamp"],
            },
        ),
        # --- Branch Tools ---
        Tool(
            name="branch_list",
            description=(
                "List all memory branches. Branches are filtered views of the knowledge graph. "
                "Main branch sees everything; other branches see filtered subsets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "include_archived": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include archived branches",
                    },
                },
            },
        ),
        Tool(
            name="branch_current",
            description="Get the name and details of the currently active branch.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="branch_create",
            description=(
                "Create a new branch with seed entities. Uses BFS to expand from seeds to N-hop neighbors. "
                "Branch names follow format: <type>/<name> where type is project|feature|domain|spike|archive."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Branch name (e.g., 'project/auth-service', 'feature/jwt')",
                    },
                    "seed_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names to seed the branch from",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional branch description",
                    },
                    "depth": {
                        "type": "integer",
                        "default": 2,
                        "description": "How many hops from seeds to include (default: 2)",
                    },
                    "checkout": {
                        "type": "boolean",
                        "default": False,
                        "description": "Switch to the new branch after creation",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="branch_checkout",
            description=(
                "Switch to a different branch. Changes what entities/relations are visible in queries. "
                "Use 'main' to see everything."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Branch name to switch to",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="branch_add",
            description=(
                "Add entities to the current branch. Only works on non-main branches. "
                "Optionally includes relations between added entities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names to add to current branch",
                    },
                    "include_relations": {
                        "type": "boolean",
                        "default": True,
                        "description": "Also include relations between added entities",
                    },
                },
                "required": ["entity_names"],
            },
        ),
        Tool(
            name="branch_diff",
            description=(
                "Show differences between two branches. Returns entities and relations that are "
                "only in one branch vs the other, plus what's common to both."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "branch_a": {
                        "type": "string",
                        "description": "First branch (default: current branch)",
                    },
                    "branch_b": {
                        "type": "string",
                        "description": "Second branch to compare",
                    },
                },
                "required": ["branch_b"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool call: {name}")
    logger.debug(f"Arguments: {arguments}")
    try:
        if name == "create_entities":
            results = engine.create_entities(arguments["entities"])
            # Results can be Entity objects or warning dicts
            output = []
            for r in results:
                if isinstance(r, dict):
                    output.append(r)  # Warning dict
                else:
                    output.append(r.model_dump(mode="json"))  # Entity object
            return [TextContent(type="text", text=json.dumps(output, indent=2, default=str))]

        elif name == "create_entities_force":
            # Bypass duplicate check
            created = engine.create_entities_force(arguments["entities"])
            result = [e.model_dump(mode="json") for e in created]
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "remember":
            result = engine.remember(
                name=arguments["name"],
                entity_type=arguments["entity_type"],
                observations=arguments.get("observations"),
                relations=arguments.get("relations"),
                force=arguments.get("force", False),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "create_relations":
            result = engine.create_relations(arguments["relations"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "add_observations":
            results = engine.add_observations(arguments["observations"])
            return [TextContent(type="text", text=json.dumps(results, indent=2))]

        elif name == "delete_entities":
            count = engine.delete_entities(arguments["entityNames"])
            return [TextContent(type="text", text=f"Deleted {count} entities")]

        elif name == "delete_relations":
            count = engine.delete_relations(arguments["relations"])
            return [TextContent(type="text", text=f"Deleted {count} relations")]

        elif name == "delete_observations":
            count = engine.delete_observations(arguments["deletions"])
            return [TextContent(type="text", text=f"Deleted {count} observations")]

        elif name == "recall":
            result = engine.recall(
                depth=arguments["depth"],
                query=arguments.get("query"),
                focus=arguments.get("focus"),
                max_tokens=arguments.get("max_tokens"),
                format=arguments.get("format", "prose"),
            )
            # Return full result dict for structure-only responses
            if result.get("structure_only"):
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            # Return the formatted content directly (it's markdown)
            return [TextContent(type="text", text=result["content"])]

        # --- Event Rewind Tools ---
        elif name == "get_state_at":
            state = engine.state_at(arguments["timestamp"])
            result = {
                "timestamp": arguments["timestamp"],
                "entity_count": len(state.entities),
                "relation_count": len(state.relations),
                "entities": [e.to_summary() for e in state.entities.values()],
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "diff_timerange":
            result = engine.diff_between(
                start=arguments["start"],
                end=arguments.get("end"),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_entity_history":
            history = engine.get_entity_history(arguments["entity_name"])
            return [TextContent(type="text", text=json.dumps(history, indent=2, default=str))]

        # --- Edge Weight Tools ---
        elif name == "get_relation_weight":
            result = engine.get_relation_weight(arguments["relation_id"])
            if result is None:
                return [TextContent(type="text", text=f"Relation not found: {arguments['relation_id']}")]
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "set_relation_importance":
            result = engine.set_relation_importance(
                relation_id=arguments["relation_id"],
                importance=arguments["importance"],
            )
            if result is None:
                return [TextContent(type="text", text=f"Relation not found: {arguments['relation_id']}")]
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_strongest_connections":
            result = engine.get_strongest_connections(
                entity_name=arguments["entity_name"],
                limit=arguments.get("limit", 10),
            )
            if not result:
                return [TextContent(type="text", text=f"No connections found for: {arguments['entity_name']}")]
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_weak_relations":
            result = engine.get_weak_relations(
                max_weight=arguments.get("max_weight", 0.1),
                limit=arguments.get("limit", 20),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        # --- Universal Agent Tools ---
        elif name == "get_primer":
            result = engine.get_primer()
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "session_start":
            result = engine.session_start(project_hint=arguments.get("project_hint"))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "session_end":
            result = engine.session_end(summary=arguments.get("summary"))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        # --- Graph Coherence Tools ---
        elif name == "find_similar":
            result = engine.find_similar(
                name=arguments["name"],
                threshold=arguments.get("threshold", 0.7),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "find_orphans":
            result = engine.find_orphans()
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "merge_entities":
            result = engine.merge_entities(
                source=arguments["source"],
                target=arguments["target"],
                delete_source=arguments.get("delete_source", True),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "get_graph_health":
            result = engine.get_graph_health()
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "suggest_relations":
            result = engine.suggest_relations(
                entity=arguments["entity"],
                limit=arguments.get("limit", 5),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "clear_graph":
            result = engine.clear_graph(
                reason=arguments.get("reason"),
                confirm_token=arguments.get("confirm_token"),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        # --- Recovery Tools ---
        elif name == "reload":
            result = engine.reload()
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "rewind":
            result = engine.rewind(
                steps=arguments.get("steps", 1),
                to_commit=arguments.get("to_commit"),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "restore_state_at":
            result = engine.restore_state_at(
                timestamp=arguments["timestamp"],
                reason=arguments.get("reason", ""),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        # --- Branch Tools ---
        elif name == "branch_list":
            branches = engine.branch_manager.list(
                include_archived=arguments.get("include_archived", False)
            )
            current = engine.branch_manager.current_branch_name()
            result = [
                {
                    "name": b.name,
                    "is_current": b.name == current,
                    "entity_count": len(b.entity_ids) if b.name != "main" else len(engine.state.entities),
                    "relation_count": len(b.relation_ids) if b.name != "main" else len(engine.state.relations),
                    "description": b.description,
                    "is_active": b.is_active,
                }
                for b in branches
            ]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "branch_current":
            branch = engine.branch_manager.current_branch()
            result = {
                "name": branch.name,
                "description": branch.description,
                "entity_count": len(branch.entity_ids) if branch.name != "main" else len(engine.state.entities),
                "relation_count": len(branch.relation_ids) if branch.name != "main" else len(engine.state.relations),
                "parent": branch.parent,
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "branch_create":
            branch = engine.branch_manager.create(
                name=arguments["name"],
                seed_entities=arguments.get("seed_entities", []),
                description=arguments.get("description", ""),
                depth=arguments.get("depth", 2),
            )
            if arguments.get("checkout", False):
                engine.branch_manager.checkout(arguments["name"])

            result = {
                "name": branch.name,
                "entity_count": len(branch.entity_ids),
                "relation_count": len(branch.relation_ids),
                "checked_out": arguments.get("checkout", False),
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "branch_checkout":
            branch = engine.branch_manager.checkout(arguments["name"])
            result = {
                "switched_to": branch.name,
                "entity_count": len(branch.entity_ids) if branch.name != "main" else len(engine.state.entities),
                "relation_count": len(branch.relation_ids) if branch.name != "main" else len(engine.state.relations),
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "branch_add":
            current = engine.branch_manager.current_branch_name()
            if current == "main":
                return [TextContent(type="text", text=json.dumps({
                    "error": "Cannot add entities to main branch (main sees everything)"
                }))]

            # Resolve entity names to IDs
            entity_ids = []
            not_found = []
            for name_str in arguments["entity_names"]:
                eid = engine._resolve_entity(name_str)
                if eid:
                    entity_ids.append(eid)
                else:
                    not_found.append(name_str)

            if entity_ids:
                branch = engine.branch_manager.add_entities(
                    current,
                    entity_ids,
                    include_relations=arguments.get("include_relations", True),
                )
                result: dict = {
                    "added": len(entity_ids),
                    "total_entities": len(branch.entity_ids),
                    "total_relations": len(branch.relation_ids),
                }
                if not_found:
                    result["not_found"] = not_found
            else:
                result = {"error": "No valid entities found", "not_found": not_found}

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "branch_diff":
            branch_a = arguments.get("branch_a") or engine.branch_manager.current_branch_name()
            branch_b = arguments["branch_b"]
            diff = engine.branch_manager.diff(branch_a, branch_b)

            # Convert sets to lists for JSON
            result = {
                "branch_a": diff["branch_a"],
                "branch_b": diff["branch_b"],
                "only_in_a": {
                    "entity_count": len(diff["only_in_a"]["entities"]),
                    "relation_count": len(diff["only_in_a"]["relations"]),
                },
                "only_in_b": {
                    "entity_count": len(diff["only_in_b"]["entities"]),
                    "relation_count": len(diff["only_in_b"]["relations"]),
                },
                "in_both": {
                    "entity_count": len(diff["in_both"]["entities"]),
                    "relation_count": len(diff["in_both"]["relations"]),
                },
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        logger.error(traceback.format_exc())
        return [TextContent(type="text", text=f"Error: {e}")]


def main():
    """Entry point for the MCP server."""
    logger.info(f"Mnemograph MCP Server starting (memory_dir={memory_dir}, session={session_id})")
    logger.info(f"Loaded {len(engine.state.entities)} entities, {len(engine.state.relations)} relations")
    try:
        asyncio.run(_run_server())
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        logger.error(traceback.format_exc())
        raise


async def _run_server():
    """Run the MCP server."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    main()
