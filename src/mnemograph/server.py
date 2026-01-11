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
            name="read_graph",
            description="Read the entire knowledge graph",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_graph",
            description=(
                "Search entities by text. ALWAYS SEARCH BEFORE CREATING to avoid duplicates. "
                "Try canonical names first, then variants ('PostgreSQL', 'Postgres', 'psql')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text (try canonical name, then variants)"}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="open_nodes",
            description="Get specific entities by name with their relations",
            inputSchema={
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names to retrieve",
                    }
                },
                "required": ["names"],
            },
        ),
        Tool(
            name="search_semantic",
            description="Search entities using semantic similarity (meaning-based, not just keyword matching)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "limit": {"type": "integer", "default": 10, "description": "Max results"},
                    "type": {
                        "type": "string",
                        "description": "Filter by entity type",
                        "enum": [
                            "concept", "decision", "project",
                            "pattern", "question", "learning", "entity"
                        ],
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="recall",
            description=(
                "Get relevant context for a query. Use at session start or before decisions. "
                "shallow=quick summary, medium=semantic search+neighbors (default), deep=full exploration."
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
                        "description": "Entity names to start traversal from",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Override default token budget",
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
            description="Signal session start and get initial context",
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
                "Event-sourced: can rewind to before clear with get_state_at(timestamp)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for clearing (recorded in event history)",
                    },
                },
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

        elif name == "read_graph":
            result = engine.read_graph()
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "search_graph":
            result = engine.search_graph(arguments["query"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "open_nodes":
            result = engine.open_nodes(arguments["names"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "search_semantic":
            result = engine.search_semantic(
                query=arguments["query"],
                limit=arguments.get("limit", 10),
                type_filter=arguments.get("type"),
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "recall":
            result = engine.recall(
                depth=arguments["depth"],
                query=arguments.get("query"),
                focus=arguments.get("focus"),
                max_tokens=arguments.get("max_tokens"),
            )
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
            result = engine.clear_graph(reason=arguments.get("reason", ""))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

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
