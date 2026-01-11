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
            description="Create new entities in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Entity name"},
                                "entityType": {
                                    "type": "string",
                                    "enum": [
                                        "concept", "decision", "project",
                                        "pattern", "question", "learning", "entity"
                                    ],
                                    "description": "Type of entity",
                                },
                                "observations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Initial observations about this entity",
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
            name="create_relations",
            description="Create relations between entities. Use active voice for relation types.",
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
                                    "description": "Relation type (active voice, e.g., 'implements', 'uses')",
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
            description="Add observations to existing entities",
            inputSchema={
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entityName": {"type": "string"},
                                "contents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Observations to add",
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
            name="search_nodes",
            description="Search entities by text query (matches name, type, observations)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
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
            name="memory_context",
            description="Get memory context at varying depth levels for efficient retrieval",
            inputSchema={
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "string",
                        "enum": ["shallow", "medium", "deep"],
                        "default": "shallow",
                        "description": "shallow=summary (~500 tokens), medium=search+neighbors (~2000), deep=multi-hop (~5000)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for medium depth (uses semantic search)",
                    },
                    "focus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names to focus on",
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool call: {name}")
    logger.debug(f"Arguments: {arguments}")
    try:
        if name == "create_entities":
            created = engine.create_entities(arguments["entities"])
            result = [e.model_dump(mode="json") for e in created]
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "create_relations":
            created = engine.create_relations(arguments["relations"])
            result = [r.model_dump(mode="json") for r in created]
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

        elif name == "search_nodes":
            result = engine.search_nodes(arguments["query"])
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

        elif name == "memory_context":
            result = engine.memory_context(
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
