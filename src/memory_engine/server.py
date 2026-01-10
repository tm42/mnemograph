"""MCP server for the memory engine."""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .engine import MemoryEngine

# Initialize engine from environment
memory_dir = Path(os.environ.get("MEMORY_PATH", ".claude/memory"))
session_id = os.environ.get("SESSION_ID", "default")
engine = MemoryEngine(memory_dir, session_id)

server = Server("graphmem")


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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
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

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def main():
    """Entry point for the MCP server."""
    print("GraphMem MCP Server starting...", file=sys.stderr)
    asyncio.run(_run_server())


async def _run_server():
    """Run the MCP server."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    main()
