#!/usr/bin/env python3
"""Seed script to populate memory with test data.

Usage:
    MEMORY_PATH=/path/to/memory python scripts/seed.py

    # Or with default path:
    python scripts/seed.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnemograph.engine import MemoryEngine


def seed_basic_knowledge(engine: MemoryEngine) -> None:
    """Seed with basic software engineering knowledge."""

    # --- Concepts ---
    engine.create_entities([
        {
            "name": "Event Sourcing",
            "entityType": "concept",
            "observations": [
                "Pattern where state is derived from an append-only log of events",
                "Enables time-travel debugging and audit trails",
                "Used in GraphMem for memory persistence",
            ],
        },
        {
            "name": "Knowledge Graph",
            "entityType": "concept",
            "observations": [
                "Data structure with entities (nodes) and relations (edges)",
                "Enables semantic queries and relationship traversal",
                "More expressive than flat key-value storage",
            ],
        },
        {
            "name": "Semantic Search",
            "entityType": "concept",
            "observations": [
                "Search based on meaning rather than exact keyword matching",
                "Uses embeddings to represent text as vectors",
                "Enables finding related content even with different wording",
            ],
        },
        {
            "name": "MCP Protocol",
            "entityType": "concept",
            "observations": [
                "Model Context Protocol for LLM tool integration",
                "JSON-RPC over stdio transport",
                "Standard interface for Claude Code extensions",
            ],
        },
    ])

    # --- Decisions ---
    engine.create_entities([
        {
            "name": "Use Python for GraphMem",
            "entityType": "decision",
            "observations": [
                "Python better for ML/embeddings (sentence-transformers)",
                "MCP has official Python SDK",
                "Easier data processing and prototyping",
            ],
        },
        {
            "name": "Use sqlite-vec for vectors",
            "entityType": "decision",
            "observations": [
                "Single file database, no external dependencies",
                "Good enough for local use case",
                "Simpler than Postgres with pgvector",
            ],
        },
        {
            "name": "ULID for IDs",
            "entityType": "decision",
            "observations": [
                "Sortable by creation time",
                "URL-safe, no special characters",
                "Better than UUID for event logs",
            ],
        },
    ])

    # --- Patterns ---
    engine.create_entities([
        {
            "name": "Lazy Loading",
            "entityType": "pattern",
            "observations": [
                "Defer expensive operations until needed",
                "Used for vector index in GraphMem",
                "Improves startup time",
            ],
        },
        {
            "name": "Tiered Retrieval",
            "entityType": "pattern",
            "observations": [
                "Shallow: summary stats (~500 tokens)",
                "Medium: search + neighbors (~2000 tokens)",
                "Deep: multi-hop traversal (~5000 tokens)",
            ],
        },
    ])

    # --- Project ---
    engine.create_entities([
        {
            "name": "GraphMem",
            "entityType": "project",
            "observations": [
                "Event-sourced knowledge graph memory for Claude Code",
                "Implements MCP protocol with 11 tools",
                "Local-first, no external services required",
            ],
        },
    ])

    # --- Relations ---
    engine.create_relations([
        {"from": "GraphMem", "to": "Event Sourcing", "relationType": "uses"},
        {"from": "GraphMem", "to": "Knowledge Graph", "relationType": "implements"},
        {"from": "GraphMem", "to": "Semantic Search", "relationType": "provides"},
        {"from": "GraphMem", "to": "MCP Protocol", "relationType": "exposes_via"},
        {"from": "Use Python for GraphMem", "to": "GraphMem", "relationType": "decided_for"},
        {"from": "Use sqlite-vec for vectors", "to": "Semantic Search", "relationType": "enables"},
        {"from": "Lazy Loading", "to": "Semantic Search", "relationType": "optimizes"},
        {"from": "Tiered Retrieval", "to": "GraphMem", "relationType": "implemented_in"},
        {"from": "ULID for IDs", "to": "Event Sourcing", "relationType": "supports"},
    ])

    print(f"Created {len(engine.state.entities)} entities and {len(engine.state.relations)} relations")


def seed_sample_session(engine: MemoryEngine) -> None:
    """Seed with a sample development session."""

    engine.create_entities([
        {
            "name": "Auth Refactor",
            "entityType": "project",
            "observations": [
                "Refactoring authentication from session-based to JWT",
                "Started 2025-01-10",
            ],
        },
        {
            "name": "JWT vs Session",
            "entityType": "decision",
            "observations": [
                "Chose JWT for stateless API compatibility",
                "Session cookies have CORS issues with mobile app",
                "Trade-off: harder to revoke tokens",
            ],
        },
        {
            "name": "Token Refresh Flow",
            "entityType": "question",
            "observations": [
                "Should refresh tokens be stored in httpOnly cookies or localStorage?",
                "Security vs UX trade-off",
            ],
        },
    ])

    engine.create_relations([
        {"from": "Auth Refactor", "to": "JWT vs Session", "relationType": "decided"},
        {"from": "Token Refresh Flow", "to": "Auth Refactor", "relationType": "blocks"},
    ])

    print("Added sample session data")


def main():
    memory_path = Path(os.environ.get("MEMORY_PATH", ".claude/memory"))
    session_id = os.environ.get("SESSION_ID", "seed-script")

    print(f"Seeding memory at: {memory_path}")
    engine = MemoryEngine(memory_path, session_id)

    existing = len(engine.state.entities)
    if existing > 0:
        print(f"Warning: Memory already has {existing} entities")
        response = input("Continue and add more? [y/N] ")
        if response.lower() != "y":
            print("Aborted")
            return

    seed_basic_knowledge(engine)
    seed_sample_session(engine)

    print(f"\nFinal state: {len(engine.state.entities)} entities, {len(engine.state.relations)} relations")
    print(f"Log file: {memory_path / 'mnemograph.log'}")


if __name__ == "__main__":
    main()
