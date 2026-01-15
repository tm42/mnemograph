---
description: Store knowledge in the mnemograph knowledge graph
argument-hint: [what to remember]
---

# Remember to Knowledge Graph

Use the mnemograph MCP server to store new knowledge in the knowledge graph.

## Your Task

The user wants to remember something. Their input: **$ARGUMENTS**

## Instructions

1. **Parse the input** to identify:
   - **What** is being remembered (the entity/concept)
   - **Type** of knowledge (decision, pattern, learning, concept, question, project)
   - **Key facts** (observations to store)
   - **Connections** (relations to existing entities)

2. **Check for duplicates first**:
   - Use `find_similar` with the entity name
   - If a similar entity exists (>80% match), use `add_observations` instead of creating new

3. **Store using the `remember` tool** with:
   - `name`: Clear, canonical name (e.g., "Decision: Use SQLite", "Pattern: Repository")
   - `entity_type`: One of: concept, decision, project, pattern, question, learning
   - `observations`: Array of atomic facts (one idea per observation)
   - `relations`: Connections to existing entities (use: implements, uses, part_of, decided_for, etc.)

4. **Confirm what was stored** and suggest related entities to connect.

## Entity Type Guidelines

| Type | Use For | Example |
|------|---------|---------|
| decision | Choices WITH rationale | "Decision: Use JWT for auth because stateless" |
| pattern | Recurring solutions | "Pattern: Error boundary in React components" |
| learning | Discoveries, gotchas | "Learning: SQLite WAL mode needs shared memory" |
| concept | Ideas, approaches | "Event sourcing", "CQRS" |
| question | Open unknowns | "Question: Should we add real-time sync?" |
| project | Codebases, systems | "mnemograph", "auth-service" |

## Example Usage

- `/remember we decided to use SQLite for simplicity` → creates decision entity
- `/remember gotcha: pytest must be run with python -m` → creates learning entity
- `/remember the API uses rate limiting pattern` → creates pattern, links to API
