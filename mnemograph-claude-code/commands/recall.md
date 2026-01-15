---
description: Query the mnemograph knowledge graph for relevant context
argument-hint: [query or topic]
---

# Recall from Knowledge Graph

Use the mnemograph MCP server to retrieve relevant context from the knowledge graph.

## Your Task

The user wants to recall information. Their query: **$ARGUMENTS**

## Instructions

1. **Determine recall depth** based on the query:
   - If the query is a quick orientation question (e.g., "what's in memory?", "overview") → use `depth: shallow`
   - If the query is task-specific (e.g., "auth decisions", "API patterns") → use `depth: medium`
   - If the query needs deep exploration (e.g., "everything about X", "full context") → use `depth: deep`

2. **Call the mnemograph recall tool** with:
   - `query`: The user's query (use $ARGUMENTS)
   - `depth`: As determined above
   - `format`: "prose" for human-readable output

3. **Present the results** clearly:
   - Summarize the key entities and their relationships
   - Highlight any decisions or learnings that are relevant
   - If the query found nothing useful, suggest what topics might exist

## Example Queries

- `/recall auth` → medium depth search for authentication-related knowledge
- `/recall` → shallow overview of what's in the graph
- `/recall everything about the API design` → deep exploration
