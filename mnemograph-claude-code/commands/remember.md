---
description: Store knowledge in the mnemograph knowledge graph
argument-hint: [what to remember]
---

# Remember to Knowledge Graph

Store knowledge using the memory-store agent, which handles deduplication, canonical naming, and automatic relation creation.

## Your Task

The user wants to remember: **$ARGUMENTS**

## Instructions

1. **Parse the user's input** to identify:
   - What they want to remember (the content)
   - Any type hints (decision, pattern, learning, gotcha, question)
   - Any mentioned relations to existing concepts

2. **Spawn the memory-store agent** with a structured request:

```xml
<store-request>
  <item content="..." type_hint="..." related_to="..."/>
</store-request>
```

3. **Report the result** to the user:
   - What was stored (entity names, types)
   - Any duplicates that were merged
   - Any ambiguous matches the agent flagged

## Type Detection Hints

| User says... | type_hint |
|--------------|-----------|
| "we decided", "chose", "decision" | decision |
| "pattern", "approach", "we use" | pattern |
| "gotcha", "learned", "turns out", "TIL" | learning |
| "question", "should we", "wondering" | question |
| "project", "codebase", "repo" | project |
| Generic fact | concept |

## Examples

**User:** `/remember we decided to use JWT for auth`
**Action:** Spawn memory-store with:
```xml
<store-request>
  <item content="we decided to use JWT for auth" type_hint="decision"/>
</store-request>
```

**User:** `/remember gotcha: pytest needs python -m prefix`
**Action:** Spawn memory-store with:
```xml
<store-request>
  <item content="pytest needs python -m prefix" type_hint="learning"/>
</store-request>
```

**User:** `/remember the API uses rate limiting, and we chose Redis for the cache`
**Action:** Spawn memory-store with multiple items:
```xml
<store-request>
  <item content="API uses rate limiting" type_hint="pattern"/>
  <item content="chose Redis for cache" type_hint="decision" related_to="API"/>
</store-request>
```
