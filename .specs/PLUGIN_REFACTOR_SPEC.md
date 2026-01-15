# Mnemograph Claude Code Plugin Refactor Spec

**Status**: Draft
**Date**: January 2025
**Package**: `mnemograph-claude-code`

---

## Overview

The mnemograph Claude Code plugin provides automatic memory integration — context injection at session start, recall pattern detection, and save reminders. This spec covers gaps in the current implementation and performance improvements.

**Current state**: SessionStart and Stop hooks implemented, basic slash commands (`/recall`, `/remember`, `/visualize-memory-graph`).

**Goal**: Make memory feel seamless — CC "just knows" things from past sessions without explicit commands.

---

## Priority Matrix

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| UserPromptSubmit hook | High | Medium | **P0** |
| Engine warm-up optimization | Medium | Low | **P1** |
| Plugin settings schema | Medium | Low | **P1** |
| SessionEnd auto-summary | Low | Medium | P2 |
| PostToolUse co-access tracking | Low | High | P3 |

---

## P0: UserPromptSubmit Hook

### Problem

Users currently must explicitly call `/recall` or use MCP tools. The "magic" of persistent memory is lost when users have to remember to ask for it.

### Solution

Detect recall patterns in user prompts and automatically inject relevant memory context.

### Recall Patterns

```python
RECALL_PATTERNS = [
    # Explicit memory requests
    (r"what did we (decide|discuss|conclude|agree)", "decision"),
    (r"(last time|previously|before|earlier) we", "temporal"),
    (r"do you remember", "memory"),
    (r"remind me (about|of|what)", "reminder"),

    # Implicit recall (assumes shared context)
    (r"what do (you|we) know about", "knowledge"),
    (r"what('s| is) (the|our) (approach|decision|plan) (for|on|about)", "decision"),
    (r"why did we (choose|decide|go with)", "rationale"),

    # Continuity signals
    (r"continue (with|on|from|where)", "continuation"),
    (r"back to (the|that|our)", "return"),
    (r"as we discussed", "reference"),
]
```

### Implementation

**File**: `hooks/user_prompt.py`

```python
#!/usr/bin/env python3
"""UserPromptSubmit hook: Detect recall patterns and inject memory."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from mnemograph_client import MnemographClient, detect_project_name

RECALL_PATTERNS = [
    (r"what did we (decide|discuss|conclude|agree)", "decision"),
    (r"(last time|previously|before|earlier) we", "temporal"),
    (r"do you remember", "memory"),
    (r"remind me (about|of|what)", "reminder"),
    (r"what do (you|we) know about", "knowledge"),
    (r"what('s| is) (the|our) (approach|decision|plan)", "decision"),
    (r"why did we (choose|decide|go with)", "rationale"),
    (r"continue (with|on|from|where)", "continuation"),
    (r"back to (the|that|our)", "return"),
]

# Patterns to EXCLUDE (false positives)
EXCLUDE_PATTERNS = [
    r"what did we eat",
    r"what did we have for",
    r"do you remember (me|my name|who)",
]


def detect_recall_intent(prompt: str) -> tuple[bool, str, str]:
    """
    Detect if prompt suggests recall intent.

    Returns: (should_search, query_type, extracted_topic)
    """
    prompt_lower = prompt.lower().strip()

    # Check exclusions first
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, prompt_lower):
            return False, "", ""

    # Check recall patterns
    for pattern, query_type in RECALL_PATTERNS:
        match = re.search(pattern, prompt_lower)
        if match:
            topic = extract_topic(prompt_lower, match.end())
            if topic:  # Only trigger if we found a topic
                return True, query_type, topic

    return False, "", ""


def extract_topic(prompt: str, start_pos: int) -> str:
    """Extract topic from text after the pattern match."""
    remainder = prompt[start_pos:].strip()

    # Remove common filler words
    stopwords = {"the", "a", "an", "about", "for", "on", "with", "that", "this", "?"}
    words = [w.strip("?.,!") for w in remainder.split()
             if w.strip("?.,!") not in stopwords and len(w) > 2]

    # Take first 5 meaningful words as topic
    return " ".join(words[:5])


def search_memory(client: MnemographClient, topic: str, query_type: str) -> str | None:
    """Search mnemograph and return formatted context."""
    # Adjust query based on type
    if query_type == "decision":
        query = f"decision {topic}"
    elif query_type == "rationale":
        query = f"rationale why {topic}"
    else:
        query = topic

    result = client.recall(
        depth="medium",
        query=query,
        format="prose",
        timeout_seconds=3,  # Fast timeout for interactive use
    )

    if not result or not result.get("content"):
        return None

    content = result["content"].strip()
    if not content or content == "No relevant context found.":
        return None

    return content


def format_context(topic: str, content: str) -> str:
    """Format memory context for injection."""
    return f"""<mnemograph-recall query="{topic}">
{content}
</mnemograph-recall>"""


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    prompt = hook_input.get("prompt", "")
    cwd = hook_input.get("cwd", str(Path.cwd()))

    if not prompt:
        output_result(None)
        return

    should_search, query_type, topic = detect_recall_intent(prompt)

    if not should_search:
        output_result(None)
        return

    project_name = detect_project_name(cwd)
    client = MnemographClient(session_id=f"recall-{project_name}", cwd=cwd)

    if not client.available:
        output_result(None)
        return

    content = search_memory(client, topic, query_type)

    if content:
        context = format_context(topic, content)
        output_result(context)
    else:
        output_result(None)


def output_result(context: str | None):
    """Output hook result as JSON."""
    result: dict = {"continue": True}
    if context:
        result["additionalContext"] = context
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

### Hook Configuration

Update `hooks/hooks.json`:

```json
{
  "description": "Memory hooks for session lifecycle",
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/user_prompt.py",
            "timeout": 5000
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/stop_hook.py"
          }
        ]
      }
    ]
  }
}
```

### Example Flow

**User types**: "What did we decide about authentication?"

**Hook detects**:
- Pattern: `what did we decide`
- Query type: `decision`
- Topic: `authentication`

**Hook injects**:
```xml
<mnemograph-recall query="authentication">
**Decision: Use JWT for API auth** (decision)
Chose JWT over sessions for stateless API. Trade-off: need refresh token strategy.
- Refresh tokens stored in Redis with 7-day expiry
- Access tokens expire in 15 minutes

**auth-service** (project)
Handles authentication, OAuth integration, session management.
</mnemograph-recall>
```

**CC sees**: User's question + injected context → answers accurately.

### Performance Requirements

- **Timeout**: 3 seconds max (user is waiting)
- **Pattern matching**: <10ms (regex is fast)
- **Memory search**: <2s (uses existing `recall(depth="medium")`)
- **False positive rate**: <5% (tune patterns conservatively)

---

## P1: Engine Warm-Up Optimization

### Problem

First hook invocation loads `sentence-transformers` (~2-3 seconds cold start). This happens on every session start.

### Current Flow

```
SessionStart hook called
  → Import mnemograph.engine
    → Import sentence_transformers (2-3s)
  → Create MemoryEngine
  → Call recall()
  → Return context
```

### Solution 1: Lazy Vector Loading (Recommended)

Defer embedding model loading until actually needed for semantic search.

**File**: `src/mnemograph/vectors.py`

```python
class VectorIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._model = None  # Lazy load
        self._conn = None

    @property
    def model(self):
        """Lazy-load embedding model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search requires embeddings — triggers model load."""
        embedding = self.model.encode(query)  # First call loads model
        # ... rest of search
```

**Impact**: SessionStart with `depth="shallow"` won't load embeddings (no semantic search needed). Only `depth="medium"` or `depth="deep"` triggers the load.

### Solution 2: Background Pre-Warm (Alternative)

Start loading model in background thread immediately on import.

```python
import threading

_model_loader = None

def _preload_model():
    global _model_loader
    from sentence_transformers import SentenceTransformer
    _model_loader = SentenceTransformer("all-MiniLM-L6-v2")

# Start loading immediately on module import
threading.Thread(target=_preload_model, daemon=True).start()
```

**Trade-off**: Uses memory even if semantic search isn't needed.

### Recommendation

Implement Solution 1 (lazy loading) first. It's simpler and avoids unnecessary resource usage.

---

## P1: Plugin Settings Schema

### Problem

No way for users to configure plugin behavior. All settings are hardcoded.

### Solution

Add settings schema to `plugin.json` and read from `.claude/mnemograph.local.md`.

### Settings Schema

**File**: `.claude-plugin/plugin.json`

```json
{
  "name": "mnemograph",
  "version": "0.4.0",
  "description": "Persistent memory for Claude Code sessions via mnemograph knowledge graph",
  "author": {
    "name": "tm42",
    "email": "tm42@users.noreply.github.com"
  },
  "homepage": "https://github.com/tm42/mnemograph",
  "settings": {
    "auto_context_on_start": {
      "type": "boolean",
      "default": true,
      "description": "Load memory context automatically at session start"
    },
    "detect_recall_patterns": {
      "type": "boolean",
      "default": true,
      "description": "Detect recall patterns and inject memory automatically"
    },
    "prompt_to_save": {
      "type": "boolean",
      "default": true,
      "description": "Show reminder to save learnings when session ends"
    },
    "context_depth": {
      "type": "string",
      "enum": ["shallow", "medium"],
      "default": "shallow",
      "description": "How much context to load at session start"
    },
    "auto_commit": {
      "type": "boolean",
      "default": false,
      "description": "Auto-commit memory changes when session ends"
    }
  }
}
```

### Settings File

Users create `.claude/mnemograph.local.md` in their project:

```yaml
---
auto_context_on_start: true
detect_recall_patterns: true
prompt_to_save: false
context_depth: medium
auto_commit: false
---

# Project-specific memory notes

Any additional context for this project's memory usage.
```

### Reading Settings

**File**: `lib/settings.py`

```python
"""Plugin settings loader."""

from pathlib import Path
import yaml

DEFAULT_SETTINGS = {
    "auto_context_on_start": True,
    "detect_recall_patterns": True,
    "prompt_to_save": True,
    "context_depth": "shallow",
    "auto_commit": False,
}


def load_settings(cwd: str | Path) -> dict:
    """Load settings from .claude/mnemograph.local.md"""
    settings_path = Path(cwd) / ".claude" / "mnemograph.local.md"

    if not settings_path.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        content = settings_path.read_text()
        # Extract YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                user_settings = yaml.safe_load(parts[1]) or {}
                return {**DEFAULT_SETTINGS, **user_settings}
    except Exception:
        pass

    return DEFAULT_SETTINGS.copy()
```

### Hook Integration

```python
# In session_start.py
from settings import load_settings

def main():
    settings = load_settings(cwd)

    if not settings["auto_context_on_start"]:
        output_result(None)
        return

    depth = settings["context_depth"]
    # ... rest of hook
```

---

## P2: SessionEnd Auto-Summary (Deferred)

### Concept

When session ends cleanly, optionally create a summary observation in memory.

### Why Deferred

- Requires analyzing conversation transcript (complex)
- Risk of storing noise
- Stop hook reminder is sufficient for now

### Future Implementation

```python
# hooks/session_end.py
def main():
    settings = load_settings(cwd)

    if not settings.get("auto_summary", False):
        return

    # Analyze transcript for key learnings
    # Create observation on project entity
    # Commit if auto_commit enabled
```

---

## P3: PostToolUse Co-Access Tracking (Deferred)

### Concept

Track which entities are accessed together to strengthen relation weights.

### Why Deferred

- Requires PostToolUse hook support
- Complex to implement correctly
- Edge weight system already works via explicit importance

---

## File Structure After Refactor

```
mnemograph-claude-code/
├── .claude-plugin/
│   └── plugin.json              # With settings schema
├── agents/
│   └── memory-init.md           # Existing
├── commands/
│   ├── recall.md                # Existing
│   ├── remember.md              # Existing
│   └── visualize-memory-graph.md # Existing
├── hooks/
│   ├── hooks.json               # Updated with UserPromptSubmit
│   ├── session_start.py         # Existing (reads settings)
│   ├── user_prompt.py           # NEW
│   └── stop_hook.py             # Existing (reads settings)
├── lib/
│   ├── __init__.py
│   ├── mnemograph_client.py     # Existing
│   ├── pattern_detector.py      # NEW (extracted from user_prompt.py)
│   └── settings.py              # NEW
└── README.md
```

---

## Testing Plan

### Unit Tests

```python
# tests/test_pattern_detector.py

def test_detects_decision_recall():
    assert detect_recall_intent("What did we decide about auth?") == (
        True, "decision", "auth"
    )

def test_detects_temporal_recall():
    assert detect_recall_intent("Last time we discussed caching") == (
        True, "temporal", "discussed caching"
    )

def test_ignores_false_positives():
    assert detect_recall_intent("What did we eat for lunch?") == (
        False, "", ""
    )

def test_requires_topic():
    assert detect_recall_intent("What did we decide?") == (
        False, "", ""
    )
```

### Integration Tests

```bash
# Test UserPromptSubmit hook manually
echo '{"prompt": "What did we decide about authentication?", "cwd": "/path/to/project"}' \
  | python3 hooks/user_prompt.py
```

### Performance Tests

```python
# Measure pattern detection latency
import timeit

prompts = [
    "What did we decide about auth?",
    "Continue working on the API",
    "Fix the bug in login",  # Should not trigger
]

for prompt in prompts:
    time = timeit.timeit(lambda: detect_recall_intent(prompt), number=1000)
    print(f"{prompt[:30]}: {time*1000:.2f}ms per 1000 calls")
```

---

## Implementation Checklist

### Phase 1: UserPromptSubmit Hook
- [ ] Create `hooks/user_prompt.py`
- [ ] Extract `lib/pattern_detector.py`
- [ ] Update `hooks/hooks.json`
- [ ] Test with real prompts
- [ ] Tune patterns based on false positives

### Phase 2: Settings & Optimization
- [ ] Add settings schema to `plugin.json`
- [ ] Create `lib/settings.py`
- [ ] Update hooks to read settings
- [ ] Implement lazy vector loading in core
- [ ] Measure cold start improvement

### Phase 3: Documentation
- [ ] Update plugin README
- [ ] Add settings documentation
- [ ] Document pattern detection behavior

---

## Open Questions

1. **Pattern tuning**: How aggressive should recall detection be?
   - Conservative (fewer false positives, may miss valid recalls)
   - Aggressive (catches more, but may inject irrelevant context)
   - **Recommendation**: Start conservative, add patterns based on user feedback

2. **Context size in UserPromptSubmit**: How much to inject?
   - Too much slows down response and adds noise
   - Too little misses relevant context
   - **Recommendation**: Cap at ~500 tokens, use `depth="medium"`

3. **Settings file format**: YAML frontmatter vs pure YAML?
   - Frontmatter allows markdown notes
   - Pure YAML is simpler
   - **Recommendation**: Frontmatter (matches Claude Code conventions)
