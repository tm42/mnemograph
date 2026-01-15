#!/usr/bin/env python3
"""UserPromptSubmit hook: Detect recall patterns and inject memory.

When a user's message suggests they want to recall something from memory,
this hook automatically searches mnemograph and injects relevant context.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from mnemograph_client import MnemographClient, detect_project_name
from settings import load_settings

# Patterns that suggest user wants to recall something
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
    (r"continue (with|on|from|where) we", "continuation"),
    (r"back to (the|that|our)", "return"),
    (r"as we discussed", "reference"),
]

# Patterns to EXCLUDE (common false positives)
EXCLUDE_PATTERNS = [
    r"what did we eat",
    r"what did we have for",
    r"do you remember (me|my name|who i am)",
    r"remind me (to|later|tomorrow)",  # Task reminders, not memory recall
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
            if topic:  # Only trigger if we found a meaningful topic
                return True, query_type, topic

    return False, "", ""


def extract_topic(prompt: str, start_pos: int) -> str:
    """Extract topic from text after the pattern match."""
    remainder = prompt[start_pos:].strip()

    # Remove common filler words and punctuation
    stopwords = {"the", "a", "an", "about", "for", "on", "with", "that", "this", "?", ""}
    words = [w.strip("?.,!") for w in remainder.split()]
    words = [w for w in words if w not in stopwords and len(w) > 2]

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


def output_result(context: str | None):
    """Output hook result as JSON."""
    result: dict = {"continue": True}
    if context:
        result["additionalContext"] = context
    print(json.dumps(result))


def main():
    """Main hook entry point."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    prompt = hook_input.get("prompt", "")
    cwd = hook_input.get("cwd", str(Path.cwd()))

    if not prompt:
        output_result(None)
        return

    # Load settings and check if recall detection is enabled
    settings = load_settings(cwd)
    if not settings.get("detect_recall_patterns", True):
        output_result(None)
        return

    # Detect recall intent
    should_search, query_type, topic = detect_recall_intent(prompt)

    if not should_search:
        output_result(None)
        return

    # Initialize client
    project_name = detect_project_name(cwd)
    client = MnemographClient(session_id=f"recall-{project_name}", cwd=cwd)

    if not client.available:
        output_result(None)
        return

    # Search memory and inject context
    content = search_memory(client, topic, query_type)

    if content:
        context = format_context(topic, content)
        output_result(context)
    else:
        output_result(None)


if __name__ == "__main__":
    main()
