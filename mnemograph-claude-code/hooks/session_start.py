#!/usr/bin/env python3
"""SessionStart hook for mnemograph.

Injects memory context at the start of a Claude Code session.
Fails silently if mnemograph is unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from mnemograph_client import MnemographClient, detect_project_name
from settings import load_settings


def main():
    """Main hook entry point."""
    # Parse hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        # No input or invalid JSON - still try to provide context
        hook_input = {}

    # Get working directory from hook context
    cwd = hook_input.get("cwd", str(Path.cwd()))

    # Load settings
    settings = load_settings(cwd)

    # Check if auto context is enabled
    if not settings.get("auto_context_on_start", True):
        output_result(None)
        return

    project_name = detect_project_name(cwd)

    # Initialize client with cwd for proper memory detection
    client = MnemographClient(session_id=f"session-{project_name}", cwd=cwd)

    if not client.available:
        # Mnemograph not available - exit silently
        output_result(None)
        return

    # Get memory context at configured depth
    depth = settings.get("context_depth", "shallow")
    result = client.recall(
        depth=depth,
        query=project_name,
        format="prose",
        timeout_seconds=10,
    )

    if result and result.get("content"):
        context = format_context(project_name, result["content"])
        output_result(context)
    else:
        output_result(None)


def format_context(project_name: str, content: str) -> str:
    """Format memory context for injection."""
    return f"""<mnemograph-context project="{project_name}">
{content}
</mnemograph-context>"""


def output_result(context: str | None):
    """Output hook result as JSON."""
    result: dict = {"continue": True}

    if context:
        result["additionalContext"] = context

    print(json.dumps(result))


if __name__ == "__main__":
    main()
