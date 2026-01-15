#!/usr/bin/env python3
"""Stop hook for mnemograph.

Outputs a reminder to save learnings at session end.
Optionally auto-commits memory changes if configured.
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
        hook_input = {}

    # Get context
    cwd = hook_input.get("cwd", str(Path.cwd()))
    project_name = detect_project_name(cwd)
    stop_reason = hook_input.get("stopReason", "unknown")

    # Load settings
    settings = load_settings(cwd)

    # Initialize client with cwd for proper memory detection
    client = MnemographClient(session_id=f"session-{project_name}", cwd=cwd)

    output_parts = []

    # Only show reminder for normal session ends (not errors/interrupts)
    # and if prompt_to_save is enabled
    if (stop_reason in ("end_turn", "stop_sequence", "unknown")
            and client.available
            and settings.get("prompt_to_save", True)):
        output_parts.append(format_learning_reminder(project_name))

    # Auto-commit if enabled
    if settings.get("auto_commit", False) and client.available:
        committed = client.commit(
            message=f"Auto-commit: session end ({project_name})",
            timeout_seconds=5,
        )
        if committed:
            output_parts.append("\n[mnemograph: changes auto-committed]")

    output_result("\n".join(output_parts) if output_parts else None)


def format_learning_reminder(project_name: str) -> str:
    """Format the learning reminder message."""
    return f"""---
**Session ending** - Any learnings worth remembering for {project_name}?

If you discovered patterns, made decisions, or hit gotchas, use the memory-store agent:

```xml
<store-request>
  <item content="what you learned" type_hint="learning|decision|pattern"/>
</store-request>
```

Or simply: `/remember <what you learned>`
---"""


def output_result(message: str | None):
    """Output hook result as JSON."""
    result: dict = {"continue": True}

    if message:
        result["output"] = message

    print(json.dumps(result))


if __name__ == "__main__":
    main()
