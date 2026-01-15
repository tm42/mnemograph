"""Plugin settings loader.

Reads settings from .claude/mnemograph.local.md (YAML frontmatter)
or falls back to defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_SETTINGS: dict[str, Any] = {
    "auto_context_on_start": True,
    "detect_recall_patterns": True,
    "prompt_to_save": True,
    "context_depth": "shallow",
    "auto_commit": False,
}


def load_settings(cwd: str | Path) -> dict[str, Any]:
    """Load settings from .claude/mnemograph.local.md

    Settings file format (YAML frontmatter):
    ```
    ---
    auto_context_on_start: true
    detect_recall_patterns: true
    prompt_to_save: false
    context_depth: medium
    auto_commit: false
    ---

    # Optional markdown notes below
    ```

    Returns merged settings (user settings override defaults).
    """
    settings_path = Path(cwd) / ".claude" / "mnemograph.local.md"

    if not settings_path.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        content = settings_path.read_text()

        # Extract YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                # Try to parse YAML
                try:
                    import yaml
                    user_settings = yaml.safe_load(parts[1]) or {}
                except ImportError:
                    # Fall back to simple parsing if yaml not available
                    user_settings = _parse_simple_yaml(parts[1])

                # Merge with defaults
                return {**DEFAULT_SETTINGS, **user_settings}
    except Exception:
        pass

    return DEFAULT_SETTINGS.copy()


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    """Simple YAML-like parser for basic key: value pairs.

    Used as fallback when PyYAML is not available.
    """
    result: dict[str, Any] = {}

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Parse value types
            if value.lower() == "true":
                result[key] = True
            elif value.lower() == "false":
                result[key] = False
            elif value.isdigit():
                result[key] = int(value)
            else:
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                result[key] = value

    return result
