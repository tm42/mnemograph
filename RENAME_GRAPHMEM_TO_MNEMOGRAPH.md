# Rename Project: graphmem → mnemograph

## Overview

Renaming the project from `graphmem` to `mnemograph` across:
- GitHub repository
- Python package name
- All code references
- CLI entry points
- Documentation

---

## Phase 1: GitHub Repository Rename

**Do this manually in GitHub UI:**

1. Go to https://github.com/tm42/graphmem
2. Settings → General → Repository name
3. Change `graphmem` → `mnemograph`
4. Click "Rename"

GitHub automatically redirects the old URL, but update your local remote:

```bash
cd /path/to/graphmem
git remote set-url origin git@github.com:tm42/mnemograph.git
```

---

## Phase 2: Code Rename (for Claude Code)

### 2.1 Directory Structure

```bash
# Rename the package directory
mv src/graphmem src/mnemograph

# If there's a memory_engine subdirectory, keep it as-is (it's internal)
```

### 2.2 pyproject.toml

Update these fields:

```toml
[project]
name = "mnemograph"  # was: graphmem
description = "Event-sourced knowledge graph memory for Claude Code..."

[project.scripts]
mnemograph = "mnemograph.server:main"           # was: graphmem
mnemograph-cli = "mnemograph.cli.main:cli"      # was: graphmem-cli  
claude-mem = "mnemograph.cli.vcs:cli"           # keep this one (it's the VCS CLI)

[project.urls]
Homepage = "https://github.com/tm42/mnemograph"
Repository = "https://github.com/tm42/mnemograph"
```

### 2.3 Python Imports

Find and replace in all `.py` files:

```
from graphmem → from mnemograph
import graphmem → import mnemograph
```

**Files to check:**
- `src/mnemograph/**/*.py` (all Python files)
- `tests/**/*.py`
- `scripts/**/*.py`
- `cli/**/*.py`

### 2.4 __init__.py

Update `src/mnemograph/__init__.py`:

```python
"""Mnemograph: Event-sourced knowledge graph memory for Claude Code."""

__version__ = "0.1.0"
# ... rest of exports
```

### 2.5 README.md

Update all references:

```markdown
# Mnemograph

<!-- mcp-name: io.github.tm42/mnemograph -->

pip install mnemograph

## CLI

mnemograph         # MCP server
mnemograph-cli     # Event operations
claude-mem         # Git-based VCS
```

Update the MCP config example:

```json
{
  "mcpServers": {
    "mnemograph": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mnemograph", "mnemograph"],
      "env": {
        "MEMORY_PATH": "/Users/YOU/.claude/memory"
      }
    }
  }
}
```

### 2.6 CLAUDE.md (if exists)

Update project references and any import examples.

### 2.7 MCP Server Name

In `src/mnemograph/server.py` (or wherever the MCP server is defined):

```python
# Update server name/description
server = Server(
    name="mnemograph",
    description="Event-sourced knowledge graph memory for Claude Code"
)
```

### 2.8 CLI Help Text

In `cli/main.py` or similar:

```python
@click.group()
def cli():
    """Mnemograph - Event-sourced knowledge graph memory."""
    pass
```

---

## Phase 3: Verification

### 3.1 Test imports work

```bash
cd /path/to/mnemograph
uv sync
python -c "import mnemograph; print(mnemograph.__version__)"
```

### 3.2 Test CLI entry points

```bash
uv run mnemograph --help
uv run mnemograph-cli --help
uv run claude-mem --help
```

### 3.3 Run tests

```bash
uv run pytest
```

### 3.4 Check for any remaining "graphmem" references

```bash
grep -r "graphmem" --include="*.py" --include="*.md" --include="*.toml" --include="*.json" .
```

---

## Phase 4: Commit and Push

```bash
git add -A
git commit -m "Rename project: graphmem → mnemograph

- Rename package directory
- Update all imports
- Update pyproject.toml (name, scripts, URLs)
- Update README and docs
- Update MCP server name
"
git push origin main
```

---

## Quick Reference: Old → New

| Old | New |
|-----|-----|
| `graphmem` (repo) | `mnemograph` |
| `graphmem` (package) | `mnemograph` |
| `graphmem` (CLI) | `mnemograph` |
| `graphmem-cli` | `mnemograph-cli` |
| `claude-mem` | `claude-mem` (unchanged) |
| `from graphmem import` | `from mnemograph import` |
| `io.github.tm42/graphmem` | `io.github.tm42/mnemograph` |

---

## Files Checklist

- [ ] `src/graphmem/` → `src/mnemograph/`
- [ ] `pyproject.toml` - name, scripts, URLs
- [ ] `src/mnemograph/__init__.py` - docstring, version
- [ ] `src/mnemograph/server.py` - server name
- [ ] `cli/main.py` - CLI group docstring
- [ ] `README.md` - all references
- [ ] `CLAUDE.md` - if exists
- [ ] `tests/**/*.py` - imports
- [ ] `.github/workflows/*.yml` - if any reference the name
- [ ] Any config files (`.json`, `.yaml`)

---

## Notes

- GitHub redirects old repo URL automatically (for a while)
- PyPI name `mnemograph` should be available (verify before publish)
- The `claude-mem` CLI name stays the same (it's user-facing and memorable)
- Old `graphmem` installs will fail after PyPI publish (that's fine, it was never published)
