# Publish Mnemograph to MCP Registry

## Prerequisites
- Package must be on PyPI first
- GitHub account (for authentication)
- Repo renamed to `mnemograph` (see RENAME_GRAPHMEM_TO_MNEMOGRAPH.md)

## Step 1: Publish to PyPI

```bash
uv build
uv publish
```

## Step 2: Add validation metadata

Add this line anywhere in your README.md (can be in a comment):

```markdown
<!-- mcp-name: io.github.tm42/mnemograph -->
```

Commit and push.

## Step 3: Install publisher CLI

```bash
curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz" | tar xz mcp-publisher
sudo mv mcp-publisher /usr/local/bin/
```

## Step 4: Create server.json in repo root

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-07-09/server.schema.json",
  "name": "io.github.tm42/mnemograph",
  "description": "Event-sourced knowledge graph memory for Claude Code with semantic search, tiered retrieval, and git-based version control",
  "repository": {
    "url": "https://github.com/tm42/mnemograph",
    "source": "github"
  },
  "version": "0.1.0",
  "packages": [
    {
      "registry_type": "pypi",
      "identifier": "mnemograph",
      "version": "0.1.0",
      "runtime": "python",
      "runtime_arguments": ["-m", "mnemograph"],
      "transport": {
        "type": "stdio"
      },
      "environment_variables": [
        {
          "description": "Path to memory storage directory",
          "name": "MEMORY_PATH",
          "is_required": false,
          "default": "~/.claude/memory"
        }
      ]
    }
  ]
}
```

## Step 5: Login and publish

```bash
# Authenticate (opens browser for GitHub OAuth)
mcp-publisher login

# Publish to registry
mcp-publisher publish
```

## Step 6 (Optional): GitHub Actions for auto-publish

Create `.github/workflows/publish-mcp.yml`:

```yaml
name: Publish to MCP Registry
on:
  push:
    tags: ["v*"]

permissions:
  id-token: write
  contents: read

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install mcp-publisher
        run: |
          curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_linux_amd64.tar.gz" | tar xz
          sudo mv mcp-publisher /usr/local/bin/
      
      - name: Login via GitHub OIDC
        run: mcp-publisher login github-oidc
      
      - name: Publish
        run: mcp-publisher publish
```

Then: `git tag v0.1.0 && git push --tags`

## Verify

After publishing, check:
- https://registry.modelcontextprotocol.io (search for mnemograph)
- `curl https://registry.modelcontextprotocol.io/v0/servers/io.github.tm42/mnemograph`

## Checklist

- [ ] Repo renamed to `mnemograph` on GitHub
- [ ] All code references updated (see RENAME_GRAPHMEM_TO_MNEMOGRAPH.md)
- [ ] Package published to PyPI as `mnemograph`
- [ ] `<!-- mcp-name: io.github.tm42/mnemograph -->` in README.md  
- [ ] `server.json` in repo root
- [ ] `mcp-publisher login`
- [ ] `mcp-publisher publish`
- [ ] (optional) GitHub Actions workflow for auto-publish on tags
