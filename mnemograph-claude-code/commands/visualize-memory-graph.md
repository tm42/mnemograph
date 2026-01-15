---
description: Visualize the mnemograph knowledge graph in an interactive browser view
---

# Visualize Memory Graph

Open an interactive D3.js visualization of the knowledge graph in the browser.

## Your Task

Before running the visualization, ask the user which memory scope and mode they want using the AskUserQuestion tool.

## Instructions

1. **Ask the user for their preferences** using AskUserQuestion with these questions:

   **Question 1 - Memory Scope:**
   - Header: "Scope"
   - Question: "Which memory graph do you want to visualize?"
   - Options:
     - Label: "Global memory", Description: "Your personal knowledge graph stored in ~/.claude/memory (shared across all projects)"
     - Label: "Project memory", Description: "Project-specific knowledge graph stored in .claude/memory (local to this codebase)"

   **Question 2 - Mode:**
   - Header: "Mode"
   - Question: "How do you want to view the graph?"
   - Options:
     - Label: "Live view (Recommended)", Description: "Opens in browser with a Refresh button for live updates as you work"
     - Label: "Export only", Description: "Save the graph as an HTML file without opening the browser"

2. **Run the visualization** based on their choices:

   Use the Bash tool to run the `mnemograph graph` command. The mnemograph project is located at `${CLAUDE_PLUGIN_ROOT}/..` (the parent of this plugin directory).

   **Commands by choice combination:**

   | Scope | Mode | Command |
   |-------|------|---------|
   | Global | Live | `cd ${CLAUDE_PLUGIN_ROOT}/.. && uv run mnemograph --global graph --watch` |
   | Global | Export | `cd ${CLAUDE_PLUGIN_ROOT}/.. && uv run mnemograph --global graph --export ~/mnemograph-export.html` |
   | Project | Live | `cd ${CLAUDE_PLUGIN_ROOT}/.. && uv run mnemograph graph --watch` |
   | Project | Export | `cd ${CLAUDE_PLUGIN_ROOT}/.. && uv run mnemograph graph --export ./mnemograph-export.html` |

   **Run in background** for live view mode so the user can continue working while the server runs.

3. **Inform the user** about what's happening:
   - For live view: Tell them the server is running and they can click Refresh in the browser
   - For export: Tell them where the file was saved

## Notes

- The live view runs a local HTTP server — tell the user to press Ctrl+C or click "Stop Server" in the browser when done
- The visualization shows entities as nodes and relations as edges
- Node colors represent entity types (concept, decision, pattern, etc.)
- Edge thickness represents relation weight (stronger connections = thicker lines)
