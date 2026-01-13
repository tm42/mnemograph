"""Graph visualization export for D3.js viewer.

Exports the knowledge graph to JSON format suitable for the interactive
D3.js-based HTML viewer. Supports branch filtering and ghost nodes.
"""

import json
from datetime import datetime
from pathlib import Path

from ..state import GraphState

# Path to the HTML template
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "graph.html"


def export_graph_for_viz(
    state: GraphState,
    memory_dir: Path,
    branch_entity_ids: set[str] | None = None,
    branch_relation_ids: set[str] | None = None,
    branch_name: str = "main",
    include_context: bool = True,
    output_path: Path | None = None,
) -> Path:
    """Export graph state for D3 visualization.

    Args:
        state: Full graph state
        memory_dir: Base memory directory
        branch_entity_ids: Set of entity IDs on the current branch (None = all)
        branch_relation_ids: Set of relation IDs on the current branch (None = all)
        branch_name: Name of the current branch for metadata
        include_context: Include off-branch entities as ghost nodes
        output_path: Where to write JSON (default: auto-generated)

    Returns:
        Path to exported JSON file
    """
    entities = []
    for eid, entity in state.entities.items():
        on_branch = branch_entity_ids is None or eid in branch_entity_ids

        if not on_branch and not include_context:
            continue

        # Get first few observations as text
        obs_texts = [obs.text for obs in entity.observations[:3]]

        entities.append({
            "id": eid,
            "name": entity.name,
            "type": entity.type,
            "observations": obs_texts,
            "on_branch": on_branch,
        })

    # Build set of included entity IDs for relation filtering
    included_ids = {e["id"] for e in entities}

    relations = []
    for rel in state.relations:
        if rel.from_entity not in included_ids or rel.to_entity not in included_ids:
            continue

        on_branch = branch_relation_ids is None or rel.id in branch_relation_ids

        if not on_branch and not include_context:
            continue

        relations.append({
            "id": rel.id,
            "from": rel.from_entity,
            "to": rel.to_entity,
            "type": rel.type,
            "weight": rel.weight,
            "on_branch": on_branch,
        })

    data = {
        "meta": {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "branch": branch_name,
            "entity_count": len(entities),
            "relation_count": len(relations),
        },
        "entities": entities,
        "relations": relations,
    }

    if output_path is None:
        viz_dir = memory_dir / "viz" / "exports"
        viz_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        branch_slug = branch_name.replace("/", "-")
        output_path = viz_dir / f"{branch_slug}-{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    return output_path


def get_viewer_html() -> str:
    """Return the self-contained HTML viewer with embedded D3.js.

    This viewer can be used either:
    1. With a JSON file path in the URL hash: viewer.html#path/to/data.json
    2. With embedded data via a script tag with id="graph-data"
    3. With a file picker if neither is available

    The template is loaded from templates/graph.html.
    """
    return _TEMPLATE_PATH.read_text()


def ensure_viewer_exists(memory_dir: Path) -> Path:
    """Ensure the viewer HTML exists in the memory directory.

    Args:
        memory_dir: Base memory directory

    Returns:
        Path to the viewer HTML file
    """
    viz_dir = memory_dir / "viz"
    viewer_path = viz_dir / "graph-viewer.html"

    if not viewer_path.exists():
        viz_dir.mkdir(parents=True, exist_ok=True)
        viewer_path.write_text(get_viewer_html())

    return viewer_path


def create_standalone_viewer(data: dict, output_path: Path, api_enabled: bool = False) -> Path:
    """Create a standalone HTML viewer with embedded data.

    This avoids file:// CORS issues by embedding the JSON directly
    into the HTML file as a script tag.

    Args:
        data: Graph data dict with entities and relations
        output_path: Where to write the HTML file
        api_enabled: If True, show refresh button for live reload

    Returns:
        Path to the created HTML file
    """
    html = get_viewer_html()

    # Embed data as a script tag before the closing </body>
    data_json = json.dumps(data)
    api_flag = "true" if api_enabled else "false"
    embedded_data = f'<script>window.API_ENABLED = {api_flag};</script>\n'
    embedded_data += f'<script type="application/json" id="graph-data">{data_json}</script>'

    # Insert before </body>
    html = html.replace("</body>", f"{embedded_data}\n</body>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
