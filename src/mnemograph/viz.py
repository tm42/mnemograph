"""Graph visualization export for D3.js viewer.

Exports the knowledge graph to JSON format suitable for the interactive
D3.js-based HTML viewer. Supports branch filtering and ghost nodes.
"""

import json
from datetime import datetime
from pathlib import Path

from .state import GraphState


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
    """
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Mnemograph Viewer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }

        #container {
            width: 100vw;
            height: 100vh;
            position: relative;
        }

        svg {
            width: 100%;
            height: 100%;
        }

        /* Controls panel */
        #controls {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(30, 30, 50, 0.95);
            padding: 16px;
            border-radius: 8px;
            min-width: 240px;
            z-index: 100;
        }

        #controls h2 {
            font-size: 14px;
            margin-bottom: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        #controls input[type="text"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #444;
            border-radius: 4px;
            background: #2a2a4a;
            color: #eee;
            margin-bottom: 12px;
        }

        #controls label {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            cursor: pointer;
        }

        #controls input[type="checkbox"] {
            width: 16px;
            height: 16px;
        }

        .stats {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #444;
            font-size: 12px;
            color: #888;
        }

        /* Info panel (hover details) */
        #info {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(30, 30, 50, 0.95);
            padding: 16px;
            border-radius: 8px;
            max-width: 320px;
            z-index: 100;
            display: none;
        }

        #info.visible { display: block; }

        #info h3 {
            font-size: 16px;
            margin-bottom: 8px;
        }

        #info .type {
            font-size: 11px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 12px;
        }

        #info .observations {
            font-size: 13px;
            line-height: 1.5;
            color: #ccc;
        }

        #info .observation {
            margin-bottom: 8px;
            padding-left: 12px;
            border-left: 2px solid #444;
        }

        /* Legend */
        #legend {
            position: absolute;
            bottom: 16px;
            left: 16px;
            background: rgba(30, 30, 50, 0.95);
            padding: 12px 16px;
            border-radius: 8px;
            display: flex;
            gap: 16px;
            font-size: 12px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        /* Graph elements */
        .node {
            cursor: pointer;
        }

        .node circle {
            stroke: #fff;
            stroke-width: 1.5px;
        }

        .node.ghost circle {
            stroke: #666;
        }

        .node text {
            font-size: 11px;
            fill: #fff;
            pointer-events: none;
        }

        .node.ghost text {
            fill: #666;
        }

        .node.dimmed {
            opacity: 0.2;
        }

        .link {
            stroke: #666;
            stroke-opacity: 0.6;
        }

        .link.ghost {
            stroke-opacity: 0.2;
        }

        .link.highlighted {
            stroke: #fff;
            stroke-opacity: 1;
        }

        /* Tooltip */
        #tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 200;
            display: none;
        }
    </style>
</head>
<body>
    <div id="container">
        <svg></svg>

        <div id="controls">
            <h2>Mnemograph</h2>
            <input type="text" id="search" placeholder="Search entities...">
            <label>
                <input type="checkbox" id="showGhosts" checked>
                Show off-branch entities
            </label>
            <label>
                <input type="checkbox" id="showLabels" checked>
                Show labels
            </label>
            <div class="stats">
                <div id="branchName">Branch: main</div>
                <div id="entityCount">Entities: 0</div>
                <div id="relationCount">Relations: 0</div>
            </div>
        </div>

        <div id="info">
            <h3 id="infoName"></h3>
            <div class="type" id="infoType"></div>
            <div class="observations" id="infoObservations"></div>
        </div>

        <div id="legend"></div>

        <div id="tooltip"></div>
    </div>

    <script>
        // Entity type colors
        const TYPE_COLORS = {
            concept: '#6366f1',    // Indigo
            decision: '#f59e0b',   // Amber
            project: '#10b981',    // Emerald
            pattern: '#8b5cf6',    // Violet
            question: '#ef4444',   // Red
            learning: '#06b6d4',   // Cyan
            entity: '#6b7280',     // Gray
            default: '#6b7280'     // Gray
        };

        // State
        let graphData = null;
        let simulation = null;
        let selectedNode = null;

        // Escape HTML to prevent XSS
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Load data from file or embedded
        async function loadData() {
            // Check for data in URL hash
            const hash = window.location.hash.slice(1);
            if (hash) {
                try {
                    const response = await fetch(hash);
                    return await response.json();
                } catch (e) {
                    console.error('Failed to load:', hash, e);
                }
            }

            // Check for embedded data
            const embedded = document.getElementById('graph-data');
            if (embedded) {
                return JSON.parse(embedded.textContent);
            }

            // Prompt for file
            return new Promise((resolve) => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.json';
                input.onchange = async (e) => {
                    const file = e.target.files[0];
                    const text = await file.text();
                    resolve(JSON.parse(text));
                };
                input.click();
            });
        }

        // Initialize visualization
        async function init() {
            graphData = await loadData();

            // Update stats
            document.getElementById('branchName').textContent = `Branch: ${escapeHtml(graphData.meta.branch)}`;
            document.getElementById('entityCount').textContent = `Entities: ${graphData.meta.entity_count}`;
            document.getElementById('relationCount').textContent = `Relations: ${graphData.meta.relation_count}`;

            // Build legend
            const types = [...new Set(graphData.entities.map(e => e.type))];
            const legend = document.getElementById('legend');
            legend.replaceChildren();
            types.forEach(type => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                const dot = document.createElement('div');
                dot.className = 'legend-dot';
                dot.style.background = TYPE_COLORS[type] || TYPE_COLORS.default;
                const label = document.createElement('span');
                label.textContent = type;
                item.appendChild(dot);
                item.appendChild(label);
                legend.appendChild(item);
            });

            render();
            setupControls();
        }

        function render() {
            const svg = d3.select('svg');
            const width = window.innerWidth;
            const height = window.innerHeight;

            svg.selectAll('*').remove();

            // Create container for zoom
            const g = svg.append('g');

            // Setup zoom
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => g.attr('transform', event.transform));

            svg.call(zoom);

            // Filter based on ghost visibility
            const showGhosts = document.getElementById('showGhosts').checked;
            const entities = showGhosts
                ? graphData.entities
                : graphData.entities.filter(e => e.on_branch);
            const entityIds = new Set(entities.map(e => e.id));
            const relations = graphData.relations.filter(
                r => entityIds.has(r.from) && entityIds.has(r.to)
            );

            // Create node map
            const nodeMap = new Map(entities.map(e => [e.id, {...e}]));

            // Calculate node degrees
            const degrees = new Map();
            relations.forEach(r => {
                degrees.set(r.from, (degrees.get(r.from) || 0) + 1);
                degrees.set(r.to, (degrees.get(r.to) || 0) + 1);
            });

            // Create links with source/target objects
            const links = relations.map(r => ({
                ...r,
                source: nodeMap.get(r.from),
                target: nodeMap.get(r.to)
            }));

            const nodes = Array.from(nodeMap.values());

            // Create simulation
            simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30));

            // Draw links
            const link = g.append('g')
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('class', d => `link ${d.on_branch ? '' : 'ghost'}`)
                .attr('stroke-width', d => 1 + d.weight * 3);

            // Draw nodes
            const node = g.append('g')
                .selectAll('g')
                .data(nodes)
                .join('g')
                .attr('class', d => `node ${d.on_branch ? '' : 'ghost'}`)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));

            // Node circles
            node.append('circle')
                .attr('r', d => 8 + Math.sqrt(degrees.get(d.id) || 1) * 3)
                .attr('fill', d => TYPE_COLORS[d.type] || TYPE_COLORS.default)
                .attr('fill-opacity', d => d.on_branch ? 1 : 0.2);

            // Node labels
            node.append('text')
                .attr('dx', 12)
                .attr('dy', 4)
                .text(d => d.name)
                .attr('class', 'label');

            // Hover handlers
            node.on('mouseover', (event, d) => showInfo(d))
                .on('mouseout', () => hideInfo())
                .on('click', (event, d) => selectNode(d, link, node));

            link.on('mouseover', (event, d) => showTooltip(event, d.type))
                .on('mouseout', hideTooltip);

            // Simulation tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });

            // Initial zoom to fit
            setTimeout(() => {
                const bounds = g.node().getBBox();
                if (bounds.width > 0 && bounds.height > 0) {
                    const scale = 0.8 * Math.min(width / bounds.width, height / bounds.height);
                    const tx = width / 2 - scale * (bounds.x + bounds.width / 2);
                    const ty = height / 2 - scale * (bounds.y + bounds.height / 2);
                    svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
                }
            }, 500);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        function showInfo(entity) {
            const info = document.getElementById('info');
            document.getElementById('infoName').textContent = entity.name;
            document.getElementById('infoType').textContent = entity.type;

            // Build observations safely
            const obsContainer = document.getElementById('infoObservations');
            obsContainer.replaceChildren();
            entity.observations.forEach(obs => {
                const div = document.createElement('div');
                div.className = 'observation';
                div.textContent = obs;
                obsContainer.appendChild(div);
            });

            info.classList.add('visible');
        }

        function hideInfo() {
            if (!selectedNode) {
                document.getElementById('info').classList.remove('visible');
            }
        }

        function selectNode(d, links, nodes) {
            if (selectedNode === d) {
                // Deselect
                selectedNode = null;
                links.classed('highlighted', false);
                nodes.classed('selected', false);
                hideInfo();
            } else {
                // Select
                selectedNode = d;
                showInfo(d);

                // Highlight connected links
                links.classed('highlighted', l =>
                    l.source.id === d.id || l.target.id === d.id
                );
            }
        }

        function showTooltip(event, text) {
            const tooltip = document.getElementById('tooltip');
            tooltip.textContent = text;
            tooltip.style.left = event.pageX + 10 + 'px';
            tooltip.style.top = event.pageY + 10 + 'px';
            tooltip.style.display = 'block';
        }

        function hideTooltip() {
            document.getElementById('tooltip').style.display = 'none';
        }

        function setupControls() {
            // Search
            document.getElementById('search').addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                d3.selectAll('.node').classed('dimmed', d =>
                    query && !d.name.toLowerCase().includes(query)
                );
            });

            // Show ghosts toggle
            document.getElementById('showGhosts').addEventListener('change', render);

            // Show labels toggle
            document.getElementById('showLabels').addEventListener('change', (e) => {
                d3.selectAll('.label').style('display', e.target.checked ? 'block' : 'none');
            });
        }

        // Handle window resize
        window.addEventListener('resize', () => {
            if (simulation) {
                simulation.force('center', d3.forceCenter(window.innerWidth / 2, window.innerHeight / 2));
                simulation.alpha(0.3).restart();
            }
        });

        // Start
        init();
    </script>
</body>
</html>'''


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
