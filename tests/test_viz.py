"""Tests for the graph visualization export module."""

import json
import tempfile
from pathlib import Path

from mnemograph.engine import MemoryEngine
from mnemograph.viz import export_graph_for_viz, get_viewer_html, ensure_viewer_exists, create_standalone_viewer


class TestExportGraphForViz:
    """Tests for export_graph_for_viz function."""

    def test_export_basic_graph(self):
        """Test basic export with entities and relations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            # Create test entities
            engine.create_entities([
                {"name": "Python", "entityType": "concept", "observations": ["A programming language"]},
                {"name": "FastAPI", "entityType": "project", "observations": ["Web framework"]},
            ])
            engine.create_relations([
                {"from": "FastAPI", "to": "Python", "relationType": "uses"},
            ])

            # Export
            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
            )

            assert output_path.exists()
            data = json.loads(output_path.read_text())

            # Check metadata
            assert data["meta"]["branch"] == "main"
            assert data["meta"]["entity_count"] == 2
            assert data["meta"]["relation_count"] == 1
            assert "exported_at" in data["meta"]

            # Check entities
            entity_names = {e["name"] for e in data["entities"]}
            assert entity_names == {"Python", "FastAPI"}

            # Check entity structure
            python = next(e for e in data["entities"] if e["name"] == "Python")
            assert python["type"] == "concept"
            assert "A programming language" in python["observations"]
            assert python["on_branch"] is True

            # Check relations
            assert len(data["relations"]) == 1
            rel = data["relations"][0]
            assert rel["type"] == "uses"
            assert rel["on_branch"] is True
            assert "weight" in rel

    def test_export_custom_output_path(self):
        """Test export to a custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            engine.create_entities([
                {"name": "Test", "entityType": "entity"},
            ])

            custom_path = memory_dir / "custom" / "export.json"
            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
                output_path=custom_path,
            )

            assert output_path == custom_path
            assert output_path.exists()

    def test_export_empty_graph(self):
        """Test export with no entities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
            )

            data = json.loads(output_path.read_text())
            assert data["meta"]["entity_count"] == 0
            assert data["meta"]["relation_count"] == 0
            assert data["entities"] == []
            assert data["relations"] == []

    def test_export_branch_filtering(self):
        """Test export with branch filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            # Create test entities
            entities = engine.create_entities([
                {"name": "A", "entityType": "concept"},
                {"name": "B", "entityType": "concept"},
                {"name": "C", "entityType": "concept"},
            ])
            result = engine.create_relations([
                {"from": "A", "to": "B", "relationType": "relates_to"},
                {"from": "B", "to": "C", "relationType": "relates_to"},
            ])
            relations = result["created"]

            # Get IDs
            entity_ids = {e.id for e in entities}
            a_id = next(e.id for e in entities if e.name == "A")
            b_id = next(e.id for e in entities if e.name == "B")
            ab_rel_id = next(r["id"] for r in relations if r["from_entity"] == a_id)

            # Export with branch filter (only A and B on branch)
            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
                branch_entity_ids={a_id, b_id},
                branch_relation_ids={ab_rel_id},
                branch_name="test-branch",
                include_context=True,
            )

            data = json.loads(output_path.read_text())
            assert data["meta"]["branch"] == "test-branch"

            # All 3 entities included (because include_context=True)
            assert data["meta"]["entity_count"] == 3

            # Check on_branch flags
            a = next(e for e in data["entities"] if e["name"] == "A")
            b = next(e for e in data["entities"] if e["name"] == "B")
            c = next(e for e in data["entities"] if e["name"] == "C")
            assert a["on_branch"] is True
            assert b["on_branch"] is True
            assert c["on_branch"] is False  # Ghost node

    def test_export_no_context(self):
        """Test export without off-branch context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            entities = engine.create_entities([
                {"name": "A", "entityType": "concept"},
                {"name": "B", "entityType": "concept"},
            ])

            a_id = next(e.id for e in entities if e.name == "A")

            # Export without context (only on-branch entities)
            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
                branch_entity_ids={a_id},
                include_context=False,
            )

            data = json.loads(output_path.read_text())
            assert data["meta"]["entity_count"] == 1  # Only A
            assert data["entities"][0]["name"] == "A"

    def test_export_truncates_observations(self):
        """Test that observations are truncated to 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            engine.create_entities([
                {
                    "name": "ManyObs",
                    "entityType": "concept",
                    "observations": ["obs1", "obs2", "obs3", "obs4", "obs5"],
                },
            ])

            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
            )

            data = json.loads(output_path.read_text())
            entity = data["entities"][0]
            assert len(entity["observations"]) == 3
            assert entity["observations"] == ["obs1", "obs2", "obs3"]

    def test_export_relation_weight_included(self):
        """Test that relation weights are included in export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            engine.create_entities([
                {"name": "A", "entityType": "concept"},
                {"name": "B", "entityType": "concept"},
            ])
            result = engine.create_relations([
                {"from": "A", "to": "B", "relationType": "connects"},
            ])
            relations = result["created"]

            # Set a custom weight
            engine.set_relation_importance(relations[0]["id"], 0.9)

            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
            )

            data = json.loads(output_path.read_text())
            rel = data["relations"][0]
            assert "weight" in rel
            # Weight should reflect the explicit weight we set
            assert rel["weight"] > 0


class TestViewerHtml:
    """Tests for the HTML viewer generation."""

    def test_get_viewer_html_returns_valid_html(self):
        """Test that get_viewer_html returns valid HTML structure."""
        html = get_viewer_html()

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "Mnemograph" in html
        assert "d3.v7.min.js" in html

    def test_get_viewer_html_contains_d3_code(self):
        """Test that the HTML contains D3.js visualization code."""
        html = get_viewer_html()

        assert "d3.forceSimulation" in html
        assert "d3.forceManyBody" in html
        assert "d3.forceLink" in html

    def test_get_viewer_html_contains_type_colors(self):
        """Test that entity type colors are defined."""
        html = get_viewer_html()

        assert "TYPE_COLORS" in html
        assert "concept" in html
        assert "decision" in html
        assert "project" in html

    def test_get_viewer_html_escapes_html(self):
        """Test that the viewer uses safe HTML escaping."""
        html = get_viewer_html()

        # Should have escapeHtml function for XSS prevention
        assert "escapeHtml" in html

    def test_get_viewer_html_has_controls(self):
        """Test that the viewer has UI controls."""
        html = get_viewer_html()

        assert 'id="search"' in html
        assert 'id="showGhosts"' in html
        assert 'id="showLabels"' in html


class TestEnsureViewerExists:
    """Tests for ensure_viewer_exists function."""

    def test_creates_viewer_if_not_exists(self):
        """Test that the viewer is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)

            viewer_path = ensure_viewer_exists(memory_dir)

            assert viewer_path.exists()
            assert viewer_path.name == "graph-viewer.html"
            assert "Mnemograph" in viewer_path.read_text()

    def test_returns_existing_viewer(self):
        """Test that existing viewer is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)

            # Create viewer first time
            viewer_path1 = ensure_viewer_exists(memory_dir)

            # Modify it
            original_content = viewer_path1.read_text()
            viewer_path1.write_text("MODIFIED")

            # Call again - should return same path without overwriting
            viewer_path2 = ensure_viewer_exists(memory_dir)

            assert viewer_path1 == viewer_path2
            assert viewer_path2.read_text() == "MODIFIED"

    def test_creates_viz_directory(self):
        """Test that viz directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)

            viewer_path = ensure_viewer_exists(memory_dir)

            assert (memory_dir / "viz").exists()
            assert viewer_path.parent == memory_dir / "viz"


class TestExportFilenames:
    """Tests for export filename generation."""

    def test_export_filename_format(self):
        """Test that export filenames follow the expected pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            engine.create_entities([
                {"name": "Test", "entityType": "entity"},
            ])

            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
            )

            # Should be in viz/exports directory
            assert "viz" in str(output_path)
            assert "exports" in str(output_path)
            assert output_path.suffix == ".json"

    def test_export_filename_includes_branch(self):
        """Test that branch name is included in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            engine = MemoryEngine(memory_dir, "test-session")

            engine.create_entities([
                {"name": "Test", "entityType": "entity"},
            ])

            output_path = export_graph_for_viz(
                state=engine.state,
                memory_dir=memory_dir,
                branch_name="feature/auth",
            )

            # Branch name should be in filename (with / replaced)
            assert "feature-auth" in output_path.name


class TestCreateStandaloneViewer:
    """Tests for create_standalone_viewer function."""

    def test_creates_standalone_html_with_embedded_data(self):
        """Test that standalone viewer embeds JSON data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "viewer.html"

            data = {
                "meta": {"branch": "main", "entity_count": 1, "relation_count": 0},
                "entities": [{"id": "e1", "name": "Test", "type": "concept", "observations": ["obs1"], "on_branch": True}],
                "relations": [],
            }

            result = create_standalone_viewer(data, output_path)

            assert result == output_path
            assert output_path.exists()

            content = output_path.read_text()
            # Should have graph-data script tag
            assert 'id="graph-data"' in content
            # Should contain our entity data
            assert "Test" in content
            assert "concept" in content

    def test_standalone_viewer_is_valid_html(self):
        """Test that standalone viewer produces valid HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "viewer.html"

            data = {
                "meta": {"branch": "test", "entity_count": 0, "relation_count": 0},
                "entities": [],
                "relations": [],
            }

            create_standalone_viewer(data, output_path)
            content = output_path.read_text()

            assert content.startswith("<!DOCTYPE html>")
            assert "</html>" in content
            assert "<script" in content

    def test_standalone_viewer_creates_parent_directories(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path" / "viewer.html"

            data = {"meta": {}, "entities": [], "relations": []}

            create_standalone_viewer(data, output_path)

            assert output_path.exists()
            assert (Path(tmpdir) / "nested" / "path").is_dir()

    def test_standalone_viewer_api_enabled_flag(self):
        """Test that api_enabled flag injects API_ENABLED variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "viewer.html"
            data = {"meta": {}, "entities": [], "relations": []}

            # Without api_enabled
            create_standalone_viewer(data, output_path, api_enabled=False)
            content = output_path.read_text()
            assert "window.API_ENABLED = false" in content

            # With api_enabled
            create_standalone_viewer(data, output_path, api_enabled=True)
            content = output_path.read_text()
            assert "window.API_ENABLED = true" in content
