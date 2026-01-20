# tests/unit/test_enrichment.py
"""Tests for the enrichment module."""

from __future__ import annotations

from fitz_ai.ingestion.enrichment import (
    CodeEnrichmentContext,
    ContentType,
    EnrichmentConfig,
    EnrichmentContext,
    EnrichmentPipeline,
)


class TestEnrichmentContext:
    """Tests for EnrichmentContext and subclasses."""

    def test_base_context_defaults(self):
        ctx = EnrichmentContext(file_path="/path/to/file.py")
        assert ctx.file_path == "/path/to/file.py"
        assert ctx.content_type == ContentType.UNKNOWN
        assert ctx.file_extension == ".py"
        assert ctx.metadata == {}

    def test_code_context_python(self):
        ctx = CodeEnrichmentContext(
            file_path="/path/to/module.py",
            language="python",
            imports=["os", "sys"],
            exports=["class Foo", "def bar"],
            used_by=[("/path/to/other.py", "test")],
        )
        assert ctx.content_type == ContentType.CODE
        assert ctx.language == "python"
        assert len(ctx.imports) == 2
        assert len(ctx.exports) == 2
        assert len(ctx.used_by) == 1


class TestEnrichmentPipeline:
    """Tests for the EnrichmentPipeline."""

    def test_pipeline_creation(self, tmp_path):
        """Test that pipeline can be created with default config."""
        config = EnrichmentConfig()
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=None,
        )

        assert not pipeline.chunk_enrichment_enabled  # No chat client
        assert pipeline.artifacts_enabled

    def test_pipeline_from_dict(self, tmp_path):
        """Test creating pipeline from dict config."""
        pipeline = EnrichmentPipeline.from_config(
            config={"artifacts": {"auto": True}},
            project_root=tmp_path,
        )

        assert pipeline.artifacts_enabled

    def test_generate_structural_artifacts(self, tmp_path):
        """Test generating structural artifacts (no LLM required)."""
        # Create a simple Python project
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "__init__.py").write_text("")
        (proj / "module.py").write_text(
            '''
"""A simple module."""

class MyClass:
    """My class."""
    pass

def my_func():
    """My function."""
    pass
'''
        )

        config = EnrichmentConfig()
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=proj,
            chat_client=None,
        )

        artifacts = pipeline.generate_structural_artifacts()

        # Should generate structural artifacts (not architecture_narrative which needs LLM)
        assert len(artifacts) > 0
        artifact_names = [a.artifact_type.value for a in artifacts]
        assert "navigation_index" in artifact_names
        assert "interface_catalog" in artifact_names
        assert "architecture_narrative" not in artifact_names


class TestArtifactPluginDiscovery:
    """Tests for artifact plugin discovery."""

    def test_plugins_discovered(self):
        """Test that all expected plugins are discovered."""
        from fitz_ai.ingestion.enrichment.artifacts.registry import get_artifact_registry

        registry = get_artifact_registry()
        plugin_names = registry.list_plugin_names()

        assert "navigation_index" in plugin_names
        assert "interface_catalog" in plugin_names
        assert "data_model_reference" in plugin_names
        assert "dependency_summary" in plugin_names
        assert "architecture_narrative" in plugin_names
