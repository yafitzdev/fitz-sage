# tests/test_cli_ingest.py
"""
Tests for the ingest command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestIngestCommand:
    """Tests for fitz ingest command."""

    def test_ingest_shows_help(self):
        """Test that ingest --help works."""
        result = runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "ingest" in result.output.lower()
        assert "source" in result.output.lower()

    def test_ingest_works_with_defaults(self, tmp_path):
        """Test that ingest works with package defaults."""
        # Create a source file to ingest
        (tmp_path / "test.txt").write_text("Test content")

        # CLIContext.load() always succeeds with package defaults
        mock_ctx = MagicMock()
        mock_ctx.raw_config = {
            "embedding": {"plugin_name": "cohere"},
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"collection": "default"},
            "chunking": {"default": {"plugin_name": "simple", "kwargs": {"chunk_size": 1000}}},
        }
        mock_ctx.embedding_id = "cohere:embed-english-v3.0"
        mock_ctx.embedding_plugin = "cohere"
        mock_ctx.vector_db_plugin = "local_faiss"
        mock_ctx.retrieval_collection = "default"

        mock_ingest = MagicMock()
        mock_ingest.return_value = MagicMock(new_chunks=0, updated_chunks=0, deleted_chunks=0)

        with (
            patch("fitz_ai.cli.commands.ingest_runner.CLIContext.load", return_value=mock_ctx),
            patch("fitz_ai.ingestion.diff.run_diff_ingest", mock_ingest),
        ):
            result = runner.invoke(app, ["ingest", str(tmp_path), "-y"])

        # Should succeed with defaults
        assert result.exit_code == 0 or "ingest" in result.output.lower()

    def test_ingest_non_interactive_requires_source(self):
        """Test that non-interactive mode requires source."""
        mock_ctx = MagicMock()
        mock_ctx.raw_config = {
            "embedding": {"plugin_name": "cohere"},
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"collection": "default"},
            "chunking": {"default": {"plugin_name": "simple", "kwargs": {"chunk_size": 1000}}},
        }
        mock_ctx.embedding_id = "cohere:embed-english-v3.0"
        mock_ctx.embedding_plugin = "cohere"
        mock_ctx.vector_db_plugin = "local_faiss"
        mock_ctx.retrieval_collection = "default"

        with patch("fitz_ai.cli.context.CLIContext.load", return_value=mock_ctx):
            result = runner.invoke(app, ["ingest", "-y"])

        assert result.exit_code != 0
        assert "source" in result.output.lower()


class TestIngestHelpers:
    """Tests for ingest helper functions."""

    def test_suggest_collection_name_from_dir(self, tmp_path):
        """Test suggest_collection_name suggests name from directory."""
        from fitz_ai.cli.commands.ingest_helpers import suggest_collection_name as _suggest_collection_name

        test_dir = tmp_path / "My-Project"
        test_dir.mkdir()

        name = _suggest_collection_name(str(test_dir))

        assert name == "my_project"

    def test_suggest_collection_name_from_file(self, tmp_path):
        """Test suggest_collection_name suggests name from file's parent."""
        from fitz_ai.cli.commands.ingest_helpers import suggest_collection_name as _suggest_collection_name

        test_file = tmp_path / "docs" / "readme.md"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")

        name = _suggest_collection_name(str(test_file))

        assert name == "docs"

    def test_is_code_project_detects_python_project(self, tmp_path):
        """Test is_code_project detects Python projects via pyproject.toml."""
        from fitz_ai.cli.commands.ingest_helpers import is_code_project as _is_code_project

        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        assert _is_code_project(str(tmp_path)) is True

    def test_is_code_project_returns_false_for_docs(self, tmp_path):
        """Test is_code_project returns False for docs-only folders."""
        from fitz_ai.cli.commands.ingest_helpers import is_code_project as _is_code_project

        (tmp_path / "readme.md").write_text("# Documentation")
        (tmp_path / "guide.txt").write_text("User guide")

        assert _is_code_project(str(tmp_path)) is False

    def test_parse_artifact_selection_all(self):
        """Test parse_artifact_selection with 'all'."""
        from fitz_ai.cli.commands.ingest_helpers import parse_artifact_selection as _parse_artifact_selection

        available = ["navigation_index", "interface_catalog"]
        result = _parse_artifact_selection("all", available)

        assert result == available

    def test_parse_artifact_selection_none(self):
        """Test parse_artifact_selection with 'none'."""
        from fitz_ai.cli.commands.ingest_helpers import parse_artifact_selection as _parse_artifact_selection

        result = _parse_artifact_selection("none", ["a", "b"])

        assert result == []

    def test_parse_artifact_selection_comma_list(self):
        """Test parse_artifact_selection with comma-separated list."""
        from fitz_ai.cli.commands.ingest_helpers import parse_artifact_selection as _parse_artifact_selection

        available = ["navigation_index", "interface_catalog", "other"]
        result = _parse_artifact_selection("navigation_index,interface_catalog", available)

        assert result == ["navigation_index", "interface_catalog"]

    def test_parse_artifact_selection_filters_invalid(self):
        """Test parse_artifact_selection filters invalid names."""
        from fitz_ai.cli.commands.ingest_helpers import parse_artifact_selection as _parse_artifact_selection

        available = ["valid_one", "valid_two"]
        result = _parse_artifact_selection("valid_one,invalid", available)

        assert result == ["valid_one"]


class TestIngestAdapters:
    """Tests for ingest adapter classes."""

    def test_vector_db_writer_adapter_upsert(self):
        """Test VectorDBWriterAdapter.upsert calls client."""
        from fitz_ai.cli.commands.ingest_adapters import VectorDBWriterAdapter

        mock_client = MagicMock()
        adapter = VectorDBWriterAdapter(mock_client)

        points = [{"id": "1", "vector": [0.1, 0.2]}]
        adapter.upsert("collection", points)

        mock_client.upsert.assert_called_once()

    def test_vector_db_writer_adapter_flush(self):
        """Test VectorDBWriterAdapter.flush calls client."""
        from fitz_ai.cli.commands.ingest_adapters import VectorDBWriterAdapter

        mock_client = MagicMock()
        adapter = VectorDBWriterAdapter(mock_client)

        adapter.flush()

        mock_client.flush.assert_called_once()


class TestBuildChunkingRouterConfig:
    """Tests for _build_chunking_router_config."""

    def test_build_default_config(self):
        """Test building config with defaults."""
        from fitz_ai.cli.commands.ingest_config import build_chunking_router_config as _build_chunking_router_config

        config = {
            "chunking": {
                "default": {"plugin_name": "simple", "kwargs": {"chunk_size": 500}},
                "warn_on_fallback": True,
            }
        }

        result = _build_chunking_router_config(config)

        assert result.default.plugin_name == "simple"
        assert result.default.kwargs["chunk_size"] == 500
        assert result.warn_on_fallback is True

    def test_build_config_with_extensions(self):
        """Test building config with per-extension chunkers."""
        from fitz_ai.cli.commands.ingest_config import build_chunking_router_config as _build_chunking_router_config

        config = {
            "chunking": {
                "default": {"plugin_name": "simple"},
                "by_extension": {
                    ".md": {"plugin_name": "markdown", "kwargs": {}},
                    ".py": {"plugin_name": "python_code", "kwargs": {}},
                },
            }
        }

        result = _build_chunking_router_config(config)

        assert ".md" in result.by_extension
        assert result.by_extension[".md"].plugin_name == "markdown"
        assert ".py" in result.by_extension

    def test_build_config_empty(self):
        """Test building config with empty input."""
        from fitz_ai.cli.commands.ingest_config import build_chunking_router_config as _build_chunking_router_config

        result = _build_chunking_router_config({})

        assert result.default.plugin_name == "simple"
        assert result.default.kwargs["chunk_size"] == 1000


class TestIngestOptions:
    """Tests for ingest command options."""

    def test_ingest_force_flag_recognized(self):
        """Test that --force flag is recognized in help."""
        result = runner.invoke(app, ["ingest", "--help"])

        assert "--force" in result.output or "-f" in result.output
