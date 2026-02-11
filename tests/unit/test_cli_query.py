# tests/test_cli_query.py
"""
Tests for the query command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app
from fitz_ai.core import Answer
from fitz_ai.services.fitz_service import CollectionInfo, QueryError

runner = CliRunner()


class TestQueryCommand:
    """Tests for fitz query command."""

    def test_query_shows_help(self):
        """Test that query --help works."""
        result = runner.invoke(app, ["query", "--help"])

        assert result.exit_code == 0
        assert "Query your knowledge base" in result.output or "query" in result.output.lower()

    def test_query_requires_valid_collection(self):
        """Test that query requires a valid collection for collection-based queries."""
        # Mock FitzService to raise CollectionNotFoundError
        mock_service = MagicMock()
        mock_service.list_collections.return_value = []

        with patch("fitz_ai.cli.commands.query.FitzService", return_value=mock_service):
            result = runner.invoke(app, ["query", "test question", "--engine", "fitz_rag"])

        # When no collections exist, command should show message and exit
        assert "no collection" in result.output.lower() or "ingest" in result.output.lower()


class TestQueryHelpers:
    """Tests for query helper functions."""

    def test_cli_context_loads_config(self, tmp_path):
        """Test CLIContext.load() returns raw and typed config."""
        import yaml

        from fitz_ai.cli.context import CLIContext

        # Create engine-specific config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "fitz_rag.yaml"
        config = {
            "chat": "cohere",
            "embedding": "cohere",
            "vector_db": "pgvector",
            "retrieval_plugin": "dense",
            "collection": "test",
        }
        config_path.write_text(yaml.dump(config))

        with patch(
            "fitz_ai.cli.context.FitzPaths.engine_config",
            return_value=config_path,
        ):
            ctx = CLIContext.load(engine="fitz_rag")

        assert ctx.raw_config["chat"] == "cohere"
        assert ctx.typed_config.collection == "test"

    def test_get_collections_returns_list(self):
        """Test get_collections returns collection list."""
        mock_ctx = MagicMock()
        mock_ctx.get_collections.return_value = ["coll_a", "coll_b"]

        from fitz_ai.cli.utils import get_collections

        collections = get_collections(mock_ctx)

        assert sorted(collections) == ["coll_a", "coll_b"]

    def test_get_collections_handles_empty(self):
        """Test get_collections returns empty list when no collections."""
        mock_ctx = MagicMock()
        mock_ctx.get_collections.return_value = []

        from fitz_ai.cli.utils import get_collections

        collections = get_collections(mock_ctx)

        assert collections == []


class TestQueryExecution:
    """Tests for query execution with mocked service."""

    def test_query_direct_mode(self):
        """Test query with direct question argument."""
        mock_answer = Answer(
            text="This is the answer",
            provenance=[],
            mode="trustworthy",
        )

        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="test", chunk_count=10),
        ]
        mock_service.query.return_value = mock_answer

        with patch("fitz_ai.cli.commands.query.FitzService", return_value=mock_service):
            result = runner.invoke(app, ["query", "What is RAG?", "--engine", "fitz_rag"])

        # Should call service.query with the question
        mock_service.query.assert_called_once()
        assert result.exit_code == 0

    def test_query_handles_error(self):
        """Test query handles QueryError gracefully."""
        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="test", chunk_count=10),
        ]
        mock_service.query.side_effect = QueryError("Test error")

        with patch("fitz_ai.cli.commands.query.FitzService", return_value=mock_service):
            result = runner.invoke(app, ["query", "What is RAG?", "--engine", "fitz_rag"])

        # Should show error and exit with code 1
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


class TestQueryOptions:
    """Tests for query command options."""

    def test_query_with_collection_option(self):
        """Test query with --collection option."""
        mock_answer = Answer(
            text="Answer",
            provenance=[],
            mode="trustworthy",
        )

        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="custom", chunk_count=10),
        ]
        mock_service.query.return_value = mock_answer

        with patch("fitz_ai.cli.commands.query.FitzService", return_value=mock_service):
            runner.invoke(app, ["query", "question", "-c", "custom", "--engine", "fitz_rag"])

        # Service should be called with the custom collection
        mock_service.query.assert_called_once()
        call_kwargs = mock_service.query.call_args
        assert call_kwargs.kwargs["collection"] == "custom"

    def test_query_collection_not_found(self):
        """Test query shows error when collection not found."""
        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="other", chunk_count=10),
        ]

        with patch("fitz_ai.cli.commands.query.FitzService", return_value=mock_service):
            result = runner.invoke(
                app, ["query", "question", "-c", "nonexistent", "--engine", "fitz_rag"]
            )

        # Should show collection not found message
        assert "not found" in result.output.lower() or "available" in result.output.lower()
