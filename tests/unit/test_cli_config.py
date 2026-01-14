# tests/test_cli_config.py
"""
Tests for the config command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


def _make_mock_context(
    *,
    chat_plugin="cohere",
    embedding_plugin="cohere",
    vector_db_plugin="local_faiss",
    retrieval_plugin="dense",
    retrieval_collection="default",
    has_user_config=True,
    config_path=None,
    config_source="user config",
    raw_config=None,
):
    """Create a mock CLIContext for testing."""
    ctx = MagicMock()
    ctx.chat_plugin = chat_plugin
    ctx.chat_model_smart = "command-r-plus"
    ctx.chat_model_fast = "command-r"
    ctx.embedding_plugin = embedding_plugin
    ctx.embedding_model = "embed-english-v3.0"
    ctx.vector_db_plugin = vector_db_plugin
    ctx.vector_db_kwargs = {}
    ctx.retrieval_plugin = retrieval_plugin
    ctx.retrieval_collection = retrieval_collection
    ctx.retrieval_top_k = 5
    ctx.rerank_enabled = False
    ctx.rerank_plugin = ""
    ctx.rerank_model = ""
    ctx.rgs_citations = True
    ctx.rgs_strict_grounding = True
    ctx.has_user_config = has_user_config
    ctx.config_path = config_path
    ctx.config_source = config_source
    ctx.raw_config = raw_config or {"chat": {"plugin_name": chat_plugin}}
    return ctx


class TestConfigCommand:
    """Tests for fitz config command."""

    def test_config_shows_help(self):
        """Test that config --help works."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "config" in result.output.lower()
        assert "json" in result.output.lower()
        assert "path" in result.output.lower()

    def test_config_works_with_defaults(self):
        """Test that config works even without user config (uses package defaults)."""
        mock_ctx = _make_mock_context(
            has_user_config=False,
            config_path=None,
            config_source="package defaults",
        )

        with patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx):
            result = runner.invoke(app, ["config"])

        # Should succeed with defaults
        assert result.exit_code == 0
        assert "package defaults" in result.output.lower() or "cohere" in result.output.lower()


class TestConfigPathOption:
    """Tests for --path option."""

    def test_config_path_shows_path(self, tmp_path):
        """Test --path shows config file path."""
        config_path = tmp_path / "fitz_rag.yaml"
        config_path.write_text("chat:\n  plugin_name: cohere\n")

        mock_ctx = _make_mock_context(
            has_user_config=True,
            config_path=config_path,
        )

        with patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx):
            result = runner.invoke(app, ["config", "--path"])

        assert result.exit_code == 0
        assert str(config_path) in result.output

    def test_config_path_missing_file(self, tmp_path):
        """Test --path with missing config shows error."""
        nonexistent = tmp_path / "nonexistent" / "fitz_rag.yaml"

        mock_ctx = _make_mock_context(
            has_user_config=False,
            config_path=None,
        )

        with (
            patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx),
            patch(
                "fitz_ai.cli.commands.config.FitzPaths.engine_config",
                return_value=nonexistent,
            ),
            patch(
                "fitz_ai.cli.commands.config.FitzPaths.config",
                return_value=nonexistent.parent / "fitz.yaml",
            ),
        ):
            result = runner.invoke(app, ["config", "--path"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()


class TestConfigJsonOption:
    """Tests for --json option."""

    def test_config_json_output(self, tmp_path):
        """Test --json outputs valid JSON."""
        config_path = tmp_path / "fitz_rag.yaml"
        config_path.write_text("chat:\n  plugin_name: cohere\n")

        mock_ctx = _make_mock_context(
            has_user_config=True,
            config_path=config_path,
            raw_config={
                "chat": {"plugin_name": "cohere"},
                "embedding": {"plugin_name": "cohere"},
            },
        )

        with patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx):
            result = runner.invoke(app, ["config", "--json"])

        assert result.exit_code == 0
        # Output should contain JSON-like content
        assert "cohere" in result.output


class TestConfigRawOption:
    """Tests for --raw option."""

    def test_config_raw_output(self, tmp_path):
        """Test --raw outputs YAML content."""
        config_content = """# Fitz config
chat:
  plugin_name: cohere
"""
        config_path = tmp_path / "fitz_rag.yaml"
        config_path.write_text(config_content)

        mock_ctx = _make_mock_context(
            has_user_config=True,
            config_path=config_path,
        )

        with patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx):
            result = runner.invoke(app, ["config", "--raw"])

        assert result.exit_code == 0
        assert "plugin_name: cohere" in result.output


class TestConfigEditOption:
    """Tests for --edit option."""

    def test_config_edit_missing_file(self, tmp_path):
        """Test --edit with missing config shows error."""
        nonexistent = tmp_path / "nonexistent" / "fitz_rag.yaml"

        mock_ctx = _make_mock_context(
            has_user_config=False,
            config_path=None,
        )

        with (
            patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx),
            patch(
                "fitz_ai.cli.commands.config.FitzPaths.engine_config",
                return_value=nonexistent,
            ),
            patch(
                "fitz_ai.cli.commands.config.FitzPaths.config",
                return_value=nonexistent.parent / "fitz.yaml",
            ),
        ):
            result = runner.invoke(app, ["config", "--edit"])

        assert result.exit_code != 0
        assert "no config" in result.output.lower() or "init" in result.output.lower()

    def test_config_edit_no_editor(self, tmp_path):
        """Test --edit with no editor available."""
        config_path = tmp_path / "fitz_rag.yaml"
        config_path.write_text("chat:\n  plugin_name: cohere\n")

        mock_ctx = _make_mock_context(
            has_user_config=True,
            config_path=config_path,
        )

        with (
            patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx),
            patch.dict("os.environ", {}, clear=True),
            patch("subprocess.run", side_effect=Exception("not found")),
        ):
            result = runner.invoke(app, ["config", "--edit"])

        # Should show error about editor or show path as fallback
        assert "editor" in result.output.lower() or str(config_path) in result.output


class TestConfigSummaryView:
    """Tests for default summary view."""

    def test_config_summary_shows_components(self, tmp_path):
        """Test default view shows all components."""
        config_path = tmp_path / "fitz_rag.yaml"
        config_path.write_text("chat:\n  plugin_name: cohere\n")

        mock_ctx = _make_mock_context(
            chat_plugin="cohere",
            embedding_plugin="cohere",
            vector_db_plugin="local_faiss",
            retrieval_plugin="dense",
            has_user_config=True,
            config_path=config_path,
        )

        with (
            patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx),
            patch("fitz_ai.cli.commands.config.RICH", False),
        ):
            result = runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "cohere" in result.output.lower()
        assert "faiss" in result.output.lower()

    def test_config_summary_shows_rerank_when_enabled(self, tmp_path):
        """Test summary shows rerank when enabled."""
        config_path = tmp_path / "fitz_rag.yaml"
        config_path.write_text("chat:\n  plugin_name: cohere\n")

        mock_ctx = _make_mock_context(
            has_user_config=True,
            config_path=config_path,
        )
        mock_ctx.rerank_enabled = True
        mock_ctx.rerank_plugin = "cohere"
        mock_ctx.rerank_model = "rerank-v3.5"

        with (
            patch("fitz_ai.cli.commands.config.CLIContext.load", return_value=mock_ctx),
            patch("fitz_ai.cli.commands.config.RICH", False),
        ):
            result = runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "rerank" in result.output.lower()


class TestShowConfigSummary:
    """Tests for _show_config_summary function."""

    def test_show_config_summary_plain(self, capsys):
        """Test _show_config_summary plain text output."""
        mock_ctx = MagicMock()
        mock_ctx.chat_plugin = "cohere"
        mock_ctx.chat_model_smart = "command-r-plus"
        mock_ctx.embedding_plugin = "cohere"
        mock_ctx.embedding_model = "embed-english-v3.0"
        mock_ctx.vector_db_plugin = "local_faiss"
        mock_ctx.vector_db_kwargs = {}
        mock_ctx.retrieval_plugin = "dense"
        mock_ctx.retrieval_collection = "test"
        mock_ctx.retrieval_top_k = 5
        mock_ctx.rerank_enabled = False
        mock_ctx.rerank_plugin = ""
        mock_ctx.rerank_model = ""
        mock_ctx.rgs_citations = True
        mock_ctx.rgs_strict_grounding = True

        with patch("fitz_ai.cli.commands.config.RICH", False):
            from fitz_ai.cli.commands.config import _show_config_summary

            _show_config_summary(mock_ctx)

        captured = capsys.readouterr()
        assert "cohere" in captured.out.lower()
        assert "dense" in captured.out.lower()
        assert "test" in captured.out


class TestShowConfigJson:
    """Tests for _show_config_json function."""

    def test_show_config_json_plain(self, capsys):
        """Test _show_config_json plain text output."""
        import json

        with patch("fitz_ai.cli.commands.config.RICH", False):
            from fitz_ai.cli.commands.config import _show_config_json

            config = {"chat": {"plugin_name": "test"}}

            _show_config_json(config)

        captured = capsys.readouterr()
        # Should be valid JSON
        parsed = json.loads(captured.out.strip())
        assert parsed["chat"]["plugin_name"] == "test"


class TestShowConfigYaml:
    """Tests for _show_config_yaml function."""

    def test_show_config_yaml_reads_file(self, tmp_path, capsys):
        """Test _show_config_yaml reads and displays file."""
        config_content = "chat:\n  plugin_name: cohere\n"
        config_path = tmp_path / "fitz.yaml"
        config_path.write_text(config_content)

        with patch("fitz_ai.cli.commands.config.RICH", False):
            from fitz_ai.cli.commands.config import _show_config_yaml

            _show_config_yaml(config_path)

        captured = capsys.readouterr()
        assert "cohere" in captured.out
