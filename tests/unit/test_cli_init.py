# tests/unit/test_cli_init.py
"""
Tests for the init command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestInitCommand:
    """Tests for fitz init command."""

    def test_init_shows_help(self):
        """Test that init --help works."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "init" in result.output.lower()
        assert "interactive" in result.output.lower() or "wizard" in result.output.lower()

    def test_init_show_config_flag(self):
        """Test that --show flag previews config without saving."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.pgvector.available = True
        mock_system.api_keys = {
            "cohere": MagicMock(available=True),
            "openai": MagicMock(available=False),
            "anthropic": MagicMock(available=False),
        }

        # Mock CLIContext
        mock_ctx = MagicMock()
        mock_ctx.chat_plugin = "cohere"
        mock_ctx.embedding_plugin = "cohere"
        mock_ctx.vector_db_plugin = "pgvector"
        mock_ctx.rerank_enabled = False
        mock_ctx.retrieval_plugin = "dense"
        mock_ctx.retrieval_collection = "default"
        mock_ctx.parser = "simple"

        with (
            patch("fitz_ai.cli.commands.init_wizard.detect_system", return_value=mock_system),
            patch("fitz_ai.cli.commands.init_wizard.CLIContext.load", return_value=mock_ctx),
            patch(
                "fitz_ai.cli.commands.init_wizard.available_llm_plugins",
                side_effect=lambda t: ["cohere"] if t in ["chat", "embedding"] else [],
            ),
            patch(
                "fitz_ai.cli.commands.init_wizard.available_vector_db_plugins",
                return_value=["pgvector"],
            ),
            patch(
                "fitz_ai.cli.commands.init_wizard.get_default_engine",
                return_value="fitz_krag",
            ),
        ):
            result = runner.invoke(app, ["init", "-y", "--show"])

        # Should not save (exit before save)
        assert result.exit_code == 0


class TestInitHelpers:
    """Tests for init helper functions."""

    def test_filter_available_plugins_ollama(self):
        """Test filter_available_plugins filters Ollama plugins."""
        from fitz_ai.cli.commands.init_detector import filter_available_plugins

        mock_system = MagicMock()
        mock_system.ollama.available = True

        plugins = ["cohere", "local_ollama", "openai"]
        result = filter_available_plugins(plugins, "chat", mock_system)

        assert "local_ollama" in result

    def test_filter_available_plugins_ollama_unavailable(self):
        """Test filter_available_plugins excludes Ollama when unavailable."""
        from fitz_ai.cli.commands.init_detector import filter_available_plugins

        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.api_keys = {}

        plugins = ["local_ollama"]
        result = filter_available_plugins(plugins, "chat", mock_system)

        assert "local_ollama" not in result

    def test_filter_available_plugins_pgvector(self):
        """Test filter_available_plugins includes pgvector when available."""
        from fitz_ai.cli.commands.init_detector import filter_available_plugins

        mock_system = MagicMock()
        mock_system.pgvector.available = True

        plugins = ["pgvector"]
        result = filter_available_plugins(plugins, "vector_db", mock_system)

        assert "pgvector" in result

    def test_filter_available_plugins_api_keys(self):
        """Test filter_available_plugins checks API keys."""
        from fitz_ai.cli.commands.init_detector import filter_available_plugins

        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.pgvector.available = False
        mock_system.api_keys = {
            "cohere": MagicMock(available=True),
            "openai": MagicMock(available=False),
        }

        plugins = ["cohere", "openai"]
        result = filter_available_plugins(plugins, "chat", mock_system)

        assert "cohere" in result
        assert "openai" not in result

    def test_get_default_model_chat(self):
        """Test get_default_model returns correct chat models for smart/fast tiers."""
        from fitz_ai.cli.commands.init_models import get_default_model

        # Smart tier (default)
        assert get_default_model("chat", "cohere", "smart") == "command-a-03-2025"
        assert get_default_model("chat", "openai", "smart") == "gpt-4o"
        assert "llama" in get_default_model("chat", "ollama", "smart")

        # Fast tier
        assert get_default_model("chat", "cohere", "fast") == "command-r7b-12-2024"
        assert get_default_model("chat", "openai", "fast") == "gpt-4o-mini"
        assert "llama" in get_default_model("chat", "ollama", "fast")

    def test_get_default_model_embedding(self):
        """Test get_default_model returns correct embedding model."""
        from fitz_ai.cli.commands.init_models import get_default_model

        assert get_default_model("embedding", "cohere") == "embed-english-v3.0"
        assert get_default_model("embedding", "openai") == "text-embedding-3-small"

    def test_get_default_model_unknown(self):
        """Test get_default_model returns empty for unknown plugin."""
        from fitz_ai.cli.commands.init_models import get_default_model

        assert get_default_model("chat", "unknown_plugin") == ""


class TestGenerateConfig:
    """Tests for config generation functions."""

    def test_generate_global_config(self):
        """Test generate_global_config produces valid YAML."""
        import yaml

        from fitz_ai.cli.commands.init_config import generate_global_config

        config_str = generate_global_config("fitz_krag")
        config = yaml.safe_load(config_str)

        assert config["default_engine"] == "fitz_krag"

    def test_generate_fitz_krag_config_basic(self):
        """Test generate_fitz_krag_config produces valid YAML with flat format."""
        import yaml

        from fitz_ai.cli.commands.init_config import generate_fitz_krag_config

        config_str = generate_fitz_krag_config(
            chat="cohere",
            chat_model_smart="command-a-03-2025",
            chat_model_fast="command-r7b-12-2024",
            embedding="cohere",
            embedding_model="embed-english-v3.0",
            rerank=None,
            rerank_model="",
            vector_db="pgvector",
        )

        # Should be valid YAML with flat format
        config = yaml.safe_load(config_str)

        # New flat format: plugins are strings, not dicts
        assert config["chat"] == "cohere"
        assert config["embedding"] == "cohere"
        assert config["vector_db"] == "pgvector"

    def test_generate_fitz_krag_config_with_rerank(self):
        """Test generate_fitz_krag_config includes rerank when provided."""
        import yaml

        from fitz_ai.cli.commands.init_config import generate_fitz_krag_config

        config_str = generate_fitz_krag_config(
            chat="cohere",
            chat_model_smart="command-a-03-2025",
            chat_model_fast="command-r7b-12-2024",
            embedding="cohere",
            embedding_model="embed-english-v3.0",
            rerank="cohere",
            rerank_model="rerank-v3.5",
            vector_db="pgvector",
        )

        config = yaml.safe_load(config_str)

        # Rerank is a string in flat format
        assert config["rerank"] == "cohere"

    def test_generate_fitz_krag_config_without_rerank(self):
        """Test generate_fitz_krag_config sets rerank to null when not provided."""
        import yaml

        from fitz_ai.cli.commands.init_config import generate_fitz_krag_config

        config_str = generate_fitz_krag_config(
            chat="cohere",
            chat_model_smart="",
            chat_model_fast="",
            embedding="cohere",
            embedding_model="",
            rerank=None,
            rerank_model="",
            vector_db="pgvector",
        )

        config = yaml.safe_load(config_str)

        # Rerank is null in flat format
        assert config["rerank"] is None


class TestInitValidation:
    """Tests for init validation logic."""

    def _create_mock_ctx(self):
        """Create a mock CLIContext."""
        mock_ctx = MagicMock()
        mock_ctx.chat_plugin = "cohere"
        mock_ctx.embedding_plugin = "cohere"
        mock_ctx.vector_db_plugin = "pgvector"
        mock_ctx.rerank_enabled = False
        mock_ctx.retrieval_plugin = "dense"
        mock_ctx.retrieval_collection = "default"
        mock_ctx.parser = "simple"
        return mock_ctx

    def test_init_fails_without_chat_plugins(self):
        """Test init fails when no chat plugins available (fitz_krag)."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.pgvector.available = True
        mock_system.api_keys = {}

        with (
            patch("fitz_ai.cli.commands.init_wizard.detect_system", return_value=mock_system),
            patch(
                "fitz_ai.cli.commands.init_wizard.CLIContext.load",
                return_value=self._create_mock_ctx(),
            ),
            patch(
                "fitz_ai.cli.commands.init_wizard.get_default_engine",
                return_value="fitz_krag",
            ),
            patch("fitz_ai.cli.commands.init_wizard.available_llm_plugins", return_value=[]),
            patch(
                "fitz_ai.cli.commands.init_wizard.available_vector_db_plugins",
                return_value=["pgvector"],
            ),
        ):
            result = runner.invoke(app, ["init", "-y"])

        assert result.exit_code != 0
        assert "chat" in result.output.lower() or "api key" in result.output.lower()

    def test_init_fails_without_vector_db(self):
        """Test init fails when no vector DB available (fitz_krag)."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.pgvector.available = False
        mock_system.api_keys = {"cohere": MagicMock(available=True)}

        with (
            patch("fitz_ai.cli.commands.init_wizard.detect_system", return_value=mock_system),
            patch(
                "fitz_ai.cli.commands.init_wizard.CLIContext.load",
                return_value=self._create_mock_ctx(),
            ),
            patch(
                "fitz_ai.cli.commands.init_wizard.get_default_engine",
                return_value="fitz_krag",
            ),
            patch(
                "fitz_ai.cli.commands.init_wizard.available_llm_plugins",
                return_value=["cohere"],
            ),
            patch(
                "fitz_ai.cli.commands.init_wizard.available_vector_db_plugins",
                return_value=[],
            ),
        ):
            result = runner.invoke(app, ["init", "-y"])

        assert result.exit_code != 0
        assert "vector" in result.output.lower()
