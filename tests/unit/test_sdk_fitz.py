# tests/test_sdk_fitz.py
"""
Tests for the Fitz SDK.
"""

from __future__ import annotations

import pytest
import yaml


class TestFitzInit:
    """Tests for fitz initialization."""

    def test_default_collection(self):
        """Test default collection name."""
        from fitz_ai.sdk import fitz

        f = fitz()

        assert f.collection == "default"

    def test_custom_collection(self):
        """Test custom collection name."""
        from fitz_ai.sdk import fitz

        f = fitz(collection="my_collection")

        assert f.collection == "my_collection"

    def test_custom_config_path(self, tmp_path):
        """Test custom config path."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "custom.yaml"
        f = fitz(config_path=config_path)

        assert f.config_path == config_path

    def test_auto_init_default_true(self):
        """Test auto_init defaults to True."""
        from fitz_ai.sdk import fitz

        f = fitz()
        # auto_init is an internal attribute
        assert f._auto_init is True


class TestFitzConfigCreation:
    """Tests for config file creation."""

    def test_ensure_config_creates_file(self, tmp_path):
        """Test that _ensure_config creates config when auto_init=True."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / ".fitz" / "config.yaml"
        f = fitz(config_path=config_path, auto_init=True)

        # Call ensure config
        f._ensure_config()

        # Config should have been created
        assert config_path.exists()

    def test_config_has_required_sections(self, tmp_path):
        """Test that created config has all required keys (flat format)."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path, collection="test")

        # Manually call the config creation
        f._create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        # Flat format keys (v0.9.0+ KRAG)
        assert "chat" in config
        assert "embedding" in config
        assert "vector_db" in config
        assert "rerank" in config
        assert "collection" in config

    def test_config_uses_collection_name(self, tmp_path):
        """Test that config uses the fitz instance's collection name."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path, collection="my_collection")

        f._create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        # Flat format: collection is top-level
        assert config["collection"] == "my_collection"

    def test_raises_without_auto_init(self, tmp_path):
        """Test that ConfigurationError is raised when auto_init=False and no config."""
        from fitz_ai.core import ConfigurationError
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "nonexistent.yaml"
        f = fitz(config_path=config_path, auto_init=False)

        with pytest.raises(ConfigurationError):
            f._ensure_config()


class TestFitzAsk:
    """Tests for fitz.ask() method."""

    def test_raises_on_empty_question(self, tmp_path):
        """Test that QueryError is raised for empty question."""
        from fitz_ai.sdk import fitz
        from fitz_ai.services.fitz_service import QueryError

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path)
        f._create_default_config(config_path)

        with pytest.raises(QueryError, match="cannot be empty"):
            f.ask("")

    def test_raises_on_whitespace_question(self, tmp_path):
        """Test that QueryError is raised for whitespace-only question."""
        from fitz_ai.sdk import fitz
        from fitz_ai.services.fitz_service import QueryError

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path)
        f._create_default_config(config_path)

        with pytest.raises(QueryError, match="cannot be empty"):
            f.ask("   ")


class TestFitzQuery:
    """Tests for fitz.query() alias."""

    def test_query_is_alias_for_ask(self, tmp_path):
        """Test that query() is an alias for ask()."""
        from fitz_ai.sdk import fitz
        from fitz_ai.services.fitz_service import QueryError

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path)
        f._create_default_config(config_path)

        # Both should raise the same error
        with pytest.raises(QueryError, match="cannot be empty"):
            f.query("")


class TestFitzExports:
    """Tests for SDK exports."""

    def test_fitz_exported_from_sdk(self):
        """Test fitz is exported from fitz_ai.sdk."""
        from fitz_ai.sdk import fitz

        assert fitz is not None

    def test_fitz_exported_from_top_level(self):
        """Test fitz is exported from fitz_ai."""
        from fitz_ai import fitz

        assert fitz is not None

    def test_module_level_query_exported(self):
        """Test module-level query() is exported."""
        import fitz_ai

        assert hasattr(fitz_ai, "query")
        assert callable(fitz_ai.query)
