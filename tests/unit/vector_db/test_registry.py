# tests/unit/vector_db/test_registry.py
"""Tests for vector_db registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.tier1

from fitz_sage.vector_db.registry import available_vector_db_plugins, get_vector_db_plugin


class TestGetVectorDbPlugin:
    def test_rejects_non_pgvector(self):
        with pytest.raises(ValueError, match="Unsupported vector_db plugin"):
            get_vector_db_plugin("qdrant")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="Unsupported vector_db plugin"):
            get_vector_db_plugin("")

    @patch("fitz_sage.vector_db.registry.create_vector_db_plugin")
    @patch("fitz_sage.vector_db.registry.maybe_wrap")
    def test_creates_and_wraps_pgvector(self, mock_wrap, mock_create):
        mock_plugin = MagicMock()
        mock_create.return_value = mock_plugin
        mock_wrap.return_value = mock_plugin

        get_vector_db_plugin("pgvector", mode="local")

        mock_create.assert_called_once_with("pgvector", mode="local")
        mock_wrap.assert_called_once_with(
            mock_plugin,
            layer="vector_db",
            plugin_name="pgvector",
            methods_to_track={"search", "upsert", "delete", "count", "list_collections"},
        )

    @patch("fitz_sage.vector_db.registry.create_vector_db_plugin")
    @patch("fitz_sage.vector_db.registry.maybe_wrap")
    def test_default_plugin_is_pgvector(self, mock_wrap, mock_create):
        mock_create.return_value = MagicMock()
        mock_wrap.return_value = mock_create.return_value

        get_vector_db_plugin()

        mock_create.assert_called_once_with("pgvector")


class TestAvailablePlugins:
    def test_returns_pgvector_only(self):
        result = available_vector_db_plugins()
        assert result == ["pgvector"]
