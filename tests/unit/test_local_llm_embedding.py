# tests/test_local_llm_embedding.py
"""
Tests for local LLM embedding backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLocalEmbedderConfig:
    """Tests for LocalEmbedderConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedderConfig

        cfg = LocalEmbedderConfig()

        # Config exists (even if empty)
        assert cfg is not None

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedderConfig

        cfg = LocalEmbedderConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.new_attr = "value"


class TestLocalEmbedder:
    """Tests for LocalEmbedder."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_runtime = MagicMock()
        embedder = LocalEmbedder(mock_runtime)

        assert embedder._rt is mock_runtime
        assert embedder._cfg is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from fitz_ai.backends.local_llm.embedding import (
            LocalEmbedder,
            LocalEmbedderConfig,
        )

        mock_runtime = MagicMock()
        cfg = LocalEmbedderConfig()
        embedder = LocalEmbedder(mock_runtime, cfg)

        assert embedder._cfg is cfg

    def test_embed_returns_vector(self):
        """Test embed returns float vector."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_adapter = MagicMock()
        mock_adapter.embed.return_value = [0.1, 0.2, 0.3, 0.4]

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        embedder = LocalEmbedder(mock_runtime)
        result = embedder.embed("test text")

        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_adapter.embed.assert_called_once_with("test text")

    def test_embed_raises_on_non_list(self):
        """Test embed raises TypeError for non-list response."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_adapter = MagicMock()
        mock_adapter.embed.return_value = "not a list"

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        embedder = LocalEmbedder(mock_runtime)

        with pytest.raises(TypeError, match="list"):
            embedder.embed("test")

    def test_embed_raises_on_dict(self):
        """Test embed raises TypeError for dict response."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_adapter = MagicMock()
        mock_adapter.embed.return_value = {"embedding": [0.1, 0.2]}

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        embedder = LocalEmbedder(mock_runtime)

        with pytest.raises(TypeError, match="list"):
            embedder.embed("test")

    def test_embed_texts_single(self):
        """Test embed_texts with single text."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_adapter = MagicMock()
        mock_adapter.embed.return_value = [0.1, 0.2, 0.3]

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        embedder = LocalEmbedder(mock_runtime)
        result = embedder.embed_texts(["text one"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    def test_embed_texts_multiple(self):
        """Test embed_texts with multiple texts."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_adapter = MagicMock()
        # Return different vectors for each call
        mock_adapter.embed.side_effect = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        embedder = LocalEmbedder(mock_runtime)
        result = embedder.embed_texts(["one", "two", "three"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]

    def test_embed_texts_empty(self):
        """Test embed_texts with empty list."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_runtime = MagicMock()
        embedder = LocalEmbedder(mock_runtime)

        result = embedder.embed_texts([])

        assert result == []

    def test_embed_calls_runtime_llama(self):
        """Test that embed calls runtime.llama() to get adapter."""
        from fitz_ai.backends.local_llm.embedding import LocalEmbedder

        mock_adapter = MagicMock()
        mock_adapter.embed.return_value = [0.1]

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        embedder = LocalEmbedder(mock_runtime)
        embedder.embed("test")

        mock_runtime.llama.assert_called_once()
