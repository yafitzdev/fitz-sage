# tests/test_local_llm_runtime.py
"""
Tests for local LLM runtime (Ollama adapter).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.engines.fitz_rag.exceptions import LLMError


class TestLocalLLMRuntimeConfig:
    """Tests for LocalLLMRuntimeConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from fitz_ai.backends.local_llm.runtime import LocalLLMRuntimeConfig

        cfg = LocalLLMRuntimeConfig()

        assert cfg.model == "llama3.2:1b"
        assert cfg.verbose is False

    def test_custom_values(self):
        """Test custom configuration values."""
        from fitz_ai.backends.local_llm.runtime import LocalLLMRuntimeConfig

        cfg = LocalLLMRuntimeConfig(model="mistral:7b", verbose=True)

        assert cfg.model == "mistral:7b"
        assert cfg.verbose is True

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        from fitz_ai.backends.local_llm.runtime import LocalLLMRuntimeConfig

        cfg = LocalLLMRuntimeConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.model = "other"


class TestOllamaAdapter:
    """Tests for _OllamaAdapter."""

    def test_adapter_init_without_ollama_raises(self):
        """Test that adapter raises helpful error when ollama not installed."""
        with patch.dict("sys.modules", {"ollama": None}):
            from fitz_ai.backends.local_llm.runtime import _OllamaAdapter

            with pytest.raises(LLMError) as exc_info:
                _OllamaAdapter("model", verbose=False)

            assert "ollama" in str(exc_info.value).lower()

    def test_adapter_init_with_ollama(self):
        """Test adapter initializes with ollama installed."""
        mock_ollama = MagicMock()

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from fitz_ai.backends.local_llm.runtime import _OllamaAdapter

            adapter = _OllamaAdapter("test-model", verbose=True)

            assert adapter._model == "test-model"
            assert adapter._verbose is True

    def test_adapter_chat_success(self):
        """Test adapter chat returns content."""
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Hello!"}}

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from fitz_ai.backends.local_llm.runtime import _OllamaAdapter

            adapter = _OllamaAdapter("model", verbose=False)
            result = adapter.chat([{"role": "user", "content": "Hi"}])

            assert result == "Hello!"
            mock_ollama.chat.assert_called_once()

    def test_adapter_chat_failure_raises_llm_error(self):
        """Test adapter chat raises LLMError on failure."""
        mock_ollama = MagicMock()
        mock_ollama.chat.side_effect = Exception("Connection failed")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from fitz_ai.backends.local_llm.runtime import _OllamaAdapter

            adapter = _OllamaAdapter("model", verbose=False)

            with pytest.raises(LLMError):
                adapter.chat([{"role": "user", "content": "Hi"}])

    def test_adapter_embed_success(self):
        """Test adapter embed returns embedding."""
        mock_ollama = MagicMock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from fitz_ai.backends.local_llm.runtime import _OllamaAdapter

            adapter = _OllamaAdapter("model", verbose=False)
            result = adapter.embed("test text")

            assert result == [0.1, 0.2, 0.3]
            mock_ollama.embeddings.assert_called_once()

    def test_adapter_embed_failure_raises_llm_error(self):
        """Test adapter embed raises LLMError on failure."""
        mock_ollama = MagicMock()
        mock_ollama.embeddings.side_effect = Exception("Embedding failed")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from fitz_ai.backends.local_llm.runtime import _OllamaAdapter

            adapter = _OllamaAdapter("model", verbose=False)

            with pytest.raises(LLMError):
                adapter.embed("test")


class TestLocalLLMRuntime:
    """Tests for LocalLLMRuntime."""

    def test_runtime_init_default_config(self):
        """Test runtime initializes with default config."""
        from fitz_ai.backends.local_llm.runtime import LocalLLMRuntime

        runtime = LocalLLMRuntime()

        assert runtime._cfg.model == "llama3.2:1b"
        assert runtime._adapter is None

    def test_runtime_init_custom_config(self):
        """Test runtime initializes with custom config."""
        from fitz_ai.backends.local_llm.runtime import (
            LocalLLMRuntime,
            LocalLLMRuntimeConfig,
        )

        cfg = LocalLLMRuntimeConfig(model="custom:model")
        runtime = LocalLLMRuntime(cfg)

        assert runtime._cfg.model == "custom:model"

    def test_runtime_llama_creates_adapter(self):
        """Test llama() creates and caches adapter."""
        mock_ollama = MagicMock()

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from fitz_ai.backends.local_llm.runtime import LocalLLMRuntime

            runtime = LocalLLMRuntime()

            adapter1 = runtime.llama()
            adapter2 = runtime.llama()

            # Should return same instance (cached)
            assert adapter1 is adapter2

    def test_runtime_llama_raises_llm_error_on_failure(self):
        """Test llama() raises LLMError on adapter failure."""
        with patch.dict("sys.modules", {"ollama": None}):
            from fitz_ai.backends.local_llm.runtime import LocalLLMRuntime

            runtime = LocalLLMRuntime()

            with pytest.raises(LLMError):
                runtime.llama()


class TestFallbackHelp:
    """Tests for fallback help message."""

    def test_fallback_help_contains_instructions(self):
        """Test that fallback help message contains setup instructions."""
        from fitz_ai.backends.local_llm.runtime import _LOCAL_FALLBACK_HELP

        assert "ollama" in _LOCAL_FALLBACK_HELP.lower()
        assert "pull" in _LOCAL_FALLBACK_HELP.lower()
        assert "llama3.2:1b" in _LOCAL_FALLBACK_HELP
