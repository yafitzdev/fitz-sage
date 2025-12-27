# tests/test_local_llm_chat.py
"""
Tests for local LLM chat backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLocalChatConfig:
    """Tests for LocalChatConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from fitz_ai.backends.local_llm.chat import LocalChatConfig

        cfg = LocalChatConfig()

        assert cfg.max_tokens == 256
        assert cfg.temperature == 0.2

    def test_custom_values(self):
        """Test custom configuration values."""
        from fitz_ai.backends.local_llm.chat import LocalChatConfig

        cfg = LocalChatConfig(max_tokens=512, temperature=0.7)

        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.7

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        from fitz_ai.backends.local_llm.chat import LocalChatConfig

        cfg = LocalChatConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.max_tokens = 1024


class TestLocalChatLLM:
    """Tests for LocalChatLLM."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from fitz_ai.backends.local_llm.chat import LocalChatLLM

        mock_runtime = MagicMock()
        chat = LocalChatLLM(mock_runtime)

        assert chat._rt is mock_runtime
        assert chat._cfg.max_tokens == 256

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from fitz_ai.backends.local_llm.chat import LocalChatConfig, LocalChatLLM

        mock_runtime = MagicMock()
        cfg = LocalChatConfig(max_tokens=1024)
        chat = LocalChatLLM(mock_runtime, cfg)

        assert chat._cfg.max_tokens == 1024

    def test_chat_calls_runtime(self):
        """Test chat method calls runtime adapter."""
        from fitz_ai.backends.local_llm.chat import LocalChatLLM

        mock_adapter = MagicMock()
        mock_adapter.chat.return_value = {"message": {"content": "Response"}}

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        chat = LocalChatLLM(mock_runtime)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = chat.chat(messages)

        mock_runtime.llama.assert_called_once()
        mock_adapter.chat.assert_called_once()

        # Check messages were formatted correctly
        call_args = mock_adapter.chat.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"

    def test_chat_handles_missing_role(self):
        """Test chat handles messages without explicit role."""
        from fitz_ai.backends.local_llm.chat import LocalChatLLM

        mock_adapter = MagicMock()
        mock_adapter.chat.return_value = {"message": {"content": "Response"}}

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        chat = LocalChatLLM(mock_runtime)
        messages = [{"content": "No role specified"}]

        chat.chat(messages)

        call_args = mock_adapter.chat.call_args[0][0]
        assert call_args[0]["role"] == "user"  # Default role

    def test_chat_handles_missing_content(self):
        """Test chat handles messages without content."""
        from fitz_ai.backends.local_llm.chat import LocalChatLLM

        mock_adapter = MagicMock()
        mock_adapter.chat.return_value = {"message": {"content": "Response"}}

        mock_runtime = MagicMock()
        mock_runtime.llama.return_value = mock_adapter

        chat = LocalChatLLM(mock_runtime)
        messages = [{"role": "user"}]

        chat.chat(messages)

        call_args = mock_adapter.chat.call_args[0][0]
        assert call_args[0]["content"] == ""  # Default empty content


class TestExtractText:
    """Tests for _extract_text helper function."""

    def test_extract_text_from_dict(self):
        """Test extracting text from standard response dict."""
        from fitz_ai.backends.local_llm.chat import _extract_text

        resp = {"message": {"content": "Hello, world!"}}
        result = _extract_text(resp)

        assert result == "Hello, world!"

    def test_extract_text_missing_message(self):
        """Test extraction when message key is missing."""
        from fitz_ai.backends.local_llm.chat import _extract_text

        resp = {"other": "data"}
        result = _extract_text(resp)

        assert result == ""

    def test_extract_text_missing_content(self):
        """Test extraction when content key is missing."""
        from fitz_ai.backends.local_llm.chat import _extract_text

        resp = {"message": {"role": "assistant"}}
        result = _extract_text(resp)

        assert result == ""

    def test_extract_text_none_message(self):
        """Test extraction when message is None."""
        from fitz_ai.backends.local_llm.chat import _extract_text

        resp = {"message": None}
        result = _extract_text(resp)

        assert result == ""

    def test_extract_text_non_dict_response(self):
        """Test extraction falls back to str() for non-dict."""
        from fitz_ai.backends.local_llm.chat import _extract_text

        resp = "plain string response"
        result = _extract_text(resp)

        assert result == "plain string response"

    def test_extract_text_exception_fallback(self):
        """Test extraction uses str() on exception."""
        from fitz_ai.backends.local_llm.chat import _extract_text

        # Object that raises on .get()
        class BadObj:
            def get(self, *args):
                raise RuntimeError("boom")

            def __str__(self):
                return "fallback"

        result = _extract_text(BadObj())
        assert result == "fallback"
