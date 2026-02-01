# tests/unit/llm/test_client.py
"""
Unit tests for LLM client public API.

Tests the factory functions work correctly.
Provider-specific tests require the SDK to be installed.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

from fitz_ai.llm.client import get_chat, get_embedder, get_reranker, get_vision
from fitz_ai.llm.providers import ChatProvider, EmbeddingProvider, RerankProvider

# Check for optional SDKs
HAS_COHERE = importlib.util.find_spec("cohere") is not None
HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_OLLAMA = importlib.util.find_spec("ollama") is not None


class TestGetReranker:
    """Test get_reranker function."""

    def test_none_returns_none(self) -> None:
        """None spec returns None."""
        reranker = get_reranker(None)
        assert reranker is None


class TestGetVision:
    """Test get_vision function."""

    def test_none_returns_none(self) -> None:
        """None spec returns None."""
        vision = get_vision(None)
        assert vision is None


class TestUnknownProvider:
    """Test unknown provider handling."""

    def test_unknown_chat_provider_raises(self) -> None:
        """Unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown chat provider"):
            get_chat("unknown_provider")

    def test_unknown_embedding_provider_raises(self) -> None:
        """Unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder("unknown_provider")

    def test_unknown_rerank_provider_raises(self) -> None:
        """Unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown rerank provider"):
            get_reranker("unknown_provider")

    def test_unknown_vision_provider_raises(self) -> None:
        """Unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown vision provider"):
            get_vision("unknown_provider")


# Provider-specific tests - only run if SDK is installed


@pytest.mark.skipif(not HAS_COHERE, reason="cohere not installed")
class TestCohereProviders:
    """Test Cohere providers when SDK is available."""

    def test_cohere_chat(self) -> None:
        """Cohere chat provider works."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                chat = get_chat("cohere")
                assert isinstance(chat, ChatProvider)

    def test_cohere_embedder(self) -> None:
        """Cohere embedding provider works."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                embedder = get_embedder("cohere")
                assert isinstance(embedder, EmbeddingProvider)

    def test_cohere_reranker(self) -> None:
        """Cohere rerank provider works."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                reranker = get_reranker("cohere")
                assert reranker is not None
                assert isinstance(reranker, RerankProvider)


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestOpenAIProviders:
    """Test OpenAI providers when SDK is available."""

    def test_openai_chat(self) -> None:
        """OpenAI chat provider works."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                chat = get_chat("openai")
                assert isinstance(chat, ChatProvider)

    def test_openai_vision(self) -> None:
        """OpenAI vision provider works."""
        from fitz_ai.llm.providers import VisionProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                vision = get_vision("openai/gpt-4o")
                assert vision is not None
                assert isinstance(vision, VisionProvider)


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestAnthropicProviders:
    """Test Anthropic providers when SDK is available."""

    def test_anthropic_chat(self) -> None:
        """Anthropic chat provider works."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                chat = get_chat("anthropic")
                assert isinstance(chat, ChatProvider)

    def test_anthropic_vision(self) -> None:
        """Anthropic vision provider works."""
        from fitz_ai.llm.providers import VisionProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                vision = get_vision("anthropic/claude-sonnet-4")
                assert vision is not None
                assert isinstance(vision, VisionProvider)


@pytest.mark.skipif(not HAS_OLLAMA, reason="ollama not installed")
class TestOllamaProviders:
    """Test Ollama providers when SDK is available."""

    def test_ollama_chat(self) -> None:
        """Ollama chat provider works."""
        with patch("httpx.Client"):
            chat = get_chat("ollama")
            assert isinstance(chat, ChatProvider)

    def test_ollama_embedder(self) -> None:
        """Ollama embedding provider works."""
        with patch("httpx.Client"):
            embedder = get_embedder("ollama")
            assert isinstance(embedder, EmbeddingProvider)
