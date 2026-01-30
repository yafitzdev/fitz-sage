# tests/unit/llm/test_client.py
"""
Unit tests for LLM client public API.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fitz_ai.llm.client import get_chat, get_embedder, get_reranker, get_vision
from fitz_ai.llm.providers import (
    ChatProvider,
    CohereChat,
    CohereEmbedding,
    CohereRerank,
    EmbeddingProvider,
    OllamaChat,
    OllamaEmbedding,
    RerankProvider,
)


class TestGetChat:
    """Test get_chat function."""

    def test_returns_chat_provider(self) -> None:
        """Returns a ChatProvider instance."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                chat = get_chat("cohere")
                assert isinstance(chat, ChatProvider)
                assert isinstance(chat, CohereChat)

    def test_tier_selection(self) -> None:
        """Tier selects appropriate model."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                smart = get_chat("cohere", tier="smart")
                fast = get_chat("cohere", tier="fast")
                assert smart._model == "command-a-03-2025"
                assert fast._model == "command-r7b-12-2024"

    def test_model_override(self) -> None:
        """Model in spec overrides tier default."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                chat = get_chat("cohere/custom-model", tier="fast")
                assert chat._model == "custom-model"

    def test_ollama_no_auth(self) -> None:
        """Ollama works without auth."""
        with patch("httpx.Client"):
            chat = get_chat("ollama")
            assert isinstance(chat, OllamaChat)

    def test_config_passed_through(self) -> None:
        """Config is passed to provider."""
        with patch("httpx.Client"):
            chat = get_chat("ollama", config={"base_url": "http://custom:11434"})
            assert chat._base_url == "http://custom:11434"


class TestGetEmbedder:
    """Test get_embedder function."""

    def test_returns_embedding_provider(self) -> None:
        """Returns an EmbeddingProvider instance."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                embedder = get_embedder("cohere")
                assert isinstance(embedder, EmbeddingProvider)
                assert isinstance(embedder, CohereEmbedding)

    def test_model_override(self) -> None:
        """Model in spec is used."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                embedder = get_embedder("cohere/embed-english-v3.0")
                assert embedder._model == "embed-english-v3.0"

    def test_dimensions_config(self) -> None:
        """Dimensions from config are used."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                embedder = get_embedder("cohere", config={"dimensions": 512})
                assert embedder._dimensions == 512

    def test_ollama_embedder(self) -> None:
        """Ollama embedding works."""
        with patch("httpx.Client"):
            embedder = get_embedder("ollama")
            assert isinstance(embedder, OllamaEmbedding)


class TestGetReranker:
    """Test get_reranker function."""

    def test_none_returns_none(self) -> None:
        """None spec returns None."""
        reranker = get_reranker(None)
        assert reranker is None

    def test_returns_rerank_provider(self) -> None:
        """Returns a RerankProvider instance."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                reranker = get_reranker("cohere")
                assert reranker is not None
                assert isinstance(reranker, RerankProvider)
                assert isinstance(reranker, CohereRerank)

    def test_model_override(self) -> None:
        """Model in spec is used."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                reranker = get_reranker("cohere/rerank-english-v3.0")
                assert reranker is not None
                assert reranker._model == "rerank-english-v3.0"


class TestGetVision:
    """Test get_vision function."""

    def test_none_returns_none(self) -> None:
        """None spec returns None."""
        vision = get_vision(None)
        assert vision is None

    def test_unsupported_provider_raises(self) -> None:
        """Unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unknown vision provider"):
            get_vision("cohere")


# Skip OpenAI/Anthropic tests if SDKs not installed
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestGetChatOpenAI:
    """Test get_chat with OpenAI."""

    def test_openai_chat(self) -> None:
        """OpenAI chat provider works."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                chat = get_chat("openai")
                assert chat._model == "gpt-4o"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestGetVisionOpenAI:
    """Test get_vision with OpenAI."""

    def test_openai_vision(self) -> None:
        """OpenAI vision provider works."""
        from fitz_ai.llm.providers import OpenAIVision, VisionProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                vision = get_vision("openai/gpt-4o")
                assert vision is not None
                assert isinstance(vision, VisionProvider)
                assert isinstance(vision, OpenAIVision)


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestGetChatAnthropic:
    """Test get_chat with Anthropic."""

    def test_anthropic_chat(self) -> None:
        """Anthropic chat provider works."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                chat = get_chat("anthropic")
                assert chat._model == "claude-sonnet-4-20250514"


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestGetVisionAnthropic:
    """Test get_vision with Anthropic."""

    def test_anthropic_vision(self) -> None:
        """Anthropic vision provider works."""
        from fitz_ai.llm.providers import AnthropicVision, VisionProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                vision = get_vision("anthropic/claude-sonnet-4")
                assert vision is not None
                assert isinstance(vision, VisionProvider)
                assert isinstance(vision, AnthropicVision)
