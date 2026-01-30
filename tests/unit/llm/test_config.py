# tests/unit/llm/test_config.py
"""
Unit tests for LLM config parser.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.llm.auth import ApiKeyAuth, M2MAuth
from fitz_ai.llm.config import (
    create_chat_provider,
    create_embedding_provider,
    create_rerank_provider,
    create_vision_provider,
    parse_provider_string,
    resolve_auth,
)


class TestParseProviderString:
    """Test provider string parsing."""

    def test_provider_only(self) -> None:
        """Parse provider without model."""
        provider, model = parse_provider_string("cohere")
        assert provider == "cohere"
        assert model is None

    def test_provider_with_model(self) -> None:
        """Parse provider with model."""
        provider, model = parse_provider_string("cohere/command-a-03-2025")
        assert provider == "cohere"
        assert model == "command-a-03-2025"

    def test_openai_with_model(self) -> None:
        """Parse OpenAI provider with model."""
        provider, model = parse_provider_string("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_strips_whitespace(self) -> None:
        """Whitespace is stripped."""
        provider, model = parse_provider_string("  cohere / command-a-03-2025  ")
        assert provider == "cohere"
        assert model == "command-a-03-2025"

    def test_model_with_slashes(self) -> None:
        """Model names can contain slashes (only first split)."""
        provider, model = parse_provider_string("huggingface/meta-llama/Llama-2-70b")
        assert provider == "huggingface"
        assert model == "meta-llama/Llama-2-70b"


class TestResolveAuth:
    """Test auth resolution."""

    def test_cohere_api_key(self) -> None:
        """Cohere uses COHERE_API_KEY with bearer format."""
        auth = resolve_auth("cohere")
        assert isinstance(auth, ApiKeyAuth)
        assert auth.env_var == "COHERE_API_KEY"
        assert auth.header_format == "bearer"

    def test_openai_api_key(self) -> None:
        """OpenAI uses OPENAI_API_KEY with bearer format."""
        auth = resolve_auth("openai")
        assert isinstance(auth, ApiKeyAuth)
        assert auth.env_var == "OPENAI_API_KEY"
        assert auth.header_format == "bearer"

    def test_anthropic_api_key(self) -> None:
        """Anthropic uses ANTHROPIC_API_KEY with x-api-key format."""
        auth = resolve_auth("anthropic")
        assert isinstance(auth, ApiKeyAuth)
        assert auth.env_var == "ANTHROPIC_API_KEY"
        assert auth.header_format == "x-api-key"

    def test_ollama_no_auth(self) -> None:
        """Ollama doesn't need auth."""
        auth = resolve_auth("ollama")
        assert auth is None

    def test_m2m_auth(self, temp_certificate) -> None:
        """M2M auth is created from config."""
        cert_path, _ = temp_certificate
        config = {
            "auth": {
                "type": "m2m",
                "token_url": "https://auth.example.com/token",
                "client_id": "my-client",
                "client_secret": "my-secret",
                "scope": "read write",
            },
            "cert_path": cert_path,
        }
        auth = resolve_auth("openai", config)
        assert isinstance(auth, M2MAuth)
        assert auth.token_url == "https://auth.example.com/token"
        assert auth.client_id == "my-client"
        assert auth.client_secret == "my-secret"
        assert auth.cert_path == cert_path
        assert auth.scope == "read write"

    def test_m2m_auth_minimal(self) -> None:
        """M2M auth works with minimal config."""
        config = {
            "auth": {
                "type": "m2m",
                "token_url": "https://auth.example.com/token",
                "client_id": "my-client",
                "client_secret": "my-secret",
            }
        }
        auth = resolve_auth("openai", config)
        assert isinstance(auth, M2MAuth)
        assert auth.cert_path is None
        assert auth.scope is None


class TestCreateChatProvider:
    """Test chat provider creation."""

    def test_cohere_provider(self) -> None:
        """Create Cohere chat provider."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_chat_provider("cohere")
                assert provider._model == "command-a-03-2025"  # smart tier default

    def test_cohere_with_model(self) -> None:
        """Create Cohere chat provider with specific model."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_chat_provider("cohere/command-r-plus")
                assert provider._model == "command-r-plus"

    def test_cohere_fast_tier(self) -> None:
        """Create Cohere chat provider with fast tier."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_chat_provider("cohere", tier="fast")
                assert provider._model == "command-r7b-12-2024"

    def test_ollama_no_auth(self) -> None:
        """Ollama provider doesn't require auth."""
        with patch("httpx.Client"):
            provider = create_chat_provider("ollama")
            assert provider._model == "llama3.1:70b"  # smart tier default

    def test_ollama_with_model(self) -> None:
        """Ollama provider with specific model."""
        with patch("httpx.Client"):
            provider = create_chat_provider("ollama/mistral:7b")
            assert provider._model == "mistral:7b"

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chat provider: unknown"):
            create_chat_provider("unknown")

    def test_base_url_passed_through(self) -> None:
        """Base URL is passed to provider."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                # Cohere doesn't use base_url, but the config is passed
                provider = create_chat_provider(
                    "cohere", config={"base_url": "https://proxy.example.com"}
                )
                # Provider is created (doesn't fail)
                assert provider is not None


class TestCreateEmbeddingProvider:
    """Test embedding provider creation."""

    def test_cohere_provider(self) -> None:
        """Create Cohere embedding provider."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_embedding_provider("cohere")
                assert provider._model == "embed-multilingual-v3.0"

    def test_cohere_with_model(self) -> None:
        """Create Cohere embedding provider with specific model."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_embedding_provider("cohere/embed-english-v3.0")
                assert provider._model == "embed-english-v3.0"

    def test_cohere_with_dimensions(self) -> None:
        """Create Cohere embedding provider with custom dimensions."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_embedding_provider("cohere", config={"dimensions": 512})
                assert provider._dimensions == 512

    def test_cohere_with_input_type(self) -> None:
        """Create Cohere embedding provider with input type."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_embedding_provider(
                    "cohere", config={"input_type": "search_query"}
                )
                assert provider._input_type == "search_query"

    def test_ollama_provider(self) -> None:
        """Create Ollama embedding provider."""
        with patch("httpx.Client"):
            provider = create_embedding_provider("ollama")
            assert provider._model == "nomic-embed-text"

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider: unknown"):
            create_embedding_provider("unknown")


class TestCreateRerankProvider:
    """Test rerank provider creation."""

    def test_none_returns_none(self) -> None:
        """None spec returns None."""
        provider = create_rerank_provider(None)
        assert provider is None

    def test_cohere_provider(self) -> None:
        """Create Cohere rerank provider."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_rerank_provider("cohere")
                assert provider is not None
                assert provider._model == "rerank-multilingual-v3.0"

    def test_cohere_with_model(self) -> None:
        """Create Cohere rerank provider with specific model."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                provider = create_rerank_provider("cohere/rerank-english-v3.0")
                assert provider._model == "rerank-english-v3.0"

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown rerank provider: openai"):
            create_rerank_provider("openai")


class TestCreateVisionProvider:
    """Test vision provider creation."""

    def test_none_returns_none(self) -> None:
        """None spec returns None."""
        provider = create_vision_provider(None)
        assert provider is None

    def test_unknown_provider_raises(self) -> None:
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vision provider: cohere"):
            create_vision_provider("cohere")


# Skip OpenAI/Anthropic vision tests if SDKs not installed
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
class TestCreateVisionProviderOpenAI:
    """Test OpenAI vision provider creation."""

    def test_openai_provider(self) -> None:
        """Create OpenAI vision provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                provider = create_vision_provider("openai/gpt-4o")
                assert provider is not None
                assert provider._model == "gpt-4o"


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestCreateVisionProviderAnthropic:
    """Test Anthropic vision provider creation."""

    def test_anthropic_provider(self) -> None:
        """Create Anthropic vision provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                provider = create_vision_provider("anthropic/claude-sonnet-4")
                assert provider is not None
                assert provider._model == "claude-sonnet-4"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestCreateChatProviderOpenAI:
    """Test OpenAI chat provider creation."""

    def test_openai_provider(self) -> None:
        """Create OpenAI chat provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                provider = create_chat_provider("openai")
                assert provider._model == "gpt-4o"

    def test_openai_with_base_url(self) -> None:
        """Create OpenAI chat provider with custom base URL."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                create_chat_provider("openai", config={"base_url": "https://api.proxy.com/v1"})
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["base_url"] == "https://api.proxy.com/v1"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestCreateEmbeddingProviderOpenAI:
    """Test OpenAI embedding provider creation."""

    def test_openai_provider(self) -> None:
        """Create OpenAI embedding provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                provider = create_embedding_provider("openai")
                assert provider._model == "text-embedding-3-small"

    def test_openai_with_dimensions(self) -> None:
        """Create OpenAI embedding provider with dimensions."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                provider = create_embedding_provider("openai", config={"dimensions": 256})
                assert provider._dimensions == 256


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestCreateChatProviderAnthropic:
    """Test Anthropic chat provider creation."""

    def test_anthropic_provider(self) -> None:
        """Create Anthropic chat provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                provider = create_chat_provider("anthropic")
                assert provider._model == "claude-sonnet-4-20250514"

    def test_anthropic_with_model(self) -> None:
        """Create Anthropic chat provider with specific model."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                provider = create_chat_provider("anthropic/claude-opus-4-20250514")
                assert provider._model == "claude-opus-4-20250514"
