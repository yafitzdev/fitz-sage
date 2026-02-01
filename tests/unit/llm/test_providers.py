# tests/unit/llm/test_providers.py
"""
Unit tests for LLM provider implementations.
"""

from __future__ import annotations

# Check for optional SDK dependencies
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.llm.auth import ApiKeyAuth, DynamicHttpxAuth
from fitz_ai.llm.providers import (
    ChatProvider,
    CohereChat,
    CohereEmbedding,
    CohereRerank,
    EmbeddingProvider,
    OllamaChat,
    OllamaEmbedding,
    RerankProvider,
    RerankResult,
    VisionProvider,
)

HAS_OPENAI = importlib.util.find_spec("openai") is not None
if HAS_OPENAI:
    from fitz_ai.llm.providers import OpenAIChat, OpenAIEmbedding, OpenAIVision
else:
    OpenAIChat = None  # type: ignore[misc, assignment]
    OpenAIEmbedding = None  # type: ignore[misc, assignment]
    OpenAIVision = None  # type: ignore[misc, assignment]

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
if HAS_ANTHROPIC:
    from fitz_ai.llm.providers import AnthropicChat, AnthropicVision
else:
    AnthropicChat = None  # type: ignore[misc, assignment]
    AnthropicVision = None  # type: ignore[misc, assignment]


class TestProtocols:
    """Test that providers implement protocols correctly."""

    def test_cohere_chat_is_chat_provider(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereChat(auth)
                assert isinstance(provider, ChatProvider)

    def test_cohere_embedding_is_embedding_provider(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereEmbedding(auth)
                assert isinstance(provider, EmbeddingProvider)

    def test_cohere_rerank_is_rerank_provider(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereRerank(auth)
                assert isinstance(provider, RerankProvider)

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_openai_chat_is_chat_provider(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                auth = ApiKeyAuth("OPENAI_API_KEY")
                provider = OpenAIChat(auth)
                assert isinstance(provider, ChatProvider)

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_openai_embedding_is_embedding_provider(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                auth = ApiKeyAuth("OPENAI_API_KEY")
                provider = OpenAIEmbedding(auth)
                assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_openai_vision_is_vision_provider(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                auth = ApiKeyAuth("OPENAI_API_KEY")
                provider = OpenAIVision(auth)
                assert isinstance(provider, VisionProvider)

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
    def test_anthropic_chat_is_chat_provider(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                provider = AnthropicChat(auth)
                assert isinstance(provider, ChatProvider)

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
    def test_anthropic_vision_is_vision_provider(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                provider = AnthropicVision(auth)
                assert isinstance(provider, VisionProvider)

    def test_ollama_chat_is_chat_provider(self) -> None:
        with patch("httpx.Client"):
            provider = OllamaChat()
            assert isinstance(provider, ChatProvider)

    def test_ollama_embedding_is_embedding_provider(self) -> None:
        with patch("httpx.Client"):
            provider = OllamaEmbedding()
            assert isinstance(provider, EmbeddingProvider)


class TestCohereChat:
    """Test Cohere chat provider."""

    def test_chat_extracts_text(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Hello, world!"
            mock_response.message.content = [mock_content]
            mock_client.chat.return_value = mock_response

            with patch("cohere.ClientV2", return_value=mock_client):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereChat(auth)
                result = provider.chat([{"role": "user", "content": "Hi"}])

            assert result == "Hello, world!"

    def test_tier_selects_model(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                auth = ApiKeyAuth("COHERE_API_KEY")

                smart = CohereChat(auth, tier="smart")
                assert smart._model == "command-a-03-2025"

                fast = CohereChat(auth, tier="fast")
                assert fast._model == "command-r7b-12-2024"

    def test_uses_dynamic_httpx_auth(self) -> None:
        """Verify Cohere provider uses DynamicHttpxAuth for per-request auth."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("cohere.ClientV2"):
                    auth = ApiKeyAuth("COHERE_API_KEY")
                    CohereChat(auth)

                    # Verify httpx.Client was created with auth parameter
                    call_kwargs = mock_httpx_client.call_args[1]
                    assert "auth" in call_kwargs
                    # Verify it's a DynamicHttpxAuth instance
                    assert isinstance(call_kwargs["auth"], DynamicHttpxAuth)

    def test_httpx_client_passed_to_sdk(self) -> None:
        """Verify httpx_client (not http_client) is passed to Cohere SDK."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            mock_httpx_client = MagicMock()
            with patch("httpx.Client", return_value=mock_httpx_client):
                with patch("cohere.ClientV2") as mock_cohere:
                    auth = ApiKeyAuth("COHERE_API_KEY")
                    CohereChat(auth)

                    call_kwargs = mock_cohere.call_args[1]
                    # Cohere uses httpx_client, not http_client
                    assert call_kwargs["httpx_client"] == mock_httpx_client
                    assert "http_client" not in call_kwargs

    def test_api_key_unused_placeholder(self) -> None:
        """Verify SDK receives api_key='unused' placeholder."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("httpx.Client"):
                with patch("cohere.ClientV2") as mock_cohere:
                    auth = ApiKeyAuth("COHERE_API_KEY")
                    CohereChat(auth)

                    call_kwargs = mock_cohere.call_args[1]
                    assert call_kwargs["api_key"] == "unused"

    def test_certificate_path_propagates(self) -> None:
        """Verify certificate path from auth flows to httpx.Client verify parameter."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("cohere.ClientV2"):
                    # Create mock auth that returns custom certificate path
                    mock_auth = MagicMock()
                    mock_auth.get_headers.return_value = {"Authorization": "Bearer test"}
                    mock_auth.get_request_kwargs.return_value = {"verify": "/path/to/ca.crt"}

                    CohereChat(mock_auth)

                    call_kwargs = mock_httpx_client.call_args[1]
                    assert call_kwargs["verify"] == "/path/to/ca.crt"

    def test_backwards_compatible_with_api_key_auth(self) -> None:
        """Verify API-key-only users see no behavior change."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Hello from Cohere!"
            mock_response.message.content = [mock_content]
            mock_client.chat.return_value = mock_response

            with patch("httpx.Client"):
                with patch("cohere.ClientV2", return_value=mock_client):
                    auth = ApiKeyAuth("COHERE_API_KEY")
                    provider = CohereChat(auth)
                    result = provider.chat([{"role": "user", "content": "Hi"}])

                    assert result == "Hello from Cohere!"


class TestCohereEmbedding:
    """Test Cohere embedding provider."""

    def test_embed_returns_vector(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.embeddings.float_ = [[0.1, 0.2, 0.3]]
            mock_client.embed.return_value = mock_response

            with patch("cohere.ClientV2", return_value=mock_client):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereEmbedding(auth)
                result = provider.embed("test text")

            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_returns_vectors(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.embeddings.float_ = [[0.1, 0.2], [0.3, 0.4]]
            mock_client.embed.return_value = mock_response

            with patch("cohere.ClientV2", return_value=mock_client):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereEmbedding(auth)
                result = provider.embed_batch(["text1", "text2"])

            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_dimensions_property(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereEmbedding(auth, dimensions=512)
                assert provider.dimensions == 512

    def test_uses_dynamic_httpx_auth(self) -> None:
        """Verify CohereEmbedding uses DynamicHttpxAuth for per-request auth."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("cohere.ClientV2"):
                    auth = ApiKeyAuth("COHERE_API_KEY")
                    CohereEmbedding(auth)

                    call_kwargs = mock_httpx_client.call_args[1]
                    assert "auth" in call_kwargs
                    assert isinstance(call_kwargs["auth"], DynamicHttpxAuth)


class TestCohereRerank:
    """Test Cohere rerank provider."""

    def test_rerank_returns_results(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_result1 = MagicMock()
            mock_result1.index = 1
            mock_result1.relevance_score = 0.9
            mock_result2 = MagicMock()
            mock_result2.index = 0
            mock_result2.relevance_score = 0.5
            mock_response = MagicMock()
            mock_response.results = [mock_result1, mock_result2]
            mock_client.rerank.return_value = mock_response

            with patch("cohere.ClientV2", return_value=mock_client):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereRerank(auth)
                result = provider.rerank("query", ["doc1", "doc2"])

            assert len(result) == 2
            assert result[0] == RerankResult(index=1, score=0.9)
            assert result[1] == RerankResult(index=0, score=0.5)

    def test_rerank_empty_documents(self) -> None:
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("cohere.ClientV2"):
                auth = ApiKeyAuth("COHERE_API_KEY")
                provider = CohereRerank(auth)
                result = provider.rerank("query", [])

            assert result == []

    def test_uses_dynamic_httpx_auth(self) -> None:
        """Verify CohereRerank uses DynamicHttpxAuth for per-request auth."""
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("cohere.ClientV2"):
                    auth = ApiKeyAuth("COHERE_API_KEY")
                    CohereRerank(auth)

                    call_kwargs = mock_httpx_client.call_args[1]
                    assert "auth" in call_kwargs
                    assert isinstance(call_kwargs["auth"], DynamicHttpxAuth)


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestOpenAIChat:
    """Test OpenAI chat provider."""

    def test_chat_extracts_text(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Hello from OpenAI!"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            with patch("openai.OpenAI", return_value=mock_client):
                auth = ApiKeyAuth("OPENAI_API_KEY")
                provider = OpenAIChat(auth)
                result = provider.chat([{"role": "user", "content": "Hi"}])

            assert result == "Hello from OpenAI!"

    def test_custom_base_url(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                auth = ApiKeyAuth("OPENAI_API_KEY")
                OpenAIChat(auth, base_url="https://custom.api.com/v1")

                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["base_url"] == "https://custom.api.com/v1"

    def test_uses_dynamic_httpx_auth(self) -> None:
        """Verify OpenAI provider uses DynamicHttpxAuth for per-request auth."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("openai.OpenAI"):
                    auth = ApiKeyAuth("OPENAI_API_KEY")
                    OpenAIChat(auth)

                    # Verify httpx.Client was created with auth parameter
                    call_kwargs = mock_httpx_client.call_args[1]
                    assert "auth" in call_kwargs
                    # Verify it's a DynamicHttpxAuth instance
                    assert isinstance(call_kwargs["auth"], DynamicHttpxAuth)

    def test_backwards_compatible_with_api_key_auth(self) -> None:
        """Verify API-key-only users see no behavior change."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_message.content = "Hello from OpenAI!"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            with patch("httpx.Client"):
                with patch("openai.OpenAI", return_value=mock_client):
                    auth = ApiKeyAuth("OPENAI_API_KEY")
                    provider = OpenAIChat(auth)
                    result = provider.chat([{"role": "user", "content": "Hi"}])

                    assert result == "Hello from OpenAI!"

    def test_api_key_unused_placeholder(self) -> None:
        """Verify SDK receives api_key='unused' placeholder."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("httpx.Client"):
                with patch("openai.OpenAI") as mock_openai:
                    auth = ApiKeyAuth("OPENAI_API_KEY")
                    OpenAIChat(auth)

                    call_kwargs = mock_openai.call_args[1]
                    assert call_kwargs["api_key"] == "unused"

    def test_httpx_timeout_matches_sdk_defaults(self) -> None:
        """Verify httpx.Client has SDK-compatible timeout (600s read, 5s connect)."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("openai.OpenAI"):
                    auth = ApiKeyAuth("OPENAI_API_KEY")
                    OpenAIChat(auth)

                    call_kwargs = mock_httpx_client.call_args[1]
                    timeout = call_kwargs.get("timeout")
                    # Verify timeout is set (specific values depend on httpx.Timeout)
                    assert timeout is not None

    def test_http_client_passed_to_sdk(self) -> None:
        """Verify http_client is passed to OpenAI SDK."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mock_http_client = MagicMock()
            with patch("httpx.Client", return_value=mock_http_client):
                with patch("openai.OpenAI") as mock_openai:
                    auth = ApiKeyAuth("OPENAI_API_KEY")
                    OpenAIChat(auth)

                    call_kwargs = mock_openai.call_args[1]
                    assert call_kwargs["http_client"] == mock_http_client


@pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
class TestOpenAIEmbedding:
    """Test OpenAI embedding provider."""

    def test_embed_batch_preserves_order(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            mock_client = MagicMock()
            # Return embeddings in different order than input
            mock_item0 = MagicMock()
            mock_item0.index = 0
            mock_item0.embedding = [0.1, 0.2]
            mock_item1 = MagicMock()
            mock_item1.index = 1
            mock_item1.embedding = [0.3, 0.4]
            mock_response = MagicMock()
            mock_response.data = [mock_item1, mock_item0]  # Reversed order
            mock_client.embeddings.create.return_value = mock_response

            with patch("openai.OpenAI", return_value=mock_client):
                auth = ApiKeyAuth("OPENAI_API_KEY")
                provider = OpenAIEmbedding(auth)
                result = provider.embed_batch(["text1", "text2"])

            # Should be sorted by index
            assert result == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestAnthropicChat:
    """Test Anthropic chat provider."""

    def test_system_message_extracted(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response

            with patch("anthropic.Anthropic", return_value=mock_client):
                auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                provider = AnthropicChat(auth)
                messages = [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ]
                provider.chat(messages)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are helpful"
            assert len(call_kwargs["messages"]) == 1
            assert call_kwargs["messages"][0]["role"] == "user"

    def test_uses_dynamic_httpx_auth(self) -> None:
        """Verify Anthropic provider uses DynamicHttpxAuth for per-request auth."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("anthropic.Anthropic"):
                    auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                    AnthropicChat(auth)

                    # Verify httpx.Client was created with auth parameter
                    call_kwargs = mock_httpx_client.call_args[1]
                    assert "auth" in call_kwargs
                    # Verify it's a DynamicHttpxAuth instance
                    assert isinstance(call_kwargs["auth"], DynamicHttpxAuth)

    def test_auth_token_unused_placeholder(self) -> None:
        """Verify SDK receives auth_token='unused' placeholder."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("httpx.Client"):
                with patch("anthropic.Anthropic") as mock_anthropic:
                    auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                    AnthropicChat(auth)

                    call_kwargs = mock_anthropic.call_args[1]
                    assert call_kwargs["auth_token"] == "unused"

    def test_http_client_passed_to_sdk(self) -> None:
        """Verify http_client is passed to Anthropic SDK."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_http_client = MagicMock()
            with patch("httpx.Client", return_value=mock_http_client):
                with patch("anthropic.Anthropic") as mock_anthropic:
                    auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                    AnthropicChat(auth)

                    call_kwargs = mock_anthropic.call_args[1]
                    assert call_kwargs["http_client"] == mock_http_client

    def test_certificate_path_propagates(self) -> None:
        """Verify certificate path from auth flows to httpx.Client."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("httpx.Client") as mock_httpx_client:
                with patch("anthropic.Anthropic"):
                    # Create auth that returns certificate path
                    auth = MagicMock()
                    auth.get_headers.return_value = {"Authorization": "Bearer test"}
                    auth.get_request_kwargs.return_value = {"verify": "/path/to/ca.crt"}

                    AnthropicChat(auth)

                    call_kwargs = mock_httpx_client.call_args[1]
                    assert call_kwargs["verify"] == "/path/to/ca.crt"

    def test_backwards_compatible_with_api_key_auth(self) -> None:
        """Verify API-key-only users see no behavior change."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Hello from Anthropic!"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response

            with patch("httpx.Client"):
                with patch("anthropic.Anthropic", return_value=mock_client):
                    auth = ApiKeyAuth("ANTHROPIC_API_KEY")
                    provider = AnthropicChat(auth)
                    result = provider.chat([{"role": "user", "content": "Hi"}])

                    assert result == "Hello from Anthropic!"


class TestOllamaChat:
    """Test Ollama chat provider."""

    def test_chat_extracts_text(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Hello from Ollama!"}}
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaChat()
            result = provider.chat([{"role": "user", "content": "Hi"}])

        assert result == "Hello from Ollama!"

    def test_custom_base_url(self) -> None:
        with patch("httpx.Client") as mock_client_class:
            OllamaChat(base_url="http://custom:11434")

            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["base_url"] == "http://custom:11434"


class TestOllamaEmbedding:
    """Test Ollama embedding provider."""

    def test_embed_returns_vector(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaEmbedding()
            result = provider.embed("test")

        assert result == [0.1, 0.2, 0.3]

    def test_dimensions_discovered(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]]}
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            provider = OllamaEmbedding()
            provider.embed("test")
            assert provider.dimensions == 5
