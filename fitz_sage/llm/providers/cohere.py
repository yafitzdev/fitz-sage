# fitz_sage/llm/providers/cohere.py
"""
Cohere provider wrappers using the official SDK.

Uses DynamicHttpxAuth for per-request token refresh, solving the frozen
token bug where M2M tokens captured at __init__ never refresh.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from fitz_sage.llm.auth import AuthProvider
from fitz_sage.llm.auth.httpx_auth import DynamicHttpxAuth
from fitz_sage.llm.providers.base import ModelTier, RerankResult

logger = logging.getLogger(__name__)

# Default models by tier
CHAT_MODELS: dict[ModelTier, str] = {
    "smart": "command-a-03-2025",
    "balanced": "command-r7b-12-2024",
    "fast": "command-r7b-12-2024",
}

EMBEDDING_MODEL = "embed-multilingual-v3.0"
RERANK_MODEL = "rerank-multilingual-v3.0"


class CohereChat:
    """
    Cohere chat provider using the official SDK.

    Args:
        auth: Authentication provider (uses api_key property).
        model: Model name override.
        tier: Model tier (smart, balanced, fast).
        **kwargs: Additional default kwargs for chat calls.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        tier: ModelTier = "smart",
        models: dict[ModelTier, str] | None = None,
        **kwargs: Any,
    ) -> None:
        import cohere
        import httpx

        request_kwargs = auth.get_request_kwargs()

        httpx_client = httpx.Client(
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(300.0, connect=5.0),
        )

        self._client = cohere.ClientV2(
            api_key="unused",  # SDK requires non-empty, httpx_client auth overrides
            httpx_client=httpx_client,
        )
        # Use provided models dict, falling back to defaults
        tier_models = models or CHAT_MODELS
        self._model = model or tier_models.get(tier) or CHAT_MODELS[tier]
        self._defaults = kwargs

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion."""
        params = {**self._defaults, **kwargs}

        response = self._client.chat(
            model=params.pop("model", self._model),
            messages=messages,
            **params,
        )

        # Extract text from response
        if response.message and response.message.content:
            for block in response.message.content:
                if hasattr(block, "text"):
                    return block.text
        return ""

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """Generate a streaming chat completion."""
        params = {**self._defaults, **kwargs}

        stream = self._client.chat_stream(
            model=params.pop("model", self._model),
            messages=messages,
            **params,
        )

        for event in stream:
            if hasattr(event, "delta") and hasattr(event.delta, "message"):
                if event.delta.message and event.delta.message.content:
                    if hasattr(event.delta.message.content, "text"):
                        yield event.delta.message.content.text


class CohereEmbedding:
    """
    Cohere embedding provider using the official SDK.

    Args:
        auth: Authentication provider.
        model: Model name override.
        input_type: Input type for embeddings (search_document, search_query, etc.)
        dimensions: Output dimensions (None for model default).
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        input_type: str = "search_document",
        dimensions: int | None = None,
    ) -> None:
        import cohere
        import httpx

        request_kwargs = auth.get_request_kwargs()

        httpx_client = httpx.Client(
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(300.0, connect=5.0),
        )

        self._client = cohere.ClientV2(
            api_key="unused",  # SDK requires non-empty, httpx_client auth overrides
            httpx_client=httpx_client,
        )
        self._model = model or EMBEDDING_MODEL
        self._input_type = input_type
        self._dimensions = dimensions

    _TASK_TYPE_MAP = {
        "query": "search_query",
        "document": "search_document",
    }

    def embed(self, text: str, *, task_type: str | None = None) -> list[float]:
        """Embed a single text."""
        result = self.embed_batch([text], task_type=task_type)
        return result[0]

    def embed_batch(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
        """Embed multiple texts with automatic batching."""
        if not texts:
            return []

        input_type = (
            self._TASK_TYPE_MAP.get(task_type, self._input_type) if task_type else self._input_type
        )

        all_embeddings: list[list[float]] = []
        batch_size = 96  # Cohere's limit

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._embed_single_batch(batch, input_type=input_type)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _embed_single_batch(
        self, texts: list[str], input_type: str | None = None
    ) -> list[list[float]]:
        """Embed a single batch."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "texts": texts,
            "input_type": input_type or self._input_type,
            "embedding_types": ["float"],
        }
        if self._dimensions:
            kwargs["output_dimension"] = self._dimensions

        response = self._client.embed(**kwargs)

        # V2 API returns embeddings.float
        if hasattr(response, "embeddings") and hasattr(response.embeddings, "float_"):
            return [list(e) for e in response.embeddings.float_]
        elif hasattr(response, "embeddings") and isinstance(response.embeddings, dict):
            return [list(e) for e in response.embeddings.get("float", [])]

        raise RuntimeError(f"Unexpected embedding response format: {response}")

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        if self._dimensions is None:
            try:
                result = self.embed("test")
                self._dimensions = len(result)
            except Exception:
                return 1024
        return self._dimensions or 1024


class CohereRerank:
    """
    Cohere rerank provider using the official SDK.

    Args:
        auth: Authentication provider.
        model: Model name override.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
    ) -> None:
        import cohere
        import httpx

        request_kwargs = auth.get_request_kwargs()

        httpx_client = httpx.Client(
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(300.0, connect=5.0),
        )

        self._client = cohere.ClientV2(
            api_key="unused",  # SDK requires non-empty, httpx_client auth overrides
            httpx_client=httpx_client,
        )
        self._model = model or RERANK_MODEL

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query."""
        if not documents:
            return []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            kwargs["top_n"] = top_n

        response = self._client.rerank(**kwargs)

        return [RerankResult(index=r.index, score=r.relevance_score) for r in response.results]


__all__ = [
    "CohereChat",
    "CohereEmbedding",
    "CohereRerank",
    "CHAT_MODELS",
    "EMBEDDING_MODEL",
    "RERANK_MODEL",
]
