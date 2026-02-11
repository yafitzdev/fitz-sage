# fitz_ai/llm/providers/ollama.py
"""
Ollama provider wrappers using direct HTTP calls.

Ollama runs locally and doesn't require authentication.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import httpx

from fitz_ai.llm.providers.base import ModelTier, RerankResult

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

# Default models by tier (common Ollama models)
CHAT_MODELS: dict[ModelTier, str] = {
    "smart": "qwen2.5:14b",
    "balanced": "qwen2.5:7b",
    "fast": "qwen2.5:3b",
}

EMBEDDING_MODEL = "nomic-embed-text"
RERANK_MODEL = "qllama/bge-reranker-v2-m3"


class OllamaChat:
    """
    Ollama chat provider using direct HTTP calls.

    Args:
        model: Model name.
        tier: Model tier (smart, balanced, fast).
        base_url: Ollama server URL.
        **kwargs: Additional default kwargs for chat calls.
    """

    def __init__(
        self,
        model: str | None = None,
        tier: ModelTier = "smart",
        base_url: str | None = None,
        models: dict[ModelTier, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url or DEFAULT_BASE_URL
        # Use provided models dict, falling back to defaults
        tier_models = models or CHAT_MODELS
        self._model = model or tier_models.get(tier) or CHAT_MODELS[tier]
        self._defaults = kwargs
        self._client = httpx.Client(base_url=self._base_url, timeout=120.0)

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion."""
        params = {**self._defaults, **kwargs}

        payload = {
            "model": params.pop("model", self._model),
            "messages": messages,
            "stream": False,
        }

        # Map standard parameters to Ollama options format
        options = params.pop("options", {})
        if "max_tokens" in params:
            options["num_predict"] = params.pop("max_tokens")
        if "temperature" in params:
            options["temperature"] = params.pop("temperature")
        if options:
            payload["options"] = options

        payload.update(params)

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        return ""

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """Generate a streaming chat completion."""
        params = {**self._defaults, **kwargs}

        payload = {
            "model": params.pop("model", self._model),
            "messages": messages,
            "stream": True,
        }

        # Map standard parameters to Ollama options format
        options = params.pop("options", {})
        if "max_tokens" in params:
            options["num_predict"] = params.pop("max_tokens")
        if "temperature" in params:
            options["temperature"] = params.pop("temperature")
        if options:
            payload["options"] = options

        payload.update(params)

        with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


class OllamaEmbedding:
    """
    Ollama embedding provider using direct HTTP calls.

    Args:
        model: Model name.
        base_url: Ollama server URL.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._base_url = base_url or DEFAULT_BASE_URL
        self._model = model or EMBEDDING_MODEL
        self._client = httpx.Client(base_url=self._base_url, timeout=60.0)
        self._dimensions: int | None = None

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        payload = {
            "model": self._model,
            "input": text,
        }

        response = self._client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        if "embeddings" in data and data["embeddings"]:
            embedding = data["embeddings"][0]
            if self._dimensions is None:
                self._dimensions = len(embedding)
            return list(embedding)

        raise RuntimeError(f"Unexpected embedding response: {data}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        payload = {
            "model": self._model,
            "input": texts,
        }

        response = self._client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        if "embeddings" in data:
            embeddings = [list(e) for e in data["embeddings"]]
            if embeddings and self._dimensions is None:
                self._dimensions = len(embeddings[0])
            return embeddings

        raise RuntimeError(f"Unexpected embedding response: {data}")

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        if self._dimensions is None:
            # Fetch dimensions by embedding a test string
            self.embed("test")
        return self._dimensions or 768  # Default fallback

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


class OllamaRerank:
    """
    Ollama reranker using cross-encoder models like qwen3-reranker.

    Uses the chat API to score query-document relevance.

    Args:
        model: Model name (e.g., "sam860/qwen3-reranker").
        base_url: Ollama server URL.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._base_url = base_url or DEFAULT_BASE_URL
        self._model = model or RERANK_MODEL
        self._client = httpx.Client(base_url=self._base_url, timeout=60.0)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query."""
        if not documents:
            return []

        # Score each document
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            score = self._score_document(query, doc)
            scored.append((i, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n limit
        if top_n is not None:
            scored = scored[:top_n]

        return [RerankResult(index=idx, score=score) for idx, score in scored]

    def _score_document(self, query: str, document: str) -> float:
        """Score a single document's relevance to the query."""
        # Truncate long documents to avoid token limits
        max_doc_chars = 2000
        if len(document) > max_doc_chars:
            document = document[:max_doc_chars] + "..."

        # Use a simple relevance scoring prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance scoring assistant. "
                    "Given a query and a document, output ONLY a relevance score "
                    "from 0.0 to 1.0 where 1.0 means highly relevant. "
                    "Output nothing but the number."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument: {document}\n\nRelevance score:",
            },
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10},
        }

        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"].strip()
                # Parse the score - handle various formats
                score = self._parse_score(content)
                return score
        except Exception as e:
            logger.warning(f"Failed to score document: {e}")

        return 0.0  # Default to low score on error

    def _parse_score(self, content: str) -> float:
        """Parse a relevance score from model output."""
        import re

        # Try to extract a number from the response
        # Handle formats like "0.8", "0.8/1.0", "Score: 0.8", etc.
        match = re.search(r"(\d+\.?\d*)", content)
        if match:
            score = float(match.group(1))
            # Normalize if score > 1 (might be out of 10 or 100)
            if score > 1.0:
                if score <= 10:
                    score = score / 10.0
                elif score <= 100:
                    score = score / 100.0
                else:
                    score = 1.0
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        return 0.0

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


VISION_MODEL = "llava:7b"


class OllamaVision:
    """
    Ollama vision provider using multimodal models (e.g., LLaVA).

    Uses the /api/chat endpoint with images in messages for VLM inference.

    Args:
        model: Model name (e.g., "llava:7b").
        base_url: Ollama server URL.
        **kwargs: Additional default kwargs.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url or DEFAULT_BASE_URL
        self._model = model or VISION_MODEL
        self._client = httpx.Client(base_url=self._base_url, timeout=120.0)
        self._defaults = kwargs

    def describe_image(self, image_base64: str, prompt: str | None = None) -> str:
        """
        Describe an image using a vision model.

        Args:
            image_base64: Base64-encoded image data (raw, no data URI prefix).
            prompt: Custom prompt for description.

        Returns:
            Text description of the image.
        """
        actual_prompt = prompt or (
            "Describe this figure, chart, or diagram in detail. "
            "Include all data values, labels, trends, and key insights visible in the image."
        )

        # Ollama multimodal API: images go in messages[].images as raw base64 array
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": actual_prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
        }

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        return ""

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


__all__ = [
    "OllamaChat",
    "OllamaEmbedding",
    "OllamaRerank",
    "OllamaVision",
    "CHAT_MODELS",
    "EMBEDDING_MODEL",
    "RERANK_MODEL",
    "VISION_MODEL",
    "DEFAULT_BASE_URL",
]
