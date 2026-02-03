# fitz_ai/llm/providers/ollama.py
"""
Ollama provider wrappers using direct HTTP calls.

Ollama runs locally and doesn't require authentication.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import httpx

from fitz_ai.llm.providers.base import ModelTier

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

# Default models by tier (common Ollama models)
CHAT_MODELS: dict[ModelTier, str] = {
    "smart": "qwen2.5:14b",
    "balanced": "qwen2.5:7b",
    "fast": "qwen2.5:3b",
}

EMBEDDING_MODEL = "nomic-embed-text"


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


__all__ = [
    "OllamaChat",
    "OllamaEmbedding",
    "CHAT_MODELS",
    "EMBEDDING_MODEL",
    "DEFAULT_BASE_URL",
]
