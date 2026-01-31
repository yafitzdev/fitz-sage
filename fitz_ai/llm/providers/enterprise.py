# fitz_ai/llm/providers/enterprise.py
"""
Enterprise gateway provider.

Simple httpx-based client for enterprise LLM gateways. No SDK dependencies.
Assumes OpenAI-compatible API format (POST /chat/completions).

Model strings are passed through verbatim - the gateway interprets them.
Examples: "openai/gpt-4o" (BMW), "gpt-4o" (generic), "my-deployment" (Azure)
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import httpx

from fitz_ai.llm.auth import AuthProvider
from fitz_ai.llm.auth.httpx_auth import DynamicHttpxAuth

logger = logging.getLogger(__name__)


class EnterpriseChat:
    """
    Enterprise gateway chat provider.

    Args:
        auth: Authentication provider (M2MAuth, CompositeAuth, etc.)
        base_url: Gateway URL (e.g., "https://llm.corp.internal/v1")
        model: Model string passed verbatim to gateway
        **kwargs: Default kwargs for chat calls (temperature, max_tokens, etc.)
    """

    def __init__(
        self,
        auth: AuthProvider,
        base_url: str,
        model: str,
        **kwargs: Any,
    ) -> None:
        request_kwargs = auth.get_request_kwargs()

        self._client = httpx.Client(
            base_url=base_url,
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(600.0, connect=5.0),
        )
        self._model = model
        self._defaults = kwargs

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion."""
        params = {**self._defaults, **kwargs}

        body = {
            "model": params.pop("model", self._model),
            "messages": messages,
            **params,
        }

        response = self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        # OpenAI-compatible response format
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "")
        return ""

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """Generate a streaming chat completion."""
        params = {**self._defaults, **kwargs}

        body = {
            "model": params.pop("model", self._model),
            "messages": messages,
            "stream": True,
            **params,
        }

        with self._client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(chunk)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except (json.JSONDecodeError, KeyError):
                        continue


class EnterpriseEmbedding:
    """
    Enterprise gateway embedding provider.

    Args:
        auth: Authentication provider
        base_url: Gateway URL
        model: Model string passed verbatim
        dimensions: Optional embedding dimensions (if gateway supports)
    """

    def __init__(
        self,
        auth: AuthProvider,
        base_url: str,
        model: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        request_kwargs = auth.get_request_kwargs()

        self._client = httpx.Client(
            base_url=base_url,
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(300.0, connect=5.0),
        )
        self._model = model
        self._dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        body: dict[str, Any] = {
            "model": self._model,
            "input": text,
        }
        if self._dimensions:
            body["dimensions"] = self._dimensions

        response = self._client.post("/embeddings", json=body)
        response.raise_for_status()
        data = response.json()

        # OpenAI-compatible response format
        if "data" in data and data["data"]:
            return data["data"][0].get("embedding", [])
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        body: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions:
            body["dimensions"] = self._dimensions

        response = self._client.post("/embeddings", json=body)
        response.raise_for_status()
        data = response.json()

        # OpenAI-compatible response format
        if "data" in data:
            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            return [item.get("embedding", []) for item in sorted_data]
        return []

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        if self._dimensions:
            return self._dimensions
        # Default assumption for OpenAI-compatible embeddings
        return 1536


__all__ = [
    "EnterpriseChat",
    "EnterpriseEmbedding",
]
