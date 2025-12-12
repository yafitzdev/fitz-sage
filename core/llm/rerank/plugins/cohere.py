# core/llm/rerank/plugins/cohere.py
from __future__ import annotations

from typing import Any
import os

from rag.exceptions.retriever import RerankError

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereChatClient:
    plugin_name = "cohere"
    plugin_type = "chat"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "command-r-plus",
        temperature: float = 0.2,
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        self.model = model
        self.temperature = temperature

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is not set for CohereChatClient")

        self._client = cohere.ClientV2(api_key=key)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        msg = getattr(resp, "message", None)
        if msg is None:
            return str(resp)

        content = getattr(msg, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text

        text = getattr(msg, "text", None)
        if isinstance(text, str):
            return text

        return str(resp)


class CohereRerankClient:
    plugin_name = "cohere"
    plugin_type = "rerank"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        self.model = model

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is not set for CohereRerankClient")

        self._client = cohere.ClientV2(api_key=key)

    def rerank(self, query: str, chunks: list["Chunk"]) -> list["Chunk"]:
        # Local import to avoid a hard dependency cycle at import time.
        from rag.models.chunk import Chunk

        if not all(isinstance(c, Chunk) for c in chunks):
            raise TypeError("CohereRerankClient.rerank expects list[Chunk]")

        try:
            docs = [c.content for c in chunks]
            response = self._client.rerank(model=self.model, query=query, documents=docs)

            sorted_results = sorted(
                response.results,
                key=lambda r: r.relevance_score,
                reverse=True,
            )
            return [chunks[r.index] for r in sorted_results]
        except Exception as exc:
            raise RerankError("Reranking failed") from exc
