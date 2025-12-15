# fitz/core/llm/embedding/plugins/local.py
from __future__ import annotations

from typing import Any, List

from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.core.llm.embedding.base import EmbeddingPlugin


class LocalEmbeddingClient(EmbeddingPlugin):
    """
    Local fallback embedding plugin.

    This uses the same local runtime as local chat/rerank.
    If no local runtime is available, a clean LLMError is raised upstream.
    """

    plugin_name = "local"
    plugin_type = "embedding"
    availability = "local"

    def __init__(self, **kwargs: Any) -> None:
        # Keep config minimal and deterministic
        cfg = LocalLLMRuntimeConfig(
            model="llama3",
        )
        self._rt = LocalLLMRuntime(cfg)

    def embed(self, text: str) -> List[float]:
        # This will raise the clean local-fallback error if unavailable
        llm = self._rt.llama()

        # llama-cpp-python embedding API (best-effort across versions)
        if hasattr(llm, "embed"):
            vec = llm.embed(text)
            return list(vec)

        if hasattr(llm, "create_embedding"):
            resp = llm.create_embedding(input=text)
            data = resp.get("data") or []
            if not data:
                return []
            return list(data[0].get("embedding") or [])

        # Hard fallback: deterministic dummy embedding
        # (keeps pipeline runnable even without embedding support)
        return [0.0] * 384