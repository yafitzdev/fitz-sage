# fitz/core/llm/embedding/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.embedding import LocalEmbedder, LocalEmbedderConfig
from fitz.core.llm.embedding.base import EmbeddingPlugin


class LocalEmbeddingClient(EmbeddingPlugin):
    """
    Local fallback embedding plugin.

    Thin adapter around fitz.backends.local_llm.embedding.LocalEmbedder.
    """

    plugin_name = "local"
    plugin_type = "embedding"
    availability = "local"

    def __init__(self, **kwargs: Any):
        cfg = LocalEmbedderConfig(**kwargs)
        self._embedder = LocalEmbedder(cfg)

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)
