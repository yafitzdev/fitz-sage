# fitz/core/llm/rerank/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.rerank import LocalReranker, LocalRerankerConfig
from fitz.core.llm.rerank.base import RerankPlugin
from fitz.core.models.chunk import Chunk


class LocalRerankClient(RerankPlugin):
    """
    Local fallback rerank plugin.

    Thin adapter around fitz.backends.local_llm.rerank.LocalReranker.
    """

    plugin_name = "local"
    plugin_type = "rerank"
    availability = "local"

    def __init__(self, **kwargs: Any):
        cfg = LocalRerankerConfig(**kwargs)
        self._reranker = LocalReranker(cfg)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return self._reranker.rerank(query, chunks)
