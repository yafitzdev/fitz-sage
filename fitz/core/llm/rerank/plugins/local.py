from __future__ import annotations

from typing import List

from fitz.core.llm.rerank.base import RerankPlugin
from fitz.core.models.chunk import Chunk


class LocalRerankClient(RerankPlugin):
    """
    Local fallback reranker.

    Baseline quality:
    - token overlap
    - substring presence
    - deterministic ordering

    Purpose:
    - make the pipeline runnable without API keys
    - validate retrieval + RGS wiring
    """

    plugin_name = "local"
    plugin_type = "rerank"
    availability = "local"

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks

        q = query.lower()
        q_tokens = set(q.split())

        def score(chunk: Chunk) -> float:
            text = chunk.content.lower()

            # token overlap
            tokens = set(text.split())
            overlap = len(tokens & q_tokens)

            # substring boost
            substring = 1.0 if q in text else 0.0

            # mild length normalization (avoid giant chunks winning)
            length_penalty = max(len(tokens), 1)

            return (overlap * 2.0 + substring) / length_penalty

        return sorted(chunks, key=score, reverse=True)
