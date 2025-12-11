"""
Cohere reranking client for Fitz-RAG.

This client accepts either:
- a list of strings
- a list of dict chunks with a 'text' field
- a list of objects with a .text attribute

It returns a list of integer indices indicating the ranking
(order of indices into the original list).
"""

from __future__ import annotations

from typing import Any, List
import os

import cohere

from fitz_rag.exceptions.retriever import RerankError


class CohereRerankClient:
    """
    Minimal Cohere reranking wrapper.

    Parameters
    ----------
    plugin_name:
        For future plugin-routing; currently unused.
    api_key:
        Optional explicit API key. If not provided, COHERE_API_KEY env is used.
    model:
        Cohere rerank model name.
    """

    def __init__(
        self,
        plugin_name: str = "cohere",
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:
        self.plugin_name = plugin_name
        self.model = model

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            # In normal use, tests call require_api_key() first and skip
            # if the env var is not set, so this should not trigger there.
            raise ValueError("COHERE_API_KEY is not set for CohereRerankClient")

        self._client = cohere.Client(self.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rerank(self, query: str, chunks: List[Any]) -> List[int]:
        """
        Parameters
        ----------
        query:
            User query / search string.
        chunks:
            Either:
                - list[str]
                - list[dict] with "text"
                - list[objects] with .text

        Returns
        -------
        List[int]:
            List of indices into the `chunks` list, sorted by descending relevance.
        """
        try:
            # Normalize all inputs to a list of document strings
            documents: List[str] = []

            for idx, c in enumerate(chunks):
                if isinstance(c, str):
                    documents.append(c)
                elif isinstance(c, dict) and isinstance(c.get("text"), str):
                    documents.append(c["text"])
                else:
                    text = getattr(c, "text", None)
                    if not isinstance(text, str):
                        raise TypeError(
                            f"Unsupported chunk format at index {idx}: {type(c)}"
                        )
                    documents.append(text)

            # Call Cohere rerank
            response = self._client.rerank(
                query=query,
                documents=documents,
                model=self.model,
            )

            # response.results elements usually have: index, relevance_score
            results_sorted = sorted(
                response.results,
                key=lambda r: r.relevance_score,
                reverse=True,
            )

            # Return just the indices into the original list
            return [r.index for r in results_sorted]

        except Exception as e:
            # Wrap *any* underlying problem in a domain-specific error
            raise RerankError("Reranking failed") from e
