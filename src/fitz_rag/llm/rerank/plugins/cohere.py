"""
Cohere reranking client for Fitz-RAG (ClientV2 version).

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

from fitz_rag.exceptions.retriever import RerankError

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereRerankClient:
    """
    Minimal Cohere reranking wrapper using Cohere ClientV2.

    Parameters
    ----------
    plugin_name:
        For future plugin-routing; currently unused.
    api_key:
        Optional explicit API key. If not provided, COHERE_API_KEY env is used.
    model:
        Cohere rerank model name (e.g., "rerank-english-v3.0").
    """

    def __init__(
        self,
        plugin_name: str = "cohere",
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:

        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        self.plugin_name = plugin_name
        self.model = model

        # Resolve API key (explicit or env)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set for CohereRerankClient")

        try:
            # Modern interface
            self._client = cohere.ClientV2(api_key=self.api_key)
        except Exception as e:
            raise RerankError("Failed to initialize Cohere rerank client") from e

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
            # Normalize input to list[str]
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

            # Call Cohere v2 rerank endpoint
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
            )

            # Cohere v2 returns:
            # response.results → list of objects with fields:
            #   - index (int)
            #   - relevance_score (float)
            #
            # We sort by score (descending) and return the indices.
            sorted_results = sorted(
                response.results,
                key=lambda r: r.relevance_score,
                reverse=True,
            )

            return [r.index for r in sorted_results]

        except Exception as e:
            raise RerankError("Reranking failed") from e
