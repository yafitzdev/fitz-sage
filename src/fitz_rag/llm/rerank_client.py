from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional, Iterable
import os

from fitz_rag.exceptions.retriever import RerankError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RERANK

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore

logger = get_logger(__name__)


class RerankClient(Protocol):
    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        ...


@dataclass
class CohereRerankClient:
    """
    Cohere Rerank API wrapper.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError("cohere is not installed. Run `pip install cohere`.")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set.")

        self.model = (
                self.model
                or os.getenv("COHERE_RERANK_MODEL")
                or "rerank-v3.5"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise RerankError("Failed to initialize Cohere Rerank client") from e

        logger.info(f"{RERANK} Initialized CohereRerankClient (model={self.model})")

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        if not documents:
            logger.debug(f"{RERANK} Called rerank() with zero documents — returning empty list")
            return []

        n = top_n if top_n is not None else len(documents)

        logger.debug(f"{RERANK} Reranking {len(documents)} docs (top_n={n}, query_len={len(query)})")

        try:
            res = self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=n,
            )
        except Exception as e:
            logger.error(f"{RERANK} Rerank request failed")
            raise RerankError("Rerank request failed") from e

        # Extract results
        try:
            indices = [r.index for r in res.results]
            logger.debug(f"{RERANK} Rerank succeeded — returning {len(indices)} indices")
            return indices
        except Exception as e:
            logger.error(f"{RERANK} Rerank response malformed")
            raise RerankError("Malformed rerank response") from e


@dataclass
class DummyRerankClient:
    """Deterministic test reranker."""

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        logger.debug(f"{RERANK} Dummy reranker invoked for {len(documents)} documents")

        if not documents:
            return []

        indices = list(range(len(documents)))
        indices.sort(key=lambda i: len(documents[i]), reverse=True)

        if top_n is not None:
            indices = indices[:top_n]

        return indices
