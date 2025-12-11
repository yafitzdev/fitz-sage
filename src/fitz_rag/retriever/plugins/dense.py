from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fitz_rag.core import Chunk
from fitz_rag.llm.embedding_client import EmbeddingClient
from fitz_rag.llm.rerank_client import RerankClient
from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
    RerankError,
)
from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RETRIEVER, EMBEDDING, RERANK

logger = get_logger(__name__)


@dataclass
class RAGRetriever:
    """
    Dense vector retriever plugin (default retriever in fitz-rag).

    Steps:
      1) Embed query
      2) Vector search in Qdrant
      3) Optional rerank with Cohere
      4) Return Chunk objects
    """

    client: any
    embed_cfg: EmbeddingConfig
    retriever_cfg: RetrieverConfig
    rerank_cfg: RerankConfig | None = None
    embedder: EmbeddingClient | None = None
    reranker: RerankClient | None = None

    # ---------------------------------------------------------
    # Constructor: instantiate embedder/reranker lazily
    # ---------------------------------------------------------
    def __post_init__(self):
        from fitz_rag.llm.embedding_client import CohereEmbeddingClient
        from fitz_rag.llm.rerank_client import CohereRerankClient

        if self.embedder is None:
            self.embedder = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
            )

        if self.rerank_cfg and self.rerank_cfg.enabled:
            if self.reranker is None:
                self.reranker = CohereRerankClient(
                    api_key=self.rerank_cfg.api_key,
                    model=self.rerank_cfg.model,
                )

    # ---------------------------------------------------------
    # Main retrieval method
    # ---------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        """
        Full retrieval pipeline:
        1) embed query
        2) vector search
        3) optional rerank
        4) return Chunk objects
        """
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        # 1. EMBEDDING
        try:
            logger.debug(f"{EMBEDDING} Embedding query...")
            query_vector = self.embedder.embed(query)
        except Exception as e:
            logger.error(f"{EMBEDDING} Embedding failed for query='{query}'")
            raise EmbeddingError(f"Failed to embed query: {query}") from e

        # 2. VECTOR SEARCH
        try:
            logger.debug(
                f"{RETRIEVER} Vector search: collection='{self.retriever_cfg.collection}', "
                f"top_k={self.retriever_cfg.top_k}"
            )

            hits = self.client.search(
                collection_name=self.retriever_cfg.collection,
                vector=query_vector,
                limit=self.retriever_cfg.top_k,
                with_payload=True,
            )

        except Exception as e:
            logger.error(
                f"{RETRIEVER} Vector search failed for collection '{self.retriever_cfg.collection}'"
            )
            raise VectorSearchError(
                f"Vector search failed for collection '{self.retriever_cfg.collection}'"
            ) from e

        chunks: List[Chunk] = []
        for hit in hits:
            chunks.append(
                Chunk(
                    text=hit.payload.get("text", ""),
                    score=hit.score,
                    metadata=hit.payload,
                )
            )

        # 3. OPTIONAL RERANK
        if self.reranker and self.rerank_cfg.enabled and len(chunks) > 1:
            try:
                logger.debug(f"{RERANK} Running rerank on {len(chunks)} chunks")

                docs = [c.text for c in chunks]
                order = self.reranker.rerank(query, docs, top_n=len(docs))

                chunks = [chunks[i] for i in order]

            except Exception as e:
                logger.error(f"{RERANK} Rerank failed: {e}")
                raise RerankError("Reranking failed") from e

        return chunks
