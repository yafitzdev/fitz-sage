from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient

from fitz_rag.models.chunk import Chunk
from fitz_rag.config.schema import (
    RetrieverConfig,
    EmbeddingConfig,
    RerankConfig,
)
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.embedding_client import DummyEmbeddingClient

from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
    RerankError,
)


class RAGRetriever:
    """
    Unified-config retriever.

    Supports:
      - embedding (via EmbeddingConfig)
      - vector search (Qdrant)
      - optional reranking (via RerankConfig)
      - structured exceptions
    """

    def __init__(
        self,
        client: QdrantClient,
        embed_cfg: EmbeddingConfig,
        retriever_cfg: RetrieverConfig,
        rerank_cfg: Optional[RerankConfig] = None,
    ):
        self.client = client
        self.retriever_cfg = retriever_cfg

        # -------------------------------------------------------
        # Embedding backend (strict)
        # -------------------------------------------------------
        provider = embed_cfg.provider.lower()

        if provider not in ("cohere",):
            raise ValueError(f"Unsupported embedding provider: {embed_cfg.provider}")

        self.embedder = CohereEmbeddingClient(
            api_key=embed_cfg.api_key,
            model=embed_cfg.model,
        )

        # Test hack: allow replacing embedder manually after init
        self._allow_embedder_patch = True

        # -------------------------------------------------------
        # Reranker backend
        # -------------------------------------------------------
        if rerank_cfg and rerank_cfg.enabled:
            if rerank_cfg.provider.lower() != "cohere":
                raise ValueError(f"Unsupported rerank provider: {rerank_cfg.provider}")
            if not rerank_cfg.model:
                raise ValueError("rerank.model must be provided")

            self.reranker = CohereRerankClient(
                api_key=rerank_cfg.api_key,
                model=rerank_cfg.model,
            )
        else:
            self.reranker = None

    # ---------------------------------------------------------------------
    # Main public API
    # ---------------------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        """
        Full retrieval pipeline with structured exceptions:
        1) embed query
        2) vector search
        3) optional rerank
        4) return Chunk objects
        """

        # 1. EMBEDDING
        try:
            query_vector = self.embedder.embed(query)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {query}") from e

        # 2. VECTOR SEARCH
        try:
            hits = self.client.search(
                collection_name=self.retriever_cfg.collection,
                query_vector=query_vector,
                limit=self.retriever_cfg.top_k,
                with_payload=True,
            )
        except Exception as e:
            raise VectorSearchError(
                f"Vector search failed for collection '{self.retriever_cfg.collection}'"
            ) from e

        # Convert to chunks
        chunks = self._to_chunks(hits)

        # 3. RERANK
        if self.reranker:
            try:
                chunks = self._apply_rerank(query, chunks)
            except Exception as e:
                raise RerankError("Reranking failed") from e

        return chunks

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _to_chunks(self, hits) -> List[Chunk]:
        chunks: List[Chunk] = []

        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            doc_id = payload.get("doc_id", "unknown")

            chunks.append(
                Chunk(
                    id=str(hit.id),
                    doc_id=doc_id,
                    content=text,
                    metadata=metadata,
                    chunk_index=metadata.get("chunk_index", 0),
                )
            )

        return chunks

    def _apply_rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        result = self.reranker.rerank(query=query, documents=texts)
        return [chunks[i] for i in result.indices]
