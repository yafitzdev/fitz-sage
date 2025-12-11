from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fitz_rag.core import Chunk

# Embedding
from fitz_rag.llm.embedding.engine import EmbeddingEngine
from fitz_rag.llm.embedding.plugins.cohere import CohereEmbeddingClient

# Reranking
from fitz_rag.llm.rerank.engine import RerankEngine
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient

# Exceptions
from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
)

# Config types
from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

# Logging
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RETRIEVER, EMBEDDING, VECTOR_SEARCH, RERANK

logger = get_logger(__name__)


@dataclass
class RAGRetriever:
    """
    Dense vector retriever plugin for fitz-rag.

    Pipeline:
      1) EmbeddingEngine embeds the query
      2) Dense vector search via Qdrant
      3) RerankEngine optionally reranks the chunks
      4) Returns Chunk objects
    """

    client: any
    embed_cfg: EmbeddingConfig
    retriever_cfg: RetrieverConfig
    rerank_cfg: RerankConfig | None = None

    # Injected dependencies (overridden in tests)
    embedder: EmbeddingEngine | None = None
    rerank_engine: RerankEngine | None = None

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    def __post_init__(self):
        # 1. EMBEDDING ENGINE
        if self.embedder is None:
            embed_plugin = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
                input_type=self.embed_cfg.input_type,
                output_dimension=self.embed_cfg.output_dimension,
            )
            self.embedder = EmbeddingEngine(embed_plugin)

        # 2. RERANK ENGINE (if enabled)
        if self.rerank_cfg and self.rerank_cfg.enabled:
            rerank_plugin = CohereRerankClient(
                api_key=self.rerank_cfg.api_key,
                model=self.rerank_cfg.model,
            )
            self.rerank_engine = RerankEngine(rerank_plugin)

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        # -----------------------------
        # 1) EMBED QUERY
        # -----------------------------
        try:
            logger.debug(f"{EMBEDDING} Embedding query...")
            query_vector = self.embedder.embed(query)
        except Exception as e:
            logger.error(f"{EMBEDDING} Embedding failed for query='{query}'")
            raise EmbeddingError(f"Failed to embed query: {query}") from e

        # -----------------------------
        # 2) VECTOR SEARCH (Qdrant)
        # -----------------------------
        try:
            logger.debug(
                f"{VECTOR_SEARCH} Qdrant search → collection='{self.retriever_cfg.collection}', "
                f"top_k={self.retriever_cfg.top_k}"
            )

            hits = self.client.search(
                collection_name=self.retriever_cfg.collection,
                vector=query_vector,
                limit=self.retriever_cfg.top_k,
                with_payload=True,
            )

        except Exception as e:
            logger.error(f"{VECTOR_SEARCH} Search failed: {e}")
            raise VectorSearchError(
                f"Vector search failed for collection '{self.retriever_cfg.collection}'"
            ) from e

        # Convert hits → Chunk objects
        chunks: List[Chunk] = []

        for hit in hits:
            text = hit.payload.get("text", "")
            metadata = hit.payload
            score = getattr(hit, "score", 0.0)

            chunks.append(
                Chunk(
                    text,
                    metadata,
                    score,
                )
            )

        # -----------------------------
        # 3) OPTIONAL RERANKING
        # -----------------------------
        if self.rerank_engine:
            logger.debug(f"{RERANK} Running rerank engine on {len(chunks)} chunks")
            chunks = self.rerank_engine.rerank(query, chunks)

        return chunks
