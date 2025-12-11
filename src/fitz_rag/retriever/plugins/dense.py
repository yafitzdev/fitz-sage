from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

from fitz_rag.core import Chunk  # universal chunk

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
    RerankError,
)

# Config types
from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

# Logging
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RETRIEVER, EMBEDDING, VECTOR_SEARCH, RERANK

from fitz_rag.retriever.base import RetrievalPlugin

logger = get_logger(__name__)


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    """
    Dense vector retrieval plugin for fitz-rag.

    Default strategy:
      1) Embed query via EmbeddingEngine
      2) Dense vector search via Qdrant
      3) Optional rerank via RerankEngine
      4) Returns Chunk objects (universal model)

    This class is auto-registered in the retrieval registry.
    """

    plugin_name: str = "dense"

    # Core dependencies
    client: Any = None
    embed_cfg: EmbeddingConfig | None = None
    retriever_cfg: RetrieverConfig | None = None
    rerank_cfg: RerankConfig | None = None

    # Injected dependencies (overridden in tests)
    embedder: Any | None = None
    rerank_engine: RerankEngine | None = None

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    def __post_init__(self) -> None:
        if self.embed_cfg is None or self.retriever_cfg is None:
            raise ValueError("embed_cfg and retriever_cfg must be provided")

        # 1. EMBEDDING ENGINE
        if self.embedder is None:
            embed_plugin = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
                input_type="search_query",
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
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")

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
        # 2) VECTOR SEARCH
        # -----------------------------
        try:
            logger.debug(
                f"{VECTOR_SEARCH} Qdrant search → collection='{self.retriever_cfg.collection}', "
                f"top_k={self.retriever_cfg.top_k}"
            )

            try:
                # Real Qdrant client (keyword args)
                hits = self.client.search(
                    collection_name=self.retriever_cfg.collection,
                    query_vector=query_vector,
                    limit=self.retriever_cfg.top_k,
                    with_payload=True,
                )
            except TypeError:
                # MockQdrantSearchClient in tests (positional)
                hits = self.client.search(
                    self.retriever_cfg.collection,
                    query_vector,
                    self.retriever_cfg.top_k,
                )

        except Exception as e:
            logger.error(f"{VECTOR_SEARCH} Search failed: {e}")
            raise VectorSearchError(
                f"Vector search failed for collection '{self.retriever_cfg.collection}'"
            ) from e

        # -----------------------------
        # Convert hits → universal Chunk
        # -----------------------------
        chunks: List[Chunk] = []

        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}

            chunks.append(
                Chunk(
                    id=getattr(hit, "id", None),              # universal Chunk.id
                    text=payload.get("text", ""),             # universal Chunk.text
                    metadata=payload,                         # universal Chunk.metadata
                    score=getattr(hit, "score", None),        # universal Chunk.score
                )
            )

        # -----------------------------
        # 3) OPTIONAL RERANKING
        # -----------------------------
        if self.rerank_engine:
            logger.debug(f"{RERANK} Running rerank engine on {len(chunks)} chunks")

            try:
                chunks = self.rerank_engine.plugin.rerank(query, chunks)
            except Exception as e:
                logger.error(f"{RERANK} Rerank failed: {e}")
                raise RerankError("Reranking failed") from e

        return chunks


# -------------------------------------------------------------------
# Backwards compatible alias used throughout the codebase & tests.
# -------------------------------------------------------------------
@dataclass
class RAGRetriever(DenseRetrievalPlugin):
    """
    Backwards-compatible alias for DenseRetrievalPlugin.

    Existing code can continue to import:

        from fitz_rag.retriever.plugins.dense import RAGRetriever
    """
    pass
