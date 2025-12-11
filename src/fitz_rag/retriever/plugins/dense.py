from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

from fitz_rag.core import Chunk

from fitz_rag.llm.embedding.engine import EmbeddingEngine
from fitz_rag.llm.embedding.plugins.cohere import CohereEmbeddingClient

from fitz_rag.llm.rerank.engine import RerankEngine
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient

from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
    RerankError,
)

from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RETRIEVER, EMBEDDING, VECTOR_SEARCH, RERANK

from fitz_rag.retriever.base import RetrievalPlugin

logger = get_logger(__name__)


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    plugin_name: str = "dense"

    client: Any = None
    embed_cfg: EmbeddingConfig | None = None
    retriever_cfg: RetrieverConfig | None = None
    rerank_cfg: RerankConfig | None = None

    embedder: Any | None = None
    rerank_engine: RerankEngine | None = None

    def __post_init__(self) -> None:
        # retriever_cfg is ALWAYS required
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")

        # embed_cfg is required ONLY when no embedder is injected
        if self.embedder is None and self.embed_cfg is None:
            raise ValueError("embed_cfg must be provided when no embedder is injected")

        # ------------------------------------------------
        # 1) Build embedder if not injected
        # ------------------------------------------------
        if self.embedder is None:
            embed_plugin = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
                input_type="search_query",
                output_dimension=self.embed_cfg.output_dimension,
            )
            self.embedder = EmbeddingEngine(embed_plugin)

        # ------------------------------------------------
        # 2) Build rerank engine ONLY if not injected
        # ------------------------------------------------
        if self.rerank_cfg and self.rerank_cfg.enabled and self.rerank_engine is None:
            rerank_plugin = CohereRerankClient(
                api_key=self.rerank_cfg.api_key,
                model=self.rerank_cfg.model,
            )
            self.rerank_engine = RerankEngine(rerank_plugin)

    # ---------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        # -----------------------------
        # 1) EMBED
        # -----------------------------
        try:
            query_vector = self.embedder.embed(query)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {query}") from e

        # -----------------------------
        # 2) SEARCH
        # -----------------------------
        try:
            try:
                hits = self.client.search(
                    collection_name=self.retriever_cfg.collection,
                    query_vector=query_vector,
                    limit=self.retriever_cfg.top_k,
                    with_payload=True,
                )
            except TypeError:
                hits = self.client.search(
                    self.retriever_cfg.collection,
                    query_vector,
                    self.retriever_cfg.top_k,
                )
        except Exception as e:
            raise VectorSearchError("Vector search failed") from e

        # -----------------------------
        # 3) BUILD CHUNKS
        # -----------------------------
        chunks: List[Chunk] = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            chunks.append(
                Chunk(
                    id=getattr(hit, "id", None),
                    text=payload.get("text", ""),
                    metadata=payload,
                    score=getattr(hit, "score", None),
                )
            )

        # -----------------------------
        # 4) OPTIONAL RERANK
        # -----------------------------
        if self.rerank_engine:
            try:
                chunks = self.rerank_engine.plugin.rerank(query, chunks)
            except Exception as e:
                raise RerankError("Reranking failed") from e

        return chunks


@dataclass
class RAGRetriever(DenseRetrievalPlugin):
    pass
