# fitz_ai/engines/fitz_rag/engine.py
"""
FitzRagEngine - Knowledge engine implementation for Fitz RAG paradigm.

This engine wraps the existing RAG pipeline (retrieval + generation) behind
the paradigm-agnostic KnowledgeEngine interface.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fitz_ai.cloud import CloudClient
from fitz_ai.cloud.cache_key import CacheVersions, compute_retrieval_fingerprint
from fitz_ai.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    Query,
    QueryError,
)
from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import RGSAnswer
from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline
from fitz_ai.logging import get_logger

if TYPE_CHECKING:
    from fitz_ai.cloud.client import CacheLookupResult

logger = get_logger(__name__)


class FitzRagEngine:
    """
    Fitz RAG engine implementation.

    This engine implements the retrieval-augmented generation paradigm:
    1. Embed the query
    2. Retrieve relevant chunks from vector DB
    3. Optionally rerank chunks
    4. Generate answer using LLM + retrieved context

    The engine wraps the existing RAGPipeline and adapts it to the
    KnowledgeEngine protocol.

    Examples:
        >>> from fitz_ai.engines.fitz_rag.config import load_config
        >>>
        >>> config = load_config("fitz.yaml")
        >>> engine = FitzRagEngine(config)
        >>>
        >>> query = Query(text="What is quantum computing?")
        >>> answer = engine.answer(query)
        >>> print(answer.text)
        >>> for source in answer.provenance:
        ...     print(f"Source: {source.source_id}")
    """

    def __init__(self, config: FitzRagConfig):
        """
        Initialize the Fitz RAG engine.

        Args:
            config: FitzRagConfig object with all RAG settings

        Raises:
            ConfigurationError: If configuration is invalid or required
                              components cannot be initialized
        """
        try:
            # Use the factory method to create RAGPipeline from config
            # This properly initializes all components (retrieval, llm, rgs, context)
            self._pipeline = RAGPipeline.from_config(config)
            self._config = config

            # Initialize cloud client if enabled
            self._cloud_client: CloudClient | None = None
            if config.cloud and config.cloud.enabled:
                config.cloud.validate_config()
                # Get org_id from environment or config
                org_id = os.environ.get("FITZ_ORG_ID")
                if not org_id:
                    raise ConfigurationError(
                        "FITZ_ORG_ID environment variable required when cloud is enabled"
                    )
                self._cloud_client = CloudClient(config.cloud, org_id)
                logger.info("Fitz Cloud enabled", extra={"org_id": org_id[:8]})

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Fitz RAG engine: {e}") from e

    def answer(self, query: Query) -> Answer:
        """
        Execute a query using Fitz RAG approach with optional cloud caching.

        Flow:
        1. Validate query
        2. Check cloud cache (if enabled) - COMING SOON
        3. Run RAG pipeline (retrieve → generate)
        4. Store in cloud cache (if enabled)
        5. Return answer

        Args:
            query: Query object with the question and optional constraints

        Returns:
            Answer object with generated text and source provenance

        Raises:
            QueryError: If query is invalid
            KnowledgeError: If retrieval fails
            GenerationError: If answer generation fails
        """
        # Validate query
        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        # TODO: Check cloud cache (requires embedder access)
        # if self._cloud_client:
        #     cache_result = self._check_cloud_cache(query.text)
        #     if cache_result.hit:
        #         logger.info("Cloud cache hit")
        #         return cache_result.answer

        try:
            # Run the RAG pipeline
            rgs_answer: RGSAnswer = self._pipeline.run(query.text)

            # Convert to standard Answer format
            provenance = [
                Provenance(
                    source_id=src.doc_id or src.source_id,
                    excerpt=src.content or "",
                    metadata=src.metadata,
                )
                for src in rgs_answer.sources
            ]

            answer = Answer(
                text=rgs_answer.answer,
                provenance=provenance,
                metadata={
                    "engine": "fitz_rag",
                    "query": query.text,
                },
            )

            # Store in cloud cache (if enabled)
            if self._cloud_client:
                self._store_in_cloud_cache(query.text, answer, rgs_answer)

            return answer

        except Exception as e:
            # Determine error type and re-raise appropriately
            error_msg = str(e).lower()
            if "retriev" in error_msg or "search" in error_msg:
                raise KnowledgeError(f"Retrieval failed: {e}") from e
            elif "generat" in error_msg or "llm" in error_msg:
                raise GenerationError(f"Generation failed: {e}") from e
            else:
                raise KnowledgeError(f"RAG pipeline error: {e}") from e

    def _store_in_cloud_cache(self, query_text: str, answer: Answer, rgs_answer: RGSAnswer) -> None:
        """
        Store answer in cloud cache.

        Note: For MVP, we're skipping cache storage since we need:
        - Query embedding (requires access to embedder)
        - Retrieval fingerprint (requires chunk IDs)
        - Collection version hash

        This will be implemented when we add cache lookup.
        """
        # TODO: Implement cache storage
        # Need to:
        # 1. Get query embedding from embedder
        # 2. Compute retrieval fingerprint from chunk IDs
        # 3. Get collection version
        # 4. Call cloud_client.store_cache()
        logger.debug("Cloud cache storage not yet implemented")

    @property
    def config(self) -> FitzRagConfig:
        """Get the engine's configuration."""
        return self._config

    @classmethod
    def from_yaml(cls, config_path: str) -> "FitzRagEngine":
        """
        Create engine from a YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configured FitzRagEngine instance
        """
        from fitz_ai.engines.fitz_rag.config import load_config

        config = load_config(config_path)
        return cls(config)
