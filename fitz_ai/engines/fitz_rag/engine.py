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
from fitz_ai.logging.logger import get_logger

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
        >>> from fitz_ai.config import load_engine_config
        >>>
        >>> config = load_engine_config("fitz_rag")
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
            self._config = config

            # Initialize cloud client if enabled (must be before pipeline creation)
            self._cloud_client: CloudClient | None = None
            if config.cloud and config.cloud.enabled:
                config.cloud.validate_config()
                # Get org_id from config first, then environment variable
                org_id = config.cloud.org_id or os.environ.get("FITZ_ORG_ID")
                if not org_id:
                    raise ConfigurationError(
                        "cloud.org_id or FITZ_ORG_ID environment variable required when cloud is enabled"
                    )
                self._cloud_client = CloudClient(config.cloud, org_id)
                logger.info("Fitz Cloud enabled", extra={"org_id": org_id[:8]})

            # Use the factory method to create RAGPipeline from config
            # This properly initializes all components (retrieval, llm, rgs, context)
            # Pass cloud_client to enable cache operations
            self._pipeline = RAGPipeline.from_config(config, cloud_client=self._cloud_client)

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Fitz RAG engine: {e}") from e

    def answer(self, query: Query) -> Answer:
        """
        Execute a query using Fitz RAG approach with optional cloud caching.

        Flow:
        1. Validate query
        2. Run RAG pipeline (retrieve → cache check → generate → cache store)
        3. Return answer

        Cache operations (lookup and storage) are handled transparently within
        the RAG pipeline when cloud is enabled.

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
        import yaml
        from pathlib import Path

        with Path(config_path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Unwrap fitz_rag key if present
        if "fitz_rag" in raw:
            config_dict = raw["fitz_rag"]
        else:
            config_dict = raw

        config = FitzRagConfig(**config_dict)
        return cls(config)
