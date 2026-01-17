# fitz_ai/integrations/base.py
"""Framework-agnostic Fitz Cloud optimizer.

This module provides the base FitzOptimizer class that handles all core
optimization logic. Framework-specific adapters (LangChain, LlamaIndex)
use this as their backend.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fitz_ai.cloud.cache_key import CacheVersions, compute_retrieval_fingerprint
from fitz_ai.cloud.client import REQUIRED_EMBEDDING_DIM, CloudClient
from fitz_ai.cloud.config import CloudConfig
from fitz_ai.core import Answer, Provenance
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result from cache lookup.

    Attributes:
        hit: Whether cache hit occurred
        answer: Answer text (if hit)
        sources: Source documents with excerpts (if hit)
        routing_advice: Model routing recommendation (if miss, Pro+ tiers)
    """

    hit: bool
    answer: Optional[str] = None
    sources: Optional[list[dict]] = None
    routing_advice: Optional[dict] = None


class FitzOptimizer:
    """
    Framework-agnostic Fitz Cloud optimizer.

    Handles cache lookup/store, encryption, and routing advice.
    Framework adapters (LangChain, LlamaIndex) use this as their core.

    Example:
        >>> optimizer = FitzOptimizer(
        ...     api_key="fitz_abc123...",
        ...     org_key="<64-char-hex-key>",
        ...     embedding_fn=embeddings.embed_query,
        ... )
        >>>
        >>> # Check cache before LLM call
        >>> result = optimizer.lookup(
        ...     query="What is the refund policy?",
        ...     query_embedding=query_vec,
        ...     chunk_ids=["chunk_1", "chunk_2"],
        ...     llm_model="gpt-4o",
        ... )
        >>>
        >>> if result.hit:
        ...     return result.answer  # Skip LLM call
        >>> else:
        ...     # Apply routing advice if available
        ...     model = result.routing_advice.get("recommended_model") or "gpt-4o"
    """

    def __init__(
        self,
        api_key: str,
        org_key: str,
        org_id: Optional[str] = None,
        base_url: str = "https://api.fitz-ai.cloud/v1",
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        engine_version: str = "0.5.0",
        collection_version: str = "default",
        timeout: int = 30,
    ):
        """
        Initialize optimizer.

        Args:
            api_key: Fitz Cloud API key (fitz_xxx format)
            org_key: Encryption key (64-char hex, NEVER sent to server)
            org_id: Organization ID (auto-generated if not provided)
            base_url: Cloud API URL
            embedding_fn: Optional function to embed query text → 1536-dim vector
            engine_version: Version string for cache invalidation
            collection_version: Collection version for cache invalidation
            timeout: HTTP request timeout in seconds
        """
        self.org_id = org_id or self._generate_org_id(api_key)
        self.embedding_fn = embedding_fn
        self.engine_version = engine_version
        self.collection_version = collection_version

        config = CloudConfig(
            enabled=True,
            api_key=api_key,
            org_id=self.org_id,
            org_key=org_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.client = CloudClient(config, self.org_id)

        logger.info(
            "FitzOptimizer initialized",
            extra={"org_id": self.org_id[:8] + "...", "base_url": base_url},
        )

    def lookup(
        self,
        query: str,
        query_embedding: list[float],
        chunk_ids: list[str],
        llm_model: str,
        chunk_embeddings: Optional[list[list[float]]] = None,
        prompt_template: str = "default",
    ) -> OptimizationResult:
        """
        Look up cached answer.

        Args:
            query: Query text
            query_embedding: Query embedding (must be 1536-dim)
            chunk_ids: List of retrieved chunk IDs (for fingerprinting)
            llm_model: LLM model name (for version tracking)
            chunk_embeddings: Optional chunk embeddings for routing advice (Pro+ tiers)
            prompt_template: Prompt template version

        Returns:
            OptimizationResult with hit status, answer (if hit), or routing advice (if miss)
        """
        # Validate embedding dimension
        if len(query_embedding) != REQUIRED_EMBEDDING_DIM:
            logger.debug(
                f"Embedding dimension mismatch: {len(query_embedding)} != {REQUIRED_EMBEDDING_DIM}"
            )
            return OptimizationResult(hit=False)

        retrieval_fingerprint = compute_retrieval_fingerprint(chunk_ids)
        versions = CacheVersions(
            optimizer="1.0.0",
            engine=self.engine_version,
            collection=self.collection_version,
            llm_model=llm_model,
            prompt_template=prompt_template,
        )

        try:
            result = self.client.lookup_cache(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_fingerprint=retrieval_fingerprint,
                versions=versions,
                chunk_embeddings=chunk_embeddings,
            )

            if result.hit and result.answer:
                logger.info("Cache hit", extra={"query_len": len(query)})
                return OptimizationResult(
                    hit=True,
                    answer=result.answer.text,
                    sources=[
                        {
                            "source_id": p.source_id,
                            "excerpt": p.excerpt,
                            "metadata": p.metadata,
                        }
                        for p in result.answer.provenance
                    ],
                )

            # Cache miss - return routing advice if available
            routing = None
            if result.routing:
                routing = {
                    "recommended_model": result.routing.recommended_model,
                    "complexity": result.routing.complexity,
                    "dedup_chunks": result.routing.dedup_chunks,
                }

            logger.info("Cache miss", extra={"query_len": len(query)})
            return OptimizationResult(hit=False, routing_advice=routing)

        except Exception as e:
            logger.warning("Cache lookup failed", extra={"error": str(e)})
            return OptimizationResult(hit=False)

    def store(
        self,
        query: str,
        query_embedding: list[float],
        chunk_ids: list[str],
        llm_model: str,
        answer_text: str,
        sources: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
        prompt_template: str = "default",
    ) -> bool:
        """
        Store answer in cache.

        Args:
            query: Query text
            query_embedding: Query embedding (must be 1536-dim)
            chunk_ids: List of retrieved chunk IDs
            llm_model: LLM model name
            answer_text: Generated answer text
            sources: Optional source documents for provenance
            metadata: Optional metadata (tokens, latency, etc.)
            prompt_template: Prompt template version

        Returns:
            True if stored successfully
        """
        # Validate embedding dimension
        if len(query_embedding) != REQUIRED_EMBEDDING_DIM:
            logger.debug(
                f"Embedding dimension mismatch: {len(query_embedding)} != {REQUIRED_EMBEDDING_DIM}"
            )
            return False

        retrieval_fingerprint = compute_retrieval_fingerprint(chunk_ids)
        versions = CacheVersions(
            optimizer="1.0.0",
            engine=self.engine_version,
            collection=self.collection_version,
            llm_model=llm_model,
            prompt_template=prompt_template,
        )

        # Build Answer object with provenance
        provenance = []
        if sources:
            for src in sources:
                provenance.append(
                    Provenance(
                        source_id=src.get("source_id", src.get("id", "unknown")),
                        excerpt=src.get("excerpt", src.get("content", "")[:500]),
                        metadata=src.get("metadata", {}),
                    )
                )

        answer = Answer(
            text=answer_text,
            provenance=provenance,
            mode=None,
            metadata=metadata or {},
        )

        try:
            stored = self.client.store_cache(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_fingerprint=retrieval_fingerprint,
                versions=versions,
                answer=answer,
                metadata=metadata,
            )

            if stored:
                logger.info("Cache stored", extra={"query_len": len(query)})
            return stored

        except Exception as e:
            logger.warning("Cache store failed", extra={"error": str(e)})
            return False

    def embed_query(self, query: str) -> Optional[list[float]]:
        """
        Embed query text using configured embedding function.

        Args:
            query: Query text

        Returns:
            Query embedding or None if embedding_fn not configured
        """
        if self.embedding_fn is None:
            logger.warning("No embedding_fn configured")
            return None

        try:
            embedding = self.embedding_fn(query)
            if len(embedding) != REQUIRED_EMBEDDING_DIM:
                logger.warning(
                    f"Embedding dimension mismatch: {len(embedding)} != {REQUIRED_EMBEDDING_DIM}"
                )
                return None
            return embedding
        except Exception as e:
            logger.warning("Embedding failed", extra={"error": str(e)})
            return None

    def close(self):
        """Close cloud client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _generate_org_id(api_key: str) -> str:
        """Generate org_id from API key prefix."""
        # Use API key prefix to generate deterministic UUID
        prefix = api_key[:16] if len(api_key) >= 16 else api_key
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"fitz.{prefix}"))
