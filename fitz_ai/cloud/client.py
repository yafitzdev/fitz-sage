# fitz_ai/cloud/client.py
"""Client for Fitz Cloud Query-Time RAG Optimizer API."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from fitz_ai.cloud.cache_key import CacheVersions, compute_cache_key
from fitz_ai.cloud.config import CloudConfig
from fitz_ai.cloud.crypto import CacheEncryption, EncryptedBlob
from fitz_ai.core import Answer
from fitz_ai.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingAdvice:
    """Model routing advice from Fitz Cloud."""

    recommended_model: Optional[str]
    complexity: str  # simple | moderate | complex
    dedup_chunks: Optional[list[int]]


@dataclass
class CacheLookupResult:
    """Result from cache lookup."""

    hit: bool
    answer: Optional[Answer] = None
    routing: Optional[RoutingAdvice] = None


@dataclass
class TierFeatures:
    """Tier features from Fitz Cloud."""

    tier: str
    features: dict[str, Any]
    limits: dict[str, Any]


class CloudClient:
    """
    Client for Fitz Cloud Query-Time RAG Optimizer.

    Provides:
    - Encrypted cache lookup/store
    - Model routing recommendations
    - Tier feature checking
    """

    def __init__(self, config: CloudConfig, org_id: str):
        """
        Initialize cloud client.

        Args:
            config: Cloud configuration
            org_id: Organization ID (UUID string)
        """
        self.config = config
        self.org_id = org_id
        self.encryption = CacheEncryption(config.org_key) if config.org_key else None

        # HTTP client
        self.client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    def lookup_cache(
        self,
        query_text: str,
        query_embedding: list[float],
        retrieval_fingerprint: str,
        versions: CacheVersions,
    ) -> CacheLookupResult:
        """
        Look up cached answer.

        Args:
            query_text: Query string
            query_embedding: Query embedding vector
            retrieval_fingerprint: Hash of chunk IDs
            versions: Version info

        Returns:
            CacheLookupResult with hit status, answer (if hit), or routing advice (if miss)
        """
        if not self.config.enabled or not self.encryption:
            return CacheLookupResult(hit=False)

        cache_key = compute_cache_key(query_text, retrieval_fingerprint, versions)

        payload = {
            "cache_key": cache_key,
            "query_embedding": query_embedding,
            "retrieval_fingerprint": retrieval_fingerprint,
            "versions": {
                "optimizer": versions.optimizer,
                "engine": versions.engine,
                "collection": versions.collection,
                "llm_model": versions.llm_model,
                "prompt_template": versions.prompt_template,
            },
        }

        try:
            response = self.client.post("/cache/lookup", json=payload)
            response.raise_for_status()
            data = response.json()

            if data["hit"]:
                # Decrypt blob
                blob = EncryptedBlob(
                    ciphertext=base64.b64decode(data["blob"]),
                    timestamp=data["timestamp"],
                    org_id=self.org_id,
                )
                plaintext = self.encryption.decrypt(blob)
                answer_data = json.loads(plaintext)

                # Reconstruct Answer
                answer = Answer(**answer_data)

                logger.info("Cache hit", extra={"cache_key": cache_key[:16]})
                return CacheLookupResult(hit=True, answer=answer)
            else:
                # Cache miss - return routing advice
                routing_data = data.get("routing")
                routing = (
                    RoutingAdvice(
                        recommended_model=routing_data.get("recommended_model"),
                        complexity=routing_data.get("complexity", "moderate"),
                        dedup_chunks=routing_data.get("dedup_chunks"),
                    )
                    if routing_data
                    else None
                )

                logger.info("Cache miss", extra={"cache_key": cache_key[:16]})
                return CacheLookupResult(hit=False, routing=routing)

        except Exception as e:
            logger.warning("Cache lookup failed", extra={"error": str(e)})
            return CacheLookupResult(hit=False)

    def store_cache(
        self,
        query_text: str,
        query_embedding: list[float],
        retrieval_fingerprint: str,
        versions: CacheVersions,
        answer: Answer,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Store answer in cache.

        Args:
            query_text: Query string
            query_embedding: Query embedding vector
            retrieval_fingerprint: Hash of chunk IDs
            versions: Version info
            answer: Answer to cache
            metadata: Optional metadata (model_used, tokens, latency)

        Returns:
            True if stored successfully
        """
        if not self.config.enabled or not self.encryption:
            return False

        cache_key = compute_cache_key(query_text, retrieval_fingerprint, versions)

        # Serialize answer
        answer_data = {
            "text": answer.text,
            "provenance": [
                {
                    "source_id": p.source_id,
                    "excerpt": p.excerpt,
                    "metadata": p.metadata,
                }
                for p in answer.provenance
            ],
            "mode": answer.mode.value if answer.mode else None,
            "metadata": answer.metadata,
        }
        plaintext = json.dumps(answer_data)

        # Encrypt
        blob = self.encryption.encrypt(plaintext, self.org_id)

        payload = {
            "cache_key": cache_key,
            "query_embedding": query_embedding,
            "encrypted_blob": base64.b64encode(blob.ciphertext).decode(),
            "timestamp": blob.timestamp,
            "versions": {
                "optimizer": versions.optimizer,
                "engine": versions.engine,
                "collection": versions.collection,
                "llm_model": versions.llm_model,
                "prompt_template": versions.prompt_template,
            },
            "metadata": metadata or {},
        }

        try:
            response = self.client.post("/cache/store", json=payload)
            response.raise_for_status()
            data = response.json()

            if data["stored"]:
                logger.info("Cache stored", extra={"cache_key": cache_key[:16]})
                return True
            else:
                logger.warning("Cache store failed", extra={"cache_key": cache_key[:16]})
                return False

        except Exception as e:
            logger.warning("Cache store request failed", extra={"error": str(e)})
            return False

    def invalidate_cache(self, reason: str, scope: str) -> int:
        """
        Invalidate cache entries.

        Args:
            reason: "key_rotation", "collection_update", or "manual"
            scope: "all", "collection:{version}", or "before:{timestamp}"

        Returns:
            Number of entries deleted
        """
        if not self.config.enabled:
            return 0

        payload = {"reason": reason, "scope": scope}

        try:
            response = self.client.delete("/cache", json=payload)
            response.raise_for_status()
            data = response.json()

            deleted = data.get("deleted_count", 0)
            logger.info("Cache invalidated", extra={"deleted": deleted, "scope": scope})
            return deleted

        except Exception as e:
            logger.warning("Cache invalidation failed", extra={"error": str(e)})
            return 0

    def get_features(self) -> Optional[TierFeatures]:
        """
        Get tier features and limits.

        Returns:
            TierFeatures or None if request fails
        """
        if not self.config.enabled:
            return None

        try:
            response = self.client.get("/features")
            response.raise_for_status()
            data = response.json()

            return TierFeatures(
                tier=data["tier"],
                features=data["features"],
                limits=data["limits"],
            )

        except Exception as e:
            logger.warning("Failed to fetch features", extra={"error": str(e)})
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
