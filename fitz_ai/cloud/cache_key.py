# fitz_ai/cloud/cache_key.py
"""Cache key computation for Query-Time RAG Optimizer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class CacheVersions:
    """Version information for cache key computation."""

    optimizer: str  # Fitz Cloud optimizer version
    engine: str  # fitz-ai engine version
    collection: str  # Collection version hash
    llm_model: str  # LLM model name
    prompt_template: str = "default"  # Prompt template version


def compute_cache_key(
    query_text: str,
    retrieval_fingerprint: str,
    versions: CacheVersions,
) -> str:
    """
    Compute deterministic cache key.

    Two queries hit the same cache entry if:
    - Query text is identical (after normalization)
    - Retrieved chunks are identical (same fingerprint)
    - All versions match (optimizer, engine, collection, llm_model, prompt)

    Args:
        query_text: Normalized query string
        retrieval_fingerprint: Hash of chunk IDs retrieved
        versions: Version info

    Returns:
        Cache key (hex string)
    """
    # Normalize query text (lowercase, strip whitespace)
    normalized_query = query_text.lower().strip()

    # Concatenate all components
    key_input = ":".join(
        [
            normalized_query,
            retrieval_fingerprint,
            versions.optimizer,
            versions.engine,
            versions.collection,
            versions.llm_model,
            versions.prompt_template,
        ]
    )

    # SHA-256 hash
    return hashlib.sha256(key_input.encode()).hexdigest()


def compute_retrieval_fingerprint(chunk_ids: list[str]) -> str:
    """
    Compute fingerprint of retrieved chunks.

    Args:
        chunk_ids: List of chunk IDs in retrieval order

    Returns:
        Fingerprint (hex string)
    """
    # Sort for determinism (order shouldn't affect cache key)
    sorted_ids = sorted(chunk_ids)
    fingerprint_input = ":".join(sorted_ids)
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()
