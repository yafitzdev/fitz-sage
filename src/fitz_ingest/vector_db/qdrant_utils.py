"""
Qdrant utility functions for fitz_ingest.

This module provides:
- ensure_collection(): Create a Qdrant collection if it does not exist.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    distance: str = "cosine",
) -> None:
    """
    Ensures that the Qdrant collection exists.
    If it already exists, do nothing.
    """
    existing = client.get_collections()

    for col in existing.collections:
        if col.name == name:
            # Collection exists → do nothing
            return

    # Create new collection
    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=dist),
    )
