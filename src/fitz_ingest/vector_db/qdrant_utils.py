from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from fitz_rag.exceptions.retriever import VectorSearchError


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

    # --- Query existing collections ---
    try:
        existing = client.get_collections()
    except Exception as e:
        raise VectorSearchError("Failed to fetch list of Qdrant collections") from e

    for col in existing.collections:
        if col.name == name:
            return  # exists → nothing to do

    # --- Create new collection ---
    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT

    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )
    except Exception as e:
        raise VectorSearchError(f"Failed to create Qdrant collection '{name}'") from e
