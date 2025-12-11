from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from fitz_rag.exceptions.retriever import VectorSearchError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import VECTOR_DB

logger = get_logger(__name__)


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

    logger.debug(f"{VECTOR_DB} Checking existence of collection '{name}'")

    # --- Query existing collections ---
    try:
        existing = client.get_collections()
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed to fetch list of Qdrant collections: {e}")
        raise VectorSearchError("Failed to fetch list of Qdrant collections") from e

    for col in existing.collections:
        if col.name == name:
            logger.debug(f"{VECTOR_DB} Collection '{name}' already exists")
            return  # exists → nothing to do

    # --- Create new collection ---
    dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT

    logger.info(f"{VECTOR_DB} Creating new Qdrant collection '{name}'")

    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed to create Qdrant collection '{name}': {e}")
        raise VectorSearchError(f"Failed to create Qdrant collection '{name}'") from e
