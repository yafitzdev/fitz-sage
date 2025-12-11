from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from fitz_ingest.exceptions.vector import IngestionVectorError
from fitz_ingest.exceptions.config import IngestionConfigError

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

    Raises:
        IngestionVectorError  – Qdrant communication failures.
        IngestionConfigError  – bad parameters or distance type.
    """

    logger.debug(f"{VECTOR_DB} Checking existence of collection '{name}'")

    # -------------------------------------------------------
    # 1. Fetch existing collections
    # -------------------------------------------------------
    try:
        existing = client.get_collections()
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed fetching Qdrant collections: {e}")
        raise IngestionVectorError("Failed to fetch list of Qdrant collections") from e

    # -------------------------------------------------------
    # 2. Check if collection already exists
    # -------------------------------------------------------
    for col in existing.collections:
        if col.name == name:
            logger.debug(f"{VECTOR_DB} Collection '{name}' already exists")
            return

    # -------------------------------------------------------
    # 3. Determine distance metric
    # -------------------------------------------------------
    try:
        dist = Distance.COSINE if distance.lower() == "cosine" else Distance.DOT
    except Exception as e:
        logger.error(f"{VECTOR_DB} Invalid distance value '{distance}': {e}")
        raise IngestionConfigError(f"Invalid distance metric: {distance}") from e

    logger.info(f"{VECTOR_DB} Creating new Qdrant collection '{name}'")

    # -------------------------------------------------------
    # 4. Create new collection
    # -------------------------------------------------------
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed creating collection '{name}': {e}")
        raise IngestionVectorError(
            f"Failed to create Qdrant collection '{name}'"
        ) from e
