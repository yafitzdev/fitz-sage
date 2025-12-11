"""
Qdrant client wrapper for fitz_rag.

Provides:
- create_qdrant_client()
- ensure_collection()

Now includes structured exceptions:
- ConfigError
- VectorSearchError
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from fitz_rag.config import get_config
from fitz_rag.exceptions.config import ConfigError
from fitz_rag.exceptions.retriever import VectorSearchError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import VECTOR_DB

logger = get_logger(__name__)

# Load unified config
try:
    _cfg = get_config()
except Exception as e:
    raise ConfigError("Failed to load unified fitz_rag configuration") from e


ENV_QDRANT_URL = "FITZ_RAG_QDRANT_URL"
ENV_QDRANT_API_KEY = "FITZ_RAG_QDRANT_API_KEY"
ENV_QDRANT_TIMEOUT = "FITZ_RAG_QDRANT_TIMEOUT"


def create_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
) -> QdrantClient:
    """
    Build a QdrantClient from:
    - explicit args
    - environment variables
    - unified config (fitz_rag.config)
    - default localhost fallback

    Raises:
        ConfigError – invalid config
        VectorSearchError – Qdrant initialization failure
    """

    logger.debug(f"{VECTOR_DB} Creating Qdrant client…")

    try:
        qcfg = _cfg.get("qdrant", {})
    except Exception as e:
        logger.error(f"{VECTOR_DB} Invalid config structure: {e}")
        raise ConfigError("Invalid config structure for qdrant section") from e

    # Build base config URL
    try:
        cfg_url = (
            f"{'https' if qcfg.get('https', False) else 'http'}://"
            f"{qcfg.get('host', 'localhost')}:{qcfg.get('port', 6333)}"
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed constructing Qdrant URL: {e}")
        raise ConfigError("Failed constructing Qdrant URL from config") from e

    # URL selection
    chosen_url = (
        url
        or os.getenv(ENV_QDRANT_URL)
        or cfg_url
    )

    # API key selection
    chosen_api_key = (
        api_key
        or os.getenv(ENV_QDRANT_API_KEY)
        or None
    )

    # Timeout handling
    try:
        if timeout is None:
            env_val = os.getenv(ENV_QDRANT_TIMEOUT)
            if env_val:
                timeout = int(env_val)
            else:
                timeout = int(qcfg.get("timeout", 30))
    except Exception as e:
        logger.error(f"{VECTOR_DB} Invalid timeout value: {e}")
        raise ConfigError("Invalid Qdrant timeout value") from e

    logger.debug(
        f"{VECTOR_DB} Initializing Qdrant client (url={chosen_url}, timeout={timeout})"
    )

    # Initialize QdrantClient
    try:
        return QdrantClient(
            url=chosen_url,
            api_key=chosen_api_key,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed initializing Qdrant client: {e}")
        raise VectorSearchError(f"Failed initializing Qdrant client for URL: {chosen_url}") from e


def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Ensure a Qdrant collection exists.

    Raises:
        VectorSearchError – Qdrant calls failed
        ConfigError – bad parameters
    """

    logger.debug(f"{VECTOR_DB} Ensuring collection '{name}' exists")

    # Fetch existing collections
    try:
        collections = client.get_collections().collections
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed fetching collections: {e}")
        raise VectorSearchError("Failed fetching collections from Qdrant") from e

    # Check existence
    try:
        names = {c.name for c in collections}
    except Exception as e:
        logger.error(f"{VECTOR_DB} Invalid Qdrant collections structure: {e}")
        raise VectorSearchError("Invalid structure in Qdrant collections response") from e

    if name in names:
        logger.debug(f"{VECTOR_DB} Collection '{name}' already exists")
        return

    # Create new collection
    logger.info(f"{VECTOR_DB} Creating new Qdrant collection '{name}'")

    try:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=distance,
            ),
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed creating collection '{name}': {e}")
        raise VectorSearchError(f"Failed creating Qdrant collection '{name}'") from e
