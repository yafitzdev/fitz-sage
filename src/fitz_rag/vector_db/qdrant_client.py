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

    try:
        qcfg = _cfg.get("qdrant", {})
    except Exception as e:
        raise ConfigError("Invalid config structure for qdrant section") from e

    try:
        cfg_url = (
            f"{'https' if qcfg.get('https', False) else 'http'}://"
            f"{qcfg.get('host', 'localhost')}:{qcfg.get('port', 6333)}"
        )
    except Exception as e:
        raise ConfigError("Failed constructing Qdrant URL from config") from e

    # URL selection: explicit > ENV > CFG > default
    chosen_url = (
        url
        or os.getenv(ENV_QDRANT_URL)
        or cfg_url
    )

    # API key: explicit > ENV > None
    chosen_api_key = (
        api_key
        or os.getenv(ENV_QDRANT_API_KEY)
        or None
    )

    # Timeout: explicit > ENV > CFG > default
    try:
        if timeout is None:
            env_val = os.getenv(ENV_QDRANT_TIMEOUT)
            if env_val:
                timeout = int(env_val)
            else:
                timeout = int(qcfg.get("timeout", 30))
    except Exception as e:
        raise ConfigError("Invalid Qdrant timeout value") from e

    try:
        return QdrantClient(
            url=chosen_url,
            api_key=chosen_api_key,
            timeout=timeout,
        )
    except Exception as e:
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

    # Fetch existing collections
    try:
        collections = client.get_collections().collections
    except Exception as e:
        raise VectorSearchError("Failed fetching collections from Qdrant") from e

    # Check existence
    try:
        names = {c.name for c in collections}
    except Exception as e:
        raise VectorSearchError("Invalid structure in Qdrant collections response") from e

    if name in names:
        return

    # Create new collection
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
        raise VectorSearchError(f"Failed creating Qdrant collection '{name}'") from e
