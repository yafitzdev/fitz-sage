# fitz/vector_db/plugins/qdrant.py
"""
Smart Qdrant vector database plugin.

Features:
- Auto-creates collections when they don't exist
- Auto-detects vector dimensions from first upsert
- Auto-detects named vs unnamed vectors
- Converts string IDs to UUIDs automatically
- Provides helpful error messages with fix suggestions
- Environment variable configuration
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# Support both import paths (fitz.vector_db and fitz.core.vector_db)
try:
    from fitz.vector_db.base import SearchResult, VectorDBPlugin
except ImportError:
    from fitz.vector_db.base import SearchResult, VectorDBPlugin

logger = logging.getLogger(__name__)


# =============================================================================
# Errors with helpful messages
# =============================================================================


class QdrantConnectionError(Exception):
    """Raised when unable to connect to Qdrant."""

    def __init__(self, host: str, port: int, original_error: Exception):
        self.host = host
        self.port = port
        self.original_error = original_error

        message = f"""
❌ Cannot connect to Qdrant at {host}:{port}

Possible fixes:
  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant
  2. Check if Qdrant is running: curl http://{host}:{port}/collections
  3. Set correct host/port via environment variables:
       export QDRANT_HOST=your-host
       export QDRANT_PORT=6333

Original error: {original_error}
"""
        super().__init__(message)


class QdrantCollectionError(Exception):
    """Raised for collection-related errors with helpful messages."""

    pass


# =============================================================================
# Helper functions
# =============================================================================


def _string_to_uuid(s: str) -> str:
    """Convert any string to a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def _get_env_or_default(env_var: str, default: Any) -> Any:
    """Get environment variable with type coercion."""
    value = os.getenv(env_var)
    if value is None:
        return default

    # Coerce to same type as default
    if isinstance(default, int):
        return int(value)
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    return value


# =============================================================================
# Smart Qdrant Plugin
# =============================================================================


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    """
    Smart Qdrant vector database plugin with auto-configuration.

    Environment variables:
        QDRANT_HOST: Qdrant server hostname (default: localhost)
        QDRANT_PORT: Qdrant server port (default: 6333)

    Features:
        - Auto-creates collections on first upsert
        - Auto-detects vector dimensions
        - Handles both named and unnamed vectors
        - Converts string IDs to UUIDs
        - Provides helpful error messages

    Usage:
        # Basic usage (uses env vars or defaults)
        db = QdrantVectorDB()

        # Custom configuration
        db = QdrantVectorDB(host="192.168.1.100", port=6333)
    """

    plugin_name: str = "qdrant"
    plugin_type: str = "vector_db"

    # Connection settings (from env or defaults)
    host: str = field(default_factory=lambda: _get_env_or_default("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: _get_env_or_default("QDRANT_PORT", 6333))

    # Internal state
    _client: Any = field(init=False, repr=False, default=None)
    _collection_cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize Qdrant client with connection validation."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise RuntimeError("qdrant-client is required. Install with: pip install qdrant-client")

        try:
            self._client = QdrantClient(host=self.host, port=self.port, timeout=10)
            # Test connection
            self._client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            raise QdrantConnectionError(self.host, self.port, e)

    # =========================================================================
    # Collection Management
    # =========================================================================

    def _get_collection_info(self, collection_name: str) -> Optional[dict]:
        """Get collection info, returns None if doesn't exist."""
        if collection_name in self._collection_cache:
            return self._collection_cache[collection_name]

        try:
            info = self._client.get_collection(collection_name)

            # Parse vector config
            vectors_config = info.config.params.vectors

            # Detect if named or unnamed vectors
            if hasattr(vectors_config, "size"):
                # Unnamed vectors (simple config)
                config = {
                    "exists": True,
                    "named_vectors": False,
                    "vector_name": None,
                    "size": vectors_config.size,
                    "distance": str(vectors_config.distance),
                }
            else:
                # Named vectors (dict-like config)
                # Get the first (or only) vector name
                vector_names = (
                    list(vectors_config.keys()) if hasattr(vectors_config, "keys") else []
                )
                if vector_names:
                    first_name = vector_names[0]
                    first_config = vectors_config[first_name]
                    config = {
                        "exists": True,
                        "named_vectors": True,
                        "vector_name": first_name,
                        "vector_names": vector_names,
                        "size": first_config.size,
                        "distance": str(first_config.distance),
                    }
                else:
                    config = {"exists": True, "named_vectors": False, "vector_name": None}

            self._collection_cache[collection_name] = config
            return config

        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "doesn't exist" in error_msg:
                return None
            raise

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        info = self._get_collection_info(collection_name)
        return info is not None and info.get("exists", False)

    def _create_collection(self, collection_name: str, vector_size: int) -> None:
        """Create a collection with the given vector size."""
        from qdrant_client.http.models import Distance, VectorParams

        logger.info(f"Auto-creating collection '{collection_name}' with dimension {vector_size}")

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        # Invalidate cache
        self._collection_cache.pop(collection_name, None)

        logger.info(f"✓ Created collection '{collection_name}'")

    def _ensure_collection(self, collection_name: str, vector_size: int) -> dict:
        """Ensure collection exists, create if needed. Returns collection info."""
        info = self._get_collection_info(collection_name)

        if info is None:
            # Collection doesn't exist - create it
            self._create_collection(collection_name, vector_size)
            info = self._get_collection_info(collection_name)

        # Validate vector dimensions match
        if info and info.get("size") and info["size"] != vector_size:
            raise QdrantCollectionError(
                f"""
❌ Vector dimension mismatch for collection '{collection_name}'

Collection expects: {info['size']} dimensions
You're sending:     {vector_size} dimensions

Possible fixes:
  1. Use a different collection name
  2. Delete and recreate the collection:
       curl -X DELETE "http://{self.host}:{self.port}/collections/{collection_name}"
  3. Use an embedding model with {info['size']} dimensions
"""
            )

        return info or {}

    # =========================================================================
    # Core Operations
    # =========================================================================

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[SearchResult]:
        """
        Search for similar vectors in a collection.

        Automatically handles:
        - Named vs unnamed vectors
        - Missing collection errors with helpful messages
        """
        info = self._get_collection_info(collection_name)

        if info is None:
            raise QdrantCollectionError(
                f"""
❌ Collection '{collection_name}' not found

To fix:
  1. Ingest documents first:
       fitz-ingest run ./your_docs --collection {collection_name}

  2. Or check available collections:
       curl http://{self.host}:{self.port}/collections

Available at: http://{self.host}:{self.port}
"""
            )

        try:
            # Use query_points (qdrant-client >= 1.7)
            if info.get("named_vectors") and info.get("vector_name"):
                # Named vectors - use 'using' parameter
                result = self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using=info["vector_name"],
                    limit=limit,
                    with_payload=with_payload,
                )
            else:
                # Unnamed vectors
                result = self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=with_payload,
                )

            return [
                SearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload or {},
                )
                for point in result.points
            ]

        except Exception as e:
            error_msg = str(e).lower()

            if "vector name" in error_msg:
                # Named vector issue
                raise QdrantCollectionError(
                    f"""
❌ Vector name mismatch for collection '{collection_name}'

The collection uses named vectors but the search didn't specify the correct name.
Collection vector names: {info.get('vector_names', ['unknown'])}

This usually means the collection was created with a different configuration.
Try recreating the collection:
  curl -X DELETE "http://{self.host}:{self.port}/collections/{collection_name}"
  fitz-ingest run ./your_docs --collection {collection_name}
"""
                )
            raise

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """
        Upsert points into a collection.

        Automatically handles:
        - Collection creation if missing
        - String ID to UUID conversion
        - Named vs unnamed vectors
        """
        if not points:
            logger.warning("upsert called with empty points list")
            return

        from qdrant_client.http.models import PointStruct

        # Get vector dimension from first point
        first_vector = points[0].get("vector", [])
        vector_size = len(first_vector)

        # Ensure collection exists (auto-create if needed)
        info = self._ensure_collection(collection, vector_size)

        # Build Qdrant points
        q_points = []
        for p in points:
            original_id = p["id"]

            # Convert string ID to UUID (Qdrant requires UUID or int)
            if isinstance(original_id, str):
                qdrant_id = _string_to_uuid(original_id)
            else:
                qdrant_id = original_id

            # Store original ID in payload for reference
            payload = p.get("payload", {}) or {}
            payload["_original_id"] = original_id

            # Handle vector format (named vs unnamed)
            vector = p["vector"]
            if info.get("named_vectors") and info.get("vector_name"):
                vector = {info["vector_name"]: vector}

            q_points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Perform upsert
        self._client.upsert(collection_name=collection, points=q_points)
        logger.info(f"Upserted {len(q_points)} points to collection '{collection}'")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_collections(self) -> list[str]:
        """List all collections."""
        result = self._client.get_collections()
        return [c.name for c in result.collections]

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection. Returns True if deleted, False if didn't exist."""
        try:
            self._client.delete_collection(collection_name)
            self._collection_cache.pop(collection_name, None)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                return False
            raise

    def get_collection_stats(self, collection_name: str) -> dict:
        """Get statistics about a collection."""
        try:
            info = self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status),
            }
        except Exception as e:
            if "not found" in str(e).lower():
                return {"name": collection_name, "exists": False}
            raise
