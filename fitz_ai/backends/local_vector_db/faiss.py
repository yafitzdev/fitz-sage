# fitz_ai/backends/local_vector_db/faiss.py
"""
Local FAISS-backed Vector Database.

Design principles:
- Per-collection storage: Each collection has its own index with independent dimensions
- Lazy initialization: dimension detected on first upsert
- Uses FitzPaths for storage location
- Implements standard VectorDBPlugin contract
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import VECTOR_DB
from fitz_ai.vector_db.base import SearchResult

logger = get_logger(__name__)


@dataclass
class _FaissRecord:
    """Simple record object for FAISS scroll results."""

    id: str
    payload: Dict[str, Any]


@dataclass
class _CollectionData:
    """Data for a single collection."""

    dim: Optional[int] = None
    index: Optional[Any] = None  # faiss.IndexFlatL2
    payloads: List[Dict] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)


class FaissLocalVectorDB:
    """
    Local FAISS-backed VectorDB plugin with per-collection storage.

    Key design:
    - Each collection has its own FAISS index with independent dimensions
    - Collections stored in separate subdirectories
    - Lazy initialization - dimension detected on first upsert

    Usage:
        db = FaissLocalVectorDB()

        # Each collection can have different dimensions
        db.upsert("collection_a", [{"id": "1", "vector": [0.1] * 768, ...}])  # 768-dim
        db.upsert("collection_b", [{"id": "1", "vector": [0.1] * 1536, ...}])  # 1536-dim
    """

    plugin_name = "local-faiss"
    plugin_type = "vector_db"

    def __init__(
        self,
        *,
        path: Optional[str | Path] = None,
        persist: bool = True,
    ):
        """
        Initialize FAISS vector database.

        Args:
            path: Storage path. If None, uses FitzPaths.vector_db()
            persist: Whether to persist index to disk (default: True)
        """
        # Import here to allow graceful failure if faiss not installed
        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss is required for local-faiss plugin. Install with: pip install faiss-cpu"
            )

        # Resolve path
        if path is None:
            from fitz_ai.core.paths import FitzPaths

            self._base_path = FitzPaths.vector_db()
        else:
            self._base_path = Path(path)

        self._persist = persist

        # Per-collection data storage
        self._collections: Dict[str, _CollectionData] = {}

        # Discover existing collections
        self._discover_collections()

        logger.info(f"{VECTOR_DB} Local FAISS initialized (path={self._base_path})")

    def _get_collection_path(self, collection: str) -> Path:
        """Get storage path for a collection."""
        return self._base_path / collection

    def _discover_collections(self) -> None:
        """Discover existing collections from disk."""
        if not self._base_path.exists():
            return

        for coll_path in self._base_path.iterdir():
            if coll_path.is_dir() and (coll_path / "index.faiss").exists():
                collection_name = coll_path.name
                self._load_collection(collection_name)

    def _load_collection(self, collection: str) -> bool:
        """Load a collection from disk. Returns True if successful."""
        coll_path = self._get_collection_path(collection)
        index_path = coll_path / "index.faiss"

        if not index_path.exists():
            return False

        try:
            coll_data = _CollectionData()

            # Load dimension
            dim_path = coll_path / "dim.txt"
            if dim_path.exists():
                coll_data.dim = int(dim_path.read_text().strip())

            # Load index
            coll_data.index = self._faiss.read_index(str(index_path))

            # Infer dimension from index if not stored
            if coll_data.dim is None and coll_data.index.ntotal > 0:
                coll_data.dim = coll_data.index.d

            # Load metadata
            meta_path = coll_path / "payloads.npy"
            if meta_path.exists():
                coll_data.payloads = list(np.load(str(meta_path), allow_pickle=True))

            # Load IDs
            ids_path = coll_path / "ids.npy"
            if ids_path.exists():
                coll_data.ids = list(np.load(str(ids_path), allow_pickle=True))
            else:
                # Fallback: generate generic IDs
                coll_data.ids = [f"id_{i}" for i in range(coll_data.index.ntotal)]

            self._collections[collection] = coll_data

            logger.info(
                f"{VECTOR_DB} Loaded collection '{collection}' "
                f"({coll_data.index.ntotal} vectors, dim={coll_data.dim})"
            )
            return True

        except Exception as e:
            logger.warning(f"{VECTOR_DB} Failed to load collection '{collection}': {e}")
            return False

    def _get_or_create_collection(self, collection: str) -> _CollectionData:
        """Get collection data, creating if it doesn't exist."""
        if collection not in self._collections:
            self._collections[collection] = _CollectionData()
        return self._collections[collection]

    def _ensure_initialized(self, collection: str, dim: int) -> None:
        """Initialize collection index with given dimension if not already initialized."""
        coll = self._get_or_create_collection(collection)
        coll_path = self._get_collection_path(collection)

        if coll.index is not None:
            # Verify dimension matches
            if coll.dim != dim:
                raise ValueError(
                    f"Dimension mismatch: collection '{collection}' has dim={coll.dim}, "
                    f"but got vectors with dim={dim}. "
                    f"Delete {coll_path} to reset."
                )
            return

        # Initialize new index for this collection
        coll.dim = dim
        coll.index = self._faiss.IndexFlatL2(dim)
        coll.payloads = []
        coll.ids = []

        # Ensure directory exists
        coll_path.mkdir(parents=True, exist_ok=True)

        # Save dimension for future loads
        (coll_path / "dim.txt").write_text(str(dim))

        logger.info(f"{VECTOR_DB} Initialized collection '{collection}' (dim={dim})")

    def _save_collection(self, collection: str) -> None:
        """Save a collection's index and metadata to disk."""
        if collection not in self._collections:
            return

        coll = self._collections[collection]
        if coll.index is None:
            return

        coll_path = self._get_collection_path(collection)
        coll_path.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(coll.index, str(coll_path / "index.faiss"))
        np.save(str(coll_path / "payloads.npy"), coll.payloads, allow_pickle=True)
        np.save(str(coll_path / "ids.npy"), coll.ids, allow_pickle=True)
        (coll_path / "dim.txt").write_text(str(coll.dim))

        logger.debug(f"{VECTOR_DB} Persisted collection '{collection}' to {coll_path}")

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _matches_filter(self, payload: Dict[str, Any], filter_cond: Dict[str, Any]) -> bool:
        """Check if payload matches Qdrant-style filter conditions."""
        if not filter_cond:
            return True

        # Handle "must" conditions (AND)
        if "must" in filter_cond:
            for cond in filter_cond["must"]:
                if not self._matches_filter(payload, cond):
                    return False
            return True

        # Handle "should" conditions (OR)
        if "should" in filter_cond:
            for cond in filter_cond["should"]:
                if self._matches_filter(payload, cond):
                    return True
            return False

        # Handle direct field conditions
        key = filter_cond.get("key")
        if key is None:
            return True

        # Support nested key lookup via dot notation
        value = payload
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

        # Match condition (exact equality)
        if "match" in filter_cond:
            match_value = filter_cond["match"].get("value")
            return value == match_value

        # Range condition
        if "range" in filter_cond:
            range_cond = filter_cond["range"]
            if value is None:
                return False
            if "gte" in range_cond and value < range_cond["gte"]:
                return False
            if "gt" in range_cond and value <= range_cond["gt"]:
                return False
            if "lte" in range_cond and value > range_cond["lte"]:
                return False
            if "lt" in range_cond and value >= range_cond["lt"]:
                return False
            return True

        return True

    # =========================================================================
    # Public API - Standard VectorDBPlugin Contract
    # =========================================================================

    def upsert(
        self,
        collection: str,
        points: List[Dict[str, Any]],
        defer_persist: bool = False,
    ) -> None:
        """
        Upsert points into a collection.

        Args:
            collection: Collection name
            points: List of points with 'id', 'vector', and 'payload' keys
            defer_persist: If True, don't persist to disk (call flush() later)
        """
        if not points:
            return

        # Auto-detect dimension from first vector
        first_vector = points[0]["vector"]
        dim = len(first_vector)
        self._ensure_initialized(collection, dim)

        coll = self._collections[collection]

        for point in points:
            point_id = str(point["id"])
            vector = np.asarray([point["vector"]], dtype="float32")
            payload = point.get("payload", {})

            # Check if ID exists (for update)
            if point_id in coll.ids:
                idx = coll.ids.index(point_id)
                # FAISS doesn't support in-place vector updates
                # Just update payload
                coll.payloads[idx] = payload
            else:
                # Add new point
                coll.index.add(vector)
                coll.ids.append(point_id)
                coll.payloads.append(payload)

        # Auto-persist if enabled and not deferred
        if self._persist and not defer_persist:
            self._save_collection(collection)

    def flush(self) -> None:
        """Explicitly persist all collections to disk."""
        if self._persist:
            for collection in self._collections:
                self._save_collection(collection)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        with_payload: bool = True,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors in a collection.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Maximum number of results
            with_payload: Whether to include payload in results
            query_filter: Optional metadata filter conditions

        Returns:
            List of SearchResult objects
        """
        if collection_name not in self._collections:
            return []

        coll = self._collections[collection_name]
        if coll.index is None or coll.index.ntotal == 0:
            return []

        # Search more than limit to account for filtering
        search_limit = min(limit * 3, coll.index.ntotal)

        query = np.asarray([query_vector], dtype="float32")
        distances, indices = coll.index.search(query, search_limit)

        results: List[SearchResult] = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            payload = coll.payloads[idx] if idx < len(coll.payloads) else {}

            # Apply metadata filter if provided
            if query_filter and not self._matches_filter(payload, query_filter):
                continue

            # Convert L2 distance to similarity score
            score = 1.0 / (1.0 + float(distance))

            result_payload = dict(payload) if with_payload else {}

            results.append(
                SearchResult(
                    id=coll.ids[idx],
                    score=score,
                    payload=result_payload,
                )
            )

            if len(results) >= limit:
                break

        return results

    def retrieve(
        self,
        collection_name: str,
        ids: List[str],
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by their IDs."""
        if not ids or collection_name not in self._collections:
            return []

        coll = self._collections[collection_name]
        id_set = set(ids)
        results = []

        for i, point_id in enumerate(coll.ids):
            if point_id in id_set:
                payload = coll.payloads[i] if i < len(coll.payloads) else {}
                result_payload = dict(payload) if with_payload else {}

                results.append(
                    {
                        "id": point_id,
                        "payload": result_payload,
                    }
                )

        return results

    def list_collections(self) -> List[str]:
        """List all collections."""
        return sorted(self._collections.keys())

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if collection not in self._collections:
            return {
                "points_count": 0,
                "vectors_count": 0,
                "status": "not_found",
                "vector_size": None,
            }

        coll = self._collections[collection]
        count = coll.index.ntotal if coll.index else 0

        return {
            "points_count": count,
            "vectors_count": count,
            "status": "ready",
            "vector_size": coll.dim,
        }

    def scroll(
        self,
        collection: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[_FaissRecord], Optional[int]]:
        """Scroll through records in a collection."""
        if collection not in self._collections:
            return [], None

        coll = self._collections[collection]

        matching = []
        for i, payload in enumerate(coll.payloads):
            record_id = coll.ids[i] if i < len(coll.ids) else f"id_{i}"
            matching.append(_FaissRecord(record_id, dict(payload)))

        # Apply offset and limit
        start = offset
        end = offset + limit
        batch = matching[start:end]

        next_offset = end if end < len(matching) else None

        return batch, next_offset

    def scroll_with_vectors(
        self,
        collection: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """Scroll through records in a collection, including vectors."""
        if collection not in self._collections:
            return [], None

        coll = self._collections[collection]
        if coll.index is None:
            return [], None

        matching = []
        for i, payload in enumerate(coll.payloads):
            record_id = coll.ids[i] if i < len(coll.ids) else f"id_{i}"

            # Reconstruct vector from FAISS index
            vector = coll.index.reconstruct(i).tolist()

            matching.append(
                {
                    "id": record_id,
                    "payload": dict(payload),
                    "vector": vector,
                }
            )

        # Apply offset and limit
        start = offset
        end = offset + limit
        batch = matching[start:end]

        next_offset = end if end < len(matching) else None

        return batch, next_offset

    def count(self, collection: Optional[str] = None) -> int:
        """Return the number of vectors."""
        if collection is None:
            return sum(c.index.ntotal for c in self._collections.values() if c.index)

        if collection not in self._collections:
            return 0

        coll = self._collections[collection]
        return coll.index.ntotal if coll.index else 0

    def delete_collection(self, collection: str) -> int:
        """Delete a collection entirely."""
        if collection not in self._collections:
            return 0

        coll = self._collections[collection]
        deleted = coll.index.ntotal if coll.index else 0

        # Remove from memory
        del self._collections[collection]

        # Remove from disk
        coll_path = self._get_collection_path(collection)
        if coll_path.exists():
            import shutil

            shutil.rmtree(coll_path)

        logger.info(f"{VECTOR_DB} Deleted collection '{collection}' ({deleted} vectors)")
        return deleted


__all__ = ["FaissLocalVectorDB"]
