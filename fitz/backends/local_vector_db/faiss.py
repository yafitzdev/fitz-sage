# fitz/backends/local_vector_db/faiss.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import faiss
import numpy as np

from fitz.logging.logger import get_logger
from fitz.logging.tags import VECTOR_DB
from fitz.engines.classic_rag.models.chunk import Chunk
from fitz.vector_db.base import SearchResult

from .config import LocalVectorDBConfig

logger = get_logger(__name__)


class FaissLocalVectorDB:
    """
    Local FAISS-backed VectorDB plugin.

    Implements the canonical VectorDBPlugin contract with upsert support.

    Can be initialized in two ways:

    1. Direct initialization (for programmatic use):
        db = FaissLocalVectorDB(dim=768, config=LocalVectorDBConfig())

    2. From config kwargs (for YAML-based config):
        db = FaissLocalVectorDB(dim=768, path=".fitz/vector_db", persist=True)
    """

    plugin_name = "local-faiss"
    plugin_type = "vector_db"

    def __init__(
            self,
            *,
            dim: int,
            config: Optional[LocalVectorDBConfig] = None,
            # These kwargs allow config-based initialization from YAML
            path: Optional[str] = None,
            persist: Optional[bool] = None,
    ):
        """
        Initialize the FAISS vector database.

        Args:
            dim: Vector dimension (required)
            config: LocalVectorDBConfig object (optional)
            path: Storage path (alternative to config)
            persist: Whether to persist to disk (alternative to config)
        """
        self._dim = dim

        # Build config from kwargs if not provided directly
        if config is None:
            config_kwargs = {}
            if path is not None:
                config_kwargs["path"] = Path(path)
            if persist is not None:
                config_kwargs["persist"] = persist
            config = LocalVectorDBConfig(**config_kwargs)

        self._config = config

        self._base_path = Path(config.path)
        self._index_path = self._base_path / "index.faiss"
        self._meta_path = self._base_path / "payloads.npy"
        self._ids_path = self._base_path / "ids.npy"

        self._base_path.mkdir(parents=True, exist_ok=True)

        self._index = faiss.IndexFlatL2(dim)

        # payloads and IDs are aligned with FAISS vector order
        self._payloads: list[dict] = []
        self._ids: list[str] = []

        if self._config.persist and self._index_path.exists():
            self._load()

        logger.info(f"{VECTOR_DB} Local FAISS initialized (dim={dim}, path={self._base_path})")

    # ------------------------------------------------------------------ public

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """
        Upsert points into the vector database.

        Args:
            collection: Collection name (ignored for FAISS, single collection)
            points: List of points with 'id', 'vector', and 'payload' keys
        """
        for point in points:
            point_id = str(point["id"])
            vector = np.asarray([point["vector"]], dtype="float32")
            payload = point.get("payload", {})

            # Check if ID exists (for update)
            if point_id in self._ids:
                idx = self._ids.index(point_id)
                # FAISS doesn't support in-place updates, so we just update payload
                self._payloads[idx] = payload
                # Note: vector update would require index rebuild
            else:
                # Add new point
                self._index.add(vector)
                self._ids.append(point_id)
                self._payloads.append(payload)

        # Auto-persist if enabled
        if self._config.persist:
            self.persist()

    def add(self, chunks: Iterable[Chunk]) -> None:
        """
        Add chunks with pre-computed embeddings to the index.

        This is the legacy interface that expects chunks to have
        an 'embedding' attribute attached at runtime.
        """
        for chunk in chunks:
            embedding = getattr(chunk, "embedding", None)
            if embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding attached")

            vec = np.asarray([embedding], dtype="float32")
            self._index.add(vec)
            self._ids.append(chunk.id)
            self._payloads.append({
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                **(chunk.metadata or {}),
            })

    def search(
            self,
            collection_name: str,
            query_vector: list[float],
            limit: int,
            with_payload: bool = True,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection name (ignored for FAISS)
            query_vector: Query vector
            limit: Maximum number of results
            with_payload: Whether to include payload in results

        Returns:
            List of SearchResult objects
        """
        if self._index.ntotal == 0:
            return []

        query = np.asarray([query_vector], dtype="float32")
        distances, indices = self._index.search(query, limit)

        results: list[SearchResult] = []

        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            payload = self._payloads[idx] if with_payload else {}

            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    payload=payload,
                )
            )

        return results

    def count(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal

    def persist(self) -> None:
        """Save index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        np.save(str(self._meta_path), self._payloads, allow_pickle=True)
        np.save(str(self._ids_path), self._ids, allow_pickle=True)
        logger.info(f"{VECTOR_DB} Persisted FAISS index to {self._base_path}")

    def _load(self) -> None:
        """Load index and metadata from disk."""
        self._index = faiss.read_index(str(self._index_path))
        self._payloads = list(np.load(str(self._meta_path), allow_pickle=True))

        # Handle IDs file (may not exist in older versions)
        if self._ids_path.exists():
            self._ids = list(np.load(str(self._ids_path), allow_pickle=True))
        else:
            # Fallback: generate generic IDs
            self._ids = [f"id_{i}" for i in range(self._index.ntotal)]

        logger.info(
            f"{VECTOR_DB} Loaded FAISS index from {self._base_path} "
            f"({self._index.ntotal} vectors)"
        )