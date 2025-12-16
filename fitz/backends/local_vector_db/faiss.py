# fitz/backends/local_vector_db/faiss.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np

from fitz.core.models.chunk import Chunk
from fitz.core.vector_db.base import SearchResult, VectorDBPlugin
from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import VECTOR_DB
from .config import LocalVectorDBConfig

logger = get_logger(__name__)


class FaissLocalVectorDB:
    """
    Local FAISS-backed VectorDB plugin.

    Implements the canonical VectorDBPlugin contract.
    """

    plugin_name = "local-faiss"
    plugin_type = "vector_db"

    def __init__(self, *, dim: int, config: LocalVectorDBConfig):
        self._dim = dim
        self._config = config

        self._base_path = Path(config.path)
        self._index_path = self._base_path / "index.faiss"
        self._meta_path = self._base_path / "payloads.npy"

        self._base_path.mkdir(parents=True, exist_ok=True)

        self._index = faiss.IndexFlatL2(dim)

        # payloads are aligned with FAISS vector order
        self._payloads: list[dict] = []
        self._ids: list[str] = []

        if self._config.persist and self._index_path.exists():
            self._load()

        logger.info(
            f"{VECTOR_DB} Local FAISS initialized (dim={dim}, path={self._base_path})"
        )

    # ------------------------------------------------------------------ public

    def add(self, chunks: Iterable[Chunk]) -> None:
        vectors = []

        for chunk in chunks:
            embedding = getattr(chunk, "embedding", None)
            if embedding is None:
                continue

            vectors.append(embedding)
            self._ids.append(chunk.id)
            self._payloads.append(chunk.metadata or {})

        if not vectors:
            return

        matrix = np.asarray(vectors, dtype="float32")
        self._index.add(matrix)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[SearchResult]:
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

    def persist(self) -> None:
        if not self._config.persist:
            return

        faiss.write_index(self._index, str(self._index_path))
        np.save(
            self._meta_path,
            {"ids": self._ids, "payloads": self._payloads},
            allow_pickle=True,
        )

        logger.info(f"{VECTOR_DB} Local FAISS index persisted")

    def count(self) -> int:
        return self._index.ntotal

    # ----------------------------------------------------------------- private

    def _load(self) -> None:
        self._index = faiss.read_index(str(self._index_path))
        data = np.load(self._meta_path, allow_pickle=True).item()

        self._ids = list(data["ids"])
        self._payloads = list(data["payloads"])

        logger.info(
            f"{VECTOR_DB} Loaded FAISS index with {self._index.ntotal} vectors"
        )
