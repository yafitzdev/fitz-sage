"""
Generic ingestion engine for fitz-ingest.

Handles:
- Chunking
- Optional embedding
- Upsert into Qdrant
- File iteration
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Any

from qdrant_client import QdrantClient

from fitz_ingest.vector_db.qdrant_utils import ensure_collection

from fitz_rag.exceptions.retriever import EmbeddingError, VectorSearchError
from fitz_rag.exceptions.config import ConfigError


class IngestionEngine:
    """
    Ingest files into Qdrant using:
      - Chunker
      - Optional embedding client
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector_size: int,
        embedder: Optional[Any] = None,
        distance: str = "cosine",
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.vector_size = vector_size
        self.distance = distance

    # ---------------------------------------------------------
    # Ingest a directory tree
    # ---------------------------------------------------------
    def ingest_path(
        self,
        chunker,
        path: str | Path,
        glob_pattern: str = "**/*",
    ) -> None:
        try:
            path = Path(path)
            files = list(path.glob(glob_pattern))
        except Exception as e:
            raise ConfigError(f"Invalid ingestion path: {path}") from e

        if not files:
            return

        try:
            ensure_collection(
                client=self.client,
                name=self.collection,
                vector_size=self.vector_size,
            )
        except Exception as e:
            raise VectorSearchError(f"Failed ensuring collection '{self.collection}'") from e

        for file in files:
            if file.is_file():
                try:
                    self.ingest_file(chunker, file)
                except Exception as e:
                    raise VectorSearchError(f"Failed ingesting file: {file}") from e

    # ---------------------------------------------------------
    # Ingest a single file
    # ---------------------------------------------------------
    def ingest_file(self, chunker, file_path: str | Path) -> None:
        try:
            chunks = chunker.chunk_file(str(file_path))
        except Exception as e:
            raise ConfigError(f"Chunker failed on file: {file_path}") from e

        if not chunks:
            return

        points = []

        for chunk in chunks:
            text = chunk.text
            meta = dict(chunk.metadata)
            meta["file"] = str(file_path)

            # Embedding step (optional)
            try:
                vector = (
                    self.embedder.embed(text)
                    if self.embedder is not None
                    else None
                )
            except Exception as e:
                raise EmbeddingError(f"Failed embedding chunk from: {file_path}") from e

            point_id = str(uuid.uuid4())

            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "text": text,
                        **meta,
                    },
                }
            )

        # Upsert into Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
        except Exception as e:
            raise VectorSearchError(f"Failed upserting points into '{self.collection}'") from e
