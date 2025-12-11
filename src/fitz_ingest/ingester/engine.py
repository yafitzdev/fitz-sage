"""
Generic ingestion engine for fitz-ingest.

Handles:
- Chunking via ChunkingEngine (plugin-based)
- Optional embedding
- Validation
- Upsert into Qdrant
- Logging
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Any

from qdrant_client import QdrantClient

from fitz_ingest.vector_db.qdrant_utils import ensure_collection
from fitz_ingest.ingester.validation import (
    IngestionValidator,
    IngestionValidationError,
)
from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.base import Chunk

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import INGEST

from fitz_rag.exceptions.retriever import EmbeddingError, VectorSearchError
from fitz_rag.exceptions.config import ConfigError


logger = get_logger(__name__)


class IngestionEngine:
    """
    Ingest files into Qdrant using:
      - ChunkingEngine (plugin-based chunkers)
      - Optional embedding client
      - Validation
      - Unified logging
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector_size: int,
        chunker_engine: ChunkingEngine,
        embedder: Optional[Any] = None,
        distance: str = "cosine",
        validator: Optional[IngestionValidator] = None,
    ) -> None:

        self.client = client
        self.collection = collection
        self.chunker_engine = chunker_engine
        self.embedder = embedder
        self.vector_size = vector_size
        self.distance = distance
        self.validator = validator or IngestionValidator()

        logger.info(f"{INGEST} IngestionEngine initialized (collection='{collection}')")

    # ---------------------------------------------------------
    # Ingest a directory tree
    # ---------------------------------------------------------
    def ingest_path(
        self,
        path: str | Path,
        glob_pattern: str = "**/*",
    ) -> None:

        logger.info(f"{INGEST} Starting ingestion for directory: {path}")

        try:
            path = Path(path)
            files = list(path.glob(glob_pattern))
        except Exception as e:
            logger.error(f"{INGEST} Failed resolving path '{path}': {e}")
            raise ConfigError(f"Invalid ingestion path: {path}") from e

        if not files:
            logger.info(f"{INGEST} No files found — ingestion finished.")
            return

        # Ensure Qdrant collection exists
        try:
            ensure_collection(
                client=self.client,
                name=self.collection,
                vector_size=self.vector_size,
            )
            logger.info(f"{INGEST} Collection '{self.collection}' is ready.")
        except Exception as e:
            logger.error(f"{INGEST} Failed ensuring Qdrant collection: {e}")
            raise VectorSearchError(
                f"Failed ensuring collection '{self.collection}'"
            ) from e

        # Process all files
        for file in files:
            if file.is_file():
                logger.info(f"{INGEST} Ingesting file: {file}")
                try:
                    self.ingest_file(file)
                except Exception as e:
                    logger.error(f"{INGEST} Error ingesting file {file}: {e}")
                    raise

    # ---------------------------------------------------------
    # Ingest a single file
    # ---------------------------------------------------------
    def ingest_file(self, file_path: str | Path) -> None:
        """
        Chunk the file using ChunkingEngine, then optionally embed, then upsert.
        """

        logger.debug(f"{INGEST} Chunking file: {file_path}")

        try:
            chunks: list[Chunk] = self.chunker_engine.chunk_file(file_path)
        except Exception as e:
            logger.error(f"{INGEST} Chunker engine failed for file {file_path}: {e}")
            raise

        if not chunks:
            logger.info(f"{INGEST} No chunks extracted from '{file_path}', skipping.")
            return

        # Validate BEFORE embedding / Qdrant
        try:
            self.validator.validate_chunks(chunks, file_path)
            logger.debug(f"{INGEST} Validation passed for '{file_path}' ({len(chunks)} chunks)")
        except IngestionValidationError as e:
            logger.error(f"{INGEST} Validation failed for '{file_path}': {e}")
            raise ConfigError(
                f"Ingestion validation failed for file '{file_path}': {e}"
            ) from e

        points = []

        # ---------------------------------------------------------
        # Build Qdrant point structures
        # ---------------------------------------------------------
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
                if self.embedder is not None:
                    logger.debug(f"{INGEST} Embedded a chunk from '{file_path}'")
            except Exception as e:
                logger.error(f"{INGEST} Embedding failed for chunk in '{file_path}': {e}")
                raise EmbeddingError(
                    f"Failed embedding chunk from: {file_path}"
                ) from e

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

        # ---------------------------------------------------------
        # Upsert into Qdrant
        # ---------------------------------------------------------
        points_any: Any = points  # to fix PyCharm false positive

        try:
            self.client.upsert(
                self.collection,
                points_any,
            )
            logger.info(f"{INGEST} Upserted {len(points)} chunks from '{file_path}'")
        except Exception as e:
            logger.error(f"{INGEST} Upsert failed for '{file_path}': {e}")
            raise VectorSearchError(
                f"Failed upserting points into '{self.collection}'"
            ) from e
