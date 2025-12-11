"""
Generic ingestion engine for fitz-ingest.

Handles:
- Chunking
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

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import INGEST

from fitz_rag.exceptions.retriever import EmbeddingError, VectorSearchError
from fitz_rag.exceptions.config import ConfigError


logger = get_logger(__name__)


class IngestionEngine:
    """
    Ingest files into Qdrant using:
      - Chunker
      - Optional embedding client
      - Central validation
      - Unified logging
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector_size: int,
        embedder: Optional[Any] = None,
        distance: str = "cosine",
        validator: Optional[IngestionValidator] = None,
    ) -> None:
        self.client = client
        self.collection = collection
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
        chunker,
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

        # Qdrant ensure collection
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

        # Process files
        for file in files:
            if file.is_file():
                logger.info(f"{INGEST} Ingesting file: {file}")
                try:
                    self.ingest_file(chunker, file)
                except Exception as e:
                    logger.error(f"{INGEST} Error ingesting file {file}: {e}")
                    raise

    # ---------------------------------------------------------
    # Ingest a single file
    # ---------------------------------------------------------
    def ingest_file(self, chunker, file_path: str | Path) -> None:
        logger.debug(f"{INGEST} Chunking file: {file_path}")

        try:
            chunks = chunker.chunk_file(str(file_path))
        except Exception as e:
            logger.error(f"{INGEST} Chunker failed for file {file_path}: {e}")
            raise ConfigError(f"Chunker failed on file: {file_path}") from e

        if not chunks:
            logger.info(f"{INGEST} No chunks extracted from '{file_path}', skipping.")
            return

        # Validation BEFORE embedding / Qdrant
        try:
            self.validator.validate_chunks(chunks, file_path)
            logger.debug(f"{INGEST} Validation passed for '{file_path}' ({len(chunks)} chunks)")
        except IngestionValidationError as e:
            logger.error(f"{INGEST} Validation failed for '{file_path}': {e}")
            raise ConfigError(
                f"Ingestion validation failed for file '{file_path}': {e}"
            ) from e

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

        # Upsert into Qdrant — cast to Any to avoid PyCharm warning
        points_any: Any = points

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
