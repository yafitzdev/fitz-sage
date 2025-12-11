from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.ingester.validation import IngestionValidator
from fitz_ingest.vector_db.qdrant_utils import ensure_collection


class IngestionEngine:
    """
    High-level ingestion coordinator.

    Responsibilities:
    - Chunk raw files via a ChunkingEngine.
    - Validate resulting chunks (text length, metadata, etc.).
    - Embed chunks (optional embedder).
    - Upsert into a Qdrant-like vector store client.
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        vector_size: int,
        chunker_engine: ChunkingEngine,
        embedder: Optional[Any] = None,
        validator: Optional[IngestionValidator] = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.vector_size = int(vector_size)
        self.chunker_engine = chunker_engine
        self.embedder = embedder
        self.validator = validator or IngestionValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_file(self, path: str | Path) -> None:
        """
        Ingest a single file:
        - chunk
        - validate
        - ensure collection exists
        - upsert points into Qdrant-style client

        Parameters
        ----------
        path:
            Path to the text file to ingest.
        """
        p = Path(path)

        # 1) Chunk the file into chunk dicts
        chunks = self.chunker_engine.chunk_file(p)

        if not chunks:
            # nothing to ingest
            return

        # 2) Validate chunks (raises on error)
        self.validator.validate_chunks(chunks, str(p))

        # 3) Ensure collection exists
        ensure_collection(self.client, self.collection, self.vector_size)

        # 4) Build points and upsert
        points = list(self._build_points(chunks))

        if not points:
            return

        self.client.upsert(self.collection, points)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_points(self, chunks: Iterable[Any]) -> Iterable[Dict[str, Any]]:
        """
        Convert normalized chunks into Qdrant point dicts:

            {
                "id": <int>,
                "vector": <list[float]>,
                "payload": {
                    "text": <chunk_text>,
                    "file": <file_path>,
                    ... other metadata ...
                }
            }
        """
        for idx, ch in enumerate(chunks):
            text, metadata = self._extract_text_and_metadata(ch)

            # Derive a 'file' field from metadata["source_file"] or fallback.
            file_val = None
            if isinstance(metadata, dict):
                file_val = metadata.get("source_file") or metadata.get("file")

            payload: Dict[str, Any] = dict(metadata or {})
            payload.setdefault("text", text)
            if file_val is not None:
                payload.setdefault("file", file_val)

            vector = self._embed_text(text)

            yield {
                "id": idx,
                "vector": vector,
                "payload": payload,
            }

    def _extract_text_and_metadata(self, chunk: Any) -> tuple[str, Dict[str, Any]]:
        """
        Normalize chunk-like objects into (text, metadata) pair.

        Supports:
        - dict with 'text' and 'metadata'
        - legacy objects with .text and .metadata
        """
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            return str(text), metadata

        # Fallback to attribute-based access
        text = getattr(chunk, "text", "")
        metadata = getattr(chunk, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        return str(text), metadata

    def _embed_text(self, text: str) -> List[float]:
        """
        Embed text using the configured embedder.

        If embedder is None, returns a dummy zero vector so tests can run
        without a real embedding provider.
        """
        if self.embedder is None:
            # Simple deterministic zero vector for tests / dry runs.
            return [0.0] * self.vector_size

        vec = self.embedder.embed(text)
        # Ensure we always get a list of floats
        return list(vec)
