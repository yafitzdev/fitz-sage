# fitz_ai/ingestion/pipeline/ingestion_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from fitz_ai.engines.fitz_rag.config import IngestConfig
from fitz_ai.ingestion.chunking.router import ChunkingRouter
from fitz_ai.ingestion.diff.scanner import FileScanner
from fitz_ai.ingestion.parser import ParserRouter
from fitz_ai.ingestion.source.base import SourceFile
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE
from fitz_ai.vector_db.writer import VectorDBWriter

if TYPE_CHECKING:
    from fitz_ai.ingestion.enrichment.pipeline import EnrichmentPipeline

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline using three-layer architecture:

        source
          -> file discovery (FileScanner)
          -> parsing (ParserRouter → ParsedDocument)
          -> chunking (ChunkingRouter → Chunks)
          -> enrichment (optional: summaries + artifacts)
          -> embedding (vectors)   [must be injected upstream]
          -> vector db writer

    Architecture:
    - Uses Source → Parser → Chunker three-layer architecture.
    - Provider-specific embedding selection is NOT allowed here.
      Vectors must be provided by an injected embedder.
    - Enrichment is optional and adds summaries to chunks + generates artifacts.
    """

    def __init__(
        self,
        *,
        config: IngestConfig,
        writer: VectorDBWriter,
        embedder: object,
        enrichment: "EnrichmentPipeline | None" = None,
    ) -> None:
        self.config = config
        self.writer = writer
        self.embedder = embedder
        self.enrichment = enrichment

        # Initialize three-layer components
        self.parser_router = ParserRouter()
        self.chunking_router = ChunkingRouter.from_config(config.chunker)
        self.scanner = FileScanner()

        self.collection = config.collection

    def run(self, source: str) -> int:
        logger.info(f"{PIPELINE} Starting ingestion pipeline")

        # 1. Discover files
        scan_result = self.scanner.scan(source)
        if not scan_result.files:
            logger.info(f"{PIPELINE} No files found in {source}")
            return 0

        # 2. Parse and chunk documents
        all_chunks: List = []
        for file_info in scan_result.files:
            try:
                source_file = SourceFile(
                    uri=Path(file_info.path).as_uri(),
                    local_path=Path(file_info.path),
                    metadata={},
                )

                # Parse file → ParsedDocument
                parsed_doc = self.parser_router.parse(source_file)
                if not parsed_doc.full_text.strip():
                    continue

                # Chunk document → Chunks
                ext = Path(file_info.path).suffix or ".txt"
                chunker = self.chunking_router.get_chunker(ext)
                chunks = chunker.chunk(parsed_doc)
                if chunks:
                    all_chunks.extend(chunks)

            except Exception as e:
                logger.warning(f"{PIPELINE} Failed to process {file_info.path}: {e}")

        if not all_chunks:
            logger.info(f"{PIPELINE} No chunks to process")
            return 0

        logger.info(f"{PIPELINE} Chunked {len(all_chunks)} chunks from source")

        # Enrichment stage (if enabled)
        artifacts = []
        if self.enrichment and self.enrichment.is_enabled:
            logger.info(f"{PIPELINE} Running enrichment pipeline")
            result = self.enrichment.enrich(all_chunks)
            all_chunks = result.chunks
            artifacts = result.artifacts
            logger.info(
                f"{PIPELINE} Enrichment complete: "
                f"{len(all_chunks)} chunks, {len(artifacts)} artifacts"
            )

        # Embed and store chunks
        vectors = [self.embedder.embed(c.content) for c in all_chunks]
        self.writer.upsert(
            collection=self.collection,
            chunks=all_chunks,
            vectors=vectors,
        )
        total_written = len(all_chunks)

        # Embed and store artifacts separately
        if artifacts:
            artifact_chunks = [a.to_chunk() for a in artifacts]
            artifact_vectors = [self.embedder.embed(c.content) for c in artifact_chunks]
            self.writer.upsert(
                collection=self.collection,
                chunks=artifact_chunks,
                vectors=artifact_vectors,
            )
            total_written += len(artifact_chunks)
            logger.info(f"{PIPELINE} Stored {len(artifact_chunks)} artifacts")

        logger.info(f"{PIPELINE} Ingestion finished, written={total_written}")
        return total_written
