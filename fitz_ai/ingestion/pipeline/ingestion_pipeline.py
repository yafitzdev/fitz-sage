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
from fitz_ai.tabular.models import create_schema_chunk_for_stored_table
from fitz_ai.tabular.parser import can_parse as is_table_file
from fitz_ai.tabular.parser import parse_csv
from fitz_ai.tabular.store import PostgresTableStore
from fitz_ai.vector_db.writer import VectorDBWriter

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk
    from fitz_ai.ingestion.enrichment.pipeline import EnrichmentPipeline
    from fitz_ai.tabular.store.base import TableStore

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
        table_store: "TableStore | None" = None,
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

        # Table storage (defaults to PostgreSQL via pgvector)
        self.table_store = table_store or PostgresTableStore(config.collection)

    def run(self, source: str) -> int:
        logger.info(f"{PIPELINE} Starting ingestion pipeline")

        # 1. Discover files
        scan_result = self.scanner.scan(source)
        if not scan_result.files:
            logger.info(f"{PIPELINE} No files found in {source}")
            return 0

        # 2. Parse and chunk documents
        all_chunks: List = []
        table_count = 0

        for file_info in scan_result.files:
            try:
                file_path = Path(file_info.path)

                # Check if file is a table file (CSV, TSV)
                if is_table_file(file_path):
                    chunk = self._process_table_file(file_path)
                    if chunk:
                        all_chunks.append(chunk)
                        table_count += 1
                    continue

                # Regular document processing
                source_file = SourceFile(
                    uri=file_path.as_uri(),
                    local_path=file_path,
                    metadata={},
                )

                # Parse file → ParsedDocument
                parsed_doc = self.parser_router.parse(source_file)
                if not parsed_doc.full_text.strip():
                    continue

                # Chunk document → Chunks
                ext = file_path.suffix or ".txt"
                chunker = self.chunking_router.get_chunker(ext)
                chunks = chunker.chunk(parsed_doc)
                if chunks:
                    all_chunks.extend(chunks)

            except Exception as e:
                logger.warning(f"{PIPELINE} Failed to process {file_info.path}: {e}")

        if table_count > 0:
            logger.info(f"{PIPELINE} Processed {table_count} table files")

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

        # Embed and store chunks (with contextual embeddings)
        vectors = [self.embedder.embed(self._get_embedding_text(c)) for c in all_chunks]
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

    def _get_embedding_text(self, chunk: "Chunk") -> str:
        """
        Get text for embedding with contextual prefix.

        Prepends the chunk summary (if available) to the content for contextual
        embeddings. This helps disambiguate pronouns and references by providing
        document context in the embedding.

        See: Anthropic's "Contextual Retrieval" approach.
        """
        summary = chunk.metadata.get("summary", "")
        if summary:
            return f"{summary}\n\n{chunk.content}"
        return chunk.content

    def _process_table_file(self, file_path: Path) -> "Chunk | None":
        """
        Process a table file (CSV, TSV) into TableStore and create schema chunk.

        Args:
            file_path: Path to the table file.

        Returns:
            Schema chunk for the table, or None on failure.
        """

        try:
            # Parse CSV file
            parsed = parse_csv(file_path)

            # Store in TableStore (SQLite or Qdrant)
            table_hash = self.table_store.store(
                table_id=parsed.table_id,
                columns=parsed.columns,
                rows=parsed.rows,
                source_file=str(file_path),
            )

            # Create lightweight schema chunk (no embedded table_data)
            chunk = create_schema_chunk_for_stored_table(
                table_id=parsed.table_id,
                columns=parsed.columns,
                row_count=parsed.row_count,
                source_file=str(file_path),
                table_hash=table_hash,
                sample_rows=parsed.rows[:3],
            )

            # Register table chunk ID for direct retrieval at query time
            from fitz_ai.tabular.registry import add_table_id

            add_table_id(self.collection, chunk.id)

            logger.info(
                f"{PIPELINE} Stored table {file_path.name}: "
                f"{len(parsed.columns)} cols, {parsed.row_count} rows"
            )

            return chunk

        except Exception as e:
            logger.warning(f"{PIPELINE} Failed to process table {file_path}: {e}")
            return None
