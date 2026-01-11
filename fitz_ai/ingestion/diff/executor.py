# fitz_ai/ingestion/diff/executor.py
"""
Executor for incremental (diff) ingestion.

Orchestrates the full pipeline:
1. (Optional) Generate and ingest project artifacts
2. Scan files
3. Compute diff (detect content AND config changes)
4. Ingest new/changed files: Source → Parser → Chunker
5. (Optional) Enrich chunks with LLM-generated descriptions
6. Mark deletions
7. Update state
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from fitz_ai.ingestion.chunking.router import ChunkingRouter
from fitz_ai.ingestion.diff.differ import Differ, FileCandidate
from fitz_ai.ingestion.diff.scanner import FileScanner
from fitz_ai.ingestion.hashing import compute_chunk_id
from fitz_ai.ingestion.parser.router import ParserRouter
from fitz_ai.ingestion.source.base import SourceFile
from fitz_ai.ingestion.state.manager import IngestStateManager
from fitz_ai.tabular import TableExtractor
from fitz_ai.tabular.models import create_schema_chunk_for_stored_table
from fitz_ai.tabular.parser import can_parse as is_table_file
from fitz_ai.tabular.parser import get_sample_rows, parse_csv
from fitz_ai.tabular.store import SqliteTableStore

if TYPE_CHECKING:
    from fitz_ai.ingestion.enrichment.pipeline import EnrichmentPipeline
    from fitz_ai.tabular.store.base import TableStore

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorDBWriter(Protocol):
    """Protocol for writing to vector DB."""

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        """Upsert points into collection."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text."""

    def embed(self, text: str) -> List[float]:
        """Embed text and return vector."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts and return vectors."""
        ...


@dataclass
class IngestSummary:
    """Summary of an ingestion run."""

    scanned: int = 0
    ingested: int = 0
    skipped: int = 0
    marked_deleted: int = 0
    errors: int = 0
    rechunked: int = 0  # Files re-ingested due to config change
    enriched: int = 0  # Chunks enriched with descriptions
    enrichment_cached: int = 0  # Chunks with cached descriptions
    artifacts_generated: int = 0  # Number of artifacts generated
    hierarchy_summaries: int = 0  # Number of hierarchy summary chunks generated
    error_details: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def __str__(self) -> str:
        """Get summary string."""
        base = (
            f"scanned {self.scanned}, ingested {self.ingested}, "
            f"skipped {self.skipped}, marked_deleted {self.marked_deleted}, "
            f"errors {self.errors}"
        )
        if self.enriched > 0 or self.enrichment_cached > 0:
            base += f", enriched {self.enriched} (cached {self.enrichment_cached})"
        if self.artifacts_generated > 0:
            base += f", artifacts {self.artifacts_generated}"
        return base


class DiffIngestExecutor:
    """
    Executes the incremental ingestion pipeline.

    Uses the three-layer architecture:
    - ParserRouter: Routes files to appropriate parsers (SourceFile → ParsedDocument)
    - ChunkingRouter: Routes documents to type-specific chunkers (ParsedDocument → Chunks)

    Usage:
        executor = DiffIngestExecutor(
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser_router=parser_router,
            chunking_router=chunking_router,
            collection="my_collection",
            embedding_id="cohere:embed-english-v3.0",
            vector_db_id="qdrant",
        )

        summary = executor.run("/path/to/documents")
    """

    def __init__(
        self,
        *,
        state_manager: IngestStateManager,
        vector_db_writer: VectorDBWriter,
        embedder: Embedder,
        parser_router: ParserRouter,
        chunking_router: ChunkingRouter,
        collection: str,
        embedding_id: str,
        vector_db_id: Optional[str] = None,
        enrichment_pipeline: Optional["EnrichmentPipeline"] = None,
        table_store: Optional["TableStore"] = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            state_manager: Manager for ingest state.
            vector_db_writer: Writer for upserting vectors.
            embedder: Embedder for creating vectors.
            parser_router: Router for file-type specific parsing.
            chunking_router: Router for file-type specific chunking.
            collection: Vector DB collection name.
            embedding_id: Current embedding configuration ID.
            vector_db_id: Vector DB plugin name (e.g., "qdrant", "local_faiss").
            enrichment_pipeline: Optional unified enrichment pipeline.
                                Handles both chunk summaries and project artifacts.
            table_store: Optional table storage backend for CSV/table files.
        """
        self._state = state_manager
        self._vdb_writer = vector_db_writer
        self._embedder = embedder
        self._parser_router = parser_router
        self._chunking_router = chunking_router
        self._collection = collection
        self._embedding_id = embedding_id
        self._vector_db_id = vector_db_id
        self._enrichment_pipeline = enrichment_pipeline
        self._table_store = table_store or SqliteTableStore(collection)
        self._summary: Optional[IngestSummary] = None

    @property
    def _enricher_id(self) -> Optional[str]:
        """Get the enricher ID if enrichment is enabled."""
        if self._enrichment_pipeline is None or not self._enrichment_pipeline.hierarchy_enrichment_enabled:
            return None
        return getattr(self._enrichment_pipeline, "_enricher_id", None)

    def run(
        self,
        source: str | Path,
        force: bool = False,
        on_progress: Optional[callable] = None,
        skip_artifacts: bool = False,
    ) -> IngestSummary:
        """
        Run the incremental ingestion pipeline.

        Args:
            source: Path to directory or file to ingest.
            force: If True, ingest everything regardless of state.
            on_progress: Optional callback(current, total, file_path) for progress updates.
            skip_artifacts: If True, skip artifact generation (caller handles separately).

        Returns:
            IngestSummary with results.
        """
        summary = IngestSummary()
        self._summary = summary  # Store for access in _ingest_file

        # 1. Scan files
        logger.info(f"Scanning {source}...")
        scanner = FileScanner()
        scan_result = scanner.scan(source)
        summary.scanned = scan_result.total_scanned

        for path, error in scan_result.errors:
            summary.errors += 1
            summary.error_details.append(f"Scan error: {path}: {error}")

        # 1.5 Generate and ingest artifacts (if pipeline provided and enabled)
        if (
            not skip_artifacts
            and self._enrichment_pipeline is not None
            and self._enrichment_pipeline.artifacts_enabled
        ):
            count, errors = self.ingest_artifacts()
            summary.artifacts_generated = count
            summary.errors += len(errors)
            summary.error_details.extend(errors)

        # 2. Compute diff (state + config aware)
        logger.info("Computing diff...")
        differ = Differ(
            state_reader=self._state.state,
            config_provider=self._chunking_router,
            parser_id_func=lambda ext: self._parser_router.get_parser_id(ext),
            embedding_id=self._embedding_id,
            vector_db_id=self._vector_db_id,
            collection=self._collection,
        )

        root_path = str(Path(source).resolve())
        diff = differ.compute_diff(scan_result.files, force=force, root=root_path)

        summary.skipped = len(diff.to_skip)

        # 3. Ingest files - three phases: prepare, summarize, embed
        summaries_enabled = (
            self._enrichment_pipeline and self._enrichment_pipeline.hierarchy_enrichment_enabled
        )
        enrichment_status = "enabled" if summaries_enabled else "disabled"
        logger.info(f"Ingesting {len(diff.to_ingest)} files (summaries: {enrichment_status})...")

        total_to_ingest = len(diff.to_ingest)

        # Phase 1: Prepare all files (parse, chunk) - NO enrichment yet
        all_prepared: List[tuple] = []  # (candidate, file_data)

        for i, candidate in enumerate(diff.to_ingest):
            if on_progress:
                on_progress(i, total_to_ingest, f"Preparing {Path(candidate.path).name}")

            try:
                file_data = self._prepare_file_no_enrich(candidate)
                if file_data is not None:
                    all_prepared.append((candidate, file_data))
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Prepare error: {candidate.path}: {e}")
                logger.warning(f"Failed to prepare {candidate.path}: {e}")

        # Collect all chunks for batch enrichment
        all_chunks: List = [
            chunk_data["chunk"]
            for _, file_data in all_prepared
            for chunk_data in file_data["chunk_data"]
        ]

        # Phase 1.5: Batch enrich ALL chunks at once (summary, keywords, entities)
        if (
            summaries_enabled
            and all_chunks
            and self._enrichment_pipeline._chunk_enricher is not None
        ):
            t0 = time.perf_counter()
            enrich_result = self._enrichment_pipeline._chunk_enricher.enrich(all_chunks)
            enrich_time = time.perf_counter() - t0
            # Count chunks that got summaries
            summary.enriched = sum(
                1 for c in enrich_result.chunks if c.metadata.get("summary")
            )
            logger.info(f"Enriched {len(all_chunks)} chunks in {enrich_time:.2f}s")

        # Build texts to embed (use summary if available, otherwise content)
        all_texts: List[str] = []
        for candidate, file_data in all_prepared:
            for chunk_data in file_data["chunk_data"]:
                chunk = chunk_data["chunk"]
                # Get summary from chunk metadata (set by ChunkEnricher)
                chunk_summary = chunk.metadata.get("summary")
                chunk_data["description"] = chunk_summary
                # Embed summary if available, otherwise content
                text_to_embed = chunk_summary if chunk_summary else chunk.content
                all_texts.append(text_to_embed)

        # Phase 1.75: Hierarchy enrichment (generates group + corpus summaries)
        hierarchy_chunks: List = []
        if (
            self._enrichment_pipeline
            and hasattr(self._enrichment_pipeline, "_hierarchy_enricher")
            and self._enrichment_pipeline._hierarchy_enricher is not None
            and all_chunks
        ):
            t0 = time.perf_counter()
            enriched_chunks = self._enrichment_pipeline._hierarchy_enricher.enrich(all_chunks)
            hierarchy_time = time.perf_counter() - t0

            # Extract only the new hierarchy summary chunks
            hierarchy_chunks = [
                c for c in enriched_chunks if c.metadata.get("is_hierarchy_summary", False)
            ]

            if hierarchy_chunks:
                logger.info(
                    f"Generated {len(hierarchy_chunks)} hierarchy summaries "
                    f"in {hierarchy_time:.2f}s"
                )
                # Add hierarchy chunk texts for embedding
                for hc in hierarchy_chunks:
                    all_texts.append(hc.content)

        # Phase 2: Batch embed ALL texts at once
        if all_texts:
            t0 = time.perf_counter()
            all_vectors = self._embedder.embed_batch(all_texts)
            embed_time = time.perf_counter() - t0
            logger.info(f"Embedded {len(all_texts)} chunks in {embed_time:.2f}s")
        else:
            all_vectors = []

        # Phase 3: Upsert to vector DB and update state
        vector_offset = 0
        for idx, (candidate, file_data) in enumerate(all_prepared):
            if on_progress:
                on_progress(idx + 1, len(all_prepared), f"Saving {Path(candidate.path).name}")

            try:
                num_chunks = len(file_data["chunk_data"])
                # Get vectors for this file
                file_vectors = all_vectors[vector_offset : vector_offset + num_chunks]
                vector_offset += num_chunks

                # Build and upsert points (defer persist to batch at end)
                self._upsert_file(candidate, file_data, file_vectors, defer_persist=True)
                summary.ingested += 1

                # Update state on success
                self._state.mark_active(
                    file_path=candidate.path,
                    root=candidate.root,
                    content_hash=candidate.content_hash,
                    ext=candidate.ext,
                    size_bytes=candidate.size_bytes,
                    mtime_epoch=candidate.mtime_epoch,
                    chunker_id=candidate.chunker_id,
                    parser_id=candidate.parser_id,
                    embedding_id=candidate.embedding_id,
                    enricher_id=self._enricher_id,
                    vector_db_id=self._vector_db_id,
                    collection=self._collection,
                )
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Upsert error: {candidate.path}: {e}")
                logger.warning(f"Failed to upsert {candidate.path}: {e}")

        # Phase 3.5: Upsert hierarchy summary chunks
        if hierarchy_chunks:
            try:
                # Hierarchy vectors are at the end of all_vectors
                hierarchy_vector_start = len(all_vectors) - len(hierarchy_chunks)
                hierarchy_vectors = all_vectors[hierarchy_vector_start:]

                hierarchy_points = []
                for hc, vec in zip(hierarchy_chunks, hierarchy_vectors):
                    hierarchy_points.append(
                        {
                            "id": hc.id,
                            "vector": vec,
                            "payload": {
                                "content": hc.content,
                                "doc_id": hc.doc_id,
                                "chunk_index": hc.chunk_index,
                                **hc.metadata,
                            },
                        }
                    )

                self._vdb_writer.upsert(self._collection, hierarchy_points, defer_persist=True)
                logger.info(f"Upserted {len(hierarchy_points)} hierarchy summary chunks")
                summary.hierarchy_summaries = len(hierarchy_chunks)
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Hierarchy upsert error: {e}")
                logger.warning(f"Failed to upsert hierarchy chunks: {e}")

        # Flush vector DB once after all upserts
        if hasattr(self._vdb_writer, "flush"):
            self._vdb_writer.flush()
        elif hasattr(self._vdb_writer, "_client") and hasattr(self._vdb_writer._client, "flush"):
            self._vdb_writer._client.flush()

        # Phase 4: Auto-detect vocabulary keywords from all ingested chunks
        if all_prepared:
            self._detect_vocabulary(all_prepared)

        # Final progress update
        if on_progress and total_to_ingest > 0:
            on_progress(total_to_ingest, total_to_ingest, "Done")

        # 4. Update state for skipped files (they're still active)
        for candidate in diff.to_skip:
            self._state.mark_active(
                file_path=candidate.path,
                root=candidate.root,
                content_hash=candidate.content_hash,
                ext=candidate.ext,
                size_bytes=candidate.size_bytes,
                mtime_epoch=candidate.mtime_epoch,
                chunker_id=candidate.chunker_id,
                parser_id=candidate.parser_id,
                embedding_id=candidate.embedding_id,
                enricher_id=self._enricher_id,
                vector_db_id=self._vector_db_id,
                collection=self._collection,
            )

        # 5. Mark deletions
        for deleted_path in diff.to_mark_deleted:
            self._state.mark_deleted(root_path, deleted_path)
            summary.marked_deleted += 1

        # 6. Save state
        self._state.save()

        summary.finished_at = datetime.now(timezone.utc)
        self._summary = None

        logger.info(f"Ingestion complete: {summary}")

        return summary

    def _prepare_file_no_enrich(self, candidate: FileCandidate) -> Optional[Dict]:
        """
        Prepare a file for ingestion using Source → Parser → Chunker flow.

        Enrichment happens in a separate batch phase for efficiency.

        Args:
            candidate: File candidate with all metadata.

        Returns:
            file_data dict or None if file should be skipped.
        """
        file_path = Path(candidate.path)

        # Handle table files (CSV, TSV) specially
        if is_table_file(file_path):
            return self._prepare_table_file(candidate, file_path)

        # 1. Create SourceFile from candidate path
        source_file = SourceFile(
            uri=file_path.as_uri(),
            local_path=file_path,
            metadata={
                "content_hash": candidate.content_hash,
            },
        )

        # 2. Parse file using ParserRouter → ParsedDocument
        parsed_doc = self._parser_router.parse(source_file)

        # Check for empty content
        if not parsed_doc.full_text.strip():
            logger.warning(f"Empty content from {candidate.path}, skipping")
            return None

        # 3. Add metadata to parsed document
        doc_id = file_path.stem
        parsed_doc.metadata.update(
            {
                "source_file": candidate.path,
                "doc_id": doc_id,
                "content_hash": candidate.content_hash,
                "parser_id": candidate.parser_id,
                "chunker_id": candidate.chunker_id,
            }
        )

        # 3.5 Extract tables before chunking
        # Tables are converted to schema chunks with embedded JSON data
        table_extractor = TableExtractor()
        parsed_doc, table_chunks = table_extractor.extract(parsed_doc)

        # 4. Get chunker and chunk the ParsedDocument (tables removed)
        chunker = self._chunking_router.get_chunker(candidate.ext)
        chunks = chunker.chunk(parsed_doc)

        # Add table chunks to the regular chunks
        chunks.extend(table_chunks)

        if not chunks:
            logger.warning(f"No chunks from {candidate.path}, skipping")
            return None

        # 5. Prepare chunk data (no enrichment yet - that's batched later)
        chunk_data = []
        for chunk in chunks:
            chunk_id = compute_chunk_id(
                content_hash=candidate.content_hash,
                chunk_index=chunk.chunk_index,
                parser_id=candidate.parser_id,
                chunker_id=candidate.chunker_id,
                embedding_id=candidate.embedding_id,
            )

            chunk_content_hash = _hash_text(chunk.content)

            chunk_data.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": chunk,
                    "content_hash": chunk_content_hash,
                    "description": None,  # Filled in by batch summarization
                }
            )

        return {"chunk_data": chunk_data}

    def _prepare_table_file(
        self,
        candidate: FileCandidate,
        file_path: Path,
    ) -> Optional[Dict]:
        """
        Process a table file (CSV, TSV) into TableStore and create schema chunk.

        Args:
            candidate: File candidate with metadata.
            file_path: Path to the table file.

        Returns:
            file_data dict with schema chunk, or None on error.
        """
        try:
            # Parse CSV file
            parsed = parse_csv(file_path)

            # Store in TableStore
            table_hash = self._table_store.store(
                table_id=parsed.table_id,
                columns=parsed.columns,
                rows=parsed.rows,
                source_file=str(file_path),
            )

            # Create schema chunk for vector search
            sample_rows = get_sample_rows(parsed, n=3)
            schema_chunk = create_schema_chunk_for_stored_table(
                table_id=parsed.table_id,
                columns=parsed.columns,
                row_count=parsed.row_count,
                source_file=str(file_path),
                table_hash=table_hash,
                sample_rows=sample_rows,
            )

            logger.info(
                f"Stored table {file_path.name}: {len(parsed.columns)} columns, "
                f"{parsed.row_count} rows"
            )

            # Compute chunk_id for the schema chunk
            chunk_id = compute_chunk_id(
                content_hash=candidate.content_hash,
                chunk_index=0,
                parser_id="table",
                chunker_id="table",
                embedding_id=candidate.embedding_id,
            )

            chunk_content_hash = _hash_text(schema_chunk.content)

            return {
                "chunk_data": [
                    {
                        "chunk_id": chunk_id,
                        "chunk": schema_chunk,
                        "content_hash": chunk_content_hash,
                        "description": None,
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Failed to process table {file_path}: {e}")
            return None

    def _upsert_file(
        self,
        candidate: FileCandidate,
        file_data: Dict,
        vectors: List[List[float]],
        defer_persist: bool = False,
    ) -> None:
        """
        Build points and upsert to vector DB.

        Args:
            candidate: File candidate with all metadata.
            file_data: Prepared file data from _prepare_file_no_enrich.
            vectors: Embedding vectors for this file's chunks.
            defer_persist: If True, defer disk persistence (for batching).
        """
        chunk_data = file_data["chunk_data"]
        points = []

        for i, data in enumerate(chunk_data):
            chunk = data["chunk"]
            payload = {
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content_hash": candidate.content_hash,
                "source_path": candidate.path,
                "ext": candidate.ext,
                "chunk_text_hash": f"sha256:{_hash_text(chunk.content)}",
                "parser_id": candidate.parser_id,
                "chunker_id": candidate.chunker_id,
                "embedding_id": candidate.embedding_id,
                "is_deleted": False,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "metadata": dict(chunk.metadata or {}),
            }

            if data["description"] is not None:
                payload["description"] = data["description"]
                payload["enricher_id"] = self._enricher_id

            points.append(
                {
                    "id": data["chunk_id"],
                    "vector": vectors[i],
                    "payload": payload,
                }
            )

        self._vdb_writer.upsert(self._collection, points, defer_persist=defer_persist)
        logger.debug(f"Upserted {len(points)} chunks from {candidate.path}")

    def _detect_vocabulary(self, prepared_files: List[tuple]) -> None:
        """
        Auto-detect keywords from ingested chunks and save to vocabulary.

        This scans all chunks for patterns like test case IDs, ticket numbers,
        version numbers, etc. and saves them to .fitz/keywords/{collection}.yaml.

        Args:
            prepared_files: List of (candidate, file_data) tuples from ingestion.
        """
        try:
            from fitz_ai.ingestion.vocabulary import KeywordDetector, VocabularyStore

            # Collect all chunks (as simple objects with content, doc_id, metadata)
            all_chunks = []
            for candidate, file_data in prepared_files:
                for chunk_data in file_data["chunk_data"]:
                    chunk = chunk_data["chunk"]
                    all_chunks.append(chunk)

            if not all_chunks:
                return

            # Detect keywords
            detector = KeywordDetector()
            keywords = detector.detect_from_chunks(all_chunks)

            if keywords:
                # Merge with existing vocabulary (preserves user edits)
                # Store per-collection so different collections have separate vocabularies
                store = VocabularyStore(collection=self._collection)
                merged = store.merge_and_save(keywords, source_docs=len(prepared_files))
                logger.info(
                    f"Vocabulary [{self._collection}]: detected {len(keywords)}, "
                    f"saved {len(merged)} keywords"
                )

        except Exception as e:
            # Non-fatal: vocabulary detection is optional
            logger.warning(f"Vocabulary detection failed: {e}")

    def ingest_artifacts(
        self,
        on_progress: Optional[callable] = None,
    ) -> tuple[int, List[str]]:
        """
        Generate and ingest project artifacts.

        Artifacts are high-level summaries that provide context for code retrieval.
        They are stored with is_artifact=True and always retrieved with score=1.0.

        Args:
            on_progress: Optional callback(current, total, artifact_name) for progress.

        Returns:
            Tuple of (artifacts_generated, error_messages).
        """
        if self._enrichment_pipeline is None:
            return 0, []

        errors: List[str] = []
        logger.info("Generating project artifacts...")

        try:
            # Generate artifacts using the pipeline
            artifacts = self._enrichment_pipeline.generate_artifacts()

            if not artifacts:
                logger.info("No artifacts generated")
                return 0, []

            # Batch embed all artifacts at once
            total = len(artifacts)
            if on_progress:
                on_progress(0, total, "Embedding...")

            texts_to_embed = [artifact.content for artifact in artifacts]
            vectors = self._embedder.embed_batch(texts_to_embed)

            # Build points with vectors
            points = []
            for i, artifact in enumerate(artifacts):
                if on_progress:
                    on_progress(i + 1, total, artifact.artifact_type.value)

                # Create deterministic ID for artifact
                artifact_id = f"artifact:{artifact.artifact_type.value}:{self._collection}"

                # Build payload using artifact's to_payload method
                payload = artifact.to_payload()
                payload["collection"] = self._collection
                payload["embedding_id"] = self._embedding_id
                payload["ingested_at"] = datetime.now(timezone.utc).isoformat()

                points.append(
                    {
                        "id": artifact_id,
                        "vector": vectors[i],
                        "payload": payload,
                    }
                )

            # Upsert all artifacts
            self._vdb_writer.upsert(self._collection, points)
            logger.info(f"Ingested {len(artifacts)} artifacts")

            return len(artifacts), errors

        except Exception as e:
            errors.append(f"Artifact generation error: {e}")
            logger.warning(f"Failed to generate/ingest artifacts: {e}")
            return 0, errors


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_diff_ingest(
    source: str | Path,
    *,
    state_manager: IngestStateManager,
    vector_db_writer: VectorDBWriter,
    embedder: Embedder,
    parser_router: ParserRouter,
    chunking_router: ChunkingRouter,
    collection: str,
    embedding_id: str,
    vector_db_id: Optional[str] = None,
    enrichment_pipeline: Optional["EnrichmentPipeline"] = None,
    force: bool = False,
    on_progress: Optional[callable] = None,
    skip_artifacts: bool = False,
) -> IngestSummary:
    """
    Convenience function to run diff ingestion.

    Args:
        source: Path to directory or file.
        state_manager: Manager for ingest state.
        vector_db_writer: Writer for upserting vectors.
        embedder: Embedder for creating vectors.
        parser_router: Router for file-type specific parsing.
        chunking_router: Router for file-type specific chunking.
        collection: Vector DB collection name.
        embedding_id: Current embedding configuration ID.
        vector_db_id: Vector DB plugin name (e.g., "qdrant", "local_faiss").
        enrichment_pipeline: Optional unified enrichment pipeline.
        force: If True, ingest everything regardless of state.
        on_progress: Optional callback(current, total, file_path) for progress updates.
        skip_artifacts: If True, skip artifact generation (caller handles separately).

    Returns:
        IngestSummary with results.
    """
    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=vector_db_writer,
        embedder=embedder,
        parser_router=parser_router,
        chunking_router=chunking_router,
        collection=collection,
        embedding_id=embedding_id,
        vector_db_id=vector_db_id,
        enrichment_pipeline=enrichment_pipeline,
    )
    return executor.run(source, force=force, on_progress=on_progress, skip_artifacts=skip_artifacts)


__all__ = [
    "VectorDBWriter",
    "Embedder",
    "IngestSummary",
    "DiffIngestExecutor",
    "run_diff_ingest",
]
