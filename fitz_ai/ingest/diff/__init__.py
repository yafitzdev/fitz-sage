# fitz_ai/ingest/diff/__init__.py
"""
Incremental (diff) ingestion for Fitz.

This package implements content-hash based incremental ingestion:
- Only ingest files that have changed
- Skip files that already exist in vector DB
- Track deletions via state file

Key components:
- Scanner: Walks directories and computes content hashes
- Differ: Computes action plan by querying vector DB (authoritative)
- Executor: Orchestrates the full pipeline

Usage:
    from fitz_ai.ingest.diff import run_diff_ingest
    from fitz_ai.ingest.state import IngestStateManager

    summary = run_diff_ingest(
        source="./documents",
        state_manager=IngestStateManager(),
        vector_db_reader=reader,
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunker=chunker_plugin,  # ChunkerPlugin with chunk_text method
        collection="my_docs",
    )
    print(summary)  # "scanned 10, ingested 3, skipped 7, ..."
"""

from .differ import (
    Differ,
    DiffResult,
    FileCandidate,
    StateReader,
    VectorDBReader,
    compute_diff,
)
from .executor import (
    DiffIngestExecutor,
    Embedder,
    IngestSummary,
    Parser,
    VectorDBWriter,
    run_diff_ingest,
)
from .scanner import (
    SUPPORTED_EXTENSIONS,
    FileScanner,
    scan_directory,
    ScannedFile,
    ScanResult,
)

__all__ = [
    # Scanner
    "SUPPORTED_EXTENSIONS",
    "ScannedFile",
    "ScanResult",
    "FileScanner",
    "scan_directory",
    # Differ
    "VectorDBReader",
    "StateReader",
    "FileCandidate",
    "DiffResult",
    "Differ",
    "compute_diff",
    # Executor
    "VectorDBWriter",
    "Embedder",
    "Parser",
    "IngestSummary",
    "DiffIngestExecutor",
    "run_diff_ingest",
]