# fitz_ai/ingest/state/__init__.py
"""
State management for incremental ingestion.

This package handles the .fitz/ingest.json state file that tracks:
- Which files have been ingested
- Content hashes for change detection
- Deletion tracking
- Config snapshots for staleness detection

Key exports:
- IngestStateManager: Load/save state, update entries
- IngestState: Root Pydantic model
- FileEntry: Per-file tracking
"""

from .manager import IngestStateManager
from .schema import (
    ChunkingConfigEntry,
    EmbeddingConfig,
    FileEntry,
    FileStatus,
    IngestState,
    ParsingConfigEntry,
    RootEntry,
)

__all__ = [
    # Manager
    "IngestStateManager",
    # Schema
    "IngestState",
    "RootEntry",
    "FileEntry",
    "FileStatus",
    "EmbeddingConfig",
    "ChunkingConfigEntry",
    "ParsingConfigEntry",
]