# fitz_ai/ingest/state/schema.py
"""
State schema for incremental ingestion.

Defines the Pydantic models for .fitz/ingest.json as specified in §4.2.

Key concepts:
- State exists for speed and bookkeeping
- Vector DB is authoritative (state is not truth)
- State tracks file paths for deletion detection
- State stores config IDs for future staleness checks
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class FileStatus(str, Enum):
    """Status of a tracked file."""
    ACTIVE = "active"
    DELETED = "deleted"


class FileEntry(BaseModel):
    """
    Per-file tracking entry.

    Stored under roots.<root_path>.files.<file_path>
    """
    model_config = ConfigDict(extra="forbid")

    ext: str = Field(..., description="File extension (e.g., '.md')")
    size_bytes: int = Field(..., description="File size in bytes (for debugging)")
    mtime_epoch: float = Field(..., description="Modification time as Unix epoch (for debugging)")
    content_hash: str = Field(..., description="SHA-256 hash of file content (sha256:...)")
    status: FileStatus = Field(default=FileStatus.ACTIVE, description="File status")
    last_seen_at: datetime = Field(default_factory=datetime.utcnow, description="Last scan time")

    def is_active(self) -> bool:
        """Check if file is active (not deleted)."""
        return self.status == FileStatus.ACTIVE


class RootEntry(BaseModel):
    """
    Per-root directory tracking.

    Tracks all files under a given root path.
    """
    model_config = ConfigDict(extra="forbid")

    last_run_at: datetime = Field(default_factory=datetime.utcnow, description="Last ingest run time")
    files: Dict[str, FileEntry] = Field(default_factory=dict, description="File entries keyed by abs path")


class EmbeddingConfig(BaseModel):
    """
    Embedding configuration snapshot.

    Used to detect when embedding config changes (future staleness checks).
    """
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., description="Embedding provider (e.g., 'openai', 'cohere', 'ollama')")
    model: str = Field(..., description="Model name (e.g., 'text-embedding-3-small')")
    dimension: Optional[int] = Field(default=None, description="Vector dimension (optional)")
    normalize: Optional[bool] = Field(default=None, description="Whether vectors are normalized")
    id: str = Field(..., description="Composite ID: provider:model")

    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> "EmbeddingConfig":
        """Create an EmbeddingConfig with auto-generated ID."""
        return cls(
            provider=provider,
            model=model,
            dimension=kwargs.get("dimension"),
            normalize=kwargs.get("normalize"),
            id=f"{provider}:{model}",
        )


class ChunkingConfigEntry(BaseModel):
    """
    Per-extension chunking configuration.

    Stored under chunking_by_ext.<ext>
    """
    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(..., description="Chunking strategy (e.g., 'tokens', 'chars')")
    chunk_size: int = Field(..., description="Chunk size")
    overlap: int = Field(default=0, description="Overlap between chunks")
    id: str = Field(..., description="Composite ID: strategy_size_overlap")

    @classmethod
    def create(cls, strategy: str, chunk_size: int, overlap: int = 0) -> "ChunkingConfigEntry":
        """Create a ChunkingConfigEntry with auto-generated ID."""
        return cls(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            id=f"{strategy}_{chunk_size}_{overlap}",
        )


class ParsingConfigEntry(BaseModel):
    """
    Per-extension parsing configuration.

    Stored under parsing_by_ext.<ext>
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Parser ID (e.g., 'md.v1', 'pdf.v1')")


# Default configurations per spec §9
DEFAULT_PARSING_BY_EXT: Dict[str, ParsingConfigEntry] = {
    ".md": ParsingConfigEntry(id="md.v1"),
    ".txt": ParsingConfigEntry(id="txt.v1"),
    ".py": ParsingConfigEntry(id="py.v1"),
    ".pdf": ParsingConfigEntry(id="pdf.v1"),
}

DEFAULT_CHUNKING_BY_EXT: Dict[str, ChunkingConfigEntry] = {
    ".md": ChunkingConfigEntry.create("tokens", 800, 120),
    ".txt": ChunkingConfigEntry.create("tokens", 800, 120),
    ".py": ChunkingConfigEntry.create("tokens", 800, 120),
    ".pdf": ChunkingConfigEntry.create("tokens", 800, 120),
}


class IngestState(BaseModel):
    """
    Root state model for .fitz/ingest.json.

    This is the complete schema as specified in §4.2.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, description="Schema version for migrations")
    project_id: str = Field(..., description="UUID for this project")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    roots: Dict[str, RootEntry] = Field(default_factory=dict, description="Tracked roots keyed by abs path")
    embedding: Optional[EmbeddingConfig] = Field(default=None, description="Current embedding config")
    chunking_by_ext: Dict[str, ChunkingConfigEntry] = Field(
        default_factory=lambda: dict(DEFAULT_CHUNKING_BY_EXT),
        description="Chunking config by extension",
    )
    parsing_by_ext: Dict[str, ParsingConfigEntry] = Field(
        default_factory=lambda: dict(DEFAULT_PARSING_BY_EXT),
        description="Parsing config by extension",
    )

    def get_file_entry(self, root: str, file_path: str) -> Optional[FileEntry]:
        """Get file entry if it exists."""
        root_entry = self.roots.get(root)
        if root_entry is None:
            return None
        return root_entry.files.get(file_path)

    def get_known_paths(self, root: str) -> set[str]:
        """Get all known file paths for a root."""
        root_entry = self.roots.get(root)
        if root_entry is None:
            return set()
        return set(root_entry.files.keys())

    def get_active_paths(self, root: str) -> set[str]:
        """Get all active (non-deleted) file paths for a root."""
        root_entry = self.roots.get(root)
        if root_entry is None:
            return set()
        return {
            path for path, entry in root_entry.files.items()
            if entry.is_active()
        }

    def get_parser_id(self, ext: str) -> str:
        """Get parser ID for extension, with fallback."""
        entry = self.parsing_by_ext.get(ext)
        if entry:
            return entry.id
        # Fallback: extension without dot + .v1
        return f"{ext.lstrip('.')}.v1"

    def get_chunker_id(self, ext: str) -> str:
        """Get chunker ID for extension, with fallback."""
        entry = self.chunking_by_ext.get(ext)
        if entry:
            return entry.id
        # Fallback: default tokens config
        return "tokens_800_120"

    def get_embedding_id(self) -> str:
        """Get embedding ID, with fallback."""
        if self.embedding:
            return self.embedding.id
        return "unknown:unknown"


__all__ = [
    "FileStatus",
    "FileEntry",
    "RootEntry",
    "EmbeddingConfig",
    "ChunkingConfigEntry",
    "ParsingConfigEntry",
    "IngestState",
    "DEFAULT_PARSING_BY_EXT",
    "DEFAULT_CHUNKING_BY_EXT",
]