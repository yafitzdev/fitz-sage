# fitz_ai/ingestion/source/base.py
"""
Source protocol for file discovery and access.

Sources handle the "where" of ingestion - finding files and providing
local access to them, regardless of storage backend (filesystem, S3, etc.)

Flow: Source.discover() → SourceFile → Parser.parse() → ParsedDocument
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable


@dataclass
class SourceFile:
    """
    A file discovered by a Source.

    Provides unified access regardless of storage backend.
    The local_path is always available - Sources download remote files
    to a temp location if needed.
    """

    uri: str  # Original URI (file://, s3://, mongodb://, etc.)
    local_path: Path  # Local path for parser access (may be temp file)
    mime_type: Optional[str] = None  # Detected or inferred MIME type
    size: Optional[int] = None  # File size in bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        """File extension (lowercase, with dot)."""
        return self.local_path.suffix.lower()

    @property
    def name(self) -> str:
        """Filename without path."""
        return self.local_path.name

    def __repr__(self) -> str:
        return f"SourceFile({self.uri!r})"


@runtime_checkable
class Source(Protocol):
    """
    Protocol for file sources.

    Sources discover files and provide local access to them.
    Implementations handle storage-specific logic (S3 auth, MongoDB queries, etc.)

    Example implementations:
    - FileSystemSource: Local filesystem
    - S3Source: AWS S3 buckets
    - MongoDBSource: GridFS or document attachments
    - URLSource: HTTP/HTTPS URLs
    """

    plugin_name: str

    def discover(
        self,
        root: str,
        patterns: Optional[List[str]] = None,
    ) -> Iterable[SourceFile]:
        """
        Discover files from the source.

        Args:
            root: Root location (path, bucket, collection, etc.)
            patterns: Optional glob patterns to filter files (e.g., ["*.pdf", "*.docx"])
                      If None, discover all supported files.

        Yields:
            SourceFile for each discovered file.

        Note:
            For remote sources, this may download files lazily or eagerly
            depending on the implementation.
        """
        ...


__all__ = [
    "SourceFile",
    "Source",
]
