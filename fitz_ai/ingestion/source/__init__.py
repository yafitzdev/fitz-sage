# fitz_ai/ingestion/source/__init__.py
"""
Source plugins for file discovery and access.

Sources handle the "where" of ingestion - finding files regardless of
storage backend (filesystem, S3, MongoDB, etc.)
"""

from fitz_ai.ingestion.source.base import Source, SourceFile
from fitz_ai.ingestion.source.plugins.filesystem import FileSystemSource

__all__ = [
    "Source",
    "SourceFile",
    "FileSystemSource",
]
