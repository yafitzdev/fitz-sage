# fitz_sage/ingestion/source/plugins/__init__.py
"""Source plugins for file discovery."""

from fitz_sage.ingestion.source.plugins.filesystem import FileSystemSource

__all__ = ["FileSystemSource"]
