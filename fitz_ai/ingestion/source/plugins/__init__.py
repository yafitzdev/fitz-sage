# fitz_ai/ingestion/source/plugins/__init__.py
"""Source plugins for file discovery."""

from fitz_ai.ingestion.source.plugins.filesystem import FileSystemSource

__all__ = ["FileSystemSource"]
