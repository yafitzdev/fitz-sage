# fitz_ai/ingestion/enrichment/base.py
"""
Base types for the enrichment system.

This module defines content type classification used by artifact plugins.
"""

from __future__ import annotations

from enum import Enum


class ContentType(str, Enum):
    """Content type classification for artifact plugin routing."""

    PYTHON = "python"
    CODE = "code"  # Non-Python code
    DOCUMENT = "document"  # Markdown, text, etc.
    STRUCTURED = "structured"  # JSON, YAML, etc.
    UNKNOWN = "unknown"


__all__ = [
    "ContentType",
]
