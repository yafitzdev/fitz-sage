# fitz_sage/ingestion/parser/plugins/__init__.py
"""
Parser plugins for document format understanding.
"""

from fitz_sage.ingestion.parser.plugins.docling import DoclingParser
from fitz_sage.ingestion.parser.plugins.plaintext import PlainTextParser

__all__ = ["DoclingParser", "PlainTextParser"]
