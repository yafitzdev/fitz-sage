# fitz_ai/ingestion/parser/__init__.py
"""
Parser plugins for document format understanding.

Parsers handle the "how" of extraction - converting files into
structured ParsedDocument with preserved semantics.
"""

from fitz_ai.ingestion.parser.base import ParseError, Parser
from fitz_ai.ingestion.parser.plugins.docling import DoclingParser
from fitz_ai.ingestion.parser.plugins.plaintext import PlainTextParser

__all__ = [
    "Parser",
    "ParseError",
    "DoclingParser",
    "PlainTextParser",
]
