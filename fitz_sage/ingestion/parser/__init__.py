# fitz_sage/ingestion/parser/__init__.py
"""
Parser plugins for document format understanding.

Parsers handle the "how" of extraction - converting files into
structured ParsedDocument with preserved semantics.
"""

from fitz_sage.ingestion.parser.base import ParseError, Parser
from fitz_sage.ingestion.parser.router import ParserRouter

__all__ = [
    "Parser",
    "ParseError",
    "ParserRouter",
]
