# fitz_ai/tabular/parser/__init__.py
"""Table file parsers."""

from fitz_ai.tabular.parser.csv_parser import (
    SUPPORTED_EXTENSIONS,
    ParsedTableFile,
    can_parse,
    parse_csv,
)

__all__ = [
    "ParsedTableFile",
    "parse_csv",
    "can_parse",
    "SUPPORTED_EXTENSIONS",
]
