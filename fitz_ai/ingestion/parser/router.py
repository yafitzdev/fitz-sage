# fitz_ai/ingestion/parser/router.py
"""
ParserRouter - Routes files to appropriate parsers based on extension.

Architecture:
    ┌─────────────────────────────────────┐
    │           ParserRouter              │
    │  Routes files to appropriate parser │
    └─────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
  PlainTextParser DoclingParser  (future)
   (.txt, .md)    (.pdf, .docx)

Usage:
    router = ParserRouter()
    parsed_doc = router.parse(source_file)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from fitz_ai.core.document import ParsedDocument
from fitz_ai.ingestion.parser.base import ParseError, Parser
from fitz_ai.ingestion.source.base import SourceFile

logger = logging.getLogger(__name__)


@dataclass
class ParserRouter:
    """
    Routes files to appropriate parsers based on extension.

    Priority order:
    1. Extension-specific parser (if registered)
    2. Default fallback parser (PlainTextParser)

    Example:
        router = ParserRouter()
        doc = router.parse(source_file)
    """

    # Map of extension -> parser instance
    _parsers: Dict[str, Parser] = field(default_factory=dict)
    _default_parser: Optional[Parser] = field(default=None)
    _warned_extensions: Set[str] = field(default_factory=set)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize with default parsers."""
        if not self._initialized:
            self._initialize_default_parsers()
            self._initialized = True

    def _initialize_default_parsers(self) -> None:
        """Set up default parsers."""
        from fitz_ai.ingestion.parser.plugins.plaintext import (
            PLAINTEXT_EXTENSIONS,
            PlainTextParser,
        )

        # Default parser for text files
        plaintext = PlainTextParser()
        self._default_parser = plaintext

        # Register plaintext parser for its extensions
        for ext in PLAINTEXT_EXTENSIONS:
            self._parsers[ext] = plaintext

        # Register Docling parser for complex documents (PDF, DOCX, images, etc.)
        from fitz_ai.ingestion.parser.plugins.docling import (
            DOCLING_EXTENSIONS,
            DoclingParser,
        )

        docling = DoclingParser()
        for ext in DOCLING_EXTENSIONS:
            # Docling handles these formats - override any plaintext registrations
            self._parsers[ext] = docling

    def register_parser(self, parser: Parser, extensions: Optional[List[str]] = None) -> None:
        """
        Register a parser for specific extensions.

        Args:
            parser: Parser instance to register.
            extensions: List of extensions to register for.
                       If None, uses parser.supported_extensions.
        """
        exts = extensions or list(getattr(parser, "supported_extensions", []))
        for ext in exts:
            normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            self._parsers[normalized] = parser
            logger.debug(f"Registered parser '{parser.plugin_name}' for {normalized}")

    def get_parser(self, ext: str) -> Parser:
        """
        Get the appropriate parser for a file extension.

        Args:
            ext: File extension (e.g., ".md", ".pdf", "txt").

        Returns:
            Parser instance for the extension.
        """
        normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"

        parser = self._parsers.get(normalized)
        if parser is not None:
            return parser

        # Fall back to default
        if normalized not in self._warned_extensions:
            self._warned_extensions.add(normalized)
            logger.debug(
                f"No parser registered for '{normalized}', "
                f"using default '{self._default_parser.plugin_name}'"
            )

        return self._default_parser

    def can_parse(self, file: SourceFile) -> bool:
        """
        Check if any parser can handle the file.

        Args:
            file: SourceFile to check.

        Returns:
            True if a parser can handle the file.
        """
        parser = self.get_parser(file.extension)
        return parser.can_parse(file)

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a file using the appropriate parser.

        Args:
            file: SourceFile to parse.

        Returns:
            ParsedDocument with structured elements.

        Raises:
            ParseError: If parsing fails.
        """
        parser = self.get_parser(file.extension)
        logger.debug(
            f"Parsing '{file.name}' with {parser.plugin_name} parser"
        )

        try:
            return parser.parse(file)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(
                f"Parser '{parser.plugin_name}' failed: {e}",
                source=file.uri,
                cause=e,
            ) from e

    def get_parser_id(self, ext: str) -> str:
        """
        Get a unique ID for the parser handling an extension.

        Args:
            ext: File extension.

        Returns:
            Parser ID string (e.g., "plaintext:v1", "docling:v1").
        """
        parser = self.get_parser(ext)
        return f"{parser.plugin_name}:v1"

    @property
    def registered_extensions(self) -> List[str]:
        """Get list of extensions with registered parsers."""
        return sorted(self._parsers.keys())

    def __repr__(self) -> str:
        ext_list = ", ".join(self.registered_extensions[:5])
        if len(self.registered_extensions) > 5:
            ext_list += f", ... ({len(self.registered_extensions)} total)"
        return f"ParserRouter(extensions=[{ext_list}])"


__all__ = ["ParserRouter"]
