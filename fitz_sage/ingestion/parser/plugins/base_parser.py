# fitz_sage/ingestion/parser/plugins/base_parser.py
"""
Base parser class with common functionality for all parser plugins.

This reduces code duplication across parser implementations by providing:
- Common can_parse() implementation
- Standard metadata building
- Error handling patterns
- Default field definitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Set

from fitz_sage.core.document import ParsedDocument
from fitz_sage.ingestion.parser.base import ParseError
from fitz_sage.ingestion.source.base import SourceFile

logger = logging.getLogger(__name__)


@dataclass
class BaseParser:
    """
    Base class for parser implementations.

    Provides common functionality that all parsers need:
    - Extension-based can_parse() check
    - Standard metadata building
    - Common error handling patterns

    Subclasses should:
    1. Set default values for plugin_name and supported_extensions
    2. Implement parse() method
    3. Override can_parse() if more sophisticated detection needed

    Example:
        @dataclass
        class MyParser(BaseParser):
            plugin_name: str = field(default="my_parser")
            supported_extensions: Set[str] = field(
                default_factory=lambda: {".xyz", ".abc"}
            )

            def parse(self, file: SourceFile) -> ParsedDocument:
                # Custom parsing logic here
                ...
    """

    # Required fields - subclasses should provide defaults
    plugin_name: str
    supported_extensions: Set[str]

    def can_parse(self, file: SourceFile) -> bool:
        """
        Check if this parser can handle the file.

        Default implementation checks file extension against supported_extensions.
        Override for more sophisticated detection (e.g., MIME type, magic bytes).

        Args:
            file: SourceFile to check

        Returns:
            True if this parser can handle the file
        """
        return file.extension in self.supported_extensions

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a file into structured content.

        Must be implemented by subclasses.

        Args:
            file: SourceFile with local_path for reading

        Returns:
            ParsedDocument with structured elements

        Raises:
            ParseError: If the file cannot be parsed
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement parse()")

    def _build_metadata(self, file: SourceFile, **kwargs) -> dict[str, Any]:
        """
        Build standard metadata for parsed documents.

        Provides consistent metadata across all parsers.
        Subclasses can override to add parser-specific metadata.

        Args:
            file: Source file being parsed
            **kwargs: Additional metadata fields

        Returns:
            Dict with standard metadata fields
        """
        metadata = {
            "parser": self.plugin_name,
            "source_extension": file.extension,
            "source_filename": file.local_path.name if file.local_path else None,
        }

        # Add any additional metadata passed in
        metadata.update(kwargs)

        return metadata

    def _read_file_text(
        self, file: SourceFile, encoding: str = "utf-8", fallback_encoding: str = "latin-1"
    ) -> str:
        """
        Read text content from a file with encoding fallback.

        Common pattern for text-based parsers.

        Args:
            file: SourceFile to read
            encoding: Primary encoding to try
            fallback_encoding: Fallback if primary fails

        Returns:
            File content as string

        Raises:
            ParseError: If file cannot be read
        """
        if not file.local_path:
            raise ParseError("No local path available for file", source=file.uri)

        try:
            # Try primary encoding
            try:
                content = file.local_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                # Fallback to alternative encoding
                logger.debug(
                    f"Failed to decode {file.uri} with {encoding}, " f"trying {fallback_encoding}"
                )
                content = file.local_path.read_text(encoding=fallback_encoding)

            return content

        except Exception as e:
            raise ParseError(
                f"Failed to read file: {e}",
                source=file.uri,
                cause=e,
            ) from e

    def _read_file_bytes(self, file: SourceFile) -> bytes:
        """
        Read binary content from a file.

        Common pattern for binary parsers (PDF, images, etc).

        Args:
            file: SourceFile to read

        Returns:
            File content as bytes

        Raises:
            ParseError: If file cannot be read
        """
        if not file.local_path:
            raise ParseError("No local path available for file", source=file.uri)

        try:
            return file.local_path.read_bytes()
        except Exception as e:
            raise ParseError(
                f"Failed to read file: {e}",
                source=file.uri,
                cause=e,
            ) from e

    def __repr__(self) -> str:
        """Concise representation showing plugin name and extension count."""
        ext_count = len(self.supported_extensions) if self.supported_extensions else 0
        return f"{self.__class__.__name__}(plugin='{self.plugin_name}', extensions={ext_count})"
