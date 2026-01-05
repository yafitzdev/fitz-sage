# fitz_ai/ingestion/parser/plugins/docling.py
"""
Docling-based parser for PDF, DOCX, images, and more.

Uses IBM's Docling library for advanced document understanding including:
- Layout analysis and reading order
- Table structure extraction
- Figure/image detection
- Code block recognition
- Formula handling

Requires: pip install docling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Set

from fitz_ai.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_ai.ingestion.parser.base import ParseError
from fitz_ai.ingestion.source.base import SourceFile

logger = logging.getLogger(__name__)

# Supported file extensions
DOCLING_EXTENSIONS: Set[str] = {
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}


@dataclass
class DoclingParser:
    """
    Parser using Docling for document understanding.

    Docling provides advanced PDF/document parsing with:
    - Vision-based layout analysis
    - Table structure detection
    - Reading order inference
    - Multi-format support

    Example:
        parser = DoclingParser()
        doc = parser.parse(source_file)
        for element in doc.elements:
            print(element.type, element.content[:50])
    """

    plugin_name: str = field(default="docling", repr=False)
    supported_extensions: Set[str] = field(default_factory=lambda: DOCLING_EXTENSIONS)

    # Lazy-loaded converter
    _converter: object = field(default=None, repr=False)

    def _get_converter(self):
        """Lazy-load the DocumentConverter."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
            except ImportError as e:
                raise ImportError(
                    "Docling is required for this parser. Install it with: pip install docling"
                ) from e
        return self._converter

    def can_parse(self, file: SourceFile) -> bool:
        """Check if this parser can handle the file."""
        return file.extension in self.supported_extensions

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a file into structured content using Docling.

        Args:
            file: SourceFile with local_path for reading.

        Returns:
            ParsedDocument with structured elements.

        Raises:
            ParseError: If parsing fails.
        """
        if not self.can_parse(file):
            raise ParseError(
                f"Unsupported file type: {file.extension}",
                source=file.uri,
            )

        try:
            return self._parse_with_docling(file)
        except ImportError:
            raise
        except Exception as e:
            raise ParseError(
                f"Failed to parse document: {e}",
                source=file.uri,
                cause=e,
            ) from e

    def _parse_with_docling(self, file: SourceFile) -> ParsedDocument:
        """Internal parsing using Docling."""
        converter = self._get_converter()

        # Convert the document
        logger.info(f"Parsing with Docling: {file.local_path}")
        result = converter.convert(str(file.local_path))
        doc = result.document

        # Extract elements
        elements: list[DocumentElement] = []

        for item, level in doc.iterate_items():
            element = self._convert_item(item, level, doc)
            if element:
                elements.append(element)

        # Build metadata
        metadata = {
            "parser": self.plugin_name,
            "source_extension": file.extension,
        }

        # Add page count if available
        if hasattr(doc, "pages") and doc.pages:
            metadata["page_count"] = len(doc.pages)

        return ParsedDocument(
            source=file.uri,
            elements=elements,
            metadata=metadata,
        )

    def _convert_item(
        self,
        item,
        level: int,
        doc,
    ) -> DocumentElement | None:
        """Convert a Docling item to DocumentElement."""
        from docling_core.types.doc.labels import DocItemLabel

        label = getattr(item, "label", None)
        if label is None:
            return None

        # Map Docling labels to our ElementType
        if label == DocItemLabel.SECTION_HEADER:
            text = self._get_text(item)
            if not text:
                return None
            heading_level = getattr(item, "level", level) or 1
            return DocumentElement(
                type=ElementType.HEADING,
                content=text,
                level=min(heading_level, 6),  # Cap at h6
            )

        elif label == DocItemLabel.TITLE:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.HEADING,
                content=text,
                level=1,  # Title is always h1
            )

        elif label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
            )

        elif label == DocItemLabel.LIST_ITEM:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.LIST_ITEM,
                content=text,
                level=level,
            )

        elif label == DocItemLabel.TABLE:
            # Export table as markdown for content
            try:
                table_md = item.export_to_markdown(doc=doc)
                return DocumentElement(
                    type=ElementType.TABLE,
                    content=table_md,
                    metadata={"rows": len(item.data.grid) if item.data else 0},
                )
            except Exception as e:
                logger.warning(f"Failed to export table: {e}")
                return None

        elif label == DocItemLabel.PICTURE:
            # For pictures, store description or placeholder
            caption = ""
            if hasattr(item, "captions") and item.captions:
                # Try to get caption text
                for cap_ref in item.captions:
                    if hasattr(cap_ref, "text"):
                        caption = cap_ref.text
                        break

            return DocumentElement(
                type=ElementType.FIGURE,
                content=caption or "[Figure]",
                metadata={
                    "has_image": item.image is not None if hasattr(item, "image") else False,
                },
            )

        elif label == DocItemLabel.CODE:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.CODE_BLOCK,
                content=text,
                language=None,  # Docling doesn't detect language
            )

        elif label == DocItemLabel.FORMULA:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"is_formula": True},
            )

        elif label in (DocItemLabel.FOOTNOTE, DocItemLabel.REFERENCE):
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"is_footnote": label == DocItemLabel.FOOTNOTE},
            )

        elif label == DocItemLabel.CAPTION:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"is_caption": True},
            )

        elif label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
            # Skip headers/footers by default (furniture)
            return None

        else:
            # Unknown label - try to extract text anyway
            text = self._get_text(item)
            if text:
                return DocumentElement(
                    type=ElementType.TEXT,
                    content=text,
                    metadata={"original_label": str(label)},
                )
            return None

    def _get_text(self, item) -> str:
        """Extract text from a Docling item."""
        # TextItem has 'text' attribute
        if hasattr(item, "text") and item.text:
            return item.text.strip()
        # Some items have 'orig' for original text
        if hasattr(item, "orig") and item.orig:
            return item.orig.strip()
        return ""


__all__ = ["DoclingParser", "DOCLING_EXTENSIONS"]
