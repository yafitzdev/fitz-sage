# fitz_ai/core/document.py
"""
Core document types for the ingestion pipeline.

ParsedDocument represents structured content extracted from any file format.
It preserves document structure (headings, tables, code blocks, etc.) so
chunkers can make intelligent splitting decisions.

Flow: Source → Parser → ParsedDocument → Chunker → Chunks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ElementType(Enum):
    """Types of structural elements in a document."""

    TEXT = "text"  # Regular paragraph text
    HEADING = "heading"  # Section heading (h1-h6)
    TABLE = "table"  # Tabular data
    FIGURE = "figure"  # Image or diagram
    CODE_BLOCK = "code_block"  # Source code
    LIST_ITEM = "list_item"  # Bulleted or numbered list item
    QUOTE = "quote"  # Block quote
    PAGE_BREAK = "page_break"  # Page boundary marker


@dataclass
class DocumentElement:
    """
    A single structural element in a parsed document.

    Elements represent semantic units: paragraphs, headings, tables, etc.
    Chunkers use element boundaries to make better splitting decisions.
    """

    type: ElementType
    content: str
    level: Optional[int] = None  # Heading level (1-6), list depth, etc.
    language: Optional[str] = None  # Programming language for code blocks
    page: Optional[int] = None  # Page number (for PDFs)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"DocumentElement({self.type.value}, {preview!r})"


@dataclass
class ParsedDocument:
    """
    Structured representation of a parsed file.

    Produced by Parsers, consumed by Chunkers.
    Preserves document structure for intelligent chunking.
    """

    source: str  # Original source URI or path
    elements: List[DocumentElement]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get all text content concatenated (for simple chunkers)."""
        return "\n\n".join(el.content for el in self.elements if el.content)

    @property
    def element_count(self) -> int:
        """Number of elements in the document."""
        return len(self.elements)

    @property
    def page_count(self) -> Optional[int]:
        """Number of pages (if available from metadata)."""
        return self.metadata.get("page_count")

    def elements_by_type(self, element_type: ElementType) -> List[DocumentElement]:
        """Get all elements of a specific type."""
        return [el for el in self.elements if el.type == element_type]

    def __repr__(self) -> str:
        return f"ParsedDocument({self.source!r}, {self.element_count} elements)"


__all__ = [
    "ElementType",
    "DocumentElement",
    "ParsedDocument",
]
