# fitz_ai/ingestion/chunking/base.py
"""
Base protocol and classes for chunking plugins.

Each chunker plugin must implement:
- plugin_name: str - The plugin identifier (e.g., "simple", "semantic")
- chunker_id: str - Unique ID including params that affect output
- chunk(document) -> List[Chunk]

The chunker_id is used for:
1. State tracking - stored per file to detect config changes
2. Re-chunking decisions - if chunker_id changes, file is re-ingested
3. Reproducibility - same chunker_id guarantees same chunking behavior

Flow: ParsedDocument → Chunker.chunk() → List[Chunk]
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.document import ParsedDocument


@runtime_checkable
class Chunker(Protocol):
    """
    Protocol for document chunking plugins.

    Chunkers split ParsedDocument into retrieval-sized Chunks.
    They can use document structure (headings, paragraphs, etc.) for
    intelligent splitting decisions.

    Contract:
    - plugin_name: Identifies the plugin type (e.g., "simple", "semantic")
    - chunker_id: Unique ID including params that affect chunk output
    - chunk: Splits ParsedDocument into chunks

    The chunker_id format is: "{plugin_name}:{param1}:{param2}:..."

    Example:
        >>> chunker = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        >>> chunker.plugin_name
        'simple'
        >>> chunker.chunker_id
        'simple:1000:100'
    """

    plugin_name: str

    @property
    def chunker_id(self) -> str:
        """
        Unique identifier including parameters that affect chunk output.

        This ID is stored in the state file per-file. When the ID changes
        (e.g., chunk_size changed from 1000 to 800), the file will be
        re-chunked on the next ingestion run.

        Format: "{plugin_name}:{param1}:{param2}:..."

        Returns:
            Deterministic string ID for this chunker configuration.
        """
        ...

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """
        Split a parsed document into chunks.

        Args:
            document: ParsedDocument with structured elements.

        Returns:
            List of Chunk objects with content and metadata.

        Note:
            Chunkers can use document.elements for structure-aware chunking,
            or document.full_text for simple text-based chunking.
        """
        ...


__all__ = ["Chunker", "Chunk"]
