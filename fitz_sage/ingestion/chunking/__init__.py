# fitz_sage/ingestion/chunking/__init__.py
"""
Chunking subsystem for ingestion.

Provides:
- Chunker protocol for implementing chunkers
- ChunkingRouter for file-type specific routing
- Built-in chunker plugins (simple, recursive, etc.)

Usage:
    from fitz_sage.ingestion.chunking import ChunkingRouter, Chunker
    from fitz_sage.ingestion.chunking.config import ChunkingRouterConfig

    config = ChunkingRouterConfig(...)
    router = ChunkingRouter.from_config(config)
    chunker = router.get_chunker(".md")
    chunks = chunker.chunk(parsed_doc)
"""

from fitz_sage.core.chunk import Chunk
from fitz_sage.ingestion.chunking.base import Chunker
from fitz_sage.ingestion.chunking.router import ChunkingRouter

__all__ = [
    # Protocol
    "Chunker",
    "Chunk",
    # Router
    "ChunkingRouter",
]
