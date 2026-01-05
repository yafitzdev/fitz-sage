# fitz_ai/ingestion/chunking/__init__.py
"""
Chunking subsystem for ingestion.

Provides:
- Chunker protocol for implementing chunkers
- ChunkingRouter for file-type specific routing
- ChunkingEngine for orchestration
- Built-in chunker plugins (simple, etc.)

Usage:
    from fitz_ai.ingestion.chunking import (
        ChunkingRouter,
        ChunkingEngine,
        Chunker,
    )
    from fitz_ai.engines.fitz_rag.config import ChunkingRouterConfig

    config = ChunkingRouterConfig(...)
    router = ChunkingRouter.from_config(config)
    engine = ChunkingEngine(router)

    chunks = engine.run(raw_doc)
"""

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.chunking.base import Chunker
from fitz_ai.ingestion.chunking.engine import ChunkingEngine
from fitz_ai.ingestion.chunking.router import ChunkingRouter

__all__ = [
    # Protocol
    "Chunker",
    "Chunk",
    # Router
    "ChunkingRouter",
    # Engine
    "ChunkingEngine",
]
