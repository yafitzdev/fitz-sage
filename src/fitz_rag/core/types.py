"""
Core types and dataclasses shared across the fitz_rag library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Protocol, List


# ---------------------------------------------------------
# RetrievedChunk: common object returned by all strategies
# ---------------------------------------------------------
@dataclass
class RetrievedChunk:
    collection: str
    score: float
    text: str
    metadata: Dict[str, Any]
    chunk_id: Any


# ---------------------------------------------------------
# Chunk: generic chunk produced by chunkers and retrievers
# ---------------------------------------------------------
@dataclass
class Chunk:
    id: str = "unknown"               # retriever doesn't provide an ID
    text: str = ""
    metadata: Dict[str, Any] | None = None
    score: float | None = None        # added for retriever tests

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
