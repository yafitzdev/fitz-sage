from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Chunk:
    text: str
    metadata: dict


class BaseChunker:
    """
    Abstract chunking plugin.

    Plugins MUST implement:
        chunk_text(text: str, base_meta: dict) -> List[Chunk]

    They do NOT:
    - read files
    - log
    - perform exception handling
    - construct metadata
    - interact with paths

    All of that is handled by ChunkingEngine.
    """

    def chunk_text(self, text: str, base_meta: Dict) -> List[Chunk]:
        raise NotImplementedError("Chunking plugins must implement chunk_text()")
