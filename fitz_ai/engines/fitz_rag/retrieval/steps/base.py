# fitz_ai/engines/fitz_rag/retrieval/steps/base.py
"""
Base classes for retrieval steps.

All retrieval steps inherit from RetrievalStep and implement execute().
Duck typing is used for dependencies (vector clients, embedders, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fitz_ai.core.chunk import Chunk


@dataclass
class RetrievalStep(ABC):
    """
    Base class for retrieval steps.

    All steps take a query and list of chunks, and return an updated list of chunks.
    Steps are stateless and composable.
    """

    @abstractmethod
    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Execute step and return updated chunks."""
        ...

    @property
    def name(self) -> str:
        """Return the step class name."""
        return self.__class__.__name__
