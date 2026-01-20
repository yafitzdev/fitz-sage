# fitz_ai/ingestion/enrichment/modules/base.py
"""
Base classes for enrichment modules.

EnrichmentModule ABC defines the interface for pluggable enrichment components
that extract metadata from chunks during the enrichment bus pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class EnrichmentModule(ABC):
    """
    Base class for enrichment modules.

    Each module defines:
    - What to extract (prompt_instruction)
    - How to parse the result (parse_result)
    - Where to store it (apply_to_chunk or collect separately)

    To add a new enrichment:
    1. Subclass EnrichmentModule
    2. Implement the abstract methods
    3. Add to ChunkEnricher's module list
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this module."""
        ...

    @property
    @abstractmethod
    def json_key(self) -> str:
        """Key in the JSON response for this module's output."""
        ...

    @abstractmethod
    def prompt_instruction(self) -> str:
        """
        Return the instruction to include in the prompt.

        Should describe what to extract and the expected format.
        """
        ...

    @abstractmethod
    def parse_result(self, data: Any) -> Any:
        """
        Parse and validate the module's output from the JSON response.

        Args:
            data: The value from response[json_key]

        Returns:
            Parsed/validated result
        """
        ...

    def apply_to_chunk(self, chunk: "Chunk", result: Any) -> None:
        """
        Apply the enrichment result to a chunk's metadata.

        Override this if the module should attach data to chunks.
        Default implementation does nothing (for modules like keywords
        that collect data separately).
        """
        pass


__all__ = [
    "ChatClient",
    "EnrichmentModule",
]
