# fitz_ai/ingestion/enrichment/entities/linker.py
"""
Entity linking module.

Links entities that co-occur in the same chunk, storing relationships
in chunk metadata for knowledge graph queries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk


@dataclass
class EntityLink:
    """
    A link between two entities that co-occur in a chunk.

    Attributes:
        source: Source entity name
        target: Target entity name
        source_type: Type of source entity (class, function, api, etc.)
        target_type: Type of target entity
        chunk_id: ID of the chunk where they co-occur
    """

    source: str
    target: str
    source_type: str
    target_type: str
    chunk_id: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class EntityLinker:
    """
    Links entities that co-occur in the same chunk.

    For each chunk containing 2+ entities, creates pairwise links
    representing co-occurrence relationships. These links can be
    used for knowledge graph construction and entity relationship queries.

    Example:
        If a chunk contains entities [A, B, C], the following links are created:
        - A <-> B
        - A <-> C
        - B <-> C
    """

    def link(self, chunks: List["Chunk"]) -> List["Chunk"]:
        """
        Add entity_links to chunks based on co-occurrence.

        For each chunk with entities, creates pairwise links
        between all entities in that chunk.

        Args:
            chunks: List of chunks with entities in metadata

        Returns:
            Same chunks with entity_links added to metadata
        """
        for chunk in chunks:
            entities = chunk.metadata.get("entities", [])
            if len(entities) < 2:
                continue

            links = []
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    links.append(
                        EntityLink(
                            source=e1["name"],
                            target=e2["name"],
                            source_type=e1["type"],
                            target_type=e2["type"],
                            chunk_id=chunk.id,
                        ).to_dict()
                    )

            chunk.metadata["entity_links"] = links

        return chunks


__all__ = ["EntityLink", "EntityLinker"]
