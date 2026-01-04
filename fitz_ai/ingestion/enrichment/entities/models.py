# fitz_ai/ingestion/enrichment/entities/models.py
"""Entity models for extraction results."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Entity:
    """
    An extracted entity from content.

    Attributes:
        name: Entity name (e.g., "UserAuthService", "John Smith")
        type: Entity type (e.g., "class", "function", "person", "organization")
        description: Brief context about the entity from the source content
    """

    name: str
    type: str
    description: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# Standard entity types
DOMAIN_ENTITY_TYPES = frozenset(
    {
        "class",
        "function",
        "method",
        "api",
        "module",
        "config",
        "system",
        "service",
        "endpoint",
    }
)

NAMED_ENTITY_TYPES = frozenset(
    {
        "person",
        "organization",
        "location",
        "date",
        "product",
        "concept",
    }
)

ALL_ENTITY_TYPES = DOMAIN_ENTITY_TYPES | NAMED_ENTITY_TYPES
