# fitz_ai/retrieval/vocabulary/models.py
"""
Data models for keyword vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Keyword:
    """
    A detected keyword with its variations.

    Attributes:
        id: Unique identifier for this keyword (e.g., "TC-1001")
        category: Category of the keyword (e.g., "testcase", "ticket", "version")
        match: List of variations that match this keyword
        occurrences: Number of times this keyword appears in the corpus
        first_seen: Source file where the keyword was first detected
        user_defined: Whether this keyword was manually added by user
        auto_generated: Original variations auto-generated (preserved for merge)
    """

    id: str
    category: str
    match: list[str]
    occurrences: int = 1
    first_seen: str | None = None
    user_defined: bool = False
    auto_generated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        data: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "match": self.match,
            "occurrences": self.occurrences,
        }
        if self.first_seen:
            data["first_seen"] = self.first_seen
        if self.user_defined:
            data["user_defined"] = True
        if self.auto_generated:
            data["_auto_generated"] = self.auto_generated
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Keyword":
        """Create from dictionary (YAML deserialization)."""
        return cls(
            id=data["id"],
            category=data["category"],
            match=data.get("match", []),
            occurrences=data.get("occurrences", 1),
            first_seen=data.get("first_seen"),
            user_defined=data.get("user_defined", False),
            auto_generated=data.get("_auto_generated", []),
        )

    def add_variation(self, variation: str) -> None:
        """Add a variation if not already present."""
        if variation not in self.match:
            self.match.append(variation)

    def matches_text(self, text: str) -> bool:
        """Check if any variation matches in the text (case-insensitive)."""
        text_lower = text.lower()
        return any(var.lower() in text_lower for var in self.match)


@dataclass
class VocabularyConfig:
    """
    Configuration for vocabulary detection and matching.

    Attributes:
        enabled: Whether vocabulary detection is enabled
        detect_on_ingest: Auto-detect keywords during ingestion
        min_occurrences: Minimum occurrences to include a keyword
        categories: Which categories to detect
    """

    enabled: bool = True
    detect_on_ingest: bool = True
    min_occurrences: int = 1
    categories: list[str] = field(
        default_factory=lambda: [
            "testcase",
            "ticket",
            "version",
            "pull_request",
            "person",
            "file",
            "internal_id",
        ]
    )


@dataclass
class VocabularyMetadata:
    """
    Metadata for the vocabulary file.

    Attributes:
        generated: Timestamp when vocabulary was generated
        source_docs: Number of source documents scanned
        auto_detected: Number of auto-detected keywords
        user_modified: Number of user-modified keywords
    """

    generated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_docs: int = 0
    auto_detected: int = 0
    user_modified: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "generated": self.generated.isoformat(),
            "source_docs": self.source_docs,
            "auto_detected": self.auto_detected,
            "user_modified": self.user_modified,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VocabularyMetadata":
        """Create from dictionary (YAML deserialization)."""
        generated = data.get("generated")
        if isinstance(generated, str):
            generated = datetime.fromisoformat(generated)
        elif not isinstance(generated, datetime):
            generated = datetime.now(timezone.utc)

        return cls(
            generated=generated,
            source_docs=data.get("source_docs", 0),
            auto_detected=data.get("auto_detected", 0),
            user_modified=data.get("user_modified", 0),
        )
