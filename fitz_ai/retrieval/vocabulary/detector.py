# fitz_ai/retrieval/vocabulary/detector.py
"""
Pattern-based keyword detector for auto-discovering identifiers in content.

Scans chunks during ingestion to find identifier patterns like:
- Test cases: TC-1001, testcase_42
- Tickets: JIRA-4521, BUG-789
- Versions: v2.0.1, 1.0.0-beta
- Pull requests: PR #123, PR-456
- People: John Smith, Jane Doe
- Files: config.yaml, report.pdf
- Internal IDs: ORD-12345, USR-98765
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from fitz_ai.logging.logger import get_logger

from .models import Keyword
from .variations import generate_variations

if TYPE_CHECKING:
    from fitz_ai.core.chunk import ChunkLike

logger = get_logger(__name__)


@dataclass
class DetectorPattern:
    """A pattern for detecting keywords."""

    pattern: str
    category: str
    description: str
    min_occurrences: int = 1  # Minimum times it must appear to be included

    def compile(self) -> re.Pattern:
        """Compile the regex pattern."""
        return re.compile(self.pattern, re.IGNORECASE)


# Default patterns for keyword detection
DEFAULT_PATTERNS: list[DetectorPattern] = [
    # Test cases
    DetectorPattern(
        pattern=r"\b(TC[_\-]?\d+)\b",
        category="testcase",
        description="Test case IDs (TC-1001, TC_42)",
    ),
    DetectorPattern(
        pattern=r"\b(testcase[_\-\s]?\d+)\b",
        category="testcase",
        description="Testcase with number (testcase 1001)",
    ),
    # Tickets / Issues
    DetectorPattern(
        pattern=r"\b([A-Z]{2,5}-\d{3,})\b",
        category="ticket",
        description="Issue tracker IDs (JIRA-4521, BUG-789)",
    ),
    # Pull requests
    DetectorPattern(
        pattern=r"\b(PR[_\-\s#]?\d+)\b",
        category="pull_request",
        description="Pull request IDs (PR #123, PR-456)",
    ),
    DetectorPattern(
        pattern=r"\b(pull\s*request[_\-\s#]?\d+)\b",
        category="pull_request",
        description="Pull request spelled out",
    ),
    # Versions
    DetectorPattern(
        pattern=r"\b(v\d+\.\d+(?:\.\d+)?(?:-\w+)?)\b",
        category="version",
        description="Version numbers with v prefix (v2.0.1)",
    ),
    DetectorPattern(
        pattern=r"\b(\d+\.\d+\.\d+(?:-\w+)?)\b",
        category="version",
        description="Semantic versions (1.0.0-beta)",
        min_occurrences=2,  # Require multiple mentions to avoid false positives
    ),
    # People (only if mentioned multiple times)
    DetectorPattern(
        pattern=r"\b([A-Z][a-z]+\s[A-Z][a-z]+)\b",
        category="person",
        description="Person names (John Smith)",
        min_occurrences=2,  # People must be mentioned multiple times
    ),
    # Files
    DetectorPattern(
        pattern=r"\b(\w+\.(?:yaml|yml|json|py|md|txt|pdf|docx?|xlsx?))\b",
        category="file",
        description="File names (config.yaml, report.pdf)",
    ),
    # Internal IDs (generic pattern for uppercase prefix + numbers)
    DetectorPattern(
        pattern=r"\b([A-Z]{2,4}\d{4,})\b",
        category="internal_id",
        description="Internal IDs (ORD12345, USR98765)",
        min_occurrences=2,
    ),
]


@dataclass
class DetectionResult:
    """Result of keyword detection."""

    keyword_id: str
    category: str
    occurrences: int
    first_seen: str | None
    locations: list[str] = field(default_factory=list)


class KeywordDetector:
    """
    Detects keywords in chunk content using regex patterns.

    Usage:
        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)
    """

    def __init__(
        self,
        patterns: list[DetectorPattern] | None = None,
        min_occurrences: int = 1,
    ):
        """
        Initialize the detector.

        Args:
            patterns: Custom patterns to use (defaults to DEFAULT_PATTERNS)
            min_occurrences: Global minimum occurrences override
        """
        self.patterns = patterns or DEFAULT_PATTERNS
        self.min_occurrences = min_occurrences
        self._compiled = [(p, p.compile()) for p in self.patterns]

    def detect_from_chunks(
        self,
        chunks: Sequence["ChunkLike"],
    ) -> list[Keyword]:
        """
        Detect keywords from a list of chunks.

        Args:
            chunks: Chunks to scan for keywords

        Returns:
            List of detected keywords with variations
        """
        # Track detections: keyword_id → DetectionResult
        detections: dict[str, DetectionResult] = defaultdict(
            lambda: DetectionResult(
                keyword_id="",
                category="",
                occurrences=0,
                first_seen=None,
            )
        )

        for chunk in chunks:
            content = chunk.content
            source = chunk.metadata.get("source_file", chunk.doc_id)

            for pattern_spec, compiled in self._compiled:
                matches = compiled.findall(content)
                for match in matches:
                    # Normalize the match
                    keyword_id = match.strip()
                    key = keyword_id.lower()

                    if not detections[key].keyword_id:
                        detections[key] = DetectionResult(
                            keyword_id=keyword_id,
                            category=pattern_spec.category,
                            occurrences=0,
                            first_seen=source,
                        )

                    detections[key].occurrences += 1
                    if source and source not in detections[key].locations:
                        detections[key].locations.append(source)

        # Convert to Keywords, filtering by min_occurrences
        keywords: list[Keyword] = []
        for key, detection in detections.items():
            # Get pattern-specific min_occurrences
            pattern_min = self.min_occurrences
            for pattern_spec, _ in self._compiled:
                if pattern_spec.category == detection.category:
                    pattern_min = max(pattern_min, pattern_spec.min_occurrences)
                    break

            if detection.occurrences >= pattern_min:
                # Generate variations for this keyword
                variations = generate_variations(
                    detection.keyword_id,
                    detection.category,
                )

                keyword = Keyword(
                    id=detection.keyword_id,
                    category=detection.category,
                    match=variations,
                    occurrences=detection.occurrences,
                    first_seen=detection.first_seen,
                    auto_generated=variations.copy(),
                )
                keywords.append(keyword)

        logger.info(f"[VOCABULARY] Detected {len(keywords)} keywords from {len(chunks)} chunks")

        # Sort by occurrences (most common first)
        keywords.sort(key=lambda k: (-k.occurrences, k.id))

        return keywords

    def detect_from_text(self, text: str, source: str | None = None) -> list[Keyword]:
        """
        Detect keywords from raw text.

        Args:
            text: Text to scan
            source: Optional source identifier

        Returns:
            List of detected keywords
        """

        # Create a simple chunk-like object
        @dataclass
        class _TextChunk:
            content: str
            doc_id: str
            metadata: dict

        chunk = _TextChunk(content=text, doc_id=source or "text", metadata={})
        return self.detect_from_chunks([chunk])  # type: ignore


def suggest_keywords(
    chunks: Sequence["ChunkLike"],
    min_occurrences: int = 2,
) -> list[Keyword]:
    """
    Suggest keywords from chunks with higher occurrence threshold.

    This is useful for CLI suggestions where we want to show
    only frequently occurring patterns.

    Args:
        chunks: Chunks to scan
        min_occurrences: Minimum occurrences to suggest

    Returns:
        List of suggested keywords
    """
    detector = KeywordDetector(min_occurrences=min_occurrences)
    return detector.detect_from_chunks(chunks)
