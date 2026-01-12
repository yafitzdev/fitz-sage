# fitz_ai/retrieval/vocabulary/matcher.py
"""
Query-time keyword matching.

Detects keywords in user queries and filters chunks to those
containing any variation of the matched keywords.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from fitz_ai.logging.logger import get_logger

from .models import Keyword
from .variations import normalize_for_matching

if TYPE_CHECKING:
    from fitz_ai.core.chunk import ChunkLike

logger = get_logger(__name__)


class KeywordMatcher:
    """
    Matches keywords in queries and filters chunks.

    Usage:
        matcher = KeywordMatcher(keywords)

        # Find keywords in query
        matches = matcher.find_in_query("what happened in TC-1001?")
        # Returns: [Keyword(id="TC-1001", ...)]

        # Filter chunks
        filtered = matcher.filter_chunks("what happened in TC-1001?", chunks)
        # Returns only chunks containing any variation of TC-1001
    """

    def __init__(self, keywords: list[Keyword]):
        """
        Initialize the matcher.

        Args:
            keywords: List of keywords to match against
        """
        self.keywords = keywords

        # Build lookup: normalized variation → Keyword
        self._lookup: dict[str, Keyword] = {}
        for kw in keywords:
            for variation in kw.match:
                normalized = normalize_for_matching(variation)
                # Only store first keyword for each variation (avoid duplicates)
                if normalized not in self._lookup:
                    self._lookup[normalized] = kw

        # Sort variations by length (longest first) for greedy matching
        self._sorted_variations = sorted(
            self._lookup.keys(),
            key=lambda v: -len(v),
        )

    def find_in_query(self, query: str) -> list[Keyword]:
        """
        Find all keywords mentioned in a query.

        Args:
            query: User query string

        Returns:
            List of matched keywords (deduplicated)
        """
        query_normalized = normalize_for_matching(query)
        found: dict[str, Keyword] = {}  # keyword.id → Keyword

        for variation in self._sorted_variations:
            if variation in query_normalized:
                kw = self._lookup[variation]
                if kw.id not in found:
                    found[kw.id] = kw
                    logger.debug(f"[VOCABULARY] Matched keyword: {kw.id} via {variation!r}")

        if found:
            logger.info(f"[VOCABULARY] Found {len(found)} keywords in query: {list(found.keys())}")

        return list(found.values())

    def filter_chunks(
        self,
        query: str,
        chunks: Sequence["ChunkLike"],
    ) -> list["ChunkLike"]:
        """
        Filter chunks to those containing matched keywords.

        If no keywords are found in the query, returns all chunks unchanged.
        If keywords are found, returns only chunks containing ANY variation
        of ALL matched keywords.

        Args:
            query: User query string
            chunks: Chunks to filter

        Returns:
            Filtered chunks (or all chunks if no keywords matched)
        """
        matched_keywords = self.find_in_query(query)

        if not matched_keywords:
            return list(chunks)

        # Filter chunks that contain ALL matched keywords
        filtered = []
        for chunk in chunks:
            if self._chunk_matches_all(chunk, matched_keywords):
                filtered.append(chunk)

        logger.info(
            f"[VOCABULARY] Filtered {len(chunks)} → {len(filtered)} chunks "
            f"(keywords: {[k.id for k in matched_keywords]})"
        )

        return filtered

    def _chunk_matches_all(
        self,
        chunk: "ChunkLike",
        keywords: list[Keyword],
    ) -> bool:
        """Check if chunk contains all keywords."""
        content_normalized = normalize_for_matching(chunk.content)

        for kw in keywords:
            if not self._chunk_matches_keyword(content_normalized, kw):
                return False

        return True

    def _chunk_matches_keyword(
        self,
        content_normalized: str,
        keyword: Keyword,
    ) -> bool:
        """Check if content contains any variation of the keyword."""
        for variation in keyword.match:
            if normalize_for_matching(variation) in content_normalized:
                return True
        return False

    def chunk_matches_any(
        self,
        chunk: "ChunkLike",
        keywords: list[Keyword] | None = None,
    ) -> bool:
        """
        Check if chunk contains any of the keywords.

        Args:
            chunk: Chunk to check
            keywords: Keywords to check (defaults to all keywords)

        Returns:
            True if chunk contains any keyword
        """
        keywords = keywords or self.keywords
        content_normalized = normalize_for_matching(chunk.content)

        for kw in keywords:
            if self._chunk_matches_keyword(content_normalized, kw):
                return True

        return False

    def get_matching_keywords(
        self,
        chunk: "ChunkLike",
    ) -> list[Keyword]:
        """
        Get all keywords that appear in a chunk.

        Args:
            chunk: Chunk to check

        Returns:
            List of keywords found in the chunk
        """
        content_normalized = normalize_for_matching(chunk.content)
        matched = []

        for kw in self.keywords:
            if self._chunk_matches_keyword(content_normalized, kw):
                matched.append(kw)

        return matched


def create_matcher_from_store(collection: str | None = None) -> KeywordMatcher | None:
    """
    Create a KeywordMatcher from the vocabulary store.

    Args:
        collection: Collection name to load vocabulary for

    Returns:
        KeywordMatcher if vocabulary exists, None otherwise
    """
    from .store import VocabularyStore

    store = VocabularyStore(collection=collection)
    if not store.exists():
        return None

    keywords = store.load()
    if not keywords:
        return None

    return KeywordMatcher(keywords)
