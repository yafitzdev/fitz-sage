# fitz_ai/retrieval/vocabulary/__init__.py
"""
Auto-detected keyword vocabulary for exact matching.

This module provides:
- Pattern detection during ingestion (auto-discover identifiers)
- Variation generation (TC-1001 → testcase_1001, testcase 1001, etc.)
- Vocabulary storage (keywords.yaml)
- Query-time matching (find keywords in queries, filter chunks)

Usage:
    from fitz_ai.retrieval.vocabulary import (
        VocabularyStore,
        KeywordDetector,
        KeywordMatcher,
    )

    # During ingestion
    detector = KeywordDetector()
    keywords = detector.detect_from_chunks(chunks)
    store = VocabularyStore()
    store.merge_and_save(keywords)

    # During query
    matcher = KeywordMatcher(store.load())
    filtered_chunks = matcher.filter_chunks(query, chunks)
"""

from .detector import KeywordDetector
from .matcher import KeywordMatcher, create_matcher_from_store
from .models import Keyword, VocabularyConfig
from .store import VocabularyStore
from .variations import generate_variations

__all__ = [
    "KeywordDetector",
    "KeywordMatcher",
    "VocabularyStore",
    "Keyword",
    "VocabularyConfig",
    "generate_variations",
    "create_matcher_from_store",
]
