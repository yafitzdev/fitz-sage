# tests/unit/conftest.py
"""
Test fixtures for unit tests.

Provides mock embedders for testing semantic matching without
requiring actual embedding API calls.
"""

from __future__ import annotations

from typing import Callable

import pytest

from fitz_ai.core.guardrails import SemanticMatcher

from .mock_embedder import create_deterministic_embedder


@pytest.fixture
def mock_embedder() -> Callable[[str], list[float]]:
    """Fixture providing a deterministic mock embedder."""
    return create_deterministic_embedder()


@pytest.fixture
def semantic_matcher(mock_embedder) -> SemanticMatcher:
    """Fixture providing a SemanticMatcher with mock embedder."""
    return SemanticMatcher(
        embedder=mock_embedder,
        # Thresholds tuned for mock embedder clusters
        causal_threshold=0.70,
        assertion_threshold=0.70,
        query_threshold=0.70,
        conflict_threshold=0.70,
    )
