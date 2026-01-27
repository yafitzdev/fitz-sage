# tests/unit/conftest.py
"""
Test fixtures for unit tests.

Provides mock embedders for testing semantic matching without
requiring actual embedding API calls.
"""

from __future__ import annotations

import os
from typing import Callable

import pytest

from fitz_ai.core.guardrails import SemanticMatcher

from .mock_embedder import create_deterministic_embedder


def _is_parallel_run() -> bool:
    """Check if running under pytest-xdist with multiple workers."""
    # PYTEST_XDIST_WORKER is set when running under xdist
    return "PYTEST_XDIST_WORKER" in os.environ


def pytest_collection_modifyitems(items):
    """Add tier and postgres markers to unit tests based on type.

    Tier 1 (every commit): Pure logic tests with no I/O or mocks
    Tier 2 (PR merge): Tests with mocks but no real services
    Tier 3+: Already marked in specific files (integration, e2e)

    Postgres marker: Tests that use PostgreSQL (can't run in parallel due to pgserver)
    """
    # Files that should be tier1 (pure logic, fast, no external deps)
    # NOTE: Do NOT include postgres tests here - they can't run in parallel
    TIER1_PATTERNS = [
        "test_answer_mode",
        "test_chunker_id",
        "test_constraints",
        "test_causal_attribution",
        "test_model_tier_resolution",
        "test_query_router",
        "test_semantic_grouping",
        "test_context_pipeline",
        "test_rgs",
        "test_writer_basic",
        # Tabular pure logic
        "tabular/test_models",
        "tabular/test_parser",
        # Structured pure logic
        "structured/test_types",
        "structured/test_formatter",
        "structured/test_router",
        "structured/test_schema",
    ]

    # Files that use PostgreSQL (pgserver) - must run serially
    POSTGRES_PATTERNS = [
        "test_pgvector",
        "test_postgres",
        "test_ingest_executor",
        "test_ingest_state",
        "test_ingest_timing",
        "test_vocabulary",
        "test_entity_graph",
        "test_retrieval_yaml_plugins",
        "test_direct_query",
        "test_vector_search_derived",
    ]

    for item in items:
        fspath = str(item.fspath)

        # Only process tests in unit directory
        if "/unit/" not in fspath and "\\unit\\" not in fspath:
            continue

        # Add postgres marker if file uses postgres
        is_postgres = any(pattern in fspath for pattern in POSTGRES_PATTERNS)
        if is_postgres:
            item.add_marker(pytest.mark.postgres)
            # Auto-skip postgres tests when running in parallel (pgserver can't handle it)
            if _is_parallel_run():
                item.add_marker(pytest.mark.skip(reason="Postgres tests skipped in parallel mode (use -m postgres separately)"))

        # Skip tier marking if already has a tier marker
        has_tier = any(marker.name.startswith("tier") for marker in item.iter_markers())
        if has_tier:
            continue

        # Check if matches tier1 pattern
        is_tier1 = any(pattern in fspath for pattern in TIER1_PATTERNS)

        if is_tier1:
            item.add_marker(pytest.mark.tier1)
        else:
            item.add_marker(pytest.mark.tier2)


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
