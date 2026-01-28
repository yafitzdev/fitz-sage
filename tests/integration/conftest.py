# tests/integration/conftest.py
"""
Integration test fixtures.

These tests require real services (postgres, cloud APIs, etc.)

Cloud Cache E2E Tests:
    Require environment variables:
    - FITZ_CLOUD_TEST_API_KEY: API key for test organization
    - FITZ_CLOUD_TEST_ORG_KEY: 64-character hex encryption key
    - FITZ_CLOUD_TEST_ORG_ID: UUID of test organization
    - FITZ_CLOUD_URL: Cloud API base URL (optional, defaults to localhost:8000)
"""

from __future__ import annotations

import pytest

# Import cloud fixtures to make them available to tests
from .cloud_fixtures import (
    cache_versions,
    cloud_available,
    cloud_client,
    cloud_config,
    cloud_org_id,
    cloud_pipeline,
    test_queries,
    unique_collection_name,
)

# Re-export fixtures for pytest discovery
__all__ = [
    "cache_versions",
    "cloud_available",
    "cloud_client",
    "cloud_config",
    "cloud_org_id",
    "cloud_pipeline",
    "test_queries",
    "unique_collection_name",
]


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "cloud: marks tests that require cloud connection")


def pytest_collection_modifyitems(items):
    """Add tier3 and integration markers to all tests in this directory."""
    for item in items:
        if "/integration/" in str(item.fspath) or "\\integration\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier3)
            item.add_marker(pytest.mark.integration)
