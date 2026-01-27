# tests/load/conftest.py
"""
Load/scalability test fixtures.

Imports e2e fixtures directly (not via root conftest to avoid parallel execution issues).
"""

from __future__ import annotations

import pytest

# Import e2e fixtures for load tests (these run serially, not in parallel)
from tests.e2e.conftest import *  # noqa: F401, F403


def pytest_collection_modifyitems(items):
    """Add tier4 and scalability markers to all tests in this directory."""
    for item in items:
        if "/load/" in str(item.fspath) or "\\load\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.scalability)
