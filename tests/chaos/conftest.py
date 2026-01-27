# tests/chaos/conftest.py
"""
Chaos test fixtures.

Imports e2e fixtures directly (not via root conftest to avoid parallel execution issues).
"""

from __future__ import annotations

import pytest

# Import e2e fixtures for chaos tests (these run serially, not in parallel)
from tests.e2e.conftest import *  # noqa: F401, F403


def pytest_collection_modifyitems(items):
    """Add tier4 and chaos markers to all tests in this directory."""
    for item in items:
        if "/chaos/" in str(item.fspath) or "\\chaos\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.chaos)
