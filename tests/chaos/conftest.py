# tests/chaos/conftest.py
"""
Chaos test fixtures.

Note: e2e fixtures (e2e_runner) are imported via tests/conftest.py
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    """Add tier4 and chaos markers to all tests in this directory."""
    for item in items:
        if "/chaos/" in str(item.fspath) or "\\chaos\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.chaos)
