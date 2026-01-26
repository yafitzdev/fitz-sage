# tests/load/conftest.py
"""
Load/scalability test fixtures.

Note: e2e fixtures (e2e_runner) are imported via tests/conftest.py
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    """Add tier4 and scalability markers to all tests in this directory."""
    for item in items:
        if "/load/" in str(item.fspath) or "\\load\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.scalability)
