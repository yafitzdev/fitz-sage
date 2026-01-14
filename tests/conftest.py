# tests/conftest.py
"""
Root conftest - imports fixtures from all test modules.

This allows tests in any subdirectory to use fixtures from
e2e, unit, or other test modules.
"""

from __future__ import annotations

# Import e2e fixtures so they're available to performance/security/chaos tests
from tests.e2e.conftest import *  # noqa: F401, F403

# Import unit test fixtures
from tests.unit.conftest import *  # noqa: F401, F403
