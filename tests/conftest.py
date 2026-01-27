# tests/conftest.py
"""
Root conftest - imports fixtures from all test modules.

All tests use the centralized tests/test_config.yaml.
Config structure matches .fitz/config/ format.

Test Tiers (for CI/CD optimization):
=====================================
- tier1: Critical path tests - pure logic, no I/O (<30s)
         Run: pytest -m tier1
- tier2: Unit tests with mocks - no real services (<2min)
         Run: pytest -m "tier1 or tier2"
- tier3: Integration tests - real postgres, embeddings (<10min)
         Run: pytest -m "tier1 or tier2 or tier3"
- tier4: Heavy tests - security, chaos, load, performance (30min+)
         Run: pytest (all tests)

Recommended CI Configuration:
- Every commit:    pytest -m tier1
- PR merge:        pytest -m "tier1 or tier2"
- Merge to main:   pytest -m "tier1 or tier2 or tier3"
- Nightly:         pytest

Feature Markers:
- postgres: Postgres-specific tests (pgvector, table store)
- llm: Tests requiring real LLM API calls
- embeddings: Tests requiring real embedding API calls
- integration: Tests requiring real services
- e2e: End-to-end tests
- slow: Slow tests (>10s)
- security/chaos/performance/scalability: Category markers
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import yaml

# =============================================================================
# Dependency Availability Checks
# =============================================================================


def postgres_deps_available() -> bool:
    """Check if PostgreSQL dependencies (psycopg, pgvector, pgserver) are available."""
    try:
        import pgserver  # noqa: F401
        import pgvector  # noqa: F401
        import psycopg  # noqa: F401
        import psycopg_pool  # noqa: F401

        return True
    except ImportError:
        return False


# Export for use in test files
POSTGRES_DEPS_AVAILABLE = postgres_deps_available()
SKIP_POSTGRES_REASON = "PostgreSQL dependencies (psycopg, pgvector, pgserver) not installed"


# =============================================================================
# pgdata Reset (prevents corruption from interrupted test runs)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


def pytest_configure(config):
    """Reset pgdata before any tests run to ensure clean state.

    Uses production code's _force_remove_pgdata which handles zombie processes.
    Gracefully skips if postgres dependencies aren't installed.
    """
    try:
        from fitz_ai.storage.postgres import _force_remove_pgdata
    except ImportError:
        # Postgres dependencies not installed - skip pgdata cleanup
        return

    pgdata_path = PROJECT_ROOT / ".fitz" / "pgdata"

    if pgdata_path.exists():
        if not _force_remove_pgdata(pgdata_path):
            import warnings

            warnings.warn(f"Could not clean pgdata directory at {pgdata_path}")


# =============================================================================
# Test Configuration
# =============================================================================

TEST_CONFIG_PATH = Path(__file__).parent / "test_config.yaml"


@lru_cache(maxsize=1)
def load_test_config() -> dict:
    """Load the centralized test configuration."""
    with open(TEST_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_test_embedder():
    """Get embedder configured for tests (from first tier)."""
    from fitz_ai.llm.registry import get_llm_plugin

    config = load_test_config()
    # Get embedding config from first tier
    first_tier = config["tiers"][0]
    return get_llm_plugin(
        plugin_type="embedding",
        plugin_name=first_tier["embedding"],
        **first_tier.get("embedding_kwargs", {}),
    )


def get_test_chat(tier: str = "smart"):
    """
    Get chat client configured for tests (local Ollama).

    Args:
        tier: Model tier - "smart", "fast", or "balanced"
    """
    from fitz_ai.llm.registry import get_llm_plugin

    config = load_test_config()
    # Get chat config from first tier (local)
    first_tier = config["tiers"][0]
    return get_llm_plugin(
        plugin_type="chat",
        plugin_name=first_tier["chat"],
        tier=tier,
        **first_tier.get("chat_kwargs", {}),
    )


@pytest.fixture
def test_embedder():
    """Fixture providing the test embedder (local Ollama)."""
    return get_test_embedder()


@pytest.fixture
def test_chat():
    """Fixture providing the test chat client (local Ollama, smart tier)."""
    return get_test_chat("smart")


@pytest.fixture
def test_config():
    """Fixture providing the full test config dict."""
    return load_test_config()


# =============================================================================
# Import fixtures from submodules
# =============================================================================

# Note: E2E fixtures NOT imported here - they have autouse=True session fixtures
# that conflict with pytest-xdist parallel execution. E2E tests get their fixtures
# from tests/e2e/conftest.py directly via pytest's conftest discovery.

# Import unit test fixtures
from tests.unit.conftest import *  # noqa: E402, F401, F403
