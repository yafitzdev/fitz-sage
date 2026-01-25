# tests/conftest.py
"""
Root conftest - imports fixtures from all test modules.

All tests use the centralized tests/test_config.yaml.
Config structure matches .fitz/config/ format.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

# =============================================================================
# pgdata Reset (prevents corruption from interrupted test runs)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


def pytest_configure(config):
    """Reset pgdata before any tests run to avoid corruption issues."""
    pgdata_path = PROJECT_ROOT / ".fitz" / "pgdata"
    if pgdata_path.exists():
        shutil.rmtree(pgdata_path, ignore_errors=True)

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

# Import e2e fixtures so they're available to performance/security/chaos tests
from tests.e2e.conftest import *  # noqa: E402, F401, F403

# Import unit test fixtures
from tests.unit.conftest import *  # noqa: E402, F401, F403
