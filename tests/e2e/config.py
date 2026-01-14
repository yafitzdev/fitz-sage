# tests/e2e/config.py
"""E2E test configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

E2E_CONFIG_PATH = Path(__file__).parent / "e2e_config.yaml"


def load_e2e_config() -> dict[str, Any]:
    """Load e2e test configuration."""
    with open(E2E_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_tier_names(config: dict[str, Any] | None = None) -> list[str]:
    """Get list of tier names in order."""
    if config is None:
        config = load_e2e_config()
    return [tier["name"] for tier in config.get("tiers", [])]


def get_tier_config(tier_name: str, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build full config dict for a specific tier.

    Args:
        tier_name: Name of the tier (e.g., "local", "cloud")
        base_config: Base e2e config (loaded if not provided)

    Returns:
        Complete config dict ready for FitzRagConfig.from_dict()
    """
    if base_config is None:
        base_config = load_e2e_config()

    tiers = base_config.get("tiers", [])
    tier = next((t for t in tiers if t["name"] == tier_name), None)
    if not tier:
        available = [t["name"] for t in tiers]
        raise ValueError(f"Unknown tier: {tier_name}. Available: {available}")

    return {
        "chat": tier["chat"],
        "embedding": base_config["embedding"],
        "vector_db": base_config["vector_db"],
        "retrieval": {
            "plugin_name": "dense",
            "collection": None,  # Set by runner
            "top_k": 20,
        },
        "multihop": {"max_hops": 2},
        "rgs": {
            "strict_grounding": False,
            "max_chunks": 50,
        },
    }


def get_cache_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get cache configuration."""
    if config is None:
        config = load_e2e_config()
    return config.get("cache", {"enabled": False, "max_entries": 1000, "ttl_days": 30})
