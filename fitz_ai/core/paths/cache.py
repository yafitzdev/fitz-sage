# fitz_ai/core/paths/cache.py
"""Cache and knowledge map paths."""

from __future__ import annotations

from pathlib import Path

from .workspace import workspace


def cache() -> Path:
    """
    Cache directory root.

    Location: {workspace}/cache/
    """
    return workspace() / "cache"


def knowledge_map() -> Path:
    """
    Knowledge map state file for visualization caching.

    Location: {workspace}/knowledge_map.json
    """
    return workspace() / "knowledge_map.json"


def knowledge_map_html() -> Path:
    """
    Default output path for knowledge map HTML visualization.

    Location: {workspace}/knowledge_map.html
    """
    return workspace() / "knowledge_map.html"
