# fitz_ai/core/constants.py
"""
Core constants for the fitz-ai library.

Centralizes magic strings and numbers to improve maintainability
and provide self-documenting code.
"""

from __future__ import annotations

import uuid

# =============================================================================
# Ollama Service Constants
# =============================================================================

OLLAMA_DEFAULT_HOST = "localhost"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_HEALTH_TIMEOUT = 2.0  # seconds
OLLAMA_API_TAGS_PATH = "/api/tags"


def ollama_url(host: str = OLLAMA_DEFAULT_HOST, port: int = OLLAMA_DEFAULT_PORT) -> str:
    """Construct Ollama base URL from host and port."""
    return f"http://{host}:{port}"


# =============================================================================
# UUID Namespace
# =============================================================================

# DNS namespace UUID (RFC 4122) for deterministic UUID5 generation
# Used for converting string IDs to UUIDs in a reproducible way
UUID_NAMESPACE_DNS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


__all__ = [
    "OLLAMA_DEFAULT_HOST",
    "OLLAMA_DEFAULT_PORT",
    "OLLAMA_HEALTH_TIMEOUT",
    "OLLAMA_API_TAGS_PATH",
    "ollama_url",
    "UUID_NAMESPACE_DNS",
]
