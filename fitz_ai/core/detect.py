# fitz_ai/core/detect.py
"""
Centralized service detection for Fitz.

This module provides auto-discovery of external services (Ollama, etc.)
and is used by:
- CLI commands (doctor, init, quickstart)
- LLM plugins (auto-detection of Ollama)

Usage:
    from fitz_ai.core.detect import detect_ollama, detect_api_key

    # Check Ollama
    ollama = detect_ollama()
    if ollama.available:
        print(f"Ollama at {ollama.host}:{ollama.port}")

    # Check API keys
    cohere = detect_api_key("cohere")
    if cohere.available:
        print(f"Cohere key set: {cohere.details}")
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ServiceStatus:
    """Status of a detected service."""

    name: str
    available: bool
    host: Optional[str] = None
    port: Optional[int] = None
    details: str = ""


@dataclass
class ApiKeyStatus:
    """Status of an API key."""

    name: str
    available: bool
    env_var: str = ""
    details: str = ""


@dataclass
class SystemStatus:
    """Complete system status."""

    ollama: ServiceStatus
    pgvector: ServiceStatus
    api_keys: dict[str, ApiKeyStatus]

    @property
    def best_llm(self) -> str:
        """Return the best available LLM provider."""
        if self.api_keys["cohere"].available:
            return "cohere"
        if self.api_keys["openai"].available:
            return "openai"
        if self.api_keys["anthropic"].available:
            return "anthropic"
        if self.ollama.available:
            return "ollama"
        return "ollama"  # Default fallback

    @property
    def best_embedding(self) -> str:
        """Return the best available embedding provider."""
        if self.api_keys["cohere"].available:
            return "cohere"
        if self.api_keys["openai"].available:
            return "openai"
        if self.ollama.available:
            return "ollama"
        return "ollama"  # Default fallback

    @property
    def best_vector_db(self) -> str:
        """Return the best available vector database."""
        # pgvector is the only option now
        return "pgvector"

    @property
    def best_rerank(self) -> Optional[str]:
        """Return the best available rerank provider (or None if none available)."""
        # Currently only Cohere supports reranking
        if self.api_keys["cohere"].available:
            return "cohere"
        return None


# =============================================================================
# Network Helpers
# =============================================================================


def _get_local_ip() -> Optional[str]:
    """Get the local machine's IP address on the LAN."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        # Connect to a public IP (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None


# =============================================================================
# Service Detection
# =============================================================================


def detect_ollama() -> ServiceStatus:
    """
    Detect if Ollama is running.

    Checks HTTP endpoint at localhost:11434 and 127.0.0.1:11434.

    Returns:
        ServiceStatus with available=True if Ollama responds
    """
    try:
        import httpx
    except ImportError:
        return ServiceStatus(
            name="Ollama",
            available=False,
            details="httpx not installed",
        )

    from fitz_ai.core.constants import (
        OLLAMA_API_TAGS_PATH,
        OLLAMA_DEFAULT_PORT,
        OLLAMA_HEALTH_TIMEOUT,
    )

    hosts_to_try = ["localhost", "127.0.0.1"]
    port = OLLAMA_DEFAULT_PORT

    for host in hosts_to_try:
        try:
            response = httpx.get(
                f"http://{host}:{port}{OLLAMA_API_TAGS_PATH}",
                timeout=OLLAMA_HEALTH_TIMEOUT,
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "?") for m in models[:3]]

                if model_names:
                    details = f"Models: {', '.join(model_names)}"
                    if len(models) > 3:
                        details += f" (+{len(models) - 3} more)"
                else:
                    details = "No models installed"

                return ServiceStatus(
                    name="Ollama",
                    available=True,
                    host=host,
                    port=port,
                    details=details,
                )
        except Exception as e:
            logger.debug(f"Ollama not found at {host}:{port}: {e}")
            continue

    return ServiceStatus(
        name="Ollama",
        available=False,
        details="Not running (tried localhost:11434)",
    )


def detect_pgvector() -> ServiceStatus:
    """
    Check if pgvector/psycopg is installed.

    Returns:
        ServiceStatus with available=True if psycopg can be imported
    """
    try:
        import pgvector  # noqa: F401
        import psycopg  # noqa: F401

        return ServiceStatus(
            name="pgvector",
            available=True,
            details="Installed (PostgreSQL vector DB)",
        )
    except ImportError:
        return ServiceStatus(
            name="pgvector",
            available=False,
            details="Not installed (pip install psycopg pgvector pgserver)",
        )


def detect_api_key(provider: str) -> ApiKeyStatus:
    """
    Check if an API key is set for a provider.

    Args:
        provider: Provider name (cohere, openai, anthropic)

    Returns:
        ApiKeyStatus with available=True if key is set
    """
    env_var_map = {
        "cohere": "COHERE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    provider_lower = provider.lower()
    env_var = env_var_map.get(provider_lower)

    if not env_var:
        return ApiKeyStatus(
            name=provider,
            available=False,
            env_var="",
            details=f"Unknown provider: {provider}",
        )

    key = os.getenv(env_var)

    if key:
        # Show first 8 chars for verification
        masked = f"{key[:8]}..." if len(key) > 8 else key
        return ApiKeyStatus(
            name=provider.capitalize(),
            available=True,
            env_var=env_var,
            details=f"Set ({masked})",
        )

    return ApiKeyStatus(
        name=provider.capitalize(),
        available=False,
        env_var=env_var,
        details=f"Not set (export {env_var}=...)",
    )


def detect_system_status() -> SystemStatus:
    """Get complete system status."""
    return SystemStatus(
        ollama=detect_ollama(),
        pgvector=detect_pgvector(),
        api_keys={
            "cohere": detect_api_key("cohere"),
            "openai": detect_api_key("openai"),
            "anthropic": detect_api_key("anthropic"),
        },
    )


# =============================================================================
# Connection Helpers
# =============================================================================


def get_ollama_connection() -> tuple[str, int]:
    """
    Get Ollama connection details, auto-detecting if needed.

    Returns:
        (host, port) tuple
    """
    status = detect_ollama()

    if status.available and status.host and status.port:
        return status.host, status.port

    return "localhost", 11434
