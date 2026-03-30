# fitz_sage/llm/auth/base.py
"""
Authentication provider protocol for LLM clients.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthProvider(Protocol):
    """
    Protocol for authentication providers.

    Implementations provide headers and request kwargs for HTTP clients.
    """

    def get_headers(self) -> dict[str, str]:
        """Return authentication headers to include in requests."""
        ...

    def get_request_kwargs(self) -> dict[str, Any]:
        """Return additional request kwargs (e.g., cert paths, timeouts)."""
        ...
