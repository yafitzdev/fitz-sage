# fitz_ai/llm/auth/composite.py
"""Composite auth for multi-header scenarios (BMW enterprise gateway)."""
from __future__ import annotations

from typing import Any

from fitz_ai.llm.auth.base import AuthProvider


class CompositeAuth(AuthProvider):
    """
    Combines multiple AuthProviders into single header set.

    Use case: BMW gateway requires both:
    - Authorization: Bearer <M2M-token> (dynamic, refreshed)
    - X-API-Key: <BMW-LLM-API-key> (static)

    Example:
        m2m = M2MAuth(token_url=..., client_id=..., client_secret=...)
        api_key = ApiKeyAuth("BMW_LLM_API_KEY", header_format="x-api-key")
        composite = CompositeAuth(m2m, api_key)
        # composite.get_headers() returns both headers
    """

    def __init__(self, *providers: AuthProvider) -> None:
        if not providers:
            raise ValueError("CompositeAuth requires at least one provider")
        self._providers = providers

    def get_headers(self) -> dict[str, str]:
        """Merge headers from all providers (later providers override earlier)."""
        headers: dict[str, str] = {}
        for provider in self._providers:
            headers.update(provider.get_headers())
        return headers

    def get_request_kwargs(self) -> dict[str, Any]:
        """Merge request kwargs from all providers."""
        kwargs: dict[str, Any] = {}
        for provider in self._providers:
            kwargs.update(provider.get_request_kwargs())
        return kwargs
