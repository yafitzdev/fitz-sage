# fitz_ai/llm/auth/httpx_auth.py
"""httpx.Auth wrapper for dynamic token injection."""
from __future__ import annotations

import typing

import httpx

from fitz_ai.llm.auth.base import AuthProvider


class DynamicHttpxAuth(httpx.Auth):
    """
    httpx.Auth implementation that injects fresh tokens per-request.

    Wraps an AuthProvider and calls get_headers() for each request,
    enabling automatic token refresh for M2M authentication.

    Usage:
        auth_provider = M2MAuth(...)
        http_client = httpx.Client(auth=DynamicHttpxAuth(auth_provider))

    Source: https://www.python-httpx.org/advanced/authentication/
    """

    def __init__(self, auth_provider: AuthProvider) -> None:
        self._auth_provider = auth_provider

    def auth_flow(
        self, request: httpx.Request
    ) -> typing.Generator[httpx.Request, httpx.Response, None]:
        """Inject fresh auth headers into each request."""
        headers = self._auth_provider.get_headers()
        for key, value in headers.items():
            request.headers[key] = value
        yield request
