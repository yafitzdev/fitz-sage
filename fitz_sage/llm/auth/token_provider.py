# fitz_sage/llm/auth/token_provider.py
"""Token provider adapter for OpenAI SDK azure_ad_token_provider pattern."""

from __future__ import annotations

from fitz_sage.llm.auth.base import AuthProvider


class TokenProviderAdapter:
    """
    Adapts AuthProvider to OpenAI SDK's azure_ad_token_provider callback pattern.

    The OpenAI SDK accepts a callable that returns a token string. This adapter
    wraps AuthProvider and extracts the bearer token from get_headers().

    Usage:
        auth_provider = M2MAuth(...)
        client = openai.AzureOpenAI(azure_ad_token_provider=TokenProviderAdapter(auth_provider))

    Source: https://github.com/openai/openai-python/blob/main/examples/azure_ad.py
    """

    def __init__(self, auth_provider: AuthProvider) -> None:
        self._auth_provider = auth_provider

    def __call__(self) -> str:
        """Called by OpenAI SDK before each request to get fresh token."""
        headers = self._auth_provider.get_headers()
        auth_header = headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            error_preview = f"{auth_header[:20]}..." if auth_header else "No Authorization header"
            raise ValueError(
                f"TokenProviderAdapter requires AuthProvider with Bearer token. "
                f"Got: {error_preview}"
            )

        return auth_header[7:]  # Strip "Bearer " prefix
