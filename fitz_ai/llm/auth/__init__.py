# fitz_ai/llm/auth/__init__.py
"""
Authentication providers for LLM clients.

Supports:
- API key authentication (simple, env var based)
- M2M OAuth2 client credentials (enterprise, auto-refresh)
"""

from fitz_ai.llm.auth.api_key import ApiKeyAuth
from fitz_ai.llm.auth.base import AuthProvider
from fitz_ai.llm.auth.m2m import M2MAuth

__all__ = [
    "AuthProvider",
    "ApiKeyAuth",
    "M2MAuth",
]
