# fitz_ai/llm/auth/m2m.py
"""
M2M OAuth2 client credentials authentication provider.

Supports enterprise deployments with:
- OAuth2 client credentials flow (RFC 6749 Section 4.4)
- Automatic token refresh
- Custom CA certificates
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx


class M2MAuth:
    """
    OAuth2 client credentials (M2M) authentication with auto-refresh.

    Args:
        token_url: OAuth2 token endpoint URL.
        client_id: OAuth2 client ID (or env var with ${VAR} syntax).
        client_secret: OAuth2 client secret (or env var with ${VAR} syntax).
        cert_path: Path to CA certificate bundle for enterprise CAs.
        scope: Optional OAuth2 scope.
        refresh_margin_seconds: Refresh token this many seconds before expiry.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        cert_path: str | None = None,
        scope: str | None = None,
        refresh_margin_seconds: int = 60,
    ) -> None:
        self.token_url = token_url
        self.client_id = self._resolve_env_var(client_id)
        self.client_secret = self._resolve_env_var(client_secret)
        self.cert_path = cert_path
        self.scope = scope
        self.refresh_margin_seconds = refresh_margin_seconds

        self._token: str | None = None
        self._expires_at: float = 0
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_env_var(value: str) -> str:
        """Resolve ${VAR} syntax to environment variable value."""
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved = os.environ.get(env_var)
            if not resolved:
                raise ValueError(f"Environment variable {env_var} not set")
            return resolved
        return value

    def _refresh_token(self) -> None:
        """Fetch a new access token from the token endpoint."""
        verify: bool | str = self.cert_path if self.cert_path else True

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope

        with httpx.Client(verify=verify) as client:
            response = client.post(
                self.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()

        self._token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in - self.refresh_margin_seconds

    def _ensure_valid_token(self) -> str:
        """Ensure we have a valid token, refreshing if necessary."""
        with self._lock:
            if self._token is None or time.time() >= self._expires_at:
                self._refresh_token()
            return self._token  # type: ignore[return-value]

    def get_headers(self) -> dict[str, str]:
        """Return authentication headers with valid bearer token."""
        token = self._ensure_valid_token()
        return {"Authorization": f"Bearer {token}"}

    def get_request_kwargs(self) -> dict[str, Any]:
        """Return additional request kwargs for certificate verification."""
        if self.cert_path:
            return {"verify": self.cert_path}
        return {}
