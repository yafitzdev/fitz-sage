# fitz_sage/llm/auth/m2m.py
"""
M2M OAuth2 client credentials authentication provider.

Supports enterprise deployments with:
- OAuth2 client credentials flow (RFC 6749 Section 4.4)
- Automatic token refresh with exponential backoff
- Circuit breaker protection against token endpoint failures
- Custom CA certificates
- mTLS (mutual TLS) client certificate authentication
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx
from circuitbreaker import circuit
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from fitz_sage.llm.auth.certificates import validate_certificate_file, validate_key_file

logger = logging.getLogger(__name__)

# Transient network errors that should be retried
# Do NOT include HTTPStatusError - 401/403 are permanent failures
TRANSIENT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


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
        client_cert_path: Path to client certificate for mTLS authentication.
        client_key_path: Path to client private key for mTLS authentication.
        client_key_password: Password for encrypted client key (or env var with ${VAR} syntax).
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        cert_path: str | None = None,
        scope: str | None = None,
        refresh_margin_seconds: int = 60,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
        client_key_password: str | None = None,
    ) -> None:
        self.token_url = token_url
        self.client_id = self._resolve_env_var(client_id)
        self.client_secret = self._resolve_env_var(client_secret)
        self.cert_path = cert_path
        self.scope = scope
        self.refresh_margin_seconds = refresh_margin_seconds

        # Resolve client key password (may use ${VAR} syntax)
        resolved_key_password = (
            self._resolve_env_var(client_key_password) if client_key_password else None
        )

        # Validate certificates at init time (fail fast with actionable errors)
        if cert_path:
            validate_certificate_file(cert_path, "CA certificate")
        if client_cert_path:
            validate_certificate_file(client_cert_path, "Client certificate")
        if client_key_path:
            validate_key_file(client_key_path, "Client key", resolved_key_password)

        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.client_key_password = resolved_key_password

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

    @retry(
        wait=wait_exponential_jitter(initial=1, max=60, jitter=1),  # 1s, 2s, 4s... up to 60s
        stop=stop_after_attempt(5),  # Give up after 5 attempts
        retry=retry_if_exception_type(TRANSIENT_ERRORS),  # Only retry transient errors
        before_sleep=before_sleep_log(logger, logging.WARNING),  # Log retries at WARNING
    )
    @circuit(
        failure_threshold=5,  # Open circuit after 5 consecutive failures
        recovery_timeout=30,  # Allow test request after 30s (half-open state)
        expected_exception=TRANSIENT_ERRORS,  # Only count transient errors as failures
    )
    def _refresh_token(self) -> None:
        """Fetch a new access token from the token endpoint.

        Resilience:
        - Retries transient errors (timeout, network) with exponential backoff
        - Circuit breaker prevents retry storms on sustained outages
        - Non-transient errors (401, 403) fail immediately without retry
        """
        verify: bool | str = self.cert_path if self.cert_path else True

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope

        with httpx.Client(verify=verify, timeout=10.0) as client:
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
        logger.debug("Token refreshed successfully, expires in %d seconds", expires_in)

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
        """Return additional request kwargs for certificate verification and mTLS."""
        kwargs: dict[str, Any] = {}

        if self.cert_path:
            kwargs["verify"] = self.cert_path

        if self.client_cert_path:
            if self.client_key_path:
                if self.client_key_password:
                    kwargs["cert"] = (
                        self.client_cert_path,
                        self.client_key_path,
                        self.client_key_password,
                    )
                else:
                    kwargs["cert"] = (self.client_cert_path, self.client_key_path)
            else:
                kwargs["cert"] = self.client_cert_path

        return kwargs
