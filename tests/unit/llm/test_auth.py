# tests/unit/llm/test_auth.py
"""
Unit tests for LLM authentication providers.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.llm.auth import ApiKeyAuth, AuthProvider, M2MAuth


class TestAuthProvider:
    """Test AuthProvider protocol."""

    def test_api_key_auth_is_auth_provider(self) -> None:
        """ApiKeyAuth implements AuthProvider protocol."""
        with patch.dict("os.environ", {"TEST_KEY": "secret"}):
            auth = ApiKeyAuth("TEST_KEY")
            assert isinstance(auth, AuthProvider)

    def test_m2m_auth_is_auth_provider(self) -> None:
        """M2MAuth implements AuthProvider protocol."""
        with patch.dict("os.environ", {"CLIENT_ID": "id", "CLIENT_SECRET": "secret"}):
            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="${CLIENT_ID}",
                client_secret="${CLIENT_SECRET}",
            )
            assert isinstance(auth, AuthProvider)


class TestApiKeyAuth:
    """Test API key authentication."""

    def test_bearer_format(self) -> None:
        """Bearer format produces correct header."""
        with patch.dict("os.environ", {"MY_API_KEY": "test-key-123"}):
            auth = ApiKeyAuth("MY_API_KEY", header_format="bearer")
            headers = auth.get_headers()
            assert headers == {"Authorization": "Bearer test-key-123"}

    def test_x_api_key_format(self) -> None:
        """X-Api-Key format produces correct header."""
        with patch.dict("os.environ", {"MY_API_KEY": "test-key-123"}):
            auth = ApiKeyAuth("MY_API_KEY", header_format="x-api-key")
            headers = auth.get_headers()
            assert headers == {"X-Api-Key": "test-key-123"}

    def test_basic_format(self) -> None:
        """Basic format produces correct header."""
        with patch.dict("os.environ", {"MY_API_KEY": "test-key-123"}):
            auth = ApiKeyAuth("MY_API_KEY", header_format="basic")
            headers = auth.get_headers()
            assert headers == {"Authorization": "Basic test-key-123"}

    def test_missing_env_var_raises(self) -> None:
        """Missing environment variable raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            auth = ApiKeyAuth("NONEXISTENT_KEY")
            with pytest.raises(ValueError, match="NONEXISTENT_KEY not set"):
                auth.get_headers()

    def test_lazy_loading(self) -> None:
        """API key is loaded lazily on first access."""
        with patch.dict("os.environ", {"LAZY_KEY": "lazy-value"}):
            auth = ApiKeyAuth("LAZY_KEY")
            assert auth._api_key is None
            _ = auth.get_headers()
            assert auth._api_key == "lazy-value"

    def test_request_kwargs_empty(self) -> None:
        """API key auth has no additional request kwargs."""
        with patch.dict("os.environ", {"MY_KEY": "value"}):
            auth = ApiKeyAuth("MY_KEY")
            assert auth.get_request_kwargs() == {}


class TestM2MAuth:
    """Test M2M OAuth2 authentication."""

    def test_env_var_resolution(self) -> None:
        """Environment variables are resolved from ${VAR} syntax."""
        with patch.dict("os.environ", {"MY_ID": "resolved-id", "MY_SECRET": "resolved-secret"}):
            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="${MY_ID}",
                client_secret="${MY_SECRET}",
            )
            assert auth.client_id == "resolved-id"
            assert auth.client_secret == "resolved-secret"

    def test_literal_values_unchanged(self) -> None:
        """Literal values (not ${VAR}) are used as-is."""
        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="literal-id",
            client_secret="literal-secret",
        )
        assert auth.client_id == "literal-id"
        assert auth.client_secret == "literal-secret"

    def test_missing_env_var_raises(self) -> None:
        """Missing environment variable raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MISSING_VAR not set"):
                M2MAuth(
                    token_url="https://auth.example.com/token",
                    client_id="${MISSING_VAR}",
                    client_secret="secret",
                )

    def test_token_refresh(self) -> None:
        """Token is fetched from token endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-token-abc",
            "expires_in": 3600,
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
            )
            headers = auth.get_headers()

            assert headers == {"Authorization": "Bearer new-token-abc"}
            mock_client.return_value.__enter__.return_value.post.assert_called_once()

    def test_token_cached(self) -> None:
        """Token is cached and reused."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "cached-token",
            "expires_in": 3600,
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
            )

            # First call fetches token
            auth.get_headers()
            # Second call uses cached token
            auth.get_headers()

            # Only one HTTP call
            assert mock_client.return_value.__enter__.return_value.post.call_count == 1

    def test_token_refresh_on_expiry(self) -> None:
        """Token is refreshed when expired."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 1,  # Expires in 1 second
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
                refresh_margin_seconds=0,
            )

            auth.get_headers()
            time.sleep(1.1)  # Wait for expiry
            auth.get_headers()

            # Two HTTP calls due to refresh
            assert mock_client.return_value.__enter__.return_value.post.call_count == 2

    def test_scope_included_in_request(self) -> None:
        """Scope is included in token request when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "token", "expires_in": 3600}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
                scope="read write",
            )
            auth.get_headers()

            call_kwargs = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_kwargs[1]["data"]["scope"] == "read write"

    def test_cert_path_in_request_kwargs(self) -> None:
        """Certificate path is returned in request kwargs."""
        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="my-client",
            client_secret="my-secret",
            cert_path="/etc/ssl/corp-ca.crt",
        )
        kwargs = auth.get_request_kwargs()
        assert kwargs == {"verify": "/etc/ssl/corp-ca.crt"}

    def test_no_cert_path_empty_kwargs(self) -> None:
        """No cert path means empty request kwargs."""
        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="my-client",
            client_secret="my-secret",
        )
        assert auth.get_request_kwargs() == {}

    def test_thread_safety(self) -> None:
        """Token refresh is thread-safe."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "thread-safe-token", "expires_in": 3600}

        call_count = 0
        lock = threading.Lock()

        def mock_post(*args, **kwargs):
            nonlocal call_count
            with lock:
                call_count += 1
            time.sleep(0.01)  # Simulate network delay
            return mock_response

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post = mock_post

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
            )

            threads = [threading.Thread(target=auth.get_headers) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Only one token refresh despite concurrent access
            assert call_count == 1
