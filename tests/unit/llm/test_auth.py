# tests/unit/llm/test_auth.py
"""
Unit tests for LLM authentication providers.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fitz_ai.llm.auth import ApiKeyAuth, AuthProvider, CompositeAuth, M2MAuth
from fitz_ai.llm.config import resolve_auth


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

    def test_cert_path_in_request_kwargs(self, temp_certificate) -> None:
        """Certificate path is returned in request kwargs."""
        cert_path, _ = temp_certificate
        auth = M2MAuth(
            token_url="https://auth.example.com/token",
            client_id="my-client",
            client_secret="my-secret",
            cert_path=cert_path,
        )
        kwargs = auth.get_request_kwargs()
        assert kwargs == {"verify": cert_path}

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

    def test_retry_on_transient_error(self) -> None:
        """Token refresh retries on transient network errors."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "token", "expires_in": 3600}

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Connection timeout")
            return mock_response

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post = mock_post

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
            )
            headers = auth.get_headers()

            assert headers == {"Authorization": "Bearer token"}
            assert call_count == 3  # Failed twice, succeeded on third

    def test_no_retry_on_auth_error(self) -> None:
        """Token refresh does NOT retry on 401/403 errors."""
        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status_code = 401
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=response
            )
            return response

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post = mock_post

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="wrong-secret",
            )

            with pytest.raises(httpx.HTTPStatusError):
                auth.get_headers()

            # Only one attempt - no retry on auth errors
            assert call_count == 1

    def test_circuit_breaker_opens_after_failures(self) -> None:
        """Circuit breaker opens after consecutive transient failures."""
        from circuitbreaker import CircuitBreakerError, CircuitBreakerMonitor
        from tenacity import RetryError

        # Reset all circuit breakers before test
        for cb in CircuitBreakerMonitor.get_circuits():
            cb.reset()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.NetworkError("Connection refused")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post = mock_post

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
            )

            # First call will retry 5 times then fail with RetryError or CircuitBreakerError
            with pytest.raises((httpx.NetworkError, CircuitBreakerError, RetryError)):
                auth.get_headers()

            # Verify multiple attempts were made (retry attempts)
            # 5 retry attempts with exponential backoff
            assert call_count >= 5

        # Reset circuit breaker for other tests
        for cb in CircuitBreakerMonitor.get_circuits():
            cb.reset()

    def test_recovery_after_transient_failure(self) -> None:
        """System recovers automatically when token endpoint becomes available."""
        from circuitbreaker import CircuitBreakerMonitor

        # Reset all circuit breakers before test
        for cb in CircuitBreakerMonitor.get_circuits():
            cb.reset()

        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "recovered-token", "expires_in": 3600}

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Timeout")
            return mock_response

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post = mock_post

            auth = M2MAuth(
                token_url="https://auth.example.com/token",
                client_id="my-client",
                client_secret="my-secret",
            )

            # Call succeeds after retries (simulating endpoint recovery)
            headers = auth.get_headers()

            assert headers == {"Authorization": "Bearer recovered-token"}
            assert call_count == 3  # 2 failures + 1 success


class TestCompositeAuth:
    """Test CompositeAuth for multi-header scenarios."""

    def test_requires_at_least_one_provider(self) -> None:
        """CompositeAuth raises ValueError with no providers."""
        with pytest.raises(ValueError, match="at least one provider"):
            CompositeAuth()

    def test_single_provider_passthrough(self) -> None:
        """Single provider headers are passed through unchanged."""
        with patch.dict("os.environ", {"TEST_KEY": "secret123"}):
            api_key = ApiKeyAuth("TEST_KEY")
            composite = CompositeAuth(api_key)
            headers = composite.get_headers()
            assert headers == {"Authorization": "Bearer secret123"}

    def test_merges_multiple_provider_headers(self) -> None:
        """Multiple providers have their headers merged."""
        with patch.dict("os.environ", {"KEY1": "token1", "KEY2": "token2"}):
            # First provider: standard Bearer
            auth1 = ApiKeyAuth("KEY1")
            # Second provider: custom header
            auth2 = ApiKeyAuth("KEY2", header_format="x-api-key")
            composite = CompositeAuth(auth1, auth2)

            headers = composite.get_headers()
            assert headers["Authorization"] == "Bearer token1"
            assert headers["X-Api-Key"] == "token2"

    def test_later_provider_overrides_earlier(self) -> None:
        """When providers have conflicting headers, later wins."""
        with patch.dict("os.environ", {"KEY1": "first", "KEY2": "second"}):
            auth1 = ApiKeyAuth("KEY1")
            auth2 = ApiKeyAuth("KEY2")  # Same header as auth1
            composite = CompositeAuth(auth1, auth2)

            headers = composite.get_headers()
            # auth2 should override auth1
            assert headers["Authorization"] == "Bearer second"

    def test_merges_request_kwargs(self) -> None:
        """Request kwargs are merged from all providers."""
        with patch.dict("os.environ", {"KEY": "value"}):
            # Mock provider with custom request kwargs
            mock_provider = MagicMock()
            mock_provider.get_headers.return_value = {}
            mock_provider.get_request_kwargs.return_value = {"verify": "/path/to/ca.crt"}

            api_key = ApiKeyAuth("KEY")
            composite = CompositeAuth(api_key, mock_provider)

            kwargs = composite.get_request_kwargs()
            assert kwargs["verify"] == "/path/to/ca.crt"

    def test_m2m_plus_api_key_pattern(self) -> None:
        """Verify BMW pattern: M2M bearer + API key headers."""
        # This test uses mocks since M2MAuth needs actual token endpoint
        mock_m2m = MagicMock()
        mock_m2m.get_headers.return_value = {"Authorization": "Bearer m2m_token_123"}
        mock_m2m.get_request_kwargs.return_value = {}

        with patch.dict("os.environ", {"BMW_API_KEY": "bmw_secret"}):
            api_key = ApiKeyAuth("BMW_API_KEY", header_format="x-api-key")
            composite = CompositeAuth(mock_m2m, api_key)

            headers = composite.get_headers()
            assert headers["Authorization"] == "Bearer m2m_token_123"
            assert headers["X-Api-Key"] == "bmw_secret"


class TestEnterpriseAuth:
    """Test enterprise authentication (M2M + API key)."""

    def test_enterprise_auth_creates_composite(self) -> None:
        """Enterprise auth creates CompositeAuth with M2M + ApiKey."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "m2m-token", "expires_in": 3600}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            with patch.dict("os.environ", {
                "TEST_LLM_KEY": "llm-api-key-123",
                "CLIENT_ID": "my-client",
                "CLIENT_SECRET": "my-secret",
            }):
                config = {
                    "auth": {
                        "type": "enterprise",
                        "token_url": "https://auth.example.com/token",
                        "client_id": "${CLIENT_ID}",
                        "client_secret": "${CLIENT_SECRET}",
                        "llm_api_key_env": "TEST_LLM_KEY",
                    }
                }

                auth = resolve_auth("openai", config)

                assert isinstance(auth, CompositeAuth)

                # Verify headers include both M2M token and API key
                headers = auth.get_headers()
                assert headers["Authorization"] == "Bearer m2m-token"
                assert headers["X-Api-Key"] == "llm-api-key-123"

    def test_enterprise_auth_missing_fields_lists_all(self) -> None:
        """Missing enterprise auth fields error lists ALL missing fields."""
        config = {
            "auth": {
                "type": "enterprise",
                "token_url": "https://auth.example.com/token",
                # Missing: client_id, client_secret, llm_api_key_env
            }
        }

        with pytest.raises(ValueError) as exc_info:
            resolve_auth("openai", config)

        error_msg = str(exc_info.value)
        # All three missing fields should be mentioned
        assert "client_id" in error_msg
        assert "client_secret" in error_msg
        assert "llm_api_key_env" in error_msg

    def test_enterprise_auth_validates_api_key_env_exists(self) -> None:
        """Enterprise auth validates API key env var exists at startup."""
        config = {
            "auth": {
                "type": "enterprise",
                "token_url": "https://auth.example.com/token",
                "client_id": "my-client",
                "client_secret": "my-secret",
                "llm_api_key_env": "NONEXISTENT_API_KEY",
            }
        }

        # Ensure the env var doesn't exist
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                resolve_auth("openai", config)

            error_msg = str(exc_info.value)
            assert "NONEXISTENT_API_KEY" in error_msg

    def test_enterprise_auth_custom_header(self) -> None:
        """Enterprise auth supports custom LLM API key header."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "m2m-token", "expires_in": 3600}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            with patch.dict("os.environ", {"CUSTOM_KEY": "custom-api-key"}):
                config = {
                    "auth": {
                        "type": "enterprise",
                        "token_url": "https://auth.example.com/token",
                        "client_id": "my-client",
                        "client_secret": "my-secret",
                        "llm_api_key_env": "CUSTOM_KEY",
                        "llm_api_key_header": "Authorization",  # Use bearer instead
                    }
                }

                auth = resolve_auth("openai", config)
                headers = auth.get_headers()

                # With Authorization header, API key should override M2M token
                # (later provider overrides earlier in CompositeAuth)
                assert "Authorization" in headers

    def test_enterprise_auth_with_mtls(self, temp_certificate) -> None:
        """Enterprise auth passes through mTLS options to M2MAuth."""
        cert_path, key_path = temp_certificate
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "m2m-token", "expires_in": 3600}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            with patch.dict("os.environ", {"TEST_KEY": "api-key"}):
                config = {
                    "auth": {
                        "type": "enterprise",
                        "token_url": "https://auth.example.com/token",
                        "client_id": "my-client",
                        "client_secret": "my-secret",
                        "llm_api_key_env": "TEST_KEY",
                        "client_cert_path": cert_path,
                        "client_key_path": key_path,
                    },
                    "cert_path": cert_path,  # CA cert for verification
                }

                auth = resolve_auth("openai", config)

                # Verify request kwargs include cert options
                kwargs = auth.get_request_kwargs()
                assert kwargs.get("verify") == cert_path
                assert kwargs.get("cert") == (cert_path, key_path)

    def test_enterprise_auth_with_scope(self) -> None:
        """Enterprise auth passes scope to M2MAuth."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "m2m-token", "expires_in": 3600}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response

            with patch.dict("os.environ", {"TEST_KEY": "api-key"}):
                config = {
                    "auth": {
                        "type": "enterprise",
                        "token_url": "https://auth.example.com/token",
                        "client_id": "my-client",
                        "client_secret": "my-secret",
                        "llm_api_key_env": "TEST_KEY",
                        "scope": "llm-access read",
                    }
                }

                auth = resolve_auth("openai", config)
                # Trigger token fetch
                auth.get_headers()

                # Verify scope was passed in token request
                call_kwargs = mock_client.return_value.__enter__.return_value.post.call_args
                assert call_kwargs[1]["data"]["scope"] == "llm-access read"
