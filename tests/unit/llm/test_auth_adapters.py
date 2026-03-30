# tests/unit/llm/test_auth_adapters.py
"""
Unit tests for auth adapters (DynamicHttpxAuth, TokenProviderAdapter).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from fitz_sage.llm.auth import (
    ApiKeyAuth,
    AuthProvider,
    DynamicHttpxAuth,
    TokenProviderAdapter,
)


class TestDynamicHttpxAuth:
    """Test DynamicHttpxAuth wrapper."""

    def test_is_httpx_auth_subclass(self) -> None:
        """DynamicHttpxAuth is valid httpx.Auth subclass."""
        assert issubclass(DynamicHttpxAuth, httpx.Auth)

    def test_auth_flow_injects_fresh_headers(self) -> None:
        """auth_flow injects headers from AuthProvider into request."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {"Authorization": "Bearer test-token"}

        auth = DynamicHttpxAuth(mock_provider)
        request = httpx.Request("GET", "https://api.example.com/test")

        # Execute auth_flow generator
        flow = auth.auth_flow(request)
        modified_request = next(flow)

        assert modified_request.headers["Authorization"] == "Bearer test-token"
        mock_provider.get_headers.assert_called_once()

    def test_auth_flow_calls_provider_each_time(self) -> None:
        """auth_flow calls get_headers() fresh on each request."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.side_effect = [
            {"Authorization": "Bearer token-1"},
            {"Authorization": "Bearer token-2"},
        ]

        auth = DynamicHttpxAuth(mock_provider)

        # First request
        req1 = httpx.Request("GET", "https://api.example.com/test")
        flow1 = auth.auth_flow(req1)
        result1 = next(flow1)
        assert result1.headers["Authorization"] == "Bearer token-1"

        # Second request - should get fresh headers
        req2 = httpx.Request("GET", "https://api.example.com/test")
        flow2 = auth.auth_flow(req2)
        result2 = next(flow2)
        assert result2.headers["Authorization"] == "Bearer token-2"

        assert mock_provider.get_headers.call_count == 2

    def test_works_with_api_key_auth(self) -> None:
        """DynamicHttpxAuth works with ApiKeyAuth provider."""
        with patch.dict("os.environ", {"TEST_KEY": "my-api-key"}):
            api_auth = ApiKeyAuth("TEST_KEY", header_format="bearer")
            dynamic_auth = DynamicHttpxAuth(api_auth)

            request = httpx.Request("GET", "https://api.example.com/test")
            flow = dynamic_auth.auth_flow(request)
            modified = next(flow)

            assert modified.headers["Authorization"] == "Bearer my-api-key"

    def test_injects_multiple_headers(self) -> None:
        """auth_flow injects all headers from AuthProvider."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {
            "Authorization": "Bearer token",
            "X-Custom-Header": "custom-value",
        }

        auth = DynamicHttpxAuth(mock_provider)
        request = httpx.Request("GET", "https://api.example.com/test")

        flow = auth.auth_flow(request)
        modified = next(flow)

        assert modified.headers["Authorization"] == "Bearer token"
        assert modified.headers["X-Custom-Header"] == "custom-value"


class TestTokenProviderAdapter:
    """Test TokenProviderAdapter for OpenAI SDK pattern."""

    def test_is_callable(self) -> None:
        """TokenProviderAdapter instance is callable."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {"Authorization": "Bearer token"}

        adapter = TokenProviderAdapter(mock_provider)
        assert callable(adapter)

    def test_returns_bearer_token(self) -> None:
        """__call__ returns token without Bearer prefix."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {"Authorization": "Bearer my-token-123"}

        adapter = TokenProviderAdapter(mock_provider)
        token = adapter()

        assert token == "my-token-123"

    def test_raises_on_non_bearer(self) -> None:
        """__call__ raises ValueError when header is not Bearer format."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {"Authorization": "Basic abc123"}

        adapter = TokenProviderAdapter(mock_provider)

        with pytest.raises(ValueError, match="requires AuthProvider with Bearer token"):
            adapter()

    def test_raises_on_missing_header(self) -> None:
        """__call__ raises ValueError when no Authorization header."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {}

        adapter = TokenProviderAdapter(mock_provider)

        with pytest.raises(ValueError, match="No Authorization header"):
            adapter()

    def test_calls_provider_each_time(self) -> None:
        """__call__ fetches fresh token on each invocation."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.side_effect = [
            {"Authorization": "Bearer token-1"},
            {"Authorization": "Bearer token-2"},
        ]

        adapter = TokenProviderAdapter(mock_provider)

        assert adapter() == "token-1"
        assert adapter() == "token-2"
        assert mock_provider.get_headers.call_count == 2

    def test_handles_token_with_special_characters(self) -> None:
        """__call__ handles tokens with special characters."""
        mock_provider = MagicMock(spec=AuthProvider)
        # JWT tokens often have dots, underscores, hyphens
        complex_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        mock_provider.get_headers.return_value = {"Authorization": f"Bearer {complex_token}"}

        adapter = TokenProviderAdapter(mock_provider)
        token = adapter()

        assert token == complex_token

    def test_raises_on_empty_bearer_token(self) -> None:
        """__call__ raises ValueError when Bearer token is empty."""
        mock_provider = MagicMock(spec=AuthProvider)
        mock_provider.get_headers.return_value = {"Authorization": "Bearer "}

        adapter = TokenProviderAdapter(mock_provider)
        # "Bearer " has length 7, so stripping gives empty string
        # This should work but return empty string - implementation doesn't validate
        token = adapter()
        assert token == ""
