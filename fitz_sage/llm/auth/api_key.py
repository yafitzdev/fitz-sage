# fitz_sage/llm/auth/api_key.py
"""
API key authentication provider.
"""

from __future__ import annotations

import os
from typing import Any, Literal


class ApiKeyAuth:
    """
    Simple API key authentication from environment variable.

    Args:
        env_var: Environment variable name containing the API key.
        header_format: How to format the Authorization header.
            - "bearer": Authorization: Bearer <key>
            - "x-api-key": X-Api-Key: <key>
            - "basic": Authorization: Basic <key>
    """

    def __init__(
        self,
        env_var: str,
        header_format: Literal["bearer", "x-api-key", "basic"] = "bearer",
    ) -> None:
        self.env_var = env_var
        self.header_format = header_format
        self._api_key: str | None = None

    @property
    def api_key(self) -> str:
        """Lazily load API key from environment."""
        if self._api_key is None:
            key = os.environ.get(self.env_var)
            if not key:
                raise ValueError(f"Environment variable {self.env_var} not set")
            self._api_key = key
        return self._api_key

    def get_headers(self) -> dict[str, str]:
        """Return authentication headers."""
        key = self.api_key
        if self.header_format == "bearer":
            return {"Authorization": f"Bearer {key}"}
        elif self.header_format == "x-api-key":
            return {"X-Api-Key": key}
        elif self.header_format == "basic":
            return {"Authorization": f"Basic {key}"}
        else:
            return {"Authorization": f"Bearer {key}"}

    def get_request_kwargs(self) -> dict[str, Any]:
        """No additional request kwargs needed for API key auth."""
        return {}
