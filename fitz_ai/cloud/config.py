# fitz_ai/cloud/config.py
"""Configuration for Fitz Cloud integration."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CloudConfig(BaseModel):
    """
    Fitz Cloud configuration.

    Example YAML:
        cloud:
          enabled: true
          api_key: "fitz_xxx..."
          org_id: "your-org-uuid"  # Can also use FITZ_ORG_ID env var
          org_key: "64-char-hex-string"
          base_url: "https://api.fitz-ai.cloud/v1"
    """

    enabled: bool = Field(
        default=False,
        description="Enable Fitz Cloud features (cache, routing)",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="Fitz Cloud API key (fitz_xxx format)",
    )

    org_id: Optional[str] = Field(
        default=None,
        description="Organization ID (UUID). Can also be set via FITZ_ORG_ID env var.",
    )

    org_key: Optional[str] = Field(
        default=None,
        description="Organization encryption key (NEVER sent to server)",
    )

    base_url: str = Field(
        default="https://api.fitz-ai.cloud/v1",
        description="Fitz Cloud API base URL",
    )

    timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
    )

    def validate_config(self) -> None:
        """Validate that required fields are set when enabled."""
        if self.enabled:
            if not self.api_key:
                raise ValueError("cloud.api_key is required when cloud.enabled=true")
            if not self.org_key:
                raise ValueError("cloud.org_key is required when cloud.enabled=true")
