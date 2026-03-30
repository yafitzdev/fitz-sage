# fitz_sage/cloud/config.py
"""Configuration for Fitz Cloud integration."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from fitz_sage.core.config_base import ApiServiceConfig, FeatureToggleMixin


class CloudConfig(ApiServiceConfig, FeatureToggleMixin):
    """
    Fitz Cloud configuration.

    Inherits from ApiServiceConfig for API settings and timeouts,
    and FeatureToggleMixin for enable/disable functionality.

    Example YAML:
        cloud:
          enabled: true
          api_key: "fitz_xxx..."
          org_id: "your-org-uuid"  # Can also use FITZ_ORG_ID env var
          org_key: "64-char-hex-string"
          base_url: "https://api.fitz-sage.cloud/v1"
          timeout: 30
    """

    # Override base_url default for Fitz Cloud
    base_url: str = Field(
        default="https://api.fitz-sage.cloud/v1",
        description="Fitz Cloud API base URL",
    )

    # Cloud-specific fields
    org_id: Optional[str] = Field(
        default=None,
        description="Organization ID (UUID). Can also be set via FITZ_ORG_ID env var.",
    )

    org_key: Optional[str] = Field(
        default=None,
        description="Organization encryption key (NEVER sent to server)",
    )

    def validate_config(self) -> None:
        """Validate that required fields are set when enabled."""
        if self.enabled:
            if not self.api_key:
                raise ValueError("cloud.api_key is required when cloud.enabled=true")
            if not self.org_key:
                raise ValueError("cloud.org_key is required when cloud.enabled=true")
