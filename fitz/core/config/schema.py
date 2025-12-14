# core/config/schema.py
"""
Pydantic schema for Fitz configuration.

Rules:
- Strict validation
- No unknown keys
- Provider selection lives in config, never in non-plugin code paths.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PluginConfig(BaseModel):
    """
    Generic plugin configuration block.

    All provider-specific configuration must be expressed via:
    - plugin_name: plugin id in the central registry
    - kwargs: arbitrary plugin init kwargs
    """

    plugin_name: str = Field(..., description="Plugin name in the central registry")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin init kwargs")

    model_config = ConfigDict(extra="forbid")


class FitzConfig(BaseModel):
    """
    Core Fitz config.

    This config is intentionally minimal and provider-agnostic.
    """

    llm: PluginConfig

    model_config = ConfigDict(extra="forbid")
