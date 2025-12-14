# ingest/config/schema.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngesterConfig(BaseModel):
    plugin_name: str = Field(..., description="Ingester plugin name")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Ingester init kwargs")

    model_config = ConfigDict(extra="forbid")


class ChunkerConfig(BaseModel):
    plugin_name: str = Field(..., description="Chunker plugin name")
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=0, ge=0)
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Chunker init kwargs")

    model_config = ConfigDict(extra="forbid")


class IngestConfig(BaseModel):
    """
    Central configuration for the entire ingestion pipeline.

    Rules:
    - Engines are built FROM config (engines do not own config).
    - Provider/plugin selection lives only here (plugin_name).
    """

    ingester: IngesterConfig
    chunker: ChunkerConfig
    collection: str = Field(..., description="Target vector DB collection")

    model_config = ConfigDict(extra="forbid")
