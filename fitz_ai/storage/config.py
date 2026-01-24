# fitz_ai/storage/config.py
"""Configuration schema for unified PostgreSQL storage."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class StorageMode(str, Enum):
    """Storage deployment mode."""

    LOCAL = "local"  # pgserver embedded PostgreSQL
    EXTERNAL = "external"  # External PostgreSQL connection string


class StorageConfig(BaseModel):
    """
    Configuration for unified PostgreSQL storage.

    Supports two modes:
    - LOCAL: Uses pgserver for embedded PostgreSQL (zero-friction local dev)
    - EXTERNAL: Uses user-provided PostgreSQL connection string (shared deployments)
    """

    mode: StorageMode = Field(
        default=StorageMode.LOCAL,
        description="Storage mode: 'local' (pgserver) or 'external' (connection_string)",
    )

    # Local mode settings
    data_dir: Optional[Path] = Field(
        default=None,
        description="Data directory for pgserver. None = use FitzPaths.pgdata()",
    )

    # External mode settings
    connection_string: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection string for external mode (e.g., postgresql://user:pass@host:5432/db)",
    )

    # Connection pool settings
    pool_min_size: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Minimum connections in pool",
    )

    pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum connections in pool",
    )

    # HNSW index settings
    hnsw_m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="HNSW max connections per layer (higher = better recall, more memory)",
    )

    hnsw_ef_construction: int = Field(
        default=64,
        ge=16,
        le=512,
        description="HNSW construction ef (higher = better recall, slower builds)",
    )

    def validate_mode(self) -> None:
        """Validate configuration based on mode."""
        if self.mode == StorageMode.EXTERNAL and not self.connection_string:
            raise ValueError("connection_string required for external mode")
