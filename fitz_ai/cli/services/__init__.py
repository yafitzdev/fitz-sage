# fitz_ai/cli/services/__init__.py
"""Service layer for CLI commands - business logic without UI concerns."""

from .init_service import InitService
from .ingest_service import IngestService

__all__ = ["InitService", "IngestService"]
