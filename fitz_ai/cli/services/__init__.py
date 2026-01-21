# fitz_ai/cli/services/__init__.py
"""Service layer for CLI commands - business logic without UI concerns."""

from .ingest_service import IngestService
from .init_service import InitService

__all__ = ["InitService", "IngestService"]
