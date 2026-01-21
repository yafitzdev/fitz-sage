# fitz_ai/ingestion/enrichment/registry.py
"""
Unified registry for enrichment modules.

Provides factory functions and module discovery for the enrichment system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fitz_ai.ingestion.enrichment.modules import (
    EnrichmentModule,
    EntityModule,
    KeywordModule,
    SummaryModule,
)

if TYPE_CHECKING:
    pass


# Default modules in order of execution
DEFAULT_MODULES: list[type[EnrichmentModule]] = [
    SummaryModule,
    KeywordModule,
    EntityModule,
]


def get_default_modules() -> list[EnrichmentModule]:
    """Create instances of all default enrichment modules."""
    return [module_cls() for module_cls in DEFAULT_MODULES]


def get_module_by_name(name: str) -> type[EnrichmentModule] | None:
    """
    Get a module class by name.

    Args:
        name: Module name (e.g., "summary", "keywords", "entities")

    Returns:
        Module class if found, None otherwise
    """
    module_map = {module_cls().name: module_cls for module_cls in DEFAULT_MODULES}
    return module_map.get(name)


def list_available_modules() -> list[str]:
    """List names of all available enrichment modules."""
    return [module_cls().name for module_cls in DEFAULT_MODULES]


__all__ = [
    "DEFAULT_MODULES",
    "get_default_modules",
    "get_module_by_name",
    "list_available_modules",
]
