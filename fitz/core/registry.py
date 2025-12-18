# File: fitz/core/registry.py
"""
Centralized Plugin Registry System.

Single implementation of plugin registry logic used by all plugin types:
- Vector DB
- Ingest
- Chunking
- Retriever
- Pipeline

NOTE: LLM plugins (chat, embedding, rerank) have their own registry at fitz.llm.registry
      They use YAML-based discovery, not Python module scanning.

Design principle: NO SILENT FALLBACK
- If you ask for "local", you get local or an error
- No magic substitution
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class PluginRegistryError(Exception):
    """Base error for plugin registry operations."""
    pass


class PluginNotFoundError(PluginRegistryError, ValueError):
    """Raised when requested plugin doesn't exist.

    Inherits from ValueError for backwards compatibility.
    """
    pass


class DuplicatePluginError(PluginRegistryError):
    """Raised when two plugins have the same name."""
    pass


# Specific error classes for backwards compat
class LLMRegistryError(PluginNotFoundError):
    """Error from LLM registry.

    NOTE: LLM plugins are managed by fitz.llm.registry, not here.
    This class is kept for backwards compatibility.
    """
    pass


class VectorDBRegistryError(PluginNotFoundError):
    """Error from Vector DB registry."""
    pass


# =============================================================================
# Generic Plugin Registry
# =============================================================================


@dataclass
class PluginRegistry:
    """
    Generic plugin registry with lazy auto-discovery.

    Args:
        name: Registry name (for error messages and logging)
        scan_packages: List of package names to scan for plugins
        required_method: Method name that plugins must have
        plugin_name_attr: Attribute containing the plugin name (default: "plugin_name")
        plugin_type_filter: If set, only register plugins with this plugin_type
        check_module_match: If True, only register classes defined in the scanned module
    """

    name: str
    scan_packages: List[str]
    required_method: str
    plugin_name_attr: str = "plugin_name"
    plugin_type_filter: str | None = None
    check_module_match: bool = True
    _plugins: Dict[str, Type[Any]] = field(default_factory=dict, repr=False)
    _discovered: bool = field(default=False, repr=False)

    def get(self, plugin_name: str) -> Type[Any]:
        """Get a plugin by name. Raises PluginNotFoundError if not found."""
        self._ensure_discovered()

        if plugin_name not in self._plugins:
            available = sorted(self._plugins.keys())
            raise PluginNotFoundError(
                f"Unknown {self.name} plugin: {plugin_name!r}. Available: {available}"
            )

        return self._plugins[plugin_name]

    def list_available(self) -> List[str]:
        """List all available plugin names."""
        self._ensure_discovered()
        return sorted(self._plugins.keys())

    def register(self, plugin_class: Type[Any]) -> None:
        """Manually register a plugin class."""
        if not hasattr(plugin_class, self.required_method):
            raise PluginRegistryError(
                f"{self.name} plugin {plugin_class.__name__} missing required "
                f"method {self.required_method!r}"
            )

        if not hasattr(plugin_class, self.plugin_name_attr):
            raise PluginRegistryError(
                f"{self.name} plugin {plugin_class.__name__} missing required "
                f"attribute {self.plugin_name_attr!r}"
            )

        name = getattr(plugin_class, self.plugin_name_attr)

        if name in self._plugins:
            existing = self._plugins[name]
            if existing is not plugin_class:
                raise DuplicatePluginError(
                    f"Duplicate {self.name} plugin: {name!r}. "
                    f"Found in {existing.__module__} and {plugin_class.__module__}"
                )
            return  # Same class, already registered

        self._plugins[name] = plugin_class
        logger.debug(f"Registered {self.name} plugin: {name!r}")

    def _ensure_discovered(self) -> None:
        """Run auto-discovery if not already done."""
        if self._discovered:
            return

        for package_name in self.scan_packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.debug(f"Could not import {package_name}: {e}")
                continue

            self._scan_package(package)

        self._discovered = True
        logger.debug(
            f"Discovered {len(self._plugins)} {self.name} plugin(s): "
            f"{sorted(self._plugins.keys())}"
        )

    def _scan_package(self, package: Any) -> None:
        """Scan a package for plugin classes."""
        package_path = getattr(package, "__path__", None)
        if not package_path:
            return

        for importer, modname, ispkg in pkgutil.walk_packages(
            package_path, prefix=f"{package.__name__}."
        ):
            if ispkg:
                continue

            try:
                module = importlib.import_module(modname)
            except Exception as e:
                logger.debug(f"Could not import {modname}: {e}")
                continue

            self._scan_module(module)

    def _scan_module(self, module: Any) -> None:
        """Scan a module for plugin classes."""
        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Must be a class
            if not isinstance(obj, type):
                continue

            # Must have required method
            if not hasattr(obj, self.required_method):
                continue

            # Must have plugin_name attribute
            if not hasattr(obj, self.plugin_name_attr):
                continue

            # Optional: check plugin_type matches
            if self.plugin_type_filter:
                plugin_type = getattr(obj, "plugin_type", None)
                if plugin_type != self.plugin_type_filter:
                    continue

            # Optional: only accept classes defined in this module
            if self.check_module_match:
                if obj.__module__ != module.__name__:
                    continue

            # Register it
            try:
                self.register(obj)
            except (PluginRegistryError, DuplicatePluginError) as e:
                logger.debug(f"Skipping {name}: {e}")


# =============================================================================
# Pre-configured Registries
# =============================================================================

VECTOR_DB_REGISTRY = PluginRegistry(
    name="vector_db",
    scan_packages=["fitz.vector_db.plugins"],
    required_method="search",
    plugin_type_filter="vector_db",
    check_module_match=False,  # Plugins can be re-exported from other modules
)

INGEST_REGISTRY = PluginRegistry(
    name="ingest",
    scan_packages=["fitz.ingest.ingestion.plugins"],
    required_method="ingest",
)

CHUNKING_REGISTRY = PluginRegistry(
    name="chunking",
    scan_packages=["fitz.ingest.chunking.plugins"],
    required_method="chunk_text",
)

RETRIEVER_REGISTRY = PluginRegistry(
    name="retriever",
    scan_packages=["fitz.engines.classic_rag.retrieval.runtime.plugins"],
    required_method="retrieve",
)

PIPELINE_REGISTRY = PluginRegistry(
    name="pipeline",
    scan_packages=["fitz.engines.classic_rag.pipeline.pipeline.plugins"],
    required_method="build",
)


# =============================================================================
# Vector DB Functions
# =============================================================================


def get_vector_db_plugin(plugin_name: str) -> Type[Any]:
    """Get a vector DB plugin by name."""
    try:
        return VECTOR_DB_REGISTRY.get(plugin_name)
    except PluginNotFoundError as e:
        raise VectorDBRegistryError(str(e)) from e


def available_vector_db_plugins() -> List[str]:
    """List available vector DB plugins."""
    return VECTOR_DB_REGISTRY.list_available()


def resolve_vector_db_plugin(requested_name: str) -> Type[Any]:
    """Alias for get_vector_db_plugin (backwards compatibility)."""
    return get_vector_db_plugin(requested_name)


# =============================================================================
# Ingest Functions
# =============================================================================


def get_ingest_plugin(plugin_name: str) -> Type[Any]:
    """Get an ingestion plugin by name."""
    return INGEST_REGISTRY.get(plugin_name)


def available_ingest_plugins() -> List[str]:
    """List available ingestion plugins."""
    return INGEST_REGISTRY.list_available()


# =============================================================================
# Chunking Functions
# =============================================================================


def get_chunker_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunker plugin by name."""
    return CHUNKING_REGISTRY.get(plugin_name)


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Alias for get_chunker_plugin."""
    return CHUNKING_REGISTRY.get(plugin_name)


def available_chunking_plugins() -> List[str]:
    """List available chunking plugins."""
    return CHUNKING_REGISTRY.list_available()


# =============================================================================
# Retriever Functions
# =============================================================================


def get_retriever_plugin(plugin_name: str) -> Type[Any]:
    """Get a retriever plugin by name."""
    return RETRIEVER_REGISTRY.get(plugin_name)


def available_retriever_plugins() -> List[str]:
    """List available retriever plugins."""
    return RETRIEVER_REGISTRY.list_available()


# =============================================================================
# Pipeline Functions
# =============================================================================


def get_pipeline_plugin(plugin_name: str) -> Type[Any]:
    """Get a pipeline plugin by name."""
    return PIPELINE_REGISTRY.get(plugin_name)


def available_pipeline_plugins() -> List[str]:
    """List available pipeline plugins."""
    return PIPELINE_REGISTRY.list_available()