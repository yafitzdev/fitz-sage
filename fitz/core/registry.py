# fitz/core/registry.py
"""
Centralized Plugin Registry System.

Single implementation of plugin registry logic used by all plugin types:
- Ingest
- Chunking
- Retriever
- Pipeline

NOTE: LLM plugins use fitz.llm.registry (YAML-based)
NOTE: Vector DB plugins use fitz.vector_db.registry (YAML-based)
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class PluginRegistryError(Exception):
    """Base error for plugin registry operations."""
    pass


class PluginNotFoundError(PluginRegistryError):
    """Raised when requested plugin doesn't exist."""
    pass


class DuplicatePluginError(PluginRegistryError):
    """Raised when two plugins have the same name."""
    pass


class LLMRegistryError(PluginNotFoundError):
    """Error from LLM registry."""
    pass


class VectorDBRegistryError(PluginNotFoundError):
    """Error from Vector DB registry."""
    pass


@dataclass
class PluginRegistry:
    """
    Generic plugin registry with lazy auto-discovery.

    Args:
        name: Registry name (for error messages)
        scan_packages: List of package names to scan for plugins
        required_method: Method name that plugins must have
        plugin_name_attr: Attribute containing the plugin name
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
        """Get a plugin by name."""
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
            return

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

            if not isinstance(obj, type):
                continue

            if not hasattr(obj, self.required_method):
                continue

            if not hasattr(obj, self.plugin_name_attr):
                continue

            if self.plugin_type_filter:
                plugin_type = getattr(obj, "plugin_type", None)
                if plugin_type != self.plugin_type_filter:
                    continue

            if self.check_module_match:
                if obj.__module__ != module.__name__:
                    continue

            try:
                self.register(obj)
            except (PluginRegistryError, DuplicatePluginError) as e:
                logger.debug(f"Skipping {name}: {e}")


# Pre-configured registries
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


__all__ = [
    # Exceptions
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
    "LLMRegistryError",
    "VectorDBRegistryError",
    # Registry class
    "PluginRegistry",
    # Pre-configured registries
    "INGEST_REGISTRY",
    "CHUNKING_REGISTRY",
    "RETRIEVER_REGISTRY",
    "PIPELINE_REGISTRY",
]