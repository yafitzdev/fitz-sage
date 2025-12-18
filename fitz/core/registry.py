# fitz/core/registry.py
"""
Centralized Plugin Registry System.

LLM plugins (chat, embedding, rerank) are YAML-based only.
Other plugin types (vector_db, ingest, etc.) still use Python class discovery.

Design principle: NO SILENT FALLBACK
- If you ask for "cohere", you get cohere or an error
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


# =============================================================================
# Generic Plugin Registry (for non-LLM plugins)
# =============================================================================


@dataclass
class PluginRegistry:
    """
    Generic plugin registry with lazy auto-discovery.
    Used for vector_db, ingest, chunking, retriever, pipeline plugins.
    NOT used for LLM plugins (those are YAML-based).
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

    def _ensure_discovered(self) -> None:
        """Run discovery if not already done."""
        if self._discovered:
            return

        for package_name in self.scan_packages:
            self._scan_package(package_name)

        self._discovered = True
        logger.debug(f"[{self.name}] Discovered plugins: {list(self._plugins.keys())}")

    def _scan_package(self, package_name: str) -> None:
        """Scan a package for plugin classes."""
        try:
            pkg = importlib.import_module(package_name)
        except ImportError:
            logger.debug(f"[{self.name}] Package not found: {package_name}")
            return

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return

        for module_info in pkgutil.iter_modules(pkg_path):
            module_name = f"{package_name}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                logger.debug(f"[{self.name}] Failed to import {module_name}: {e}")
                continue

            for cls in self._iter_plugin_classes(module):
                self._register(cls)

    def _iter_plugin_classes(self, module: object) -> Iterable[type]:
        """Find plugin classes in a module."""
        mod_name = getattr(module, "__name__", "")

        for obj in vars(module).values():
            if not isinstance(obj, type):
                continue

            if self.check_module_match and getattr(obj, "__module__", None) != mod_name:
                continue

            plugin_name = getattr(obj, self.plugin_name_attr, None)
            if not isinstance(plugin_name, str) or not plugin_name:
                continue

            if self.plugin_type_filter:
                obj_type = getattr(obj, "plugin_type", None)
                if obj_type != self.plugin_type_filter:
                    continue

            method = getattr(obj, self.required_method, None)
            if not callable(method):
                continue

            yield obj

    def _register(self, cls: Type[Any]) -> None:
        """Register a plugin class."""
        name = getattr(cls, self.plugin_name_attr)

        if name in self._plugins:
            existing = self._plugins[name]
            if existing is not cls:
                raise DuplicatePluginError(
                    f"Duplicate {self.name} plugin: {name!r}. "
                    f"Found in {existing.__module__} and {cls.__module__}"
                )
            return

        self._plugins[name] = cls


# =============================================================================
# Non-LLM Registries (still use Python class discovery)
# =============================================================================

VECTOR_DB_REGISTRY = PluginRegistry(
    name="vector_db",
    scan_packages=["fitz.vector_db.plugins"],
    required_method="search",
    plugin_type_filter="vector_db",
    check_module_match=False,
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
# LLM Registry (YAML-based only)
# =============================================================================

VALID_LLM_TYPES = frozenset({"chat", "embedding", "rerank"})

LLM_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {}
_LLM_DISCOVERED = False


def _create_yaml_plugin_wrapper(plugin_type: str, plugin_name: str) -> Type[Any]:
    """Create a wrapper class for a YAML plugin."""
    from fitz.llm.yaml_wrappers import create_yaml_plugin_wrapper
    return create_yaml_plugin_wrapper(plugin_type, plugin_name)


def _discover_llm_plugins() -> None:
    """Discover YAML-based LLM plugins."""
    global _LLM_DISCOVERED
    if _LLM_DISCOVERED:
        return

    from fitz.llm.loader import list_yaml_plugins

    for plugin_type in ("chat", "embedding", "rerank"):
        LLM_REGISTRY[plugin_type] = {}

        try:
            plugin_names = list_yaml_plugins(plugin_type)
            for name in plugin_names:
                try:
                    wrapper_cls = _create_yaml_plugin_wrapper(plugin_type, name)
                    LLM_REGISTRY[plugin_type][name] = wrapper_cls
                    logger.debug(f"[LLM] Registered {plugin_type} plugin: {name}")
                except Exception as e:
                    logger.warning(f"[LLM] Failed to create wrapper for {plugin_type}/{name}: {e}")
        except Exception as e:
            logger.warning(f"[LLM] Failed to list {plugin_type} plugins: {e}")

    _LLM_DISCOVERED = True
    logger.debug(f"[LLM] Discovery complete: {dict(LLM_REGISTRY)}")


def get_llm_plugin(*, plugin_name: str, plugin_type: str) -> Type[Any]:
    """Get an LLM plugin by name and type."""
    if plugin_type not in VALID_LLM_TYPES:
        raise ValueError(
            f"Invalid LLM plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(VALID_LLM_TYPES)}"
        )

    _discover_llm_plugins()

    if plugin_name not in LLM_REGISTRY[plugin_type]:
        available = sorted(LLM_REGISTRY[plugin_type].keys())
        raise LLMRegistryError(
            f"Unknown {plugin_type} plugin: {plugin_name!r}. Available: {available}"
        )

    return LLM_REGISTRY[plugin_type][plugin_name]


def available_llm_plugins(plugin_type: str) -> List[str]:
    """List available LLM plugins for a type."""
    _discover_llm_plugins()
    return sorted(LLM_REGISTRY.get(plugin_type, {}).keys())


def resolve_llm_plugin(*, plugin_type: str, requested_name: str) -> Type[Any]:
    """Alias for get_llm_plugin."""
    return get_llm_plugin(plugin_name=requested_name, plugin_type=plugin_type)


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
    """Alias for get_vector_db_plugin."""
    return get_vector_db_plugin(requested_name)


# =============================================================================
# Ingest Functions
# =============================================================================


def get_ingest_plugin(plugin_name: str) -> Type[Any]:
    """Get an ingestion plugin by name."""
    try:
        return INGEST_REGISTRY.get(plugin_name)
    except PluginNotFoundError as e:
        raise PluginRegistryError(str(e)) from e


def available_ingest_plugins() -> List[str]:
    """List available ingest plugins."""
    return INGEST_REGISTRY.list_available()


# =============================================================================
# Chunking Functions
# =============================================================================


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunking plugin by name."""
    try:
        return CHUNKING_REGISTRY.get(plugin_name)
    except PluginNotFoundError as e:
        raise PluginRegistryError(str(e)) from e


def available_chunking_plugins() -> List[str]:
    """List available chunking plugins."""
    return CHUNKING_REGISTRY.list_available()


# =============================================================================
# Retriever Functions
# =============================================================================


def get_retriever_plugin(plugin_name: str) -> Type[Any]:
    """Get a retriever plugin by name."""
    try:
        return RETRIEVER_REGISTRY.get(plugin_name)
    except PluginNotFoundError as e:
        raise PluginRegistryError(str(e)) from e


def available_retriever_plugins() -> List[str]:
    """List available retriever plugins."""
    return RETRIEVER_REGISTRY.list_available()


# =============================================================================
# Pipeline Functions
# =============================================================================


def get_pipeline_plugin(plugin_name: str) -> Type[Any]:
    """Get a pipeline plugin by name."""
    try:
        return PIPELINE_REGISTRY.get(plugin_name)
    except PluginNotFoundError as e:
        raise PluginRegistryError(str(e)) from e


def available_pipeline_plugins() -> List[str]:
    """List available pipeline plugins."""
    return PIPELINE_REGISTRY.list_available()