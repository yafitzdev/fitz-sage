# fitz/core/registry.py
"""
Centralized Plugin Registry System.

This module provides a generic, reusable plugin registry that eliminates
code duplication across the codebase. Previously, we had 5+ nearly identical
registry implementations:
- fitz/llm/registry.py
- fitz/vector_db/registry.py
- fitz/ingest/ingestion/registry.py
- fitz/ingest/chunking/registry.py
- fitz/engines/classic_rag/retrieval/runtime/registry.py
- fitz/engines/classic_rag/pipeline/pipeline/registry.py

Now there's ONE registry implementation that all can use.

Usage:
    # Create a registry for your plugin type
    from fitz.core.registry import PluginRegistry

    chat_registry = PluginRegistry(
        name="chat",
        required_attrs=["plugin_name", "plugin_type"],
        required_method="chat",
        scan_packages=["fitz.llm.chat.plugins"],
    )

    # Get a plugin
    plugin_cls = chat_registry.get("cohere")

    # List available plugins
    available = chat_registry.list_available()

Design Principles:
    - NO SILENT FALLBACK: If you ask for "cohere", you get cohere or an error
    - Lazy discovery: Plugins are discovered on first access
    - Duplicate detection: Raises error if same plugin_name registered twice
    - Clear error messages: Tells you what's available when lookup fails
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PluginRegistryError(Exception):
    """Raised when plugin lookup or registration fails."""
    pass


class DuplicatePluginError(PluginRegistryError):
    """Raised when two plugins have the same name."""
    pass


class PluginNotFoundError(PluginRegistryError):
    """Raised when requested plugin doesn't exist."""
    pass


@dataclass
class PluginRegistry(Generic[T]):
    """
    Generic plugin registry with auto-discovery.

    This is the SINGLE implementation of plugin registry logic.
    All plugin types (chat, embedding, vector_db, etc.) use this.

    Attributes:
        name: Human-readable name for this registry (for error messages)
        required_attrs: Class attributes that must exist (e.g., ["plugin_name"])
        required_method: Method that must be callable (e.g., "chat", "embed")
        scan_packages: List of package paths to scan for plugins
        attr_filter: Optional dict of {attr: value} that must match

    Example:
        # For chat plugins
        registry = PluginRegistry(
            name="chat",
            required_attrs=["plugin_name", "plugin_type"],
            required_method="chat",
            scan_packages=["fitz.llm.chat.plugins"],
            attr_filter={"plugin_type": "chat"},
        )

        # For vector_db plugins
        registry = PluginRegistry(
            name="vector_db",
            required_attrs=["plugin_name"],
            required_method="search",
            scan_packages=["fitz.vector_db.plugins"],
        )
    """

    name: str
    required_attrs: List[str] = field(default_factory=lambda: ["plugin_name"])
    required_method: Optional[str] = None
    scan_packages: List[str] = field(default_factory=list)
    attr_filter: Dict[str, Any] = field(default_factory=dict)
    plugin_name_attr: str = "plugin_name"

    # Internal state
    _registry: Dict[str, Type[T]] = field(default_factory=dict, init=False, repr=False)
    _discovered: bool = field(default=False, init=False, repr=False)

    def get(self, name: str) -> Type[T]:
        """
        Get a plugin by exact name.

        No fallback, no magic - returns exactly what you ask for or raises.

        Args:
            name: Exact plugin name (e.g., "cohere", "qdrant", "local")

        Returns:
            Plugin class

        Raises:
            PluginNotFoundError: If plugin doesn't exist
        """
        self._ensure_discovered()

        if name not in self._registry:
            available = sorted(self._registry.keys())
            raise PluginNotFoundError(
                f"Unknown {self.name} plugin: {name!r}. "
                f"Available: {available}"
            )

        return self._registry[name]

    def list_available(self) -> List[str]:
        """List all available plugin names."""
        self._ensure_discovered()
        return sorted(self._registry.keys())

    def register(self, cls: Type[T]) -> None:
        """
        Manually register a plugin class.

        Useful for testing or dynamic registration.

        Args:
            cls: Plugin class to register

        Raises:
            DuplicatePluginError: If plugin name already registered
        """
        name = getattr(cls, self.plugin_name_attr, None)
        if not name:
            raise PluginRegistryError(
                f"Cannot register {cls}: missing {self.plugin_name_attr} attribute"
            )

        existing = self._registry.get(name)
        if existing is not None and existing is not cls:
            raise DuplicatePluginError(
                f"Duplicate {self.name} plugin_name={name!r}: "
                f"{existing.__module__}.{existing.__name__} vs "
                f"{cls.__module__}.{cls.__name__}"
            )

        self._registry[name] = cls
        logger.debug(f"Registered {self.name} plugin: {name}")

    def is_discovered(self) -> bool:
        """Check if discovery has run."""
        return self._discovered

    def reset(self) -> None:
        """Reset registry (mainly for testing)."""
        self._registry.clear()
        self._discovered = False

    def _ensure_discovered(self) -> None:
        """Run discovery if not already done."""
        if self._discovered:
            return

        for pkg_name in self.scan_packages:
            self._scan_package(pkg_name)

        logger.debug(
            f"Discovered {self.name} plugins: {list(self._registry.keys())}"
        )
        self._discovered = True

    def _scan_package(self, package_name: str) -> None:
        """Scan a package for plugin classes."""
        try:
            pkg = importlib.import_module(package_name)
        except ImportError as e:
            logger.debug(f"Could not import {package_name}: {e}")
            return

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            # Single module, not a package - scan it directly
            self._scan_module(pkg)
            return

        # Iterate over submodules
        for module_info in pkgutil.iter_modules(pkg_path):
            module_name = f"{package_name}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
                self._scan_module(module)
            except Exception as e:
                logger.debug(f"Could not import {module_name}: {e}")
                continue

    def _scan_module(self, module: object) -> None:
        """Scan a module for plugin classes."""
        mod_name = getattr(module, "__name__", "")

        for obj in vars(module).values():
            if not isinstance(obj, type):
                continue

            # Must be defined in this module (not imported)
            if getattr(obj, "__module__", None) != mod_name:
                continue

            # Check if it's a valid plugin
            if self._is_valid_plugin(obj):
                self.register(obj)

    def _is_valid_plugin(self, cls: type) -> bool:
        """Check if a class is a valid plugin for this registry."""
        # Check required attributes
        for attr in self.required_attrs:
            val = getattr(cls, attr, None)
            if not isinstance(val, str) or not val:
                return False

        # Check attribute filters (e.g., plugin_type == "chat")
        for attr, expected in self.attr_filter.items():
            actual = getattr(cls, attr, None)
            if actual != expected:
                return False

        # Check required method
        if self.required_method:
            method = getattr(cls, self.required_method, None)
            if not callable(method):
                return False

        return True


# =============================================================================
# Pre-configured Registries
# =============================================================================
# These are singleton instances for each plugin type.
# Import and use these directly instead of creating new registries.


# LLM Chat plugins
CHAT_REGISTRY = PluginRegistry[Any](
    name="chat",
    required_attrs=["plugin_name", "plugin_type"],
    required_method="chat",
    scan_packages=["fitz.llm.chat.plugins"],
    attr_filter={"plugin_type": "chat"},
)

# LLM Embedding plugins
EMBEDDING_REGISTRY = PluginRegistry[Any](
    name="embedding",
    required_attrs=["plugin_name", "plugin_type"],
    required_method="embed",
    scan_packages=["fitz.llm.embedding.plugins"],
    attr_filter={"plugin_type": "embedding"},
)

# LLM Rerank plugins
RERANK_REGISTRY = PluginRegistry[Any](
    name="rerank",
    required_attrs=["plugin_name", "plugin_type"],
    required_method="rerank",
    scan_packages=["fitz.llm.rerank.plugins"],
    attr_filter={"plugin_type": "rerank"},
)

# Vector DB plugins
VECTOR_DB_REGISTRY = PluginRegistry[Any](
    name="vector_db",
    required_attrs=["plugin_name"],
    required_method="search",
    scan_packages=["fitz.vector_db.plugins"],
)

# Retrieval plugins
RETRIEVAL_REGISTRY = PluginRegistry[Any](
    name="retrieval",
    required_attrs=["plugin_name"],
    required_method="retrieve",
    scan_packages=["fitz.engines.classic_rag.retrieval.runtime.plugins"],
)

# Pipeline plugins
PIPELINE_REGISTRY = PluginRegistry[Any](
    name="pipeline",
    required_attrs=["plugin_name"],
    required_method="build",
    scan_packages=["fitz.engines.classic_rag.pipeline.pipeline.plugins"],
)

# Ingestion plugins
INGEST_REGISTRY = PluginRegistry[Any](
    name="ingest",
    required_attrs=["plugin_name"],
    required_method="ingest",
    scan_packages=["fitz.ingest.ingestion.plugins"],
)

# Chunking plugins
CHUNKING_REGISTRY = PluginRegistry[Any](
    name="chunking",
    required_attrs=["plugin_name"],
    required_method="chunk_text",
    scan_packages=["fitz.ingest.chunking.plugins"],
)


# =============================================================================
# Convenience Functions
# =============================================================================
# These provide a familiar API and handle the LLM registry multiplexing.


def get_llm_plugin(*, plugin_name: str, plugin_type: str) -> Type[Any]:
    """
    Get an LLM plugin by name and type.

    This is the main entry point for LLM plugins (chat, embedding, rerank).

    Args:
        plugin_name: Plugin name (e.g., "cohere", "openai", "local")
        plugin_type: Plugin type ("chat", "embedding", "rerank")

    Returns:
        Plugin class

    Raises:
        PluginNotFoundError: If plugin doesn't exist
        ValueError: If plugin_type is invalid
    """
    registry = _get_llm_registry(plugin_type)
    return registry.get(plugin_name)


def available_llm_plugins(plugin_type: str) -> List[str]:
    """List available plugins for an LLM type."""
    registry = _get_llm_registry(plugin_type)
    return registry.list_available()


def _get_llm_registry(plugin_type: str) -> PluginRegistry:
    """Get the appropriate registry for an LLM plugin type."""
    registries = {
        "chat": CHAT_REGISTRY,
        "embedding": EMBEDDING_REGISTRY,
        "rerank": RERANK_REGISTRY,
    }

    if plugin_type not in registries:
        raise ValueError(
            f"Invalid LLM plugin_type: {plugin_type!r}. "
            f"Must be one of: {list(registries.keys())}"
        )

    return registries[plugin_type]


def get_vector_db_plugin(plugin_name: str) -> Type[Any]:
    """Get a vector DB plugin by name."""
    return VECTOR_DB_REGISTRY.get(plugin_name)


def available_vector_db_plugins() -> List[str]:
    """List available vector DB plugins."""
    return VECTOR_DB_REGISTRY.list_available()


def get_retriever_plugin(plugin_name: str) -> Type[Any]:
    """Get a retrieval plugin by name."""
    return RETRIEVAL_REGISTRY.get(plugin_name)


def available_retriever_plugins() -> List[str]:
    """List available retrieval plugins."""
    return RETRIEVAL_REGISTRY.list_available()


def get_pipeline_plugin(plugin_name: str) -> Type[Any]:
    """Get a pipeline plugin by name."""
    return PIPELINE_REGISTRY.get(plugin_name)


def available_pipeline_plugins() -> List[str]:
    """List available pipeline plugins."""
    return PIPELINE_REGISTRY.list_available()


def get_ingest_plugin(plugin_name: str) -> Type[Any]:
    """Get an ingestion plugin by name."""
    return INGEST_REGISTRY.get(plugin_name)


def available_ingest_plugins() -> List[str]:
    """List available ingestion plugins."""
    return INGEST_REGISTRY.list_available()


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunking plugin by name."""
    return CHUNKING_REGISTRY.get(plugin_name)


def available_chunking_plugins() -> List[str]:
    """List available chunking plugins."""
    return CHUNKING_REGISTRY.list_available()


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================
# These match the old function signatures so existing code doesn't break.


def resolve_llm_plugin(*, plugin_type: str, requested_name: str) -> Type[Any]:
    """Alias for get_llm_plugin (backwards compatibility)."""
    return get_llm_plugin(plugin_name=requested_name, plugin_type=plugin_type)


def resolve_vector_db_plugin(requested_name: str) -> Type[Any]:
    """Alias for get_vector_db_plugin (backwards compatibility)."""
    return get_vector_db_plugin(requested_name)