# core/llm/registry.py
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Iterable, Type

LLMPluginType = str  # "chat" | "embedding" | "rerank" | "vector_db"

LLM_REGISTRY: Dict[LLMPluginType, Dict[str, Type[Any]]] = {}
_DISCOVERED = False

_REQUIRED_METHOD: dict[str, str] = {
    "chat": "chat",
    "embedding": "embed",
    "rerank": "rerank",
    "vector_db": "search",
}


def register_llm_plugin(
    cls: Type[Any],
    *,
    plugin_name: str,
    plugin_type: LLMPluginType,
) -> None:
    """
    Optional manual registration hook.

    Not required when using Option A (class scanning discovery),
    but kept to avoid breaking older plugin modules.
    """
    _register(plugin_name=plugin_name, plugin_type=plugin_type, cls=cls)


def get_llm_plugin(*, plugin_name: str, plugin_type: LLMPluginType) -> Type[Any]:
    _auto_discover()
    try:
        return LLM_REGISTRY[plugin_type][plugin_name]
    except KeyError as exc:
        raise ValueError(f"Unknown {plugin_type} plugin: {plugin_name!r}") from exc


def available_llm_plugins(plugin_type: LLMPluginType) -> list[str]:
    _auto_discover()
    return sorted(LLM_REGISTRY.get(plugin_type, {}).keys())


def _auto_discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    # Scan all plugin packages where provider implementations may live.
    # (kept explicit to avoid accidental import of unrelated modules)
    for pkg_name in (
        "core.llm.plugins",
        "core.llm.chat.plugins",
        "core.llm.embedding.plugins",
        "core.llm.rerank.plugins",
        "core.vector_db.plugins",
    ):
        _scan_package(pkg_name)

    _DISCOVERED = True


def _scan_package(package_name: str) -> None:
    pkg = importlib.import_module(package_name)

    for module_info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for cls in _iter_plugin_classes(module):
            name = getattr(cls, "plugin_name")
            ptype = getattr(cls, "plugin_type")
            _register(plugin_name=name, plugin_type=ptype, cls=cls)


def _iter_plugin_classes(module: object) -> Iterable[type]:
    mod_name = getattr(module, "__name__", "")
    for obj in vars(module).values():
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod_name:
            continue

        plugin_name = getattr(obj, "plugin_name", None)
        plugin_type = getattr(obj, "plugin_type", None)

        if not isinstance(plugin_name, str) or not plugin_name:
            continue
        if not isinstance(plugin_type, str) or not plugin_type:
            continue

        required = _REQUIRED_METHOD.get(plugin_type)
        if required is None:
            continue

        fn = getattr(obj, required, None)
        if not callable(fn):
            continue

        yield obj


def _register(*, plugin_name: str, plugin_type: str, cls: Type[Any]) -> None:
    bucket = LLM_REGISTRY.setdefault(plugin_type, {})
    existing = bucket.get(plugin_name)

    if existing is not None and existing is not cls:
        raise ValueError(
            f"Duplicate {plugin_type} plugin_name={plugin_name!r}: "
            f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
        )

    bucket[plugin_name] = cls
