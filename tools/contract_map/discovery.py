# tools/contract_map/discovery.py
"""Plugin discovery and namespace scanning."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, List, Tuple

from .common import DiscoveryReport

# ---------------------------------------------------------------------------
# Plugin predicates (declared, not inferred)
# ---------------------------------------------------------------------------

PluginPredicate = Tuple[Callable[[type], bool], Callable[[type], str]]


def _simple_plugin_id(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


PLUGIN_PREDICATES: Dict[str, PluginPredicate] = {
    # --- LLM plugins (new fitz.llm.* paths) ---
    "fitz.llm.chat.plugins": (
        lambda cls: (
            isinstance(cls, type)
            and isinstance(getattr(cls, "plugin_name", None), str)
            and getattr(cls, "plugin_type", None) == "chat"
            and callable(getattr(cls, "chat", None))
        ),
        _simple_plugin_id,
    ),
    "fitz.llm.embedding.plugins": (
        lambda cls: (
            isinstance(cls, type)
            and isinstance(getattr(cls, "plugin_name", None), str)
            and getattr(cls, "plugin_type", None) == "embedding"
            and callable(getattr(cls, "embed", None))
        ),
        _simple_plugin_id,
    ),
    "fitz.llm.rerank.plugins": (
        lambda cls: (
            isinstance(cls, type)
            and isinstance(getattr(cls, "plugin_name", None), str)
            and getattr(cls, "plugin_type", None) == "rerank"
            and callable(getattr(cls, "rerank", None))
        ),
        _simple_plugin_id,
    ),
    # --- vector DB (new fitz.vector_db.* path) ---
    "fitz.vector_db.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and getattr(cls, "plugin_type", None) == "vector_db"
            and callable(getattr(cls, "search", None))
        ),
        _simple_plugin_id,
    ),
    # --- retrieval (under engines/classic_rag) ---
    "fitz.engines.classic_rag.retrieval.runtime.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "retrieve", None))
        ),
        _simple_plugin_id,
    ),
    # --- pipeline (under engines/classic_rag) ---
    "fitz.engines.classic_rag.pipeline.pipeline.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "build", None))
        ),
        _simple_plugin_id,
    ),
    # --- ingest ---
    "fitz.ingest.chunking.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "chunk_text", None))
        ),
        _simple_plugin_id,
    ),
    "fitz.ingest.ingestion.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "ingest", None))
        ),
        _simple_plugin_id,
    ),
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def scan_discovery(namespace: str, note: str) -> DiscoveryReport:
    """Scan a namespace for plugins using declared discovery predicates."""
    failures: List[str] = []
    duplicates: List[str] = []
    found: Dict[str, str] = {}
    modules_scanned = 0

    try:
        pkg = importlib.import_module(namespace)
    except Exception as exc:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[f"{namespace}: {type(exc).__name__}: {exc}"],
            duplicates=[],
        )

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[],
            duplicates=[],
        )

    predicate, plugin_id = PLUGIN_PREDICATES.get(
        namespace,
        (lambda _: False, _simple_plugin_id),
    )

    for mod_info in pkgutil.iter_modules(pkg_path):
        modules_scanned += 1
        mod_name = f"{namespace}.{mod_info.name}"

        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            failures.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue

        actual_name = getattr(mod, "__name__", mod_name)
        for obj in vars(mod).values():
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__module__", None) != actual_name:
                continue
            if not predicate(obj):
                continue

            name = getattr(obj, "plugin_name")
            pid = plugin_id(obj)

            existing = found.get(name)
            if existing and existing != pid:
                duplicates.append(f"{name!r}: {existing} vs {pid}")
            else:
                found[name] = pid

    return DiscoveryReport(
        namespace=namespace,
        note=note,
        modules_scanned=modules_scanned,
        plugins_found=[f"{k} -> {found[k]}" for k in sorted(found)],
        failures=sorted(failures),
        duplicates=sorted(duplicates),
    )


def scan_all_discoveries() -> List[DiscoveryReport]:
    """Scan all declared plugin namespaces."""
    return [
        scan_discovery("fitz.llm.chat.plugins", "LLM chat plugins (Option A discovery)"),
        scan_discovery("fitz.llm.embedding.plugins", "LLM embedding plugins (Option A discovery)"),
        scan_discovery("fitz.llm.rerank.plugins", "LLM rerank plugins (Option A discovery)"),
        scan_discovery("fitz.vector_db.plugins", "Vector DB plugins (Option A discovery)"),
        scan_discovery(
            "fitz.engines.classic_rag.retrieval.runtime.plugins",
            "RAG retriever plugins (Option A discovery)",
        ),
        scan_discovery(
            "fitz.engines.classic_rag.pipeline.pipeline.plugins",
            "RAG pipeline plugins (Option A discovery)",
        ),
        scan_discovery("fitz.ingest.chunking.plugins", "Ingest chunking plugins (Option A discovery)"),
        scan_discovery("fitz.ingest.ingestion.plugins", "Ingest ingestion plugins (Option A discovery)"),
    ]


def render_discovery_section(reports: List[DiscoveryReport]) -> str:
    """Render the Discovery Report section."""
    lines = ["## Discovery Report"]

    for r in reports:
        lines.append(f"### `{r.namespace}`")
        if r.note:
            lines.append(f"- {r.note}")
        lines.append(f"- modules_scanned: `{r.modules_scanned}`")

        if r.plugins_found:
            lines.append("- plugins:")
            for p in r.plugins_found:
                lines.append(f"  - `{p}`")

        if r.failures:
            lines.append("- failures:")
            for f in r.failures:
                lines.append(f"  - `{f}`")

        if r.duplicates:
            lines.append("- duplicates:")
            for d in r.duplicates:
                lines.append(f"  - `{d}`")

        lines.append("")

    return "\n".join(lines)