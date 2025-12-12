# tools/contract_map.py
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, get_type_hints
from typing import get_args, get_origin

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]


# -----------------------------
# Data structures
# -----------------------------


@dataclass(frozen=True, slots=True)
class ModelField:
    name: str
    type: str
    required: bool
    default: str | None = None


@dataclass(frozen=True, slots=True)
class ModelContract:
    module: str
    name: str
    fields: List[ModelField] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MethodContract:
    name: str
    signature: str
    returns: str | None


@dataclass(frozen=True, slots=True)
class ProtocolContract:
    module: str
    name: str
    methods: List[MethodContract] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RegistryContract:
    module: str
    name: str
    plugins: List[str] = field(default_factory=list)
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ContractMap:
    models: List[ModelContract] = field(default_factory=list)
    protocols: List[ProtocolContract] = field(default_factory=list)
    registries: List[RegistryContract] = field(default_factory=list)


# -----------------------------
# Helpers
# -----------------------------


def _safe_import(module: str) -> object | None:
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _fmt_type(tp: Any) -> str:
    if tp is None:
        return "None"

    origin = get_origin(tp)
    if origin is None:
        return getattr(tp, "__name__", None) or str(tp)

    args = get_args(tp)

    # Pretty-print unions (including Optional)
    if origin is type(None):  # pragma: no cover
        return "None"
    if str(origin).endswith("types.UnionType") or origin is getattr(__import__("typing"), "Union", None):
        inner = " | ".join(_fmt_type(a) for a in args)
        return inner

    origin_name = getattr(origin, "__name__", None) or str(origin)
    if args:
        return f"{origin_name}[{', '.join(_fmt_type(a) for a in args)}]"
    return origin_name


def _is_pydantic_model(obj: Any) -> bool:
    try:
        return isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    except Exception:
        return False


def _extract_pydantic_fields(model_cls: Type[Any]) -> List[ModelField]:
    # pydantic v2: model_fields; v1: __fields__
    fields: List[ModelField] = []

    if hasattr(model_cls, "model_fields"):
        mf: Dict[str, Any] = getattr(model_cls, "model_fields")
        for name in sorted(mf.keys()):
            f = mf[name]
            ann = getattr(f, "annotation", None)
            required = bool(getattr(f, "is_required", False))
            default = getattr(f, "default", None)
            fields.append(
                ModelField(
                    name=name,
                    type=_fmt_type(ann),
                    required=required,
                    default=None if required else repr(default),
                )
            )
        return fields

    if hasattr(model_cls, "__fields__"):
        ff: Dict[str, Any] = getattr(model_cls, "__fields__")
        for name in sorted(ff.keys()):
            f = ff[name]
            ann = getattr(f, "type_", None)
            required = bool(getattr(f, "required", False))
            default = getattr(f, "default", None)
            fields.append(
                ModelField(
                    name=name,
                    type=_fmt_type(ann),
                    required=required,
                    default=None if required else repr(default),
                )
            )
        return fields

    # fallback: annotations only
    hints = getattr(model_cls, "__annotations__", {}) or {}
    for name in sorted(hints.keys()):
        fields.append(ModelField(name=name, type=_fmt_type(hints[name]), required=True, default=None))
    return fields


def _looks_like_protocol(obj: Any) -> bool:
    if not isinstance(obj, type):
        return False
    # typing.Protocol has _is_protocol in stdlib implementations
    return bool(getattr(obj, "_is_protocol", False))


def _extract_protocol_methods(proto_cls: Type[Any]) -> List[MethodContract]:
    methods: List[MethodContract] = []
    # Only show methods defined on the protocol (not inherited from Protocol base)
    for name, member in inspect.getmembers(proto_cls):
        if name.startswith("_"):
            continue
        if not inspect.isfunction(member) and not inspect.ismethod(member):
            continue
        if getattr(member, "__qualname__", "").split(".")[0] != proto_cls.__name__:
            continue

        try:
            sig = str(inspect.signature(member))
        except Exception:
            sig = "(...)"

        returns: str | None = None
        try:
            hints = get_type_hints(member, include_extras=True)
            if "return" in hints:
                returns = _fmt_type(hints["return"])
        except Exception:
            returns = None

        methods.append(MethodContract(name=name, signature=sig, returns=returns))

    methods.sort(key=lambda m: m.name)
    return methods


def _maybe_call(module_obj: object, fn_name: str) -> None:
    fn = getattr(module_obj, fn_name, None)
    if callable(fn):
        try:
            fn()
        except Exception:
            return


def _extract_registry_plugins(
    module_name: str,
    *,
    dict_attr: str,
    discover_fns: Iterable[str] = (),
    note: str | None = None,
) -> RegistryContract | None:
    mod = _safe_import(module_name)
    if mod is None:
        return None

    for fn in discover_fns:
        _maybe_call(mod, fn)

    reg = getattr(mod, dict_attr, None)
    if not isinstance(reg, dict):
        return None

    plugins = sorted(str(k) for k in reg.keys())
    return RegistryContract(module=module_name, name=dict_attr, plugins=plugins, note=note)


def _extract_registry_via_list_fn(
    module_name: str,
    *,
    list_fn: str,
    note: str | None = None,
) -> RegistryContract | None:
    mod = _safe_import(module_name)
    if mod is None:
        return None

    fn = getattr(mod, list_fn, None)
    if not callable(fn):
        return None

    try:
        plugins = fn()
    except Exception:
        return None

    if not isinstance(plugins, list):
        return None

    return RegistryContract(
        module=module_name,
        name=list_fn,
        plugins=sorted(str(x) for x in plugins),
        note=note,
    )


# -----------------------------
# Collection
# -----------------------------


def build_contract_map() -> ContractMap:
    cm = ContractMap()

    # --- Models (pydantic) ---
    model_modules = [
        "rag.models.chunk",
        "rag.models.document",
        "rag.config.schema",
        "ingest.config.schema",
        "core.config.schema",
        "ingest.ingestion.base",
    ]
    for mod_name in model_modules:
        mod = _safe_import(mod_name)
        if mod is None:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue
            if _is_pydantic_model(obj):
                cm.models.append(
                    ModelContract(
                        module=mod_name,
                        name=obj.__name__,
                        fields=_extract_pydantic_fields(obj),
                    )
                )
            # dataclass-like models (RawDocument) show annotations as fields
            if obj.__name__ in {"RawDocument"} and hasattr(obj, "__annotations__"):
                fields = []
                hints = obj.__annotations__ or {}
                for k in sorted(hints.keys()):
                    fields.append(ModelField(name=k, type=_fmt_type(hints[k]), required=True))
                cm.models.append(ModelContract(module=mod_name, name=obj.__name__, fields=fields))

    cm.models.sort(key=lambda m: (m.module, m.name))

    # --- Protocols ---
    protocol_modules = [
        "rag.retrieval.base",
        "core.llm.chat.base",
        "core.llm.embedding.base",
        "core.llm.rerank.base",
        "ingest.chunking.base",
        "ingest.ingestion.base",
        "rag.pipeline.base",
    ]
    for mod_name in protocol_modules:
        mod = _safe_import(mod_name)
        if mod is None:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue
            if _looks_like_protocol(obj):
                cm.protocols.append(
                    ProtocolContract(
                        module=mod_name,
                        name=obj.__name__,
                        methods=_extract_protocol_methods(obj),
                    )
                )

    cm.protocols.sort(key=lambda p: (p.module, p.name))

    # --- Registries ---
    # Retrieval registry: lazy discovery inside get_retriever_plugin() via _auto_discover
    rr = _extract_registry_plugins(
        "rag.retrieval.registry",
        dict_attr="RETRIEVER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over rag.retrieval.plugins.*",
    )
    if rr:
        cm.registries.append(rr)

    # Pipeline registry: explicit dict
    pr = _extract_registry_via_list_fn(
        "rag.pipeline.registry",
        list_fn="available_pipeline_plugins",
        note="Explicit pipeline plugin registry",
    )
    if pr:
        cm.registries.append(pr)

    # Chunker registry: lazy discovery
    cr = _extract_registry_plugins(
        "ingest.chunking.registry",
        dict_attr="CHUNKER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.chunking.plugins.*",
    )
    if cr:
        cm.registries.append(cr)

    # Ingest registry: explicit dict
    ir = _extract_registry_plugins(
        "ingest.ingestion.registry",
        dict_attr="REGISTRY",
        discover_fns=(),
        note="Explicit ingestion plugin registry (registration via decorator)",
    )
    if ir:
        cm.registries.append(ir)

    # LLM central registry (if present)
    llm_reg = _extract_registry_plugins(
        "core.llm.registry",
        dict_attr="LLM_REGISTRY",
        discover_fns=(),
        note="Central plugin registry (chat/embedding/rerank/vector_db)",
    )
    if llm_reg:
        cm.registries.append(llm_reg)

    cm.registries.sort(key=lambda r: (r.module, r.name))
    return cm


# -----------------------------
# Rendering
# -----------------------------


def render_markdown(cm: ContractMap) -> str:
    lines: List[str] = []
    lines.append("# Contract Map")
    lines.append("")

    lines.append("## Models")
    for m in cm.models:
        lines.append(f"### `{m.module}.{m.name}`")
        if not m.fields:
            lines.append("- (no fields found)")
            lines.append("")
            continue
        for f in m.fields:
            req = "required" if f.required else "optional"
            default = "" if f.default is None else f" (default={f.default})"
            lines.append(f"- `{f.name}`: `{f.type}` — {req}{default}")
        lines.append("")

    lines.append("## Protocols")
    for p in cm.protocols:
        lines.append(f"### `{p.module}.{p.name}`")
        if not p.methods:
            lines.append("- (no methods found)")
            lines.append("")
            continue
        for m in p.methods:
            ret = f" -> {m.returns}" if m.returns else ""
            lines.append(f"- `{m.name}{m.signature}`{ret}")
        lines.append("")

    lines.append("## Registries")
    for r in cm.registries:
        lines.append(f"### `{r.module}.{r.name}`")
        if r.note:
            lines.append(f"- {r.note}")
        if not r.plugins:
            lines.append("- (no plugins found)")
            lines.append("")
            continue
        for name in r.plugins:
            lines.append(f"- `{name}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_json(cm: ContractMap) -> str:
    payload = asdict(cm)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# -----------------------------
# CLI
# -----------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a contract map from the current codebase.")
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write output to a file (otherwise prints to stdout)",
    )

    args = parser.parse_args(argv)

    cm = build_contract_map()

    if args.format == "json":
        text = render_json(cm)
    else:
        text = render_markdown(cm)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return 0

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
