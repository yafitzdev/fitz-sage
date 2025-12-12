# rag/config/loader.py
from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files as pkg_files
from typing import Any

import yaml

from rag.config.schema import RAGConfig


class ConfigError(RuntimeError):
    pass


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_file(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise ConfigError(f"Failed to load config file: {path}") from exc


def _load_default_config() -> dict[str, Any]:
    try:
        default_path = pkg_files("rag.config").joinpath("default.yaml")
        return yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigError("Failed to load bundled rag default config") from exc


@lru_cache
def load_config(path: str | None = None) -> dict[str, Any]:
    config = _load_default_config()

    override_path = path or os.getenv("FITZ_RAG_CONFIG")
    if override_path:
        override = _load_yaml_file(override_path)
        config = _deep_merge(config, override)

    return config


def load_rag_config(path: str | None = None) -> RAGConfig:
    raw = load_config(path)
    try:
        return RAGConfig.from_dict(raw)
    except Exception as exc:
        raise ConfigError("Invalid rag configuration") from exc
