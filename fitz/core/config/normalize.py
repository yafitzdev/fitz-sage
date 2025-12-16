# fitz/core/config/normalize.py
from __future__ import annotations

from typing import Any


def normalize_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """
    Convert human-facing preset config into engine-facing config.
    """

    out: dict[str, Any] = {}

    # LLM block (chat / embedding / rerank)
    if "llm" in preset:
        # for now, collapse to chat plugin
        # (later this can expand cleanly)
        llm_block = preset["llm"]
        if "chat" not in llm_block:
            raise ValueError("Preset llm must define chat plugin")

        out["llm"] = {
            "plugin_name": llm_block["chat"]["plugin"],
            "kwargs": {},
        }

    # vector_db
    if "vector_db" in preset:
        out["vector_db"] = {
            "plugin_name": preset["vector_db"]["plugin"],
            "kwargs": {},
        }

    # pipeline
    if "pipeline" in preset:
        out["pipeline"] = {
            "plugin_name": preset["pipeline"]["plugin"],
            "kwargs": {},
        }

    return out
