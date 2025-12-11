from __future__ import annotations

from typing import Dict

from fitz_rag.pipeline.base import PipelinePlugin
from fitz_rag.pipeline.plugins.debug import DebugPipelinePlugin
from fitz_rag.pipeline.plugins.easy import EasyPipelinePlugin
from fitz_rag.pipeline.plugins.fast import FastPipelinePlugin
from fitz_rag.pipeline.plugins.standard import StandardPipelinePlugin


_REGISTRY: Dict[str, PipelinePlugin] = {
    "debug": DebugPipelinePlugin(),
    "easy": EasyPipelinePlugin(),
    "fast": FastPipelinePlugin(),
    "standard": StandardPipelinePlugin(),
}


def get_pipeline_plugin(name: str) -> PipelinePlugin:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline plugin: {name!r}") from exc


def available_pipeline_plugins() -> list[str]:
    return sorted(_REGISTRY.keys())
