# pipeline/pipeline/base.py
from __future__ import annotations

from typing import Protocol

from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig


class Pipeline(Protocol):
    def run(self, query: str): ...


class PipelinePlugin(Protocol):
    """
    Contract:
    - Plugins do NOT implement wiring.
    - Plugins may only adjust config, then delegate to RAGPipeline.from_config(cfg).
    """

    plugin_name: str

    def build(self, cfg: FitzRagConfig) -> Pipeline: ...
