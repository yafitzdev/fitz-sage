# rag/pipeline/base.py
from __future__ import annotations

from typing import Protocol

from rag.config.schema import RAGConfig


class Pipeline(Protocol):
    def run(self, query: str): ...


class PipelinePlugin(Protocol):
    """
    Minimal interface for pipeline plugins.

    Contract:
    - Plugins MUST NOT implement wiring (LLM/vector_db/retriever construction).
    - Plugins may only mutate/override config, then delegate to RAGPipeline.from_config().
    - Return value must expose .run(query).
    """

    def build(self, cfg: RAGConfig) -> Pipeline:
        ...
