from __future__ import annotations

from dataclasses import dataclass

from fitz_rag.config.schema import RAGConfig
from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.pipeline.base import PipelinePlugin

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class StandardPipelinePlugin(PipelinePlugin):
    """
    Balanced default pipeline plugin.

    Mirrors the behavior of the standard RAG pipeline using the provided
    config without additional mutations.
    """

    def build(self, cfg: RAGConfig) -> RAGPipeline:
        logger.info(f"{PIPELINE} Building Standard pipeline")
        return RAGPipeline.from_config(cfg)
