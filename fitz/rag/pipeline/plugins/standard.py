# rag/pipeline/plugins/standard.py
from __future__ import annotations

from dataclasses import dataclass

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE
from fitz.rag.config.schema import RAGConfig
from fitz.rag.pipeline.base import Pipeline, PipelinePlugin
from fitz.rag.pipeline.engine import RAGPipeline

logger = get_logger(__name__)


@dataclass
class StandardPipelinePlugin(PipelinePlugin):
    """
    Balanced default pipeline plugin.

    Must only delegate to RAGPipeline.from_config().
    """

    plugin_name: str = "standard"

    def build(self, cfg: RAGConfig) -> Pipeline:
        logger.info(f"{PIPELINE} Building Standard pipeline")
        return RAGPipeline.from_config(cfg)
