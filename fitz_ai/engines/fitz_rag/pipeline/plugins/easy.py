# pipeline/pipeline/plugins/easy.py
from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
from fitz_ai.engines.fitz_rag.pipeline.base import Pipeline, PipelinePlugin
from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class EasyPipelinePlugin(PipelinePlugin):
    """
    Minimal convenience pipeline plugin.

    Must only delegate to RAGPipeline.from_config().
    """

    plugin_name: str = "easy"

    def build(self, cfg: FitzRagConfig) -> Pipeline:
        logger.info(f"{PIPELINE} Building Easy pipeline")
        return RAGPipeline.from_config(cfg)
