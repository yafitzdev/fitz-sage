from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config
from fitz_rag.pipeline.engine import RAGPipeline

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class StandardRAG:
    """
    Standard RAG preset — configurable but user-friendly.

    Preferred:
        rag = StandardRAG(config=my_cfg)

    Backward compatible:
        rag = StandardRAG(collection="docs", top_k=20)
    """

    # New unified config
    config: Optional[RAGConfig] = None

    # Legacy parameters (optional)
    collection: Optional[str] = None
    cohere_api_key: Optional[str] = None

    top_k: int = 20
    final_top_k: int = 5
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        # -------------------------------------
        # 1) Unified CONFIG path
        # -------------------------------------
        if self.config is not None:
            logger.info(f"{PIPELINE} Initializing StandardRAG from explicit config")
            self.pipeline = RAGPipeline.from_config(self.config)
            return

        # -------------------------------------
        # 2) Legacy fallback path
        # -------------------------------------
        logger.info(f"{PIPELINE} Initializing StandardRAG using legacy parameters")

        raw = load_config()

        raw["retriever"]["collection"] = self.collection
        raw["retriever"]["top_k"] = self.top_k

        if self.cohere_api_key:
            logger.debug(f"{PIPELINE} Overriding API keys with provided cohere_api_key")
            raw["llm"]["api_key"] = self.cohere_api_key
            raw["embedding"]["api_key"] = self.cohere_api_key
            raw["rerank"]["api_key"] = self.cohere_api_key

        cfg = RAGConfig.from_dict(raw)

        logger.debug(f"{PIPELINE} Constructing StandardRAG pipeline (top_k={self.top_k})")
        self.pipeline = RAGPipeline.from_config(cfg)

    # -------------------------------------
    # User API
    # -------------------------------------
    def ask(self, query: str):
        logger.info(f"{PIPELINE} StandardRAG.ask called (query='{query[:50]}...')")
        return self.pipeline.run(query)
