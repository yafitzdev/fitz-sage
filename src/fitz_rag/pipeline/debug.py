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
class DebugRAG:
    """
    Debug-oriented RAG pipeline for introspection.

    Preferred:
        rag = DebugRAG(config=my_cfg)

    Legacy:
        rag = DebugRAG(collection="docs")
    """

    config: Optional[RAGConfig] = None

    # Legacy overrides
    collection: Optional[str] = None
    cohere_api_key: Optional[str] = None
    top_k: int = 20
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        # -------------------------------------
        # 1) Unified config path
        # -------------------------------------
        if self.config is not None:
            logger.info(f"{PIPELINE} Initializing DebugRAG from explicit config")
            self.pipeline = RAGPipeline.from_config(self.config)
            return

        # -------------------------------------
        # 2) Legacy path (for backward compat)
        # -------------------------------------
        logger.info(f"{PIPELINE} Initializing DebugRAG using legacy parameters")

        raw = load_config()

        raw["retriever"]["collection"] = self.collection
        raw["retriever"]["top_k"] = self.top_k

        if self.cohere_api_key:
            logger.debug(f"{PIPELINE} Overriding API keys with provided cohere_api_key")
            raw["llm"]["api_key"] = self.cohere_api_key
            raw["embedding"]["api_key"] = self.cohere_api_key
            raw["rerank"]["api_key"] = self.cohere_api_key

        cfg = RAGConfig.from_dict(raw)
        self.pipeline = RAGPipeline.from_config(cfg)

    # -------------------------------------
    # Developer utilities
    # -------------------------------------
    def explain(self, query: str):
        """
        Returns:
          - retrieved chunks
          - RGS prompt
          - model answer
        Useful for debugging how RAG arrives at a conclusion.
        """

        logger.info(f"{PIPELINE} Debug explain() called for query='{query}'")

        # Step 1: retrieve chunks
        logger.debug(f"{PIPELINE} Retrieving chunks")
        chunks = self.pipeline.retriever.retrieve(query)

        # Step 2: build prompt
        logger.debug(f"{PIPELINE} Building RGS prompt")
        prompt = self.pipeline.rgs.build_prompt(query, chunks)

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user",  "content": prompt.user},
        ]

        # Step 3: LLM answer
        logger.debug(f"{PIPELINE} Querying LLM for answer")
        raw = self.pipeline.llm.chat(messages)

        logger.info(f"{PIPELINE} explain() completed")

        return {
            "query": query,
            "chunks": chunks,
            "prompt": prompt,
            "answer": raw,
        }
