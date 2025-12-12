# fitz_rag/pipeline/engine.py
from __future__ import annotations

from typing import Any, Optional

from rag.config.loader import load_config
from rag.config.schema import RAGConfig
from rag.exceptions.llm import LLMError
from rag.exceptions.pipeline import PipelineError, RGSGenerationError
from rag.generation.rgs import RGS, RGSAnswer, RGSConfig as RGSRuntimeConfig
from rag.retrieval.engine import RetrieverEngine

from core.llm.chat import ChatEngine
from core.llm.embedding.engine import EmbeddingEngine
from core.llm.registry import get_llm_plugin
from core.llm.rerank.engine import RerankEngine
from core.vector_db import get_vector_db_plugin

from core.logging.logger import get_logger
from core.logging.tags import PIPELINE, VECTOR_DB

logger = get_logger(__name__)


def _cfg_to_kwargs(cfg: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    if cfg is None:
        return {}

    exclude = exclude or set()

    if hasattr(cfg, "model_dump"):
        data = cfg.model_dump(exclude_none=True)
    elif hasattr(cfg, "dict"):
        data = cfg.dict(exclude_none=True)
    elif isinstance(cfg, dict):
        data = {k: v for k, v in cfg.items() if v is not None}
    else:
        data = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and v is not None}

    for k in exclude:
        data.pop(k, None)

    data.pop("plugin_name", None)
    return data


class RAGPipeline:
    """
    Final clean RAG pipeline.
    Architecture:
        vector_db → retrieval → rgs → llm → final answer
    """

    def __init__(self, retriever: RetrieverEngine, llm: ChatEngine, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

        logger.info(f"{PIPELINE} RAGPipeline initialized")

    def run(self, query: str) -> RGSAnswer:
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # 1) Retrieve
        try:
            chunks = self.retriever.retrieve(query)
        except Exception as e:
            logger.error(f"{PIPELINE} Retriever failed: {e}")
            raise PipelineError("Retriever failed") from e

        # 2) Build RGS prompt
        try:
            prompt = self.rgs.build_prompt(query, chunks)
        except Exception as e:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {e}")
            raise RGSGenerationError("Failed to build RGS prompt") from e

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3) LLM call
        try:
            raw = self.llm.chat(messages)
        except Exception as e:
            logger.error(f"{PIPELINE} LLM chat failed: {e}")
            raise LLMError("LLM chat operation failed") from e

        # 4) Build RGS answer
        try:
            answer = self.rgs.build_answer(raw, chunks)
            logger.info(f"{PIPELINE} Pipeline run completed")
            return answer
        except Exception as e:
            logger.error(f"{PIPELINE} Failed to structure RGS answer: {e}")
            raise RGSGenerationError("Failed to build RGS answer") from e

    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        # ---------------- Vector DB client ----------------
        VectorDBCls = get_vector_db_plugin("qdrant")
        vector_client = VectorDBCls(
            host=cfg.retriever.qdrant_host,
            port=cfg.retriever.qdrant_port,
        )
        logger.info(
            f"{VECTOR_DB} Using vector DB plugin='qdrant', "
            f"collection='{cfg.retriever.collection}'"
        )

        # ---------------- Chat Engine ------------------
        chat_engine = ChatEngine.from_name(
            cfg.llm.provider.lower(),
            api_key=cfg.llm.api_key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
        )

        # ---------------- Embedder (resolved upstream) ----------------
        embed_provider = cfg.llm.provider.lower()
        EmbedCls = get_llm_plugin(plugin_name=embed_provider, plugin_type="embedding")
        embed_plugin = EmbedCls(**_cfg_to_kwargs(cfg.embedding))
        embedder = EmbeddingEngine(embed_plugin)

        # ---------------- Optional rerank (resolved upstream) ----------
        rerank_engine: RerankEngine | None = None
        if getattr(cfg.rerank, "enabled", False):
            RerankCls = get_llm_plugin(plugin_name=embed_provider, plugin_type="rerank")
            rerank_plugin = RerankCls(**_cfg_to_kwargs(cfg.rerank, exclude={"enabled"}))
            rerank_engine = RerankEngine(rerank_plugin)

        # ---------------- Retriever (via registry) --------------------
        retriever_name = getattr(cfg.retriever, "plugin_name", None) or "dense"
        retriever = RetrieverEngine.from_name(
            retriever_name,
            client=vector_client,
            retriever_cfg=cfg.retriever,
            embedder=embedder,
            rerank_engine=rerank_engine,
        )

        # ---------------- RGS Config -------------------
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
            include_query_in_context=cfg.rgs.include_query_in_context,
            source_label_prefix=getattr(cfg.rgs, "source_label_prefix", "S"),
        )
        rgs = RGS(config=rgs_cfg)

        logger.info(f"{PIPELINE} RAGPipeline successfully created")
        return cls(retriever=retriever, llm=chat_engine, rgs=rgs)


def create_pipeline_from_yaml(path: Optional[str] = None) -> RAGPipeline:
    logger.debug(f"{PIPELINE} Loading config from YAML")
    raw = load_config(path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)
