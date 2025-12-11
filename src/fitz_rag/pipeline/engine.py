from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config

from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient

from fitz_rag.generation.rgs import RGS, RGSConfig as RGSRuntimeConfig
from fitz_rag.generation.rgs import RGSAnswer

from fitz_rag.exceptions.pipeline import PipelineError, RGSGenerationError
from fitz_rag.exceptions.llm import LLMError


class RAGPipeline:
    """
    Final v0.1.0 RAG pipeline using ONLY the unified config system.

    Pipeline:
        1. Retriever retrieves + reranks chunks
        2. RGS builds prompt
        3. LLM executes
        4. RGS builds answer structure
    """

    def __init__(self, retriever, llm, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

    # ---------------------------------------------------------
    # Main RAG execution
    # ---------------------------------------------------------
    def run(self, query: str) -> RGSAnswer:
        # 1) RETRIEVE
        try:
            chunks = self.retriever.retrieve(query)
        except Exception as e:
            raise PipelineError("Retriever failed") from e

        # 2) BUILD RGS PROMPT
        try:
            prompt = self.rgs.build_prompt(query, chunks)
        except Exception as e:
            raise RGSGenerationError("Failed to build RGS prompt") from e

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3) LLM CALL
        try:
            raw = self.llm.chat(messages)
        except Exception as e:
            raise LLMError("LLM chat operation failed") from e

        # 4) STRUCTURE ANSWER
        try:
            return self.rgs.build_answer(raw, chunks)
        except Exception as e:
            raise RGSGenerationError("Failed to build RGS answer") from e

    # ---------------------------------------------------------
    # Factory from unified config
    # ---------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        # Resolve API key (shared for embedding/chat/rerank)
        key = (
            cfg.llm.api_key
            or cfg.embedding.api_key
            or cfg.rerank.api_key
        )
        if key is None:
            raise PipelineError("No API key provided in any config section.")

        # ---------------- Qdrant Client ----------------
        qdrant = QdrantClient(
            host=cfg.retriever.qdrant_host,
            port=cfg.retriever.qdrant_port,
        )

        # ---------------- Embedder ---------------------
        if cfg.embedding.provider.lower() != "cohere":
            raise PipelineError(f"Unsupported embedding provider: {cfg.embedding.provider}")

        embedder = CohereEmbeddingClient(
            api_key=key,
            model=cfg.embedding.model,
        )

        # ---------------- Reranker (optional) ---------
        if cfg.rerank.enabled:
            if cfg.rerank.provider.lower() != "cohere":
                raise PipelineError(f"Unsupported rerank provider: {cfg.rerank.provider}")
            if not cfg.rerank.model:
                raise PipelineError("rerank.model must be set if reranking is enabled")

            reranker = CohereRerankClient(
                api_key=key,
                model=cfg.rerank.model,
            )
        else:
            reranker = None

        # ---------------- Chat LLM ---------------------
        if cfg.llm.provider.lower() != "cohere":
            raise PipelineError(f"Unsupported LLM provider: {cfg.llm.provider}")

        chat = CohereChatClient(
            api_key=key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
        )

        # ---------------- Retriever --------------------
        retriever = RAGRetriever(
            client=qdrant,
            embed_cfg=cfg.embedding,
            retriever_cfg=cfg.retriever,
            rerank_cfg=cfg.rerank,
        )
        # override embedder constructed inside retriever
        retriever.embedder = embedder
        retriever.reranker = reranker

        # ---------------- RGS --------------------------
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
        )
        rgs = RGS(config=rgs_cfg)

        return cls(retriever=retriever, llm=chat, rgs=rgs)


# ---------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------
def create_pipeline(name: str, config: Optional[RAGConfig] = None, **kwargs) -> RAGPipeline:
    """
    Modern factory:
        - If config is provided → unified RAG pipeline
        - Otherwise → backward-compatible wrappers (EasyRAG, FastRAG, etc.)
    """
    if config is not None:
        return RAGPipeline.from_config(config)

    name = name.lower()

    if name in ("easy", "simple"):
        from fitz_rag.pipeline.easy import EasyRAG
        return EasyRAG(**kwargs).pipeline

    if name in ("fast",):
        from fitz_rag.pipeline.fast import FastRAG
        return FastRAG(**kwargs).pipeline

    if name in ("standard", "default"):
        from fitz_rag.pipeline.standard import StandardRAG
        return StandardRAG(**kwargs).pipeline

    if name in ("debug", "verbose"):
        from fitz_rag.pipeline.debug import DebugRAG
        return DebugRAG(**kwargs).pipeline

    raise PipelineError(f"Unknown pipeline type: {name}")


def create_pipeline_from_yaml(user_config_path: Optional[str] = None) -> RAGPipeline:
    raw = load_config(user_config_path=user_config_path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)
