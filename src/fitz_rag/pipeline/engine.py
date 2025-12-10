from __future__ import annotations

from typing import List
from fitz_rag.models.chunk import Chunk
from fitz_rag.generation.rgs import RGS, RGSAnswer


class RAGPipeline:
    """
    Final v0.1.0 RAG pipeline:
        retriever → chunks
        RGS → prompt
        LLM → answer
        RGS → structured answer
    """

    def __init__(self, retriever, llm, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

    def run(self, query: str) -> RGSAnswer:
        # 1. Retrieve + rerank (retriever handles all of it)
        chunks: List[Chunk] = self.retriever.retrieve(query)

        # 2. Build prompt using RGS
        prompt = self.rgs.build_prompt(query, chunks)

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3. LLM call
        raw = self.llm.chat(messages)

        # 4. Structured answer
        return self.rgs.build_answer(raw, chunks)


def create_pipeline(name: str, **kwargs) -> RAGPipeline:
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

    raise ValueError(f"Unknown pipeline type: {name}")
