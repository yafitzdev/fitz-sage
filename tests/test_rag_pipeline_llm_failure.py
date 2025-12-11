# tests/test_rag_pipeline_llm_failure.py

import pytest

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.exceptions.llm import LLMError
from fitz_rag.generation.rgs import RGS, RGSConfig

class DummyRetriever:
    def retrieve(self, q): return [{"id": "1", "text": "hello", "metadata": {}}]

class FailingLLM:
    def chat(self, msgs):
        raise RuntimeError("LLM is down")

def test_rag_pipeline_llm_failure():
    pipe = RAGPipeline(
        retriever=DummyRetriever(),
        llm=FailingLLM(),
        rgs=RGS(RGSConfig())
    )

    with pytest.raises(LLMError):
        pipe.run("hello?")
