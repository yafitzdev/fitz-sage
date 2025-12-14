# tests/test_rag_pipeline_llm_failure.py

import pytest

from core.exceptions.llm import LLMError
from rag.generation.rgs import RGS, RGSConfig
from rag.pipeline.engine import RAGPipeline


class DummyRetriever:
    def retrieve(self, q):
        return [{"id": "1", "text": "hello", "metadata": {}}]


class FailingLLM:
    def chat(self, msgs):
        raise RuntimeError("LLM is down")


def test_rag_pipeline_llm_failure():
    pipe = RAGPipeline(retriever=DummyRetriever(), llm=FailingLLM(), rgs=RGS(RGSConfig()))

    with pytest.raises(LLMError):
        pipe.run("hello?")
