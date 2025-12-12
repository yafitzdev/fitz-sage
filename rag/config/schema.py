# rag/config/schema.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VectorDBConfig(BaseModel):
    plugin_name: str = Field(..., description="Vector DB plugin name (e.g. 'qdrant')")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin kwargs")


class LLMConfig(BaseModel):
    plugin_name: str = Field(..., description="Chat LLM plugin name")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin kwargs")


class EmbeddingConfig(BaseModel):
    plugin_name: str = Field(..., description="Embedding plugin name")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin kwargs")


class RerankConfig(BaseModel):
    enabled: bool = False
    plugin_name: str | None = Field(default=None, description="Rerank plugin name")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin kwargs")


class RetrieverConfig(BaseModel):
    plugin_name: str = Field(..., description="Retriever plugin name (e.g. 'dense')")
    collection: str = Field(..., description="Vector DB collection name")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")


class RGSSettings(BaseModel):
    max_chunks: int = 8
    max_chars: int = 4000
    include_citations: bool = True
    strict_grounding: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class RAGConfig(BaseModel):
    vector_db: VectorDBConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    rgs: RGSSettings = Field(default_factory=RGSSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RAGConfig":
        return cls(**raw)
