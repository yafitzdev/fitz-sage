# rag/config/schema.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def _validate_enabled_plugin_name(self) -> "RerankConfig":
        if self.enabled and not self.plugin_name:
            raise ValueError("rerank.plugin_name must be set when rerank.enabled=True")
        return self


class RetrieverConfig(BaseModel):
    plugin_name: str = Field(..., description="Retriever plugin name (e.g. 'dense')")
    collection: str = Field(..., description="Vector DB collection name")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")


class RGSSettings(BaseModel):
    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: str | None = None
    max_chunks: int | None = 8
    max_answer_chars: int | None = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"


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
