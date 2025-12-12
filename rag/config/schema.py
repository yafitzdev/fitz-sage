# rag/config/schema.py

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    plugin_name: Optional[str] = Field(
        default=None,
        description="Embedding plugin name (canonical field)",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Alias for plugin_name (legacy / test compatibility)",
    )
    model: str
    api_key: Optional[str] = None
    output_dimension: Optional[int] = None

    @model_validator(mode="after")
    def _normalize_plugin_name(self) -> "EmbeddingConfig":
        if self.plugin_name is None and self.provider is not None:
            self.plugin_name = self.provider
        if self.plugin_name is None:
            raise ValueError("Either plugin_name or provider must be set")
        return self


class RerankConfig(BaseModel):
    enabled: bool = False
    plugin_name: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    top_k: Optional[int] = None


class RetrieverConfig(BaseModel):
    collection: str
    top_k: int = 5


class RGSSettings(BaseModel):
    max_chunks: int = 8
    max_chars: int = 4000
    include_citations: bool = True
    strict_grounding: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"


class RAGConfig(BaseModel):
    embedding: EmbeddingConfig
    retriever: RetrieverConfig
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    rgs: RGSSettings = Field(default_factory=RGSSettings)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
