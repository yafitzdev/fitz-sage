# fitz_ai/engines/fitz_rag/pipeline/components.py
"""Pipeline component groupings for cleaner dependency injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from fitz_ai.core.guardrails import ConstraintPlugin, SemanticMatcher


@dataclass
class StructuredComponents:
    """
    Structured data query components.

    Groups all dependencies for table/CSV querying via SQL.
    """

    router: Any | None = None  # StructuredQueryRouter
    executor: Any | None = None  # StructuredExecutor
    sql_generator: Any | None = None  # SQLGenerator
    result_formatter: Any | None = None  # ResultFormatter
    derived_store: Any | None = None  # DerivedStore

    def is_enabled(self) -> bool:
        """Check if structured querying is enabled."""
        return self.router is not None


@dataclass
class CloudComponents:
    """
    Cloud cache components.

    Groups dependencies for Fitz Cloud cache integration.
    """

    embedder: Any | None = None  # Embedder for query embeddings
    client: Any | None = None  # CloudClient for cache operations
    fast_chat: Any | None = None  # Fast-tier chat for routing optimization

    def is_enabled(self) -> bool:
        """Check if cloud caching is enabled."""
        return self.client is not None and self.embedder is not None


@dataclass
class GuardrailComponents:
    """
    Guardrail components for epistemic safety.

    Groups constraint plugins and semantic matching.
    """

    constraints: Sequence[ConstraintPlugin] | None = None
    semantic_matcher: SemanticMatcher | None = None


@dataclass
class RoutingComponents:
    """
    Query routing components.

    Groups all query routing and filtering dependencies.
    """

    query_router: Any | None = None  # QueryRouter for global/specific routing
    keyword_matcher: Any | None = None  # KeywordMatcher for vocabulary filtering
    hop_controller: Any | None = None  # HopController for multi-hop retrieval


@dataclass
class PipelineComponents:
    """
    Complete set of RAGPipeline dependencies.

    Groups all optional and required components into logical units.
    This replaces the 17-parameter constructor with 4 structured groups.
    """

    # Required core
    retrieval: Any  # RetrievalPipelineFromYaml
    chat: Any  # Chat LLM (smart tier)
    rgs: Any  # RGS (Retrieval-Guided Synthesis)

    # Optional context processor (has default)
    context: Any | None = None  # ContextPipeline

    # Optional component groups
    guardrails: GuardrailComponents | None = None
    routing: RoutingComponents | None = None
    cloud: CloudComponents | None = None
    structured: StructuredComponents | None = None
