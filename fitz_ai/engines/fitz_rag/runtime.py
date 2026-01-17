# fitz_ai/engines/fitz_rag/runtime.py
"""
Fitz RAG Runtime - Canonical entry point for Fitz RAG execution.

This module provides the single, stable way to execute Fitz RAG queries.
All other entry points (CLI, API, etc.) should route through this runtime.
"""

from typing import Any, Dict, Optional

from fitz_ai.config import load_engine_config
from fitz_ai.core import Answer, Constraints, Provenance
from fitz_ai.engines.fitz_rag.config import FitzRagConfig
from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline


def run_fitz_rag(
    query: str,
    config: Optional[FitzRagConfig] = None,
    config_path: Optional[str] = None,
    constraints: Optional[Constraints] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Answer:
    """
    Execute a Fitz RAG query.

    This is the canonical entry point for all Fitz RAG execution.

    Args:
        query: The question text
        config: Optional pre-loaded FitzRagConfig
        config_path: Optional path to config file
        constraints: Optional query-time constraints
        metadata: Optional engine-specific metadata

    Returns:
        Answer object with generated text and source provenance
    """
    if config is None:
        if config_path is None:
            config = load_engine_config("fitz_rag")
        else:
            import yaml
            from pathlib import Path

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_rag" in raw:
                config_dict = raw["fitz_rag"]
            else:
                config_dict = raw

            config = FitzRagConfig(**config_dict)

    pipeline = RAGPipeline.from_config(config)
    rag_answer = pipeline.run(query)

    # Convert to core Answer type
    provenance = []
    if hasattr(rag_answer, "sources") and rag_answer.sources:
        for source_ref in rag_answer.sources:
            prov = Provenance(
                source_id=getattr(source_ref, "source_id", str(source_ref)),
                excerpt=getattr(source_ref, "text", None),
                metadata=getattr(source_ref, "metadata", {}),
            )
            provenance.append(prov)

    answer_metadata = {
        "engine": "fitz_rag",
        "query_text": query,
    }

    if hasattr(rag_answer, "metadata") and rag_answer.metadata:
        answer_metadata["rag_metadata"] = rag_answer.metadata

    return Answer(
        text=rag_answer.answer if hasattr(rag_answer, "answer") else str(rag_answer),
        provenance=provenance,
        metadata=answer_metadata,
    )


def create_fitz_rag_engine(
    config: Optional[FitzRagConfig] = None,
    config_path: Optional[str] = None,
) -> RAGPipeline:
    """
    Create and return a Fitz RAG pipeline instance.

    Args:
        config: Optional pre-loaded FitzRagConfig
        config_path: Optional path to config file

    Returns:
        Initialized RAGPipeline instance
    """
    if config is None:
        if config_path is None:
            config = load_engine_config("fitz_rag")
        else:
            import yaml
            from pathlib import Path

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_rag" in raw:
                config_dict = raw["fitz_rag"]
            else:
                config_dict = raw

            config = FitzRagConfig(**config_dict)

    return RAGPipeline.from_config(config)


# Convenience alias
run = run_fitz_rag


# =============================================================================
# AUTO-REGISTRATION WITH GLOBAL REGISTRY
# =============================================================================


def _register_fitz_rag_engine():
    """Register Fitz RAG engine with the global registry."""
    from fitz_ai.engines.fitz_rag.config import get_default_config_path
    from fitz_ai.engines.fitz_rag.engine import FitzRagEngine
    from fitz_ai.runtime.registry import EngineCapabilities, EngineRegistry

    def fitz_rag_factory(config):
        """Factory for creating Fitz RAG engine."""
        if config is None:
            config = load_engine_config("fitz_rag")
        elif isinstance(config, dict):
            config = FitzRagConfig(**config)

        # Return FitzRagEngine which implements KnowledgeEngine protocol
        return FitzRagEngine(config)

    def fitz_rag_config_loader(config_path):
        """Load config for fitz_rag engine."""
        if config_path is None:
            return load_engine_config("fitz_rag")
        else:
            import yaml
            from pathlib import Path

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_rag" in raw:
                config_dict = raw["fitz_rag"]
            else:
                config_dict = raw

            return FitzRagConfig(**config_dict)

    # Define capabilities
    capabilities = EngineCapabilities(
        supports_collections=True,
        requires_documents_at_query=False,
        supports_chat=True,
        supports_streaming=False,
        requires_config=True,
        requires_api_key=True,
        api_key_env_var="COHERE_API_KEY",
    )

    try:
        registry = EngineRegistry.get_global()
        registry.register(
            name="fitz_rag",
            factory=fitz_rag_factory,
            description="Vector search and LLM synthesis RAG",
            config_type=FitzRagConfig,
            config_loader=fitz_rag_config_loader,
            default_config_path=get_default_config_path,
            capabilities=capabilities,
        )
    except ValueError:
        pass  # Already registered


# Auto-register on import
_register_fitz_rag_engine()
