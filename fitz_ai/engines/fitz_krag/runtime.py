# fitz_ai/engines/fitz_krag/runtime.py
"""
Fitz KRAG Runtime - Engine registration and entry points.

Auto-registers the fitz_krag engine with the global registry on import.
"""

from typing import Any, Dict, Optional

from fitz_ai.config import load_engine_config
from fitz_ai.core import Answer
from fitz_ai.engines.fitz_krag.config import FitzKragConfig


def run_fitz_krag(
    query: str,
    config: Optional[FitzKragConfig] = None,
    config_path: Optional[str] = None,
) -> Answer:
    """
    Execute a Fitz KRAG query.

    Args:
        query: The question text
        config: Optional pre-loaded FitzKragConfig
        config_path: Optional path to config file

    Returns:
        Answer object with generated text and source provenance
    """
    if config is None:
        if config_path is None:
            config = load_engine_config("fitz_krag")
        else:
            from pathlib import Path

            import yaml

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_krag" in raw:
                config_dict = raw["fitz_krag"]
            else:
                config_dict = raw

            config = FitzKragConfig(**config_dict)

    from fitz_ai.core import Query
    from fitz_ai.engines.fitz_krag.engine import FitzKragEngine

    engine = FitzKragEngine(config)
    return engine.answer(Query(text=query))


run = run_fitz_krag


# =============================================================================
# AUTO-REGISTRATION WITH GLOBAL REGISTRY
# =============================================================================


def _register_fitz_krag_engine():
    """Register Fitz KRAG engine with the global registry."""
    from fitz_ai.engines.fitz_krag.config import get_default_config_path
    from fitz_ai.engines.fitz_krag.engine import FitzKragEngine
    from fitz_ai.runtime.registry import EngineCapabilities, EngineRegistry

    def fitz_krag_factory(config):
        """Factory for creating Fitz KRAG engine."""
        if config is None:
            config = load_engine_config("fitz_krag")
        elif isinstance(config, dict):
            config = FitzKragConfig(**config)

        return FitzKragEngine(config)

    def fitz_krag_config_loader(config_path):
        """Load config for fitz_krag engine."""
        if config_path is None:
            return load_engine_config("fitz_krag")
        else:
            from pathlib import Path

            import yaml

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_krag" in raw:
                config_dict = raw["fitz_krag"]
            else:
                config_dict = raw

            return FitzKragConfig(**config_dict)

    capabilities = EngineCapabilities(
        supports_collections=True,
        requires_documents_at_query=False,
        supports_persistent_ingest=True,
        supports_chat=True,
        supports_streaming=False,
        requires_config=True,
        requires_api_key=True,
        api_key_env_var="COHERE_API_KEY",
    )

    try:
        registry = EngineRegistry.get_global()
        registry.register(
            name="fitz_krag",
            factory=fitz_krag_factory,
            description="Knowledge Routing Augmented Generation (code + docs)",
            config_type=FitzKragConfig,
            config_loader=fitz_krag_config_loader,
            default_config_path=get_default_config_path,
            capabilities=capabilities,
        )
    except ValueError:
        pass  # Already registered


# Auto-register on import
_register_fitz_krag_engine()
