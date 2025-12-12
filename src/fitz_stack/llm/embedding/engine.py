from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from fitz_stack.llm.embedding.base import EmbeddingPlugin
from fitz_stack.llm.registry import get_llm_plugin
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import EMBEDDING

logger = get_logger(__name__)


@dataclass
class EmbeddingEngine:
    """
    Orchestration for embedding plugins.

    Responsibilities:
        - Normalize input to a single text string
        - Run plugin.embed(...)
        - Provide factory constructor from_name(...)
    """

    plugin: EmbeddingPlugin

    def embed(self, text: str) -> List[float]:
        logger.info(f"{EMBEDDING} Embedding text of length={len(text)}")
        return self.plugin.embed(text)

    # ---------------------------------------------------------
    # Factory from registry
    # ---------------------------------------------------------
    @classmethod
    def from_name(cls, plugin_name: str, **kwargs) -> "EmbeddingEngine":
        PluginCls = get_llm_plugin(plugin_name, plugin_type="embedding")
        plugin = PluginCls(**kwargs)
        return cls(plugin=plugin)
