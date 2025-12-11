# src/fitz_rag/sourcer/rag_base.py
"""
Core plugin and retrieval system for fitz-rag.

Defines:
- ArtefactRetrievalStrategy (base)
- SourceConfig (plugin description)
- RAGContextBuilder (retrieval orchestrator)
- load_source_configs() (dynamic plugin loader)
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from fitz_rag.core import RetrievedChunk
from fitz_rag.config import get_config

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import SOURCER

_cfg = get_config()
logger = get_logger(__name__)


# ---------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------

class StrategyError(Exception):
    """Errors raised during retrieval strategy execution."""


class PluginLoadError(Exception):
    """Raised when plugin modules cannot be imported."""


# ---------------------------------------------------------
# Base Strategy Interface
# ---------------------------------------------------------

class ArtefactRetrievalStrategy:
    """
    Base class for all retrieval strategies.

    A strategy accepts:
        trf (any structured input)
        query (string)

    It returns:
        list[RetrievedChunk]
    """

    def retrieve(self, trf: Dict, query: str) -> List[RetrievedChunk]:
        raise NotImplementedError


# ---------------------------------------------------------
# SourceConfig — describes a plugin
# ---------------------------------------------------------

@dataclass
class SourceConfig:
    name: str
    order: int
    strategy: ArtefactRetrievalStrategy
    label: Optional[str] = None


# ---------------------------------------------------------
# RetrievalContext — final result structure
# ---------------------------------------------------------

@dataclass
class RetrievalContext:
    query: str
    artefacts: Dict[str, List[RetrievedChunk]]


# ---------------------------------------------------------
# RAGContextBuilder — orchestrates retrieval
# ---------------------------------------------------------

class RAGContextBuilder:
    """
    Given:
        - list of SourceConfig
        - TRF-like dict
    Produces:
        RetrievalContext(query, artefacts)
    """

    def __init__(self, sources: List[SourceConfig]) -> None:
        self.sources = sources

    def retrieve_for(self, trf: Dict, query: str) -> RetrievalContext:
        artefacts: Dict[str, List[RetrievedChunk]] = {}

        for src in self.sources:
            logger.debug(f"{SOURCER} Running strategy '{src.name}' for query='{query}'")

            try:
                artefacts[src.name] = src.strategy.retrieve(trf, query)
                logger.debug(
                    f"{SOURCER} Strategy '{src.name}' returned {len(artefacts[src.name])} chunks"
                )

            except Exception as e:
                logger.error(f"{SOURCER} Strategy '{src.name}' failed: {e}")

                # Fail-open but store an error chunk with all required fields
                artefacts[src.name] = []
                artefacts[src.name + '_error'] = [
                    RetrievedChunk(
                        text=f"[Retrieval error in {src.name}: {e}]",
                        score=0.0,
                        collection="internal",
                        chunk_id=f"{src.name}_error",
                        metadata={"error": True},
                    )
                ]

        return RetrievalContext(query=query, artefacts=artefacts)


# ---------------------------------------------------------
# Dynamic Plugin Loader
# ---------------------------------------------------------

def load_source_configs() -> List[SourceConfig]:
    """
    Load all plugins in fitz_rag.sourcer.* that define SOURCE_CONFIG.
    """

    pkg_dir = Path(__file__).parent
    base_pkg = __name__.rsplit(".", 1)[0]

    configs: List[SourceConfig] = []

    for info in pkgutil.iter_modules([str(pkg_dir)]):

        name = info.name
        full_name = f"{base_pkg}.{name}"

        logger.debug(f"{SOURCER} Loading plugin module '{full_name}'")

        try:
            module = importlib.import_module(full_name)
        except Exception as e:
            logger.error(f"{SOURCER} Failed to import plugin '{full_name}': {e}")
            raise PluginLoadError(
                f"Failed to import plugin module '{full_name}': {e}"
            ) from e

        cfg = getattr(module, "SOURCE_CONFIG", None)

        if isinstance(cfg, SourceConfig):
            logger.debug(f"{SOURCER} Registered plugin '{full_name}'")
            configs.append(cfg)
        else:
            # Silence is intentional — internal modules don't define SOURCE_CONFIG.
            logger.debug(f"{SOURCER} Module '{full_name}' has no SOURCE_CONFIG")

    return configs
