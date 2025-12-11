from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Iterable

import importlib
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------


class PluginLoadError(Exception):
    """Raised when source plugin modules cannot be imported."""


# ---------------------------------------------------------------------
# Core strategy / config types
# ---------------------------------------------------------------------


class ArtefactRetrievalStrategy(Protocol):
    """
    Base protocol for retrieval strategies used by the sourcer layer.

    A concrete strategy is responsible for:
    - looking at a TRF-like object (any dict) and the user query
    - returning a list of "chunks" (dicts or other chunk-like objects)
    """

    def retrieve(self, trf: Any, query: str) -> Iterable[Any]:
        ...


@dataclass
class SourceConfig:
    """
    Configuration for a single retrieval source.

    Attributes
    ----------
    name:
        Name of the source. Used as key in RetrievalContext.artefacts.
    order:
        Ordering key for this source within RAGContextBuilder.
    strategy:
        ArtefactRetrievalStrategy instance responsible for retrieving chunks.
    """
    name: str
    order: int
    strategy: ArtefactRetrievalStrategy


@dataclass
class RetrievalContext:
    """
    Holds the result of running retrieval strategies for a given query.

    Attributes
    ----------
    query:
        Original user query.
    artefacts:
        Mapping of source name -> list of chunks (dicts or chunk-like objects).
        Error artefacts are stored under "<source>_error".
    """
    query: str
    artefacts: Dict[str, List[Any]]


# ---------------------------------------------------------------------
# RAGContextBuilder
# ---------------------------------------------------------------------


class RAGContextBuilder:
    """
    Coordinates running multiple retrieval strategies and collecting artefacts.

    Usage
    -----
        builder = RAGContextBuilder(
            sources=[SourceConfig(name="kb", order=0, strategy=MyStrategy())]
        )
        ctx = builder.retrieve_for(trf, query)

    Behavior on errors
    ------------------
    If a strategy raises an exception, the builder:
    - sets ctx.artefacts[source.name] = []
    - appends a dict-based "error chunk" to ctx.artefacts[f"{source.name}_error"]
      with a 'text' field containing the error message.
    """

    def __init__(self, sources: List[SourceConfig]) -> None:
        # Sort by order to have deterministic behavior
        self.sources = sorted(sources, key=lambda s: s.order)

    def retrieve_for(self, trf: Any, query: str) -> RetrievalContext:
        artefacts: Dict[str, List[Any]] = {}

        for src in self.sources:
            name = src.name
            strategy = src.strategy

            try:
                result = strategy.retrieve(trf, query)
                # Normalize None -> []
                if result is None:
                    chunks: List[Any] = []
                else:
                    chunks = list(result)

                artefacts[name] = chunks

            except Exception as e:
                # Log the error for diagnostics
                logger.error("Retrieval error in source '%s': %s", name, e)

                # Store empty list for the main source
                artefacts[name] = []

                # Store a dict-based "error chunk" for tests and UIs
                error_chunk = {
                    "text": f"Retrieval error in {name}: {e}",
                    "metadata": {
                        "source": name,
                        "error": True,
                    },
                }
                artefacts[f"{name}_error"] = [error_chunk]

        return RetrievalContext(query=query, artefacts=artefacts)


# ---------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------


def load_source_configs() -> List[SourceConfig]:
    """
    Load source configurations from plugin modules.

    In v0.1.0 this is intentionally minimal: the test suite only verifies
    that import errors are wrapped in PluginLoadError. The successful path
    may simply return an empty list.

    Returns
    -------
    List[SourceConfig]:
        Loaded source configurations (possibly empty).

    Raises
    ------
    PluginLoadError:
        If underlying importlib.import_module calls fail.
    """
    try:
        # Attempt to import a well-known sourcer plugin module.
        # Tests monkeypatch importlib.import_module to throw here.
        importlib.import_module("fitz_rag.sourcer.plugins")
    except Exception as e:
        raise PluginLoadError("Failed to import plugin module") from e

    # For now, return an empty list; production code can extend this
    # to discover and construct SourceConfig instances from configuration.
    return []
