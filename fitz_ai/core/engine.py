"""KnowledgeEngine Protocol - Query → Answer contract. See docs/api_reference.md for details."""

from typing import Protocol, runtime_checkable

from .answer import Answer
from .query import Query


@runtime_checkable
class KnowledgeEngine(Protocol):
    """Paradigm-agnostic protocol: all engines implement answer(Query) -> Answer."""

    def answer(self, query: Query) -> Answer:
        """Execute query and return answer with provenance."""
        ...
