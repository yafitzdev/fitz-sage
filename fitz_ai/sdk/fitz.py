# fitz_ai/sdk/fitz.py
"""
Fitz class - Stateful SDK for the Fitz KRAG framework.

This is a thin wrapper around FitzService that adds:
- Stateful collection management (remembers collection across calls)
- Auto-initialization of config
- Simplified API for common use cases
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from fitz_ai.core import Answer, ConfigurationError
from fitz_ai.logging.logger import get_logger
from fitz_ai.services import FitzService

if TYPE_CHECKING:
    from fitz_ai.retrieval.rewriter.types import ConversationContext

logger = get_logger(__name__)


@dataclass
class IngestStats:
    """Statistics from an ingestion operation."""

    documents: int
    sections: int
    symbols: int
    collection: str


class fitz:
    """
    Stateful SDK for the Fitz RAG framework.

    Provides a simple two-step workflow:
    1. Ingest documents: fitz.ingest("./docs")
    2. Ask questions: answer = fitz.ask("question?")

    The Fitz object remembers its collection and configuration,
    allowing multiple queries without re-specifying settings.

    Examples:
        Simple usage:
        >>> f = fitz()
        >>> f.ingest("./docs")
        >>> answer = f.ask("What is the refund policy?")
        >>> print(answer.text)

        With collection name:
        >>> f = fitz(collection="physics")
        >>> f.ingest("./physics_papers")
        >>> answer = f.ask("Explain entanglement")

        With custom config:
        >>> f = fitz(config_path="my_config.yaml")

        Access provenance:
        >>> for source in answer.provenance:
        ...     print(source.excerpt)
    """

    def __init__(
        self,
        collection: str = "default",
        config_path: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
    ) -> None:
        """
        Initialize the Fitz SDK.

        Args:
            collection: Name for the vector DB collection. Documents ingested
                       with this Fitz instance will be stored in this collection.
            config_path: Path to a YAML config file. If not provided, uses
                        the default config at .fitz/config.yaml or creates one.
            auto_init: If True and no config exists, create a default one.
                      If False, raise ConfigurationError when config missing.
        """
        self._collection = collection
        self._config_path = Path(config_path) if config_path else None
        self._auto_init = auto_init

        # Service layer - does all the real work
        self._service = FitzService()

    @property
    def collection(self) -> str:
        """The vector DB collection name."""
        return self._collection

    @property
    def config_path(self) -> Path:
        """Path to the configuration file."""
        if self._config_path:
            return self._config_path
        from fitz_ai.core.paths import FitzPaths

        return FitzPaths.config()

    def ingest(
        self,
        source: Union[str, Path],
        *,
        force: bool = False,
        artifacts: Optional[str] = "none",
    ) -> IngestStats:
        """
        Ingest documents into the knowledge base.

        Incremental by default - only processes new/changed files.

        Args:
            source: Path to a file or directory to ingest.
            force: Re-ingest all files regardless of state.
            artifacts: "none", "all", or comma-separated list of artifact types.

        Returns:
            IngestStats with document, section, and symbol counts.

        Raises:
            ConfigurationError: If config cannot be loaded/created.
            ValueError: If source path doesn't exist or no documents found.
        """
        # Ensure config exists
        self._ensure_config()

        # Delegate to service
        result = self._service.ingest(
            source=source,
            collection=self._collection,
            force=force,
            artifacts=artifacts,
        )

        logger.info(
            f"Ingested {result.documents_processed} documents "
            f"({result.sections_created} sections, {result.symbols_created} symbols) "
            f"into collection '{self._collection}'"
        )

        return IngestStats(
            documents=result.documents_processed,
            sections=result.sections_created,
            symbols=result.symbols_created,
            collection=self._collection,
        )

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        conversation_context: Optional["ConversationContext"] = None,
    ) -> Answer:
        """
        Ask a question about the ingested documents.

        Args:
            question: The question to ask.
            top_k: Override the number of results to retrieve.
            conversation_context: Optional ConversationContext for query rewriting.
                Enables conversational pronoun resolution (e.g., "their" → "TechCorp's").

        Returns:
            Answer object with text and provenance.

        Raises:
            ConfigurationError: If not configured.
            QueryError: If query fails or question is empty.
        """
        # Ensure config exists
        self._ensure_config()

        # Delegate to service
        return self._service.query(
            question=question,
            collection=self._collection,
            top_k=top_k,
            conversation_context=conversation_context,
        )

    def query(
        self,
        question: str,
        conversation_context: Optional["ConversationContext"] = None,
        **kwargs: Any,
    ) -> Answer:
        """Alias for ask(). Provided for API consistency."""
        return self.ask(question, conversation_context=conversation_context, **kwargs)

    def _ensure_config(self) -> None:
        """Ensure configuration file exists, creating if needed."""
        config_path = self.config_path

        if config_path.exists():
            return

        if not self._auto_init:
            raise ConfigurationError(
                f"Config file not found: {config_path}. Run 'fitz init' or pass auto_init=True."
            )

        # Create default config
        self._create_default_config(config_path)

    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file."""
        config_content = f"""\
# Fitz KRAG Configuration
# Generated by: Fitz SDK

chat: cohere
embedding: cohere
vector_db: pgvector

collection: {self._collection}

rerank: cohere

chat_kwargs:
  models:
    smart: command-a-03-2025
    fast: command-r7b-12-2024

embedding_kwargs:
  model: embed-english-v3.0

rerank_kwargs:
  model: rerank-v3.5
"""

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content)
        logger.info(f"Created default config at {config_path}")
