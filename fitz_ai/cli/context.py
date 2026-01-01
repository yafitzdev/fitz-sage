# fitz_ai/cli/context.py
"""
Central CLI context - single source of truth for all CLI commands.

All configuration reading happens here. Commands import CLIContext and use it.
No more scattered dict.get() chains across every command file.

Config Loading Strategy:
    1. Engine defaults (fitz_ai/engines/<engine>/config/default.yaml) - always loaded
    2. User config (.fitz/config/<engine>.yaml) - overrides defaults
    3. Plugin defaults (fitz_ai/llm/<type>/<plugin>.yaml) - for model info

Values are ALWAYS guaranteed to exist. No fallback logic needed in CLI code.

Usage:
    from fitz_ai.cli.context import CLIContext

    # Load merged config (defaults + user overrides)
    ctx = CLIContext.load()
    print(ctx.chat_plugin)        # always exists
    print(ctx.chat_display)       # "cohere (command-r-plus)"
    print(ctx.retrieval_top_k)    # always exists
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fitz_ai.config.loader import get_config_source, load_engine_config
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Default engine for CLI operations
DEFAULT_ENGINE = "fitz_rag"


@dataclass
class CLIContext:
    """
    Central CLI context containing all configuration values.

    This is the single source of truth for CLI commands. All config values
    are extracted once and exposed as typed properties.
    """

    # Raw and typed config
    raw_config: dict = field(repr=False)
    typed_config: Any = field(default=None, repr=False)
    config_path: Path = field(default=None)

    # Chat LLM
    chat_plugin: str = ""
    chat_model_smart: str = ""
    chat_model_fast: str = ""

    # Embedding
    embedding_plugin: str = ""
    embedding_model: str = ""

    # Vector DB
    vector_db_plugin: str = ""
    vector_db_kwargs: dict = field(default_factory=dict)

    # Retrieval
    retrieval_plugin: str = ""
    retrieval_collection: str = ""
    retrieval_top_k: int = 5

    # Rerank
    rerank_enabled: bool = False
    rerank_plugin: str = ""
    rerank_model: str = ""

    # RGS (Response Generation System)
    rgs_citations: bool = True
    rgs_strict_grounding: bool = True

    # -------------------------------------------------------------------------
    # Display Properties (formatted strings for UI)
    # -------------------------------------------------------------------------

    @property
    def chat_display(self) -> str:
        """Chat info for display: 'plugin (model)' or just 'plugin'."""
        if self.chat_model_smart:
            return f"{self.chat_plugin} ({self.chat_model_smart})"
        return self.chat_plugin or "?"

    @property
    def chat_display_fast(self) -> str:
        """Chat fast model info for display."""
        if self.chat_model_fast:
            return f"{self.chat_plugin} ({self.chat_model_fast})"
        return self.chat_plugin or "?"

    @property
    def embedding_display(self) -> str:
        """Embedding info for display: 'plugin (model)' or just 'plugin'."""
        if self.embedding_model:
            return f"{self.embedding_plugin} ({self.embedding_model})"
        return self.embedding_plugin or "?"

    @property
    def rerank_display(self) -> Optional[str]:
        """Rerank info for display, or None if disabled."""
        if not self.rerank_enabled:
            return None
        if self.rerank_model:
            return f"{self.rerank_plugin} ({self.rerank_model})"
        return self.rerank_plugin or "?"

    @property
    def vector_db_display(self) -> str:
        """Vector DB info for display."""
        host = self.vector_db_kwargs.get("host", "")
        port = self.vector_db_kwargs.get("port", "")
        if host:
            return f"{self.vector_db_plugin} ({host}:{port})"
        return self.vector_db_plugin or "?"

    @property
    def retrieval_display(self) -> str:
        """Retrieval info for display."""
        return f"{self.retrieval_plugin} (collection={self.retrieval_collection}, top_k={self.retrieval_top_k})"

    @property
    def embedding_id(self) -> str:
        """Unique embedding ID for caching (e.g., 'cohere:embed-english-v3.0')."""
        plugin = self.embedding_plugin or "unknown"
        model = self.embedding_model or "default"
        return f"{plugin}:{model}"

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, engine: str = DEFAULT_ENGINE) -> "CLIContext":
        """
        Load CLI context with merged config (defaults + user overrides).

        This ALWAYS succeeds because package defaults always exist.
        Values are guaranteed to be present - no fallback logic needed.

        Args:
            engine: Engine name (default: "fitz_rag")

        Returns:
            CLIContext with all values populated from merged config.
        """
        try:
            # Load merged config (defaults + user overrides)
            merged_config = load_engine_config(engine)

            # Get config source for display
            source = get_config_source(engine)
            config_path = FitzPaths.engine_config(engine)

            # Load typed config if user config exists
            typed_config = cls._load_typed_config(config_path, merged_config)

            # Extract all values - no .get() fallbacks needed!
            return cls._from_merged_config(merged_config, typed_config, config_path, source)

        except FileNotFoundError as e:
            # This should only happen if package defaults are missing (bug)
            logger.error(f"Package defaults missing: {e}")
            raise

    @classmethod
    def load_or_none(cls, engine: str = DEFAULT_ENGINE) -> Optional["CLIContext"]:
        """
        Load CLI context, returning None only if loading fails.

        Note: With the layered config system, this almost always succeeds
        because package defaults are always available. It only returns None
        if there's an actual loading error.

        Returns:
            CLIContext or None if loading failed.
        """
        try:
            return cls.load(engine)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return None

    @classmethod
    def _load_typed_config(cls, config_path: Path, raw_config: dict) -> Any:
        """Load typed FitzRagConfig if user config exists.

        Currently disabled - importing the engine config schema triggers
        the engine package __init__, which does heavy initialization.
        CLIContext works fine with raw config, so typed config is unnecessary.
        """
        return None

    @classmethod
    def _from_merged_config(
        cls, config: dict, typed: Any, path: Path, source: str
    ) -> "CLIContext":
        """
        Extract all values from merged config + plugin defaults.

        Config provides plugin names. Plugin YAMLs provide model defaults.
        User kwargs override plugin defaults.
        """
        from fitz_ai.llm.loader import load_plugin

        # Get config sections
        chat = config["chat"]
        emb = config["embedding"]
        vdb = config["vector_db"]
        ret = config["retrieval"]
        rerank = config["rerank"]
        rgs = config["rgs"]

        # Load plugin specs to get model defaults
        chat_spec = load_plugin("chat", chat["plugin_name"])
        emb_spec = load_plugin("embedding", emb["plugin_name"])

        # User kwargs override plugin defaults
        chat_kwargs = chat.get("kwargs", {})
        chat_models = chat_kwargs.get("models", chat_spec.defaults.get("models", {}))

        emb_kwargs = emb.get("kwargs", {})
        emb_model = emb_kwargs.get("model", emb_spec.defaults.get("model", ""))

        vdb_kwargs = vdb.get("kwargs", {})

        # Rerank is optional - only load spec if enabled
        rerank_model = ""
        if rerank["enabled"]:
            rerank_spec = load_plugin("rerank", rerank["plugin_name"])
            rerank_kwargs = rerank.get("kwargs", {})
            rerank_model = rerank_kwargs.get("model", rerank_spec.defaults.get("model", ""))

        return cls(
            raw_config=config,
            typed_config=typed,
            config_path=path,
            # Chat
            chat_plugin=chat["plugin_name"],
            chat_model_smart=chat_models.get("smart", ""),
            chat_model_fast=chat_models.get("fast", ""),
            # Embedding
            embedding_plugin=emb["plugin_name"],
            embedding_model=emb_model,
            # Vector DB
            vector_db_plugin=vdb["plugin_name"],
            vector_db_kwargs=vdb_kwargs,
            # Retrieval
            retrieval_plugin=ret["plugin_name"],
            retrieval_collection=ret["collection"],
            retrieval_top_k=ret["top_k"],
            # Rerank
            rerank_enabled=rerank["enabled"],
            rerank_plugin=rerank["plugin_name"],
            rerank_model=rerank_model,
            # RGS
            rgs_citations=rgs["enable_citations"],
            rgs_strict_grounding=rgs["strict_grounding"],
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_vector_db_client(self):
        """Get vector DB client instance."""
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        return get_vector_db_plugin(self.vector_db_plugin, **self.vector_db_kwargs)

    def get_collections(self) -> list[str]:
        """Get list of collections from vector DB."""
        try:
            client = self.get_vector_db_client()
            return sorted(client.list_collections())
        except Exception:
            return []

    def info_line(self, include_rerank: bool = True) -> str:
        """
        Get a single-line info string for display.

        Example: "Collection: default | VectorDB: local_faiss | Chat: cohere (command-r-plus)"
        """
        parts = [
            f"Collection: {self.retrieval_collection}",
            f"VectorDB: {self.vector_db_plugin}",
            f"Retrieval: {self.retrieval_plugin}",
            f"Chat: {self.chat_display}",
            f"Embedding: {self.embedding_display}",
        ]
        if include_rerank and self.rerank_display:
            parts.append(f"Rerank: {self.rerank_display}")
        return " | ".join(parts)


__all__ = ["CLIContext"]
