# fitz_ai/engines/fitz_rag/retrieval/loader.py
"""
YAML-based Retrieval Plugin Loader.

Loads retrieval plugin definitions from YAML files and builds
executable pipelines using the standard step classes.

Plugin YAML files live in: fitz_ai/engines/fitz_rag/retrieval/plugins/*.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import yaml

from fitz_ai.engines.fitz_rag.retrieval.steps import (
    ChatClient,
    Embedder,
    EntityGraphClient,
    KeywordMatcherClient,
    Reranker,
    RetrievalStep,
    VectorClient,
    get_step_class,
)
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

logger = get_logger(__name__)

# Path to plugin YAML files
PLUGINS_DIR = Path(__file__).parent / "plugins"


# =============================================================================
# Plugin Spec (parsed from YAML)
# =============================================================================


@dataclass
class StepSpec:
    """Specification for a single retrieval step."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled_if: str | None = None  # Dependency name that must be present
    use_config: str | None = None  # Config key to override a param

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepSpec":
        step_type = data.get("type")
        if not step_type:
            raise ValueError("Step must have 'type' field")

        return cls(
            type=step_type,
            params={k: v for k, v in data.items() if k not in ("type", "enabled_if", "use_config")},
            enabled_if=data.get("enabled_if"),
            use_config=data.get("use_config"),
        )


@dataclass
class RetrievalPluginSpec:
    """Specification for a retrieval plugin (loaded from YAML)."""

    plugin_name: str
    description: str
    steps: list[StepSpec]

    @classmethod
    def from_yaml(cls, path: Path) -> "RetrievalPluginSpec":
        """Load plugin spec from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        plugin_name = data.get("plugin_name")
        if not plugin_name:
            raise ValueError(f"Plugin YAML must have 'plugin_name': {path}")

        steps_data = data.get("steps", [])
        steps = [StepSpec.from_dict(s) for s in steps_data]

        return cls(
            plugin_name=plugin_name,
            description=data.get("description", ""),
            steps=steps,
        )


# =============================================================================
# Dependencies Container
# =============================================================================


@dataclass
class RetrievalDependencies:
    """
    Container for retrieval dependencies.

    These are injected when building the pipeline.
    """

    vector_client: VectorClient
    embedder: Embedder
    collection: str
    reranker: Reranker | None = None
    chat: ChatClient | None = None  # Fast-tier chat for multi-query expansion
    keyword_matcher: KeywordMatcherClient | None = None  # Exact keyword matching
    entity_graph: EntityGraphClient | None = None  # Entity graph for related chunk discovery
    max_entity_expansion: int = 10  # Max related chunks per query
    table_store: Any | None = None  # TableStore for CSV file queries

    # Config overrides (from retrieval config)
    top_k: int = 5

    # Optional artifact fetching
    fetch_artifacts: bool = False


# =============================================================================
# Plugin Loader
# =============================================================================


def load_plugin_spec(plugin_name: str) -> RetrievalPluginSpec:
    """
    Load a retrieval plugin specification by name.

    Searches for {plugin_name}.yaml in the plugins directory.
    """
    yaml_path = PLUGINS_DIR / f"{plugin_name}.yaml"

    if not yaml_path.exists():
        available = list_available_plugins()
        raise FileNotFoundError(
            f"Retrieval plugin not found: {plugin_name!r}. "
            f"Available: {available}. "
            f"Searched: {yaml_path}"
        )

    return RetrievalPluginSpec.from_yaml(yaml_path)


def list_available_plugins() -> list[str]:
    """List all available retrieval plugins."""
    if not PLUGINS_DIR.exists():
        return []

    return [p.stem for p in PLUGINS_DIR.glob("*.yaml")]


def build_pipeline_from_spec(
    spec: RetrievalPluginSpec,
    deps: RetrievalDependencies,
) -> list[RetrievalStep]:
    """
    Build executable step instances from a plugin spec.

    Args:
        spec: Plugin specification (from YAML)
        deps: Injected dependencies

    Returns:
        List of configured RetrievalStep instances
    """
    steps: list[RetrievalStep] = []

    for step_spec in spec.steps:
        # Check if step should be enabled
        if step_spec.enabled_if:
            dep_value = getattr(deps, step_spec.enabled_if, None)
            if not dep_value:
                logger.debug(
                    f"{RETRIEVER} Skipping step {step_spec.type}: {step_spec.enabled_if} not provided"
                )
                continue

        # Build params
        params = dict(step_spec.params)

        # Apply config overrides
        if step_spec.use_config:
            config_value = getattr(deps, step_spec.use_config, None)
            if config_value is not None:
                # Override 'k' with config value
                params["k"] = config_value

        # Inject dependencies based on step type
        step = _build_step(step_spec.type, params, deps)
        steps.append(step)

    # Auto-inject table_query after vector_search if chat is available
    # This is "always on" like keyword matching and multi-query expansion
    if deps.chat is not None:
        steps = _inject_table_query_step(steps, deps)

    return steps


def _inject_table_query_step(
    steps: list[RetrievalStep],
    deps: RetrievalDependencies,
) -> list[RetrievalStep]:
    """
    Inject table_query step after vector_search.

    Table query processing is "always on" when chat is available,
    similar to keyword matching and multi-query expansion.
    """
    from fitz_ai.engines.fitz_rag.retrieval.steps.vector_search import VectorSearchStep

    # Find vector_search step index
    vector_search_idx = None
    for i, step in enumerate(steps):
        if isinstance(step, VectorSearchStep):
            vector_search_idx = i
            break

    if vector_search_idx is None:
        # No vector search step, can't inject
        return steps

    # Build table_query step
    table_query_step = _build_step("table_query", {}, deps)

    # Insert after vector_search
    result = steps[: vector_search_idx + 1] + [table_query_step] + steps[vector_search_idx + 1 :]

    logger.debug(f"{RETRIEVER} Auto-injected table_query step after vector_search")

    return result


def _build_step(
    step_type: str,
    params: dict[str, Any],
    deps: RetrievalDependencies,
) -> RetrievalStep:
    """Build a single step instance."""
    step_cls = get_step_class(step_type)

    # Inject dependencies based on step type
    if step_type == "vector_search":
        params.setdefault("client", deps.vector_client)
        params.setdefault("embedder", deps.embedder)
        params.setdefault("collection", deps.collection)
        params.setdefault("chat", deps.chat)  # Enables multi-query expansion
        params.setdefault("keyword_matcher", deps.keyword_matcher)  # Enables keyword filtering
        params.setdefault("entity_graph", deps.entity_graph)  # Enables entity-based expansion
        params.setdefault("max_entity_expansion", deps.max_entity_expansion)

    elif step_type == "rerank":
        if deps.reranker is None:
            raise ValueError("Rerank step requires reranker dependency")
        params.setdefault("reranker", deps.reranker)

    elif step_type == "artifact_fetch":
        from fitz_ai.engines.fitz_rag.retrieval.steps.artifact_fetch import (
            SimpleArtifactClient,
        )

        params.setdefault("artifact_client", SimpleArtifactClient(deps.vector_client))
        params.setdefault("collection", deps.collection)

    elif step_type == "table_query":
        if deps.chat is None:
            raise ValueError("Table query step requires chat dependency")
        params.setdefault("chat", deps.chat)
        if deps.table_store is not None:
            params.setdefault("table_store", deps.table_store)

    return step_cls(**params)


# =============================================================================
# Main Entry Point
# =============================================================================


def create_retrieval_pipeline(
    plugin_name: str,
    vector_client: VectorClient,
    embedder: Embedder,
    collection: str,
    reranker: Reranker | None = None,
    chat: ChatClient | None = None,
    keyword_matcher: KeywordMatcherClient | None = None,
    entity_graph: EntityGraphClient | None = None,
    max_entity_expansion: int = 10,
    table_store: Any | None = None,
    top_k: int = 5,
    fetch_artifacts: bool = False,
) -> "RetrievalPipelineFromYaml":
    """
    Create a retrieval pipeline from a YAML plugin definition.

    Args:
        plugin_name: Name of the plugin (e.g., "dense", "dense_rerank")
        vector_client: Vector database client
        embedder: Embedding service
        collection: Collection name
        reranker: Optional reranking service
        chat: Optional fast-tier chat client for multi-query expansion
        keyword_matcher: Optional keyword matcher for exact term filtering
        entity_graph: Optional entity graph for related chunk discovery
        max_entity_expansion: Maximum related chunks to add per query
        table_store: Optional TableStore for CSV file queries
        top_k: Final number of chunks to return
        fetch_artifacts: Whether to fetch artifacts (always with score=1.0)

    Returns:
        Configured retrieval pipeline
    """
    spec = load_plugin_spec(plugin_name)

    deps = RetrievalDependencies(
        vector_client=vector_client,
        embedder=embedder,
        collection=collection,
        reranker=reranker,
        chat=chat,
        keyword_matcher=keyword_matcher,
        entity_graph=entity_graph,
        max_entity_expansion=max_entity_expansion,
        table_store=table_store,
        top_k=top_k,
        fetch_artifacts=fetch_artifacts,
    )

    steps = build_pipeline_from_spec(spec, deps)

    return RetrievalPipelineFromYaml(
        plugin_name=spec.plugin_name,
        description=spec.description,
        steps=steps,
    )


# =============================================================================
# Pipeline Class
# =============================================================================


@dataclass
class RetrievalPipelineFromYaml:
    """
    Retrieval pipeline built from YAML plugin definition.

    This is the runtime object that executes retrieval.
    """

    plugin_name: str
    description: str
    steps: list[RetrievalStep]

    def retrieve(self, query: str, filter_override: dict[str, Any] | None = None) -> list:
        """
        Execute the retrieval pipeline.

        Args:
            query: Query string
            filter_override: Optional filter to apply to vector search (for query routing)
        """
        from fitz_ai.core.chunk import Chunk
        from fitz_ai.engines.fitz_rag.retrieval.steps.vector_search import VectorSearchStep

        logger.info(f"{RETRIEVER} Running {self.plugin_name} pipeline ({len(self.steps)} steps)")

        # Apply filter override to vector search steps if provided
        if filter_override:
            for step in self.steps:
                if isinstance(step, VectorSearchStep):
                    step.filter_conditions = filter_override
                    logger.debug(f"{RETRIEVER} Applied filter override to {step.name}")
                    break

        chunks: list[Chunk] = []

        for i, step in enumerate(self.steps):
            logger.debug(f"{RETRIEVER} Step {i + 1}/{len(self.steps)}: {step.name}")
            chunks = step.execute(query, chunks)

        logger.info(f"{RETRIEVER} Pipeline complete: {len(chunks)} chunks")
        return chunks
