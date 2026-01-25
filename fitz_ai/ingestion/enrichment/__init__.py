# fitz_ai/ingestion/enrichment/__init__.py
"""
Enrichment module for enhancing document ingestion.

This module provides:
- EnrichmentPipeline: Unified entry point for all enrichment operations
- EnrichmentConfig: Configuration for enrichment behavior
- ChunkEnricher: Bus for per-chunk metadata extraction
- HierarchyEnricher: Multi-level summarization
- Artifacts: Project-level insights (navigation index, interface catalog, etc.)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    EnrichmentPipeline                        │
    │            (single entry point for all enrichment)           │
    └─────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ ChunkEnricher│    │  Hierarchy   │    │  Artifacts   │
    │    (bus)     │    │  Enricher    │    │  (project)   │
    └──────────────┘    └──────────────┘    └──────────────┘

Usage:
    from fitz_ai.ingestion.enrichment import EnrichmentPipeline, EnrichmentConfig
    from fitz_ai.llm import get_chat_factory

    # Create pipeline
    factory = get_chat_factory("cohere")
    pipeline = EnrichmentPipeline.from_config(
        config=config.get("enrichment"),
        project_root=Path("./src"),
        chat_factory=factory,
    )

    # Unified enrichment (preferred - the "box" interface)
    result = pipeline.enrich(chunks)
    enriched_chunks = result.chunks    # Chunks with summaries in metadata
    artifacts = result.artifacts        # Corpus-level artifacts
"""

# Artifact components
from fitz_ai.ingestion.enrichment.artifacts import (
    Artifact,
    ArtifactGenerator,
    ArtifactType,
    ProjectAnalysis,
    ProjectAnalyzer,
)

# Registries
from fitz_ai.ingestion.enrichment.artifacts.registry import (
    ArtifactRegistry,
    get_artifact_plugin,
    get_artifact_registry,
    list_artifact_plugins,
)

# Base types
from fitz_ai.ingestion.enrichment.base import (
    ContentType,
)

# Chunk enrichment bus
from fitz_ai.ingestion.enrichment.bus import (
    ChunkEnricher,
    ChunkEnrichmentResult,
    EnrichmentBatchResult,
    create_default_enricher,
)

# Configuration
from fitz_ai.ingestion.enrichment.config import (
    ArtifactConfig,
    EnrichmentConfig,
    HierarchyConfig,
    HierarchyRule,
)

# Hierarchy components
from fitz_ai.ingestion.enrichment.hierarchy import (
    ChunkGrouper,
    ChunkMatcher,
    HierarchyEnricher,
)

# Models
from fitz_ai.ingestion.enrichment.models import (
    EnrichmentResult,
)

# Modules
from fitz_ai.ingestion.enrichment.modules import (
    EnrichmentModule,
    EntityModule,
    KeywordModule,
    SummaryModule,
)

# Pipeline (main entry point)
from fitz_ai.ingestion.enrichment.pipeline import (
    EnrichmentPipeline,
)

# Registry
from fitz_ai.ingestion.enrichment.registry import (
    DEFAULT_MODULES,
    get_default_modules,
    get_module_by_name,
    list_available_modules,
)

__all__ = [
    # Configuration
    "EnrichmentConfig",
    "ArtifactConfig",
    "HierarchyConfig",
    "HierarchyRule",
    # Pipeline
    "EnrichmentPipeline",
    # Models
    "EnrichmentResult",
    # Chunk Enrichment Bus
    "ChunkEnricher",
    "ChunkEnrichmentResult",
    "EnrichmentBatchResult",
    "create_default_enricher",
    # Modules
    "EnrichmentModule",
    "SummaryModule",
    "KeywordModule",
    "EntityModule",
    # Module Registry
    "DEFAULT_MODULES",
    "get_default_modules",
    "get_module_by_name",
    "list_available_modules",
    # Base types
    "ContentType",
    # Hierarchy
    "HierarchyEnricher",
    "ChunkMatcher",
    "ChunkGrouper",
    # Artifacts
    "Artifact",
    "ArtifactType",
    "ArtifactGenerator",
    "ProjectAnalysis",
    "ProjectAnalyzer",
    # Artifact Registry
    "ArtifactRegistry",
    "get_artifact_registry",
    "get_artifact_plugin",
    "list_artifact_plugins",
]
