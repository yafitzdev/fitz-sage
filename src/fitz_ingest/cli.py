"""
Command-line ingestion tool for fitz_ingest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.plugins.simple import SimpleChunker
from fitz_ingest.ingester.engine import IngestionEngine

from fitz_ingest.exceptions.base import IngestionError
from fitz_ingest.exceptions.config import IngestionConfigError
from fitz_ingest.exceptions.vector import IngestionVectorError
from fitz_ingest.exceptions.chunking import IngestionChunkingError


# ============================================================
# Chunker plugin registry
# ============================================================

CHUNKER_REGISTRY = {
    "simple": SimpleChunker,
    # Future plugins:
    # "fixed_word": FixedWordChunker,
    # "markdown": MarkdownChunker,
    # "json": JSONHierarchyChunker,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fitz_ingest CLI")

    parser.add_argument("--path", type=str, required=True, help="File or directory to ingest")
    parser.add_argument("--collection", type=str, required=True, help="Qdrant collection name")
    parser.add_argument("--vector-size", type=int, required=True, help="Vector dimension (e.g., 1536)")
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")

    # NEW: choose the chunker plugin
    parser.add_argument(
        "--chunker",
        type=str,
        default="simple",
        help="Chunker plugin to use (default: simple)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (used only by simple chunker)"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Qdrant connection
    # ---------------------------------------------------------
    try:
        client = QdrantClient(host=args.host, port=args.port)
    except Exception as e:
        raise IngestionVectorError(
            f"Failed to connect to Qdrant at {args.host}:{args.port}"
        ) from e

    # ---------------------------------------------------------
    # Load chunker plugin
    # ---------------------------------------------------------
    plugin_name = args.chunker.lower()

    if plugin_name not in CHUNKER_REGISTRY:
        raise IngestionConfigError(
            f"Unknown chunker plugin '{plugin_name}'. "
            f"Available: {', '.join(CHUNKER_REGISTRY.keys())}"
        )

    PluginClass = CHUNKER_REGISTRY[plugin_name]

    # Plugin constructor signature differs per plugin.
    # For simple chunker, we pass chunk_size.
    try:
        if plugin_name == "simple":
            plugin = PluginClass(chunk_size=args.chunk_size)
        else:
            plugin = PluginClass()  # For future plugins
    except Exception as e:
        raise IngestionConfigError(f"Failed to initialize chunker plugin '{plugin_name}': {e}") from e

    chunker_engine = ChunkingEngine(plugin)

    # ---------------------------------------------------------
    # Build ingestion engine
    # ---------------------------------------------------------
    engine = IngestionEngine(
        client=client,
        collection=args.collection,
        vector_size=args.vector_size,
        chunker_engine=chunker_engine,
        embedder=None,  # (Not implemented yet for fitz_ingest)
    )

    target_path = Path(args.path)

    # ---------------------------------------------------------
    # Run ingestion
    # ---------------------------------------------------------
    try:
        if target_path.is_file():
            engine.ingest_file(target_path)
        else:
            engine.ingest_path(target_path)
    except (IngestionConfigError, IngestionVectorError, IngestionChunkingError) as e:
        raise
    except IngestionError as e:
        raise
    except Exception as e:
        raise IngestionError(f"Unexpected error during ingestion: {e}") from e

    print(f"Ingestion complete → {args.collection}")


if __name__ == "__main__":
    main()
