"""
Command-line ingestion tool for fitz_ingest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from fitz_ingest.chunker.simple_chunker import SimpleChunker
from fitz_ingest.ingester.engine import IngestionEngine

from fitz_rag.exceptions.config import ConfigError
from fitz_rag.exceptions.retriever import VectorSearchError
from fitz_rag.exceptions.pipeline import PipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fitz_ingest CLI")

    parser.add_argument("--path", type=str, required=True, help="File or directory to ingest")
    parser.add_argument("--collection", type=str, required=True, help="Qdrant collection name")
    parser.add_argument("--vector-size", type=int, required=True, help="Vector dimension (e.g., 1536)")
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = QdrantClient(host=args.host, port=args.port)
    except Exception as e:
        raise VectorSearchError(f"Failed to connect to Qdrant at {args.host}:{args.port}") from e

    chunker = SimpleChunker(chunk_size=args.chunk_size)

    engine = IngestionEngine(
        client=client,
        collection=args.collection,
        vector_size=args.vector_size,
        embedder=None,
    )

    target_path = Path(args.path)

    try:
        if target_path.is_file():
            engine.ingest_file(chunker, target_path)
        else:
            engine.ingest_path(chunker, target_path)
    except (ConfigError, VectorSearchError, PipelineError) as e:
        raise
    except Exception as e:
        raise PipelineError(f"Unexpected error during ingestion: {e}") from e

    print(f"Ingestion complete → {args.collection}")


if __name__ == "__main__":
    main()
