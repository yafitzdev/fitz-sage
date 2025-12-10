"""
Command-line ingestion tool for fitz_ingest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from fitz_ingest.chunker.simple_chunker import SimpleChunker
from fitz_ingest.ingester.engine import IngestionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fitz_ingest CLI")

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="File or directory to ingest",
    )

    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection name",
    )

    parser.add_argument(
        "--vector-size",
        type=int,
        required=True,
        help="Vector dimension (e.g., 1536)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Qdrant host",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    client = QdrantClient(
        host=args.host,
        port=args.port,
    )

    chunker = SimpleChunker(chunk_size=args.chunk_size)

    engine = IngestionEngine(
        client=client,
        collection=args.collection,
        vector_size=args.vector_size,
        embedder=None,  # user injects one manually if needed
    )

    target_path = Path(args.path)

    if target_path.is_file():
        engine.ingest_file(chunker, target_path)
    else:
        engine.ingest_path(chunker, target_path)

    print(f"Ingestion complete → {args.collection}")


if __name__ == "__main__":
    main()
