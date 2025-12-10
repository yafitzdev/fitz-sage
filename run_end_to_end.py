"""
End-to-end test: fitz_ingest + fitz_rag + Qdrant + Cohere.

Flow:
  1) Ingest .txt files from DATA_PATH using fitz_ingest.SimpleChunker
  2) Embed & store in Qdrant
  3) Dense retrieve from Qdrant
  4) Rerank with CohereRerankClient
  5) Build context with fitz_rag.context.builder
  6) Ask CohereChatClient with a clean prompt
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------------------------------------------------------------------
# 0) Make ./src importable → import fitz_ingest, fitz_rag
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fitz_ingest.chunker.simple_chunker import SimpleChunker
from fitz_rag.context.builder import build_context
from fitz_rag.llm.chat_client import CohereChatClient
from fitz_rag.llm.rerank_client import CohereRerankClient  # :contentReference[oaicite:1]{index=1}


# ---------------------------------------------------------------------
# 1) Config
# ---------------------------------------------------------------------
QDRANT_HOST = "192.168.178.2"
QDRANT_PORT = 6333
COLLECTION = "fitz_stack_test"

DATA_PATH = Path(r"C:\Users\yanfi\Downloads\test_data")

EMBED_MODEL = "embed-english-v3.0"
VECTOR_SIZE = 1024            # must match embedding dim
TOP_K = 15                    # dense candidates before rerank
TOP_K_RERANK = 5              # after rerank, how many to keep


# ---------------------------------------------------------------------
# 2) Cohere Embedder (v2 embed API)
# ---------------------------------------------------------------------
class CohereEmbedder:
    """
    Small helper around Cohere embeddings.

    Uses:
      - input_type="search_document" for stored chunks
      - input_type="search_query" for user queries
    """

    def __init__(self, api_key: str, model: str = EMBED_MODEL):
        self._client = cohere.Client(api_key)
        self.model = model

    def embed_doc(self, text: str) -> List[float]:
        resp = self._client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document",
        )
        return resp.embeddings[0]

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query",
        )
        return resp.embeddings[0]


# ---------------------------------------------------------------------
# 3) Ingestion
# ---------------------------------------------------------------------
def ingest_folder(
    client: QdrantClient,
    embedder: CohereEmbedder,
    folder: Path,
    collection: str,
) -> None:
    if not folder.is_dir():
        raise ValueError(f"Data path does not exist or is not a directory: {folder}")

    # For this demo: recreate the collection each run
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    chunker = SimpleChunker(chunk_size=500)
    points: List[PointStruct] = []

    for file in folder.rglob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunker.chunk_file(str(file))

        for ch in chunks:
            vec = embedder.embed_doc(ch.text)
            payload: Dict[str, Any] = {
                "text": ch.text,
                "file": str(file),
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                )
            )

    if not points:
        print(f"No .txt files found under {folder}")
        return

    client.upsert(collection_name=collection, points=points)
    print(f"Ingested {len(points)} chunks into collection '{collection}'.")


# ---------------------------------------------------------------------
# 4) Retrieval + rerank + context
# ---------------------------------------------------------------------
def retrieve_context(
    client: QdrantClient,
    embedder: CohereEmbedder,
    reranker: CohereRerankClient,
    collection: str,
    query: str,
    max_chars: int = 4000,
) -> str:
    # Dense retrieval
    query_vec = embedder.embed_query(query)

    result = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=TOP_K,
        with_payload=True,
    )

    hits = result.points
    if not hits:
        return "No relevant context found."

    texts = [h.payload.get("text", "") for h in hits]

    # Rerank using your official client (returns indices)
    order = reranker.rerank(query, texts, top_n=TOP_K_RERANK)

    # Map indices back to chunks
    selected_chunks: List[Dict[str, Any]] = []
    for idx in order:
        payload = hits[idx].payload or {}
        selected_chunks.append(
            {
                "text": payload.get("text", ""),
                "file": payload.get("file", "unknown"),
            }
        )

    # Build RAG context (merge/group/dedupe/pack)
    context = build_context(selected_chunks, max_chars=max_chars)
    return context


# ---------------------------------------------------------------------
# 5) Orchestration
# ---------------------------------------------------------------------
def run_end_to_end() -> None:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = CohereEmbedder(api_key)
    reranker = CohereRerankClient(api_key=api_key)
    chat = CohereChatClient(api_key=api_key)

    print("=== INGEST ===")
    ingest_folder(client, embedder, DATA_PATH, COLLECTION)

    print("\n=== RAG QUERY ===")
    query = "What are these documents roughly about?"
    context = retrieve_context(client, embedder, reranker, COLLECTION, query)

    user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer ONLY using the CONTEXT above.
- If the context is insufficient, say that you don't know.
""".strip()

    system_prompt = (
        "You are a precise assistant. "
        "Use only the given CONTEXT. If it does not contain enough information, say so explicitly."
    )

    answer = chat.chat(system_prompt=system_prompt, user_content=user_prompt)

    print("\n=== FINAL PROMPT SENT TO LLM ===")
    print(user_prompt)

    print("\n=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    run_end_to_end()
