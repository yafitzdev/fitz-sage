"""
Integration tests for Cohere LLM components:
- Embedding
- Reranking
- Chat

IMPORTANT:
These tests require COHERE_API_KEY to be present in the environment.
If it's missing, the tests will skip automatically.
"""

from __future__ import annotations
import os
import inspect

from fitz_rag.llm.embedding.plugins.cohere import CohereEmbeddingClient
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient
from fitz_rag.llm.chat.plugins.cohere import CohereChatClient


def require_api_key():
    key = os.getenv("COHERE_API_KEY")
    if not key:
        print("❌ COHERE_API_KEY not set — skipping live tests.")
        return None
    return key


# ---------------------------------------------------------
# Test: Embedding
# ---------------------------------------------------------
def test_embedding():
    if not require_api_key():
        return

    print("\n=== TEST: Cohere Embedding ===")

    embedder = CohereEmbeddingClient()

    text = "Hello world from Fitz-RAG!"
    vec = embedder.embed(text)

    print(f"Embedding length: {len(vec)}")
    print(f"First 5 values: {vec[:5]}")

    assert isinstance(vec, list)
    assert len(vec) > 10

    print("✅ Embedding test passed.")


# ---------------------------------------------------------
# Test: Reranking
# ---------------------------------------------------------
def test_rerank():
    if not require_api_key():
        return

    print("\n=== TEST: Cohere Rerank ===")

    reranker = CohereRerankClient()

    query = "What is torque?"
    docs = [
        "Torque is a measure of rotational force.",
        "Bananas are yellow.",
        "Cars use engines to generate torque.",
    ]

    # Introspect the signature to call correctly
    sig = inspect.signature(reranker.rerank)
    params = sig.parameters

    if "top_n" in params:
        order = reranker.rerank(query, docs, top_n=2)
    elif "limit" in params:
        order = reranker.rerank(query, docs, limit=2)
    else:
        # fallback: assume no param needed
        order = reranker.rerank(query, docs)

    print(f"Rerank order: {order}")

    assert isinstance(order, list)
    assert len(order) >= 1  # at least one result

    print("Top results:")
    for idx in order:
        print(f"- {docs[idx]}")

    print("✅ Rerank test passed.")


# ---------------------------------------------------------
# Test: Chat
# ---------------------------------------------------------
def test_chat():
    if not require_api_key():
        return

    print("\n=== TEST: Cohere Chat ===")

    chat = CohereChatClient()

    response = chat.chat([
        {"role": "system", "content": "You are a helpful test assistant."},
        {"role": "user", "content": "Tell me something about RAG systems."},
    ])

    print(f"Response: {response}")

    assert isinstance(response, str)
    assert len(response) > 5

    print("✅ Chat test passed.")
