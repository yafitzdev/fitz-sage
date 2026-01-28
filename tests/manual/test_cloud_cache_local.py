# tests/manual/test_cloud_cache_local.py
"""
Test Fitz Cloud cache integration with local API.

Run with: python tests/manual/test_cloud_cache_local.py

Prerequisites:
- fitz-ai-cloud API running at localhost:8000
- OpenAI API key in environment (OPENAI_API_KEY)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fitz_ai.cloud.client import CloudClient
from fitz_ai.cloud.config import CloudConfig
from fitz_ai.cloud.cache_key import CacheVersions
from fitz_ai.core import Answer, Provenance


def get_embedding(text: str) -> list[float]:
    """Get embedding - tries OpenAI (1536-dim), falls back to deterministic mock (768-dim)."""
    import hashlib

    # Try OpenAI first
    try:
        import openai
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"    (OpenAI unavailable: {str(e)[:50]}... using mock embedding)")

    # Fallback: deterministic mock embedding based on text hash
    # This allows testing cache mechanics without OpenAI
    # Using 768-dim to test multi-dimension support (like nomic-embed-text)
    import random

    # Seed random with hash for deterministic results
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    random.seed(hash_val)

    # Generate 768 floats in [-1, 1] range (nomic-embed-text dimension)
    embedding = [random.uniform(-1, 1) for _ in range(768)]

    # Normalize the vector
    norm = sum(x*x for x in embedding) ** 0.5
    return [x / norm for x in embedding]


def main():
    print("=" * 60)
    print("FITZ CLOUD CACHE INTEGRATION TEST")
    print("=" * 60)

    # Configuration for local testing
    config = CloudConfig(
        enabled=True,
        api_key="fitz_97d05efb982f24d2a05c6d44898869e2694dc0ae6f6032a0",
        org_key="6624e9824e80156a446829b47cfe9d497f376f0d28c1e0e874bc951f55cb77ef",
        base_url="http://localhost:8000/v1",
        timeout=30,
    )
    org_id = "4462b5ae-66d3-4903-89d1-05e804ce8af6"

    # Create cloud client
    print("\n[1] Creating CloudClient...")
    client = CloudClient(config, org_id)
    print("    OK - CloudClient created")

    # Check features
    print("\n[2] Checking tier features...")
    features = client.get_features()
    if features:
        print(f"    [OK] Tier: {features.tier}")
        print(f"    [OK] Cache enabled: {features.features.get('cross_cache', False)}")
    else:
        print("    [FAIL] Failed to get features")
        return

    # Test query
    query_text = "What is the tallest mountain in the world?"
    print(f"\n[3] Test query: '{query_text}'")

    # Get embedding
    print("\n[4] Getting embedding from OpenAI...")
    try:
        query_embedding = get_embedding(query_text)
        print(f"    [OK] Got {len(query_embedding)}-dimensional embedding")
    except Exception as e:
        print(f"    [FAIL] Failed to get embedding: {e}")
        print("    Make sure OPENAI_API_KEY is set")
        return

    # Create versions
    versions = CacheVersions(
        optimizer="1.0.0",
        engine="fitz_rag",
        collection="test_collection_v1",
        llm_model="gpt-4o-mini",
        prompt_template="default_v1",
    )
    retrieval_fingerprint = "test_chunk_abc123"

    # Test 1: Cache lookup (should be MISS)
    print("\n[5] Cache lookup #1 (expecting MISS)...")
    try:
        result = client.lookup_cache(
            query_text=query_text,
            query_embedding=query_embedding,
            retrieval_fingerprint=retrieval_fingerprint,
            versions=versions,
        )
    except Exception as e:
        print(f"    [FAIL] Exception during lookup: {e}")
        import traceback
        traceback.print_exc()
        return

    if result.hit:
        print("    ! Unexpected cache HIT (entry already exists)")
        print(f"    Answer: {result.answer.text[:100] if result.answer else 'N/A'}...")
    else:
        print("    [OK] Cache MISS (as expected)")
        if result.routing:
            print(f"    [OK] Routing advice: complexity={result.routing.complexity}")

    # Simulate LLM response
    print("\n[6] Simulating LLM response...")
    answer = Answer(
        text="Mount Everest is the tallest mountain in the world at 8,849 meters (29,032 feet) above sea level.",
        provenance=[
            Provenance(
                source_id="doc_456",
                excerpt="Mount Everest, located in the Himalayas, is the highest peak on Earth.",
                metadata={"page": 1},
            )
        ],
        mode=None,
        metadata={"model": "gpt-4o-mini", "tokens": 42},
    )
    print(f"    [OK] Answer: {answer.text[:60]}...")

    # Store in cache
    print("\n[7] Storing answer in cache...")

    # Debug: manually test the store endpoint
    import httpx
    import json
    import base64
    from fitz_ai.cloud.cache_key import compute_cache_key
    from fitz_ai.cloud.crypto import CacheEncryption

    cache_key = compute_cache_key(query_text, retrieval_fingerprint, versions)
    print(f"    Cache key: {cache_key[:32]}...")

    # Manually encrypt
    encryption = CacheEncryption(config.org_key)
    answer_data = {
        "text": answer.text,
        "provenance": [{"source_id": p.source_id, "excerpt": p.excerpt, "metadata": p.metadata} for p in answer.provenance],
        "mode": None,
        "metadata": answer.metadata,
    }
    blob = encryption.encrypt(json.dumps(answer_data), org_id)
    print(f"    Encrypted blob size: {len(blob.ciphertext)} bytes")

    # Try direct HTTP request
    payload = {
        "cache_key": cache_key,
        "query_embedding": query_embedding,
        "retrieval_fingerprint": retrieval_fingerprint,
        "encrypted_blob": base64.b64encode(blob.ciphertext).decode(),
        "timestamp": str(blob.timestamp),
        "versions": {
            "optimizer": versions.optimizer,
            "engine": versions.engine,
            "collection": versions.collection,
            "llm_model": versions.llm_model,
            "prompt_template": versions.prompt_template,
        },
        "metadata": {"model_used": "gpt-4o-mini", "tokens_output": 42},
    }

    print("    Sending direct HTTP request to /cache/store...")
    try:
        resp = httpx.post(
            f"{config.base_url}/cache/store",
            json=payload,
            headers={"X-API-Key": config.api_key, "Content-Type": "application/json"},
            timeout=30,
        )
        print(f"    Response status: {resp.status_code}")
        print(f"    Response body: {resp.text[:500]}")
        stored = resp.status_code == 200 and resp.json().get("stored", False)
    except Exception as e:
        print(f"    HTTP error: {e}")
        import traceback
        traceback.print_exc()
        stored = False

    if stored:
        print("    [OK] Answer stored in cache")
    else:
        print("    [FAIL] Failed to store in cache")
        return

    # Test 2: Cache lookup (should be HIT)
    print("\n[8] Cache lookup #2 (expecting HIT)...")
    result2 = client.lookup_cache(
        query_text=query_text,
        query_embedding=query_embedding,
        retrieval_fingerprint=retrieval_fingerprint,
        versions=versions,
    )

    if result2.hit:
        print("    [OK] Cache HIT!")
        print(f"    [OK] Decrypted answer: {result2.answer.text[:60]}...")
        if result2.answer.provenance:
            print(f"    [OK] Provenance preserved: {len(result2.answer.provenance)} source(s)")
    else:
        print("    [FAIL] Unexpected cache MISS")
        return

    # Summary
    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    print("""
What happened:
1. Query embedding generated (any dimension supported)
2. Cache lookup #1 -> MISS (no cached answer)
3. Answer encrypted locally with org_key
4. Encrypted blob stored in Fitz Cloud
5. Cache lookup #2 -> HIT (answer decrypted locally)

The cloud NEVER saw:
- The query text (only embedding vector)
- The answer text (only encrypted blob)
- Your org_key (stays local)
""")

    client.close()


if __name__ == "__main__":
    main()
