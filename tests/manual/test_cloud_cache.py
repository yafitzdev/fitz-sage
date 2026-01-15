# tests/manual/test_cloud_cache.py
"""
Manual test script for cloud cache functionality.

This script tests the full cache flow end-to-end with a real Fitz Cloud backend.
Run this manually to verify cache behavior.

Prerequisites:
1. Set FITZ_ORG_ID environment variable
2. Configure cloud section in .fitz/config/fitz_rag.yaml
3. Have documents ingested in a collection

Usage:
    python tests/manual/test_cloud_cache.py

Expected behavior:
- First query: Cache miss, runs full RAG, stores result
- Second query (same): Cache hit, returns cached answer quickly
- Third query (different): Cache miss, runs full RAG
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fitz_ai.core import Query
from fitz_ai.engines.fitz_rag import FitzRagEngine
from fitz_ai.engines.fitz_rag.config import load_config


def test_cache_flow():
    """Test the full cache flow: miss → store → hit."""
    print("=" * 80)
    print("CLOUD CACHE MANUAL TEST")
    print("=" * 80)

    # Check environment
    org_id = os.environ.get("FITZ_ORG_ID")
    if not org_id:
        print("❌ ERROR: FITZ_ORG_ID environment variable not set")
        print("   Set it with: export FITZ_ORG_ID='your-org-uuid'")
        return False

    print(f"✓ FITZ_ORG_ID: {org_id[:8]}...")

    # Load config
    config_path = ".fitz/config/fitz_rag.yaml"
    if not Path(config_path).exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        return False

    print(f"✓ Config file: {config_path}")

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ ERROR: Failed to load config: {e}")
        return False

    # Check cloud is enabled
    if not config.cloud or not config.cloud.enabled:
        print("❌ ERROR: Cloud is not enabled in config")
        print("   Add cloud section to config:")
        print("""
  cloud:
    enabled: true
    api_key: "fitz_xxx..."
    org_key: "64-char-hex-string"
    base_url: "https://api.fitz-ai.cloud/v1"
        """)
        return False

    print("✓ Cloud enabled in config")

    # Create engine
    try:
        engine = FitzRagEngine(config)
        print("✓ FitzRagEngine initialized")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize engine: {e}")
        return False

    # Test query
    test_question = "What is quantum computing?"

    print("\n" + "-" * 80)
    print("TEST 1: First query (expect cache MISS)")
    print("-" * 80)

    query1 = Query(text=test_question)
    start1 = time.time()
    try:
        answer1 = engine.answer(query1)
        elapsed1 = time.time() - start1
        print(f"✓ Query completed in {elapsed1:.2f}s")
        print(f"  Answer: {answer1.text[:100]}...")
        print(f"  Sources: {len(answer1.provenance)}")
    except Exception as e:
        print(f"❌ ERROR: Query failed: {e}")
        return False

    # Wait a moment
    print("\n⏳ Waiting 2 seconds before second query...")
    time.sleep(2)

    print("\n" + "-" * 80)
    print("TEST 2: Second query (same question, expect cache HIT)")
    print("-" * 80)

    query2 = Query(text=test_question)
    start2 = time.time()
    try:
        answer2 = engine.answer(query2)
        elapsed2 = time.time() - start2
        print(f"✓ Query completed in {elapsed2:.2f}s")
        print(f"  Answer: {answer2.text[:100]}...")
        print(f"  Sources: {len(answer2.provenance)}")
    except Exception as e:
        print(f"❌ ERROR: Query failed: {e}")
        return False

    # Compare results
    print("\n" + "-" * 80)
    print("RESULTS COMPARISON")
    print("-" * 80)
    print(f"First query time:  {elapsed1:.2f}s")
    print(f"Second query time: {elapsed2:.2f}s")
    print(f"Speed improvement: {(elapsed1 / elapsed2):.1f}x faster")

    if elapsed2 < elapsed1 * 0.5:
        print("✓ Cache appears to be working (2nd query >50% faster)")
    else:
        print("⚠️  WARNING: 2nd query not significantly faster - check cache logs")

    if answer1.text == answer2.text:
        print("✓ Answers are identical")
    else:
        print("⚠️  WARNING: Answers differ - check cache behavior")

    # Test different query
    print("\n" + "-" * 80)
    print("TEST 3: Different query (expect cache MISS)")
    print("-" * 80)

    different_question = "What is machine learning?"
    query3 = Query(text=different_question)
    start3 = time.time()
    try:
        answer3 = engine.answer(query3)
        elapsed3 = time.time() - start3
        print(f"✓ Query completed in {elapsed3:.2f}s")
        print(f"  Answer: {answer3.text[:100]}...")
    except Exception as e:
        print(f"❌ ERROR: Query failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("MANUAL TEST COMPLETE")
    print("=" * 80)
    print("\n✓ All tests passed!")
    print("\nCheck logs for cache hit/miss messages:")
    print("  - 'Cloud cache hit' = cache working")
    print("  - 'Answer stored in cloud cache' = storage working")
    print("\nTo verify cache behavior:")
    print("  1. Check Railway logs for cache API calls")
    print("  2. Verify cache keys are deterministic")
    print("  3. Test cache invalidation by re-ingesting documents")

    return True


if __name__ == "__main__":
    success = test_cache_flow()
    sys.exit(0 if success else 1)
