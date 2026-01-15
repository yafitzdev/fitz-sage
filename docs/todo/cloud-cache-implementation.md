# Cloud Cache Implementation TODO

**Session**: 2026-01-15
**Goal**: Complete cloud cache implementation for Fitz RAG engine
**Status**: Implementation complete, testing & validation pending

---

## ✅ Completed Tasks

### Core Implementation
- [x] Updated `fitz_ai.__version__` to 0.5.2
- [x] Added embedder and cloud_client parameters to RAGPipeline.__init__()
- [x] Updated RAGPipeline.from_config() to accept and pass cloud_client
- [x] Implemented helper methods in RAGPipeline:
  - [x] `_get_collection_version()` - computes hash from ingestion state
  - [x] `_get_llm_model_id()` - extracts model identifier from chat client
  - [x] `_answer_to_rgs_answer()` - converts Answer to RGSAnswer format
- [x] Implemented `_check_cloud_cache()` method in RAGPipeline
- [x] Implemented `_store_in_cloud_cache()` method in RAGPipeline
- [x] Integrated cache check in RAGPipeline.run() after Step 1 retrieval
- [x] Integrated cache storage in RAGPipeline.run() after answer generation
- [x] Updated FitzRagEngine to pass cloud_client to RAGPipeline.from_config()
- [x] Removed placeholder cache methods from FitzRagEngine

### Code Review Fixes (by Opus)
- [x] Fixed `_get_llm_model_id()` to use `self.chat.params["model"]`
- [x] Optimized to reuse query embedding (avoid 2nd API call on cache miss)
- [x] Added `CLOUD_OPTIMIZER_VERSION` constant
- [x] Fixed import paths for `get_logger` in cloud/client.py and engine.py

### Test Review Fixes (by Sonnet, after Opus review)
- [x] Fixed hardcoded version "0.5.2" to use `fitz_ai.__version__`
- [x] Renamed `test_embedding_cache_cleared_between_queries` to `test_embedding_computed_for_each_query` for clarity

---

## 🔄 In Progress

### Step 1: Manual Testing (Setup Complete, Execution Pending)
- [x] Create test configuration file (tests/manual/example_config_with_cache.yaml)
- [x] Create manual test script (tests/manual/test_cloud_cache.py)
- [x] Create manual test README (tests/manual/README.md)
- [ ] Set up environment variables (FITZ_ORG_ID)
- [ ] Test cache miss scenario (first query)
- [ ] Test cache hit scenario (second identical query)
- [ ] Verify cache logs and behavior
- [ ] Test with different queries
- [ ] Test cache invalidation on collection changes

### Step 2: Write Tests ✅ COMPLETE
- [x] Unit tests for cache methods (21 tests, all passing)
  - [x] Test `_get_collection_version()` determinism
  - [x] Test `_get_llm_model_id()` extraction
  - [x] Test `_answer_to_rgs_answer()` conversion
  - [x] Test `_check_cloud_cache()` with mock CloudClient
  - [x] Test `_store_in_cloud_cache()` with mock CloudClient
- [x] Integration tests (7 tests, all passing)
  - [x] Test full cache flow: miss → store → hit
  - [x] Test cache key determinism
  - [x] Test fail-open behavior on cache errors
  - [x] Test embedding reuse optimization
- [x] Edge case tests (20 tests, all passing)
  - [x] Test with cloud disabled
  - [x] Test with embedder unavailable
  - [x] Test with empty chunks
  - [x] Test cache API errors
  - [x] Test collection version computation failure
  - [x] Test LLM model ID extraction edge cases
  - [x] Test answer conversion edge cases
  - [x] Test embedding cache behavior
  - [x] Test retrieval fingerprint edge cases

**Test Coverage**: 48/48 tests passing
- `tests/unit/test_cloud_cache.py` - 21 tests
- `tests/unit/test_cloud_cache_edge_cases.py` - 20 tests
- `tests/integration/test_cloud_cache_integration.py` - 7 tests

---

## 📋 Pending Tasks

### Step 3: Documentation
- [ ] Update main README with cloud cache section
- [ ] Create cloud cache setup guide
- [ ] Document configuration options
- [ ] Add performance characteristics documentation
- [ ] Create troubleshooting guide
- [ ] Add examples of cache behavior

### Step 4: Observability
- [ ] Add metrics for cache hit rate
- [ ] Add metrics for cache latency
- [ ] Add metrics for embedding API calls saved
- [ ] Create dashboard/logging recommendations

### Step 5: Production Readiness
- [ ] Review error handling and edge cases
- [ ] Verify fail-open behavior is comprehensive
- [ ] Add cache statistics to query metadata
- [ ] Consider adding cache warming utilities
- [ ] Review security implications of cache keys

---

## 🐛 Known Issues / Tech Debt

None currently identified.

---

## 📝 Implementation Notes

### Cache Flow
```
Query arrives
  ↓
Step 1: Retrieve chunks
  ↓
Step 1.25: Check cloud cache
  - If HIT: Return cached answer (skip Steps 2-7)
  - If MISS: Continue pipeline
  ↓
Steps 2-7: Constraints → Context → RGS → LLM → Answer
  ↓
Step 8: Store in cloud cache
  ↓
Return answer
```

### Version Tracking
Cache keys include version info for automatic invalidation:
- `engine`: "0.5.2" (from `fitz_ai.__version__`)
- `optimizer`: "1.0" (hardcoded `CLOUD_OPTIMIZER_VERSION`)
- `collection`: Hash of all active files in collection
- `llm_model`: e.g., "openai:gpt-4"
- `prompt_template`: "default"

### Performance Impact
- **Cache HIT**: Saves ~1-3 seconds (LLM latency) + token costs
- **Cache MISS**: Adds ~100-200ms overhead (2 HTTP calls)
- **Break-even**: ~10% hit rate

### Key Files Modified
- `fitz_ai/__init__.py` - Updated version
- `fitz_ai/engines/fitz_rag/pipeline/engine.py` - Main cache implementation
- `fitz_ai/engines/fitz_rag/engine.py` - CloudClient wiring
- `fitz_ai/cloud/client.py` - Fixed import

---

## 🎯 Success Criteria

### Manual Testing
- [ ] First query logs "Cloud cache miss" or runs full pipeline
- [ ] Answer stored in cloud cache successfully
- [ ] Second identical query logs "Cloud cache hit"
- [ ] Second query returns in <500ms (vs 2-3s for first query)
- [ ] Cache keys are deterministic for same query + chunks
- [ ] Different queries produce different cache keys

### Test Coverage
- [ ] All cache methods have unit tests
- [ ] Integration test passes for full cache flow
- [ ] Edge cases are tested and pass
- [ ] Test coverage for cache code is >80%

### Documentation
- [ ] User can follow README to enable cloud cache
- [ ] Configuration options are documented
- [ ] Troubleshooting guide covers common issues

---

## 🔗 Related Files

- Plan file: `C:\Users\yanfi\.claude\plans\kind-strolling-planet.md`
- Backend: `fitz-ai-cloud` (deployed to Railway)
- Client library: `fitz_ai/cloud/` (client.py, crypto.py, cache_key.py, config.py)
- Main implementation: `fitz_ai/engines/fitz_rag/pipeline/engine.py`

---

## 🚀 Next Session

After completing this session's work:
1. Deploy to staging/test environment
2. Monitor cache hit rate and performance
3. Gather user feedback
4. Iterate on cache warming strategies
5. Consider semantic cache (fuzzy matching) for similar queries
