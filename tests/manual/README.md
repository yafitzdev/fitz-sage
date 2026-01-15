# Manual Tests

This directory contains manual test scripts for features that require live external services.

## Cloud Cache Testing

Test the Fitz Cloud cache functionality end-to-end.

### Prerequisites

1. **Fitz Cloud Account**: Sign up at https://fitz-ai.cloud (or use your deployed Railway instance)

2. **Generate Encryption Key**:
   ```bash
   python -c "import os; print(os.urandom(32).hex())"
   ```

3. **Set Environment Variable**:
   ```bash
   # On Linux/Mac:
   export FITZ_ORG_ID="your-org-uuid-here"

   # On Windows (PowerShell):
   $env:FITZ_ORG_ID="your-org-uuid-here"

   # On Windows (CMD):
   set FITZ_ORG_ID=your-org-uuid-here
   ```

4. **Configure Cloud Settings**:
   - Copy `example_config_with_cache.yaml` to `.fitz/config/fitz_rag.yaml`
   - Fill in your `api_key` and `org_key`
   - Update other settings (chat, embedding, vector_db) as needed

5. **Ingest Test Documents**:
   ```bash
   fitz ingest ./docs --collection test_docs
   ```

### Running the Test

```bash
python tests/manual/test_cloud_cache.py
```

### Expected Output

```
================================================================================
CLOUD CACHE MANUAL TEST
================================================================================
✓ FITZ_ORG_ID: 12345678...
✓ Config file: .fitz/config/fitz_rag.yaml
✓ Cloud enabled in config
✓ FitzRagEngine initialized

--------------------------------------------------------------------------------
TEST 1: First query (expect cache MISS)
--------------------------------------------------------------------------------
✓ Query completed in 2.34s
  Answer: Quantum computing is a type of computing that uses quantum-mechanical...
  Sources: 5

⏳ Waiting 2 seconds before second query...

--------------------------------------------------------------------------------
TEST 2: Second query (same question, expect cache HIT)
--------------------------------------------------------------------------------
✓ Query completed in 0.18s
  Answer: Quantum computing is a type of computing that uses quantum-mechanical...
  Sources: 5

--------------------------------------------------------------------------------
RESULTS COMPARISON
--------------------------------------------------------------------------------
First query time:  2.34s
Second query time: 0.18s
Speed improvement: 13.0x faster
✓ Cache appears to be working (2nd query >50% faster)
✓ Answers are identical

--------------------------------------------------------------------------------
TEST 3: Different query (expect cache MISS)
--------------------------------------------------------------------------------
✓ Query completed in 2.15s
  Answer: Machine learning is a subset of artificial intelligence...

================================================================================
MANUAL TEST COMPLETE
================================================================================

✓ All tests passed!
```

### What to Check

1. **Logs**: Look for these messages in the output:
   - `"Cloud cache hit"` - Cache is working correctly
   - `"Answer stored in cloud cache"` - Storage is working
   - `"Cloud cache lookup failed"` - Check your credentials

2. **Performance**: Second identical query should be 5-10x faster

3. **Cache Keys**: Check Railway logs to see cache key computation

4. **Version Tracking**: Verify cache invalidates when:
   - Documents are re-ingested (collection version changes)
   - Model is changed in config (llm_model version changes)
   - fitz-ai is upgraded (engine version changes)

### Troubleshooting

**Error: `FITZ_ORG_ID environment variable not set`**
- Set the environment variable as shown above

**Error: `cloud.api_key is required when cloud.enabled=true`**
- Add `api_key` to the `cloud:` section in your config

**Error: `cloud.org_key is required when cloud.enabled=true`**
- Generate an encryption key and add it to config

**Second query not faster:**
- Check Railway logs for cache API calls
- Verify `cloud.enabled: true` in config
- Check for cache lookup errors in logs
- Ensure identical query text (case-sensitive)

**Cache not invalidating after re-ingestion:**
- Verify collection version is computed correctly
- Check `.fitz/ingest.json` for updated file hashes
- Clear cache manually if needed (future feature)
