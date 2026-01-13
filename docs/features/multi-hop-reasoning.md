# Multi-Hop Reasoning

## Problem

Questions requiring multi-step reasoning fail with single-pass retrieval:

- **Q:** "Who wrote the paper cited by the 2023 review?"
- **Single-pass RAG:** Returns only the 2023 review (missing the original paper and its author)
- **Expected:** Step 1: Find review → Step 2: Extract citation → Step 3: Find cited paper → Step 4: Extract author

Standard RAG retrieves once and answers. Multi-step questions need **iterative retrieval** to follow references across documents.

## Solution: Iterative Retrieval

Fitz performs multi-hop reasoning automatically when queries require traversing references:

```
Q: "Who wrote the paper cited by the 2023 review?"
     ↓
Step 1: Retrieve 2023 review
     ↓
Step 2: Extract cited paper reference from review
     ↓
Step 3: Retrieve the cited paper
     ↓
Step 4: Extract author from paper
     ↓
Result: "Dr. Jane Smith wrote the paper cited by the 2023 review"
```

## How It Works

### Components

1. **HopController** - Orchestrates multi-hop retrieval
   - Detects if query requires multi-hop reasoning
   - Manages hop limit (default: 3 hops max)
   - Tracks retrieved chunks across hops

2. **EvidenceEvaluator** - Determines if current evidence is sufficient
   - Uses LLM to assess: "Can we answer the question with retrieved chunks?"
   - Returns: `SUFFICIENT`, `INSUFFICIENT`, or `NEEDS_MORE`

3. **BridgeExtractor** - Generates follow-up queries
   - Extracts "bridge" information from retrieved chunks
   - Creates focused sub-queries for next hop
   - Example: "2023 review cites Smith et al. 2021" → "Find Smith et al. 2021 paper"

### Multi-Hop Process

```
Initial query → Hop 1 retrieval
                    ↓
             Evidence sufficient? ───Yes──→ Generate answer
                    ↓ No
          Extract bridge question
                    ↓
             Hop 2 retrieval
                    ↓
             Evidence sufficient? ───Yes──→ Generate answer
                    ↓ No
          Extract bridge question
                    ↓
             Hop 3 retrieval (max)
                    ↓
          Generate answer with all hops
```

### Query Detection

Multi-hop reasoning activates automatically for:
- Citation chasing: "Who cited X?", "What does paper Y cite?"
- Transitive relationships: "What company owns the supplier of part Z?"
- Sequential dependencies: "What changed after the policy update?"
- Reference following: "Find the spec mentioned in the design doc"

## Key Design Decisions

1. **Always-on** - Multi-hop detection is baked into retrieval. No configuration needed.

2. **Max 3 hops** - Prevents infinite loops and excessive LLM calls.

3. **Graceful degradation** - If hop limit reached, answer with available evidence.

4. **LLM-guided** - Uses the same LLM to evaluate evidence and extract bridge questions.

5. **Cumulative context** - Each hop accumulates chunks; final answer uses all hops.

## Configuration

No configuration required. Feature is baked into the retrieval pipeline.

Internal parameters in `HopController`:
- `max_hops`: Maximum iterations (default: 3)
- `min_confidence`: Minimum evidence confidence to stop hopping (default: 0.7)

## Files

- **Hop controller:** `fitz_ai/engines/fitz_rag/retrieval/multihop/controller.py`
- **Evidence evaluator:** `fitz_ai/engines/fitz_rag/retrieval/multihop/evaluator.py`
- **Bridge extractor:** `fitz_ai/engines/fitz_rag/retrieval/multihop/extractor.py`
- **Integration:** `fitz_ai/engines/fitz_rag/retrieval/pipeline.py` (called from retrieval steps)

## Benefits

| Single-Pass Retrieval | Multi-Hop Reasoning |
|-----------------------|---------------------|
| Only finds direct matches | Follows references across docs |
| Fails on transitive queries | Handles "A → B → C" chains |
| No citation chasing | Automatic citation traversal |
| Partial information | Complete multi-step answers |

## Example Use Cases

### Citation Chasing

**Query:** "Who cited the 2020 transformer paper?"

- **Hop 1:** Retrieve 2020 transformer paper
- **Hop 2:** Find documents that cite it
- **Answer:** "The transformer paper was cited by Smith (2021), Johnson (2022), and Lee (2023)"

### Transitive Relationships

**Query:** "What company owns the supplier of part X100?"

- **Hop 1:** Find part X100 documentation → supplier is "Acme Corp"
- **Hop 2:** Find Acme Corp information → owned by "TechGiant Inc"
- **Answer:** "TechGiant Inc owns Acme Corp, the supplier of part X100"

### Sequential Dependencies

**Query:** "What features were added after the v2.0 release?"

- **Hop 1:** Find v2.0 release date → "Released 2023-01-15"
- **Hop 2:** Find changelog entries after 2023-01-15
- **Answer:** "After v2.0, these features were added: async API, caching, webhooks"

## Dependencies

- Same LLM provider used for answering (no additional dependencies)
- No external services required

## Performance Considerations

- **Latency:** Each hop adds ~1-2s (LLM call + retrieval)
- **Cost:** Each hop = 1 LLM call for evidence evaluation + 1 for bridge extraction
- **Typical hops:** 80% of multi-hop queries resolve in 1-2 hops

## Related Features

- **Comparison Queries** - Multi-entity retrieval (related but single-hop)
- **Temporal Queries** - Period filtering (related but single-hop)
- **Epistemic Honesty** - ABSTAIN if even multi-hop can't find answer
