# HyDE (Hypothetical Document Embeddings)

## Problem

Abstract or conceptual queries often embed far from concrete documents:

- "What's TechCorp's approach to sustainability?" - Concrete docs discuss specific EV features, battery specs, emissions data
- "How does the architecture ensure reliability?" - Technical docs describe specific patterns, not abstract concepts
- "What would I need to know before buying?" - Documents have specs and prices, not buyer-oriented passages

The semantic gap between question phrasing and document content leads to poor retrieval recall.

## Solution: Hypothetical Document Generation

Generate hypothetical documents that would answer the query, then search with both the original query and the hypothetical documents:

```
Original query:     "What's TechCorp's approach to sustainability?"
                              ↓
Generate hypotheses: LLM creates 3 hypothetical document passages
                     that WOULD contain the answer
                              ↓
Hypothetical docs:  ["TechCorp's sustainability strategy centers on...",
                     "The company's environmental commitment includes...",
                     "TechCorp addresses climate concerns through..."]
                              ↓
                    Search with original + all hypotheses
                              ↓
                    RRF fusion of results
```

## How It Works

### At Query Time

1. Query is sent to fast-tier LLM with hypothesis generation prompt
2. LLM generates 3 hypothetical document passages (single call)
3. Each hypothesis is embedded alongside the original query
4. All embeddings search the vector database
5. Results are merged using Reciprocal Rank Fusion (RRF)

### Why Hypothetical Documents Help

Hypothetical documents are written in document style, not question style:
- They contain concrete terminology that appears in real documents
- They bridge the vocabulary gap between queries and documents
- They embed closer to actual relevant documents than the query alone

## Key Design Decisions

1. **Always-on** - Activates automatically when chat client is available. No configuration.

2. **Single LLM call** - All 3 hypotheses generated in one prompt, not 3 separate calls.

3. **Fast tier** - Uses the fast-tier model for background generation (speed-optimized).

4. **Hybrid search** - Hypotheses extend query variations, merged with RRF alongside:
   - Original query embedding
   - Synonym/acronym expansions
   - Sparse (BM25) search results

5. **Graceful degradation** - If hypothesis generation fails, search continues with original query.

6. **Externalized prompt** - Prompt template in `prompts/hypothesis.txt` for easy tuning.

## Files

- **Generator module:** `fitz_ai/retrieval/hyde/`
- **Prompt template:** `fitz_ai/retrieval/hyde/prompts/hypothesis.txt`
- **Integration:** `fitz_ai/engines/fitz_rag/retrieval/steps/strategies/semantic.py`
- **Wiring:** `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py`

## Benefits

| Without HyDE | With HyDE |
|--------------|-----------|
| Abstract queries match poorly | Hypothetical docs bridge semantic gap |
| "Sustainability approach" misses concrete EV docs | Hypotheses contain "electric", "emissions", "battery" |
| User must phrase queries like documents | Natural conceptual questions work |
| Lower recall on abstract queries | Higher recall across query styles |

## Example

**Query:** "What's TechCorp's approach to sustainability?"

**Generated hypotheses:**
1. "TechCorp's sustainability strategy centers on transitioning consumers to electric vehicles, reducing dependence on fossil fuels while maintaining competitive pricing and range."
2. "The company addresses environmental concerns through its battery recycling program and partnership with renewable energy providers for factory operations."
3. "TechCorp's commitment to sustainability includes an 8-year battery warranty, reducing waste by ensuring long product lifecycles."

**Result:** Documents about EV features, battery programs, and environmental initiatives are now retrieved, even though they don't explicitly discuss "sustainability approach".

## Performance

- One additional LLM call per query (fast tier, ~200-500ms)
- 3 additional embedding calls (one per hypothesis)
- 3 additional search calls (one per hypothesis)
- RRF merge is fast (in-memory)

Typical overhead: ~500ms for hypothesis generation + 3x embedding/search. Worth it for abstract queries.

## Dependencies

- Requires chat client configured (fast tier used)
- No additional packages beyond core fitz-ai
