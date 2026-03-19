# Test Corpus for Document Retrieval Eval

Place 3-5 test documents here. Good candidates:

- **1 dense technical PDF** (spec, whitepaper) — tests section extraction + deep retrieval
- **1 structured DOCX** (with clear headings, numbered sections) — tests heading hierarchy
- **1 document with tables** (pricing, comparison matrix, config tables) — tests table retrieval
- **1 mixed content doc** (text + tables + figures) — tests multi-type retrieval
- **1 short doc** (2-3 pages) — tests that simple docs aren't over-chunked

After adding documents, update `../doc_ground_truth.json` with queries and expected sources.

## Ground Truth Format

Each query specifies what should be retrieved:

```json
{
  "id": 1,
  "query": "What are the memory requirements?",
  "document": "technical_spec.pdf",
  "category": "section-lookup",
  "critical_sections": [
    {
      "heading": "Hardware Requirements",
      "page": 12,
      "keywords": ["RAM", "VRAM", "GPU"]
    }
  ],
  "relevant_sections": [
    {
      "heading": "System Overview",
      "page": 3
    }
  ]
}
```

- `critical_sections`: MUST be found (like critical files in code eval)
- `relevant_sections`: nice to have (like relevant files in code eval)
- `heading`: matched fuzzy (substring) against provenance title/metadata
- `page`: matched within +/-1 page tolerance
- `keywords`: at least 2/3 must appear in the retrieved excerpt
