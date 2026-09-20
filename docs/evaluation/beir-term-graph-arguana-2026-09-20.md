# Frozen ArguAna Term-Graph Ablation

Run date: 2026-09-20. Run ID: `1789904444-15b97a44`. Product revision:
`6248b2922cf0f8aa8210244904c467144ce2183c` (`v0.16.1`).

This run measures the SQLite corpus term graph that replaced generative query
expansion. It uses the frozen ArguAna portion of the existing semantic holdout,
with the same query IDs and lexical-overlap strata used by the earlier run.

## Method

- 8,674 source documents and 120 frozen queries: 40 low-overlap, 40 medium,
  and 40 high.
- Manifest: `beir-semantic-vocabulary-holdout-v1`, SHA-256
  `606baf0b2be00b78aecc737a9d2f550521254be5e2845232029a69b824b1df78`.
- Source-only index shared by four fresh-process variants.
- Identical query order, candidate budgets, evidence compilation, and Pyrrho
  integration across variants.
- 2,000-sample deterministic paired percentile bootstrap, seed `20260730`.
- Python 3.10.11 on Windows. The reranker used ONNX Runtime 1.23.2 with the
  available CPU/Azure providers.
- All query checkpoints were created by this run. The paired measurement-
  integrity gate passed with no failures.

Command:

```powershell
.venv\Scripts\python.exe -m benchmarks.fitz_bench.beir_ablation `
  --dataset arguana `
  --offline `
  --query-manifest benchmarks\fixtures\beir_semantic_holdout_v1.json
```

The worktree was dirty only because the benchmark harness was being corrected
to default to the package's source-only retrieval contract. The measured
product code was the tagged v0.16.1 revision above.

## Results

Plain whole-document BM25 scored 0.4652 nDCG@10 and 0.9250 Recall@50.

| Variant | Recall nDCG@10 | Final nDCG@10 | Delivered nDCG@10 | Recall@50 | Mean latency | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Literal | 0.4509 | 0.4413 | 0.2929 | 0.9000 | 5.00s | 9.73s |
| Term graph | 0.4510 | 0.4413 | 0.3023 | 0.8917 | 5.17s | 10.69s |
| Reranker | 0.4509 | 0.4562 | 0.3357 | 0.9000 | 9.85s | 16.74s |
| Term graph + reranker | 0.4510 | 0.4563 | 0.3357 | 0.8917 | 9.34s | 16.48s |

Paired effects:

| Added component | Recall nDCG@10 | Final nDCG@10 | Delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|
| Term graph, no reranker | +0.0001 [-0.0090, +0.0090] | -0.0000 [-0.0090, +0.0094] | +0.0094 [-0.0158, +0.0357] | +0.17s [-0.07, +0.40] |
| Term graph, reranker on | +0.0001 [-0.0084, +0.0086] | +0.0001 [-0.0098, +0.0083] | +0.0000 [0.0000, 0.0000] | -0.50s [-1.05, -0.07] |
| Reranker, no term graph | +0.0000 [0.0000, 0.0000] | +0.0149 [-0.0504, +0.0799] | +0.0429 [-0.0355, +0.1255] | +4.85s [+4.29, +5.44] |
| Both versus literal | +0.0001 [-0.0088, +0.0092] | +0.0150 [-0.0526, +0.0808] | +0.0429 [-0.0409, +0.1257] | +4.35s [+3.95, +4.74] |

The term graph's measured internal work averaged 0.07 seconds without the
reranker and 0.05 seconds with it. The negative total-latency delta in the
reranked comparison is run noise or unattributed work; it is not evidence that
the graph makes queries faster.

## Lexical-Overlap Strata

| Stratum | Queries | Term-graph recall delta | Final delta, no reranker | Final delta, reranker on |
|---|---:|---:|---:|---:|
| Low | 40 | +0.0090 [-0.0026, +0.0274] | -0.0057 [-0.0259, +0.0087] | +0.0115 [0.0000, +0.0293] |
| Medium | 40 | -0.0080 [-0.0207, +0.0040] | -0.0080 [-0.0207, +0.0037] | +0.0000 [0.0000, 0.0000] |
| High | 40 | -0.0009 [-0.0196, +0.0143] | +0.0136 [-0.0003, +0.0335] | -0.0111 [-0.0334, 0.0000] |

The broad quality effects are inconclusive. ArguAna uses long argumentative
passages as queries and does not strongly exercise company-document
abbreviations, aliases, product/error clusters, or identifiers. The result
therefore supports a narrower conclusion: the safe source-backed graph did not
materially change quality on this frozen external proxy and its own runtime
cost was small. The internal hardened-boundary suite separately passed all
11 retrieval and delivery contracts that include application-shaped bridges.

## Quora Status

The official Quora archive was verified and its 522,931 documents were
projected. A fresh source-plus-term-graph build processed 5,277 files at about
22 files per second, projecting to roughly 6.5 hours before the 480 paired
query executions. The run was stopped and no Quora score is reported. This is
the documented extreme-file-count scan/index boundary, not a retrieval-quality
result. Its verified corpus and checkpoints remain reusable for a dedicated
long run.

The ignored machine-readable outputs are
`benchmarks/results/beir_ablation_latest.json` and
`benchmarks/results/beir_ablation_latest.md`.
