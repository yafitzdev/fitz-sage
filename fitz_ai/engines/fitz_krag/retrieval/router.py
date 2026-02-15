# fitz_ai/engines/fitz_krag/retrieval/router.py
"""
Retrieval router — dispatches queries to available strategies and merges results.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from fitz_ai.engines.fitz_krag.types import Address

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.query_analyzer import QueryAnalysis
    from fitz_ai.engines.fitz_krag.retrieval.strategies.chunk_fallback import (
        ChunkFallbackStrategy,
    )
    from fitz_ai.engines.fitz_krag.retrieval.strategies.code_search import (
        CodeSearchStrategy,
    )
    from fitz_ai.engines.fitz_krag.retrieval.strategies.section_search import (
        SectionSearchStrategy,
    )
    from fitz_ai.engines.fitz_krag.retrieval.strategies.table_search import (
        TableSearchStrategy,
    )

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """Routes queries to available strategies, merges results."""

    def __init__(
        self,
        code_strategy: "CodeSearchStrategy",
        chunk_strategy: "ChunkFallbackStrategy | None",
        config: "FitzKragConfig",
        section_strategy: "SectionSearchStrategy | None" = None,
        table_strategy: "TableSearchStrategy | None" = None,
        chat_factory: Any = None,
        agentic_strategy: Any = None,
        embedder: Any = None,
        hyde_generator: Any = None,
    ):
        self._code_strategy = code_strategy
        self._chunk_strategy = chunk_strategy
        self._section_strategy = section_strategy
        self._table_strategy = table_strategy
        self._config = config
        self._chat_factory = chat_factory
        self._agentic_strategy = agentic_strategy
        self._embedder = embedder
        self._hyde_generator = hyde_generator
        self._keyword_matcher: Any = None  # Set by engine for vocabulary filtering

    @staticmethod
    def _should_run_hyde(
        analysis: "QueryAnalysis | None", detection: "Any | None"
    ) -> bool:
        """Decide whether HyDE is likely to improve retrieval for this query.

        HyDE helps abstract/conceptual queries where user words don't match
        document vocabulary. It adds no value for direct code lookups, data/SQL
        queries, or high-confidence queries that already match well.
        """
        if not analysis:
            return True  # No signal — run HyDE to be safe

        # Complex intent detected — always run HyDE
        if detection and (
            getattr(detection, "has_comparison_intent", False)
            or getattr(detection, "has_temporal_intent", False)
            or getattr(detection, "has_aggregation_intent", False)
        ):
            return True

        # Code queries: keyword/AST search is primary, HyDE prose doesn't help
        if analysis.primary_type.value == "code" and analysis.confidence >= 0.7:
            return False

        # Data queries: table search uses SQL, not vector similarity
        if analysis.primary_type.value == "data":
            return False

        # Very high confidence: query already matches document vocabulary well
        if analysis.confidence >= 0.9:
            return False

        return True

    @staticmethod
    def _should_run_multi_query(
        analysis: "QueryAnalysis | None",
    ) -> bool:
        """Decide whether multi-query expansion adds value."""
        if not analysis:
            return True

        # Code/data queries don't benefit from prose decomposition
        if analysis.primary_type.value in ("code", "data"):
            return False

        # High confidence means the query is already focused
        if analysis.confidence >= 0.8:
            return False

        return True

    @staticmethod
    def _should_run_agentic(
        analysis: "QueryAnalysis | None",
    ) -> bool:
        """Decide whether agentic file scanning adds value.

        Agentic search scans unindexed files — most valuable for code queries
        or low-confidence queries where indexed content may be insufficient.
        For clear, high-confidence queries, indexed content suffices.
        """
        if not analysis:
            return True

        # Data queries: agentic scans files, not tables
        if analysis.primary_type.value == "data":
            return False

        # High-confidence queries: indexed strategies should find what's needed.
        # Only code queries at moderate confidence benefit (unindexed source files).
        if analysis.confidence >= 0.85 and analysis.primary_type.value != "code":
            return False

        return True

    def retrieve(
        self,
        query: str,
        analysis: "QueryAnalysis | None" = None,
        detection: "Any | None" = None,
        rewrite_result: "Any | None" = None,
        progress: Callable[[str], None] | None = None,
        precomputed_query_vectors: dict[str, list[float]] | None = None,
    ) -> list[Address]:
        """
        Retrieve addresses using strategy weights from query analysis.

        When analysis is provided, strategies with near-zero weight are skipped
        and results are ranked using CrossStrategyRanker. Without analysis,
        all strategies run equally (backward compatible).

        When detection is provided (DetectionSummary), it enhances retrieval:
        - Query expansion: additional retrievals with detection.query_variations
        - Comparison: search both detection.comparison_entities
        - Fetch multiplier: increase limit by detection.fetch_multiplier
        """
        from fitz_ai.engines.fitz_krag.retrieval.ranker import CrossStrategyRanker

        limit = self._config.top_addresses

        # Apply fetch multiplier from detection
        if detection and hasattr(detection, "fetch_multiplier"):
            limit = limit * detection.fetch_multiplier

        weights = analysis.strategy_weights if analysis else None
        all_addresses: list[Address] = []

        # Collect queries to run (original + detection expansions + multi-query)
        # Each entry is (query_text, tag) where tag is used for temporal metadata
        tagged_queries: list[tuple[str, str | None]] = [(query, None)]
        if detection:
            if hasattr(detection, "query_variations") and detection.query_variations:
                # Tag temporal variations with their reference text
                temporal_refs = self._extract_temporal_refs(detection)
                for i, variation in enumerate(detection.query_variations):
                    tag = temporal_refs[i] if i < len(temporal_refs) else None
                    tagged_queries.append((variation, tag))
            if hasattr(detection, "comparison_queries") and detection.comparison_queries:
                # Use comparison queries already generated by ComparisonModule
                for cq in detection.comparison_queries:
                    tagged_queries.append((cq, None))
            elif hasattr(detection, "comparison_entities") and detection.comparison_entities:
                # Fallback: naive entity expansion (no LLM call)
                for entity in detection.comparison_entities:
                    tagged_queries.append((f"{query} {entity}", None))

        # Multi-query expansion: prefer rewriter's decomposed queries over separate LLM call
        rewrite_variations = (
            rewrite_result.all_query_variations if rewrite_result else []
        )
        if len(rewrite_variations) > 1:
            # Rewriter already decomposed — use its variations (skip [0] = original)
            for var in rewrite_variations[1:]:
                tagged_queries.append((var, None))
        elif (
            self._config.enable_multi_query
            and self._chat_factory
            and len(query) >= self._config.multi_query_min_length
            and self._should_run_multi_query(analysis)
        ):
            # Fallback: only when rewriter is disabled or didn't decompose
            expanded = self._expand_query(query)
            for eq in expanded:
                tagged_queries.append((eq, None))

        # Pre-compute embeddings and HyDE vectors once for all queries
        import time as _time

        unique_queries = list(dict.fromkeys(q for q, _ in tagged_queries))
        query_vectors: dict[str, list[float]] = {}
        # Use pre-computed vectors from engine (overlapped with analysis+detection)
        if precomputed_query_vectors:
            query_vectors.update(precomputed_query_vectors)
        # Embed any queries not already pre-computed (expansions, variations)
        missing = [q for q in unique_queries if q not in query_vectors]
        if missing and self._embedder:
            try:
                _t0 = _time.perf_counter()
                vectors = self._embedder.embed_batch(missing)
                query_vectors.update(dict(zip(missing, vectors)))
                _embed_ms = (_time.perf_counter() - _t0) * 1000
            except Exception as e:
                _embed_ms = 0
                logger.warning(f"Batch embedding failed, strategies will embed individually: {e}")
        else:
            _embed_ms = 0

        run_hyde = self._should_run_hyde(analysis, detection)
        run_agentic = self._should_run_agentic(analysis)
        logger.debug(
            f"Retrieval gates: hyde={run_hyde}, agentic={run_agentic}, "
            f"type={analysis.primary_type.value if analysis else 'none'}, "
            f"conf={analysis.confidence if analysis else 0:.2f}, "
            f"queries={len(unique_queries)}, embed={_embed_ms:.0f}ms"
        )

        hyde_vectors_map: dict[str, list[list[float]]] = {}
        if self._hyde_generator and self._embedder and run_hyde:
            all_hypotheses: list[str] = []
            hyp_ranges: list[tuple[str, int, int]] = []  # (query_text, start_idx, count)
            for q_text in unique_queries:
                try:
                    hyps = self._hyde_generator.generate(q_text)
                    hyp_ranges.append((q_text, len(all_hypotheses), len(hyps)))
                    all_hypotheses.extend(hyps)
                except Exception:
                    hyp_ranges.append((q_text, 0, 0))
            if all_hypotheses:
                try:
                    hyp_vecs = self._embedder.embed_batch(all_hypotheses)
                    for q_text, start, count in hyp_ranges:
                        if count > 0:
                            hyde_vectors_map[q_text] = hyp_vecs[start : start + count]
                except Exception as e:
                    logger.warning(f"Batch HyDE embedding failed: {e}")

        # Submit all strategy calls concurrently
        _t_strategies = _time.perf_counter()
        _progress = progress or (lambda _: None)
        pool = ThreadPoolExecutor(max_workers=4)
        futures: list[tuple[Future, str | None]] = []  # (future, temporal_tag)
        try:
            for q, temporal_tag in tagged_queries:
                qv = query_vectors.get(q)
                # Empty list signals "HyDE intentionally skipped" to strategies
                # (vs None = "not pre-computed, generate your own")
                hv = hyde_vectors_map.get(q) if run_hyde else []

                if not weights or weights.get("code", 1.0) > 0.05:
                    fut = pool.submit(
                        self._run_strategy, self._code_strategy, q, limit, detection, qv, hv
                    )
                    futures.append((fut, temporal_tag))

                if self._section_strategy and (not weights or weights.get("section", 1.0) > 0.05):
                    fut = pool.submit(
                        self._run_strategy, self._section_strategy, q, limit, detection, qv, hv
                    )
                    futures.append((fut, temporal_tag))

                if self._table_strategy and (not weights or weights.get("table", 1.0) > 0.05):
                    fut = pool.submit(
                        self._run_strategy_table, self._table_strategy, q, limit, detection, qv
                    )
                    futures.append((fut, temporal_tag))

            # Submit agentic search in the same pool
            agentic_future: Future | None = None
            if self._agentic_strategy and run_agentic:
                _progress("Agentic search: scanning unindexed files...")
                agentic_future = pool.submit(self._run_agentic, query, limit, _progress)

            # Collect strategy results
            for fut, temporal_tag in futures:
                try:
                    batch = fut.result(timeout=60)
                    if temporal_tag:
                        batch = self._tag_temporal(batch, temporal_tag)
                    all_addresses.extend(batch)
                except Exception as e:
                    logger.warning(f"Strategy failed: {e}")

            # Collect agentic results
            if agentic_future:
                try:
                    agentic_addresses = agentic_future.result(timeout=60)
                    all_addresses.extend(agentic_addresses)
                except Exception as e:
                    logger.warning(f"Agentic strategy failed: {e}")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        _strategies_ms = (_time.perf_counter() - _t_strategies) * 1000
        logger.debug(f"Retrieval breakdown: strategies={_strategies_ms:.0f}ms")

        # Chunk fallback when other results are insufficient
        if (
            self._chunk_strategy
            and self._config.fallback_to_chunks
            and (not weights or weights.get("chunk", 1.0) > 0.05)
            and len(all_addresses) < self._config.top_addresses // 2
        ):
            try:
                chunk_limit = self._config.top_addresses - len(all_addresses)
                chunk_addresses = self._chunk_strategy.retrieve(query, chunk_limit)
                all_addresses.extend(chunk_addresses)
            except Exception as e:
                logger.warning(f"Chunk fallback failed: {e}")

        # Deduplicate
        deduped = self._deduplicate(all_addresses)

        # Keyword vocabulary boost
        if self._keyword_matcher:
            deduped = self._apply_keyword_boost(query, deduped)

        # Rank using analysis if available
        if analysis:
            ranker = CrossStrategyRanker()
            ranked = ranker.rank(deduped, analysis)
            result = ranked[: self._config.top_addresses]
        else:
            # Fallback: sort by score
            deduped.sort(key=lambda a: a.score, reverse=True)
            result = deduped[: self._config.top_addresses]

        # Filter out addresses below minimum relevance threshold
        min_score = getattr(self._config, "min_relevance_score", 0)
        if isinstance(min_score, (int, float)) and min_score > 0:
            filtered = [a for a in result if a.score >= min_score]
            if len(filtered) < len(result):
                logger.debug(
                    f"Relevance filter: {len(result)} -> {len(filtered)} addresses "
                    f"(min_score={min_score})"
                )
            result = filtered

        return result

    def _run_strategy(
        self,
        strategy: Any,
        query: str,
        limit: int,
        detection: Any,
        query_vector: list[float] | None,
        hyde_vectors: list[list[float]] | None,
    ) -> list[Address]:
        """Run a single strategy with pre-computed vectors."""
        try:
            return strategy.retrieve(
                query, limit, detection=detection,
                query_vector=query_vector, hyde_vectors=hyde_vectors,
            )
        except Exception as e:
            logger.warning(f"{type(strategy).__name__} failed for '{query[:50]}': {e}")
            return []

    def _run_strategy_table(
        self,
        strategy: Any,
        query: str,
        limit: int,
        detection: Any,
        query_vector: list[float] | None,
    ) -> list[Address]:
        """Run table strategy with pre-computed query vector (no HyDE)."""
        try:
            return strategy.retrieve(
                query, limit, detection=detection,
                query_vector=query_vector,
            )
        except Exception as e:
            logger.warning(f"TableSearchStrategy failed for '{query[:50]}': {e}")
            return []

    def _run_agentic(
        self,
        query: str,
        limit: int,
        progress: Callable[[str], None],
    ) -> list[Address]:
        """Run agentic search with progress reporting."""
        try:
            agentic_addresses = self._agentic_strategy.retrieve(query, limit)
            if agentic_addresses:
                paths = set()
                for a in agentic_addresses:
                    dp = a.metadata.get("disk_path")
                    if dp:
                        parts = dp.replace("\\", "/").split("/")
                        paths.add(parts[-1] if parts else dp)
                files_str = ", ".join(sorted(paths)[:5])
                progress(
                    f"Agentic search: found {len(agentic_addresses)} results "
                    f"from {len(paths)} files ({files_str})"
                )
            else:
                progress("Agentic search: no matching files found")
            return agentic_addresses
        except Exception as e:
            logger.warning(f"Agentic strategy failed: {e}")
            return []

    def _extract_temporal_refs(self, detection: Any) -> list[str]:
        """Extract temporal reference texts from detection metadata."""
        try:
            temporal = getattr(detection, "temporal", None)
            if not temporal:
                return []
            refs = temporal.metadata.get("references", [])
            result = []
            for r in refs:
                if isinstance(r, dict):
                    result.append(r.get("text", ""))
                elif isinstance(r, str):
                    result.append(r)
            return result
        except Exception:
            return []

    def _tag_temporal(self, addresses: list[Address], ref_text: str) -> list[Address]:
        """Tag addresses with the temporal reference that produced them."""
        tagged: list[Address] = []
        for addr in addresses:
            meta = dict(addr.metadata)
            existing = meta.get("temporal_refs", [])
            if ref_text not in existing:
                meta["temporal_refs"] = existing + [ref_text]
            tagged.append(
                Address(
                    kind=addr.kind,
                    source_id=addr.source_id,
                    location=addr.location,
                    summary=addr.summary,
                    score=addr.score,
                    metadata=meta,
                )
            )
        return tagged

    def _expand_query(self, query: str) -> list[str]:
        """Decompose a long query into focused sub-queries via LLM."""
        try:
            chat = self._chat_factory("fast")
            prompt = (
                "Break this complex query into 2-3 simpler, focused search queries.\n"
                "Return ONLY a JSON array of strings.\n\n"
                f"Query: {query}\n\n"
                'Output: ["query1", "query2", ...]'
            )
            import json

            response = chat.chat([{"role": "user", "content": prompt}])
            text = response.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    return [str(q) for q in parsed[:3] if isinstance(q, str) and q.strip()]
        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}")
        return []

    def _apply_keyword_boost(self, query: str, addresses: list[Address]) -> list[Address]:
        """Boost addresses matching vocabulary keywords found in the query."""
        try:
            matched_keywords = self._keyword_matcher.find_in_query(query)
            if not matched_keywords:
                return addresses

            # Collect all variations from matched keywords for text matching
            match_terms = set()
            for kw in matched_keywords:
                match_terms.add(kw.id.lower())
                for variation in getattr(kw, "match", []):
                    match_terms.add(variation.lower())

            boosted: list[Address] = []
            for addr in addresses:
                text = (addr.summary or "") + " " + (addr.location or "")
                text_lower = text.lower()
                boost = sum(1 for term in match_terms if term in text_lower)
                if boost > 0:
                    boosted.append(
                        Address(
                            kind=addr.kind,
                            source_id=addr.source_id,
                            location=addr.location,
                            summary=addr.summary,
                            score=addr.score + 0.1 * boost,
                            metadata=addr.metadata,
                        )
                    )
                else:
                    boosted.append(addr)
            return boosted
        except Exception as e:
            logger.warning(f"Keyword boost failed: {e}")
            return addresses

    def _deduplicate(self, addresses: list[Address]) -> list[Address]:
        """Deduplicate addresses by source_id+location, merging temporal tags."""
        seen: dict[tuple[str, str], int] = {}
        result: list[Address] = []

        for addr in addresses:
            key = (addr.source_id, addr.location)
            if key not in seen:
                seen[key] = len(result)
                result.append(addr)
            else:
                # Merge temporal_refs from duplicate into existing address
                new_refs = addr.metadata.get("temporal_refs", [])
                if new_refs:
                    idx = seen[key]
                    existing = result[idx]
                    existing_refs = existing.metadata.get("temporal_refs", [])
                    merged_refs = list(existing_refs)
                    for ref in new_refs:
                        if ref not in merged_refs:
                            merged_refs.append(ref)
                    if merged_refs != existing_refs:
                        meta = dict(existing.metadata)
                        meta["temporal_refs"] = merged_refs
                        result[idx] = Address(
                            kind=existing.kind,
                            source_id=existing.source_id,
                            location=existing.location,
                            summary=existing.summary,
                            score=max(existing.score, addr.score),
                            metadata=meta,
                        )

        return result
