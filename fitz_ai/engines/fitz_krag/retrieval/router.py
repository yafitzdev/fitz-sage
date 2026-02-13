# fitz_ai/engines/fitz_krag/retrieval/router.py
"""
Retrieval router — dispatches queries to available strategies and merges results.
"""

from __future__ import annotations

import logging
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
    ):
        self._code_strategy = code_strategy
        self._chunk_strategy = chunk_strategy
        self._section_strategy = section_strategy
        self._table_strategy = table_strategy
        self._config = config
        self._chat_factory = chat_factory
        self._agentic_strategy = agentic_strategy
        self._keyword_matcher: Any = None  # Set by engine for vocabulary filtering

    def retrieve(
        self,
        query: str,
        analysis: "QueryAnalysis | None" = None,
        detection: "Any | None" = None,
        progress: Callable[[str], None] | None = None,
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
            if hasattr(detection, "comparison_entities") and detection.comparison_entities:
                # Generate entity-focused queries via LLM instead of naive append
                comparison_queries = self._generate_comparison_queries(
                    query, detection.comparison_entities
                )
                for cq in comparison_queries:
                    tagged_queries.append((cq, None))

        # Multi-query expansion for long queries
        if (
            self._config.enable_multi_query
            and self._chat_factory
            and len(query) >= self._config.multi_query_min_length
        ):
            expanded = self._expand_query(query)
            for eq in expanded:
                tagged_queries.append((eq, None))

        for q, temporal_tag in tagged_queries:
            batch: list[Address] = []

            # Run code strategy (skip if weight below threshold)
            if not weights or weights.get("code", 1.0) > 0.05:
                try:
                    code_addresses = self._code_strategy.retrieve(q, limit)
                    batch.extend(code_addresses)
                except Exception as e:
                    logger.warning(f"Code strategy failed for query '{q[:50]}': {e}")

            # Run section strategy if available and weighted
            if self._section_strategy and (not weights or weights.get("section", 1.0) > 0.05):
                try:
                    section_addresses = self._section_strategy.retrieve(q, limit)
                    batch.extend(section_addresses)
                except Exception as e:
                    logger.warning(f"Section strategy failed for query '{q[:50]}': {e}")

            # Run table strategy if available and weighted
            if self._table_strategy and (not weights or weights.get("table", 1.0) > 0.05):
                try:
                    table_addresses = self._table_strategy.retrieve(q, limit)
                    batch.extend(table_addresses)
                except Exception as e:
                    logger.warning(f"Table strategy failed for query '{q[:50]}': {e}")

            # Tag addresses with temporal reference if this query is temporal
            if temporal_tag:
                batch = self._tag_temporal(batch, temporal_tag)

            all_addresses.extend(batch)

        # Agentic search for unindexed files
        if self._agentic_strategy:
            try:
                _progress = progress or (lambda _: None)
                _progress("Agentic search: scanning unindexed files...")
                agentic_addresses = self._agentic_strategy.retrieve(query, limit)
                if agentic_addresses:
                    # Summarize what was found
                    paths = set()
                    for a in agentic_addresses:
                        dp = a.metadata.get("disk_path")
                        if dp:
                            parts = dp.replace("\\", "/").split("/")
                            paths.add(parts[-1] if parts else dp)
                    files_str = ", ".join(sorted(paths)[:5])
                    _progress(f"Agentic search: found {len(agentic_addresses)} results from {len(paths)} files ({files_str})")
                else:
                    _progress("Agentic search: no matching files found")
                all_addresses.extend(agentic_addresses)
            except Exception as e:
                logger.warning(f"Agentic strategy failed: {e}")

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

    def _generate_comparison_queries(self, query: str, entities: list[str]) -> list[str]:
        """Generate entity-focused search queries for comparison via LLM."""
        if not self._chat_factory:
            # Fallback to naive approach when no LLM available
            return [f"{query} {entity}" for entity in entities]

        try:
            import json

            chat = self._chat_factory("fast")
            entities_str = ", ".join(entities)
            prompt = (
                "This is a comparison query. Generate focused search queries for "
                "each entity being compared.\n\n"
                f"Query: {query}\n"
                f"Entities: {entities_str}\n\n"
                "Instructions:\n"
                "1. Generate 2-3 search queries for EACH entity\n"
                "2. Generate 1 query that includes both entities together\n\n"
                'Return ONLY a JSON array of strings: ["query1", "query2", ...]'
            )
            response = chat.chat([{"role": "user", "content": prompt}])
            text = response.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    return [
                        str(q)
                        for q in parsed[: len(entities) * 3 + 1]
                        if isinstance(q, str) and q.strip()
                    ]
        except Exception as e:
            logger.warning(f"Comparison query expansion failed: {e}")

        # Fallback to naive approach
        return [f"{query} {entity}" for entity in entities]

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
