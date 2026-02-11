# fitz_ai/engines/fitz_rag/guardrails/plugins/insufficient_evidence.py
"""
Insufficient Evidence Constraint - Default guardrail for evidence coverage.

This constraint prevents the system from giving confident answers when
there is no relevant evidence in the retrieved chunks.

Priority order:
1. Embedding similarity (if embedder provided) - most reliable
2. Enriched metadata (entity/summary overlap)
3. Lexical overlap fallback

No LLM calls. Deterministic with tunable thresholds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..aspect_classifier import AspectClassifier, QueryAspect
from ..base import ConstraintResult

logger = get_logger(__name__)

# Type alias for embedder function
EmbedderFunc = Callable[[str], list[float]]

# Below this score, vectors are nearly orthogonal (no semantic relationship)
MIN_RELEVANCE_SCORE = 0.3
# Embedding similarity threshold for relevance
# Empirically tuned: unrelated content ~0.35-0.51, related content ~0.66-0.84
# Threshold of 0.6 provides good separation
MIN_EMBEDDING_SIMILARITY = 0.6


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# Common stopwords to ignore in overlap check
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "any",
        "both",
        "either",
        "neither",
        "much",
        "many",
    }
)


def _extract_words(text: str) -> set[str]:
    """Extract meaningful words (lowercase, no stopwords, min 3 chars)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _extract_query_entities(query: str) -> set[str]:
    """Extract potential entities from query (proper nouns, capitalized words, quoted terms)."""
    entities = set()

    # Quoted terms
    quoted = re.findall(r'"([^"]+)"', query)
    entities.update(q.lower() for q in quoted)

    # Capitalized words (potential proper nouns) - excluding sentence starters
    words = query.split()
    for i, word in enumerate(words):
        # Skip first word and common question starters
        if i > 0 and word[0].isupper() and word.lower() not in STOPWORDS:
            entities.add(word.lower())

    # Also extract meaningful words as potential topics
    entities.update(_extract_words(query))

    return entities


def _extract_specific_entities(query: str) -> tuple[set[str], set[str], set[str]]:
    """
    Extract SPECIFIC entities that must appear in context for relevance.

    These are proper nouns, product names, years, etc. that the query is specifically about.
    If query asks about "iPhone 16", context must mention "iPhone 16" (not Samsung Galaxy).

    Returns:
        Tuple of (all_entities, critical_entities, primary_entities).
        Critical entities (like years) MUST match; regular entities need ANY match.
        Primary entities are the main subject of the query — if missing, always abstain.
    """
    specific = set()
    critical = set()  # These MUST match (years, numbered qualifiers)
    primary = set()  # The main referent — missing = abstain, no similarity exception

    # Common question starters and auxiliaries to exclude
    question_words = {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "whom",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "will",
        "would",
        "can",
        "could",
        "should",
        "has",
        "have",
        "had",
        "been",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
    }

    # Action/generic words that are never primary referents.
    # These are query-intent words (what property is asked about) not subject entities
    # (what the query is about). "pricing" in "What is the pricing?" is an intent word;
    # "Tesla" in "What is Tesla's pricing?" is a subject entity.
    generic_words = {
        # Actions / verbs
        "compare",
        "explain",
        "describe",
        "list",
        "show",
        "tell",
        "give",
        "find",
        "get",
        "make",
        "use",
        "help",
        "work",
        "affect",
        "impact",
        "cause",
        "caused",
        "effect",
        "change",
        "improve",
        "reduce",
        "increase",
        "fix",
        "upgrade",
        "proceed",
        # Query-aspect words (what property is requested)
        "benefit",
        "risk",
        "cost",
        "price",
        "pricing",
        "rate",
        "rates",
        "value",
        "difference",
        "deadline",
        "budget",
        "revenue",
        "salary",
        "warranty",
        "coverage",
        "eligibility",
        "requirements",
        "prerequisites",
        "ingredients",
        "calorie",
        "calories",
        "mechanism",
        "dosage",
        "interest",
        "capacity",
        "specifications",
        "specs",
        "efficiency",
        "certification",
        "population",
        "headquarters",
        "address",
        # Adjectives / modifiers
        "best",
        "worst",
        "main",
        "key",
        "important",
        "current",
        "latest",
        "new",
        "old",
        "first",
        "last",
        "next",
        "previous",
        "average",
        "total",
        "number",
        "amount",
        "percent",
        "percentage",
        "exact",
        "minimum",
        "maximum",
        "target",
        "recommended",
        "hourly",
        "annual",
        "monthly",
        "weekly",
        "daily",
        # Format words
        "bulleted",
        "numbered",
        "formatted",
        "detailed",
        "summary",
        # Abstract nouns
        "example",
        "type",
        "kind",
        "way",
        "method",
        "process",
        "system",
        "part",
        "role",
        "result",
        "reason",
        "factor",
        "feature",
        "advantage",
        "disadvantage",
        "problem",
        "solution",
        "symptom",
        "treatment",
        "side",
        "long",
        "short",
        "term",
        "high",
        "low",
        "load",
    }

    # Quoted terms are always specific
    quoted = re.findall(r'"([^"]+)"', query)
    specific.update(q.lower() for q in quoted)
    # Quoted terms are primary — user explicitly highlighted them
    primary.update(q.lower() for q in quoted)

    # Years (4-digit numbers) - usually CRITICAL, must match exactly
    # Exception: forecasting queries ("will X be in 2026", "predict by 2030")
    # where trend/forecast data without the exact year is still useful (→ qualified not abstain)
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
    specific.update(years)

    q_lower = query.lower()
    _FORECAST_PATTERNS = (
        "will be in ",
        "by 20",
        "by 19",
        "in the next ",
        "forecast",
        "predict",
        "projection",
        "expected by",
        "estimated by",
        "outlook for",
    )
    is_forecast = any(p in q_lower for p in _FORECAST_PATTERNS)

    if years and not is_forecast:
        critical.update(years)

    # Numbered qualifiers like "type 2", "tier 1", "phase 3", "version 2.0"
    # These are CRITICAL - "type 2 diabetes" vs "type 1 diabetes" is a different entity
    numbered_qualifiers = re.findall(
        r"\b(type\s+\d+|tier\s+\d+|phase\s+\d+|version\s+[\d.]+|level\s+\d+|"
        r"class\s+\d+|grade\s+\d+|stage\s+\d+|gen\s+\d+|generation\s+\d+)\b",
        query.lower(),
    )
    specific.update(numbered_qualifiers)
    critical.update(numbered_qualifiers)

    # Product names, company names, etc. (multi-word capitalized sequences)
    # e.g., "iPhone 16", "World Series", "Microsoft", "Bitcoin"
    # Skip sequences at the very start of the query (likely question starters)
    cap_sequences = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z0-9][a-z0-9]*)*)", query)
    for seq in cap_sequences:
        seq_lower = seq.lower()
        # Skip if it's a question word or starts with one
        first_word = seq_lower.split()[0] if seq_lower else ""
        if first_word not in question_words and seq_lower not in question_words:
            # Skip if sequence is at the very beginning of query
            if not query.lower().startswith(seq_lower):
                specific.add(seq_lower)

    # Single capitalized words that aren't at sentence start (position > 0)
    # Skip ALL-CAPS words — they're emphasis markers (e.g., "What is the PRICING?"),
    # not proper nouns. Proper nouns use Title Case (e.g., "Tesla", "iPhone").
    words = query.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w]", "", word)
        if i > 0 and clean_word and clean_word[0].isupper():
            # ALL-CAPS = emphasis, not a proper noun
            if clean_word == clean_word.upper() and len(clean_word) > 1:
                continue
            clean_lower = clean_word.lower()
            if (
                clean_lower not in STOPWORDS
                and clean_lower not in question_words
                and len(clean_word) > 2
            ):
                specific.add(clean_lower)

    # Identify PRIMARY entities: proper nouns / named entities that are the subject of the query.
    # These are entities that if missing from context, mean the context is about something else.
    # e.g., "Tesla" in "What is Tesla's revenue?" — if context only has Ford data, abstain.
    #
    # Strategy: named entities (capitalized words/sequences) that aren't generic modifiers.
    for entity in specific:
        # Skip years and numbered qualifiers (already handled by critical)
        if re.match(r"^(19|20)\d{2}$", entity):
            continue
        if re.match(
            r"^(type|tier|phase|version|level|class|grade|stage|gen|generation)\s+", entity
        ):
            continue
        # Skip generic words that aren't proper nouns
        if entity in generic_words:
            continue
        # If the entity contains a capitalized word in the original query, it's a proper noun
        # Check against original query for case sensitivity
        for token in query.split():
            clean_token = re.sub(r"[^\w]", "", token)
            if clean_token.lower() == entity and clean_token and clean_token[0].isupper():
                primary.add(entity)
                break
        # Multi-word entities from cap_sequences are inherently proper nouns
        if " " in entity:
            primary.add(entity)

    return specific, critical, primary


# ── LLM-assisted primary entity extraction ────────────────────────────────────
# When heuristics produce an empty primary set, optionally ask an LLM to select
# the primary subject from deterministic candidates. The LLM cannot invent
# entities — it must choose from a closed set or answer NONE.

_PRIMARY_ENTITY_PROMPT = """Given this question, which of the following candidates is the PRIMARY SUBJECT being asked about?

Question: {query}

Candidates:
{candidates}

Rules:
- Pick the ONE candidate that is the main subject of the question
- If none are the primary subject, answer NONE
- Answer with ONLY the candidate text or NONE, nothing else"""

# Words that should never be accepted as primary entities even if LLM selects them.
# Mirrors generic_words in _extract_specific_entities plus additional LLM-specific rejections.
_LLM_PRIMARY_REJECT = frozenset(
    {
        # Actions / verbs
        "compare",
        "explain",
        "describe",
        "list",
        "show",
        "tell",
        "find",
        "get",
        "make",
        "use",
        "help",
        "work",
        "affect",
        "impact",
        "cause",
        "caused",
        "effect",
        "change",
        "improve",
        "reduce",
        "increase",
        "fix",
        "upgrade",
        "proceed",
        # Query-aspect words
        "benefit",
        "risk",
        "cost",
        "price",
        "pricing",
        "rate",
        "rates",
        "value",
        "difference",
        "deadline",
        "budget",
        "revenue",
        "salary",
        "warranty",
        "coverage",
        "eligibility",
        "requirements",
        "prerequisites",
        "ingredients",
        "calorie",
        "calories",
        "mechanism",
        "dosage",
        "interest",
        "capacity",
        "specifications",
        "specs",
        "efficiency",
        "certification",
        "population",
        "headquarters",
        "address",
        # Adjectives / modifiers
        "best",
        "worst",
        "main",
        "key",
        "important",
        "current",
        "latest",
        "new",
        "old",
        "first",
        "last",
        "next",
        "previous",
        "average",
        "total",
        "number",
        "amount",
        "percent",
        "percentage",
        "exact",
        "minimum",
        "maximum",
        "target",
        "recommended",
        "hourly",
        "annual",
        "monthly",
        "weekly",
        "daily",
        # Format words
        "bulleted",
        "numbered",
        "formatted",
        "detailed",
        "summary",
        # Abstract nouns
        "example",
        "type",
        "kind",
        "way",
        "method",
        "process",
        "system",
        "part",
        "role",
        "result",
        "reason",
        "factor",
        "feature",
        "advantage",
        "disadvantage",
        "problem",
        "solution",
        "symptom",
        "treatment",
        "side",
        "long",
        "short",
        "term",
        "high",
        "low",
        "load",
        # LLM-specific additional rejections
        "company",
        "product",
        "service",
        "customer",
        "user",
        "team",
        "project",
        "data",
        "information",
        "question",
        "answer",
    }
)


def _llm_rank_primary_entity(
    query: str,
    specific_entities: set[str],
    chat: Any,
) -> set[str]:
    """
    Ask LLM to select the primary subject from deterministic candidates.

    Returns a set with at most one entity, or empty set if LLM says NONE
    or validation rejects the choice.
    """
    if not chat or not specific_entities:
        return set()

    # Build candidate list from specific entities (deterministic, closed set)
    # Also add noun phrases from the query that aren't in specific_entities
    candidates = set(specific_entities)

    # Extract lowercase noun-like phrases (2-3 word sequences) as additional candidates
    words = query.lower().split()
    for i in range(len(words)):
        for length in (2, 3):
            if i + length <= len(words):
                phrase = " ".join(words[i : i + length])
                # Skip if starts with question word or stopword
                if words[i] not in STOPWORDS and len(words[i]) > 2:
                    candidates.add(phrase)

    # Remove obvious non-entities
    # For multi-word phrases, reject if ALL words are generic/reject words
    def _is_generic_phrase(phrase: str) -> bool:
        words_in_phrase = phrase.split()
        return all(w in _LLM_PRIMARY_REJECT or w in STOPWORDS for w in words_in_phrase)

    candidates = {
        c
        for c in candidates
        if c not in _LLM_PRIMARY_REJECT
        and not _is_generic_phrase(c)
        and not re.match(r"^(19|20)\d{2}$", c)
        and len(c) > 2
    }

    if not candidates:
        return set()

    candidates_str = "\n".join(f"- {c}" for c in sorted(candidates))
    prompt = _PRIMARY_ENTITY_PROMPT.format(query=query, candidates=candidates_str)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = chat.chat(messages).strip()

        # Validate: must be one of the candidates or NONE
        response_lower = response.lower().strip().strip('"').strip("'").strip("-").strip()

        if response_lower == "none" or not response_lower:
            return set()

        # Check if response matches any candidate (exact or close)
        for candidate in candidates:
            if candidate == response_lower or candidate in response_lower:
                # Final validation: reject abstract/generic terms
                if candidate in _LLM_PRIMARY_REJECT:
                    return set()
                return {candidate}

        # LLM returned something not in candidates — reject
        logger.debug(
            f"{PIPELINE} LLM primary entity '{response_lower}' not in candidates, rejected"
        )
        return set()

    except Exception as e:
        logger.warning(f"{PIPELINE} LLM primary entity extraction failed: {e}")
        return set()


def _context_mentions_entities(entities: set[str], context: str) -> bool:
    """Check if context mentions any of the specific entities."""
    if not entities:
        return True  # No specific entities to check

    context_lower = context.lower()
    for entity in entities:
        if entity in context_lower:
            return True

    return False


def _context_mentions_all_critical(critical: set[str], context: str) -> bool:
    """Check if context mentions ALL critical entities (years, numbered qualifiers)."""
    if not critical:
        return True  # No critical entities to check

    context_lower = context.lower()
    for entity in critical:
        if entity not in context_lower:
            return False  # Missing a critical entity

    return True


def _get_max_score(chunks: Sequence[Chunk]) -> float | None:
    """Get the highest vector_score from chunks, or None if no scores."""
    scores = []
    for chunk in chunks:
        score = chunk.metadata.get("vector_score")
        if score is not None:
            scores.append(float(score))
    return max(scores) if scores else None


def _has_entity_overlap(query: str, chunks: Sequence[Chunk]) -> bool:
    """Check if query entities appear in chunk entities (enriched metadata)."""
    query_entities = _extract_query_entities(query)
    if not query_entities:
        return True  # Can't determine, allow

    for chunk in chunks:
        chunk_entities = chunk.metadata.get("entities", [])
        if chunk_entities:
            # Extract entity names from enriched data
            chunk_entity_names = {
                e.get("name", "").lower()
                for e in chunk_entities
                if isinstance(e, dict) and e.get("name")
            }
            if query_entities & chunk_entity_names:
                return True

    return False


def _has_summary_overlap(query: str, chunks: Sequence[Chunk]) -> bool:
    """Check if query topics appear in chunk summaries (less noise than raw content)."""
    query_words = _extract_words(query)
    if not query_words:
        return True  # Can't determine, allow

    for chunk in chunks:
        summary = chunk.metadata.get("summary", "")
        if summary:
            summary_words = _extract_words(summary)
            # Require at least 1 matching word
            overlap = query_words & summary_words
            if overlap:
                return True

    return False


def _has_lexical_overlap(query: str, chunks: Sequence[Chunk]) -> bool:
    """Check if query shares any meaningful words with chunks (fallback)."""
    query_words = _extract_words(query)
    if not query_words:
        return True  # Can't determine overlap, allow

    for chunk in chunks:
        chunk_words = _extract_words(chunk.content)
        if query_words & chunk_words:  # Intersection
            return True

    return False


def _check_enriched_relevance(query: str, chunks: Sequence[Chunk]) -> tuple[bool, str]:
    """
    Check relevance using enriched metadata.

    Returns (is_relevant, method_used).
    """
    # Check if chunks have enrichment
    has_entities = any(chunk.metadata.get("entities") for chunk in chunks)
    has_summaries = any(chunk.metadata.get("summary") for chunk in chunks)

    if has_entities or has_summaries:
        # Use enriched data - stricter checks
        entity_match = _has_entity_overlap(query, chunks) if has_entities else False
        summary_match = _has_summary_overlap(query, chunks) if has_summaries else False

        if entity_match:
            return True, "entity_overlap"
        if summary_match:
            return True, "summary_overlap"

        # Enriched but no match - this is a reliable ABSTAIN signal
        return False, "no_enriched_match"

    # No enrichment - can't use this method
    return True, "no_enrichment"


@dataclass
class InsufficientEvidenceConstraint:
    """
    Constraint that prevents confident answers without relevant evidence.

    Priority order (deterministic, no LLM):
    1. No chunks = ABSTAIN
    2. Embedding similarity (if embedder provided) - most reliable
    3. Vector score from retrieval (if available)
    4. Enriched metadata (entity/summary overlap)
    5. Fallback: lexical overlap on raw content

    Attributes:
        embedder: Optional function to embed text (enables semantic similarity)
        enabled: Whether this constraint is active (default: True)
        min_score: Minimum vector score to consider relevant (default: 0.3)
        min_similarity: Minimum embedding similarity (default: 0.4)
    """

    embedder: EmbedderFunc | None = None
    chat: Any | None = None
    enabled: bool = True
    min_score: float = MIN_RELEVANCE_SCORE
    min_similarity: float = MIN_EMBEDDING_SIMILARITY
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _aspect_classifier: AspectClassifier = field(
        default_factory=AspectClassifier, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        return "insufficient_evidence"

    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding with caching."""
        if not self.embedder:
            return None

        cache_key = hash(text[:200])
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            embedding = self.embedder(text)
            self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"{PIPELINE} InsufficientEvidence: embedding failed: {e}")
            return None

    def _check_embedding_relevance(
        self, query: str, chunks: Sequence[Chunk]
    ) -> tuple[bool, float, str, dict]:
        """
        Check relevance using embedding similarity + entity matching.

        Returns (is_relevant, max_similarity, reason, diagnostics).
        """
        query_emb = self._get_embedding(query)
        if not query_emb:
            return True, 0.0, "no_embedder", {}  # Can't check, allow

        # Extract specific, critical, and primary entities from query
        specific_entities, critical_entities, primary_entities = _extract_specific_entities(query)

        # LLM fallback: if heuristic found no primary entities, ask LLM to rank candidates
        if not primary_entities and self.chat and specific_entities:
            llm_primary = _llm_rank_primary_entity(query, specific_entities, self.chat)
            if llm_primary:
                primary_entities = llm_primary
                # Also ensure these are in specific_entities
                specific_entities |= llm_primary
                logger.debug(f"{PIPELINE} LLM-assisted primary entity: {llm_primary}")

        max_sim = 0.0
        entity_match_found = False
        critical_match_found = False
        primary_match_found = False

        for chunk in chunks:
            chunk_emb = self._get_embedding(chunk.content)
            if chunk_emb:
                sim = _cosine_similarity(query_emb, chunk_emb)
                max_sim = max(max_sim, sim)

            # Check entity matching
            if specific_entities and _context_mentions_entities(specific_entities, chunk.content):
                entity_match_found = True

            # Check primary entity matching
            if primary_entities and _context_mentions_entities(primary_entities, chunk.content):
                primary_match_found = True

            # Check critical entity matching (years, numbered qualifiers must ALL match)
            if critical_entities and _context_mentions_all_critical(
                critical_entities, chunk.content
            ):
                critical_match_found = True

        # Build diagnostics dict for classifier feature extraction
        diag = {
            "ie_entity_match_found": entity_match_found,
            "ie_primary_match_found": primary_match_found,
            "ie_critical_match_found": critical_match_found,
            "ie_query_aspect": None,  # Set below if aspect classifier runs
            "ie_summary_overlap": None,  # Set below if summary check runs
            "ie_has_matching_aspect": None,  # Set below if aspect check runs
            "ie_has_conflicting_aspect": None,  # Set below if aspect check runs
        }

        # Decision logic:
        # 1. If similarity is very low (<0.45), definitely irrelevant
        if max_sim < 0.45:
            return False, max_sim, "low_similarity", diag

        # 2. CRITICAL entities (years, "type 2", etc.) MUST match - no bypass
        # "2024 World Series" query MUST have 2024 in context, even if similarity is high
        if critical_entities and not critical_match_found:
            return False, max_sim, f"missing_critical:{list(critical_entities)[:3]}", diag

        # 3. Check PRIMARY entity matching: if the main subject is missing, abstain.
        # This catches cases like: "Tesla revenue?" + Ford data, "Tokyo population?" + Osaka data.
        # Primary entity missing = context is about something else entirely. No similarity exception.
        if primary_entities and not primary_match_found and max_sim < 0.85:
            return False, max_sim, f"missing_primary:{list(primary_entities)[:3]}", diag

        # 4. Check SECONDARY entity matching for same-topic-wrong-entity detection
        # Non-primary entities (generic terms, lowercase domain words) still get the
        # similarity exception — high similarity with secondary mismatch = qualified, not abstain.
        if specific_entities and not entity_match_found and max_sim < 0.85:
            return False, max_sim, f"missing_entity:{list(specific_entities)[:3]}", diag

        # 5. ENRICHMENT BOOST: Use summaries for "same topic, wrong aspect" detection
        # Only activates when chunks are enriched (have summaries)
        # Only applies in ambiguous similarity range where embeddings aren't decisive
        if 0.45 <= max_sim < 0.70:
            has_summaries = any(chunk.metadata.get("summary") for chunk in chunks)
            if has_summaries:
                summary_match = _has_summary_overlap(query, chunks)
                diag["ie_summary_overlap"] = summary_match
                if not summary_match:
                    return False, max_sim, "no_summary_overlap", diag

        # 6. ASPECT COMPATIBILITY CHECK: Entity matches but aspect might differ
        # e.g., Query "What causes Alzheimer's?" vs Chunk "Alzheimer's symptoms include..."
        # Also check when similarity is moderate (0.5-0.85) even without entity match
        # This catches cases like "aspirin mechanism" where entity is lowercase
        query_aspect = self._aspect_classifier.classify_query(query)
        diag["ie_query_aspect"] = query_aspect.value

        # Run aspect check when:
        # 1. Entity match found AND similarity in ambiguous range (0.5-0.78), OR
        # 2. Moderate similarity (0.5-0.78) with no entities to verify (possible related content)
        # Skip aspect check for high similarity (>0.78) - embeddings are reliable there
        should_check_aspect = (entity_match_found and 0.5 <= max_sim < 0.78) or (
            not specific_entities and 0.5 <= max_sim < 0.78
        )

        logger.debug(
            f"{PIPELINE} InsufficientEvidence: aspect check - "
            f"entity_match={entity_match_found}, query_aspect={query_aspect.value}, "
            f"should_check={should_check_aspect}, sim={max_sim:.3f}"
        )
        if should_check_aspect and query_aspect != QueryAspect.GENERAL:
            # ASPECT MISMATCH DETECTION:
            # Only abstain if ALL chunks with specific aspects CONFLICT with query aspect
            # - GENERAL chunks are neutral (don't trigger abstention by themselves)
            # - If any chunk has matching or compatible aspect, allow
            # - Only abstain if there are conflicting aspects AND no matching aspects
            has_matching_aspect = False
            has_conflicting_aspect = False
            conflicting_aspects = []

            for chunk in chunks:
                chunk_aspects = self._aspect_classifier.extract_chunk_aspects(chunk.content)

                # Skip GENERAL chunks - they're neutral
                if chunk_aspects == [QueryAspect.GENERAL]:
                    continue

                # Check if this chunk has the matching aspect
                if query_aspect in chunk_aspects:
                    has_matching_aspect = True
                    break

                # This chunk has specific aspects that don't include the query aspect
                has_conflicting_aspect = True
                conflicting_aspects.extend([a.value for a in chunk_aspects])

            diag["ie_has_matching_aspect"] = has_matching_aspect
            diag["ie_has_conflicting_aspect"] = has_conflicting_aspect

            # Only abstain if we found conflicting aspects but NO matching aspects
            if has_conflicting_aspect and not has_matching_aspect:
                return (
                    False,
                    max_sim,
                    f"aspect_mismatch:query asks {query_aspect.value}, chunks have {list(set(conflicting_aspects))}",
                    diag,
                )

        # 7. If we have entity match OR no specific entities OR very high similarity, allow
        return True, max_sim, "relevant", diag

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check if there is relevant evidence to answer the query.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies if no chunks or evidence is off-topic
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Rule 1: No chunks at all
        if not chunks:
            logger.info(f"{PIPELINE} InsufficientEvidenceConstraint: no chunks retrieved")
            return ConstraintResult.deny(
                reason="No evidence retrieved",
                signal="abstain",
                evidence_count=0,
            )

        # Rule 2: Check embedding similarity + entity matching (most reliable, if available)
        if self.embedder:
            is_relevant, max_sim, reason, ie_diag = self._check_embedding_relevance(query, chunks)
            if not is_relevant:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: {reason} "
                    f"(similarity={max_sim:.3f}) -> ABSTAIN"
                )
                # Primary referent missing = context is about something else entirely.
                # Always abstain — no similarity exception.
                if "missing_primary" in reason:
                    return ConstraintResult.deny(
                        reason=f"Context lacks primary subject: {reason} (similarity={max_sim:.3f})",
                        signal="abstain",
                        evidence_count=len(chunks),
                        max_similarity=max_sim,
                        detection_reason=reason,
                        **ie_diag,
                    )

                # Secondary entity missing with moderate-high similarity = related but incomplete.
                # Return qualified — context IS topically related, just not specific enough.
                if max_sim >= 0.57 and "missing_entity" in reason:
                    return ConstraintResult.deny(
                        reason=f"Context is related but lacks specific information: {reason} (similarity={max_sim:.3f})",
                        signal="qualified",
                        evidence_count=len(chunks),
                        max_similarity=max_sim,
                        detection_reason=reason,
                        **ie_diag,
                    )

                # Low similarity or critical issues -> abstain
                return ConstraintResult.deny(
                    reason=f"Context not relevant: {reason} (similarity={max_sim:.3f})",
                    signal="abstain",
                    evidence_count=len(chunks),
                    max_similarity=max_sim,
                    detection_reason=reason,
                    **ie_diag,
                )
            logger.debug(
                f"{PIPELINE} InsufficientEvidenceConstraint: {reason} (similarity={max_sim:.3f})"
            )
            return ConstraintResult.allow(
                evidence_count=len(chunks),
                max_similarity=max_sim,
                **ie_diag,
            )

        # Rule 3: Check vector_score if available (from retrieval)
        max_score = _get_max_score(chunks)
        if max_score is not None:
            if max_score < self.min_score:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: score {max_score:.3f} "
                    f"< {self.min_score} -> ABSTAIN"
                )
                return ConstraintResult.deny(
                    reason=f"Retrieved content not relevant (score={max_score:.3f})",
                    signal="abstain",
                    evidence_count=len(chunks),
                    max_score=max_score,
                )
            return ConstraintResult.allow()

        # Rule 4: Check enriched metadata
        is_relevant, method = _check_enriched_relevance(query, chunks)
        if method != "no_enrichment":
            if is_relevant:
                logger.debug(f"{PIPELINE} InsufficientEvidenceConstraint: relevant via {method}")
                return ConstraintResult.allow()
            else:
                logger.info(f"{PIPELINE} InsufficientEvidenceConstraint: {method} -> ABSTAIN")
                return ConstraintResult.deny(
                    reason="Context not relevant (no entity or summary overlap)",
                    signal="abstain",
                    evidence_count=len(chunks),
                    method=method,
                )

        # Rule 5: Fallback to lexical overlap
        if not _has_lexical_overlap(query, chunks):
            logger.info(f"{PIPELINE} InsufficientEvidenceConstraint: no lexical overlap -> ABSTAIN")
            return ConstraintResult.deny(
                reason="Context does not appear related to query",
                signal="abstain",
                evidence_count=len(chunks),
            )

        logger.debug(f"{PIPELINE} InsufficientEvidenceConstraint: allowing (lexical overlap found)")
        return ConstraintResult.allow()


__all__ = ["InsufficientEvidenceConstraint"]
