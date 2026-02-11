# fitz_ai/governance/constraints/aspect_classifier.py
"""
Query and chunk aspect classification for intent alignment.

Aspects represent different facets of an entity:
- CAUSE: Why something happens, root causes
- SYMPTOM: Observable effects, manifestations
- TREATMENT: Solutions, interventions, fixes
- DEFINITION: What something is, core explanation
- PROCESS: How something works, steps involved
- PRICING: Cost, financial information
- COMPARISON: Evaluation against alternatives
- TIMELINE: When something happened, temporal sequence
"""

import re
from dataclasses import dataclass
from enum import Enum


class QueryAspect(Enum):
    """Query intent aspects."""

    CAUSE = "cause"
    EFFECT = "effect"  # Consequences, outcomes, results
    SYMPTOM = "symptom"
    TREATMENT = "treatment"
    DEFINITION = "definition"
    PROCESS = "process"  # How it works, mechanism, manufacturing
    APPLICATION = "application"  # Use cases, applications, purposes
    PRICING = "pricing"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    PROOF = "proof"  # Mathematical proofs, evidence, verification
    GENERAL = "general"  # Catch-all


@dataclass
class AspectMatch:
    """Result of aspect compatibility check."""

    compatible: bool
    query_aspect: QueryAspect
    chunk_aspects: list[QueryAspect]
    reason: str


class AspectClassifier:
    """Classifies query and chunk aspects for alignment checking."""

    # Query patterns for aspect detection
    CAUSE_PATTERNS = [
        r"\bwhy\s+(did|does|do|is|are|was|were)\b",  # "Why did X happen?"
        r"\b(what (causes|caused)|what leads to|what results in)\b",
        r"\b(responsible for|root cause|source of|reason for)\b",
        r"\bwhy\b.*\b(happen|occur|fail|crash|break)\b",
    ]

    EFFECT_PATTERNS = [
        r"\b(what (are|were) the (effects?|results?|consequences?|outcomes?))\b",
        r"\b(what (happened|resulted|came) (after|from))\b",
        r"\b(impact|consequence|outcome|result) of\b",
    ]

    SYMPTOM_PATTERNS = [
        r"\b(symptom|sign|indicator|manifestation|side effect)\b",
        r"\b(what are the (symptoms|signs|side effects))\b",
        r"\b(how (does|do) .* (present|manifest|appear))\b",
    ]

    TREATMENT_PATTERNS = [
        r"\b(treat|cure|fix|solve|remedy|solution|intervention|therapy)\b",
        r"\b(how (to|do I|can I) (fix|solve|treat|cure|address|resolve))\b",
        r"\b(medication|medicine|drug|prescription)\b",
    ]

    DEFINITION_PATTERNS = [
        r"^what (is|are) (a |an |the )?[A-Z]",  # "What is X?" at start
        r"\b(define|definition of|meaning of)\b",
        r"\b(what does .* mean)\b",
    ]

    PROCESS_PATTERNS = [
        r"\bhow (does|do|is|are) .* (work|function|operate)\b",
        r"\b(mechanism of action|how .* works)\b",
        r"\bhow (is|are) .* (made|manufactured|produced|created|built)\b",
        r"\b(process|procedure|steps|workflow) (of|for|to)\b",
    ]

    APPLICATION_PATTERNS = [
        r"\b(what (is|are) .* used for)\b",
        r"\b(use cases?|applications?|purposes?) of\b",
        r"\b(how (is|are) .* (used|applied|utilized))\b",
    ]

    PRICING_PATTERNS = [
        r"\b(price|cost|pricing|expensive|cheap|fee|rate|subscription)\b",
        r"\b(how much|what .* cost|pay for)\b",
        r"\b(free|premium|enterprise|tier|plan)\b.*\b(cost|price|fee)\b",
    ]

    COMPARISON_PATTERNS = [
        r"\b(vs|versus|compared to|better than|difference between)\b",
        r"\b(compare|comparison|which is better|pros and cons)\b",
        r"\b(advantage over|disadvantage (of|compared)|relative benefit)\b",
        # Note: "advantage" alone is too broad - "competitive advantage" is often a definition
    ]

    TIMELINE_PATTERNS = [
        r"\b(when|timeline|chronology|sequence|history)\b",
        r"\b(what year|what time|what date|how long)\b",
        r"\b(founded|established|started|launched|released)\b",
    ]

    PROOF_PATTERNS = [
        r"\b(proof|prove|derivation|demonstration) (of|for|that)\b",
        r"\bhow (to|do you|can I) prove\b",
        r"\b(verify|verification|evidence for)\b",
    ]

    # Chunk content markers (slightly different - look for content indicators)
    CAUSE_CONTENT = [
        r"\b(caused by|because of|reason (is|was) that)\b",
        r"\b(the cause|a cause|causes include|root cause)\b",
        r"\bwhy .* (happened|occurred|failed)\b",
        r"\b(reasons for|primary reasons?|main reasons?|cited .* as the reason)\b",
    ]

    EFFECT_CONTENT = [
        r"\b(led to|resulted in|consequence|outcome|impact)\b",
        r"\b(as a result|therefore|thus|hence)\b",
        r"\b(effects? (of|include|are)|aftermath)\b",
    ]

    SYMPTOM_CONTENT = [
        r"\b(symptoms? (include|are|of)|signs? (include|are|of))\b",
        r"\b(manifests? as|presents? with|characterized by)\b",
        r"\b(common symptoms|side effects)\b",
    ]

    TREATMENT_CONTENT = [
        r"\b(treatment|treated (with|by)|therapy|medication)\b",
        r"\b(cure|remedy|to treat|can be (fixed|treated))\b",
        r"\b(clinical trials?|FDA approved)\b",
    ]

    DEFINITION_CONTENT = [
        r"\b(is defined as|refers to|means that|states that)\b",
        r"\b(definition|known as|called|termed)\b",
        r"^[A-Z][^.]*\b(is|are) (a|an|the)\b",  # "X is a Y" definitional pattern
        r"\btheorem states\b",
        r"\b(specification|requirements?|guarantees?|criteria|eligibility)\b:\s",  # Spec format
        r"(?:^|\. )[A-Z][a-z]+:\s+\d+",  # "Uptime: 99.99%" at sentence start only
    ]

    PROCESS_CONTENT = [
        r"\b(works by|mechanism of|process of|how .* works)\b",
        r"\b(steps? (to|for|in)|procedure|workflow)\b",
        r"\b(manufactured|produced|created|built) (by|through|using)\b",
        r"\b(first|second|third),? .*(then|next|finally)\b",  # Step sequences
        r"\busing (a|an|the) .* (approach|method|technique)\b",
    ]

    APPLICATION_CONTENT = [
        r"\b(used (to|for|in)|commonly used|applications? (include|of))\b",
        r"\b(use cases?|purposes?|utilized (for|in))\b",
        r"\b(is used|are used|can be used)\b",
    ]

    PRICING_CONTENT = [
        r"\b(costs? \$|prices? (of|at|is)|pricing (is|at|for)|\$\d+|USD|EUR)\b",
        r"\b(monthly (fee|cost|price)|annual (fee|cost|price)|subscription (fee|cost))\b",
        r"\b(pay|payment|billing|invoice|charge)\b.*(per|each|every)\b",
    ]

    TIMELINE_CONTENT = [
        r"\b(in \d{4}|founded in|established in|launched in|released in)\b",
        r"\b(history of|timeline of|chronology|dates back to)\b",
        r"\b(year|decade|century|era) (of|when)\b",
    ]

    PROOF_CONTENT = [
        r"\b(proof of|proved|proven|derivation of)\b",
        r"\b(QED|therefore it follows|it can be shown)\b",
        r"\b(by induction|by contradiction|lemma)\b",
    ]

    FEATURE_CONTENT = [
        r"\b(features include|includes|comes with|provides)\b",
        r"\b(capability|functionality|supports|enables)\b",
    ]

    def classify_query(self, query: str) -> QueryAspect:
        """
        Classify query into aspect category.

        Args:
            query: User query string

        Returns:
            QueryAspect enum
        """
        query_lower = query.lower()

        # Check patterns in priority order (more specific first)
        if self._matches_patterns(query_lower, self.COMPARISON_PATTERNS):
            return QueryAspect.COMPARISON
        if self._matches_patterns(query_lower, self.PROOF_PATTERNS):
            return QueryAspect.PROOF
        if self._matches_patterns(query_lower, self.CAUSE_PATTERNS):
            return QueryAspect.CAUSE
        if self._matches_patterns(query_lower, self.EFFECT_PATTERNS):
            return QueryAspect.EFFECT
        if self._matches_patterns(query_lower, self.SYMPTOM_PATTERNS):
            return QueryAspect.SYMPTOM
        if self._matches_patterns(query_lower, self.TREATMENT_PATTERNS):
            return QueryAspect.TREATMENT
        if self._matches_patterns(query_lower, self.APPLICATION_PATTERNS):
            return QueryAspect.APPLICATION
        if self._matches_patterns(query_lower, self.PRICING_PATTERNS):
            return QueryAspect.PRICING
        if self._matches_patterns(query_lower, self.PROCESS_PATTERNS):
            return QueryAspect.PROCESS
        if self._matches_patterns(query_lower, self.TIMELINE_PATTERNS):
            return QueryAspect.TIMELINE
        if self._matches_patterns(query_lower, self.DEFINITION_PATTERNS):
            return QueryAspect.DEFINITION

        return QueryAspect.GENERAL

    def extract_chunk_aspects(self, chunk_content: str) -> list[QueryAspect]:
        """
        Extract aspects present in chunk content.

        Unlike query classification (single aspect), chunks can contain
        multiple aspects. We look for content markers.

        Args:
            chunk_content: Chunk text

        Returns:
            List of aspects found in chunk
        """
        content_lower = chunk_content.lower()
        aspects = []

        # Chunks can have multiple aspects
        if self._matches_patterns(content_lower, self.CAUSE_CONTENT):
            aspects.append(QueryAspect.CAUSE)
        if self._matches_patterns(content_lower, self.EFFECT_CONTENT):
            aspects.append(QueryAspect.EFFECT)
        if self._matches_patterns(content_lower, self.SYMPTOM_CONTENT):
            aspects.append(QueryAspect.SYMPTOM)
        if self._matches_patterns(content_lower, self.TREATMENT_CONTENT):
            aspects.append(QueryAspect.TREATMENT)
        if self._matches_patterns(content_lower, self.DEFINITION_CONTENT):
            aspects.append(QueryAspect.DEFINITION)
        if self._matches_patterns(content_lower, self.PROCESS_CONTENT):
            aspects.append(QueryAspect.PROCESS)
        if self._matches_patterns(content_lower, self.APPLICATION_CONTENT):
            aspects.append(QueryAspect.APPLICATION)
        if self._matches_patterns(content_lower, self.PRICING_CONTENT):
            aspects.append(QueryAspect.PRICING)
        if self._matches_patterns(content_lower, self.TIMELINE_CONTENT):
            aspects.append(QueryAspect.TIMELINE)
        if self._matches_patterns(content_lower, self.PROOF_CONTENT):
            aspects.append(QueryAspect.PROOF)

        # If no specific aspects found, assume general content
        if not aspects:
            aspects.append(QueryAspect.GENERAL)

        return aspects

    def check_compatibility(self, query: str, chunk_content: str) -> AspectMatch:
        """
        Check if query aspect is compatible with chunk aspects.

        Args:
            query: User query
            chunk_content: Chunk text

        Returns:
            AspectMatch with compatibility result
        """
        query_aspect = self.classify_query(query)
        chunk_aspects = self.extract_chunk_aspects(chunk_content)

        # GENERAL queries match anything
        if query_aspect == QueryAspect.GENERAL:
            return AspectMatch(
                compatible=True,
                query_aspect=query_aspect,
                chunk_aspects=chunk_aspects,
                reason="General query matches any content",
            )

        # GENERAL chunks match anything
        if QueryAspect.GENERAL in chunk_aspects:
            return AspectMatch(
                compatible=True,
                query_aspect=query_aspect,
                chunk_aspects=chunk_aspects,
                reason="General content matches any query",
            )

        # Direct match
        if query_aspect in chunk_aspects:
            return AspectMatch(
                compatible=True,
                query_aspect=query_aspect,
                chunk_aspects=chunk_aspects,
                reason=f"Query aspect {query_aspect.value} found in chunk",
            )

        # Incompatible aspects
        return AspectMatch(
            compatible=False,
            query_aspect=query_aspect,
            chunk_aspects=chunk_aspects,
            reason=f"Query asks about {query_aspect.value}, chunk discusses {[a.value for a in chunk_aspects]}",
        )

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any pattern."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


__all__ = ["AspectClassifier", "AspectMatch", "QueryAspect"]
