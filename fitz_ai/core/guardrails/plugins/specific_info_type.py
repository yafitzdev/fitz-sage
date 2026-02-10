# fitz_ai/core/guardrails/plugins/specific_info_type.py
"""
Specific Info Type Constraint - Detects when specific information types are missing.

This constraint prevents confident answers when the query asks for specific
information (price, date, count, etc.) that isn't present in the context.

It complements InsufficientEvidenceConstraint by catching cases where:
- Context is topically related (high similarity)
- But lacks the SPECIFIC type of information requested
- Should return 'qualified' instead of 'confident'

DESIGN PRINCIPLE: This constraint must be CONSERVATIVE. It is far better to
allow a confident answer through (false negative) than to wrongly downgrade
a good answer to qualified (false positive). Only fire when we are VERY
confident the specific info is genuinely absent from the context.
"""

import re
from typing import Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger

from ..base import ConstraintResult

logger = get_logger(__name__)


class SpecificInfoTypeConstraint:
    """
    Constraint that detects when specific information types are missing.

    When a query asks for specific information (price, date, quantity, etc.)
    and the context discusses the topic but lacks that specific info,
    this constraint triggers 'qualified' mode.

    Conservative by design: only fires on high-confidence detections.
    """

    def __init__(self, enabled: bool = True):
        """Initialize the constraint."""
        self.enabled = enabled
        self.name = "specific_info_type"

    def apply(self, query: str, chunks: Sequence[Chunk]) -> ConstraintResult:
        """
        Check if chunks contain the specific type of information requested.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies confident answer if specific info is missing
        """
        if not self.enabled or not chunks:
            return ConstraintResult.allow()

        # Check for entity mismatch (e.g., ProTab X1 vs ProTab X2)
        entity_mismatch = self._has_entity_mismatch(query, chunks)

        # Identify what type of information is requested (strict detection)
        info_type = self._identify_info_type(query)

        # Check if chunks contain the requested info type
        has_specific_info = (
            self._check_for_info_type(chunks, info_type, query) if info_type else None
        )

        # Build diagnostics for classifier feature extraction
        sit_diag = {
            "sit_entity_mismatch": entity_mismatch,
            "sit_info_type_requested": info_type,
            "sit_has_specific_info": has_specific_info,
        }

        if entity_mismatch:
            logger.info("SpecificInfoTypeConstraint: Entity mismatch detected -> QUALIFIED")
            return ConstraintResult.deny(
                reason="Context discusses a different entity or version",
                signal="qualified",
                **sit_diag,
            )

        if not info_type:
            # No specific info type detected, allow confident answer
            return ConstraintResult.allow(**sit_diag)

        if has_specific_info:
            # Found the specific info, allow confident answer
            return ConstraintResult.allow(**sit_diag)

        # Context is related but missing specific info -> qualified
        logger.info(f"SpecificInfoTypeConstraint: Missing {info_type} information -> QUALIFIED")

        return ConstraintResult.deny(
            reason=f"Context discusses the topic but lacks specific {info_type} information",
            signal="qualified",
            **sit_diag,
        )

    def _has_entity_mismatch(self, query: str, chunks: Sequence[Chunk]) -> bool:
        """
        Check if query asks about a specific entity but context has a different one.

        Examples:
        - Query: "ProTab X1" but context has "ProTab X2"
        - Query: "2024 sales" but context has "2023 sales"
        - Query: "online purchases" but context has "in-store purchases"
        """
        combined_text = " ".join(chunk.content.lower() for chunk in chunks)
        query_lower = query.lower()

        # Look for version mismatches (X1 vs X2, 2024 vs 2023, etc.)
        version_patterns = [
            (r"x\d+", r"x\d+"),  # X1, X2, etc.
            (r"v\d+", r"v\d+"),  # v1, v2, etc.
            (r"20\d{2}", r"20\d{2}"),  # Years
            (r"q[1-4]", r"q[1-4]"),  # Quarters
        ]

        for query_pattern, context_pattern in version_patterns:
            query_match = re.search(query_pattern, query_lower)
            if query_match:
                context_matches = re.findall(context_pattern, combined_text)
                if context_matches:
                    # Check if ANY match is different from query
                    query_version = query_match.group()
                    if all(match != query_version for match in context_matches):
                        return True  # Entity mismatch

        # Look for specific product/service mismatches
        specific_mismatches = [
            ("online", "in-store"),
            ("in-store", "online"),
            ("elderly", "children"),
            ("children", "elderly"),
            ("before", "after"),
            ("after", "before"),
        ]

        for query_term, context_term in specific_mismatches:
            if query_term in query_lower and query_term not in combined_text:
                if context_term in combined_text:
                    return True  # Entity mismatch

        return False

    def _identify_info_type(self, query: str) -> str | None:
        """
        Identify what type of specific information is being requested.

        CONSERVATIVE: Only returns an info type when the query structure
        clearly and unambiguously asks for a specific category of information.
        Generic questions ("What is X?", "How does X work?") should NOT trigger.

        Returns the info type or None if query doesn't ask for specific info.
        """
        query_lower = query.lower().strip()

        # PRICING: Only when explicitly asking about price/cost with clear intent
        # "how much does X cost" or "what is the price of X"
        if re.search(
            r"\b(price|pricing|cost|fee|charge|tariff)\b.*\b(of|for|to)\b",
            query_lower,
        ) or re.search(r"\bhow much\b.*\b(cost|charge|pay)\b", query_lower):
            return "pricing"

        # QUANTITY: Only "how many X" pattern - very specific
        if re.search(r"\bhow many\b", query_lower):
            return "quantity"

        # TEMPORAL: Only when asking for a specific date/deadline/schedule
        # NOT "when" in general - only "when is/was/will the deadline/date/etc."
        if re.search(
            r"\b(when is|when was|when will|when does|when did)\b.*"
            r"\b(deadline|due date|release|launch|expire|expiration|completion|delivery)\b",
            query_lower,
        ):
            return "temporal"
        # "what is the deadline/date for X"
        if re.search(
            r"\bwhat\b.*\b(deadline|due date|release date|launch date|expiration date)\b",
            query_lower,
        ):
            return "temporal"

        # SPECIFICATION: Only when asking for specific numeric specs/limits/requirements
        # "what is the maximum/minimum X" or "what are the requirements for X"
        if re.search(
            r"\bwhat\b.*\b(maximum|minimum|max|min|limit|capacity)\b.*\b(of|for)\b",
            query_lower,
        ):
            return "specification"
        if re.search(
            r"\b(maximum|minimum|max|min)\b.*\b(load|weight|capacity|size|speed)\b",
            query_lower,
        ):
            return "specification"

        # MEASUREMENT: Only when explicitly asking for a dose/dimension/size with clear intent
        if re.search(
            r"\bwhat\b.*\b(dosage|dose|dimension|size|weight|height|length|width)\b.*\b(of|for)\b",
            query_lower,
        ):
            return "measurement"
        if re.search(
            r"\b(recommended|maximum|correct|proper)\b.*\b(dosage|dose)\b",
            query_lower,
        ):
            return "measurement"

        # WARRANTY: Only when explicitly asking about warranty/coverage terms
        if re.search(
            r"\b(what|does|is)\b.*\b(warranty|guarantee|coverage)\b.*\b(cover|include|last|length|duration|period)\b",
            query_lower,
        ):
            return "warranty"
        if re.search(
            r"\b(warranty|guarantee)\b.*\b(for|on|of)\b",
            query_lower,
        ):
            return "warranty"

        # RATE: Only when explicitly asking for a rate, percentage, salary, or average
        # "what is the X rate" or "what is the rate of/for X"
        if re.search(
            r"\bwhat\b.*\b(rate|percentage|percent|ratio)\b",
            query_lower,
        ):
            return "rate"
        if re.search(
            r"\b(average|median|mean)\b.*\b(salary|wage|income|pay|compensation)\b",
            query_lower,
        ):
            return "rate"

        # DECISION: Only very explicit decision-seeking patterns
        if re.search(
            r"\b(should we|should i|is it worth|worth the risk|should we proceed)\b",
            query_lower,
        ):
            return "decision"

        # Everything else: DO NOT detect. Common question patterns like
        # "What is X?", "How does X work?", "Why did X happen?",
        # "What programming language...", "Where is X located?" are
        # all generic factual questions that should NOT trigger this constraint.
        return None

    def _check_for_info_type(self, chunks: Sequence[Chunk], info_type: str, query: str) -> bool:
        """
        Check if chunks contain the specific type of information.

        GENEROUS: Returns True (info found) unless we are very confident
        it is absent. Any plausible evidence counts.
        """
        combined_text = " ".join(chunk.content.lower() for chunk in chunks)

        if info_type == "pricing":
            # Look for currency symbols, price patterns, or cost mentions with numbers
            if re.search(r"[$\u20ac\u00a3\u00a5\u20b9][\d,]+", combined_text):
                return True
            if re.search(r"\d+\s*(dollar|euro|pound|cent|rupee)", combined_text):
                return True
            if re.search(r"(price|cost|fee|charge|rate).*?\d+", combined_text):
                return True
            if re.search(r"\d+.*(price|cost|fee|charge|rate)", combined_text):
                return True
            # Also check for words like "free", "no charge", "included"
            if re.search(r"\b(free|no charge|no cost|included|complimentary)\b", combined_text):
                return True

        elif info_type == "quantity":
            # Look for ANY number that could answer a "how many" question
            # Be generous: if there are numbers in relevant context, assume found
            query_entities = self._extract_query_subject(query)
            for entity in query_entities:
                entity_escaped = re.escape(entity.lower())
                # Number near entity in either direction (within 100 chars)
                if re.search(rf"\d+.{{0,100}}{entity_escaped}", combined_text):
                    return True
                if re.search(rf"{entity_escaped}.{{0,100}}\d+", combined_text):
                    return True
            # Fallback: if context contains numbers with quantity words, likely answers it
            if re.search(
                r"\d+\s*(employees|users|customers|members|people|items|units)", combined_text
            ):
                return True
            # If the context has percentages or counts, likely relevant
            if re.search(r"\d+%|\d+\s*(thousand|million|billion|hundred)", combined_text):
                return True

        elif info_type == "temporal":
            # Look for dates, times, deadlines - be generous
            date_patterns = [
                r"\d{4}",  # Year
                r"\d{1,2}/\d{1,2}",  # MM/DD
                r"(january|february|march|april|may|june|july|august|september|october|november|december)",
                r"(q1|q2|q3|q4)",  # Quarters
                r"\d{1,2}:\d{2}",  # Time
                r"(deadline|due|expire|schedule|target).*?\d",
                r"(tomorrow|yesterday|today|next week|last month|next month|this year|last year)",
                r"\d+\s*(day|week|month|year)s?",  # Duration
            ]
            for pattern in date_patterns:
                if re.search(pattern, combined_text):
                    return True

        elif info_type == "specification":
            # Look for specific requirements or limits - be generous
            if re.search(
                r"(require|minimum|maximum|limit|capacity|threshold).*?\d+", combined_text
            ):
                return True
            if re.search(r"\d+.*(require|minimum|maximum|limit|capacity|threshold)", combined_text):
                return True
            if re.search(r"(must|should|need|at least|up to|no more than).*?\d+", combined_text):
                return True
            if re.search(r"(gb|mb|tb|ghz|mhz|kg|lbs?|mph|km)", combined_text):
                return True

        elif info_type == "measurement":
            # Look for measurements with units - be generous
            if re.search(r"\d+\s*(mg|g|kg|ml|l|cm|mm|m|km|inch|inches|foot|feet)", combined_text):
                return True
            if re.search(r"\d+\s*(milligram|gram|kilogram|milliliter|liter)", combined_text):
                return True
            if re.search(r"(dose|dosage|measurement|size|dimension).*?\d+", combined_text):
                return True
            # Any number with a unit-like suffix
            if re.search(r"\d+\s*[a-z]{1,4}\b", combined_text):
                return True

        elif info_type == "decision":
            # Look for pros/cons, recommendations, analysis - be very generous
            if re.search(r"(recommend|suggest|advise|consider)", combined_text):
                return True
            if re.search(r"(pro|con|advantage|disadvantage|benefit|drawback)", combined_text):
                return True
            if re.search(r"(risk|opportunity|trade.?off|implication)", combined_text):
                return True
            if re.search(r"(should|worth|advisable|prudent)", combined_text):
                return True
            # If context discusses the topic at all with evaluative language, count it
            if re.search(r"(good|bad|better|worse|best|worst|effective|efficient)", combined_text):
                return True

        elif info_type == "rate":
            # Look for percentages, rates, ratios, salary figures
            if re.search(r"\d+(\.\d+)?\s*%", combined_text):
                return True
            if re.search(r"\d+\s*percent", combined_text):
                return True
            if re.search(r"(rate|ratio)\s*(of|is|was|:)\s*\d", combined_text):
                return True
            if re.search(r"\$[\d,]+", combined_text):
                return True
            if re.search(r"\d+\s*(per|/)\s*(year|month|hour|annum|capita)", combined_text):
                return True
            if re.search(r"(salary|wage|income|compensation|pay).*?\d", combined_text):
                return True
            if re.search(r"\d+.*(salary|wage|income|compensation|pay)", combined_text):
                return True

        elif info_type == "warranty":
            # Look for warranty/coverage details - be generous
            if re.search(r"\d+\s*(year|month|day)", combined_text):
                return True
            if re.search(r"(warranty|coverage|guarantee|protection)", combined_text):
                return True
            if re.search(r"(covers?|protects?|includes?)", combined_text):
                return True
            if re.search(r"(defects?|damage|repair|replacement)", combined_text):
                return True

        return False

    def _extract_query_subject(self, query: str) -> list[str]:
        """Extract the main subject/entity from the query."""
        # Remove question words and extract key entities
        query_lower = query.lower()

        # Remove common question starters
        for starter in ["what", "how many", "when", "where", "why", "who", "which"]:
            query_lower = query_lower.replace(starter, "")

        # Extract capitalized words (likely entities)
        entities = []
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word.lower())

        # Also extract nouns after "of" or "for"
        if " of " in query_lower:
            after_of = query_lower.split(" of ")[-1].split()[0]
            if after_of:
                entities.append(after_of)
        if " for " in query_lower:
            after_for = query_lower.split(" for ")[-1].split()[0]
            if after_for:
                entities.append(after_for)

        return entities if entities else ["users", "platform", "system", "product", "service"]


__all__ = ["SpecificInfoTypeConstraint"]
