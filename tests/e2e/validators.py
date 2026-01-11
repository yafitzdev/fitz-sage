# tests/e2e/validators.py
"""
Answer validation for E2E tests.

Validates RAG pipeline answers against expected outcomes defined in scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import RGSAnswer

    from .scenarios import TestScenario


@dataclass
class ValidationResult:
    """Result of validating an answer against a scenario."""

    passed: bool
    reason: str
    details: dict


def validate_answer(answer: "RGSAnswer", scenario: "TestScenario") -> ValidationResult:
    """
    Validate an answer against a test scenario.

    Checks:
    1. must_contain: All substrings must be present (case-insensitive)
    2. must_contain_any: At least one substring must be present (case-insensitive)
    3. must_not_contain: No substrings should be present (case-insensitive)
    4. expected_mode: Answer mode matches if specified
    5. min_sources: Minimum number of source citations
    6. custom_validator: Custom validation function if provided

    Args:
        answer: The RGSAnswer from the pipeline
        scenario: The test scenario with expected outcomes

    Returns:
        ValidationResult with pass/fail status and reason
    """
    answer_text = answer.answer.lower()
    details: dict = {}

    # 1. Check must_contain (all required)
    if scenario.must_contain:
        missing = []
        for substring in scenario.must_contain:
            if substring.lower() not in answer_text:
                missing.append(substring)
        if missing:
            return ValidationResult(
                passed=False,
                reason=f"Missing required content: {missing}",
                details={"missing": missing, "answer_preview": answer.answer[:200]},
            )
        details["must_contain"] = "all present"

    # 2. Check must_contain_any (at least one required)
    if scenario.must_contain_any:
        found_any = False
        found_items = []
        for substring in scenario.must_contain_any:
            if substring.lower() in answer_text:
                found_any = True
                found_items.append(substring)
        if not found_any:
            return ValidationResult(
                passed=False,
                reason=f"Missing at least one of: {scenario.must_contain_any}",
                details={
                    "expected_any_of": scenario.must_contain_any,
                    "answer_preview": answer.answer[:200],
                },
            )
        details["must_contain_any"] = f"found: {found_items}"

    # 3. Check must_not_contain (none should be present)
    if scenario.must_not_contain:
        found_forbidden = []
        for substring in scenario.must_not_contain:
            if substring.lower() in answer_text:
                found_forbidden.append(substring)
        if found_forbidden:
            return ValidationResult(
                passed=False,
                reason=f"Found forbidden content: {found_forbidden}",
                details={"forbidden_found": found_forbidden},
            )
        details["must_not_contain"] = "none found"

    # 4. Check expected_mode
    if scenario.expected_mode:
        actual_mode = answer.mode.value if hasattr(answer, "mode") and answer.mode else None
        if actual_mode != scenario.expected_mode:
            return ValidationResult(
                passed=False,
                reason=f"Expected mode '{scenario.expected_mode}', got '{actual_mode}'",
                details={"expected_mode": scenario.expected_mode, "actual_mode": actual_mode},
            )
        details["mode"] = actual_mode

    # 5. Check min_sources
    if scenario.min_sources > 0:
        source_count = len(answer.sources) if hasattr(answer, "sources") and answer.sources else 0
        if source_count < scenario.min_sources:
            return ValidationResult(
                passed=False,
                reason=f"Expected at least {scenario.min_sources} sources, got {source_count}",
                details={"expected_min": scenario.min_sources, "actual": source_count},
            )
        details["sources"] = source_count

    # 6. Run custom validator if provided
    if scenario.custom_validator:
        try:
            custom_passed = scenario.custom_validator(answer)
            if not custom_passed:
                return ValidationResult(
                    passed=False,
                    reason="Custom validator failed",
                    details={"custom_validator": "failed"},
                )
            details["custom_validator"] = "passed"
        except Exception as e:
            return ValidationResult(
                passed=False,
                reason=f"Custom validator raised exception: {e}",
                details={"custom_validator_error": str(e)},
            )

    # All checks passed
    return ValidationResult(
        passed=True,
        reason="All validations passed",
        details=details,
    )
