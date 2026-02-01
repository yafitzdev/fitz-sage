# fitz_ai/evaluation/benchmarks/rgb.py
"""
RGB (Robustness, Grounding, Balance) benchmark for RAG robustness testing.

RGB tests four critical robustness dimensions:
1. Noise Robustness: Can the system handle noisy/irrelevant contexts?
2. Negative Rejection: Can the system abstain when it cannot answer?
3. Information Integration: Can the system synthesize from multiple sources?
4. Counterfactual Robustness: Can the system detect conflicting information?

These map directly to Fitz's governance modes:
- Negative Rejection → ABSTAIN mode
- Counterfactual Robustness → DISPUTED mode

Usage:
    from fitz_ai.evaluation.benchmarks import RGBEvaluator, RGBTestType

    evaluator = RGBEvaluator()
    results = evaluator.evaluate(engine, test_type=RGBTestType.NEGATIVE_REJECTION)
    print(f"Abstention accuracy: {results.score:.2%}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_rag.engine import FitzRagEngine

logger = get_logger(__name__)


class RGBTestType(str, Enum):
    """Types of RGB robustness tests."""

    NOISE_ROBUSTNESS = "noise_robustness"
    """Test resilience to noisy/irrelevant context mixed with relevant info."""

    NEGATIVE_REJECTION = "negative_rejection"
    """Test ability to abstain when context doesn't support an answer."""

    INFORMATION_INTEGRATION = "information_integration"
    """Test ability to synthesize answer from multiple context pieces."""

    COUNTERFACTUAL_ROBUSTNESS = "counterfactual_robustness"
    """Test detection of conflicting information in context."""


@dataclass
class RGBCase:
    """A single RGB test case."""

    id: str
    """Unique identifier for the test case."""

    test_type: RGBTestType
    """Type of robustness test."""

    query: str
    """The question to answer."""

    contexts: list[str]
    """Context passages to use (may include noise, conflicts, etc.)."""

    expected_behavior: str
    """Description of expected behavior (e.g., "should abstain")."""

    expected_mode: AnswerMode | None = None
    """Expected Fitz answer mode (for governance tests)."""

    ground_truth: str | None = None
    """Expected answer content (for correctness tests)."""

    noise_contexts: list[str] | None = None
    """Explicitly marked noise contexts (for noise robustness)."""

    conflicting_contexts: list[tuple[str, str]] | None = None
    """Pairs of conflicting contexts (for counterfactual tests)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional test case metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "test_type": self.test_type.value,
            "query": self.query,
            "contexts": self.contexts,
            "expected_behavior": self.expected_behavior,
            "expected_mode": self.expected_mode.value if self.expected_mode else None,
            "ground_truth": self.ground_truth,
            "noise_contexts": self.noise_contexts,
            "conflicting_contexts": self.conflicting_contexts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RGBCase:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            test_type=RGBTestType(data["test_type"]),
            query=data["query"],
            contexts=data["contexts"],
            expected_behavior=data["expected_behavior"],
            expected_mode=AnswerMode(data["expected_mode"]) if data.get("expected_mode") else None,
            ground_truth=data.get("ground_truth"),
            noise_contexts=data.get("noise_contexts"),
            conflicting_contexts=data.get("conflicting_contexts"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RGBCaseResult:
    """Result for a single RGB test case."""

    case: RGBCase
    """The test case."""

    passed: bool
    """Whether the test passed."""

    answer: str
    """Generated answer."""

    actual_mode: AnswerMode
    """Actual Fitz answer mode."""

    expected_mode: AnswerMode | None
    """Expected Fitz answer mode."""

    mode_correct: bool
    """Whether the mode matched expected."""

    answer_correct: bool | None
    """Whether the answer content was correct (if verifiable)."""

    failure_reason: str | None = None
    """Explanation if test failed."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case": self.case.to_dict(),
            "passed": self.passed,
            "answer": self.answer,
            "actual_mode": self.actual_mode.value,
            "expected_mode": self.expected_mode.value if self.expected_mode else None,
            "mode_correct": self.mode_correct,
            "answer_correct": self.answer_correct,
            "failure_reason": self.failure_reason,
        }


@dataclass
class RGBTypeResult:
    """Results for a single RGB test type."""

    test_type: RGBTestType
    """The test type."""

    score: float
    """Overall score (0-1) for this test type."""

    num_passed: int
    """Number of tests passed."""

    num_total: int
    """Total number of tests."""

    case_results: list[RGBCaseResult]
    """Individual case results."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_type": self.test_type.value,
            "score": self.score,
            "num_passed": self.num_passed,
            "num_total": self.num_total,
            "case_results": [r.to_dict() for r in self.case_results],
        }


@dataclass
class RGBResult:
    """Full RGB benchmark results."""

    overall_score: float
    """Overall robustness score (0-1)."""

    noise_robustness: RGBTypeResult | None
    """Noise robustness results."""

    negative_rejection: RGBTypeResult | None
    """Negative rejection results."""

    information_integration: RGBTypeResult | None
    """Information integration results."""

    counterfactual_robustness: RGBTypeResult | None
    """Counterfactual robustness results."""

    num_cases: int
    """Total number of test cases."""

    evaluation_time_seconds: float
    """Time taken for evaluation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When evaluation was run."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "noise_robustness": self.noise_robustness.to_dict() if self.noise_robustness else None,
            "negative_rejection": (
                self.negative_rejection.to_dict() if self.negative_rejection else None
            ),
            "information_integration": (
                self.information_integration.to_dict() if self.information_integration else None
            ),
            "counterfactual_robustness": (
                self.counterfactual_robustness.to_dict() if self.counterfactual_robustness else None
            ),
            "num_cases": self.num_cases,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RGBResult:
        """Create from dictionary."""

        # Helper to reconstruct type results
        def _parse_type_result(d: dict | None) -> RGBTypeResult | None:
            if d is None:
                return None
            case_results = []
            for cr in d.get("case_results", []):
                case_results.append(
                    RGBCaseResult(
                        case=RGBCase.from_dict(cr["case"]),
                        passed=cr["passed"],
                        answer=cr["answer"],
                        actual_mode=AnswerMode(cr["actual_mode"]),
                        expected_mode=(
                            AnswerMode(cr["expected_mode"]) if cr.get("expected_mode") else None
                        ),
                        mode_correct=cr["mode_correct"],
                        answer_correct=cr.get("answer_correct"),
                        failure_reason=cr.get("failure_reason"),
                    )
                )
            return RGBTypeResult(
                test_type=RGBTestType(d["test_type"]),
                score=d["score"],
                num_passed=d["num_passed"],
                num_total=d["num_total"],
                case_results=case_results,
            )

        return cls(
            overall_score=data["overall_score"],
            noise_robustness=_parse_type_result(data.get("noise_robustness")),
            negative_rejection=_parse_type_result(data.get("negative_rejection")),
            information_integration=_parse_type_result(data.get("information_integration")),
            counterfactual_robustness=_parse_type_result(data.get("counterfactual_robustness")),
            num_cases=data["num_cases"],
            evaluation_time_seconds=data["evaluation_time_seconds"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        parts = [f"RGB Results (n={self.num_cases}):"]
        parts.append(f"  Overall Score: {self.overall_score:.2%}")
        if self.noise_robustness:
            parts.append(f"  Noise Robustness: {self.noise_robustness.score:.2%}")
        if self.negative_rejection:
            parts.append(f"  Negative Rejection: {self.negative_rejection.score:.2%}")
        if self.information_integration:
            parts.append(f"  Information Integration: {self.information_integration.score:.2%}")
        if self.counterfactual_robustness:
            parts.append(f"  Counterfactual Robustness: {self.counterfactual_robustness.score:.2%}")
        return "\n".join(parts)


class RGBEvaluator:
    """
    RGB robustness evaluator for Fitz RAG pipelines.

    Tests four dimensions of robustness that are critical for production RAG:
    1. Noise Robustness - handling irrelevant context
    2. Negative Rejection - abstaining when appropriate
    3. Information Integration - synthesizing multiple sources
    4. Counterfactual Robustness - detecting conflicts

    Example:
        evaluator = RGBEvaluator()

        # Run all tests
        results = evaluator.evaluate_all(engine, test_cases)

        # Run specific test type
        rejection_results = evaluator.evaluate(
            engine,
            test_cases,
            test_type=RGBTestType.NEGATIVE_REJECTION
        )
    """

    def __init__(self, llm_judge: Any | None = None):
        """
        Initialize RGB evaluator.

        Args:
            llm_judge: Optional LLM for judging answer correctness.
                      If not provided, only mode-based tests are run.
        """
        self._llm_judge = llm_judge

    def evaluate(
        self,
        engine: FitzRagEngine,
        cases: list[RGBCase],
        test_type: RGBTestType | None = None,
    ) -> RGBResult:
        """
        Evaluate RGB robustness.

        Args:
            engine: Fitz RAG engine to evaluate
            cases: List of RGB test cases
            test_type: Optional filter for specific test type

        Returns:
            RGBResult with scores and details
        """
        import time

        start_time = time.time()

        # Filter cases by type if specified
        if test_type:
            cases = [c for c in cases if c.test_type == test_type]

        # Group cases by type
        by_type: dict[RGBTestType, list[RGBCase]] = {}
        for case in cases:
            if case.test_type not in by_type:
                by_type[case.test_type] = []
            by_type[case.test_type].append(case)

        # Evaluate each type
        type_results: dict[RGBTestType, RGBTypeResult] = {}
        for t, t_cases in by_type.items():
            type_results[t] = self._evaluate_type(engine, t_cases, t)

        evaluation_time = time.time() - start_time

        # Calculate overall score
        if type_results:
            overall_score = sum(r.score for r in type_results.values()) / len(type_results)
        else:
            overall_score = 0.0

        return RGBResult(
            overall_score=overall_score,
            noise_robustness=type_results.get(RGBTestType.NOISE_ROBUSTNESS),
            negative_rejection=type_results.get(RGBTestType.NEGATIVE_REJECTION),
            information_integration=type_results.get(RGBTestType.INFORMATION_INTEGRATION),
            counterfactual_robustness=type_results.get(RGBTestType.COUNTERFACTUAL_ROBUSTNESS),
            num_cases=len(cases),
            evaluation_time_seconds=evaluation_time,
        )

    def _evaluate_type(
        self,
        engine: FitzRagEngine,
        cases: list[RGBCase],
        test_type: RGBTestType,
    ) -> RGBTypeResult:
        """Evaluate a specific test type."""
        case_results: list[RGBCaseResult] = []

        for case in cases:
            result = self._evaluate_case(engine, case)
            case_results.append(result)

        num_passed = sum(1 for r in case_results if r.passed)

        return RGBTypeResult(
            test_type=test_type,
            score=num_passed / len(cases) if cases else 0.0,
            num_passed=num_passed,
            num_total=len(cases),
            case_results=case_results,
        )

    def _evaluate_case(self, engine: FitzRagEngine, case: RGBCase) -> RGBCaseResult:
        """Evaluate a single test case."""
        from fitz_ai.core import Query

        # Run pipeline with injected contexts
        query = Query(text=case.query)

        # For RGB tests, we inject contexts directly rather than using retrieval
        answer = self._run_with_contexts(engine, query, case.contexts)

        # Get actual mode from governance decision
        actual_mode = self._get_answer_mode(answer)

        # Check mode correctness
        mode_correct = True
        if case.expected_mode:
            mode_correct = actual_mode == case.expected_mode

        # Check answer correctness (if ground truth available and LLM judge configured)
        answer_correct = None
        if case.ground_truth and self._llm_judge:
            answer_correct = self._judge_answer(case.query, answer.text, case.ground_truth)

        # Determine if test passed based on test type
        passed = self._check_passed(case, actual_mode, answer.text, answer_correct)

        failure_reason = None
        if not passed:
            failure_reason = self._get_failure_reason(case, actual_mode, answer_correct)

        return RGBCaseResult(
            case=case,
            passed=passed,
            answer=answer.text,
            actual_mode=actual_mode,
            expected_mode=case.expected_mode,
            mode_correct=mode_correct,
            answer_correct=answer_correct,
            failure_reason=failure_reason,
        )

    def _run_with_contexts(self, engine: FitzRagEngine, query, contexts: list[str]):
        """Run engine with injected contexts (bypass retrieval)."""
        # Create mock chunks from contexts
        from fitz_ai.core import Chunk

        chunks = [
            Chunk(
                id=f"rgb_context_{i}",
                text=ctx,
                source_file="rgb_test",
                start_char=0,
                end_char=len(ctx),
            )
            for i, ctx in enumerate(contexts)
        ]

        # Use pipeline's generation step directly with injected chunks
        pipeline = engine._pipeline

        # Get governance decision
        governance = pipeline._get_governance_decision(query.text, chunks)

        # Generate answer
        answer = pipeline._generate_answer(query.text, chunks, governance)

        return answer

    def _get_answer_mode(self, answer) -> AnswerMode:
        """Extract answer mode from answer object."""
        if hasattr(answer, "mode"):
            return answer.mode
        if hasattr(answer, "governance") and hasattr(answer.governance, "mode"):
            return answer.governance.mode
        # Default to confident if mode not available
        return AnswerMode.CONFIDENT

    def _judge_answer(self, query: str, answer: str, ground_truth: str) -> bool:
        """Use LLM to judge if answer is correct."""
        if not self._llm_judge:
            return True  # Assume correct if no judge

        prompt = f"""Judge if the answer correctly addresses the question based on the ground truth.

Question: {query}
Ground Truth: {ground_truth}
Answer: {answer}

Is the answer correct? Respond with only 'yes' or 'no'."""

        response = self._llm_judge.complete(prompt)
        return "yes" in response.lower()

    def _check_passed(
        self,
        case: RGBCase,
        actual_mode: AnswerMode,
        answer: str,
        answer_correct: bool | None,
    ) -> bool:
        """Check if test case passed based on test type."""
        if case.test_type == RGBTestType.NEGATIVE_REJECTION:
            # Should abstain when context doesn't support answer
            if case.expected_mode == AnswerMode.ABSTAIN:
                return actual_mode == AnswerMode.ABSTAIN

        if case.test_type == RGBTestType.COUNTERFACTUAL_ROBUSTNESS:
            # Should detect conflict (DISPUTED mode)
            if case.expected_mode == AnswerMode.DISPUTED:
                return actual_mode == AnswerMode.DISPUTED

        if case.test_type == RGBTestType.NOISE_ROBUSTNESS:
            # Should still get correct answer despite noise
            if answer_correct is not None:
                return answer_correct
            # Fall back to mode check
            return actual_mode in (AnswerMode.CONFIDENT, AnswerMode.QUALIFIED)

        if case.test_type == RGBTestType.INFORMATION_INTEGRATION:
            # Should synthesize from multiple sources
            if answer_correct is not None:
                return answer_correct
            return actual_mode == AnswerMode.CONFIDENT

        # Default: check mode if expected
        if case.expected_mode:
            return actual_mode == case.expected_mode

        return True

    def _get_failure_reason(
        self,
        case: RGBCase,
        actual_mode: AnswerMode,
        answer_correct: bool | None,
    ) -> str:
        """Generate failure reason."""
        if case.expected_mode and actual_mode != case.expected_mode:
            return f"Expected mode {case.expected_mode.value}, got {actual_mode.value}"

        if answer_correct is False:
            return "Answer content was incorrect"

        return "Test failed (unknown reason)"

    def evaluate_from_file(
        self,
        engine: FitzRagEngine,
        path: Path | str,
        test_type: RGBTestType | None = None,
    ) -> RGBResult:
        """Load test cases from JSON and evaluate."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        cases = [RGBCase.from_dict(c) for c in data["cases"]]
        return self.evaluate(engine, cases, test_type)

    def save_results(self, result: RGBResult, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved RGB results to {path}")

    def load_results(self, path: Path | str) -> RGBResult:
        """Load results from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return RGBResult.from_dict(data)


def create_negative_rejection_case(
    id: str,
    query: str,
    irrelevant_contexts: list[str],
) -> RGBCase:
    """Helper to create a negative rejection test case."""
    return RGBCase(
        id=id,
        test_type=RGBTestType.NEGATIVE_REJECTION,
        query=query,
        contexts=irrelevant_contexts,
        expected_behavior="Should abstain - context does not support answering the question",
        expected_mode=AnswerMode.ABSTAIN,
    )


def create_counterfactual_case(
    id: str,
    query: str,
    context_a: str,
    context_b: str,
) -> RGBCase:
    """Helper to create a counterfactual robustness test case."""
    return RGBCase(
        id=id,
        test_type=RGBTestType.COUNTERFACTUAL_ROBUSTNESS,
        query=query,
        contexts=[context_a, context_b],
        expected_behavior="Should detect conflict and report disputed information",
        expected_mode=AnswerMode.DISPUTED,
        conflicting_contexts=[(context_a, context_b)],
    )


def create_noise_robustness_case(
    id: str,
    query: str,
    relevant_context: str,
    noise_contexts: list[str],
    ground_truth: str,
) -> RGBCase:
    """Helper to create a noise robustness test case."""
    all_contexts = [relevant_context] + noise_contexts
    return RGBCase(
        id=id,
        test_type=RGBTestType.NOISE_ROBUSTNESS,
        query=query,
        contexts=all_contexts,
        expected_behavior="Should answer correctly despite noise",
        ground_truth=ground_truth,
        noise_contexts=noise_contexts,
    )
