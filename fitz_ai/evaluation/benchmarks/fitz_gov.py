# fitz_ai/evaluation/benchmarks/fitz_gov.py
"""
FITZ-GOV: Comprehensive RAG Governance Benchmark.

Fitz's core differentiator is epistemic honesty - knowing when to abstain,
dispute, or qualify answers. FITZ-GOV measures governance calibration AND
answer quality, making it a complete RAG evaluation framework.

Governance Mode Categories (maps to AnswerMode):
1. ABSTENTION - Should refuse to answer (insufficient/irrelevant context)
2. DISPUTE - Should flag conflicting information
3. QUALIFICATION - Should hedge (causal claims without evidence, etc.)
4. CONFIDENCE - Should answer confidently (clear evidence supports answer)

Answer Quality Categories (no external deps like RAGAS):
5. GROUNDING - Answer must be grounded in context (no hallucination)
6. RELEVANCE - Answer must address the actual question asked

The benchmark produces accuracy by category and a confusion matrix for
governance modes.

Usage:
    from fitz_ai.evaluation.benchmarks import FitzGovBenchmark

    benchmark = FitzGovBenchmark()
    results = benchmark.evaluate(engine)
    print(results.confusion_matrix)
    print(f"Abstention accuracy: {results.abstention.accuracy:.2%}")
    print(f"Grounding accuracy: {results.grounding.accuracy:.2%}")
"""

from __future__ import annotations

import json
from collections import defaultdict
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

# Path to bundled test data
DATA_DIR = Path(__file__).parent.parent / "data" / "fitz_gov"


class FitzGovCategory(str, Enum):
    """Categories of governance test cases."""

    # Governance Mode Categories (maps to AnswerMode)
    ABSTENTION = "abstention"
    """Cases where the system should refuse to answer."""

    DISPUTE = "dispute"
    """Cases where the system should flag conflicting information."""

    QUALIFICATION = "qualification"
    """Cases where the system should hedge or qualify the answer."""

    CONFIDENCE = "confidence"
    """Cases where the system should answer confidently."""

    # Answer Quality Categories
    GROUNDING = "grounding"
    """Cases testing if answers are grounded in context (no hallucination)."""

    RELEVANCE = "relevance"
    """Cases testing if answers address the actual question asked."""


@dataclass
class FitzGovCase:
    """A single FITZ-GOV test case."""

    id: str
    """Unique identifier for the test case."""

    category: FitzGovCategory
    """Category of governance test."""

    subcategory: str
    """More specific subcategory (e.g., "no_context", "out_of_scope")."""

    query: str
    """The question to answer."""

    contexts: list[str]
    """Context passages to use."""

    expected_mode: AnswerMode
    """Expected Fitz answer mode (for governance categories)."""

    description: str
    """Human-readable description of what's being tested."""

    rationale: str
    """Why this mode is expected."""

    # Answer quality fields (for grounding/relevance categories)
    forbidden_claims: list[str] = field(default_factory=list)
    """For GROUNDING: Claims that indicate hallucination (should NOT appear in answer)."""

    required_elements: list[str] = field(default_factory=list)
    """For RELEVANCE: Elements that MUST appear in the answer."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional test case metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "query": self.query,
            "contexts": self.contexts,
            "expected_mode": self.expected_mode.value,
            "description": self.description,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }
        # Include answer quality fields only if set
        if self.forbidden_claims:
            result["forbidden_claims"] = self.forbidden_claims
        if self.required_elements:
            result["required_elements"] = self.required_elements
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitzGovCase:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=FitzGovCategory(data["category"]),
            subcategory=data["subcategory"],
            query=data["query"],
            contexts=data["contexts"],
            expected_mode=AnswerMode(data["expected_mode"]),
            description=data["description"],
            rationale=data["rationale"],
            forbidden_claims=data.get("forbidden_claims", []),
            required_elements=data.get("required_elements", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FitzGovCaseResult:
    """Result for a single FITZ-GOV test case."""

    case: FitzGovCase
    """The test case."""

    passed: bool
    """Whether the test passed (mode matched expected)."""

    answer: str
    """Generated answer text."""

    actual_mode: AnswerMode
    """Actual Fitz answer mode."""

    triggered_constraints: list[str]
    """Constraints that were triggered."""

    failure_analysis: str | None = None
    """Analysis of why the test failed (if applicable)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case": self.case.to_dict(),
            "passed": self.passed,
            "answer": self.answer,
            "actual_mode": self.actual_mode.value,
            "triggered_constraints": self.triggered_constraints,
            "failure_analysis": self.failure_analysis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitzGovCaseResult:
        """Create from dictionary."""
        return cls(
            case=FitzGovCase.from_dict(data["case"]),
            passed=data["passed"],
            answer=data["answer"],
            actual_mode=AnswerMode(data["actual_mode"]),
            triggered_constraints=data["triggered_constraints"],
            failure_analysis=data.get("failure_analysis"),
        )


@dataclass
class FitzGovCategoryResult:
    """Results for a single governance category."""

    category: FitzGovCategory
    """The category."""

    accuracy: float
    """Accuracy for this category (0-1)."""

    num_correct: int
    """Number of correct predictions."""

    num_total: int
    """Total number of test cases."""

    case_results: list[FitzGovCaseResult]
    """Individual case results."""

    subcategory_accuracy: dict[str, float]
    """Accuracy by subcategory."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "accuracy": self.accuracy,
            "num_correct": self.num_correct,
            "num_total": self.num_total,
            "case_results": [r.to_dict() for r in self.case_results],
            "subcategory_accuracy": self.subcategory_accuracy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitzGovCategoryResult:
        """Create from dictionary."""
        return cls(
            category=FitzGovCategory(data["category"]),
            accuracy=data["accuracy"],
            num_correct=data["num_correct"],
            num_total=data["num_total"],
            case_results=[FitzGovCaseResult.from_dict(r) for r in data["case_results"]],
            subcategory_accuracy=data["subcategory_accuracy"],
        )


@dataclass
class FitzGovConfusionMatrix:
    """Confusion matrix for governance mode predictions."""

    matrix: dict[str, dict[str, int]]
    """Matrix[expected][actual] = count."""

    def __post_init__(self):
        """Initialize empty matrix if needed."""
        if not self.matrix:
            modes = [m.value for m in AnswerMode]
            self.matrix = {exp: {act: 0 for act in modes} for exp in modes}

    def add(self, expected: AnswerMode, actual: AnswerMode) -> None:
        """Add a prediction to the matrix."""
        self.matrix[expected.value][actual.value] += 1

    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        correct = sum(self.matrix[m][m] for m in self.matrix)
        total = sum(sum(row.values()) for row in self.matrix.values())
        return correct / total if total > 0 else 0.0

    def get_mode_precision(self, mode: AnswerMode) -> float:
        """Get precision for a specific mode."""
        mode_val = mode.value
        true_positive = self.matrix[mode_val][mode_val]
        predicted_positive = sum(self.matrix[exp][mode_val] for exp in self.matrix)
        return true_positive / predicted_positive if predicted_positive > 0 else 0.0

    def get_mode_recall(self, mode: AnswerMode) -> float:
        """Get recall for a specific mode."""
        mode_val = mode.value
        true_positive = self.matrix[mode_val][mode_val]
        actual_positive = sum(self.matrix[mode_val].values())
        return true_positive / actual_positive if actual_positive > 0 else 0.0

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Convert to dictionary."""
        return self.matrix

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> FitzGovConfusionMatrix:
        """Create from dictionary."""
        return cls(matrix=data)

    def __str__(self) -> str:
        """Pretty print the confusion matrix."""
        modes = [m.value for m in AnswerMode]
        lines = ["Confusion Matrix (rows=expected, cols=actual):"]
        header = "           " + " ".join(f"{m[:8]:>10}" for m in modes)
        lines.append(header)
        for exp in modes:
            row = f"{exp[:10]:>10} " + " ".join(f"{self.matrix[exp][act]:>10}" for act in modes)
            lines.append(row)
        return "\n".join(lines)


@dataclass
class FitzGovResult:
    """Full FITZ-GOV benchmark results."""

    overall_accuracy: float
    """Overall accuracy across all categories."""

    # Governance Mode Categories
    abstention: FitzGovCategoryResult | None
    """Results for abstention category."""

    dispute: FitzGovCategoryResult | None
    """Results for dispute category."""

    qualification: FitzGovCategoryResult | None
    """Results for qualification category."""

    confidence: FitzGovCategoryResult | None
    """Results for confidence category."""

    # Answer Quality Categories
    grounding: FitzGovCategoryResult | None
    """Results for grounding category (no hallucination)."""

    relevance: FitzGovCategoryResult | None
    """Results for relevance category (answers the question)."""

    confusion_matrix: FitzGovConfusionMatrix
    """Mode confusion matrix (governance categories only)."""

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
            "overall_accuracy": self.overall_accuracy,
            "abstention": self.abstention.to_dict() if self.abstention else None,
            "dispute": self.dispute.to_dict() if self.dispute else None,
            "qualification": self.qualification.to_dict() if self.qualification else None,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "relevance": self.relevance.to_dict() if self.relevance else None,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "num_cases": self.num_cases,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitzGovResult:
        """Create from dictionary."""
        return cls(
            overall_accuracy=data["overall_accuracy"],
            abstention=(
                FitzGovCategoryResult.from_dict(data["abstention"])
                if data.get("abstention")
                else None
            ),
            dispute=(
                FitzGovCategoryResult.from_dict(data["dispute"]) if data.get("dispute") else None
            ),
            qualification=(
                FitzGovCategoryResult.from_dict(data["qualification"])
                if data.get("qualification")
                else None
            ),
            confidence=(
                FitzGovCategoryResult.from_dict(data["confidence"])
                if data.get("confidence")
                else None
            ),
            grounding=(
                FitzGovCategoryResult.from_dict(data["grounding"])
                if data.get("grounding")
                else None
            ),
            relevance=(
                FitzGovCategoryResult.from_dict(data["relevance"])
                if data.get("relevance")
                else None
            ),
            confusion_matrix=FitzGovConfusionMatrix.from_dict(data["confusion_matrix"]),
            num_cases=data["num_cases"],
            evaluation_time_seconds=data["evaluation_time_seconds"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        lines = [
            f"FITZ-GOV Results (n={self.num_cases}):",
            f"  Overall Accuracy: {self.overall_accuracy:.2%}",
            "",
            "Governance Mode Categories:",
        ]
        if self.abstention:
            lines.append(
                f"  Abstention: {self.abstention.accuracy:.2%} ({self.abstention.num_correct}/{self.abstention.num_total})"
            )
        if self.dispute:
            lines.append(
                f"  Dispute: {self.dispute.accuracy:.2%} ({self.dispute.num_correct}/{self.dispute.num_total})"
            )
        if self.qualification:
            lines.append(
                f"  Qualification: {self.qualification.accuracy:.2%} ({self.qualification.num_correct}/{self.qualification.num_total})"
            )
        if self.confidence:
            lines.append(
                f"  Confidence: {self.confidence.accuracy:.2%} ({self.confidence.num_correct}/{self.confidence.num_total})"
            )

        # Answer Quality Categories
        if self.grounding or self.relevance:
            lines.append("")
            lines.append("Answer Quality Categories:")
            if self.grounding:
                lines.append(
                    f"  Grounding: {self.grounding.accuracy:.2%} ({self.grounding.num_correct}/{self.grounding.num_total})"
                )
            if self.relevance:
                lines.append(
                    f"  Relevance: {self.relevance.accuracy:.2%} ({self.relevance.num_correct}/{self.relevance.num_total})"
                )

        lines.append("")
        lines.append(str(self.confusion_matrix))
        return "\n".join(lines)


class FitzGovBenchmark:
    """
    FITZ-GOV governance calibration benchmark.

    Tests Fitz's ability to correctly classify when to:
    - Abstain (insufficient evidence)
    - Dispute (conflicting sources)
    - Qualify (uncertain claims)
    - Be confident (clear evidence)

    Example:
        benchmark = FitzGovBenchmark()

        # Run full benchmark
        results = benchmark.evaluate(engine)

        # Run specific category
        results = benchmark.evaluate(engine, categories=[FitzGovCategory.ABSTENTION])

        # Generate new test cases from corpus
        cases = benchmark.generate_test_cases(corpus_chunks)
    """

    def __init__(self, data_dir: Path | str | None = None):
        """
        Initialize FITZ-GOV benchmark.

        Args:
            data_dir: Directory containing test case JSON files.
                     Defaults to bundled test data.
        """
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR

    def evaluate(
        self,
        engine: FitzRagEngine,
        categories: list[FitzGovCategory] | None = None,
        test_cases: list[FitzGovCase] | None = None,
    ) -> FitzGovResult:
        """
        Evaluate governance calibration.

        Args:
            engine: Fitz RAG engine to evaluate
            categories: Categories to test. Defaults to all.
            test_cases: Custom test cases. If not provided, loads from data_dir.

        Returns:
            FitzGovResult with accuracy and confusion matrix
        """
        import time

        start_time = time.time()

        # Load test cases
        if test_cases is None:
            test_cases = self._load_test_cases(categories)

        # Filter by category if specified
        if categories:
            test_cases = [c for c in test_cases if c.category in categories]

        # Group by category
        by_category: dict[FitzGovCategory, list[FitzGovCase]] = defaultdict(list)
        for case in test_cases:
            by_category[case.category].append(case)

        # Initialize confusion matrix
        confusion_matrix = FitzGovConfusionMatrix(matrix={})

        # Evaluate each category
        category_results: dict[FitzGovCategory, FitzGovCategoryResult] = {}
        for cat, cat_cases in by_category.items():
            cat_result = self._evaluate_category(engine, cat_cases, cat, confusion_matrix)
            category_results[cat] = cat_result

        evaluation_time = time.time() - start_time

        # Calculate overall accuracy
        total_correct = sum(r.num_correct for r in category_results.values())
        total_cases = sum(r.num_total for r in category_results.values())
        overall_accuracy = total_correct / total_cases if total_cases > 0 else 0.0

        return FitzGovResult(
            overall_accuracy=overall_accuracy,
            abstention=category_results.get(FitzGovCategory.ABSTENTION),
            dispute=category_results.get(FitzGovCategory.DISPUTE),
            qualification=category_results.get(FitzGovCategory.QUALIFICATION),
            confidence=category_results.get(FitzGovCategory.CONFIDENCE),
            grounding=category_results.get(FitzGovCategory.GROUNDING),
            relevance=category_results.get(FitzGovCategory.RELEVANCE),
            confusion_matrix=confusion_matrix,
            num_cases=len(test_cases),
            evaluation_time_seconds=evaluation_time,
        )

    # Categories that map to AnswerMode (contribute to confusion matrix)
    GOVERNANCE_MODE_CATEGORIES = {
        FitzGovCategory.ABSTENTION,
        FitzGovCategory.DISPUTE,
        FitzGovCategory.QUALIFICATION,
        FitzGovCategory.CONFIDENCE,
    }

    # Answer quality categories (don't use confusion matrix)
    ANSWER_QUALITY_CATEGORIES = {
        FitzGovCategory.GROUNDING,
        FitzGovCategory.RELEVANCE,
    }

    def _evaluate_category(
        self,
        engine: FitzRagEngine,
        cases: list[FitzGovCase],
        category: FitzGovCategory,
        confusion_matrix: FitzGovConfusionMatrix,
    ) -> FitzGovCategoryResult:
        """Evaluate a single category."""
        case_results: list[FitzGovCaseResult] = []
        subcategory_correct: dict[str, int] = defaultdict(int)
        subcategory_total: dict[str, int] = defaultdict(int)

        # Only governance mode categories contribute to confusion matrix
        is_governance_category = category in self.GOVERNANCE_MODE_CATEGORIES

        for case in cases:
            result = self._evaluate_case(engine, case)
            case_results.append(result)

            # Update confusion matrix only for governance mode categories
            if is_governance_category:
                confusion_matrix.add(case.expected_mode, result.actual_mode)

            # Track subcategory stats
            subcategory_total[case.subcategory] += 1
            if result.passed:
                subcategory_correct[case.subcategory] += 1

        # Calculate subcategory accuracy
        subcategory_accuracy = {
            subcat: subcategory_correct[subcat] / total
            for subcat, total in subcategory_total.items()
        }

        num_correct = sum(1 for r in case_results if r.passed)

        return FitzGovCategoryResult(
            category=category,
            accuracy=num_correct / len(cases) if cases else 0.0,
            num_correct=num_correct,
            num_total=len(cases),
            case_results=case_results,
            subcategory_accuracy=subcategory_accuracy,
        )

    def _evaluate_case(self, engine: FitzRagEngine, case: FitzGovCase) -> FitzGovCaseResult:
        """Evaluate a single test case."""
        from fitz_ai.core import Query

        query = Query(text=case.query)

        # Run with injected contexts (bypass retrieval for controlled testing)
        answer = self._run_with_contexts(engine, query, case.contexts)

        # Extract governance info
        actual_mode = self._get_answer_mode(answer)
        triggered = self._get_triggered_constraints(answer)

        # Evaluate based on category type
        if case.category == FitzGovCategory.GROUNDING:
            passed, failure_analysis = self._evaluate_grounding(answer.text, case)
        elif case.category == FitzGovCategory.RELEVANCE:
            passed, failure_analysis = self._evaluate_relevance(answer.text, case)
        else:
            # Governance mode categories: check mode match
            passed = actual_mode == case.expected_mode
            failure_analysis = None
            if not passed:
                failure_analysis = self._analyze_failure(case, actual_mode, triggered)

        return FitzGovCaseResult(
            case=case,
            passed=passed,
            answer=answer.text,
            actual_mode=actual_mode,
            triggered_constraints=triggered,
            failure_analysis=failure_analysis,
        )

    def _evaluate_grounding(self, answer_text: str, case: FitzGovCase) -> tuple[bool, str | None]:
        """Evaluate grounding: answer should not contain forbidden claims (hallucinations)."""
        answer_lower = answer_text.lower()
        found_hallucinations = []

        for claim in case.forbidden_claims:
            if claim.lower() in answer_lower:
                found_hallucinations.append(claim)

        if found_hallucinations:
            return False, f"HALLUCINATION: Found forbidden claims: {found_hallucinations}"
        return True, None

    def _evaluate_relevance(self, answer_text: str, case: FitzGovCase) -> tuple[bool, str | None]:
        """Evaluate relevance: answer should contain all required elements."""
        answer_lower = answer_text.lower()
        missing_elements = []

        for element in case.required_elements:
            if element.lower() not in answer_lower:
                missing_elements.append(element)

        if missing_elements:
            return False, f"OFF-TOPIC: Missing required elements: {missing_elements}"
        return True, None

    def _run_with_contexts(self, engine: FitzRagEngine, query, contexts: list[str]):
        """Run engine with injected contexts."""
        from fitz_ai.core import Chunk

        chunks = [
            Chunk(
                id=f"fitz_gov_context_{i}",
                text=ctx,
                source_file="fitz_gov_test",
                start_char=0,
                end_char=len(ctx),
            )
            for i, ctx in enumerate(contexts)
        ]

        pipeline = engine._pipeline
        governance = pipeline._get_governance_decision(query.text, chunks)
        answer = pipeline._generate_answer(query.text, chunks, governance)
        return answer

    def _get_answer_mode(self, answer) -> AnswerMode:
        """Extract answer mode from answer object."""
        if hasattr(answer, "mode"):
            return answer.mode
        if hasattr(answer, "governance") and hasattr(answer.governance, "mode"):
            return answer.governance.mode
        return AnswerMode.CONFIDENT

    def _get_triggered_constraints(self, answer) -> list[str]:
        """Extract triggered constraints from answer."""
        if hasattr(answer, "governance") and hasattr(answer.governance, "triggered_constraints"):
            return list(answer.governance.triggered_constraints)
        return []

    def _analyze_failure(
        self,
        case: FitzGovCase,
        actual_mode: AnswerMode,
        triggered: list[str],
    ) -> str:
        """Generate failure analysis."""
        lines = [
            f"Expected {case.expected_mode.value}, got {actual_mode.value}",
            f"Triggered constraints: {triggered or 'none'}",
        ]

        # Category-specific analysis
        if case.category == FitzGovCategory.ABSTENTION:
            if actual_mode == AnswerMode.CONFIDENT:
                lines.append("OVER-CONFIDENT: Answered when should have abstained")
            elif actual_mode == AnswerMode.QUALIFIED:
                lines.append("PARTIAL: Qualified but should have fully abstained")

        elif case.category == FitzGovCategory.DISPUTE:
            if actual_mode == AnswerMode.CONFIDENT:
                lines.append("MISSED CONFLICT: Did not detect contradicting sources")
            elif actual_mode == AnswerMode.ABSTAIN:
                lines.append("OVER-CAUTIOUS: Abstained instead of reporting dispute")

        elif case.category == FitzGovCategory.CONFIDENCE:
            if actual_mode == AnswerMode.ABSTAIN:
                lines.append("UNDER-CONFIDENT: Abstained when evidence was clear")
            elif actual_mode == AnswerMode.DISPUTED:
                lines.append("FALSE CONFLICT: Saw conflict where none exists")

        return "\n".join(lines)

    def _load_test_cases(
        self,
        categories: list[FitzGovCategory] | None = None,
    ) -> list[FitzGovCase]:
        """Load test cases from data directory."""
        cases: list[FitzGovCase] = []

        if not self._data_dir.exists():
            logger.warning(f"FITZ-GOV data directory not found: {self._data_dir}")
            return cases

        # Map categories to directory names
        category_dirs = {
            # Governance mode categories
            FitzGovCategory.ABSTENTION: "abstention",
            FitzGovCategory.DISPUTE: "dispute",
            FitzGovCategory.QUALIFICATION: "qualification",
            FitzGovCategory.CONFIDENCE: "confidence",
            # Answer quality categories
            FitzGovCategory.GROUNDING: "grounding",
            FitzGovCategory.RELEVANCE: "relevance",
        }

        target_categories = categories or list(FitzGovCategory)

        for cat in target_categories:
            cat_dir = self._data_dir / category_dirs[cat]
            if not cat_dir.exists():
                continue

            for json_file in cat_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    for case_data in data.get("cases", []):
                        case_data["category"] = cat.value
                        case_data["subcategory"] = json_file.stem
                        cases.append(FitzGovCase.from_dict(case_data))

                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

        return cases

    def save_results(self, result: FitzGovResult, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved FITZ-GOV results to {path}")

    def load_results(self, path: Path | str) -> FitzGovResult:
        """Load results from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return FitzGovResult.from_dict(data)

    @staticmethod
    def get_available_categories() -> list[str]:
        """Get list of available categories."""
        return [c.value for c in FitzGovCategory]


class FitzGovDatasetGenerator:
    """
    Generate FITZ-GOV test cases from an existing corpus.

    Uses LLM to identify scenarios that should trigger different
    governance modes, creating a custom test suite for your data.
    """

    def __init__(self, chat_client: Any):
        """
        Initialize generator.

        Args:
            chat_client: Chat client for LLM-based case generation
        """
        self._chat = chat_client

    def generate_abstention_cases(
        self,
        chunks: list[Any],
        num_cases: int = 10,
    ) -> list[FitzGovCase]:
        """
        Generate abstention test cases.

        Creates questions that the corpus cannot answer, paired with
        chunks that are tangentially related but insufficient.
        """
        cases: list[FitzGovCase] = []

        # Use LLM to generate out-of-scope questions based on corpus themes
        prompt = f"""Given these document excerpts, generate {num_cases} questions that:
1. Are related to the general topic
2. Cannot be answered by the provided information
3. Would require external knowledge

Excerpts:
{self._format_chunks(chunks[:5])}

Return as JSON: {{"questions": ["q1", "q2", ...]}}"""

        response = self._chat.complete(prompt)
        questions = self._parse_json_response(response).get("questions", [])

        for i, q in enumerate(questions[:num_cases]):
            # Pair with random chunks that won't answer the question
            contexts = [c.text for c in chunks[:3]]
            cases.append(
                FitzGovCase(
                    id=f"gen_abstain_{i}",
                    category=FitzGovCategory.ABSTENTION,
                    subcategory="generated_out_of_scope",
                    query=q,
                    contexts=contexts,
                    expected_mode=AnswerMode.ABSTAIN,
                    description=f"Generated out-of-scope question: {q[:50]}...",
                    rationale="Question cannot be answered from provided context",
                )
            )

        return cases

    def generate_dispute_cases(
        self,
        chunks: list[Any],
        num_cases: int = 10,
    ) -> list[FitzGovCase]:
        """
        Generate dispute test cases by finding or creating contradictions.
        """
        cases: list[FitzGovCase] = []

        # Find pairs of chunks that might have conflicting info
        prompt = f"""Analyze these document excerpts for potential contradictions or conflicting information.
Generate {num_cases} questions where the excerpts provide conflicting answers.

Excerpts:
{self._format_chunks(chunks[:10])}

Return as JSON: {{"cases": [{{"question": "...", "chunk_indices": [0, 3], "conflict": "..."}}]}}"""

        response = self._chat.complete(prompt)
        generated = self._parse_json_response(response).get("cases", [])

        for i, case_data in enumerate(generated[:num_cases]):
            indices = case_data.get("chunk_indices", [0, 1])
            contexts = [chunks[idx].text for idx in indices if idx < len(chunks)]

            cases.append(
                FitzGovCase(
                    id=f"gen_dispute_{i}",
                    category=FitzGovCategory.DISPUTE,
                    subcategory="generated_conflict",
                    query=case_data["question"],
                    contexts=contexts,
                    expected_mode=AnswerMode.DISPUTED,
                    description=f"Generated conflict case: {case_data.get('conflict', '')}",
                    rationale="Sources provide conflicting information",
                )
            )

        return cases

    def _format_chunks(self, chunks: list[Any]) -> str:
        """Format chunks for prompt."""
        lines = []
        for i, chunk in enumerate(chunks):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            lines.append(f"[{i}] {text[:500]}...")
        return "\n\n".join(lines)

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from response
            import re

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}
