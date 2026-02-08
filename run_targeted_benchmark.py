# run_targeted_benchmark.py
"""
Run fitz-gov 2.0 benchmark.

Usage:
    python run_targeted_benchmark.py              # Governance only (249 cases)
    python run_targeted_benchmark.py --full        # Governance + relevance (289 cases)
    python run_targeted_benchmark.py --grounding   # Grounding text quality (42 cases)
"""

import json
import re
import sys
import time
from collections import Counter

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.core.chunk import Chunk
from fitz_ai.core.governance import AnswerGovernor
from fitz_ai.core.guardrails.plugins.causal_attribution import CausalAttributionConstraint
from fitz_ai.core.guardrails.plugins.conflict_aware import ConflictAwareConstraint
from fitz_ai.core.guardrails.plugins.insufficient_evidence import (
    InsufficientEvidenceConstraint,
)
from fitz_ai.core.guardrails.plugins.specific_info_type import SpecificInfoTypeConstraint
from fitz_ai.core.guardrails.staged import run_staged_constraints
from fitz_ai.llm import get_embedder
from fitz_ai.llm.factory import get_chat_factory
from fitz_gov import load_cases
from fitz_gov.models import FitzGovCategory

# Mode mapping
MODE_MAP = {
    "abstain": AnswerMode.ABSTAIN,
    "disputed": AnswerMode.DISPUTED,
    "qualified": AnswerMode.QUALIFIED,
    "confident": AnswerMode.CONFIDENT,
}

# Governance categories (core 4)
GOV_CATEGORIES = {
    FitzGovCategory.ABSTENTION,
    FitzGovCategory.DISPUTE,
    FitzGovCategory.QUALIFICATION,
    FitzGovCategory.CONFIDENCE,
}

# Governance + relevance (relevance tests mode classification, not text quality)
MODE_TESTABLE_CATEGORIES = GOV_CATEGORIES | {
    FitzGovCategory.RELEVANCE,
}

# Grounding prompt: ask LLM to answer the query using only the context
_GROUNDING_PROMPT = """Answer the following question using ONLY the provided context.
If the context does not contain the specific information needed, say so clearly.

Context:
{context}

Question: {query}

Answer:"""


def _check_grounding(case, response):
    """Check if response contains forbidden claims (hallucinations).

    Returns (passed, violations) tuple.
    Uses the same logic as fitz-gov's _evaluate_grounding.
    """
    eval_config = getattr(case, "evaluation_config", {})
    use_regex = eval_config.get("use_regex", False)
    case_insensitive = eval_config.get("case_insensitive", True)
    allowed_phrases = eval_config.get("allowed_phrases", [])

    regex_flags = re.IGNORECASE if case_insensitive else 0

    # Check if response matches any allowed phrase (e.g., "not specified")
    for allowed in allowed_phrases:
        try:
            if re.search(allowed, response, regex_flags):
                return True, []
        except re.error:
            if allowed.lower() in response.lower():
                return True, []

    # Check for forbidden claims
    violations = []
    for pattern in case.forbidden_claims:
        try:
            if use_regex:
                matches = list(re.finditer(pattern, response, regex_flags))
            else:
                search_text = response.lower() if case_insensitive else response
                search_pattern = pattern.lower() if case_insensitive else pattern
                if search_pattern in search_text:
                    matches = [pattern]
                else:
                    matches = []

            if matches:
                matched = matches[0].group() if hasattr(matches[0], "group") else pattern
                violations.append({"pattern": pattern, "matched": matched})
        except re.error:
            if pattern.lower() in response.lower():
                violations.append({"pattern": pattern, "matched": pattern})

    return len(violations) == 0, violations


def run_grounding(fast_chat):
    """Run grounding benchmark: generate answers, check for hallucinations."""
    all_cases = load_cases()
    cases = [c for c in all_cases if c.category == FitzGovCategory.GROUNDING]
    print(f"Running {len(cases)} grounding cases (text quality test)\n")

    correct = 0
    total = 0
    failures = []

    for case in cases:
        context = "\n\n---\n\n".join(case.contexts)
        prompt = _GROUNDING_PROMPT.format(context=context, query=case.query)

        try:
            response = fast_chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as e:
            response = f"Error: {e}"

        passed, violations = _check_grounding(case, response)
        total += 1

        if passed:
            correct += 1
        else:
            violation_strs = [v["matched"] for v in violations]
            print(f"  FAIL [{case.id}] {getattr(case, 'subcategory', '?')}")
            print(f"    Q: {case.query}")
            print(f"    Hallucinated: {violation_strs}")
            print(f"    Response: {response[:200]}")
            print()
            failures.append({
                "id": case.id,
                "sub": getattr(case, "subcategory", "?"),
                "query": case.query,
                "violations": violation_strs,
                "response": response[:500],
            })

    pct = correct / total * 100 if total else 0
    print(f"\nGrounding: {pct:.1f}% ({correct}/{total})")

    if failures:
        with open("grounding_failures.json", "w") as fh:
            json.dump(failures, fh, indent=2)
        print(f"Failures written to grounding_failures.json")

    return correct, total


def run_governance(full_mode, fast_chat, embedder):
    """Run governance mode classification benchmark."""
    all_cases = load_cases()
    target_categories = MODE_TESTABLE_CATEGORIES if full_mode else GOV_CATEGORIES
    cases = [c for c in all_cases if c.category in target_categories]
    mode_label = "governance + relevance (289 cases)" if full_mode else "governance only"
    print(f"Running {len(cases)} cases ({mode_label})\n")

    constraints = [
        InsufficientEvidenceConstraint(embedder=embedder, chat=fast_chat),
        SpecificInfoTypeConstraint(),
        CausalAttributionConstraint(),
        ConflictAwareConstraint(
            chat=fast_chat, use_fusion=True, adaptive=True, embedder=embedder
        ),
    ]

    governor = AnswerGovernor()

    correct = 0
    total = 0
    cat_correct = Counter()
    cat_total = Counter()
    failures = []
    transition_counts = Counter()

    for case in cases:
        chunks = [
            Chunk(
                id=f"ctx_{i}",
                doc_id="bench",
                content=ctx,
                chunk_index=i,
                metadata={"source": "fitz_gov"},
            )
            for i, ctx in enumerate(case.contexts)
        ]

        results = run_staged_constraints(case.query, chunks, constraints)
        decision = governor.decide(results)
        actual = decision.mode
        expected = MODE_MAP.get(case.expected_mode.value)

        cat = case.category.value
        cat_total[cat] += 1
        total += 1

        if actual == expected:
            correct += 1
            cat_correct[cat] += 1
        else:
            expected_str = expected.value if expected else "?"
            actual_str = actual.value

            triggered = []
            details = {}
            for r in results:
                name = r.metadata.get("constraint_name", "unknown")
                info = {
                    "denied": not r.allow_decisive_answer,
                    "signal": r.signal,
                    "reason": r.reason,
                }
                details[name] = info
                if not r.allow_decisive_answer:
                    triggered.append(f"{name}:{r.signal}")

            transition = f"{expected_str}->{actual_str}"
            transition_counts[transition] += 1

            failures.append({
                "id": case.id,
                "cat": cat,
                "sub": getattr(case, "subcategory", "?"),
                "expected": expected_str,
                "actual": actual_str,
                "query": case.query,
                "triggered": triggered,
                "details": {
                    name: {
                        "denied": info["denied"],
                        "signal": info["signal"],
                        "reason": info["reason"],
                    }
                    for name, info in details.items()
                },
            })

    pct = correct / total * 100 if total > 0 else 0
    print(f"Overall: {pct:.1f}% ({correct}/{total})\n")

    print("Per-category:")
    all_cats = sorted(cat_total.keys())
    for cat in all_cats:
        c = cat_correct[cat]
        t = cat_total[cat]
        p = c / t * 100 if t > 0 else 0
        print(f"  {cat:15s}:  {p:.1f}% ({c}/{t})")

    print(f"\nFailure transitions:")
    for transition, count in transition_counts.most_common():
        print(f"  {transition:30s}: {count}")

    print(f"\nSample failures (first 10):")
    for f in failures[:10]:
        triggered_str = ", ".join(f["triggered"]) if f["triggered"] else "(none fired)"
        print(f"  [{f['id']}] {f['expected']}->{f['actual']}  |  {triggered_str}")
        print(f"    Q: {f['query']}")

    output_file = "failure_analysis_full.json" if full_mode else "failure_analysis.json"
    with open(output_file, "w") as fh:
        json.dump(failures, fh, indent=2)
    print(f"\nAll {len(failures)} failures written to {output_file}")

    return correct, total


def main():
    grounding_mode = "--grounding" in sys.argv
    full_mode = "--full" in sys.argv
    start = time.time()

    chat_factory = get_chat_factory("ollama/qwen2.5:3b")
    fast_chat = chat_factory("fast")

    if grounding_mode:
        run_grounding(fast_chat)
    else:
        embedding = get_embedder("ollama")
        embedder = embedding.embed
        run_governance(full_mode, fast_chat, embedder)

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
