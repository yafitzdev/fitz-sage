# tools/governance/diagnose_av_votes.py
"""
Quick diagnostic: run AnswerVerification jury on a sample of cases
and log the actual vote distribution (0, 1, 2, 3).

Usage:
    python -m tools.governance.diagnose_av_votes --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from fitz_ai.config import load_engine_config
from fitz_ai.core.chunk import Chunk
from fitz_ai.governance.constraints.plugins.answer_verification import (
    AnswerVerificationConstraint,
)
from fitz_ai.llm import get_chat_factory

_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
)

# Balanced-tier confirmation prompt (runs only when 2+/3 fast prompts say NO)
BALANCED_CONFIRM_PROMPT = """You are verifying whether retrieved context is relevant to a user's question.

Question: {query}
Context: {context}

Does the context contain information that is relevant and useful for answering this question?

Answer YES if the context has any relevant information, even partial.
Answer NO if the context is completely irrelevant and contains nothing useful for the question.

Reply with ONE word: YES or NO"""


def load_cases(data_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(data_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data["cases"]:
            cases.append(case)
    return cases


def case_to_chunks(case: dict) -> list[Chunk]:
    case_id = case["id"]
    chunks = []
    for i, ctx in enumerate(case.get("contexts", [])):
        chunks.append(
            Chunk(
                id=f"{case_id}_ctx_{i}",
                doc_id=case_id,
                content=ctx,
                chunk_index=i,
                metadata={"source_file": f"case_{case_id}"},
            )
        )
    return chunks


def run_balanced_confirm(chat_balanced, query: str, context: str) -> str:
    """Run a single balanced-tier confirmation call. Returns YES/NO."""
    prompt = BALANCED_CONFIRM_PROMPT.format(query=query, context=context)
    try:
        response = chat_balanced.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.strip().upper()
    except Exception as e:
        return "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--chat", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    cfg = load_engine_config("fitz_krag")
    chat_spec = args.chat or cfg.chat
    chat_config = {k: v for k, v in cfg.chat_kwargs.model_dump().items() if v is not None}
    chat_factory = get_chat_factory(chat_spec, config=chat_config)
    chat_fast = chat_factory("fast")
    chat_balanced = chat_factory("balanced")
    print(f"Fast chat: {chat_spec} (model: {getattr(chat_fast, '_model', '?')})")
    print(f"Balanced chat: {chat_spec} (model: {getattr(chat_balanced, '_model', '?')})")

    all_cases = load_cases(args.data_dir)

    # Stratified sampling: equal number per mode
    if args.limit > 0:
        by_mode_list = {}
        for c in all_cases:
            by_mode_list.setdefault(c["expected_mode"], []).append(c)
        per_mode = args.limit // len(by_mode_list)
        cases = []
        for mode_name, mode_cases in sorted(by_mode_list.items()):
            random.seed(42)
            random.shuffle(mode_cases)
            cases.extend(mode_cases[:per_mode])
        random.seed(42)
        random.shuffle(cases)
        print(f"Stratified sample: {per_mode} per mode from {len(by_mode_list)} modes")
    else:
        cases = all_cases
    print(f"Running AV jury on {len(cases)} cases...\n")

    av = AnswerVerificationConstraint(chat=chat_fast)

    vote_dist = Counter()
    by_mode = {"abstain": Counter(), "disputed": Counter(), "trustworthy": Counter()}
    prompt_no = {"abstain": [0, 0, 0], "disputed": [0, 0, 0], "trustworthy": [0, 0, 0]}
    prompt_total = {"abstain": 0, "disputed": 0, "trustworthy": 0}
    misfires_at_2 = []

    # Conditional balanced confirmation tracking
    balanced_calls = 0
    balanced_confirmed = {"abstain": 0, "disputed": 0, "trustworthy": 0}
    balanced_rejected = {"abstain": 0, "disputed": 0, "trustworthy": 0}
    balanced_misfires = []  # confirmed non-abstain cases

    for case in tqdm(cases, desc="AV jury"):
        query = case["query"]
        chunks = case_to_chunks(case)
        mode = case["expected_mode"]

        # Run jury directly to get vote count
        context_parts = []
        total_chars = 0
        for chunk in chunks[:3]:
            remaining = 1000 - total_chars
            if remaining <= 0:
                break
            content = chunk.content[:remaining]
            context_parts.append(content)
            total_chars += len(content)
        context = "\n\n---\n\n".join(context_parts)

        no_votes, responses = av._run_jury(query, context)
        vote_dist[no_votes] += 1
        by_mode[mode][no_votes] += 1

        # Track per-prompt NO votes (all: NO = not relevant)
        prompt_total[mode] += 1
        for pi, resp in enumerate(responses):
            word = resp.strip().upper()
            if word.startswith("NO"):
                prompt_no[mode][pi] += 1

        if no_votes >= 2 and mode != "abstain":
            misfires_at_2.append({
                "case_id": case["id"],
                "mode": mode,
                "votes": no_votes,
                "responses": responses,
                "query": query[:80],
            })

        # Conditional balanced confirmation: only when 2+/3 fast say NO
        if no_votes >= 2:
            balanced_calls += 1
            confirm_resp = run_balanced_confirm(chat_balanced, query, context)
            if confirm_resp.startswith("NO"):
                # Balanced also says NO -> confirmed not relevant
                balanced_confirmed[mode] += 1
                if mode != "abstain":
                    balanced_misfires.append({
                        "case_id": case["id"],
                        "mode": mode,
                        "fast_votes": no_votes,
                        "fast_responses": responses,
                        "balanced_response": confirm_resp,
                        "query": query[:80],
                    })
            else:
                balanced_rejected[mode] += 1

    print("\n" + "=" * 60)
    print("VOTE DISTRIBUTION (all cases)")
    print("=" * 60)
    for votes in sorted(vote_dist.keys()):
        pct = 100 * vote_dist[votes] / len(cases)
        print(f"  {votes}/3 NO votes: {vote_dist[votes]:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("VOTE DISTRIBUTION BY EXPECTED MODE")
    print("=" * 60)
    for mode in ["abstain", "disputed", "trustworthy"]:
        total = sum(by_mode[mode].values())
        if total == 0:
            continue
        print(f"\n  {mode} (n={total}):")
        for votes in sorted(by_mode[mode].keys()):
            pct = 100 * by_mode[mode][votes] / total
            print(f"    {votes}/3 NO: {by_mode[mode][votes]:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("PER-PROMPT NO RATE BY MODE")
    print("=" * 60)
    prompt_names = ["P1 (entity/concept mention)", "P2 (any info)", "P3 (meaningful connection)"]
    for mode in ["abstain", "disputed", "trustworthy"]:
        total = prompt_total[mode]
        if total == 0:
            continue
        print(f"\n  {mode} (n={total}):")
        for pi in range(3):
            rate = 100 * prompt_no[mode][pi] / total
            print(f"    {prompt_names[pi]}: {prompt_no[mode][pi]}/{total} ({rate:.1f}%)")

    print("\n" + "=" * 60)
    print("FAST-ONLY: MISFIRE ANALYSIS AT 2/3 THRESHOLD")
    print("=" * 60)
    total_2plus = sum(vote_dist[v] for v in vote_dist if v >= 2)
    print(f"  Total 2+ vote cases: {total_2plus}")
    print(f"  Misfires (2+ votes, not abstain): {len(misfires_at_2)}")

    print("\n" + "=" * 60)
    print("BALANCED CONFIRMATION (2/3 fast -> 1 balanced)")
    print("=" * 60)
    print(f"  Balanced calls made: {balanced_calls}/{len(cases)} ({100*balanced_calls/len(cases):.1f}%)")
    print(f"\n  Confirmed (balanced also said NO):")
    for mode in ["abstain", "disputed", "trustworthy"]:
        n = sum(by_mode[mode].get(v, 0) for v in by_mode[mode] if v >= 2)
        conf = balanced_confirmed[mode]
        rej = balanced_rejected[mode]
        if n > 0:
            print(f"    {mode}: {conf}/{n} confirmed ({100*conf/n:.1f}%), {rej}/{n} rejected ({100*rej/n:.1f}%)")
        else:
            print(f"    {mode}: 0 cases reached balanced stage")

    total_confirmed = sum(balanced_confirmed.values())
    total_misfires = len(balanced_misfires)
    print(f"\n  Final fire rate (2/3 fast + balanced NO):")
    for mode in ["abstain", "disputed", "trustworthy"]:
        rate = 100 * balanced_confirmed[mode] / prompt_total[mode] if prompt_total[mode] else 0
        print(f"    {mode}: {balanced_confirmed[mode]}/100 ({rate:.1f}%)")
    print(f"\n  Total confirmed: {total_confirmed}")
    print(f"  Misfires (confirmed non-abstain): {total_misfires}")
    if balanced_misfires:
        print(f"\n  Misfire details:")
        for mf in balanced_misfires[:20]:
            print(f"    [{mf['mode']}] fast={mf['fast_votes']}/3 balanced={mf['balanced_response']} | {mf['case_id']} | {mf['query']}")
            print(f"      fast responses: {mf['fast_responses']}")


if __name__ == "__main__":
    main()
