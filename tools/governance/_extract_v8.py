# tools/governance/_extract_v8.py
"""One-off script: re-extract features for v8 CSV with ctx_* + temporal features."""
import json
import re
import statistics
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants (same as feature_extractor.py)
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "then", "than",
    "that", "this", "these", "those", "what", "which", "who",
    "how", "when", "where", "why", "it", "its",
}
HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "likely", "unlikely",
    "sometimes", "often", "typically", "generally", "usually", "probably",
    "approximately", "roughly", "about", "around", "estimated", "suggests",
    "appears", "seems", "potentially", "tends",
}
ASSERTION_WORDS = {
    "always", "never", "must", "certainly", "definitely", "clearly",
    "obviously", "undoubtedly", "absolutely", "exactly", "precisely",
    "proven", "confirmed", "established", "demonstrates", "proves",
    "invariably", "unquestionably",
}
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b")
PLAIN_NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
CONTRADICTION_MARKERS = [
    "however", "but", "although", "contrary", "disagree", "whereas",
    "nevertheless", "conversely", "despite", "in contrast", "on the other hand",
    "contradicts", "inconsistent", "conflicts with", "differs from",
]
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nothing",
    "hardly", "barely", "scarcely", "doesn't", "don't", "isn't",
    "wasn't", "weren't", "won't", "can't", "couldn't", "shouldn't",
}


def compute_all_chunk_features(contexts: list[str]) -> dict:
    defaults = {
        "max_pairwise_overlap": 0.0, "min_pairwise_overlap": 0.0,
        "chunk_length_cv": 0.0, "assertion_density": 0.0, "number_density": 0.0,
        "ctx_length_mean": 0.0, "ctx_length_std": 0.0, "ctx_total_chars": 0.0,
        "ctx_contradiction_count": 0.0, "ctx_negation_count": 0.0,
        "ctx_number_count": 0.0, "ctx_number_variance": 0.0,
        "ctx_max_pairwise_sim": 0.0, "ctx_mean_pairwise_sim": 0.0,
        "ctx_min_pairwise_sim": 0.0,
        "year_count": 0, "has_distinct_years": False,
    }
    if len(contexts) < 2:
        return defaults

    chunk_word_sets = []
    chunk_lengths = []
    char_lengths = []
    total_hedge = 0
    total_assert = 0
    total_numbers = 0
    all_text_lower = ""
    all_numbers: list[float] = []
    all_years: set[str] = set()

    for text in contexts:
        text_lower = text.lower()
        all_text_lower += " " + text_lower
        words = text_lower.split()
        content_words = set(words) - STOP_WORDS
        chunk_word_sets.append(content_words)
        chunk_lengths.append(len(words))
        char_lengths.append(len(text))
        word_set = set(words)
        total_hedge += len(word_set & HEDGE_WORDS)
        total_assert += len(word_set & ASSERTION_WORDS)
        total_numbers += len(NUMBER_RE.findall(text))
        for m in PLAIN_NUMBER_RE.finditer(text):
            try:
                all_numbers.append(float(m.group(0)))
            except ValueError:
                pass
        for ym in YEAR_RE.finditer(text):
            all_years.add(ym.group(0))

    # Jaccard overlaps
    overlaps = []
    for i in range(len(chunk_word_sets)):
        for j in range(i + 1, len(chunk_word_sets)):
            a, b = chunk_word_sets[i], chunk_word_sets[j]
            union = a | b
            overlaps.append(len(a & b) / len(union) if union else 0.0)

    feats: dict = {}
    feats["max_pairwise_overlap"] = max(overlaps) if overlaps else 0.0
    feats["min_pairwise_overlap"] = min(overlaps) if overlaps else 0.0

    mean_len = statistics.mean(chunk_lengths)
    feats["chunk_length_cv"] = (
        statistics.stdev(chunk_lengths) / mean_len
        if mean_len > 0 and len(chunk_lengths) > 1 else 0.0
    )

    ep = total_assert + total_hedge
    feats["assertion_density"] = (total_assert - total_hedge) / ep if ep > 0 else 0.0
    feats["number_density"] = total_numbers / len(contexts)

    feats["ctx_length_mean"] = statistics.mean(char_lengths)
    feats["ctx_length_std"] = statistics.pstdev(char_lengths)
    feats["ctx_total_chars"] = sum(char_lengths)

    feats["ctx_contradiction_count"] = float(
        sum(1 for m in CONTRADICTION_MARKERS if m in all_text_lower)
    )
    all_words = all_text_lower.split()
    feats["ctx_negation_count"] = float(
        sum(1 for w in all_words if w in NEGATION_WORDS)
    )

    feats["ctx_number_count"] = float(len(all_numbers))
    if len(all_numbers) > 1:
        mean_n = sum(all_numbers) / len(all_numbers)
        feats["ctx_number_variance"] = float(
            sum((x - mean_n) ** 2 for x in all_numbers) / len(all_numbers)
        )
    else:
        feats["ctx_number_variance"] = 0.0

    try:
        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        matrix = tfidf.fit_transform(contexts)
        sim_matrix = cosine_similarity(matrix)
        n = sim_matrix.shape[0]
        pairwise_sims = [
            float(sim_matrix[i, j]) for i in range(n) for j in range(i + 1, n)
        ]
        feats["ctx_max_pairwise_sim"] = max(pairwise_sims)
        feats["ctx_mean_pairwise_sim"] = sum(pairwise_sims) / len(pairwise_sims)
        feats["ctx_min_pairwise_sim"] = min(pairwise_sims)
    except Exception:
        feats["ctx_max_pairwise_sim"] = 0.0
        feats["ctx_mean_pairwise_sim"] = 0.0
        feats["ctx_min_pairwise_sim"] = 0.0

    feats["year_count"] = len(all_years)
    feats["has_distinct_years"] = len(all_years) > 1

    return feats


def main():
    df = pd.read_csv("tools/governance/data/eval_results_v5_clean.csv")
    print(f"Loaded {len(df)} cases from existing CSV ({len(df.columns)} cols)")

    data_dir = Path("../fitz-gov/data/tier1_core")
    case_data = {}
    for path in sorted(data_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data["cases"]:
            case_data[case["id"]] = case

    new_features = []
    for _, row in df.iterrows():
        case = case_data[row["case_id"]]
        contexts = case.get("contexts", [])
        feats = compute_all_chunk_features(contexts)
        new_features.append(feats)

    new_df = pd.DataFrame(new_features)
    result = pd.concat([df, new_df], axis=1)
    out_path = "tools/governance/data/eval_results_v8_full.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} rows x {len(result.columns)} cols to {out_path}")

    print("\nNew feature stats:")
    for col in new_df.columns:
        if new_df[col].dtype in ["float64", "int64", "bool"]:
            print(f"  {col:>25s}: mean={new_df[col].mean():.4f} std={new_df[col].std():.4f}")
        else:
            print(f"  {col:>25s}: {dict(new_df[col].value_counts())}")


if __name__ == "__main__":
    main()
