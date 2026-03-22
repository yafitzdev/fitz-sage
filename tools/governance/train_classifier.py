# tools/governance/train_classifier.py
"""
Train governance classifier from extracted features.

Loads features.csv, enriches with text-based context features from fitz-gov
cases, trains multiple models with class weighting and hyperparameter search,
builds stacking ensemble, evaluates against governor baseline.

Supports two modes:
  - 4-class: abstain/confident/disputed/qualified (legacy)
  - twostage: two binary classifiers (answerable-vs-abstain → trustworthy-vs-disputed)

Usage:
    python -m tools.governance.train_classifier
    python -m tools.governance.train_classifier --mode twostage
    python -m tools.governance.train_classifier --time-budget 600  # 10 min search
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_INPUT = _DATA_DIR / "features.csv"
_DEFAULT_MODEL_OUTPUT = _DATA_DIR / "model_v1.joblib"
_DEFAULT_TWOSTAGE_OUTPUT = _DATA_DIR / "model_v5_twostage.joblib"
_DEFAULT_CALIBRATED_OUTPUT = _DATA_DIR / "model_v5_calibrated.joblib"
_DEFAULT_FITZGOV_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
)

# Metadata columns (not features)
_META_COLS = {
    "case_id",
    "expected_mode",
    "governor_predicted",
    "classifier_predicted",
    "difficulty",
    "subcategory",
}

# Feature type sets derived from constraint schemas (single source of truth)
from fitz_ai.governance.constraints.feature_extractor import get_feature_type_sets  # noqa: E402

_CATEGORICAL_FEATURES, _BOOL_FEATURES = get_feature_type_sets()

# Dead features: constant zero/single-value across all cases — adds noise
# num_unique_sources: always 1 in fitz-gov (single doc_id per case)
# ie_max_similarity: 100% identical to max_vector_score (redundant)
# ie_summary_overlap: always False (fitz-gov chunks lack enrichment metadata)
_DEAD_FEATURES = {"num_unique_sources", "ie_max_similarity", "ie_summary_overlap"}

# Markers for context feature extraction
_CONTRADICTION_MARKERS = [
    "however",
    "but",
    "although",
    "contrary",
    "disagree",
    "whereas",
    "nevertheless",
    "conversely",
    "despite",
    "in contrast",
    "on the other hand",
    "contradicts",
    "inconsistent",
    "conflicts with",
    "differs from",
]
_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nothing",
    "hardly",
    "barely",
    "scarcely",
    "doesn't",
    "don't",
    "isn't",
    "wasn't",
    "weren't",
    "won't",
    "can't",
    "couldn't",
    "shouldn't",
}
# Stronger markers that signal opposing conclusions (not just hedging)
_OPPOSING_MARKERS = [
    "contradicts",
    "inconsistent",
    "conflicts with",
    "disagree",
    "disputed",
    "refuted",
    "disproven",
    "challenged by",
    "rejected by",
    "opposes",
    "incompatible",
    "at odds with",
    "mutually exclusive",
    "cannot both be true",
    "conflicting evidence",
    "conflicting data",
    "conflicting reports",
]


# ---------------------------------------------------------------------------
# Context feature computation (text-based, no LLM)
# ---------------------------------------------------------------------------


def load_cases_by_id(data_dir: Path) -> dict[str, dict]:
    """Load fitz-gov cases indexed by case_id."""
    cases = {}
    for path in sorted(data_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data["cases"]:
            cases[case["id"]] = case
    return cases


def compute_context_features(query: str, contexts: list[str]) -> dict[str, float]:
    """Compute text-based features from query and raw contexts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    features: dict[str, float] = {}

    # Context length stats
    if contexts:
        lengths = [len(c) for c in contexts]
        features["ctx_length_mean"] = float(np.mean(lengths))
        features["ctx_length_std"] = float(np.std(lengths))
        features["ctx_total_chars"] = float(sum(lengths))
    else:
        features["ctx_length_mean"] = 0.0
        features["ctx_length_std"] = 0.0
        features["ctx_total_chars"] = 0.0

    # Contradiction and negation markers
    all_text_lower = " ".join(contexts).lower()
    features["ctx_contradiction_count"] = float(
        sum(1 for m in _CONTRADICTION_MARKERS if m in all_text_lower)
    )
    words = all_text_lower.split()
    features["ctx_negation_count"] = float(sum(1 for w in words if w in _NEGATION_WORDS))
    # Opposing conclusion markers (stronger than contradiction markers)
    features["opposing_conclusion_count"] = float(
        sum(1 for m in _OPPOSING_MARKERS if m in all_text_lower)
    )

    # Numerical features
    numbers = [float(x) for x in re.findall(r"\b\d+\.?\d*\b", all_text_lower)]
    features["ctx_number_count"] = float(len(numbers))
    features["ctx_number_variance"] = float(np.var(numbers)) if len(numbers) > 1 else 0.0

    # Pairwise context similarity (TF-IDF cosine)
    if len(contexts) >= 2:
        try:
            tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            matrix = tfidf.fit_transform(contexts)
            sim_matrix = cosine_similarity(matrix)
            n = sim_matrix.shape[0]
            pairwise_sims = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
            features["ctx_max_pairwise_sim"] = float(max(pairwise_sims))
            features["ctx_mean_pairwise_sim"] = float(np.mean(pairwise_sims))
            features["ctx_min_pairwise_sim"] = float(min(pairwise_sims))
        except Exception:
            features["ctx_max_pairwise_sim"] = 0.0
            features["ctx_mean_pairwise_sim"] = 0.0
            features["ctx_min_pairwise_sim"] = 0.0
    else:
        features["ctx_max_pairwise_sim"] = 0.0
        features["ctx_mean_pairwise_sim"] = 0.0
        features["ctx_min_pairwise_sim"] = 0.0

    # Query-context word overlap (excluding stopwords)
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "what",
        "how",
        "does",
        "do",
        "did",
        "in",
        "of",
        "to",
        "for",
        "and",
        "or",
        "this",
        "that",
        "with",
    }
    query_words = set(query.lower().split()) - stopwords
    if query_words and contexts:
        ctx_words = set(all_text_lower.split())
        features["query_ctx_content_overlap"] = len(query_words & ctx_words) / len(query_words)
    else:
        features["query_ctx_content_overlap"] = 0.0

    # --- Text answer features (query-context alignment) ---
    _question_words = {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "is",
        "are",
        "was",
        "were",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "will",
        "would",
    }
    q_content = (
        {w.strip("?.,!;:()[]\"'") for w in query.lower().split()}
        - stopwords
        - _question_words
        - {""}
    )
    if q_content and contexts:
        # query_subject_partial: fraction of query subject words in context
        found = sum(1 for w in q_content if w in all_text_lower)
        features["query_subject_partial"] = found / len(q_content)

        # entity_substantive_score: query words mentioned 2+ times
        substantive = sum(1 for w in q_content if all_text_lower.count(w) >= 2)
        features["entity_substantive_score"] = substantive / len(q_content)

        # Sentence-level features
        best_coverage = 0.0
        best_span_len = 0
        best_span_cov = 0.0
        q_all = q_content | (query_words - _question_words - {""})

        for ctx in contexts:
            for sent in re.split(r"[.!?]+", ctx):
                sent = sent.strip()
                if not sent:
                    continue
                sent_words = [w.strip("?.,!;:()[]\"'-").lower() for w in sent.split()]
                sent_set = set(sent_words) - stopwords - {""}
                overlap = q_all & sent_set
                cov = len(overlap) / len(q_all) if q_all else 0
                if cov > best_coverage:
                    best_coverage = cov

                # Span search
                for start in range(len(sent_words)):
                    span_covered: set[str] = set()
                    for end in range(start, min(start + 30, len(sent_words))):
                        w = sent_words[end].strip("?.,!;:()[]\"'-")
                        if w in q_all:
                            span_covered.add(w)
                        if len(span_covered) >= 2:
                            sc = len(span_covered) / len(q_all)
                            if sc > best_span_cov:
                                best_span_cov = sc
                                best_span_len = end - start + 1

        features["best_sentence_coverage"] = best_coverage
        features["best_span_length"] = float(best_span_len)
        features["answer_span_coverage"] = best_span_cov
    else:
        features["query_subject_partial"] = 0.0
        features["entity_substantive_score"] = 0.0
        features["best_sentence_coverage"] = 0.0
        features["best_span_length"] = 0.0
        features["answer_span_coverage"] = 0.0

    return features


def enrich_with_context_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add context-based features by loading original fitz-gov cases.

    Skips features that are already present in the DataFrame (e.g. when
    feature_extractor.py already computed them during eval_pipeline extraction).
    """
    # Compute context features and merge (only filling missing columns)
    print(f"Computing context features from {data_dir}...")
    cases = load_cases_by_id(data_dir)

    new_features = []
    for _, row in df.iterrows():
        case = cases.get(row["case_id"], {})
        feats = compute_context_features(case.get("query", ""), case.get("contexts", []))
        new_features.append(feats)

    ctx_df = pd.DataFrame(new_features)
    # Only add columns not already present in df
    existing = set(df.columns)
    new_cols = [c for c in ctx_df.columns if c not in existing]
    if not new_cols:
        print(f"  All {len(ctx_df.columns)} context features already present, skipping")
        return df
    print(f"  Added {len(new_cols)} context features: {new_cols}")
    return pd.concat([df, ctx_df[new_cols]], axis=1)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Prepare feature matrix: encode categoricals, convert bools to int."""
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].copy()
    encoders: dict[str, LabelEncoder] = {}

    # Drop dead features (constant zero / single-value across all cases)
    dead_present = [c for c in _DEAD_FEATURES if c in X.columns]
    if dead_present:
        X = X.drop(columns=dead_present)
        print(f"  Dropped {len(dead_present)} dead features: {dead_present}")

    for col in _CATEGORICAL_FEATURES:
        if col not in X.columns:
            continue
        X[col] = X[col].fillna("none").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    for col in _BOOL_FEATURES:
        if col not in X.columns:
            continue
        X[col] = X[col].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)

    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Log-transform ctx_number_variance (max can be 7.4 trillion — huge range)
    if "ctx_number_variance" in X.columns:
        X["ctx_number_variance"] = np.log1p(X["ctx_number_variance"])

    # Interaction features (derived from existing columns, no re-extraction needed)
    if "ca_fired" in X.columns and "mean_vector_score" in X.columns:
        X["ix_ca_x_vector"] = X["ca_fired"] * X["mean_vector_score"]
    if "ctx_contradiction_count" in X.columns and "ctx_total_chars" in X.columns:
        X["ix_contradiction_density"] = X["ctx_contradiction_count"] / (X["ctx_total_chars"] + 1)
    if "ctx_negation_count" in X.columns and "ctx_total_chars" in X.columns:
        X["ix_negation_density"] = X["ctx_negation_count"] / (X["ctx_total_chars"] + 1)
    if "mean_vector_score" in X.columns and "score_spread" in X.columns:
        X["ix_score_confidence"] = X["mean_vector_score"] - X["score_spread"]
    if "ca_pairs_checked" in X.columns and "has_disputed_signal" in X.columns:
        X["ix_ca_pairs_x_disputed"] = X["ca_pairs_checked"] * X["has_disputed_signal"]
    # New: IE similarity * IE fired — strong abstain signal
    if "ie_max_similarity" in X.columns and "ie_fired" in X.columns:
        X["ix_ie_sim_x_fired"] = X["ie_max_similarity"] * X["ie_fired"]
    # New: AV votes * vector score — qualified signal
    if "av_jury_votes_no" in X.columns and "mean_vector_score" in X.columns:
        X["ix_av_votes_x_vector"] = X["av_jury_votes_no"] * X["mean_vector_score"]
    # New: disputed signal * av votes — strong disputed indicator
    if "has_disputed_signal" in X.columns and "av_jury_votes_no" in X.columns:
        X["ix_disputed_x_av"] = X["has_disputed_signal"] * X["av_jury_votes_no"]
    # New: context similarity spread — conflicting contexts signal disputed
    if "ctx_max_pairwise_sim" in X.columns and "ctx_min_pairwise_sim" in X.columns:
        X["ix_ctx_sim_spread"] = X["ctx_max_pairwise_sim"] - X["ctx_min_pairwise_sim"]
    # New: vector score * contradiction count — disputed cases with high vector match
    if "mean_vector_score" in X.columns and "ctx_contradiction_count" in X.columns:
        X["ix_vector_x_contradiction"] = X["mean_vector_score"] * X["ctx_contradiction_count"]
    # New: AV fires but IE doesn't — abstain signal missed by IE (helps Stage 1)
    if "av_fired" in X.columns and "ie_fired" in X.columns:
        X["ix_av_no_ie"] = X["av_fired"] * (1 - X["ie_fired"])
    # New: Strong AV denial without IE — strong abstain indicator
    if "av_strong_denial" in X.columns and "ie_fired" in X.columns:
        X["ix_av_strong_no_ie"] = X["av_strong_denial"] * (1 - X["ie_fired"])
    # New: Multiple constraints fire — compound caution signal
    if "num_constraints_fired" in X.columns:
        X["ix_multi_denial"] = (X["num_constraints_fired"] >= 2).astype(int)
    # New: Any denial with low vector score — irrelevant retrieval
    if "has_any_denial" in X.columns and "mean_vector_score" in X.columns:
        X["ix_denial_low_vector"] = X["has_any_denial"] * (1 - X["mean_vector_score"])
    # New: AV jury votes * abstain signal — reinforced abstain
    if "av_jury_votes_no" in X.columns and "has_abstain_signal" in X.columns:
        X["ix_av_x_abstain"] = X["av_jury_votes_no"] * X["has_abstain_signal"]
    # New: cross-chunk divergence * no CA fire — numeric conflict CA missed
    if "has_cross_chunk_divergence" in X.columns and "ca_fired" in X.columns:
        X["ix_divergence_no_ca"] = X["has_cross_chunk_divergence"] * (1 - X["ca_fired"])
    # New: any numerical divergence (within or across) * no CA fire
    if "has_within_chunk_divergence" in X.columns and "ca_fired" in X.columns:
        X["ix_any_divergence_no_ca"] = (
            X.get("has_cross_chunk_divergence", 0) | X["has_within_chunk_divergence"]
        ) * (1 - X["ca_fired"])

    # Q3-specific features: distinguish data-rich trustworthy from genuine disputes
    if "ctx_number_count" in X.columns and "num_chunks" in X.columns:
        X["numerical_richness_per_chunk"] = X["ctx_number_count"] / X["num_chunks"].clip(lower=1)
    if "within_chunk_num_conflicts" in X.columns and "cross_chunk_num_conflicts" in X.columns:
        total_conflicts = X["cross_chunk_num_conflicts"] + X["within_chunk_num_conflicts"]
        X["conflict_internality_ratio"] = X["within_chunk_num_conflicts"] / total_conflicts.clip(
            lower=1
        )
    if "ctx_total_chars" in X.columns and "num_chunks" in X.columns:
        X["chars_per_chunk"] = X["ctx_total_chars"] / X["num_chunks"].clip(lower=1)
    if "ctx_contradiction_count" in X.columns and "ctx_total_chars" in X.columns:
        X["contradiction_per_char"] = X["ctx_contradiction_count"] / X["ctx_total_chars"].clip(
            lower=1
        )

    # Conflict quality features (Q3/FT-disputed): nature of conflict, not just quantity
    if "cross_chunk_num_conflicts" in X.columns and "ctx_number_count" in X.columns:
        X["conflict_to_number_ratio"] = X["cross_chunk_num_conflicts"] / X["ctx_number_count"].clip(
            lower=1
        )
    if "ctx_negation_count" in X.columns and "ctx_total_chars" in X.columns:
        X["negation_per_char"] = X["ctx_negation_count"] / X["ctx_total_chars"].clip(lower=1)

    # Q1 recovery: partial-answer detection
    if "ctx_total_chars" in X.columns and "vocab_overlap_ratio" in X.columns:
        X["short_ctx_with_overlap"] = (
            (X["ctx_total_chars"] < 500) & (X["vocab_overlap_ratio"] > 0.3)
        ).astype(int)

    # Interaction: Q1 recovery — AV fires but retrieval is decent
    if "av_fired" in X.columns and "vocab_overlap_ratio" in X.columns:
        X["ix_av_fires_good_overlap"] = X["av_fired"] * X["vocab_overlap_ratio"]
    # Interaction: Q3 — conflict concentrated vs spread (high = real concentrated conflict)
    if "cross_chunk_max_divergence" in X.columns and "cross_chunk_num_conflicts" in X.columns:
        X["ix_max_div_per_conflict"] = X["cross_chunk_max_divergence"] / (
            X["cross_chunk_num_conflicts"].clip(lower=1)
        )
    # Interaction: single chunk with denial — common Q1 false-abstain pattern
    if "num_chunks" in X.columns and "has_any_denial" in X.columns:
        X["ix_single_chunk_denial"] = ((X["num_chunks"] == 1) & (X["has_any_denial"] == 1)).astype(
            int
        )
    # Interaction: constraint disagreement — IE and CA disagree (IE=insufficient but CA=conflict)
    if "ie_fired" in X.columns and "ca_fired" in X.columns:
        X["ix_ie_no_ca"] = X["ie_fired"] * (1 - X["ca_fired"])
        X["ix_ca_no_ie"] = X["ca_fired"] * (1 - X["ie_fired"])

    return X, encoders


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def quick_model_comparison(
    X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, seed: int
) -> dict[str, dict]:
    """Compare multiple models with class weighting. Returns name -> {mean, std, model}."""
    sample_weights = compute_sample_weight("balanced", y_train)

    models = {
        "GBT": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            random_state=seed,
        ),
        "LR": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    }

    print("\n--- Quick model comparison (5-fold CV, class-weighted) ---")
    results = {}
    for name, model in models.items():
        start = time.time()
        if name == "GBT":
            # GBT doesn't have class_weight, use manual sample_weight
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                m = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    min_samples_leaf=5,
                    random_state=seed,
                )
                m.fit(
                    X_train.iloc[train_idx],
                    y_train.iloc[train_idx],
                    sample_weight=sample_weights[train_idx],
                )
                scores.append(
                    accuracy_score(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]))
                )
            cv_scores = np.array(scores)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        elapsed = time.time() - start
        results[name] = {"mean": cv_scores.mean(), "std": cv_scores.std(), "model": model}
        print(f"  {name:6s}: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}  ({elapsed:.1f}s)")

    return results


def hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    top_models: list[tuple[str, dict]],
    time_budget: int,
    seed: int,
) -> dict[str, dict]:
    """Run RandomizedSearchCV on top models within time budget."""
    sample_weights = compute_sample_weight("balanced", y_train)
    time_per_model = time_budget / len(top_models)

    param_grids = {
        "GBT": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.3),
            "min_samples_leaf": randint(2, 20),
            "subsample": uniform(0.6, 0.4),
            "max_features": uniform(0.3, 0.7),
        },
        "RF": {
            "n_estimators": randint(100, 800),
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "min_samples_split": randint(2, 15),
        },
        "ET": {
            "n_estimators": randint(100, 800),
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "min_samples_split": randint(2, 15),
        },
        "SVM": {
            "C": uniform(0.01, 100),
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "poly"],
        },
        "LR": {
            "C": uniform(0.01, 100),
            "solver": ["lbfgs", "saga"],
            "penalty": ["l2"],
        },
    }

    base_models = {
        "GBT": GradientBoostingClassifier(random_state=seed),
        "RF": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "ET": ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=seed),
        "LR": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    }

    print(f"\n--- Hyperparameter search ({time_budget}s budget, {time_per_model:.0f}s/model) ---")
    tuned = {}
    for name, info in top_models:
        if name not in param_grids:
            continue
        start = time.time()
        # Estimate iterations from time budget: ~0.5s per CV iteration for tree models
        n_iter = max(10, int(time_per_model / 3))  # rough: 3s per full CV round

        search = RandomizedSearchCV(
            base_models[name],
            param_grids[name],
            n_iter=n_iter,
            cv=cv,
            scoring="accuracy",
            random_state=seed,
            n_jobs=-1 if name != "GBT" else 1,
        )
        if name == "GBT":
            search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            search.fit(X_train, y_train)

        elapsed = time.time() - start
        tuned[name] = {
            "model": search.best_estimator_,
            "score": search.best_score_,
            "params": search.best_params_,
        }
        total_fits = cv.get_n_splits() * n_iter
        print(
            f"  {name:6s}: {search.best_score_:.3f} (searched {total_fits} fits in {elapsed:.1f}s)"
        )
        # Print key params
        for k, v in search.best_params_.items():
            if isinstance(v, float):
                print(f"         {k}: {v:.4f}")
            else:
                print(f"         {k}: {v}")

    return tuned


def build_ensemble(
    tuned_models: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> StackingClassifier:
    """Build stacking ensemble from tuned models."""
    estimators = [(name, info["model"]) for name, info in tuned_models.items()]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    print(f"\n--- Building stacking ensemble ({len(estimators)} base models) ---")
    start = time.time()
    stack.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Ensemble trained in {elapsed:.1f}s")
    return stack


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def print_confusion_matrix(y_true, y_pred, labels):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "predicted ->".rjust(20) + "".join(f"{lbl:>12}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"actual {label}".rjust(20) + "".join(f"{cm[i, j]:>12}" for j in range(len(labels)))
        print(row)
    print()


def evaluate_model(name, model, X_test, y_test, labels):
    """Evaluate a single model and return predictions + accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f} ({sum(y_test == y_pred)}/{len(y_test)})")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    print_confusion_matrix(y_test, y_pred, labels)

    # Dispute recall diagnostic
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if "disputed" in labels and "qualified" in labels:
        d_idx = labels.index("disputed")
        q_idx = labels.index("qualified")
        d_as_q = cm[d_idx, q_idx]
        d_total = cm[d_idx].sum()
        d_correct = cm[d_idx, d_idx]
        print(f"  Disputed recall: {d_correct}/{d_total} ({100*d_correct/d_total:.0f}%)")
        print(f"  Disputed -> qualified confusion: {d_as_q}/{d_total}")

    return y_pred, acc


# ---------------------------------------------------------------------------
# Two-stage binary training
# ---------------------------------------------------------------------------

_3CLASS_LABELS = ["abstain", "disputed", "trustworthy"]


def _collapse_to_3class(labels: np.ndarray) -> np.ndarray:
    """Collapse confident+qualified → trustworthy."""
    return np.array(
        ["trustworthy" if lbl in ("confident", "qualified") else lbl for lbl in labels],
        dtype=object,
    )


def train_twostage(
    X: pd.DataFrame,
    y_4class: np.ndarray,
    feature_names: list[str],
    encoders: dict[str, LabelEncoder],
    test_size: float,
    time_budget: int,
    seed: int,
    output_path: Path,
    governor_preds: np.ndarray | None = None,
    case_ids: pd.Series | None = None,
):
    """Train a two-stage binary classifier: Stage 1 (answerable vs abstain) → Stage 2 (trustworthy vs disputed)."""

    y_3class = _collapse_to_3class(y_4class)

    # Stratified split (stratify on 3-class to preserve disputed proportion)
    X_train, X_test, y3_train, y3_test = train_test_split(
        X,
        y_3class,
        test_size=test_size,
        random_state=seed,
        stratify=y_3class,
    )
    y_s1_train = np.array(
        [0 if lbl == "abstain" else 1 for lbl in y3_train],
        dtype=int,
    )
    y_s1_test = np.array(
        [0 if lbl == "abstain" else 1 for lbl in y3_test],
        dtype=int,
    )
    _s1_label_map = {0: "abstain", 1: "answerable"}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    time_per_stage = time_budget // 2

    # ---- Stage 1: answerable vs abstain ----
    print(f"\n{'='*60}")
    print("STAGE 1: answerable vs abstain")
    print(f"{'='*60}")
    print(f"Train: {sum(y_s1_train == 1)} answerable, {sum(y_s1_train == 0)} abstain")
    print(f"Test:  {sum(y_s1_test == 1)} answerable, {sum(y_s1_test == 0)} abstain")

    s1_models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "GBT": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=seed,
        ),
    }
    if _HAS_XGB:
        n_abstain = sum(y_s1_train == 0)
        n_answerable = sum(y_s1_train == 1)
        s1_models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=n_abstain / n_answerable if n_answerable > 0 else 1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    if _HAS_LGBM:
        s1_models["LGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    print("\n--- Stage 1: Quick comparison ---")
    print("  Selection: best abstain recall (answerable recall >= 0.80)")
    s1_results = {}
    for name, model in s1_models.items():
        cv_scores = cross_val_score(model, X_train, y_s1_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_s1_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_s1_test, pred)
        cm = confusion_matrix(y_s1_test, pred, labels=[0, 1])
        abs_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
        ans_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        s1_results[name] = {
            "model": model,
            "acc": acc,
            "cv": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "abs_recall": abs_recall,
            "ans_recall": ans_recall,
        }
        print(
            f"  {name:6s}: test={acc:.4f}, CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})"
            f"  abstain={abs_recall:.3f} answerable={ans_recall:.3f}"
        )

    # Stage 1 selection: best abstain recall with answerable recall >= 0.80
    # Stage 1's job is to CATCH abstain cases — overall accuracy favors majority class
    _MIN_ANS_RECALL = 0.80
    s1_eligible = [
        (name, info) for name, info in s1_results.items() if info["ans_recall"] >= _MIN_ANS_RECALL
    ]
    if not s1_eligible:
        s1_eligible = list(s1_results.items())  # fallback: all models
    s1_ranked = sorted(s1_eligible, key=lambda x: x[1]["abs_recall"], reverse=True)
    s1_best_name = s1_ranked[0][0]
    print(
        f"\n  Selected: {s1_best_name} (abstain={s1_results[s1_best_name]['abs_recall']:.3f},"
        f" answerable={s1_results[s1_best_name]['ans_recall']:.3f})"
    )
    print(f"\n--- Stage 1: Tuning {s1_best_name} ({time_per_stage}s budget) ---")

    s1_param_grids = {
        "RF": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "ET": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "GBT": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
        },
        "XGB": {
            "n_estimators": randint(100, 600),
            "max_depth": randint(2, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.3, 0.7),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0.5, 2),
            "min_child_weight": randint(1, 10),
        },
        "LGBM": {
            "n_estimators": randint(100, 600),
            "max_depth": [-1, 5, 10, 15, 20],
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.3, 0.7),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0.5, 2),
            "num_leaves": randint(15, 63),
        },
    }

    s1_base = {
        "RF": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "ET": ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "GBT": GradientBoostingClassifier(random_state=seed),
    }
    if _HAS_XGB:
        s1_base["XGB"] = XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric="logloss", verbosity=0
        )
    if _HAS_LGBM:
        s1_base["LGBM"] = LGBMClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1
        )

    n_iter = max(10, int(time_per_stage / 3))
    # Stage 1 optimizes for balanced recall (macro), not accuracy.
    # Accuracy favors the majority class (answerable), but Stage 1's job
    # is to CATCH abstain cases — recall_macro balances both classes equally.
    s1_search = RandomizedSearchCV(
        s1_base[s1_best_name],
        s1_param_grids[s1_best_name],
        n_iter=n_iter,
        cv=cv,
        scoring="recall_macro",
        random_state=seed,
        n_jobs=-1 if s1_best_name not in ("GBT", "XGB", "LGBM") else 1,
    )
    s1_search.fit(X_train, y_s1_train)
    s1_model = s1_search.best_estimator_

    # Stage 1 threshold tuning: trade small answerable recall for better abstain recall
    s1_threshold = 0.5
    if hasattr(s1_model, "predict_proba"):
        from sklearn.model_selection import cross_val_predict as cvp_s1

        s1_cv_proba = cvp_s1(s1_model, X_train, y_s1_train, cv=cv, method="predict_proba")
        abs_mask_cv = y_s1_train == 0
        ans_mask_cv = y_s1_train == 1
        s1_candidates = []
        for thresh in np.arange(0.30, 0.70, 0.005):
            preds = (s1_cv_proba[:, 1] >= thresh).astype(int)
            abs_r = (preds[abs_mask_cv] == 0).mean() if abs_mask_cv.any() else 0
            ans_r = (preds[ans_mask_cv] == 1).mean() if ans_mask_cv.any() else 0
            s1_candidates.append((thresh, abs_r, ans_r))
        # Find threshold: maximize abstain recall with answerable recall >= 0.93
        s1_eligible = [(t, ar, anr) for t, ar, anr in s1_candidates if anr >= 0.93]
        if s1_eligible:
            s1_threshold, _, _ = max(s1_eligible, key=lambda x: x[1])
        else:
            s1_eligible = [(t, ar, anr) for t, ar, anr in s1_candidates if anr >= 0.90]
            if s1_eligible:
                s1_threshold, _, _ = max(s1_eligible, key=lambda x: x[1])
        # Show effect
        preds_at_s1 = (s1_cv_proba[:, 1] >= s1_threshold).astype(int)
        abs_r_at = (preds_at_s1[abs_mask_cv] == 0).mean()
        ans_r_at = (preds_at_s1[ans_mask_cv] == 1).mean()
        print(f"\n  Stage 1 threshold: {s1_threshold:.3f} (default 0.50)")
        print(f"  CV abstain recall:    {abs_r_at:.3f}")
        print(f"  CV answerable recall: {ans_r_at:.3f}")

    # Apply threshold to test set
    if hasattr(s1_model, "predict_proba") and s1_threshold != 0.5:
        s1_proba_test = s1_model.predict_proba(X_test)
        s1_pred_test = (s1_proba_test[:, 1] >= s1_threshold).astype(int)
    else:
        s1_pred_test = s1_model.predict(X_test)
    s1_acc = accuracy_score(y_s1_test, s1_pred_test)
    cm_s1 = confusion_matrix(y_s1_test, s1_pred_test, labels=[0, 1])
    print(f"  Best: {s1_search.best_score_:.4f} CV -> {s1_acc:.4f} test")
    print(f"  Params: {s1_search.best_params_}")
    print(f"  Abstain recall:    {cm_s1[0,0]}/{cm_s1[0].sum()} ({cm_s1[0,0]/cm_s1[0].sum():.3f})")
    print(f"  Answerable recall: {cm_s1[1,1]}/{cm_s1[1].sum()} ({cm_s1[1,1]/cm_s1[1].sum():.3f})")

    if hasattr(s1_model, "feature_importances_"):
        imp = s1_model.feature_importances_
        order = np.argsort(imp)[::-1]
        print("\n  Stage 1 top 10 features:")
        for rank, idx in enumerate(order[:10], 1):
            print(f"    {rank:2d}. {feature_names[idx]:40s} {imp[idx]:.4f}")

    # ---- Stage 2: trustworthy vs disputed (answerable subset) ----
    print(f"\n{'='*60}")
    print("STAGE 2: trustworthy vs disputed")
    print(f"{'='*60}")

    answerable_train = y_s1_train == 1
    X_train_s2 = X_train[answerable_train]
    y_train_s2_str = y3_train[answerable_train]
    # Encode: disputed=0, trustworthy=1
    y_train_s2 = np.array([0 if lbl == "disputed" else 1 for lbl in y_train_s2_str], dtype=int)
    _s2_label_map = {0: "disputed", 1: "trustworthy"}
    print(f"Train: {sum(y_train_s2 == 1)} trustworthy, {sum(y_train_s2 == 0)} disputed")

    # Full answerable data for CV
    answerable_all = np.array(
        [lbl != "abstain" for lbl in y_3class],
        dtype=bool,
    )
    X_answerable = X[answerable_all]
    y_answerable = np.array(
        [0 if lbl == "disputed" else 1 for lbl in y_3class[answerable_all]], dtype=int
    )

    s2_models = {
        "ET": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "GBT": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=seed,
        ),
    }
    if _HAS_XGB:
        n_tw = sum(y_train_s2 == 1)
        n_dp = sum(y_train_s2 == 0)
        s2_models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=n_dp / n_tw if n_tw > 0 else 1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    if _HAS_LGBM:
        s2_models["LGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    print("\n--- Stage 2: Quick comparison (recall_macro) ---")
    s2_results = {}
    for name, model in s2_models.items():
        cv_scores = cross_val_score(
            model, X_answerable, y_answerable, cv=cv, scoring="recall_macro"
        )
        model.fit(X_train_s2, y_train_s2)
        s2_results[name] = {"model": model, "cv": cv_scores.mean(), "cv_std": cv_scores.std()}
        print(f"  {name:6s}: CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

    # Tune top 3 Stage 2 models and build stacking ensemble
    s2_ranked = sorted(s2_results.items(), key=lambda x: x[1]["cv"], reverse=True)
    s2_top3 = s2_ranked[:3]
    print(f"\n--- Stage 2: Tuning top 3 for ensemble ({time_per_stage}s budget) ---")
    s2_tuned_models = {}
    time_per_s2_model = time_per_stage // len(s2_top3)
    for s2_name, _ in s2_top3:
        n_iter_s2 = max(10, int(time_per_s2_model / 3))
        s2_search = RandomizedSearchCV(
            s1_base[s2_name],
            s1_param_grids[s2_name],
            n_iter=n_iter_s2,
            cv=cv,
            scoring="recall_macro",
            random_state=seed,
            n_jobs=-1 if s2_name not in ("GBT", "XGB", "LGBM") else 1,
        )
        s2_search.fit(X_train_s2, y_train_s2)
        s2_tuned_models[s2_name] = s2_search.best_estimator_
        print(f"  {s2_name}: CV={s2_search.best_score_:.4f}")

    # Build Stage 2 stacking ensemble for more robust predictions
    s2_estimators = [(name, model) for name, model in s2_tuned_models.items()]
    s2_model = StackingClassifier(
        estimators=s2_estimators,
        final_estimator=LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed
        ),
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
    s2_model.fit(X_train_s2, y_train_s2)
    s2_best_name = "Ensemble(" + "+".join(s2_tuned_models.keys()) + ")"
    print(f"  Stage 2 ensemble: {s2_best_name}")

    # ---- Stage 2: Threshold tuning for disputed recall ----
    # Default threshold 0.5 favors trustworthy (majority). Lower threshold = more disputed.
    s2_threshold = 0.5
    if hasattr(s2_model, "predict_proba"):
        print("\n--- Stage 2: Threshold tuning for disputed recall ---")
        # Use CV predictions on answerable training data
        from sklearn.model_selection import cross_val_predict

        s2_cv_proba = cross_val_predict(
            s2_model, X_answerable, y_answerable, cv=cv, method="predict_proba"
        )
        # s2_cv_proba[:, 1] = P(trustworthy), lower threshold = more disputed
        # Strategy: find threshold that achieves disputed recall >= 0.85
        # while maximizing trustworthy recall. If impossible, find best
        # balance with minimum trustworthy recall >= 0.78.
        disp_mask = y_answerable == 0
        trust_mask = y_answerable == 1
        candidates = []
        for thresh in np.arange(0.30, 0.80, 0.005):
            preds = (s2_cv_proba[:, 1] >= thresh).astype(int)
            disp_recall = (preds[disp_mask] == 0).mean() if disp_mask.any() else 0
            trust_recall = (preds[trust_mask] == 1).mean() if trust_mask.any() else 0
            candidates.append((thresh, disp_recall, trust_recall))

        # Phase 1: find threshold achieving disputed >= 0.85 with max trustworthy
        phase1 = [(t, dr, tr) for t, dr, tr in candidates if dr >= 0.85]
        if phase1:
            # Pick highest trustworthy recall among those meeting disputed target
            best_thresh, _, _ = max(phase1, key=lambda x: x[2])
        else:
            # Phase 2: maximize disputed recall with trustworthy >= 0.78
            phase2 = [(t, dr, tr) for t, dr, tr in candidates if tr >= 0.78]
            if phase2:
                best_thresh, _, _ = max(phase2, key=lambda x: x[1])
            else:
                # Phase 3: best balance (weighted)
                best_thresh, _, _ = max(candidates, key=lambda x: x[1] * 0.6 + x[2] * 0.4)
        s2_threshold = best_thresh
        # Show effect of chosen threshold
        preds_at_thresh = (s2_cv_proba[:, 1] >= s2_threshold).astype(int)
        disp_mask = y_answerable == 0
        trust_mask = y_answerable == 1
        disp_r = (preds_at_thresh[disp_mask] == 0).mean()
        trust_r = (preds_at_thresh[trust_mask] == 1).mean()
        print(f"  Optimal threshold: {s2_threshold:.2f} (default 0.50)")
        print(f"  CV disputed recall:    {disp_r:.3f}")
        print(f"  CV trustworthy recall: {trust_r:.3f}")

    # ---- Combined evaluation ----
    print(f"\n{'='*60}")
    print("COMBINED TWO-STAGE EVALUATION")
    print(f"{'='*60}")

    final_pred = np.full(len(y3_test), "abstain", dtype=object)
    s1_answerable_mask = s1_pred_test == 1
    if s1_answerable_mask.any():
        if hasattr(s2_model, "predict_proba") and s2_threshold != 0.5:
            s2_proba = s2_model.predict_proba(X_test[s1_answerable_mask])
            s2_preds_int = (s2_proba[:, 1] >= s2_threshold).astype(int)
        else:
            s2_preds_int = s2_model.predict(X_test[s1_answerable_mask])
        answerable_positions = np.where(s1_answerable_mask)[0]
        for i, pos in enumerate(answerable_positions):
            final_pred[pos] = _s2_label_map[int(s2_preds_int[i])]

    combined_acc = accuracy_score(y3_test, final_pred)
    print(f"\nCombined accuracy: {combined_acc:.4f} ({sum(y3_test == final_pred)}/{len(y3_test)})")
    print(classification_report(y3_test, final_pred, labels=_3CLASS_LABELS, zero_division=0))

    cm_combined = confusion_matrix(y3_test, final_pred, labels=_3CLASS_LABELS)
    header = "predicted ->".rjust(20) + "".join(f"{lbl:>15}" for lbl in _3CLASS_LABELS)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(_3CLASS_LABELS):
        row = f"actual {label}".rjust(20) + "".join(
            f"{cm_combined[i, j]:>15}" for j in range(len(_3CLASS_LABELS))
        )
        print(row)

    # Per-class recall
    print("\nPer-class recall:")
    recalls = {}
    for i, label in enumerate(_3CLASS_LABELS):
        total = cm_combined[i].sum()
        recall = cm_combined[i, i] / total if total > 0 else 0
        recalls[label] = recall
        print(f"  {label:15s}: {recall:.4f} ({cm_combined[i, i]}/{total})")
    print(f"  Min recall: {min(recalls.values()):.4f}")

    # Governor comparison
    if governor_preds is not None:
        gov_3class = _collapse_to_3class(governor_preds[X_test.index])
        gov_acc = accuracy_score(y3_test, gov_3class)
        print(f"\nGovernor baseline (3-class): {gov_acc:.4f}")
        print(f"Delta vs governor: +{combined_acc - gov_acc:.4f}")

    # ---- Build test split metadata ----
    test_ids = list(case_ids.iloc[X_test.index].values) if case_ids is not None else []
    test_split_hash = (
        hashlib.sha256(",".join(sorted(str(i) for i in test_ids)).encode()).hexdigest()[:16]
        if test_ids
        else ""
    )

    # ---- Save artifact ----
    artifact = {
        "mode": "twostage",
        "stage1_model": s1_model,
        "stage1_name": f"{s1_best_name} (tuned)",
        "stage1_threshold": s1_threshold,
        "stage1_accuracy": s1_acc,
        "stage2_model": s2_model,
        "stage2_name": f"{s2_best_name} (tuned)",
        "stage2_cv": s2_search.best_score_,
        "stage2_threshold": s2_threshold,
        "combined_accuracy": combined_acc,
        "combined_recalls": recalls,
        "encoders": encoders,
        "feature_names": feature_names,
        "labels": _3CLASS_LABELS,
        "test_case_ids": test_ids,
        "test_split_hash": test_split_hash,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nTwo-stage model saved to {output_path}")
    print(f"  Stage 1: {s1_best_name} (tuned) — answerable vs abstain")
    print(f"  Stage 2: {s2_best_name} (tuned) — trustworthy vs disputed")
    print(f"  Combined: {combined_acc:.4f}")
    if test_ids:
        print(f"  Test split: {len(test_ids)} cases (hash={test_split_hash})")

    return artifact


# ---------------------------------------------------------------------------
# Cascade: 4-question atomic decomposition
# ---------------------------------------------------------------------------

# Q2 conflict routing: ca_fired indicates material conflict in evidence
_CONFLICT_FEATURE = "ca_fired"

_CASCADE_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "fitz_ai"
    / "governance"
    / "data"
    / "model_v6_cascade.joblib"
)


def _train_binary_stage(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    time_budget: int,
    seed: int,
    class_labels: tuple[str, str],
    optimize_recall_class: int = 0,
) -> tuple:
    """Train a single binary stage: compare models, tune best, return (model, threshold)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Model zoo
    models = {}
    models["RF"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    models["ET"] = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    models["GBT"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=seed,
    )
    if _HAS_XGB:
        n0, n1 = sum(y_train == 0), sum(y_train == 1)
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=n0 / n1 if n1 > 0 else 1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    if _HAS_LGBM:
        models["LGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    # Quick comparison
    print(f"\n  --- {name}: Quick comparison ---")
    results = {}
    for mname, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall_macro")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        r0 = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
        r1 = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        results[mname] = {
            "model": model,
            "acc": acc,
            "cv": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "r0": r0,
            "r1": r1,
        }
        print(
            f"    {mname:6s}: CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})"
            f"  {class_labels[0]}={r0:.3f} {class_labels[1]}={r1:.3f}"
        )

    # Select best by recall of target class (with other class >= 0.80)
    other_class = 1 - optimize_recall_class
    target_key = f"r{optimize_recall_class}"
    other_key = f"r{other_class}"
    eligible = [(n, r) for n, r in results.items() if r[other_key] >= 0.80]
    if not eligible:
        eligible = list(results.items())
    best_name = max(eligible, key=lambda x: x[1][target_key])[0]
    print(f"    Selected: {best_name}")

    # Tune
    param_grids = {
        "RF": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "ET": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "GBT": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
        },
        "XGB": {
            "n_estimators": randint(100, 600),
            "max_depth": randint(2, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.3, 0.7),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0.5, 2),
            "min_child_weight": randint(1, 10),
        },
        "LGBM": {
            "n_estimators": randint(100, 600),
            "max_depth": [-1, 5, 10, 15, 20],
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.3, 0.7),
            "num_leaves": randint(15, 63),
        },
    }
    base_models = {
        "RF": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "ET": ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "GBT": GradientBoostingClassifier(random_state=seed),
    }
    if _HAS_XGB:
        base_models["XGB"] = XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric="logloss", verbosity=0
        )
    if _HAS_LGBM:
        base_models["LGBM"] = LGBMClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1
        )

    n_iter = max(10, int(time_budget / 3))
    search = RandomizedSearchCV(
        base_models[best_name],
        param_grids[best_name],
        n_iter=n_iter,
        cv=cv,
        scoring="recall_macro",
        random_state=seed,
        n_jobs=-1 if best_name not in ("GBT", "XGB", "LGBM") else 1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"    Tuned {best_name}: CV={search.best_score_:.4f}")

    # Threshold tuning
    threshold = 0.5
    if hasattr(model, "predict_proba"):
        cv_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")
        mask0 = y_train == 0
        mask1 = y_train == 1
        candidates = []
        for t in np.arange(0.30, 0.70, 0.005):
            preds = (cv_proba[:, 1] >= t).astype(int)
            r0 = (preds[mask0] == 0).mean() if mask0.any() else 0
            r1 = (preds[mask1] == 1).mean() if mask1.any() else 0
            candidates.append((t, r0, r1))
        # Maximize target recall with other recall >= 0.85
        target_idx = 1 + optimize_recall_class  # r0 or r1
        elig = [(t, r0, r1) for t, r0, r1 in candidates if (r0, r1)[other_class] >= 0.85]
        if elig:
            threshold = max(elig, key=lambda x: x[target_idx])[0]
        else:
            elig = [(t, r0, r1) for t, r0, r1 in candidates if (r0, r1)[other_class] >= 0.80]
            if elig:
                threshold = max(elig, key=lambda x: x[target_idx])[0]
        preds_at = (cv_proba[:, 1] >= threshold).astype(int)
        r0_at = (preds_at[mask0] == 0).mean()
        r1_at = (preds_at[mask1] == 1).mean()
        print(
            f"    Threshold: {threshold:.3f} — {class_labels[0]}={r0_at:.3f} {class_labels[1]}={r1_at:.3f}"
        )

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1]
        print("    Top 5 features:")
        for rank, idx in enumerate(order[:5], 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"      {rank}. {fname:40s} {imp[idx]:.4f}")

    return model, threshold, best_name


def _train_binary_stage_cv(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    time_budget: int,
    seed: int,
    class_labels: tuple[str, str],
    optimize_recall_class: int = 0,
) -> tuple:
    """Train a binary stage with 5-fold CV, returning OOF predictions and final model on all data.

    Returns (model, threshold, best_name, oof_proba) where oof_proba is an (N, 2) array
    of out-of-fold predicted probabilities for every sample.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Model zoo
    models = {}
    models["RF"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    models["ET"] = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    models["GBT"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=seed,
    )
    if _HAS_XGB:
        n0, n1 = sum(y == 0), sum(y == 1)
        models["XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=n0 / n1 if n1 > 0 else 1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    if _HAS_LGBM:
        models["LGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    # Quick comparison using CV on all data
    print(f"\n  --- {name}: Quick comparison (5-fold CV) ---")
    results = {}
    for mname, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="recall_macro")
        # Get OOF predictions for evaluation
        oof_pred = cross_val_predict(model, X, y, cv=cv)
        acc = accuracy_score(y, oof_pred)
        cm = confusion_matrix(y, oof_pred, labels=[0, 1])
        r0 = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
        r1 = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        results[mname] = {
            "model": model,
            "acc": acc,
            "cv": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "r0": r0,
            "r1": r1,
        }
        print(
            f"    {mname:6s}: CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})"
            f"  {class_labels[0]}={r0:.3f} {class_labels[1]}={r1:.3f}"
        )

    # Select best by recall of target class (with other class >= 0.80)
    other_class = 1 - optimize_recall_class
    target_key = f"r{optimize_recall_class}"
    other_key = f"r{other_class}"
    eligible = [(n, r) for n, r in results.items() if r[other_key] >= 0.80]
    if not eligible:
        eligible = list(results.items())
    best_name = max(eligible, key=lambda x: x[1][target_key])[0]
    print(f"    Selected: {best_name}")

    # Tune with RandomizedSearchCV on all data
    param_grids = {
        "RF": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "ET": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 15),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "GBT": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
        },
        "XGB": {
            "n_estimators": randint(100, 600),
            "max_depth": randint(2, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.3, 0.7),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0.5, 2),
            "min_child_weight": randint(1, 10),
        },
        "LGBM": {
            "n_estimators": randint(100, 600),
            "max_depth": [-1, 5, 10, 15, 20],
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.3, 0.7),
            "num_leaves": randint(15, 63),
        },
    }
    base_models = {
        "RF": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "ET": ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "GBT": GradientBoostingClassifier(random_state=seed),
    }
    if _HAS_XGB:
        base_models["XGB"] = XGBClassifier(
            random_state=seed, n_jobs=-1, eval_metric="logloss", verbosity=0
        )
    if _HAS_LGBM:
        base_models["LGBM"] = LGBMClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1
        )

    n_iter = max(10, int(time_budget / 3))
    search = RandomizedSearchCV(
        base_models[best_name],
        param_grids[best_name],
        n_iter=n_iter,
        cv=cv,
        scoring="recall_macro",
        random_state=seed,
        n_jobs=-1 if best_name not in ("GBT", "XGB", "LGBM") else 1,
    )
    search.fit(X, y)
    model = search.best_estimator_
    print(f"    Tuned {best_name}: CV={search.best_score_:.4f}")

    # OOF predictions from tuned model for threshold tuning and evaluation
    oof_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    # Threshold tuning on OOF predictions
    threshold = 0.5
    mask0 = y == 0
    mask1 = y == 1
    candidates = []
    for t in np.arange(0.30, 0.70, 0.005):
        preds = (oof_proba[:, 1] >= t).astype(int)
        r0 = (preds[mask0] == 0).mean() if mask0.any() else 0
        r1 = (preds[mask1] == 1).mean() if mask1.any() else 0
        candidates.append((t, r0, r1))
    # Maximize target recall with other recall >= 0.85
    target_idx = 1 + optimize_recall_class  # r0 or r1
    elig = [(t, r0, r1) for t, r0, r1 in candidates if (r0, r1)[other_class] >= 0.85]
    if elig:
        threshold = max(elig, key=lambda x: x[target_idx])[0]
    else:
        elig = [(t, r0, r1) for t, r0, r1 in candidates if (r0, r1)[other_class] >= 0.80]
        if elig:
            threshold = max(elig, key=lambda x: x[target_idx])[0]
    preds_at = (oof_proba[:, 1] >= threshold).astype(int)
    r0_at = (preds_at[mask0] == 0).mean()
    r1_at = (preds_at[mask1] == 1).mean()
    print(
        f"    Threshold: {threshold:.3f} — {class_labels[0]}={r0_at:.3f} {class_labels[1]}={r1_at:.3f}"
    )

    # Train final model on ALL data
    model.fit(X, y)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1]
        print("    Top 5 features:")
        for rank, idx in enumerate(order[:5], 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"      {rank}. {fname:40s} {imp[idx]:.4f}")

    return model, threshold, best_name, oof_proba


def train_cascade(
    X: pd.DataFrame,
    y_4class: np.ndarray,
    feature_names: list[str],
    encoders: dict[str, LabelEncoder],
    time_budget: int,
    seed: int,
    output_path: Path,
    governor_preds: np.ndarray | None = None,
    case_ids: pd.Series | None = None,
):
    """Train 4-question atomic cascade classifier using 5-fold cross-validation.

    Every case serves as both train and test via out-of-fold predictions.
    Final models are trained on ALL data.

    Q1: Is evidence sufficient?      → No = ABSTAIN  (ML)
    Q2: Is there material conflict?  → routes Q3/Q4  (rule: ca_fired)
    Q3: Is the conflict resolved?    → Yes=TRUSTWORTHY, No=DISPUTED (ML, conflict cases only)
    Q4: Is evidence truly solid?     → Yes=TRUSTWORTHY, No=ABSTAIN  (ML, clean cases only)
    """
    y_3class = _collapse_to_3class(y_4class)

    time_per_q = time_budget // 4  # Q1, Q2, Q3, Q4 each get a share

    # ================================================================
    # Q1: Is evidence sufficient? (abstain vs answerable)
    # ================================================================
    print(f"\n{'='*60}")
    print("Q1: Is evidence sufficient? (abstain vs answerable)")
    print(f"{'='*60}")

    y_q1 = np.where(y_3class == "abstain", 0, 1)
    print(f"  Total: {sum(y_q1 == 1)} answerable, {sum(y_q1 == 0)} abstain")

    q1_model, q1_threshold, q1_name, q1_oof = _train_binary_stage_cv(
        "Q1",
        X,
        y_q1,
        feature_names,
        time_per_q,
        seed,
        class_labels=("abstain", "answerable"),
        optimize_recall_class=0,  # optimize abstain recall
    )

    # ================================================================
    # Q2: Is there material conflict? (ML router replaces hard ca_fired rule)
    # ================================================================
    print(f"\n{'='*60}")
    print("Q2: Is there material conflict? (ML router)")
    print(f"{'='*60}")

    # Train Q2 as a binary classifier: disputed cases = conflict, others = clean.
    # This lets the model learn dispute signals beyond just ca_fired (numerical
    # divergence, contradiction markers, opposing conclusions, etc.)
    y_q2 = np.where(y_3class == "disputed", 1, 0)
    print(f"  Total: {sum(y_q2 == 1)} conflict (disputed), {sum(y_q2 == 0)} clean (abstain+trustworthy)")
    print(f"  (ca_fired baseline: {X[_CONFLICT_FEATURE].astype(int).sum()} cases)")

    q2_model, q2_threshold, q2_name, q2_oof = _train_binary_stage_cv(
        "Q2",
        X,
        y_q2,
        feature_names,
        time_per_q,
        seed,
        class_labels=("clean", "conflict"),
        optimize_recall_class=1,  # optimize conflict (disputed) recall
    )

    # Use Q2 OOF predictions for routing (not hard ca_fired rule)
    has_conflict = q2_oof[:, 1] >= q2_threshold

    print(f"  ML router: {has_conflict.sum()} conflict, {(~has_conflict).sum()} clean")

    # Show class distribution in each path
    for path_name, mask in [("Conflict", has_conflict), ("Clean", ~has_conflict)]:
        dist = {lbl: (y_3class[mask] == lbl).sum() for lbl in _3CLASS_LABELS}
        print(f"  {path_name} path -- {dist}")

    # ================================================================
    # Q3: Is the conflict resolved? (trustworthy vs disputed, conflict path)
    # ================================================================
    print(f"\n{'='*60}")
    print("Q3: Is the conflict resolved? (conflict path: trustworthy vs disputed)")
    print(f"{'='*60}")

    # Q3 trains on ALL cases that could be disputes (ground truth), not just
    # Q2-routed cases. This gives Q3 maximum training signal. Q2 routing is
    # only used for the combined evaluation below.
    gt_has_conflict = y_3class == "disputed"
    # Also include cases where ca_fired detected conflict (even if not labeled disputed)
    gt_has_conflict = gt_has_conflict | (X[_CONFLICT_FEATURE].astype(int).values == 1)
    q3_mask = gt_has_conflict & (y_3class != "abstain")
    X_q3 = X[q3_mask]
    y_q3 = np.where(y_3class[q3_mask] == "trustworthy", 1, 0)

    n_abstain_removed = has_conflict.sum() - q3_mask.sum()
    print(
        f"  Total: {sum(y_q3 == 1)} resolved (trustworthy), {sum(y_q3 == 0)} unresolved (disputed)"
    )
    print(f"  Removed {n_abstain_removed} abstain cases from Q3 (noise)")

    q3_model, q3_threshold, q3_name, q3_oof = _train_binary_stage_cv(
        "Q3",
        X_q3,
        y_q3,
        feature_names,
        time_per_q,
        seed,
        class_labels=("unresolved", "resolved"),
        optimize_recall_class=1,  # optimize resolved (trustworthy) recall
    )

    # ================================================================
    # Q4: Is evidence truly solid? (trustworthy vs abstain, clean path)
    # ================================================================
    print(f"\n{'='*60}")
    print("Q4: Is evidence truly solid? (clean path: trustworthy vs abstain)")
    print(f"{'='*60}")

    # Q4 trains on all non-disputed cases (ground truth clean path).
    # Disputed cases in clean path get labeled 0 (insufficient -- safe fallback to abstain)
    q4_mask = ~gt_has_conflict
    X_q4 = X[q4_mask]
    y_q4 = np.where(y_3class[q4_mask] == "trustworthy", 1, 0)

    print(
        f"  Total: {sum(y_q4 == 1)} solid (trustworthy), {sum(y_q4 == 0)} insufficient (abstain+disputed)"
    )

    q4_model, q4_threshold, q4_name, q4_oof = _train_binary_stage_cv(
        "Q4",
        X_q4,
        y_q4,
        feature_names,
        time_per_q,
        seed,
        class_labels=("insufficient", "solid"),
        optimize_recall_class=0,  # optimize insufficient (abstain) recall
    )

    # ================================================================
    # Combined evaluation using OOF predictions
    # ================================================================
    print(f"\n{'='*60}")
    print("COMBINED CASCADE EVALUATION (out-of-fold)")
    print(f"{'='*60}")

    # Build cascade predictions from OOF probabilities
    n = len(X)
    final_oof_pred = np.full(n, "abstain", dtype=object)

    # Q1 OOF: answerable?
    q1_answerable = q1_oof[:, 1] >= q1_threshold

    # Q3 OOF: build full-size array, fill in conflict+non-abstain cases
    q3_indices = np.where(q3_mask)[0]
    q3_oof_full = np.full(n, 0.0)
    for i, idx in enumerate(q3_indices):
        q3_oof_full[idx] = q3_oof[i, 1]

    # Q4 OOF: build full-size array, fill in clean cases
    q4_indices = np.where(q4_mask)[0]
    q4_oof_full = np.full(n, 0.0)
    for i, idx in enumerate(q4_indices):
        q4_oof_full[idx] = q4_oof[i, 1]

    # Conflict path
    conflict_answerable = q1_answerable & has_conflict
    for idx in np.where(conflict_answerable)[0]:
        final_oof_pred[idx] = "trustworthy" if q3_oof_full[idx] >= q3_threshold else "disputed"

    # Clean path
    clean_answerable = q1_answerable & ~has_conflict
    for idx in np.where(clean_answerable)[0]:
        final_oof_pred[idx] = "trustworthy" if q4_oof_full[idx] >= q4_threshold else "abstain"

    combined_acc = accuracy_score(y_3class, final_oof_pred)
    print(
        f"\nCombined accuracy (OOF): {combined_acc:.4f} ({sum(y_3class == final_oof_pred)}/{len(y_3class)})"
    )
    print(classification_report(y_3class, final_oof_pred, labels=_3CLASS_LABELS, zero_division=0))

    cm = confusion_matrix(y_3class, final_oof_pred, labels=_3CLASS_LABELS)
    header = "predicted ->".rjust(20) + "".join(f"{lbl:>15}" for lbl in _3CLASS_LABELS)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(_3CLASS_LABELS):
        row = f"actual {label}".rjust(20) + "".join(f"{cm[i, j]:>15}" for j in range(3))
        print(row)

    # Safety metrics
    ft_abs = ((final_oof_pred == "trustworthy") & (y_3class == "abstain")).sum()
    ft_dis = ((final_oof_pred == "trustworthy") & (y_3class == "disputed")).sum()
    print(f"\nFalse Trustworthy: {ft_abs + ft_dis} ({ft_abs} abstain + {ft_dis} disputed)")

    # Per-class recall
    print("\nPer-class recall:")
    recalls = {}
    for label in _3CLASS_LABELS:
        total = cm[_3CLASS_LABELS.index(label)].sum()
        correct = cm[_3CLASS_LABELS.index(label), _3CLASS_LABELS.index(label)]
        recall = correct / total if total > 0 else 0
        recalls[label] = recall
        print(f"  {label:15s}: {recall:.4f} ({correct}/{total})")

    # Governor comparison (full dataset)
    if governor_preds is not None:
        gov_3class = _collapse_to_3class(governor_preds)
        gov_acc = accuracy_score(y_3class, gov_3class)
        print(f"\nGovernor baseline (3-class): {gov_acc:.4f}")
        print(f"Delta vs governor: +{combined_acc - gov_acc:.4f}")

    # ---- Build CV prediction map ----
    cv_predictions = {}
    if case_ids is not None:
        for i, cid in enumerate(case_ids.values):
            cv_predictions[cid] = final_oof_pred[i]

    # ---- Save artifact ----
    artifact = {
        "mode": "cascade",
        "q1_model": q1_model,
        "q1_name": f"{q1_name} (tuned)",
        "q1_threshold": q1_threshold,
        "q2_model": q2_model,
        "q2_name": f"{q2_name} (tuned)",
        "q2_threshold": q2_threshold,
        "q3_model": q3_model,
        "q3_name": f"{q3_name} (tuned)",
        "q3_threshold": q3_threshold,
        "q4_model": q4_model,
        "q4_name": f"{q4_name} (tuned)",
        "q4_threshold": q4_threshold,
        "conflict_feature": _CONFLICT_FEATURE,
        "combined_accuracy": combined_acc,
        "combined_recalls": recalls,
        "encoders": encoders,
        "feature_names": feature_names,
        "labels": _3CLASS_LABELS,
        "cv_predictions": cv_predictions,
        "cv_folds": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nCascade model saved to {output_path}")
    print(f"  Q1: {q1_name} (t={q1_threshold:.3f}) — evidence sufficient?")
    print(f"  Q2: {q2_name} (t={q2_threshold:.3f}) — material conflict?")
    print(f"  Q3: {q3_name} (t={q3_threshold:.3f}) — conflict resolved?")
    print(f"  Q4: {q4_name} (t={q4_threshold:.3f}) — evidence solid?")
    print(f"  Combined (OOF): {combined_acc:.4f}")
    if cv_predictions:
        print(f"  CV predictions: {len(cv_predictions)} cases (5-fold OOF)")

    return artifact


# ---------------------------------------------------------------------------
# Calibrate: safety-focused threshold tuning
# ---------------------------------------------------------------------------


def calibrate_thresholds(
    X: pd.DataFrame,
    y_4class: np.ndarray,
    feature_names: list[str],
    encoders: dict[str, LabelEncoder],
    model_path: Path,
    output_path: Path,
    seed: int = 42,
):
    """Calibrate thresholds to minimize dangerous false-trustworthy predictions.

    Loads a trained two-stage model, runs CV predictions on ALL data,
    then sweeps both thresholds jointly to find the operating point that
    minimizes cases where the model predicts trustworthy but the true label
    is abstain or disputed (the most dangerous failure mode).
    """
    y_3class = _collapse_to_3class(y_4class)

    print(f"Loading trained model from {model_path}...")
    artifact = joblib.load(model_path)
    s1_model = artifact["stage1_model"]
    s2_model = artifact["stage2_model"]
    s1_name = artifact.get("stage1_name", "unknown")
    s2_name = artifact.get("stage2_name", "unknown")
    old_s1_thresh = artifact.get("stage1_threshold", 0.5)
    old_s2_thresh = artifact.get("stage2_threshold", 0.5)
    print(f"  Stage 1: {s1_name} (threshold={old_s1_thresh:.3f})")
    print(f"  Stage 2: {s2_name} (threshold={old_s2_thresh:.3f})")

    # Stage 1: CV predictions on full dataset
    print(f"\nRunning Stage 1 CV predictions on {len(X)} samples...")
    y_s1 = np.array([0 if lbl == "abstain" else 1 for lbl in y_3class], dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    from sklearn.model_selection import cross_val_predict

    s1_cv_proba = cross_val_predict(s1_model, X, y_s1, cv=cv, method="predict_proba")

    # Stage 2: CV predictions on answerable subset
    answerable_mask = y_3class != "abstain"
    X_answerable = X[answerable_mask]
    y_answerable = np.array(
        [0 if lbl == "disputed" else 1 for lbl in y_3class[answerable_mask]], dtype=int
    )
    print(f"Running Stage 2 CV predictions on {len(X_answerable)} answerable samples...")
    s2_cv_proba = cross_val_predict(
        s2_model, X_answerable, y_answerable, cv=cv, method="predict_proba"
    )

    # Build full-dataset S2 probability array
    answerable_indices = np.where(answerable_mask)[0]
    s2_p_trust_full = np.full(len(y_3class), 0.0)
    for i, idx in enumerate(answerable_indices):
        s2_p_trust_full[idx] = s2_cv_proba[i, 1]
    # Abstain cases that leak through S1 get p=0.0 (always maps to disputed = safe)
    print(
        f"Precomputed S2 p(trustworthy) for {len(answerable_indices)} answerable samples "
        f"({sum(~answerable_mask)} abstain cases default to disputed)"
    )

    # Sweep both thresholds jointly
    print("\n" + "=" * 60)
    print("SAFETY CALIBRATION: minimizing false-trustworthy")
    print("=" * 60)

    # For each (s1_thresh, s2_thresh), simulate two-stage prediction on all data
    results = []
    p_ans = s1_cv_proba[:, 1]

    for s1_t in np.arange(0.40, 0.75, 0.005):
        for s2_t in np.arange(0.40, 0.80, 0.005):
            # Vectorized two-stage prediction
            final_pred = np.where(
                p_ans < s1_t,
                "abstain",
                np.where(s2_p_trust_full >= s2_t, "trustworthy", "disputed"),
            )

            # Count dangerous misfires: predicted trustworthy but actually abstain or disputed
            false_trust_abstain = int(
                ((final_pred == "trustworthy") & (y_3class == "abstain")).sum()
            )
            false_trust_disputed = int(
                ((final_pred == "trustworthy") & (y_3class == "disputed")).sum()
            )
            false_trustworthy = false_trust_abstain + false_trust_disputed

            accuracy = (final_pred == y_3class).mean()
            # Per-class recalls
            abs_total = (y_3class == "abstain").sum()
            disp_total = (y_3class == "disputed").sum()
            trust_total = (y_3class == "trustworthy").sum()
            abs_recall = (
                ((final_pred[y_3class == "abstain"] == "abstain").sum() / abs_total)
                if abs_total
                else 0
            )
            disp_recall = (
                ((final_pred[y_3class == "disputed"] == "disputed").sum() / disp_total)
                if disp_total
                else 0
            )
            trust_recall = (
                ((final_pred[y_3class == "trustworthy"] == "trustworthy").sum() / trust_total)
                if trust_total
                else 0
            )

            results.append(
                {
                    "s1_t": s1_t,
                    "s2_t": s2_t,
                    "accuracy": accuracy,
                    "false_trustworthy": false_trustworthy,
                    "false_trust_abstain": false_trust_abstain,
                    "false_trust_disputed": false_trust_disputed,
                    "abs_recall": abs_recall,
                    "disp_recall": disp_recall,
                    "trust_recall": trust_recall,
                }
            )

    # Strategy: minimize false_trustworthy, with accuracy >= 0.78 and trust_recall >= 0.70
    eligible = [r for r in results if r["accuracy"] >= 0.78 and r["trust_recall"] >= 0.70]
    if not eligible:
        # Relax: accuracy >= 0.75
        eligible = [r for r in results if r["accuracy"] >= 0.75 and r["trust_recall"] >= 0.65]
    if not eligible:
        eligible = results

    # Primary: minimize false_trustworthy. Tiebreak: maximize accuracy.
    best = min(eligible, key=lambda r: (r["false_trustworthy"], -r["accuracy"]))

    print(f"\nBefore calibration (s1={old_s1_thresh:.3f}, s2={old_s2_thresh:.3f}):")
    old = next(
        r
        for r in results
        if abs(r["s1_t"] - old_s1_thresh) < 0.003 and abs(r["s2_t"] - old_s2_thresh) < 0.003
    )
    print(f"  Accuracy:           {old['accuracy']:.4f}")
    print(
        f"  False trustworthy:  {old['false_trustworthy']} ({old['false_trust_abstain']} abstain, {old['false_trust_disputed']} disputed)"
    )
    print(f"  Abstain recall:     {old['abs_recall']:.4f}")
    print(f"  Disputed recall:    {old['disp_recall']:.4f}")
    print(f"  Trustworthy recall: {old['trust_recall']:.4f}")

    print(f"\nAfter calibration (s1={best['s1_t']:.3f}, s2={best['s2_t']:.3f}):")
    print(f"  Accuracy:           {best['accuracy']:.4f}")
    print(
        f"  False trustworthy:  {best['false_trustworthy']} ({best['false_trust_abstain']} abstain, {best['false_trust_disputed']} disputed)"
    )
    print(f"  Abstain recall:     {best['abs_recall']:.4f}")
    print(f"  Disputed recall:    {best['disp_recall']:.4f}")
    print(f"  Trustworthy recall: {best['trust_recall']:.4f}")

    delta_ft = best["false_trustworthy"] - old["false_trustworthy"]
    delta_acc = best["accuracy"] - old["accuracy"]
    print(f"\n  Delta false trustworthy: {delta_ft:+d}")
    print(f"  Delta accuracy:          {delta_acc:+.4f}")

    # Show top 5 alternatives for comparison
    eligible_sorted = sorted(eligible, key=lambda r: (r["false_trustworthy"], -r["accuracy"]))
    print("\nTop 5 operating points (sorted by fewest false trustworthy):")
    print(
        f"  {'s1':>6s}  {'s2':>6s}  {'acc':>6s}  {'FT':>4s}  {'abs_r':>6s}  {'dis_r':>6s}  {'tru_r':>6s}"
    )
    for r in eligible_sorted[:5]:
        marker = " <-- selected" if r is best else ""
        print(
            f"  {r['s1_t']:6.3f}  {r['s2_t']:6.3f}  {r['accuracy']:6.4f}  {r['false_trustworthy']:4d}  "
            f"{r['abs_recall']:6.4f}  {r['disp_recall']:6.4f}  {r['trust_recall']:6.4f}{marker}"
        )

    # Save calibrated artifact
    calibrated = dict(artifact)
    calibrated["stage1_threshold"] = float(best["s1_t"])
    calibrated["stage2_threshold"] = float(best["s2_t"])
    calibrated["calibration"] = {
        "false_trustworthy": best["false_trustworthy"],
        "false_trust_abstain": best["false_trust_abstain"],
        "false_trust_disputed": best["false_trust_disputed"],
        "accuracy": best["accuracy"],
        "recalls": {
            "abstain": best["abs_recall"],
            "disputed": best["disp_recall"],
            "trustworthy": best["trust_recall"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, output_path)
    print(f"\nCalibrated model saved to {output_path}")
    print(f"  s1_threshold: {best['s1_t']:.3f} (was {old_s1_thresh:.3f})")
    print(f"  s2_threshold: {best['s2_t']:.3f} (was {old_s2_thresh:.3f})")

    return calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train governance classifier")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input features CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output model path")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_FITZGOV_DIR,
        help="fitz-gov tier1_core dir for context features",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument(
        "--time-budget",
        type=int,
        default=120,
        help="Hyperparameter search time budget in seconds (default: 120)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-context-features", action="store_true", help="Skip context feature enrichment"
    )
    parser.add_argument(
        "--mode",
        choices=["4class", "twostage", "calibrate", "cascade"],
        default="4class",
        help="Training mode: 4class (legacy), twostage, cascade (4-question atomic), or calibrate",
    )
    args = parser.parse_args()

    if args.output is None:
        if args.mode == "calibrate":
            args.output = _DEFAULT_CALIBRATED_OUTPUT
        elif args.mode == "twostage":
            args.output = _DEFAULT_TWOSTAGE_OUTPUT
        elif args.mode == "cascade":
            args.output = _CASCADE_OUTPUT
        else:
            args.output = _DEFAULT_MODEL_OUTPUT

    # Load data
    print(f"Loading features from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    print(f"Class distribution:\n{df['expected_mode'].value_counts().to_string()}\n")

    # Enrich with context features
    if not args.no_context_features and args.data_dir.exists():
        df = enrich_with_context_features(df, args.data_dir)
    elif not args.no_context_features:
        print(f"WARNING: {args.data_dir} not found, skipping context features")

    # Calibrate mode: safety-focused threshold tuning on existing model
    if args.mode == "calibrate":
        X, encoders = prepare_features(df)
        feature_names = list(X.columns)
        y_4class = df["expected_mode"].values
        calibrate_thresholds(
            X,
            y_4class,
            feature_names,
            encoders,
            model_path=_DEFAULT_TWOSTAGE_OUTPUT,
            output_path=args.output,
            seed=args.seed,
        )
        return

    # Two-stage mode: delegate to dedicated function
    if args.mode == "twostage":
        X, encoders = prepare_features(df)
        feature_names = list(X.columns)
        y_4class = df["expected_mode"].values
        governor_preds = (
            df["governor_predicted"].values if "governor_predicted" in df.columns else None
        )
        train_twostage(
            X,
            y_4class,
            feature_names,
            encoders,
            args.test_size,
            args.time_budget,
            args.seed,
            args.output,
            governor_preds,
            case_ids=df["case_id"],
        )
        return

    # Cascade mode: 4-question atomic decomposition
    if args.mode == "cascade":
        X, encoders = prepare_features(df)
        feature_names = list(X.columns)
        y_4class = df["expected_mode"].values
        governor_preds = (
            df["governor_predicted"].values if "governor_predicted" in df.columns else None
        )
        train_cascade(
            X,
            y_4class,
            feature_names,
            encoders,
            args.time_budget,
            args.seed,
            args.output,
            governor_preds,
            case_ids=df["case_id"],
        )
        return

    # Prepare features
    X, encoders = prepare_features(df)
    y = df["expected_mode"]
    feature_names = list(X.columns)
    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    gov_test = df.loc[y_test.index, "governor_predicted"]
    labels = sorted(y.unique())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # Step 1: Quick model comparison
    comparison = quick_model_comparison(X_train, y_train, cv, args.seed)

    # Step 2: Pick top 3 models for hyperparameter search
    ranked = sorted(comparison.items(), key=lambda x: x[1]["mean"], reverse=True)
    top_n = min(3, len(ranked))
    top_models = ranked[:top_n]
    print(f"\nTop {top_n} models for hyperparameter search: {[n for n, _ in top_models]}")

    # Step 3: Hyperparameter search
    tuned = hyperparameter_search(X_train, y_train, cv, top_models, args.time_budget, args.seed)

    # Step 4: Build stacking ensemble
    ensemble = build_ensemble(tuned, X_train, y_train, args.seed)

    # Evaluate everything on test set
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")

    # Governor baseline
    gov_acc = accuracy_score(y_test, gov_test)
    print(f"\nGovernor baseline: {gov_acc:.3f} ({sum(y_test == gov_test)}/{len(y_test)})")
    print(classification_report(y_test, gov_test, labels=labels, zero_division=0))

    # Each tuned model
    best_acc = 0.0
    best_name = ""
    best_model = None
    for name, info in tuned.items():
        _, acc = evaluate_model(f"{name} (tuned)", info["model"], X_test, y_test, labels)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = info["model"]

    # Ensemble
    _, ens_acc = evaluate_model("Stacking Ensemble", ensemble, X_test, y_test, labels)

    # Pick winner
    if ens_acc >= best_acc:
        winner_name = "Stacking Ensemble"
        winner_model = ensemble
        winner_acc = ens_acc
    else:
        winner_name = f"{best_name} (tuned)"
        winner_model = best_model
        winner_acc = best_acc

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Governor baseline:  {gov_acc:.3f}")
    for name, info in tuned.items():
        acc = accuracy_score(y_test, info["model"].predict(X_test))
        print(f"  {name:20s}: {acc:.3f}")
    print(f"  {'Ensemble':20s}: {ens_acc:.3f}")
    print(f"  Winner: {winner_name} ({winner_acc:.3f}, +{winner_acc - gov_acc:.3f} vs governor)")

    # Feature importance (if tree-based winner)
    if hasattr(winner_model, "feature_importances_"):
        importances = winner_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\n--- Top 20 features by importance ({winner_name}) ---")
        for rank, idx in enumerate(indices[:20], 1):
            print(f"  {rank:2d}. {feature_names[idx]:<40s} {importances[idx]:.4f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": winner_model,
        "encoders": encoders,
        "feature_names": feature_names,
        "labels": labels,
        "accuracy": winner_acc,
        "model_name": winner_name,
    }
    joblib.dump(artifact, args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
