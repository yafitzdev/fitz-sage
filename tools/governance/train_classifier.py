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
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_INPUT = _DATA_DIR / "features.csv"
_DEFAULT_MODEL_OUTPUT = _DATA_DIR / "model_v1.joblib"
_DEFAULT_TWOSTAGE_OUTPUT = _DATA_DIR / "model_v5_twostage.joblib"
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

# Categorical features that need label encoding
_CATEGORICAL_FEATURES = {
    "ie_signal",
    "ca_signal",
    "ca_first_evidence_char",
    "ca_evidence_characters",
    "caa_query_type",
    "sit_info_type_requested",
}

# Boolean features to convert to int
_BOOL_FEATURES = {
    "ie_fired",
    "ca_fired",
    "ca_numerical_variance_detected",
    "caa_fired",
    "caa_has_causal_evidence",
    "caa_has_predictive_evidence",
    "sit_fired",
    "sit_entity_mismatch",
    "sit_has_specific_info",
    "has_qualified_signal",
    "detection_temporal",
    "detection_comparison",
    "has_distinct_years",
}

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

    return features


def enrich_with_context_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add context-based features by loading original fitz-gov cases.

    Skips features that are already present in the DataFrame (e.g. when
    feature_extractor.py already computed them during eval_pipeline extraction).
    """
    # Check if ctx_* features are already in the CSV (from feature_extractor.py)
    ctx_cols = [c for c in df.columns if c.startswith("ctx_")]
    if ctx_cols:
        print(
            f"  Context features already present in CSV ({len(ctx_cols)} cols), skipping enrichment"
        )
        return df

    print(f"Computing context features from {data_dir}...")
    cases = load_cases_by_id(data_dir)

    new_features = []
    for _, row in df.iterrows():
        case = cases.get(row["case_id"], {})
        feats = compute_context_features(case.get("query", ""), case.get("contexts", []))
        new_features.append(feats)

    ctx_df = pd.DataFrame(new_features)
    n_new = len(ctx_df.columns)
    print(f"  Added {n_new} context features: {list(ctx_df.columns)}")
    return pd.concat([df, ctx_df], axis=1)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Prepare feature matrix: encode categoricals, convert bools to int."""
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].copy()
    encoders: dict[str, LabelEncoder] = {}

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
        X[col] = (
            X[col].map({"True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(int)
        )

    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

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
    sample_weights = compute_sample_weight("balanced", y_train)
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
    header = "predicted ->".rjust(20) + "".join(f"{l:>12}" for l in labels)
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
        ["trustworthy" if l in ("confident", "qualified") else l for l in labels],
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
):
    """Train a two-stage binary classifier: Stage 1 (answerable vs abstain) → Stage 2 (trustworthy vs disputed)."""

    y_3class = _collapse_to_3class(y_4class)

    # Stage 1 labels: answerable vs abstain
    y_s1 = np.array(
        ["abstain" if l == "abstain" else "answerable" for l in y_3class],
        dtype=object,
    )

    # Stratified split (stratify on 3-class to preserve disputed proportion)
    X_train, X_test, y3_train, y3_test = train_test_split(
        X,
        y_3class,
        test_size=test_size,
        random_state=seed,
        stratify=y_3class,
    )
    y_s1_train = np.array(
        ["abstain" if l == "abstain" else "answerable" for l in y3_train],
        dtype=object,
    )
    y_s1_test = np.array(
        ["abstain" if l == "abstain" else "answerable" for l in y3_test],
        dtype=object,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    time_per_stage = time_budget // 2

    # ---- Stage 1: answerable vs abstain ----
    print(f"\n{'='*60}")
    print("STAGE 1: answerable vs abstain")
    print(f"{'='*60}")
    print(
        f"Train: {sum(y_s1_train == 'answerable')} answerable, {sum(y_s1_train == 'abstain')} abstain"
    )
    print(
        f"Test:  {sum(y_s1_test == 'answerable')} answerable, {sum(y_s1_test == 'abstain')} abstain"
    )

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

    print("\n--- Stage 1: Quick comparison ---")
    s1_results = {}
    for name, model in s1_models.items():
        cv_scores = cross_val_score(model, X_train, y_s1_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_s1_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_s1_test, pred)
        cm = confusion_matrix(y_s1_test, pred, labels=["abstain", "answerable"])
        abs_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
        ans_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
        s1_results[name] = {
            "model": model,
            "acc": acc,
            "cv": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }
        print(
            f"  {name:6s}: test={acc:.4f}, CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})"
            f"  abstain={abs_recall:.3f} answerable={ans_recall:.3f}"
        )

    # Hyperparameter search on top Stage 1 model
    s1_ranked = sorted(s1_results.items(), key=lambda x: x[1]["acc"], reverse=True)
    s1_best_name = s1_ranked[0][0]
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
    }

    s1_base = {
        "RF": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "ET": ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
        "GBT": GradientBoostingClassifier(random_state=seed),
    }

    n_iter = max(10, int(time_per_stage / 3))
    s1_search = RandomizedSearchCV(
        s1_base[s1_best_name],
        s1_param_grids[s1_best_name],
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1 if s1_best_name != "GBT" else 1,
    )
    s1_search.fit(X_train, y_s1_train)
    s1_model = s1_search.best_estimator_

    s1_pred_test = s1_model.predict(X_test)
    s1_acc = accuracy_score(y_s1_test, s1_pred_test)
    cm_s1 = confusion_matrix(y_s1_test, s1_pred_test, labels=["abstain", "answerable"])
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

    answerable_train = y_s1_train == "answerable"
    X_train_s2 = X_train[answerable_train]
    y_train_s2 = y3_train[answerable_train]
    print(
        f"Train: {sum(y_train_s2 == 'trustworthy')} trustworthy, {sum(y_train_s2 == 'disputed')} disputed"
    )

    # Full answerable data for CV
    answerable_all = (
        np.array(
            ["abstain" if l == "abstain" else "answerable" for l in y_3class],
            dtype=object,
        )
        == "answerable"
    )
    X_answerable = X[answerable_all]
    y_answerable = y_3class[answerable_all]

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

    print("\n--- Stage 2: Quick comparison ---")
    s2_results = {}
    for name, model in s2_models.items():
        cv_scores = cross_val_score(model, X_answerable, y_answerable, cv=cv, scoring="accuracy")
        model.fit(X_train_s2, y_train_s2)
        s2_results[name] = {"model": model, "cv": cv_scores.mean(), "cv_std": cv_scores.std()}
        print(f"  {name:6s}: CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

    # Tune top Stage 2 model
    s2_ranked = sorted(s2_results.items(), key=lambda x: x[1]["cv"], reverse=True)
    s2_best_name = s2_ranked[0][0]
    print(f"\n--- Stage 2: Tuning {s2_best_name} ({time_per_stage}s budget) ---")

    s2_search = RandomizedSearchCV(
        s1_base[s2_best_name],  # reuse base model constructors
        s1_param_grids[s2_best_name],
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=seed,
        n_jobs=-1 if s2_best_name != "GBT" else 1,
    )
    s2_search.fit(X_train_s2, y_train_s2)
    s2_model = s2_search.best_estimator_
    print(f"  Best CV: {s2_search.best_score_:.4f}")
    print(f"  Params: {s2_search.best_params_}")

    if hasattr(s2_model, "feature_importances_"):
        imp = s2_model.feature_importances_
        order = np.argsort(imp)[::-1]
        print("\n  Stage 2 top 10 features:")
        for rank, idx in enumerate(order[:10], 1):
            print(f"    {rank:2d}. {feature_names[idx]:40s} {imp[idx]:.4f}")

    # ---- Combined evaluation ----
    print(f"\n{'='*60}")
    print("COMBINED TWO-STAGE EVALUATION")
    print(f"{'='*60}")

    final_pred = np.full(len(y3_test), "abstain", dtype=object)
    s1_answerable_mask = s1_pred_test == "answerable"
    if s1_answerable_mask.any():
        s2_preds = s2_model.predict(X_test[s1_answerable_mask])
        answerable_positions = np.where(s1_answerable_mask)[0]
        for i, pos in enumerate(answerable_positions):
            final_pred[pos] = s2_preds[i]

    combined_acc = accuracy_score(y3_test, final_pred)
    print(f"\nCombined accuracy: {combined_acc:.4f} ({sum(y3_test == final_pred)}/{len(y3_test)})")
    print(classification_report(y3_test, final_pred, labels=_3CLASS_LABELS, zero_division=0))

    cm_combined = confusion_matrix(y3_test, final_pred, labels=_3CLASS_LABELS)
    header = "predicted ->".rjust(20) + "".join(f"{l:>15}" for l in _3CLASS_LABELS)
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

    # ---- Save artifact ----
    artifact = {
        "mode": "twostage",
        "stage1_model": s1_model,
        "stage1_name": f"{s1_best_name} (tuned)",
        "stage1_accuracy": s1_acc,
        "stage2_model": s2_model,
        "stage2_name": f"{s2_best_name} (tuned)",
        "stage2_cv": s2_search.best_score_,
        "combined_accuracy": combined_acc,
        "combined_recalls": recalls,
        "encoders": encoders,
        "feature_names": feature_names,
        "labels": _3CLASS_LABELS,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nTwo-stage model saved to {output_path}")
    print(f"  Stage 1: {s1_best_name} (tuned) — answerable vs abstain")
    print(f"  Stage 2: {s2_best_name} (tuned) — trustworthy vs disputed")
    print(f"  Combined: {combined_acc:.4f}")

    return artifact


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
        choices=["4class", "twostage"],
        default="4class",
        help="Training mode: 4class (legacy) or twostage (recommended)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = _DEFAULT_TWOSTAGE_OUTPUT if args.mode == "twostage" else _DEFAULT_MODEL_OUTPUT

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
