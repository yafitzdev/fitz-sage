# tools/detection/train_classifier.py
"""
Training script for the fitz-sage ML detection classifier.

Loads tier1_core JSON files from fitz-gov, derives binary labels for
temporal, comparison, aggregation, and freshness query types, trains a
MultiOutputClassifier backed by LogisticRegression, calibrates per-label
recall thresholds to >= 0.90, and saves the final artifact to:
    fitz_sage/retrieval/detection/data/model_v1_detection.joblib

Usage:
    python -m tools.detection.train_classifier
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_DATA_DIR = _SCRIPT_DIR / "../../../../fitz-gov/data/tier1_core"
_OUTPUT_PATH = (
    _SCRIPT_DIR / "../../fitz_sage/retrieval/detection/data/model_v1_detection.joblib"
).resolve()

_TIER1_GLOB = "*.json"

# Keyword patterns — high signal for short queries
_TEMPORAL_RE = re.compile(
    r"\b(when|before|after|since|until|between|during|last|first|previous|next|recent|"
    r"quarter|q1|q2|q3|q4|annually|monthly|weekly|yearly|decade|century|era|period|"
    r"timeline|history|trend|over time|change.+over|compare.+year|from \d{4}|in \d{4}|"
    r"as of|by \d{4}|prior to|following|subsequent)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(vs|versus|compare|comparison|difference|differ|better|worse|than|"
    r"contrast|distinguish|similarities|alike|both|either|which is|how does.+compare|"
    r"pros and cons|advantages|disadvantages|trade.?off|relative to|over)\b",
    re.IGNORECASE,
)
_AGGREGATION_RE = re.compile(
    r"\b(list all|list every|show all|how many|count of|enumerate|what are all|every)\b",
    re.IGNORECASE,
)
_FRESHNESS_RE = re.compile(
    r"\b(latest|most recent|current|right now|today|newest|recently|new version|"
    r"official|best practice|recommended|standard|proper way|authoritative)\b",
    re.IGNORECASE,
)


def keyword_features(queries: list[str]) -> sp.csr_matrix:
    """Return (n, 4) sparse matrix of keyword indicator features."""
    temporal = np.array([1 if _TEMPORAL_RE.search(q) else 0 for q in queries], dtype=np.float32)
    comparison = np.array([1 if _COMPARISON_RE.search(q) else 0 for q in queries], dtype=np.float32)
    aggregation = np.array(
        [1 if _AGGREGATION_RE.search(q) else 0 for q in queries], dtype=np.float32
    )
    freshness = np.array([1 if _FRESHNESS_RE.search(q) else 0 for q in queries], dtype=np.float32)
    return sp.csr_matrix(np.column_stack([temporal, comparison, aggregation, freshness]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cases(data_dir: Path) -> pd.DataFrame:
    """Load all tier1_core JSON files and return a DataFrame of cases."""
    data_dir = data_dir.resolve()
    files = sorted(data_dir.glob(_TIER1_GLOB))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            rows.append(
                {
                    "id": case.get("id", ""),
                    "query": case.get("query", ""),
                    "reasoning_type": case.get("reasoning_type", ""),
                    "query_type": case.get("query_type", ""),
                    "detection_labels": case.get("detection_labels", []),
                }
            )
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} cases from {len(files)} file(s)")
    return df


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary label columns: temporal, comparison, aggregation, freshness."""
    df = df.copy()
    df["temporal"] = (
        (df["reasoning_type"] == "temporal")
        | (df["query_type"] == "when")
        | df["detection_labels"].apply(lambda x: "temporal" in x)
    ).astype(int)
    df["comparison"] = (
        (df["reasoning_type"] == "comparative")
        | (df["query_type"] == "compare")
        | df["detection_labels"].apply(lambda x: "comparison" in x)
    ).astype(int)
    df["aggregation"] = df["detection_labels"].apply(lambda x: int("aggregation" in x))
    df["freshness"] = df["detection_labels"].apply(lambda x: int("freshness" in x))
    return df


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def calibrate_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = 0.90
) -> float:
    """Find the highest threshold where recall is still >= target_recall."""
    thresholds = np.linspace(1.0, 0.0, 201)  # sweep high → low
    best = 0.0  # fallback: predict everything positive
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if recall >= target_recall:
            best = float(t)
            break
    return best


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (recall, precision, f1) for binary arrays."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fitzgov-dir", type=Path, default=_DATA_DIR)
    args = parser.parse_args()

    labels = ["temporal", "comparison", "aggregation", "freshness"]

    # Load data
    df = load_cases(args.fitzgov_dir)
    df = derive_labels(df)

    # Print label distribution
    print("\n--- Label distribution ---")
    for label in labels:
        pos = int(df[label].sum())
        print(f"  {label}: {pos} positive / {len(df) - pos} negative")

    X_text = df["query"].tolist()
    Y = df[labels].values  # shape (n, 4)

    # Stratify target: any positive label
    stratify_col = (
        (df["temporal"] | df["comparison"] | df["aggregation"] | df["freshness"]).astype(int).values
    )

    # Features: TF-IDF + keyword indicators
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
    X_tfidf = vectorizer.fit_transform(X_text)
    X_kw = keyword_features(X_text)
    X = sp.hstack([X_tfidf, X_kw], format="csr")

    # 5-fold stratified CV for metrics + probability calibration
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- 5-fold CV metrics (threshold=0.5) ---")
    oof_probas: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, stratify_col), 1):
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Re-fit vectorizer on train split only (keyword features are fit-free)
        fold_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
        X_train_tfidf = fold_vec.fit_transform([X_text[i] for i in train_idx])
        X_val_tfidf = fold_vec.transform([X_text[i] for i in val_idx])
        X_train_fold = sp.hstack([X_train_tfidf, X_kw[train_idx]], format="csr")
        X_val_fold = sp.hstack([X_val_tfidf, X_kw[val_idx]], format="csr")

        fold_model = MultiOutputClassifier(
            LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        )
        fold_model.fit(X_train_fold, Y_train)

        # Collect OOF probabilities: shape (n_val, 4) — P(positive) per label
        fold_probas = np.column_stack(
            [est.predict_proba(X_val_fold)[:, 1] for est in fold_model.estimators_]
        )
        oof_probas.append(fold_probas)
        oof_y.append(Y_val)

    oof_probas_arr = np.vstack(oof_probas)  # (n_samples, n_labels)
    oof_y_arr = np.vstack(oof_y)  # (n_samples, n_labels)

    cv_recalls: dict[str, float] = {}
    for i, label in enumerate(labels):
        y_true = oof_y_arr[:, i]
        y_pred = (oof_probas_arr[:, i] >= 0.5).astype(int)
        recall, precision, f1 = _binary_metrics(y_true, y_pred)
        cv_recalls[label] = recall
        print(f"  {label}: recall={recall:.3f}  precision={precision:.3f}  f1={f1:.3f}")

    # Threshold calibration
    print("\n--- Calibrated thresholds (target recall >= 0.90) ---")
    thresholds: dict[str, float] = {}
    for i, label in enumerate(labels):
        y_true = oof_y_arr[:, i]
        y_proba = oof_probas_arr[:, i]
        t = calibrate_threshold(y_true, y_proba, target_recall=0.90)
        thresholds[label] = t
        # Report recall at calibrated threshold
        y_pred_cal = (y_proba >= t).astype(int)
        recall_cal, _, _ = _binary_metrics(y_true, y_pred_cal)
        print(f"  {label}: threshold={t:.3f}  recall@threshold={recall_cal:.3f}")

    # Train final model on all data
    print("\n--- Training final model on all data ---")
    final_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
    X_final_tfidf = final_vectorizer.fit_transform(X_text)
    X_final = sp.hstack([X_final_tfidf, X_kw], format="csr")
    final_model = MultiOutputClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    )
    final_model.fit(X_final, Y)

    # Save artifact
    artifact = {
        "model": final_model,
        "vectorizer": final_vectorizer,
        "labels": labels,
        "thresholds": thresholds,
    }
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, _OUTPUT_PATH)
    print(f"\nArtifact saved to: {_OUTPUT_PATH}")

    # Summary
    print("\n=== Summary ===")
    print(f"  Total samples: {len(df)}")
    for label in labels:
        print(f"  {label}: CV recall={cv_recalls[label]:.3f}  threshold={thresholds[label]:.3f}")


if __name__ == "__main__":
    main()
