# tools/detection/train_classifier.py
"""
Training script for the fitz-ai ML detection classifier.

Loads tier1_core JSON files from fitz-gov, derives binary labels for
temporal and comparison query types, trains a MultiOutputClassifier backed
by LogisticRegression, calibrates per-label recall thresholds to >= 0.90,
and saves the final artifact to:
    fitz_ai/retrieval/detection/data/model_v1_detection.joblib

Usage:
    python -m tools.detection.train_classifier
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multioutput import MultiOutputClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_DATA_DIR = _SCRIPT_DIR / "../../../../fitz-gov/data/tier1_core"
_OUTPUT_PATH = (
    _SCRIPT_DIR / "../../fitz_ai/retrieval/detection/data/model_v1_detection.joblib"
).resolve()

_TIER1_GLOB = "*.json"

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
                }
            )
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} cases from {len(files)} file(s)")
    return df


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary label columns: temporal, comparison."""
    df = df.copy()
    df["temporal"] = (
        (df["reasoning_type"] == "temporal") | (df["query_type"] == "when")
    ).astype(int)
    df["comparison"] = (
        (df["reasoning_type"] == "comparative") | (df["query_type"] == "compare")
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def calibrate_threshold(y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = 0.90) -> float:
    """Find the lowest threshold where recall >= target_recall."""
    thresholds = np.linspace(0.0, 1.0, 201)
    best = 0.5  # default fallback
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


def _binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Return (recall, precision, f1) for binary arrays."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return recall, precision, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    labels = ["temporal", "comparison"]

    # Load data
    df = load_cases(_DATA_DIR)
    df = derive_labels(df)

    # Print label distribution
    print("\n--- Label distribution ---")
    for label in labels:
        pos = int(df[label].sum())
        print(f"  {label}: {pos} positive / {len(df) - pos} negative")

    X_text = df["query"].tolist()
    Y = df[labels].values  # shape (n, 2)

    # Stratify target: temporal OR comparison
    stratify_col = (df["temporal"] | df["comparison"]).astype(int).values

    # Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, sublinear_tf=True)
    X = vectorizer.fit_transform(X_text)

    # Model
    base_clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    model = MultiOutputClassifier(base_clf)

    # 5-fold stratified CV for metrics + probability calibration
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- 5-fold CV metrics (threshold=0.5) ---")
    oof_probas: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, stratify_col), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        fold_model = MultiOutputClassifier(
            LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        )
        fold_model.fit(X_train, Y_train)

        # Collect OOF probabilities: shape (n_val, 2) — P(positive) per label
        fold_probas = np.column_stack(
            [est.predict_proba(X_val)[:, 1] for est in fold_model.estimators_]
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
    X_final = final_vectorizer.fit_transform(X_text)
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
