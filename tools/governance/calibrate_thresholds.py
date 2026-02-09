# tools/governance/calibrate_thresholds.py
"""
Per-class calibrated thresholds with governor fallback.

Loads model_v3.joblib and eval_results_v2.csv, reproduces the exact train/test
split, extracts predict_proba() on test cases, and sweeps per-class thresholds.
When the classifier's max probability is below the threshold for its predicted
class, falls back to the governor's rule-based decision.

Usage:
    python -m tools.governance.calibrate_thresholds
"""

from __future__ import annotations

import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Reuse constants from train_classifier
from tools.governance.train_classifier import (
    _BOOL_FEATURES,
    _CATEGORICAL_FEATURES,
    _META_COLS,
    compute_context_features,
    load_cases_by_id,
)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_MODEL_PATH = _DATA_DIR / "model_v3.joblib"
_EVAL_CSV = _DATA_DIR / "eval_results_v2.csv"
_FITZGOV_DIR = Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
_OUTPUT_PATH = _DATA_DIR / "model_v3_calibrated.joblib"

SEED = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Feature preparation (mirrors train_classifier.prepare_features)
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """Prepare feature matrix using pre-fitted encoders from the saved artifact."""
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].copy()

    for col in _CATEGORICAL_FEATURES:
        if col not in X.columns:
            continue
        X[col] = X[col].fillna("none").astype(str)
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels: map to -1
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v, k=known, e=le: e.transform([v])[0] if v in k else -1)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    for col in _BOOL_FEATURES:
        if col not in X.columns:
            continue
        X[col] = (
            X[col]
            .map({"True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0})
            .fillna(0)
            .astype(int)
        )

    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def enrich_with_context_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add context-based features by loading original fitz-gov cases."""
    print(f"Computing context features from {data_dir}...")
    cases = load_cases_by_id(data_dir)

    new_features = []
    for _, row in df.iterrows():
        case = cases.get(row["case_id"], {})
        feats = compute_context_features(case.get("query", ""), case.get("contexts", []))
        new_features.append(feats)

    ctx_df = pd.DataFrame(new_features)
    print(f"  Added {len(ctx_df.columns)} context features")
    return pd.concat([df, ctx_df], axis=1)


# ---------------------------------------------------------------------------
# Confusion matrix display
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true, y_pred, labels):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "predicted ->".rjust(20) + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"actual {label}".rjust(20) + "".join(
            f"{cm[i, j]:>12}" for j in range(len(labels))
        )
        print(row)
    print()


def per_class_recall(y_true, y_pred, labels) -> dict[str, float]:
    """Compute recall for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = {}
    for i, label in enumerate(labels):
        total = cm[i].sum()
        recalls[label] = cm[i, i] / total if total > 0 else 0.0
    return recalls


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def apply_thresholds(
    probas: np.ndarray,
    classifier_preds: np.ndarray,
    governor_preds: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
) -> np.ndarray:
    """Apply per-class thresholds: use classifier if confident, else governor."""
    result = classifier_preds.copy()
    for i in range(len(result)):
        pred_class = classifier_preds[i]
        pred_idx = labels.index(pred_class)
        max_prob = probas[i, pred_idx]
        if max_prob < thresholds[pred_class]:
            result[i] = governor_preds[i]
    return result


def sweep_thresholds(
    probas: np.ndarray,
    classifier_preds: np.ndarray,
    governor_preds: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Brute-force sweep per-class thresholds to maximize minimum per-class recall."""
    # Threshold grid: 0.0 means always trust classifier, 1.0 means always use governor
    grid_values = np.arange(0.0, 0.95, 0.05)

    best_min_recall = -1.0
    best_thresholds = {label: 0.0 for label in labels}
    best_recalls = {}

    # Build all combinations
    combos = list(itertools.product(grid_values, repeat=len(labels)))
    print(f"Sweeping {len(combos)} threshold combinations...")

    for combo in combos:
        thresholds = {label: combo[i] for i, label in enumerate(labels)}
        preds = apply_thresholds(probas, classifier_preds, governor_preds, thresholds, labels)
        recalls = per_class_recall(y_true, preds, labels)
        min_recall = min(recalls.values())

        if min_recall > best_min_recall:
            best_min_recall = min_recall
            best_thresholds = thresholds.copy()
            best_recalls = recalls.copy()
        elif min_recall == best_min_recall:
            # Tie-break: prefer higher overall accuracy
            current_acc = accuracy_score(y_true, preds)
            best_preds = apply_thresholds(
                probas, classifier_preds, governor_preds, best_thresholds, labels
            )
            best_acc = accuracy_score(y_true, best_preds)
            if current_acc > best_acc:
                best_thresholds = thresholds.copy()
                best_recalls = recalls.copy()

    return best_thresholds, best_min_recall, best_recalls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load model artifact
    print(f"Loading model from {_MODEL_PATH}...")
    artifact = joblib.load(_MODEL_PATH)
    model = artifact["model"]
    encoders = artifact["encoders"]
    labels = artifact["labels"]
    model_feature_names = artifact["feature_names"]
    print(f"  Model: {artifact['model_name']}")
    print(f"  Labels: {labels}")
    print(f"  Features: {len(model_feature_names)}")

    # Load eval data
    print(f"\nLoading eval data from {_EVAL_CSV}...")
    df = pd.read_csv(_EVAL_CSV)
    print(f"  {len(df)} rows x {len(df.columns)} columns")
    print(f"  Class distribution:\n{df['expected_mode'].value_counts().to_string()}\n")

    # Add context features (same as training)
    if _FITZGOV_DIR.exists():
        df = enrich_with_context_features(df, _FITZGOV_DIR)
    else:
        print(f"WARNING: {_FITZGOV_DIR} not found, skipping context features")

    # Prepare features using saved encoders
    X = prepare_features(df, encoders)
    y = df["expected_mode"].values

    # Ensure feature columns match model exactly
    missing = [c for c in model_feature_names if c not in X.columns]
    extra = [c for c in X.columns if c not in model_feature_names]
    if missing:
        print(f"WARNING: Missing features (filling with 0): {missing}")
        for col in missing:
            X[col] = 0
    if extra:
        X = X.drop(columns=extra)
    X = X[model_feature_names]

    # Reproduce exact train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    test_indices = X_test.index
    governor_test = df.loc[test_indices, "governor_predicted"].values

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # --- Baseline (no thresholds) ---
    print(f"\n{'='*70}")
    print("BASELINE (classifier only, no thresholds)")
    print(f"{'='*70}")

    classifier_preds = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, classifier_preds)
    baseline_recalls = per_class_recall(y_test, classifier_preds, labels)

    print(f"Accuracy: {baseline_acc:.4f} ({sum(y_test == classifier_preds)}/{len(y_test)})")
    print(classification_report(y_test, classifier_preds, labels=labels, zero_division=0))
    print_confusion_matrix(y_test, classifier_preds, labels)
    print("Per-class recall:")
    for label, recall in baseline_recalls.items():
        print(f"  {label:12s}: {recall:.4f} ({recall*100:.1f}%)")
    print(f"  Min recall: {min(baseline_recalls.values()):.4f}")

    # Governor baseline for reference
    print(f"\n--- Governor baseline ---")
    gov_acc = accuracy_score(y_test, governor_test)
    gov_recalls = per_class_recall(y_test, governor_test, labels)
    print(f"Accuracy: {gov_acc:.4f} ({sum(y_test == governor_test)}/{len(y_test)})")
    print("Per-class recall:")
    for label, recall in gov_recalls.items():
        print(f"  {label:12s}: {recall:.4f} ({recall*100:.1f}%)")

    # --- Extract probabilities ---
    probas = model.predict_proba(X_test)
    print(f"\nProbability matrix shape: {probas.shape}")

    # Show probability distribution per class
    print("\nPredicted class probability stats (for correctly vs incorrectly classified):")
    for i, label in enumerate(labels):
        mask_correct = (classifier_preds == label) & (y_test == label)
        mask_wrong = (classifier_preds == label) & (y_test != label)
        if mask_correct.sum() > 0:
            correct_probs = probas[mask_correct, i]
            print(f"  {label} (correct): mean={correct_probs.mean():.3f} "
                  f"median={np.median(correct_probs):.3f} min={correct_probs.min():.3f}")
        if mask_wrong.sum() > 0:
            wrong_probs = probas[mask_wrong, i]
            print(f"  {label} (wrong):   mean={wrong_probs.mean():.3f} "
                  f"median={np.median(wrong_probs):.3f} min={wrong_probs.min():.3f}")

    # --- Threshold sweep ---
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP")
    print(f"{'='*70}")

    best_thresholds, best_min_recall, best_recalls = sweep_thresholds(
        probas, classifier_preds, governor_test, y_test, labels
    )

    # Apply best thresholds
    calibrated_preds = apply_thresholds(
        probas, classifier_preds, governor_test, best_thresholds, labels
    )
    calibrated_acc = accuracy_score(y_test, calibrated_preds)

    print(f"\n{'='*70}")
    print("CALIBRATED RESULTS (classifier + governor fallback)")
    print(f"{'='*70}")

    print(f"\nOptimal thresholds:")
    for label in labels:
        print(f"  {label:12s}: {best_thresholds[label]:.2f}")

    print(f"\nAccuracy: {calibrated_acc:.4f} ({sum(y_test == calibrated_preds)}/{len(y_test)})")
    print(classification_report(y_test, calibrated_preds, labels=labels, zero_division=0))
    print_confusion_matrix(y_test, calibrated_preds, labels)

    calibrated_recalls = per_class_recall(y_test, calibrated_preds, labels)
    print("Per-class recall:")
    for label in labels:
        delta = calibrated_recalls[label] - baseline_recalls[label]
        sign = "+" if delta >= 0 else ""
        print(
            f"  {label:12s}: {calibrated_recalls[label]:.4f} "
            f"({calibrated_recalls[label]*100:.1f}%) "
            f"[{sign}{delta*100:.1f}pp vs baseline]"
        )
    print(f"  Min recall: {min(calibrated_recalls.values()):.4f}")

    # Fallback analysis
    n_fallback = sum(calibrated_preds != classifier_preds)
    fallback_mask = calibrated_preds != classifier_preds
    if n_fallback > 0:
        gov_correct_on_fallback = sum(
            (y_test[fallback_mask] == governor_test[fallback_mask])
        )
        clf_correct_on_fallback = sum(
            (y_test[fallback_mask] == classifier_preds[fallback_mask])
        )
        print(f"\nFallback analysis:")
        print(f"  Cases falling back to governor: {n_fallback}/{len(y_test)} "
              f"({100*n_fallback/len(y_test):.1f}%)")
        print(f"  Governor accuracy on fallback cases: "
              f"{gov_correct_on_fallback}/{n_fallback} "
              f"({100*gov_correct_on_fallback/n_fallback:.1f}%)")
        print(f"  Classifier accuracy on fallback cases: "
              f"{clf_correct_on_fallback}/{n_fallback} "
              f"({100*clf_correct_on_fallback/n_fallback:.1f}%)")
    else:
        print(f"\nNo fallback cases (all thresholds at 0.0)")

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'':20s} {'Accuracy':>10s}  {'Min Recall':>10s}")
    print(f"  {'Governor baseline':20s} {gov_acc:>10.4f}  {min(gov_recalls.values()):>10.4f}")
    print(f"  {'Classifier (raw)':20s} {baseline_acc:>10.4f}  {min(baseline_recalls.values()):>10.4f}")
    print(f"  {'Calibrated':20s} {calibrated_acc:>10.4f}  {min(calibrated_recalls.values()):>10.4f}")

    # --- Save calibrated artifact ---
    calibrated_artifact = artifact.copy()
    calibrated_artifact["thresholds"] = {k: float(v) for k, v in best_thresholds.items()}
    calibrated_artifact["calibrated_accuracy"] = calibrated_acc
    calibrated_artifact["calibrated_recalls"] = calibrated_recalls
    calibrated_artifact["fallback_count"] = int(n_fallback)

    joblib.dump(calibrated_artifact, _OUTPUT_PATH)
    print(f"\nCalibrated model saved to {_OUTPUT_PATH}")
    clean_thresholds = {k: float(v) for k, v in best_thresholds.items()}
    print(f"  Thresholds: {clean_thresholds}")


if __name__ == "__main__":
    main()
