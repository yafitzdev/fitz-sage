# tools/governance/calibrate_thresholds.py
"""
Per-stage calibrated thresholds for two-stage governance classifier.

Loads model_v5_twostage.joblib and eval_results_v2.csv, reproduces the exact
train/test split, extracts predict_proba() on test cases for each stage, and
sweeps per-stage thresholds.

Two-stage threshold logic:
  Stage 1 (answerable vs abstain):
    - If P(answerable) < threshold_s1 -> force abstain (safe fallback)
  Stage 2 (trustworthy vs disputed):
    - If P(trustworthy) < threshold_s2 -> force disputed (cautious fallback)

Also supports legacy 4-class mode with governor fallback.

Usage:
    python -m tools.governance.calibrate_thresholds
    python -m tools.governance.calibrate_thresholds --mode 4class
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tools.governance.train_classifier import (
    _3CLASS_LABELS,
    _BOOL_FEATURES,
    _CATEGORICAL_FEATURES,
    _META_COLS,
    _collapse_to_3class,
    compute_context_features,
    load_cases_by_id,
)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_FITZGOV_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
)

# Two-stage defaults
_TWOSTAGE_MODEL_PATH = _DATA_DIR / "model_v5_twostage.joblib"
_TWOSTAGE_OUTPUT_PATH = _DATA_DIR / "model_v5_calibrated.joblib"

# Legacy 4-class defaults
_4CLASS_MODEL_PATH = _DATA_DIR / "model_v3.joblib"
_4CLASS_OUTPUT_PATH = _DATA_DIR / "model_v3_calibrated.joblib"

_EVAL_CSV = _DATA_DIR / "eval_results_v2.csv"

SEED = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Feature preparation (mirrors train_classifier.prepare_features)
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """Prepare feature matrix using pre-fitted encoders from the saved artifact."""
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].copy()

    for col in _CATEGORICAL_FEATURES:
        if col not in X.columns:
            continue
        X[col] = X[col].fillna("none").astype(str)
        if col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            X[col] = X[col].apply(lambda v, k=known, e=le: e.transform([v])[0] if v in k else -1)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    for col in _BOOL_FEATURES:
        if col not in X.columns:
            continue
        X[col] = (
            X[col].map({"True": 1, "False": 0, True: 1, False: 0, 1: 1, 0: 0}).fillna(0).astype(int)
        )

    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def enrich_with_context_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add context-based features by loading original fitz-gov cases.

    Skips if ctx_* features are already present (from feature_extractor.py).
    """
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
    print(f"  Added {len(ctx_df.columns)} context features")
    return pd.concat([df, ctx_df], axis=1)


def align_features(X: pd.DataFrame, model_feature_names: list[str]) -> pd.DataFrame:
    """Ensure feature columns match model exactly."""
    missing = [c for c in model_feature_names if c not in X.columns]
    extra = [c for c in X.columns if c not in model_feature_names]
    if missing:
        print(f"WARNING: Missing features (filling with 0): {missing}")
        for col in missing:
            X[col] = 0
    if extra:
        X = X.drop(columns=extra)
    return X[model_feature_names]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_confusion_matrix(y_true, y_pred, labels):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = "predicted ->".rjust(20) + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"actual {label}".rjust(20) + "".join(f"{cm[i, j]:>15}" for j in range(len(labels)))
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
# Two-stage calibration
# ---------------------------------------------------------------------------


def twostage_predict(
    s1_model,
    s2_model,
    X: pd.DataFrame,
    s1_threshold: float = 0.5,
    s2_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run two-stage prediction with confidence thresholds.

    Returns:
        final_preds: 3-class predictions (abstain/disputed/trustworthy)
        s1_probas: Stage 1 probabilities (N x 2)
        s2_probas: Stage 2 probabilities for answerable cases (M x 2), padded to N
    """
    n = len(X)
    final_preds = np.full(n, "abstain", dtype=object)

    # Stage 1: answerable vs abstain
    s1_probas = s1_model.predict_proba(X)
    s1_classes = list(s1_model.classes_)
    answerable_idx = s1_classes.index("answerable")

    # Cases where P(answerable) >= threshold -> pass to Stage 2
    s1_answerable_mask = s1_probas[:, answerable_idx] >= s1_threshold

    # Full Stage 2 probabilities (filled with zeros for non-answerable)
    s2_probas_full = np.zeros((n, 2))

    if s1_answerable_mask.any():
        X_answerable = X[s1_answerable_mask]
        s2_probas = s2_model.predict_proba(X_answerable)
        s2_classes = list(s2_model.classes_)
        trustworthy_idx = s2_classes.index("trustworthy")

        s2_probas_full[s1_answerable_mask] = s2_probas

        # Cases where P(trustworthy) >= threshold -> trustworthy, else disputed
        answerable_positions = np.where(s1_answerable_mask)[0]
        for i, pos in enumerate(answerable_positions):
            if s2_probas[i, trustworthy_idx] >= s2_threshold:
                final_preds[pos] = "trustworthy"
            else:
                final_preds[pos] = "disputed"

    return final_preds, s1_probas, s2_probas_full


def sweep_twostage_thresholds(
    s1_model,
    s2_model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[float, float, float, dict[str, float]]:
    """Sweep Stage 1 and Stage 2 thresholds to maximize minimum per-class recall."""

    grid_s1 = np.arange(0.30, 0.85, 0.05)
    grid_s2 = np.arange(0.30, 0.85, 0.05)

    combos = list(itertools.product(grid_s1, grid_s2))
    print(f"Sweeping {len(combos)} threshold combinations...")

    best_min_recall = -1.0
    best_s1_t = 0.5
    best_s2_t = 0.5
    best_recalls = {}
    best_acc = 0.0

    for s1_t, s2_t in combos:
        preds, _, _ = twostage_predict(s1_model, s2_model, X_test, s1_t, s2_t)
        recalls = per_class_recall(y_test, preds, _3CLASS_LABELS)
        min_recall = min(recalls.values())
        acc = accuracy_score(y_test, preds)

        if min_recall > best_min_recall or (min_recall == best_min_recall and acc > best_acc):
            best_min_recall = min_recall
            best_s1_t = s1_t
            best_s2_t = s2_t
            best_recalls = recalls.copy()
            best_acc = acc

    return best_s1_t, best_s2_t, best_min_recall, best_recalls


def calibrate_twostage(model_path: Path, eval_csv: Path, output_path: Path):
    """Calibrate a two-stage model with per-stage confidence thresholds."""

    # Load model artifact
    print(f"Loading two-stage model from {model_path}...")
    artifact = joblib.load(model_path)
    s1_model = artifact["stage1_model"]
    s2_model = artifact["stage2_model"]
    encoders = artifact["encoders"]
    feature_names = artifact["feature_names"]
    print(f"  Stage 1: {artifact['stage1_name']}")
    print(f"  Stage 2: {artifact['stage2_name']}")
    print(f"  Uncalibrated accuracy: {artifact['combined_accuracy']:.4f}")

    # Load eval data
    print(f"\nLoading eval data from {eval_csv}...")
    df = pd.read_csv(eval_csv)
    print(f"  {len(df)} rows x {len(df.columns)} columns")

    # Add context features
    if _FITZGOV_DIR.exists():
        df = enrich_with_context_features(df, _FITZGOV_DIR)
    else:
        print(f"WARNING: {_FITZGOV_DIR} not found, skipping context features")

    # Prepare features
    X = prepare_features(df, encoders)
    X = align_features(X, feature_names)
    y_4class = df["expected_mode"].values
    y_3class = _collapse_to_3class(y_4class)

    # Reproduce exact train/test split (stratify on 3-class)
    X_train, X_test, y3_train, y3_test = train_test_split(
        X,
        y_3class,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_3class,
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Test distribution: {dict(zip(*np.unique(y3_test, return_counts=True)))}")

    # --- Baseline (default 0.5 thresholds) ---
    print(f"\n{'='*70}")
    print("BASELINE (two-stage, default thresholds s1=0.5, s2=0.5)")
    print(f"{'='*70}")

    baseline_preds, s1_probas, s2_probas = twostage_predict(
        s1_model,
        s2_model,
        X_test,
        0.5,
        0.5,
    )
    baseline_acc = accuracy_score(y3_test, baseline_preds)
    baseline_recalls = per_class_recall(y3_test, baseline_preds, _3CLASS_LABELS)

    print(f"Accuracy: {baseline_acc:.4f} ({sum(y3_test == baseline_preds)}/{len(y3_test)})")
    print(classification_report(y3_test, baseline_preds, labels=_3CLASS_LABELS, zero_division=0))
    print_confusion_matrix(y3_test, baseline_preds, _3CLASS_LABELS)
    print("Per-class recall:")
    for label, recall in baseline_recalls.items():
        print(f"  {label:15s}: {recall:.4f} ({recall*100:.1f}%)")
    print(f"  Min recall: {min(baseline_recalls.values()):.4f}")

    # Governor baseline
    governor_4class = df.loc[X_test.index, "governor_predicted"].values
    governor_3class = _collapse_to_3class(governor_4class)
    gov_acc = accuracy_score(y3_test, governor_3class)
    gov_recalls = per_class_recall(y3_test, governor_3class, _3CLASS_LABELS)
    print("\n--- Governor baseline (3-class) ---")
    print(f"Accuracy: {gov_acc:.4f}")
    print("Per-class recall:")
    for label, recall in gov_recalls.items():
        print(f"  {label:15s}: {recall:.4f} ({recall*100:.1f}%)")

    # --- Probability distribution analysis ---
    print(f"\n{'='*70}")
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")

    s1_classes = list(s1_model.classes_)
    answerable_idx = s1_classes.index("answerable")
    abstain_idx = s1_classes.index("abstain")

    # Stage 1 proba stats
    y_s1_true = np.array(
        ["abstain" if l == "abstain" else "answerable" for l in y3_test],
        dtype=object,
    )
    print("\nStage 1 P(answerable) stats:")
    for true_label in ["answerable", "abstain"]:
        mask = y_s1_true == true_label
        if mask.sum() > 0:
            probs = s1_probas[mask, answerable_idx]
            print(
                f"  {true_label:12s}: mean={probs.mean():.3f} "
                f"median={np.median(probs):.3f} min={probs.min():.3f} max={probs.max():.3f}"
            )

    # Stage 2 proba stats (for answerable cases only)
    s2_classes = list(s2_model.classes_)
    trustworthy_idx = s2_classes.index("trustworthy")

    answerable_mask = y_s1_true == "answerable"
    if answerable_mask.any():
        s2_preds_probas = s2_model.predict_proba(X_test[answerable_mask])
        y_answerable_true = y3_test[answerable_mask]
        print("\nStage 2 P(trustworthy) stats (answerable subset):")
        for true_label in ["trustworthy", "disputed"]:
            mask = y_answerable_true == true_label
            if mask.sum() > 0:
                probs = s2_preds_probas[mask, trustworthy_idx]
                print(
                    f"  {true_label:12s}: mean={probs.mean():.3f} "
                    f"median={np.median(probs):.3f} min={probs.min():.3f} max={probs.max():.3f}"
                )

    # --- Threshold sweep ---
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP")
    print(f"{'='*70}")

    best_s1_t, best_s2_t, best_min_recall, best_recalls = sweep_twostage_thresholds(
        s1_model,
        s2_model,
        X_test,
        y3_test,
    )

    calibrated_preds, _, _ = twostage_predict(
        s1_model,
        s2_model,
        X_test,
        best_s1_t,
        best_s2_t,
    )
    calibrated_acc = accuracy_score(y3_test, calibrated_preds)

    print(f"\n{'='*70}")
    print("CALIBRATED RESULTS")
    print(f"{'='*70}")

    print("\nOptimal thresholds:")
    print(f"  Stage 1 (answerable): {best_s1_t:.2f}")
    print(f"  Stage 2 (trustworthy): {best_s2_t:.2f}")

    print(f"\nAccuracy: {calibrated_acc:.4f} ({sum(y3_test == calibrated_preds)}/{len(y3_test)})")
    print(classification_report(y3_test, calibrated_preds, labels=_3CLASS_LABELS, zero_division=0))
    print_confusion_matrix(y3_test, calibrated_preds, _3CLASS_LABELS)

    calibrated_recalls = per_class_recall(y3_test, calibrated_preds, _3CLASS_LABELS)
    print("Per-class recall:")
    for label in _3CLASS_LABELS:
        delta = calibrated_recalls[label] - baseline_recalls[label]
        sign = "+" if delta >= 0 else ""
        print(
            f"  {label:15s}: {calibrated_recalls[label]:.4f} "
            f"({calibrated_recalls[label]*100:.1f}%) "
            f"[{sign}{delta*100:.1f}pp vs baseline]"
        )
    print(f"  Min recall: {min(calibrated_recalls.values()):.4f}")

    # Routing analysis
    s1_answerable_mask = s1_model.predict_proba(X_test)[:, answerable_idx] >= best_s1_t
    n_to_s2 = s1_answerable_mask.sum()
    n_abstain = (~s1_answerable_mask).sum()
    print("\nRouting analysis:")
    print(f"  Stage 1 -> abstain: {n_abstain}/{len(y3_test)} ({100*n_abstain/len(y3_test):.1f}%)")
    print(f"  Stage 1 -> Stage 2: {n_to_s2}/{len(y3_test)} ({100*n_to_s2/len(y3_test):.1f}%)")

    # --- Summary comparison ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'':20s} {'Accuracy':>10s}  {'Min Recall':>10s}")
    print(f"  {'Governor (3-class)':20s} {gov_acc:>10.4f}  {min(gov_recalls.values()):>10.4f}")
    print(f"  {'Raw (0.5/0.5)':20s} {baseline_acc:>10.4f}  {min(baseline_recalls.values()):>10.4f}")
    print(
        f"  {'Calibrated':20s} {calibrated_acc:>10.4f}  {min(calibrated_recalls.values()):>10.4f}"
    )

    # --- Save calibrated artifact ---
    calibrated_artifact = artifact.copy()
    calibrated_artifact["stage1_threshold"] = float(best_s1_t)
    calibrated_artifact["stage2_threshold"] = float(best_s2_t)
    calibrated_artifact["calibrated_accuracy"] = calibrated_acc
    calibrated_artifact["calibrated_recalls"] = calibrated_recalls
    calibrated_artifact["calibrated_min_recall"] = float(min(calibrated_recalls.values()))

    joblib.dump(calibrated_artifact, output_path)
    print(f"\nCalibrated two-stage model saved to {output_path}")
    print(f"  Stage 1 threshold: {best_s1_t:.2f}")
    print(f"  Stage 2 threshold: {best_s2_t:.2f}")


# ---------------------------------------------------------------------------
# Legacy 4-class calibration
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


def sweep_thresholds_4class(
    probas: np.ndarray,
    classifier_preds: np.ndarray,
    governor_preds: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Brute-force sweep per-class thresholds to maximize minimum per-class recall."""
    grid_values = np.arange(0.0, 0.95, 0.05)
    best_min_recall = -1.0
    best_thresholds = {label: 0.0 for label in labels}
    best_recalls = {}

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
            current_acc = accuracy_score(y_true, preds)
            best_preds = apply_thresholds(
                probas, classifier_preds, governor_preds, best_thresholds, labels
            )
            best_acc = accuracy_score(y_true, best_preds)
            if current_acc > best_acc:
                best_thresholds = thresholds.copy()
                best_recalls = recalls.copy()

    return best_thresholds, best_min_recall, best_recalls


def calibrate_4class(model_path: Path, eval_csv: Path, output_path: Path):
    """Calibrate a 4-class model with per-class governor fallback thresholds."""

    print(f"Loading model from {model_path}...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    encoders = artifact["encoders"]
    labels = artifact["labels"]
    model_feature_names = artifact["feature_names"]
    print(f"  Model: {artifact['model_name']}")
    print(f"  Labels: {labels}")

    print(f"\nLoading eval data from {eval_csv}...")
    df = pd.read_csv(eval_csv)
    print(f"  {len(df)} rows x {len(df.columns)} columns")

    if _FITZGOV_DIR.exists():
        df = enrich_with_context_features(df, _FITZGOV_DIR)

    X = prepare_features(df, encoders)
    y = df["expected_mode"].values
    X = align_features(X, model_feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    governor_test = df.loc[X_test.index, "governor_predicted"].values

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    classifier_preds = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, classifier_preds)
    baseline_recalls = per_class_recall(y_test, classifier_preds, labels)

    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    probas = model.predict_proba(X_test)

    best_thresholds, best_min_recall, best_recalls = sweep_thresholds_4class(
        probas, classifier_preds, governor_test, y_test, labels
    )

    calibrated_preds = apply_thresholds(
        probas, classifier_preds, governor_test, best_thresholds, labels
    )
    calibrated_acc = accuracy_score(y_test, calibrated_preds)

    print(f"\nCalibrated accuracy: {calibrated_acc:.4f}")
    print(f"Optimal thresholds: {best_thresholds}")
    print(f"Min recall: {best_min_recall:.4f}")

    calibrated_artifact = artifact.copy()
    calibrated_artifact["thresholds"] = {k: float(v) for k, v in best_thresholds.items()}
    calibrated_artifact["calibrated_accuracy"] = calibrated_acc
    calibrated_artifact["calibrated_recalls"] = best_recalls

    joblib.dump(calibrated_artifact, output_path)
    print(f"\nCalibrated model saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Calibrate governance classifier thresholds")
    parser.add_argument(
        "--mode",
        choices=["twostage", "4class"],
        default="twostage",
        help="Model type to calibrate (default: twostage)",
    )
    parser.add_argument("--model", type=Path, default=None, help="Model artifact path")
    parser.add_argument("--input", type=Path, default=_EVAL_CSV, help="Eval results CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output calibrated model path")
    args = parser.parse_args()

    if args.mode == "twostage":
        model_path = args.model or _TWOSTAGE_MODEL_PATH
        output_path = args.output or _TWOSTAGE_OUTPUT_PATH
        calibrate_twostage(model_path, args.input, output_path)
    else:
        model_path = args.model or _4CLASS_MODEL_PATH
        output_path = args.output or _4CLASS_OUTPUT_PATH
        calibrate_4class(model_path, args.input, output_path)


if __name__ == "__main__":
    main()
