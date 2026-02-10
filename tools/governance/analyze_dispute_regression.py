# tools/governance/analyze_dispute_regression.py
"""
Investigate disputed recall regression: 83% (Exp 5, RF, 914 cases) → 67% (Exp 6, GBT, 1113 cases).

Hypotheses tested:
  H1: GBT model type is the problem (RF might still get 83%)
  H2: New harder disputed cases dilute recall
  H3: Class rebalancing shifted decision boundaries
  H4: Old disputed cases also regressed (most concerning)

Usage:
    .venv/Scripts/python tools/governance/analyze_dispute_regression.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# Import from train_classifier
from tools.governance.train_classifier import (
    enrich_with_context_features,
    prepare_features,
    print_confusion_matrix,
)

_BASE_DIR = Path(__file__).resolve().parent
_EVAL_V1 = _BASE_DIR / "data" / "eval_results.csv"
_EVAL_V2 = _BASE_DIR / "data" / "eval_results_v2.csv"
_DATA_DIR = _BASE_DIR.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def identify_new_case_ids(v1_path: Path, v2_path: Path) -> set[str]:
    """Identify case IDs present in v2 but not v1 (the 199 new cases)."""
    v1 = pd.read_csv(v1_path)
    v2 = pd.read_csv(v2_path)
    return set(v2["case_id"]) - set(v1["case_id"])


def disputed_recall(y_true, y_pred, label="disputed") -> tuple[int, int, float]:
    """Return (correct, total, recall%) for disputed class."""
    mask = y_true == label
    total = mask.sum()
    if total == 0:
        return 0, 0, 0.0
    correct = ((y_true == label) & (y_pred == label)).sum()
    return int(correct), int(total), 100.0 * correct / total


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_subsection(title: str):
    print(f"\n--- {title} ---")


def evaluate_and_report(name: str, model, X_test, y_test, labels):
    """Full evaluation report for a model."""
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    acc = accuracy_score(y_test, y_pred)
    corr, total, recall = disputed_recall(y_test, y_pred)

    print_subsection(name)
    print(f"Overall accuracy: {acc:.3f} ({sum(y_test == y_pred)}/{len(y_test)})")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    print_confusion_matrix(y_test, y_pred, labels)
    print(f"  ** Disputed recall: {corr}/{total} = {recall:.1f}% **")

    # Disputed → X confusion breakdown
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    d_idx = labels.index("disputed")
    print("  Disputed row in confusion matrix:")
    for j, lbl in enumerate(labels):
        if j != d_idx and cm[d_idx, j] > 0:
            print(f"    disputed -> {lbl}: {cm[d_idx, j]}")

    return y_pred, acc, recall


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print_section("LOADING DATA")
    df_v2 = pd.read_csv(_EVAL_V2)
    print(f"eval_results_v2.csv: {len(df_v2)} rows")
    print(f"Class distribution:\n{df_v2['expected_mode'].value_counts().to_string()}\n")

    new_ids = identify_new_case_ids(_EVAL_V1, _EVAL_V2)
    print(f"New case IDs (in v2 but not v1): {len(new_ids)}")
    new_by_mode = df_v2[df_v2["case_id"].isin(new_ids)]["expected_mode"].value_counts()
    print(f"New cases by class:\n{new_by_mode.to_string()}\n")

    # ------------------------------------------------------------------
    # 2. Enrich with context features
    # ------------------------------------------------------------------
    print_section("ENRICHING WITH CONTEXT FEATURES")
    df_v2 = enrich_with_context_features(df_v2, _DATA_DIR)

    # ------------------------------------------------------------------
    # 3. Prepare features
    # ------------------------------------------------------------------
    print_section("PREPARING FEATURES")
    X_all, encoders = prepare_features(df_v2)
    y_all = df_v2["expected_mode"]
    is_new = df_v2["case_id"].isin(new_ids).values
    labels = sorted(y_all.unique())
    print(f"Feature matrix: {X_all.shape[0]} x {X_all.shape[1]}")
    print(f"Labels: {labels}")

    # ------------------------------------------------------------------
    # 4. Stratified 80/20 split
    # ------------------------------------------------------------------
    print_section("TRAIN/TEST SPLIT (80/20, seed=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )
    test_idx = X_test.index
    is_new_test = is_new[test_idx]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Test set disputed: {(y_test == 'disputed').sum()}")
    print(f"Test set disputed (NEW): {((y_test == 'disputed') & is_new_test).sum()}")
    print(f"Test set disputed (OLD): {((y_test == 'disputed') & ~is_new_test).sum()}")

    # ------------------------------------------------------------------
    # 5. Train RF (class_weight=balanced) and GBT (sample_weight)
    # ------------------------------------------------------------------
    print_section("TRAINING MODELS ON FULL V2 DATA (1113 cases)")

    # RF
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # GBT with sample weights
    sample_weights_train = compute_sample_weight("balanced", y_train)
    gbt = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=SEED,
    )
    gbt.fit(X_train, y_train, sample_weight=sample_weights_train)

    # ------------------------------------------------------------------
    # 6. Full evaluation: RF vs GBT
    # ------------------------------------------------------------------
    print_section("MODEL COMPARISON ON V2 TEST SET")

    rf_pred, rf_acc, rf_disp_recall = evaluate_and_report(
        "Random Forest (balanced)", rf, X_test, y_test, labels
    )
    gbt_pred, gbt_acc, gbt_disp_recall = evaluate_and_report(
        "Gradient Boosted Trees (sample_weight)", gbt, X_test, y_test, labels
    )

    # ------------------------------------------------------------------
    # 7. Breakdown: disputed recall on OLD vs NEW cases
    # ------------------------------------------------------------------
    print_section("DISPUTED RECALL BREAKDOWN: OLD vs NEW CASES")

    for name, y_pred in [("RF", rf_pred), ("GBT", gbt_pred)]:
        print_subsection(f"{name}: Disputed recall by case origin")

        # Old disputed cases in test set
        old_disp_mask = (y_test == "disputed") & ~is_new_test
        new_disp_mask = (y_test == "disputed") & is_new_test

        if old_disp_mask.sum() > 0:
            c, t, r = disputed_recall(y_test[old_disp_mask], y_pred[old_disp_mask])
            print(f"  OLD disputed cases: {c}/{t} = {r:.1f}%")
            # What do misclassified old disputed become?
            old_wrong = y_pred[old_disp_mask & (y_pred != "disputed")]
            if len(old_wrong) > 0:
                print(f"    Misclassified as: {old_wrong.value_counts().to_dict()}")
        else:
            print("  OLD disputed cases: 0 in test set")

        if new_disp_mask.sum() > 0:
            c, t, r = disputed_recall(y_test[new_disp_mask], y_pred[new_disp_mask])
            print(f"  NEW disputed cases: {c}/{t} = {r:.1f}%")
            # What do misclassified new disputed become?
            new_wrong = y_pred[new_disp_mask & (y_pred != "disputed")]
            if len(new_wrong) > 0:
                print(f"    Misclassified as: {new_wrong.value_counts().to_dict()}")
        else:
            print("  NEW disputed cases: 0 in test set")

    # ------------------------------------------------------------------
    # 8. Subcategory breakdown for new disputed misclassifications
    # ------------------------------------------------------------------
    print_section("SUBCATEGORY ANALYSIS OF NEW DISPUTED CASES")

    new_disp_test_idx = test_idx[is_new_test & (y_test.values == "disputed")]
    if len(new_disp_test_idx) > 0:
        sub_df = df_v2.loc[new_disp_test_idx, ["case_id", "subcategory"]].copy()
        sub_df["rf_pred"] = rf_pred[new_disp_test_idx].values
        sub_df["gbt_pred"] = gbt_pred[new_disp_test_idx].values
        sub_df["rf_correct"] = sub_df["rf_pred"] == "disputed"
        sub_df["gbt_correct"] = sub_df["gbt_pred"] == "disputed"

        print("Per-subcategory accuracy on NEW disputed test cases:")
        for subcat in sorted(sub_df["subcategory"].unique()):
            s = sub_df[sub_df["subcategory"] == subcat]
            rf_ok = s["rf_correct"].sum()
            gbt_ok = s["gbt_correct"].sum()
            total = len(s)
            print(f"  {subcat:40s} RF: {rf_ok}/{total}  GBT: {gbt_ok}/{total}")

    # ------------------------------------------------------------------
    # 9. Train RF on ONLY old 914 cases, report disputed recall
    # ------------------------------------------------------------------
    print_section("RF TRAINED ON OLD 914 CASES ONLY")
    print("(Tests whether old cases' disputed recall degraded with new features)")

    old_mask = ~df_v2["case_id"].isin(new_ids)
    df_old = df_v2[old_mask].copy()
    print(f"Old dataset: {len(df_old)} rows")
    print(f"Old class distribution:\n{df_old['expected_mode'].value_counts().to_string()}\n")

    # Need to re-prepare features for old-only data (to get consistent encoding)
    # Re-enrich is already done on df_v2, just filter
    X_old, _ = prepare_features(df_old)
    y_old = df_old["expected_mode"]

    X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(
        X_old, y_old, test_size=0.2, random_state=SEED, stratify=y_old
    )
    print(f"Old train: {len(X_old_train)} | Old test: {len(X_old_test)}")
    print(f"Old test disputed: {(y_old_test == 'disputed').sum()}")

    rf_old = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    rf_old.fit(X_old_train, y_old_train)

    rf_old_pred, rf_old_acc, rf_old_disp_recall = evaluate_and_report(
        "RF on Old-Only (914 cases)", rf_old, X_old_test, y_old_test, labels
    )

    # Also train GBT on old for comparison
    sw_old = compute_sample_weight("balanced", y_old_train)
    gbt_old = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=SEED,
    )
    gbt_old.fit(X_old_train, y_old_train, sample_weight=sw_old)
    gbt_old_pred, gbt_old_acc, gbt_old_disp_recall = evaluate_and_report(
        "GBT on Old-Only (914 cases)", gbt_old, X_old_test, y_old_test, labels
    )

    # ------------------------------------------------------------------
    # 10. Cross-evaluation: models trained on old, tested on new
    # ------------------------------------------------------------------
    print_section("OLD-TRAINED MODELS TESTED ON NEW DISPUTED CASES ONLY")

    new_disp_all = df_v2[df_v2["case_id"].isin(new_ids) & (df_v2["expected_mode"] == "disputed")]
    if len(new_disp_all) > 0:
        # Get features for ALL new disputed cases
        new_disp_idx = new_disp_all.index
        X_new_disp = X_all.loc[new_disp_idx]
        y_new_disp = y_all.loc[new_disp_idx]

        for name, model in [("RF-old", rf_old), ("GBT-old", gbt_old)]:
            y_pred_new = pd.Series(model.predict(X_new_disp), index=new_disp_idx)
            c, t, r = disputed_recall(y_new_disp, y_pred_new)
            print(f"  {name} on ALL new disputed ({t} cases): {c}/{t} = {r:.1f}%")
            wrong = y_pred_new[y_pred_new != "disputed"]
            if len(wrong) > 0:
                print(f"    Misclassified as: {wrong.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 11. Feature importance comparison: what RF vs GBT prioritize
    # ------------------------------------------------------------------
    print_section("TOP 15 FEATURE IMPORTANCES (RF vs GBT on V2)")

    rf_imp = pd.Series(rf.feature_importances_, index=X_all.columns)
    gbt_imp = pd.Series(gbt.feature_importances_, index=X_all.columns)

    imp_df = pd.DataFrame({"RF": rf_imp, "GBT": gbt_imp})
    imp_df["diff"] = imp_df["RF"] - imp_df["GBT"]
    imp_df = imp_df.sort_values("RF", ascending=False)

    print("\nBy RF importance:")
    for i, (feat, row) in enumerate(imp_df.head(15).iterrows()):
        print(
            f"  {i+1:2d}. {feat:40s} RF={row['RF']:.4f}  GBT={row['GBT']:.4f}  diff={row['diff']:+.4f}"
        )

    print("\nBy GBT importance:")
    imp_df_gbt = imp_df.sort_values("GBT", ascending=False)
    for i, (feat, row) in enumerate(imp_df_gbt.head(15).iterrows()):
        print(
            f"  {i+1:2d}. {feat:40s} RF={row['RF']:.4f}  GBT={row['GBT']:.4f}  diff={row['diff']:+.4f}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("SUMMARY OF FINDINGS")

    h1_verdict = (
        "YES, model matters" if abs(rf_disp_recall - gbt_disp_recall) > 5 else "Similar performance"
    )
    h3_verdict = (
        "YES, data mix matters"
        if abs(rf_old_disp_recall - rf_disp_recall) > 5
        else "Similar, data mix not main issue"
    )
    if rf_old_disp_recall < 78:
        h4_verdict = "CONCERNING: old cases degraded too"
    elif rf_old_disp_recall >= 83:
        h4_verdict = "Old cases still perform well"
    else:
        h4_verdict = "Moderate regression"

    print(
        f"""
    +--------------------------------------------------------------------+
    | DISPUTED RECALL COMPARISON                                         |
    +--------------------------------------------------------------------+
    |                                                                    |
    |  Exp 5 reference:           ~83%% (RF, 914 cases)                  |
    |                                                                    |
    |  RF on V2 (1113):           {rf_disp_recall:5.1f}%% (all disputed)               |
    |  GBT on V2 (1113):          {gbt_disp_recall:5.1f}%% (all disputed)               |
    |                                                                    |
    |  RF on Old-Only (914):      {rf_old_disp_recall:5.1f}%% (reproduced baseline)      |
    |  GBT on Old-Only (914):     {gbt_old_disp_recall:5.1f}%% (model type effect)       |
    |                                                                    |
    +--------------------------------------------------------------------+
    | HYPOTHESIS RESULTS                                                 |
    +--------------------------------------------------------------------+
    |                                                                    |
    |  H1 (GBT model type):       RF={rf_disp_recall:.1f}%% vs GBT={gbt_disp_recall:.1f}%%           |
    |      -> {h1_verdict}
    |                                                                    |
    |  H2 (new cases harder):     See OLD vs NEW breakdown above         |
    |                                                                    |
    |  H3 (class rebalancing):    RF old-only={rf_old_disp_recall:.1f}%% vs RF V2={rf_disp_recall:.1f}%%   |
    |      -> {h3_verdict}
    |                                                                    |
    |  H4 (old cases regressed):  RF old-only={rf_old_disp_recall:.1f}%% vs Exp5=~83%%    |
    |      -> {h4_verdict}
    |                                                                    |
    +--------------------------------------------------------------------+
    """
    )


if __name__ == "__main__":
    main()
