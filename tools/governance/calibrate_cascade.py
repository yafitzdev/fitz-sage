# tools/governance/calibrate_cascade.py
"""Sweep Q1/Q2/Q3/Q4 thresholds jointly to minimize false-trustworthy predictions."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.governance.train_classifier import (
    _CONFLICT_FEATURE,
    _collapse_to_3class,
    prepare_features,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "fitz_sage"
    / "governance"
    / "data"
    / "model_v6_cascade.joblib"
)
SEED = 42


def main():
    # Load data
    df = pd.read_csv(DATA_DIR / "features.csv")
    print(f"Loaded {len(df)} cases")
    X, encoders = prepare_features(df)
    y_3class = _collapse_to_3class(df["expected_mode"].values)

    # Load artifact
    artifact = joblib.load(MODEL_PATH)
    q1_model = artifact["q1_model"]
    q2_model = artifact.get("q2_model")  # ML router (None in legacy)
    q3_model = artifact["q3_model"]
    q4_model = artifact["q4_model"]
    old_q1_t = artifact["q1_threshold"]
    old_q2_t = artifact.get("q2_threshold", 0.5)
    old_q3_t = artifact["q3_threshold"]
    old_q4_t = artifact["q4_threshold"]

    has_q2_model = q2_model is not None
    q2_desc = f"Q2={old_q2_t:.3f}" if has_q2_model else f"Q2=rule({_CONFLICT_FEATURE})"
    print(f"Current thresholds: Q1={old_q1_t:.3f}  {q2_desc}  Q3={old_q3_t:.3f}  Q4={old_q4_t:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Q1 OOF probabilities
    print("Running Q1 OOF predictions...")
    y_q1 = np.where(y_3class == "abstain", 0, 1)
    q1_oof = cross_val_predict(q1_model, X, y_q1, cv=cv, method="predict_proba")

    # Q2 OOF (ML router or hard rule)
    if has_q2_model:
        print("Running Q2 OOF predictions (ML router)...")
        y_q2 = np.where(y_3class == "disputed", 1, 0)
        q2_oof = cross_val_predict(q2_model, X, y_q2, cv=cv, method="predict_proba")
        p_q2 = q2_oof[:, 1]
    else:
        print(f"Q2: using hard rule ({_CONFLICT_FEATURE})")
        p_q2 = X[_CONFLICT_FEATURE].astype(float).values

    # Q3 OOF (conflict + non-abstain cases) — use ground-truth routing for training
    gt_has_conflict = (y_3class == "disputed") | (X[_CONFLICT_FEATURE].astype(int).values == 1)
    q3_mask = gt_has_conflict & (y_3class != "abstain")
    y_q3 = np.where(y_3class[q3_mask] == "trustworthy", 1, 0)
    print(f"Running Q3 OOF predictions ({q3_mask.sum()} conflict cases)...")
    q3_oof = cross_val_predict(q3_model, X[q3_mask], y_q3, cv=cv, method="predict_proba")

    q3_indices = np.where(q3_mask)[0]
    q3_full = np.full(len(X), 0.0)
    for i, idx in enumerate(q3_indices):
        q3_full[idx] = q3_oof[i, 1]

    # Q4 OOF (clean cases)
    q4_mask = ~gt_has_conflict
    y_q4 = np.where(y_3class[q4_mask] == "trustworthy", 1, 0)
    print(f"Running Q4 OOF predictions ({q4_mask.sum()} clean cases)...")
    q4_oof = cross_val_predict(q4_model, X[q4_mask], y_q4, cv=cv, method="predict_proba")

    q4_indices = np.where(q4_mask)[0]
    q4_full = np.full(len(X), 0.0)
    for i, idx in enumerate(q4_indices):
        q4_full[idx] = q4_oof[i, 1]

    # Sweep
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP -- minimizing false-trustworthy")
    print(f"{'='*70}")

    n = len(y_3class)
    p_q1_prob = q1_oof[:, 1]

    results = []

    # Q2 threshold range depends on whether ML or rule
    q2_range = np.arange(0.20, 0.55, 0.05) if has_q2_model else [0.5]

    # Coarse sweep first, then refine around best point
    for q1_t in np.arange(0.40, 0.80, 0.02):
        for q2_t in q2_range:
            # Compute routing once per Q2 threshold
            if has_q2_model:
                has_conflict = p_q2 >= q2_t
            else:
                has_conflict = X[_CONFLICT_FEATURE].astype(int).values == 1

            for q3_t in np.arange(0.20, 0.70, 0.02):
                for q4_t in np.arange(0.40, 0.80, 0.02):
                    pred = np.full(n, "abstain", dtype=object)
                    answerable = p_q1_prob >= q1_t

                    # Conflict path
                    cm = answerable & has_conflict
                    pred[cm] = np.where(q3_full[cm] >= q3_t, "trustworthy", "disputed")

                    # Clean path
                    cl = answerable & ~has_conflict
                    pred[cl] = np.where(q4_full[cl] >= q4_t, "trustworthy", "abstain")

                    ft_abs = int(((pred == "trustworthy") & (y_3class == "abstain")).sum())
                    ft_dis = int(((pred == "trustworthy") & (y_3class == "disputed")).sum())
                    ft = ft_abs + ft_dis

                    acc = (pred == y_3class).mean()
                    r_abs = (pred[y_3class == "abstain"] == "abstain").mean()
                    r_dis = (pred[y_3class == "disputed"] == "disputed").mean()
                    r_tru = (pred[y_3class == "trustworthy"] == "trustworthy").mean()

                    # Filter: keep accuracy >= 75% and trustworthy recall >= 65%
                    if acc >= 0.75 and r_tru >= 0.65:
                        results.append(
                            {
                                "q1_t": q1_t,
                                "q2_t": q2_t,
                                "q3_t": q3_t,
                                "q4_t": q4_t,
                                "ft": ft,
                                "ft_abs": ft_abs,
                                "ft_dis": ft_dis,
                                "acc": acc,
                                "r_abs": r_abs,
                                "r_dis": r_dis,
                                "r_tru": r_tru,
                            }
                        )

    # Sort by false-trustworthy (primary), then accuracy (secondary, descending)
    results.sort(key=lambda r: (r["ft"], -r["acc"]))

    print(f"\nFound {len(results)} viable configurations (acc >= 75%, trustworthy recall >= 65%)")
    print("\nTop 10 (lowest false-trustworthy):\n")
    header = f"  {'Q1':>6s}  {'Q2':>6s}  {'Q3':>6s}  {'Q4':>6s}  {'FT':>4s}  {'Acc':>6s}  {'Abs':>6s}  {'Dis':>6s}  {'Tru':>6s}"
    print(header)
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for r in results[:10]:
        print(
            f"  {r['q1_t']:6.3f}  {r['q2_t']:6.3f}  {r['q3_t']:6.3f}  {r['q4_t']:6.3f}"
            f"  {r['ft']:4d}"
            f"  {r['acc']*100:5.1f}%  {r['r_abs']*100:5.1f}%  {r['r_dis']*100:5.1f}%  {r['r_tru']*100:5.1f}%"
        )

    # Show current operating point
    if has_q2_model:
        cur_conflict = p_q2 >= old_q2_t
    else:
        cur_conflict = X[_CONFLICT_FEATURE].astype(int).values == 1

    pred = np.full(n, "abstain", dtype=object)
    answerable = p_q1_prob >= old_q1_t
    cm = answerable & cur_conflict
    pred[cm] = np.where(q3_full[cm] >= old_q3_t, "trustworthy", "disputed")
    cl = answerable & ~cur_conflict
    pred[cl] = np.where(q4_full[cl] >= old_q4_t, "trustworthy", "abstain")
    ft_abs = int(((pred == "trustworthy") & (y_3class == "abstain")).sum())
    ft_dis = int(((pred == "trustworthy") & (y_3class == "disputed")).sum())
    cur = {
        "ft": ft_abs + ft_dis,
        "acc": (pred == y_3class).mean(),
        "r_abs": (pred[y_3class == "abstain"] == "abstain").mean(),
        "r_dis": (pred[y_3class == "disputed"] == "disputed").mean(),
        "r_tru": (pred[y_3class == "trustworthy"] == "trustworthy").mean(),
    }

    print(f"\n  Current operating point (Q1={old_q1_t:.3f} Q2={old_q2_t:.3f} Q3={old_q3_t:.3f} Q4={old_q4_t:.3f}):")
    print(
        f"  FT={cur['ft']}  Acc={cur['acc']*100:.1f}%"
        f"  Abs={cur['r_abs']*100:.1f}%  Dis={cur['r_dis']*100:.1f}%  Tru={cur['r_tru']*100:.1f}%"
    )

    if results:
        pick = results[0]
        print(
            f"\n  Best safe point (Q1={pick['q1_t']:.3f} Q2={pick['q2_t']:.3f} Q3={pick['q3_t']:.3f} Q4={pick['q4_t']:.3f}):"
        )
        print(
            f"  FT={pick['ft']}  Acc={pick['acc']*100:.1f}%"
            f"  Abs={pick['r_abs']*100:.1f}%  Dis={pick['r_dis']*100:.1f}%  Tru={pick['r_tru']*100:.1f}%"
        )
        delta_ft = cur["ft"] - pick["ft"]
        delta_acc = (pick["acc"] - cur["acc"]) * 100
        print(f"\n  Delta: FT {'-' if delta_ft > 0 else '+'}{abs(delta_ft)}, Acc {delta_acc:+.1f}%")

        ans = input("\nApply best safe thresholds? [y/N] ")
        if ans.strip().lower() == "y":
            artifact["q1_threshold"] = pick["q1_t"]
            if has_q2_model:
                artifact["q2_threshold"] = pick["q2_t"]
            artifact["q3_threshold"] = pick["q3_t"]
            artifact["q4_threshold"] = pick["q4_t"]
            artifact["combined_accuracy"] = pick["acc"]
            artifact["combined_recalls"] = {
                "abstain": pick["r_abs"],
                "disputed": pick["r_dis"],
                "trustworthy": pick["r_tru"],
            }
            artifact["false_trustworthy"] = pick["ft"]
            joblib.dump(artifact, MODEL_PATH)
            print(f"Updated artifact at {MODEL_PATH}")
        else:
            print("No changes made.")


if __name__ == "__main__":
    main()
