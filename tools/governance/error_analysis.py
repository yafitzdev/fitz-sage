# tools/governance/error_analysis.py
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from tools.governance.train_classifier import enrich_with_context_features, prepare_features  # noqa: E402

# Load data
df = pd.read_csv("tools/governance/data/eval_results_v5_clean.csv")
model_data = joblib.load("tools/governance/data/model_v6_twostage.joblib")
s1_model = model_data["stage1_model"]
s2_model = model_data["stage2_model"]
feature_cols = model_data["feature_names"]

fitzgov_dir = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
)
if fitzgov_dir.exists():
    df = enrich_with_context_features(df, fitzgov_dir)
else:
    print(f"WARNING: {fitzgov_dir} not found, skipping context features")

# Prepare features using same function as training (handles encoders properly)
X_all, _ = prepare_features(df)
X = X_all[feature_cols].copy()

# Two-stage prediction with probabilities
s1_proba = s1_model.predict_proba(X)
s1_classes = list(s1_model.classes_)
answerable_idx = s1_classes.index("answerable")
abstain_idx = s1_classes.index("abstain")
p_answerable = s1_proba[:, answerable_idx]


# 3-class labels
def to_3class(mode):
    if mode in ("confident", "qualified"):
        return "trustworthy"
    return mode


df["label_3class"] = df["expected_mode"].map(to_3class)

# Stage 1 predictions
s1_threshold = model_data.get("stage1_threshold", 0.5)
s2_threshold = model_data.get("stage2_threshold", 0.5)
print(f"Model thresholds: s1={s1_threshold}, s2={s2_threshold}")

s1_pred = np.where(p_answerable >= s1_threshold, "answerable", "abstain")
df["s1_pred"] = s1_pred
df["p_answerable"] = p_answerable

# Stage 2 (only for answerable predictions)
answerable_mask = s1_pred == "answerable"
s2_proba = s2_model.predict_proba(X[answerable_mask])
s2_classes = list(s2_model.classes_)
trustworthy_idx = s2_classes.index("trustworthy")
p_trustworthy = s2_proba[:, trustworthy_idx]

df.loc[answerable_mask, "p_trustworthy"] = p_trustworthy

# Final 3-class prediction
df["predicted_3class"] = "abstain"
df.loc[answerable_mask, "predicted_3class"] = np.where(
    p_trustworthy >= s2_threshold, "trustworthy", "disputed"
)

# Error analysis
df["correct"] = df["label_3class"] == df["predicted_3class"]
df["error_type"] = df.apply(
    lambda r: f"{r['label_3class']}->{r['predicted_3class']}" if not r["correct"] else "correct",
    axis=1,
)

print("\n=== ERROR TYPE DISTRIBUTION ===")
errors = df[~df["correct"]]
print(f"Total errors: {len(errors)}/{len(df)} ({len(errors)/len(df)*100:.1f}%)")
print()
for et, count in errors["error_type"].value_counts().items():
    pct = count / len(errors) * 100
    print(f"  {et}: {count} ({pct:.1f}%)")

print("\n=== STAGE 1 ERRORS (abstain boundary) ===")
# Abstain cases misclassified as answerable
s1_fn = df[(df["label_3class"] == "abstain") & (df["s1_pred"] == "answerable")]
print(f"Abstain missed (FN): {len(s1_fn)}")
if len(s1_fn) > 0:
    print(
        f"  P(answerable) stats: mean={s1_fn['p_answerable'].mean():.3f}, median={s1_fn['p_answerable'].median():.3f}"
    )
    print(f"  Subcategories: {dict(s1_fn['subcategory'].value_counts().head(10))}")
    print(f"  Difficulty: {dict(s1_fn['difficulty'].value_counts())}")

# Non-abstain cases misclassified as abstain
s1_fp = df[(df["label_3class"] != "abstain") & (df["s1_pred"] == "abstain")]
print(f"\nAbstain false alarms (FP): {len(s1_fp)}")
if len(s1_fp) > 0:
    print(
        f"  P(answerable) stats: mean={s1_fp['p_answerable'].mean():.3f}, median={s1_fp['p_answerable'].median():.3f}"
    )
    print(f"  Actual labels: {dict(s1_fp['label_3class'].value_counts())}")
    print(f"  Subcategories: {dict(s1_fp['subcategory'].value_counts().head(10))}")

print("\n=== STAGE 2 ERRORS (disputed boundary) ===")
s2_cases = df[answerable_mask].copy()
s2_errors = s2_cases[s2_cases["label_3class"] != s2_cases["predicted_3class"]]
# Filter out cases that were abstain (already handled by stage 1)
s2_errors = s2_errors[s2_errors["label_3class"] != "abstain"]
print(f"Stage 2 errors: {len(s2_errors)}")

# Disputed cases classified as trustworthy
disputed_missed = s2_cases[
    (s2_cases["label_3class"] == "disputed") & (s2_cases["predicted_3class"] == "trustworthy")
]
print(f"\nDisputed->trustworthy (missed contradictions): {len(disputed_missed)}")
if len(disputed_missed) > 0:
    print(
        f"  P(trustworthy) stats: mean={disputed_missed['p_trustworthy'].mean():.3f}, median={disputed_missed['p_trustworthy'].median():.3f}"
    )
    print(f"  Subcategories: {dict(disputed_missed['subcategory'].value_counts().head(10))}")
    print(f"  ca_fired: {dict(disputed_missed['ca_fired'].value_counts())}")
    print(f"  ca_signal: {dict(disputed_missed['ca_signal'].value_counts())}")

# Trustworthy cases classified as disputed
trust_as_disputed = s2_cases[
    (s2_cases["label_3class"] == "trustworthy") & (s2_cases["predicted_3class"] == "disputed")
]
print(f"\nTrustworthy->disputed (false contradictions): {len(trust_as_disputed)}")
if len(trust_as_disputed) > 0:
    print(
        f"  P(trustworthy) stats: mean={trust_as_disputed['p_trustworthy'].mean():.3f}, median={trust_as_disputed['p_trustworthy'].median():.3f}"
    )
    print(f"  Subcategories: {dict(trust_as_disputed['subcategory'].value_counts().head(10))}")
    print(f"  ca_fired: {dict(trust_as_disputed['ca_fired'].value_counts())}")
    print(f"  Original 4-class: {dict(trust_as_disputed['expected_mode'].value_counts())}")

print("\n=== FEATURE DISCRIMINATION ANALYSIS ===")
# For each feature, compare distributions between error groups
key_features = [
    "ca_fired",
    "ca_signal",
    "ie_fired",
    "ie_signal",
    "mean_vector_score",
    "score_spread",
    "vocab_overlap_ratio",
    "num_constraints_fired",
    "av_jury_votes_no",
    "ca_pairs_checked",
    "query_word_count",
    "num_chunks",
    "ctx_length_mean",
    "ctx_max_pairwise_sim",
    "ctx_mean_pairwise_sim",
    "ctx_contradiction_count",
    "ctx_number_variance",
]

print("\nFeature means by 3-class label:")
for f in key_features:
    if f in df.columns and df[f].dtype in ["int64", "float64", "bool"]:
        means = df.groupby("label_3class")[f].mean()
        print(
            f"  {f:35s}  abstain={means.get('abstain',0):.3f}  "
            f"disputed={means.get('disputed',0):.3f}  "
            f"trustworthy={means.get('trustworthy',0):.3f}"
        )

print("\n=== CONFIDENCE DISTRIBUTION BY CORRECTNESS ===")
for label in ["abstain", "disputed", "trustworthy"]:
    subset = df[df["label_3class"] == label]
    correct = subset[subset["correct"]]
    wrong = subset[~subset["correct"]]
    if label == "abstain":
        print(
            f"\n{label}: correct P(answerable)={correct['p_answerable'].mean():.3f}, "
            f"wrong P(answerable)={wrong['p_answerable'].mean():.3f}"
        )
    else:
        c_pt = correct["p_trustworthy"].dropna()
        w_pt = wrong["p_trustworthy"].dropna()
        c_pa = correct["p_answerable"].mean()
        w_pa = wrong["p_answerable"].mean() if len(wrong) > 0 else float("nan")
        print(
            f"\n{label}: correct P(trustworthy)={c_pt.mean():.3f} (n={len(c_pt)}), "
            f"wrong P(trustworthy)={w_pt.mean():.3f} (n={len(w_pt)})"
        )
        print(f"  correct P(answerable)={c_pa:.3f}, wrong P(answerable)={w_pa:.3f}")

print("\n=== HARDEST SUBCATEGORIES (error rate) ===")
for label in ["abstain", "disputed", "trustworthy"]:
    subset = df[df["label_3class"] == label]
    sub_acc = subset.groupby("subcategory")["correct"].agg(["mean", "count"])
    sub_acc = sub_acc[sub_acc["count"] >= 3].sort_values("mean")
    print(f"\n{label} - worst subcategories:")
    for idx, row in sub_acc.head(8).iterrows():
        print(f"  {idx:40s}  accuracy={row['mean']:.1%}  (n={int(row['count'])})")

print("\n=== CA_FIRED vs ACTUAL CLASS ===")
print(pd.crosstab(df["label_3class"], df["ca_fired"], margins=True))

print("\n=== CA_SIGNAL vs ACTUAL CLASS ===")
print(pd.crosstab(df["label_3class"], df["ca_signal"], margins=True))

print("\n=== IE_FIRED vs ACTUAL CLASS ===")
print(pd.crosstab(df["label_3class"], df["ie_fired"], margins=True))

print("\n=== FEATURE IMPORTANCE FROM BOTH STAGES ===")
print("\nStage 1 (answerable vs abstain):")
for name, imp in sorted(zip(feature_cols, s1_model.feature_importances_), key=lambda x: -x[1])[:15]:
    print(f"  {name:35s}  {imp:.4f}")

print("\nStage 2 (trustworthy vs disputed):")
for name, imp in sorted(zip(feature_cols, s2_model.feature_importances_), key=lambda x: -x[1])[:15]:
    print(f"  {name:35s}  {imp:.4f}")
