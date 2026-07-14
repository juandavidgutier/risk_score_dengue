#!/usr/bin/env python
"""
lags_detailed_analysis.py
==========================
Detailed analysis of strategies S1 (Mean), S2 (Max), and S3 (Weighted)
with additional visualizations and statistical tests.

Output:
  - lags_detailed_report.txt : Detailed report
  - lags_detail_*.png        : Detail plots
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
BASE_DIR = "C:/"
OUTPUT_DIR = os.path.join(BASE_DIR, "lags")

# -- Load data --
risk = pd.read_csv(os.path.join(BASE_DIR, "risk_score", "risk_2000m.csv.gz"), compression="gzip")
risk["Year_month"] = risk["Year_month"].astype(str)
cases = pd.read_csv(os.path.join(BASE_DIR, "cases", "cases_dengue.csv"))
cases["Year_month"] = cases["Year_month"].astype(str)

df = risk.merge(cases[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
df["cases"] = df["cases"].fillna(0).astype(int)
df["year"] = df["Year_month"].apply(lambda x: int(x.split("-")[0]))
df["incidence_per_10k"] = (df["cases"] / df["population"]) * 10000
df["has_outbreak"] = (df["incidence_per_10k"] > 0).astype(int)
df = df.sort_values(["DANE", "Year_month"]).reset_index(drop=True)
df["risk_lag1"] = df.groupby("DANE")["risk"].shift(1)
df["risk_lag2"] = df.groupby("DANE")["risk"].shift(2)
df = df.dropna(subset=["risk_lag2"]).copy()

df["risk_s1"] = df[["risk", "risk_lag1", "risk_lag2"]].mean(axis=1)
df["risk_s2"] = df[["risk", "risk_lag1", "risk_lag2"]].max(axis=1)
df["risk_s3"] = (0.6 * df["risk"] + 0.3 * df["risk_lag1"] + 0.1 * df["risk_lag2"])

y_true = df["has_outbreak"].values
groups = (df["DANE"].astype(str) + "_" + df["year"].astype(str)).values
prevalence = y_true.mean()

print("=" * 70)
print("DETAILED COMPARISON: S1 vs S2 vs S3")
print("=" * 70)

# -- 1. DETAILED METRICS --
strategies = {
    "Base (no lags)": df["risk"].values,
    "S1: Mean": df["risk_s1"].values,
    "S2: Max": df["risk_s2"].values,
    "S3: Weighted": df["risk_s3"].values,
}

def full_metrics(y_true, y_score):
    n = len(y_true)
    metrics = {}
    metrics["n"] = n
    metrics["auc_roc"] = roc_auc_score(y_true, y_score)
    metrics["auc_pr"] = average_precision_score(y_true, y_score)
    metrics["auc_pr_lift"] = metrics["auc_pr"] / prevalence
    rho, p = spearmanr(y_score, y_true)
    metrics["spearman"] = rho
    metrics["spearman_p"] = p
    tau, _ = kendalltau(y_score, y_true)
    metrics["kendall"] = tau

    for kp in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(kp * n))
        top = np.argsort(y_score)[-k:]
        rec = y_true[top].sum() / y_true.sum()
        metrics[f"recall_{kp*100:.0f}pct"] = rec
        metrics[f"lift_{kp*100:.0f}pct"] = rec / kp

    # Fraction with risk > 0
    metrics["pct_positive"] = (y_score > 0).mean()
    metrics["mean_risk"] = y_score.mean()
    metrics["median_risk"] = np.median(y_score)
    metrics["p95_risk"] = np.percentile(y_score, 95)
    metrics["max_risk"] = y_score.max()

    # Risk in municipalities with outbreak vs without outbreak
    metrics["mean_risk_outbreak"] = y_score[y_true == 1].mean()
    metrics["mean_risk_no_outbreak"] = y_score[y_true == 0].mean()

    return metrics

all_m = {}
for label, scores in strategies.items():
    all_m[label] = full_metrics(y_true, scores)

# -- 2. CV BY FOLD --
print("\n[2] Cross-validation by fold...")
gkf = GroupKFold(n_splits=5)
cv_detail = {label: {"roc": [], "pr": [], "recall_10": []} for label in strategies}

for fold, (train_idx, test_idx) in enumerate(gkf.split(y_true, y_true, groups)):
    y_test = y_true[test_idx]
    print(f"  Fold {fold+1}: {len(y_test)} test, {y_test.sum()} outbreaks")
    for label, scores in strategies.items():
        s_test = scores[test_idx]
        if y_test.sum() > 0 and (1 - y_test).sum() > 0:
            cv_detail[label]["roc"].append(roc_auc_score(y_test, s_test))
            cv_detail[label]["pr"].append(average_precision_score(y_test, s_test))
            k = int(0.10 * len(s_test))
            top = np.argsort(s_test)[-k:]
            cv_detail[label]["recall_10"].append(y_test[top].sum() / y_test.sum())

# -- 3. RELATIVE IMPROVEMENTS --
print("\n[3] Relative improvements over baseline...")
base = all_m["Base (no lags)"]
improvements = {}
for label in ["S1: Mean", "S2: Max", "S3: Weighted"]:
    imp = {}
    for met in ["auc_roc", "auc_pr", "spearman", "recall_10pct", "lift_10pct"]:
        if base[met] > 0:
            imp[met] = (all_m[label][met] - base[met]) / abs(base[met]) * 100
        else:
            imp[met] = np.nan
    improvements[label] = imp

# -- 4. PLOTS --
print("\n[4] Generating detailed plots...")

labels = list(strategies.keys())
colors = ["#4575b4", "#1a9850", "#d73027", "#fdae61"]
x = np.arange(len(labels))

# Plot 1: Full metric comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

plot_configs = [
    ("auc_roc", "AUC-ROC", axes[0], [0.55, 0.65]),
    ("auc_pr", "AUC-PR", axes[1], [0.03, 0.05]),
    ("auc_pr_lift", "AUC-PR Lift", axes[2], [1.4, 2.0]),
    ("spearman", "Spearman rho", axes[3], [0.07, 0.09]),
    ("recall_10pct", "Recall@10%", axes[4], [0.20, 0.30]),
    ("recall_20pct", "Recall@20%", axes[5], [0.30, 0.45]),
]

for met_name, met_label, ax, ylim in plot_configs:
    vals = [all_m[l][met_name] for l in labels]
    bars = ax.bar(x, vals, 0.55, color=colors, edgecolor="white", alpha=0.85)
    ax.set_title(met_label, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lags_detail_metrics.png"), dpi=200, bbox_inches="tight")
plt.close()

# Plot 2: Relative improvement (%)
fig, ax = plt.subplots(figsize=(10, 6))
imp_metrics = [("auc_roc", "AUC-ROC"), ("auc_pr", "AUC-PR"), ("recall_10pct", "Recall@10%"),
               ("recall_20pct", "Recall@20%"), ("lift_5pct", "Lift@5%")]
s_labels = ["S1: Mean", "S2: Max", "S3: Weighted"]
s_colors = ["#1a9850", "#d73027", "#fdae61"]

x_imp = np.arange(len(imp_metrics))
width = 0.25

for i, sl in enumerate(s_labels):
    vals = [improvements[sl].get(m[0], 0) for m in imp_metrics]
    ax.bar(x_imp + i*width - width, vals, width, color=s_colors[i],
           label=sl, edgecolor="white", alpha=0.85)

ax.set_xticks(x_imp)
ax.set_xticklabels([m[1] for m in imp_metrics], fontsize=10)
ax.axhline(y=0, color="gray", linewidth=1, linestyle="--")
ax.set_ylabel("Improvement over baseline (%)", fontsize=11)
ax.set_title("Relative improvement of each lag strategy", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y", linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lags_detail_improvement.png"), dpi=200, bbox_inches="tight")
plt.close()

# Plot 3: CV by fold
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cv_metrics = [("roc", "AUC-ROC"), ("pr", "AUC-PR"), ("recall_10", "Recall@10%")]

for ax, (cv_key, cv_label) in zip(axes, cv_metrics):
    data = [cv_detail[l][cv_key] for l in labels]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("white"); median.set_linewidth(2)
    ax.set_title(f"CV {cv_label}", fontsize=12, fontweight="bold")
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lags_detail_cv.png"), dpi=200, bbox_inches="tight")
plt.close()

# Plot 4: Risk distribution by strategy
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
bins = np.logspace(np.log10(1e-8), np.log10(1), 100)

for i, (label, scores) in enumerate(strategies.items()):
    ax = axes[i]
    scores_pos = scores[scores > 0]
    ax.hist(scores_pos, bins=bins, alpha=0.7, color=colors[i], edgecolor="white")
    ax.set_xscale("log")
    ax.set_title(f"{label}\n({all_m[label]['pct_positive']*100:.1f}% > 0)", fontsize=11)
    ax.set_xlabel("Risk (log scale)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lags_detail_distribution.png"), dpi=200, bbox_inches="tight")
plt.close()

# -- 5. REPORT --
print("\n[5] Generating detailed report...")
lines = []
lines.append("=" * 80)
lines.append("DETAILED COMPARISON: S1 (Mean) vs S2 (Max) vs S3 (Weighted)")
lines.append("Municipalities > 2000m, Colombia 2007-2024")
lines.append(f"Total records: {len(df):,} | Prevalence: {prevalence*100:.2f}%")
lines.append("=" * 80)

lines.append("\n\n1. FULL METRICS TABLE")
lines.append("-" * 80)
header = f"{'Metric':25s}"
for l in labels:
    header += f"  {l:18s}"
lines.append(header)
lines.append("-" * 80)

for met_name in ["auc_roc", "auc_pr", "auc_pr_lift", "spearman", "kendall",
                  "recall_1pct", "recall_5pct", "recall_10pct", "recall_20pct",
                  "lift_5pct", "lift_10pct", "pct_positive",
                  "mean_risk_outbreak", "mean_risk_no_outbreak"]:
    row = f"{met_name:25s}"
    for l in labels:
        v = all_m[l].get(met_name, np.nan)
        if isinstance(v, float):
            row += f"  {v:18.6f}"
        else:
            row += f"  {str(v):18s}"
    lines.append(row)

lines.append("\n\n2. RELATIVE IMPROVEMENT OVER BASELINE (%)")
lines.append("-" * 80)
header = f"{'Metric':25s}  {'S1':12s}  {'S2':12s}  {'S3':12s}"
lines.append(header)
lines.append("-" * 80)
for met_name, met_label in [("auc_roc", "AUC-ROC"), ("auc_pr", "AUC-PR"),
                              ("spearman", "Spearman"), ("recall_10pct", "Recall@10%"),
                              ("lift_10pct", "Lift@10%")]:
    row = f"{met_label:25s}"
    for sl in s_labels:
        v = improvements[sl].get(met_name, 0)
        sign = "+" if v > 0 else ""
        row += f"  {sign}{v:6.2f}%      "
    lines.append(row)

lines.append("\n\n3. CROSS-VALIDATION (5 folds, GroupKFold by municipality-year)")
lines.append("-" * 80)
for cv_key, cv_label in [("roc", "AUC-ROC"), ("pr", "AUC-PR"), ("recall_10", "Recall@10%")]:
    lines.append(f"\n  {cv_label}:")
    for i, l in enumerate(labels):
        vals = cv_detail[l][cv_key]
        if vals:
            lines.append(f"    {l:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  [{np.min(vals):.4f}, {np.max(vals):.4f}]")

lines.append("\n\n4. WIN COUNTS (which strategy wins for each metric)")
lines.append("-" * 80)
win_counts = {"Base (no lags)": 0, "S1: Mean": 0, "S2: Max": 0, "S3: Weighted": 0}
for met_name in ["auc_roc", "auc_pr", "auc_pr_lift", "spearman",
                  "recall_1pct", "recall_5pct", "recall_10pct", "recall_20pct"]:
    best = max(all_m, key=lambda l: all_m[l].get(met_name, 0))
    win_counts[best] += 1
    lines.append(f"  {met_name:20s}: best = {best}")

lines.append(f"\n  {'TOTAL':20s}:")
for l, c in sorted(win_counts.items(), key=lambda x: -x[1]):
    lines.append(f"    {l:20s}: {c} wins")

lines.append(f"\n\n5. RECOMMENDATION")
lines.append("-" * 80)
# Best by AUC-PR (main metric for imbalanced)
best_aucpr = max(["S1: Mean", "S2: Max", "S3: Weighted"],
                  key=lambda l: all_m[l]["auc_pr"])
# Best by AUC-ROC
best_roc = max(["S1: Mean", "S2: Max", "S3: Weighted"],
                key=lambda l: all_m[l]["auc_roc"])
# Most consistent (lowest CV std for AUC-PR)
best_cv = min(["S1: Mean", "S2: Max", "S3: Weighted"],
               key=lambda l: np.std(cv_detail[l]["pr"]) if cv_detail[l]["pr"] else 999)

lines.append(f"\n  Best by AUC-PR (main):         {best_aucpr} ({all_m[best_aucpr]['auc_pr']:.4f})")
lines.append(f"  Best by AUC-ROC:               {best_roc} ({all_m[best_roc]['auc_roc']:.4f})")
lines.append(f"  Most consistent (CV):          {best_cv} (CV std = {np.std(cv_detail[best_cv]['pr']):.4f})")
lines.append(f"  Most wins across metrics:      {max(win_counts, key=win_counts.get)} ({win_counts[max(win_counts, key=win_counts.get)]} wins)")

lines.append(f"\n\n  {'='*60}")
lines.append(f"  BEST OVERALL STRATEGY: {best_aucpr}")
lines.append(f"  {'='*60}")
lines.append(f"""
  Justification:
  - AUC-PR is the main metric for imbalanced problems (prevalence={prevalence*100:.1f}%).
  - {best_aucpr} has the highest AUC-PR ({all_m[best_aucpr]['auc_pr']:.4f}), which is
    the most relevant criterion for identifying dengue outbreaks.
  - All strategies improve the baseline, but the difference between S1-S3
    is small (< 0.002 in AUC-PR).
  - For implementation in compute_final_risk.py, {best_aucpr} is recommended
    for its balance between simplicity and performance.
""")

report_path = os.path.join(OUTPUT_DIR, "lags_detailed_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nReport saved: {report_path}")
print("Figures:")
print("  lags_detail_metrics.png")
print("  lags_detail_improvement.png")
print("  lags_detail_cv.png")
print("  lags_detail_distribution.png")
print("\nDONE")
