#!/usr/bin/env python
"""
test_lags.py
=============
Evaluates lag strategies on the final risk and compares them
with the baseline (no lags). Includes lag0-lag2, lag0-lag3,
and lag0-lag4 windows for S2 (max) and S3 (weighted) strategies.

Strategies:
  Base           : current risk (no lags) - baseline
  S1-lag2        : mean of lag0, lag1, lag2 of risk
  S2-lag2        : max of lag0, lag1, lag2 of risk
  S3-lag2        : weighted mean: 0.6*lag0 + 0.3*lag1 + 0.1*lag2
  S1-lag3        : mean of lag0, lag1, lag2, lag3 of risk
  S2-lag3        : max of lag0, lag1, lag2, lag3 of risk
  S3-lag3        : weighted mean: 0.5*lag0 + 0.3*lag1 + 0.2*lag2 + 0.1*lag3
  S2-lag4        : max of lag0, lag1, lag2, lag3, lag4 of risk
  S3-lag4        : weighted mean: 0.4*lag0 + 0.3*lag1 + 0.2*lag2 + 0.1*lag3 + 0.05*lag4

Output in lags/:
  - lags_results.json   : metrics for all strategies
  - lags_comparison.txt : readable comparison report
  - lags_comparison.png : comparative bar chart
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ===========================================================================
# PATHS
# ===========================================================================
BASE_DIR = "C:/"
RISK_PATH = os.path.join(BASE_DIR, "risk_score", "risk_2000m.csv.gz")
CASES_PATH = os.path.join(BASE_DIR, "cases", "cases_dengue.csv")
POP_PATH = os.path.join(BASE_DIR, "radiation", "population_per_municipality.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "lags")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
print("=" * 70)
print("LOADING DATA...")
print("=" * 70)

# Risk (already filtered >2000m)
risk = pd.read_csv(RISK_PATH, compression="gzip")
risk["Year_month"] = risk["Year_month"].astype(str)
print(f"Risk: {len(risk):,} records, {risk['DANE'].nunique()} municipalities")

# Cases
cases = pd.read_csv(CASES_PATH)
cases["Year_month"] = cases["Year_month"].astype(str)
print(f"Cases: {len(cases):,} records, {cases['DANE'].nunique()} municipalities")

# Population
pop_wide = pd.read_csv(POP_PATH)
pop_long = pop_wide.melt(
    id_vars=["DANE"],
    value_vars=[f"pop_{y}" for y in range(2007, 2025)],
    var_name="year_str", value_name="population"
)
pop_long["year"] = pop_long["year_str"].str.replace("pop_", "").astype(int)

# Merge
df = risk.merge(cases[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
df["cases"] = df["cases"].fillna(0).astype(int)
df["year"] = df["Year_month"].apply(lambda x: int(x.split("-")[0]))

# Population is already included in the risk file (column 'population')
# Only used to calculate incidence
if "population" not in df.columns:
    df = df.merge(pop_long[["DANE", "year", "population"]], on=["DANE", "year"], how="left")
    med_pop = df.groupby("DANE")["population"].transform("median")
    df["population"] = df["population"].fillna(med_pop)

# Incidence
df["incidence_per_10k"] = (df["cases"] / df["population"]) * 10000
df["has_outbreak"] = (df["incidence_per_10k"] > 0).astype(int)

print(f"\nTotal records: {len(df):,}")
print(f"Outbreak prevalence: {df['has_outbreak'].mean()*100:.2f}%")

# ===========================================================================
# 2. CREATE RISK LAGS (LAG1 TO LAG4) BY MUNICIPALITY
# ===========================================================================
print("\n" + "=" * 70)
print("CREATING LAGS (lag0 through lag4)...")
print("=" * 70)

# Sort by DANE and Year_month
df = df.sort_values(["DANE", "Year_month"]).reset_index(drop=True)

# Create lags
df["risk_lag1"] = df.groupby("DANE")["risk"].shift(1)
df["risk_lag2"] = df.groupby("DANE")["risk"].shift(2)
df["risk_lag3"] = df.groupby("DANE")["risk"].shift(3)
df["risk_lag4"] = df.groupby("DANE")["risk"].shift(4)

# ===========================================================================
# 3. CREATE STRATEGIES - lag0-lag2 window
# ===========================================================================
print("\n--- Strategies with lag0-lag2 ---")
df["risk_s1_lag2"] = df[["risk", "risk_lag1", "risk_lag2"]].mean(axis=1)
df["risk_s2_lag2"] = df[["risk", "risk_lag1", "risk_lag2"]].max(axis=1)
df["risk_s3_lag2"] = (0.6 * df["risk"] + 0.3 * df["risk_lag1"] + 0.1 * df["risk_lag2"])

# ===========================================================================
# 4. CREATE STRATEGIES - lag0-lag3 window
# ===========================================================================
print("--- Strategies with lag0-lag3 ---")
df["risk_s1_lag3"] = df[["risk", "risk_lag1", "risk_lag2", "risk_lag3"]].mean(axis=1)
df["risk_s2_lag3"] = df[["risk", "risk_lag1", "risk_lag2", "risk_lag3"]].max(axis=1)
df["risk_s3_lag3"] = (0.5 * df["risk"] + 0.3 * df["risk_lag1"] + 0.2 * df["risk_lag2"] + 0.1 * df["risk_lag3"])

# ===========================================================================
# 4b. CREATE STRATEGIES - lag0-lag4 window (S2 and S3 only)
# ===========================================================================
print("--- Strategies with lag0-lag4 ---")
df["risk_s2_lag4"] = df[["risk", "risk_lag1", "risk_lag2", "risk_lag3", "risk_lag4"]].max(axis=1)
df["risk_s3_lag4"] = (0.4 * df["risk"] + 0.3 * df["risk_lag1"] + 0.2 * df["risk_lag2"] + 0.1 * df["risk_lag3"] + 0.05 * df["risk_lag4"])

# ===========================================================================
# 5. FILTER RECORDS WITH COMPLETE LAGS
# ===========================================================================
# Use the most restrictive set (lag4 available) for fair comparison
df_valid = df.dropna(subset=["risk_lag4"]).copy()
print(f"\nRecords with full lags (lag0-lag4 available): {len(df_valid):,}")

# ===========================================================================
# 6. METRIC FUNCTIONS
# ===========================================================================
def compute_metrics(y_true, y_score, label=""):
    """Calculate main metrics for binary scoring."""
    metrics = {}
    
    # Rank correlations
    rho, p_rho = spearmanr(y_score, y_true)
    metrics["spearman_rho"] = float(rho)
    metrics["spearman_p"] = float(p_rho)
    
    tau, p_tau = kendalltau(y_score, y_true)
    metrics["kendall_tau"] = float(tau)
    
    # Binary discrimination
    auc_roc = roc_auc_score(y_true, y_score)
    metrics["auc_roc"] = float(auc_roc)
    
    prevalence = y_true.mean()
    auc_pr = average_precision_score(y_true, y_score)
    metrics["auc_pr"] = float(auc_pr)
    metrics["auc_pr_lift"] = float(auc_pr / prevalence) if prevalence > 0 else 0
    metrics["prevalence"] = float(prevalence)
    
    # Recall@k
    for k_pct in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(k_pct * len(y_true)))
        top_k_idx = np.argsort(y_score)[-k:]
        recall_k = y_true[top_k_idx].sum() / y_true.sum()
        metrics[f"recall_top_{int(k_pct*100)}pct"] = float(recall_k)
        metrics[f"lift_top_{int(k_pct*100)}pct"] = float(recall_k / k_pct)
    
    # Brier score (risk min-max scaled to [0,1])
    risk_min, risk_max = y_score.min(), y_score.max()
    if risk_max > risk_min:
        risk_scaled = (y_score - risk_min) / (risk_max - risk_min)
    else:
        risk_scaled = y_score.copy()
    brier = brier_score_loss(y_true, risk_scaled)
    brier_baseline = prevalence * (1 - prevalence)
    metrics["brier_score"] = float(brier)
    metrics["brier_skill_score"] = float(1 - brier / brier_baseline) if brier_baseline > 0 else 0
    
    return metrics


def cv_metrics(y_true, y_score, groups, n_splits=5):
    """Cross-validated metrics with GroupKFold."""
    gkf = GroupKFold(n_splits=n_splits)
    cv_roc, cv_pr, cv_recall = [], [], []
    
    for train_idx, test_idx in gkf.split(y_true, y_true, groups):
        y_test = y_true[test_idx]
        s_test = y_score[test_idx]
        if y_test.sum() > 0 and (1 - y_test).sum() > 0:
            cv_roc.append(roc_auc_score(y_test, s_test))
            cv_pr.append(average_precision_score(y_test, s_test))
            k = int(0.10 * len(s_test))
            top_k = np.argsort(s_test)[-k:]
            rec = y_test[top_k].sum() / y_test.sum()
            cv_recall.append(rec)
    
    return {
        "cv_auc_roc_mean": float(np.mean(cv_roc)) if cv_roc else 0,
        "cv_auc_roc_std": float(np.std(cv_roc)) if cv_roc else 0,
        "cv_auc_pr_mean": float(np.mean(cv_pr)) if cv_pr else 0,
        "cv_auc_pr_std": float(np.std(cv_pr)) if cv_pr else 0,
        "cv_recall_10pct_mean": float(np.mean(cv_recall)) if cv_recall else 0,
        "cv_recall_10pct_std": float(np.std(cv_recall)) if cv_recall else 0,
    }

# ===========================================================================
# 7. EVALUATE EACH STRATEGY
# ===========================================================================
print("\n" + "=" * 70)
print("EVALUATING STRATEGIES...")
print("=" * 70)

strategies = {
    "Base (no lags)": "risk",
    "S1: Mean lag0-lag2": "risk_s1_lag2",
    "S2: Max lag0-lag2": "risk_s2_lag2",
    "S3: Weighted (0.6,0.3,0.1) lag2": "risk_s3_lag2",
    "S1: Mean lag0-lag3": "risk_s1_lag3",
    "S2: Max lag0-lag3": "risk_s2_lag3",
    "S3: Weighted (0.5,0.3,0.2,0.1) lag3": "risk_s3_lag3",
    "S2: Max lag0-lag4": "risk_s2_lag4",
    "S3: Weighted (0.4,0.3,0.2,0.1,0.05) lag4": "risk_s3_lag4",
}

y_true = df_valid["has_outbreak"].values
groups = df_valid["group_st"] = (df_valid["DANE"].astype(str) + "_" + df_valid["year"].astype(str)).values

all_results = {}
best_score = -1
best_strategy = None
best_metric = "auc_pr"

for label, col in strategies.items():
    print(f"\n  --- {label} ---")
    y_score = df_valid[col].values
    
    metrics = compute_metrics(y_true, y_score, label)
    cv = cv_metrics(y_true, y_score, groups, n_splits=5)
    metrics.update(cv)
    
    all_results[label] = metrics
    
    print(f"    AUC-ROC = {metrics['auc_roc']:.4f}")
    print(f"    AUC-PR  = {metrics['auc_pr']:.4f}  (lift = {metrics['auc_pr_lift']:.2f}x)")
    print(f"    Spearman rho = {metrics['spearman_rho']:.4f}")
    print(f"    Recall@10% = {metrics['recall_top_10pct']*100:.1f}%")
    print(f"    CV AUC-PR  = {metrics['cv_auc_pr_mean']:.4f} +/- {metrics['cv_auc_pr_std']:.4f}")
    
    score = metrics[best_metric]
    if score > best_score:
        best_score = score
        best_strategy = label

# ===========================================================================
# 8. RUN STRATEGY 4 (DLNM in R) - OPTIONAL
# ===========================================================================
print("\n" + "=" * 70)
print("STRATEGY 4: DLNM (via R script)...")
print("=" * 70)

r_script = os.path.join(OUTPUT_DIR, "test_lags_dlnm.R")
r_results_path = os.path.join(OUTPUT_DIR, "dlnm_results.json")

if os.path.exists(r_script):
    try:
        # Rscript on Windows
        r_cmd = f"Rscript \"{r_script}\""
        result = subprocess.run(r_cmd, shell=True, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.returncode != 0:
            print(f"R stderr: {result.stderr}")
            print("[WARN] R script failed - strategy 4 not available")
    except Exception as e:
        print(f"[WARN] Could not run R script: {e}")
    
    if os.path.exists(r_results_path):
        with open(r_results_path) as f:
            dlnm_results = json.load(f)
        all_results["S4: DLNM"] = dlnm_results
        s4_score = dlnm_results.get(best_metric, 0)
        if s4_score > best_score:
            best_score = s4_score
            best_strategy = "S4: DLNM"
        print(f"    DLNM results loaded: AUC-PR = {dlnm_results.get('auc_pr', 'N/A')}")
else:
    print("    R script not found. Create test_lags_dlnm.R for DLNM strategy.")

# ===========================================================================
# 9. COMPARATIVE REPORT
# ===========================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Metrics to show
key_metrics = [
    ("auc_roc", "AUC-ROC"),
    ("auc_pr", "AUC-PR"),
    ("auc_pr_lift", "AUC-PR Lift"),
    ("spearman_rho", "Spearman rho"),
    ("kendall_tau", "Kendall tau"),
    ("recall_top_10pct", "Recall@10%"),
    ("recall_top_5pct", "Recall@5%"),
    ("cv_auc_pr_mean", "CV AUC-PR"),
    ("brier_skill_score", "Brier Skill Score"),
]

# Comparison table
rows = []
for label, metrics in all_results.items():
    row = {"Strategy": label}
    for key, display in key_metrics:
        val = metrics.get(key, None)
        if val is not None:
            row[display] = val
    rows.append(row)

comp_df = pd.DataFrame(rows)
print("\n" + comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Determine the best strategy for each metric
print(f"\n  {'='*50}")
print(f"  BEST OVERALL (by {best_metric}): {best_strategy}")
print(f"  {'='*50}")

# ===========================================================================
# 10. SAVE RESULTS
# ===========================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS...")
print("=" * 70)

# JSON
with open(os.path.join(OUTPUT_DIR, "lags_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Text report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("COMPARISON OF LAG STRATEGIES FOR DENGUE RISK")
report_lines.append("Municipalities > 2000 m.a.s.l., Colombia 2007-2024")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Prevalence (outbreak > 0): {df_valid['has_outbreak'].mean()*100:.2f}%")
report_lines.append(f"Total records: {len(df_valid):,}")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("EXECUTIVE SUMMARY")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append(f"Best strategy by {best_metric}: {best_strategy}")
report_lines.append("")

for key, display in key_metrics:
    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"{display}:")
    report_lines.append(f"{'─'*60}")
    best_val = -1
    best_lbl = ""
    for label in all_results:
        val = all_results[label].get(key, 0)
        if isinstance(val, (int, float)) and val > best_val:
            best_val = val
            best_lbl = label
    for label in all_results:
        val = all_results[label].get(key, 0)
        if isinstance(val, (int, float)):
            marker = "  <- BEST" if label == best_lbl else ""
            report_lines.append(f"  {label:35s}: {val:.4f}{marker}")

report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("FULL METRICS")
report_lines.append("-" * 80)
report_lines.append("")

for label in all_results:
    report_lines.append(f"\n{label}:")
    report_lines.append("  " + "-" * 50)
    for key, val in all_results[label].items():
        if isinstance(val, float):
            report_lines.append(f"    {key:30s}: {val:.6f}")
        elif isinstance(val, str):
            report_lines.append(f"    {key:30s}: {val}")
        elif isinstance(val, list):
            report_lines.append(f"    {key:30s}: {val}")

report_path = os.path.join(OUTPUT_DIR, "lags_comparison.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  Report: {report_path}")

# Comparison chart
print("\n  Creating comparison plot...")
n_strats = len(all_results)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

plot_metrics = [
    ("auc_roc", "AUC-ROC", axes[0]),
    ("auc_pr", "AUC-PR", axes[1]),
    ("spearman_rho", "Spearman rho", axes[2]),
    ("recall_top_10pct", "Recall@10%", axes[3]),
]

labels = list(all_results.keys())
x = np.arange(len(labels))
bar_width = 0.6

colors_plot = ["#4575b4", "#d73027", "#fdae61", "#1a9850", "#6a3d9a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]

for metric_key, metric_label, ax in plot_metrics:
    values = [all_results[l].get(metric_key, 0) for l in labels]
    bars = ax.bar(x, values, bar_width, color=colors_plot[:len(labels)], 
                  edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.set_title(metric_label, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    # Value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "lags_comparison.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Plot: {plot_path}")

# Summary
print(f"\n{'='*70}")
print(f"BEST STRATEGY: {best_strategy}")
print(f"{'='*70}")
print(f"\nResults saved to {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fpath):
        print(f"  {f}")

print(f"\n{'='*70}")
print("COMPLETED")
print(f"{'='*70}")
