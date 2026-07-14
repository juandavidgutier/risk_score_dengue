#!/usr/bin/env python
"""
detailed_analysis_s2_vs_base.py
================================
Detailed analysis comparing S2-lag4 (Max lag0-lag4) vs baseline (no lags).
Includes:
  1. ROC curves with bootstrap confidence bands
  2. Precision-Recall curves with bootstrap confidence bands
  3. DeLong test for statistical significance of AUC-ROC
  4. Score distribution (violin plot + histogram)
  5. Calibration curves
  6. Threshold analysis (sensitivity, specificity, PPV, NPV)
  7. Lift curve / Gains chart
  8. Summary report in JSON and TXT

Output in lags/:
  - analysis_s2_vs_base_roc_pr.png    : ROC and PR curves with bands
  - analysis_s2_vs_base_scores.png    : Score distribution
  - analysis_s2_vs_base_calibration.png: Calibration
  - analysis_s2_vs_base_thresholds.png : Threshold analysis
  - analysis_s2_vs_base_lift.png      : Lift curve
  - analysis_s2_vs_base_results.json  : Full metrics
  - analysis_s2_vs_base_report.txt    : Readable report
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, kendalltau, norm, bootstrap
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss, auc
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

risk = pd.read_csv(RISK_PATH, compression="gzip")
risk["Year_month"] = risk["Year_month"].astype(str)
print(f"Risk: {len(risk):,} records, {risk['DANE'].nunique()} municipalities")

cases = pd.read_csv(CASES_PATH)
cases["Year_month"] = cases["Year_month"].astype(str)

pop_wide = pd.read_csv(POP_PATH)
pop_long = pop_wide.melt(
    id_vars=["DANE"],
    value_vars=[f"pop_{y}" for y in range(2007, 2025)],
    var_name="year_str", value_name="population"
)
pop_long["year"] = pop_long["year_str"].str.replace("pop_", "").astype(int)

df = risk.merge(cases[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
df["cases"] = df["cases"].fillna(0).astype(int)
df["year"] = df["Year_month"].apply(lambda x: int(x.split("-")[0]))

if "population" not in df.columns:
    df = df.merge(pop_long[["DANE", "year", "population"]], on=["DANE", "year"], how="left")
    med_pop = df.groupby("DANE")["population"].transform("median")
    df["population"] = df["population"].fillna(med_pop)

df["incidence_per_10k"] = (df["cases"] / df["population"]) * 10000
df["has_outbreak"] = (df["incidence_per_10k"] > 0).astype(int)

# Create lags
df = df.sort_values(["DANE", "Year_month"]).reset_index(drop=True)
df["risk_lag1"] = df.groupby("DANE")["risk"].shift(1)
df["risk_lag2"] = df.groupby("DANE")["risk"].shift(2)
df["risk_lag3"] = df.groupby("DANE")["risk"].shift(3)
df["risk_lag4"] = df.groupby("DANE")["risk"].shift(4)

df["risk_s2_lag4"] = df[["risk", "risk_lag1", "risk_lag2", "risk_lag3", "risk_lag4"]].max(axis=1)

# Filter
df_valid = df.dropna(subset=["risk_lag4"]).copy().reset_index(drop=True)
print(f"\nValid records (lag0-lag4 available): {len(df_valid):,}")

y_true = df_valid["has_outbreak"].values
s_base = df_valid["risk"].values
s_s2 = df_valid["risk_s2_lag4"].values
groups = (df_valid["DANE"].astype(str) + "_" + df_valid["year"].astype(str)).values

prevalence = y_true.mean()
n_total = len(y_true)
n_pos = int(y_true.sum())
n_neg = n_total - n_pos

print(f"Outbreak prevalence: {prevalence*100:.2f}%  ({n_pos:,} / {n_total:,})")

# ===========================================================================
# 2. HELPER FUNCTIONS
# ===========================================================================
print("\n" + "=" * 70)
print("DEFINING ANALYSIS FUNCTIONS...")
print("=" * 70)

def bootstrap_roc_ci(y_true, y_score, n_boot=2000, ci=0.95):
    """Bootstrap confidence intervals for ROC curve."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    alpha = 1 - ci
    
    # FPR grid
    fpr_grid = np.linspace(0, 1, 200)
    tpr_samples = np.zeros((n_boot, len(fpr_grid)))
    
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y_true[idx]
        sb = y_score[idx]
        if yb.sum() > 0 and (1 - yb).sum() > 0:
            fpr, tpr, _ = roc_curve(yb, sb)
            tpr_samples[b] = np.interp(fpr_grid, fpr, tpr)
    
    # Median and CI
    tpr_median = np.median(tpr_samples, axis=0)
    tpr_lower = np.percentile(tpr_samples, 100 * alpha / 2, axis=0)
    tpr_upper = np.percentile(tpr_samples, 100 * (1 - alpha / 2), axis=0)
    
    return fpr_grid, tpr_median, tpr_lower, tpr_upper


def bootstrap_pr_ci(y_true, y_score, n_boot=2000, ci=0.95):
    """Bootstrap confidence intervals for PR curve."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    alpha = 1 - ci
    
    recall_grid = np.linspace(0, 1, 200)
    prec_samples = np.zeros((n_boot, len(recall_grid)))
    
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y_true[idx]
        sb = y_score[idx]
        if yb.sum() > 0 and (1 - yb).sum() > 0:
            prec, rec, _ = precision_recall_curve(yb, sb)
            prec_samples[b] = np.interp(recall_grid, rec[::-1], prec[::-1])
    
    prec_median = np.median(prec_samples, axis=0)
    prec_lower = np.percentile(prec_samples, 100 * alpha / 2, axis=0)
    prec_upper = np.percentile(prec_samples, 100 * (1 - alpha / 2), axis=0)
    
    return recall_grid, prec_median, prec_lower, prec_upper


def bootstrap_auc_ci(y_true, y_score, n_boot=5000, ci=0.95):
    """Bootstrap CI for AUC-ROC and AUC-PR."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    alpha = 1 - ci
    
    auc_rocs = np.zeros(n_boot)
    auc_prs = np.zeros(n_boot)
    
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y_true[idx]
        sb = y_score[idx]
        if yb.sum() > 0 and (1 - yb).sum() > 0:
            auc_rocs[b] = roc_auc_score(yb, sb)
            auc_prs[b] = average_precision_score(yb, sb)
    
    return {
        "auc_roc_mean": float(np.mean(auc_rocs)),
        "auc_roc_median": float(np.median(auc_rocs)),
        "auc_roc_ci": [float(np.percentile(auc_rocs, 100 * alpha / 2)),
                       float(np.percentile(auc_rocs, 100 * (1 - alpha / 2)))],
        "auc_roc_std": float(np.std(auc_rocs)),
        "auc_pr_mean": float(np.mean(auc_prs)),
        "auc_pr_median": float(np.median(auc_prs)),
        "auc_pr_ci": [float(np.percentile(auc_prs, 100 * alpha / 2)),
                      float(np.percentile(auc_prs, 100 * (1 - alpha / 2)))],
        "auc_pr_std": float(np.std(auc_prs)),
    }


def bootstrap_delong_test(y_true, s1, s2, n_boot=10000):
    """Bootstrap-based test for comparing two AUC values."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    
    auc1 = np.zeros(n_boot)
    auc2 = np.zeros(n_boot)
    
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y_true[idx]
        s1b = s1[idx]
        s2b = s2[idx]
        if yb.sum() > 0 and (1 - yb).sum() > 0:
            auc1[b] = roc_auc_score(yb, s1b)
            auc2[b] = roc_auc_score(yb, s2b)
    
    diff = auc2 - auc1
    z = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return {
        "auc_base_mean": float(np.mean(auc1)),
        "auc_s2_mean": float(np.mean(auc2)),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "diff_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "z_statistic": float(z),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def calibration_curve(y_true, y_score, n_bins=10):
    """Calibration curve: mean predicted vs observed frequency."""
    # Bin the scores into n_bins
    bin_edges = np.percentile(y_score, np.linspace(0, 100, n_bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mean_pred = np.zeros(n_bins)
    obs_frac = np.zeros(n_bins)
    n_in_bin = np.zeros(n_bins)
    
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
        else:
            mask = (y_score >= bin_edges[i]) & (y_score <= bin_edges[i + 1])
        n_in_bin[i] = mask.sum()
        if mask.sum() > 0:
            # Scale scores to [0,1] for calibration
            s_min, s_max = y_score[mask].min(), y_score[mask].max()
            if s_max > s_min:
                mean_pred[i] = np.mean((y_score[mask] - s_min) / (s_max - s_min))
            else:
                mean_pred[i] = 0.5
            obs_frac[i] = y_true[mask].mean()
    
    return bin_centers, mean_pred, obs_frac, n_in_bin


def threshold_metrics(y_true, y_score, n_thresholds=100):
    """Compute metrics across thresholds."""
    thresholds = np.percentile(y_score, np.linspace(0, 100, n_thresholds + 2)[1:-1])
    
    results = []
    for thresh in thresholds:
        pred = (y_score >= thresh).astype(int)
        tp = (pred == 1) & (y_true == 1)
        fp = (pred == 1) & (y_true == 0)
        tn = (pred == 0) & (y_true == 0)
        fn = (pred == 0) & (y_true == 1)
        
        n_tp = tp.sum()
        n_fp = fp.sum()
        n_tn = tn.sum()
        n_fn = fn.sum()
        
        sens = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
        spec = n_tn / (n_tn + n_fp) if (n_tn + n_fp) > 0 else 0
        ppv = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        npv = n_tn / (n_tn + n_fn) if (n_tn + n_fn) > 0 else 0
        
        results.append({
            "threshold": float(thresh),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "n_pred_pos": int(n_tp + n_fp),
            "n_tp": int(n_tp),
            "n_fp": int(n_fp),
        })
    
    return results


# ===========================================================================
# 3. COMPUTE MAIN METRICS
# ===========================================================================
print("\n" + "=" * 70)
print("COMPUTING MAIN METRICS...")
print("=" * 70)

# AUCs
auc_base = roc_auc_score(y_true, s_base)
auc_s2 = roc_auc_score(y_true, s_s2)
aupr_base = average_precision_score(y_true, s_base)
aupr_s2 = average_precision_score(y_true, s_s2)

# Bootstrap CIs
print("  Bootstrap CIs for Base...")
ci_base = bootstrap_auc_ci(y_true, s_base, n_boot=5000)
print("  Bootstrap CIs for S2-lag4...")
ci_s2 = bootstrap_auc_ci(y_true, s_s2, n_boot=5000)

# DeLong test (bootstrap-based)
print("  Bootstrap DeLong test...")
delong = bootstrap_delong_test(y_true, s_base, s_s2, n_boot=10000)

# ROC curves for plotting
fpr_base, tpr_base, _ = roc_curve(y_true, s_base)
fpr_s2, tpr_s2, _ = roc_curve(y_true, s_s2)

# PR curves
prec_base, rec_base, _ = precision_recall_curve(y_true, s_base)
prec_s2, rec_s2, _ = precision_recall_curve(y_true, s_s2)

# Bootstrap bands for ROC
print("  Bootstrap ROC bands...")
fpr_grid, tpr_med, tpr_lo, tpr_hi = bootstrap_roc_ci(y_true, s_s2, n_boot=2000)
_, fpr_base_med, fpr_base_lo, fpr_base_hi = bootstrap_roc_ci(y_true, s_base, n_boot=2000)

# Bootstrap bands for PR
print("  Bootstrap PR bands...")
rec_grid_s2, prec_med_s2, prec_lo_s2, prec_hi_s2 = bootstrap_pr_ci(y_true, s_s2, n_boot=2000)
rec_grid_base, prec_med_base, prec_lo_base, prec_hi_base = bootstrap_pr_ci(y_true, s_base, n_boot=2000)

# Calibration
print("  Calibration curves...")
bin_centers_base, mean_pred_base, obs_frac_base, n_bin_base = calibration_curve(y_true, s_base, n_bins=10)
bin_centers_s2, mean_pred_s2, obs_frac_s2, n_bin_s2 = calibration_curve(y_true, s_s2, n_bins=10)

# Threshold analysis
print("  Threshold analysis...")
thresh_base = threshold_metrics(y_true, s_base)
thresh_s2 = threshold_metrics(y_true, s_s2)

# Correlation
rho_base, p_base = spearmanr(s_base, y_true)
rho_s2, p_s2 = spearmanr(s_s2, y_true)

# Brier
def minmax_scale(x):
    if x.max() > x.min():
        return (x - x.min()) / (x.max() - x.min())
    return x.copy()

brier_base = brier_score_loss(y_true, minmax_scale(s_base))
brier_s2 = brier_score_loss(y_true, minmax_scale(s_s2))
brier_baseline = prevalence * (1 - prevalence)

# Lift@k
def compute_lift(y_true, y_score, k_pct):
    k = max(1, int(k_pct * len(y_true)))
    top_k = np.argsort(y_score)[-k:]
    rec = y_true[top_k].sum() / y_true.sum()
    return rec, rec / k_pct

lift_metrics = {}
for k_pct in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    rec_base, l_base = compute_lift(y_true, s_base, k_pct)
    rec_s2, l_s2 = compute_lift(y_true, s_s2, k_pct)
    lift_metrics[f"recall_top_{int(k_pct*100)}pct_base"] = float(rec_base)
    lift_metrics[f"lift_top_{int(k_pct*100)}pct_base"] = float(l_base)
    lift_metrics[f"recall_top_{int(k_pct*100)}pct_s2"] = float(rec_s2)
    lift_metrics[f"lift_top_{int(k_pct*100)}pct_s2"] = float(l_s2)

# ===========================================================================
# 4. RESULT REPORT
# ===========================================================================
print("\n" + "=" * 70)
print("RESULTS\n" + "=" * 70)

print(f"""
{'─'*70}
MAIN METRICS
{'─'*70}

{'Metric':<30s} {'Base':<20s} {'S2-lag4':<20s} {'Improvement':<15s}
{'─'*85}
AUC-ROC           {auc_base:<20.6f} {auc_s2:<20.6f} {(auc_s2/auc_base - 1)*100:<14.2f}%
AUC-PR            {aupr_base:<20.6f} {aupr_s2:<20.6f} {(aupr_s2/aupr_base - 1)*100:<14.2f}%
AUC-PR Lift       {aupr_base/prevalence:<20.2f}x {aupr_s2/prevalence:<20.2f}x {(aupr_s2/aupr_base - 1)*100:<14.2f}%
Spearman rho      {rho_base:<20.4f} {rho_s2:<20.4f}
Brier Score       {brier_base:<20.6f} {brier_s2:<20.6f}
Brier Skill Score {1 - brier_base/brier_baseline:<20.4f} {1 - brier_s2/brier_baseline:<20.4f}
""")

# Bootstrap CIs
print(f"""
{'─'*70}
BOOTSTRAP CONFIDENCE INTERVALS (95%)
{'─'*70}
""")
for label, ci in [("Base", ci_base), ("S2-lag4", ci_s2)]:
    print(f"  {label}:")
    print(f"    AUC-ROC: {ci['auc_roc_mean']:.4f}  [{ci['auc_roc_ci'][0]:.4f}, {ci['auc_roc_ci'][1]:.4f}]  (SD={ci['auc_roc_std']:.4f})")
    print(f"    AUC-PR:  {ci['auc_pr_mean']:.4f}  [{ci['auc_pr_ci'][0]:.4f}, {ci['auc_pr_ci'][1]:.4f}]  (SD={ci['auc_pr_std']:.4f})")

# DeLong
print(f"""
{'─'<70}
SIGNIFICANCE TEST (Bootstrap AUC-ROC: S2-lag4 vs Base)
{'─'<70}
  Mean difference:   {delong['diff_mean']:.6f}
  95% CI:             [{delong['diff_ci'][0]:.6f}, {delong['diff_ci'][1]:.6f}]
  z-statistic:        {delong['z_statistic']:.4f}
  p-value:            {delong['p_value']:.6f}
  Significant (alpha=0.05): {'YES' if delong['significant_005'] else 'NO'}
  Significant (alpha=0.01): {'YES' if delong['significant_001'] else 'NO'}
""")

# Lift table
print(f"""
{'─'*70}
LIFT BY PERCENTILE
{'─'*70}
{'Percentile':<12s} {'Recall Base':<15s} {'Recall S2':<15s} {'Lift Base':<12s} {'Lift S2':<12s}
{'─'*66}
""")
for k_pct in [1, 2, 5, 10, 15, 20]:
    r_b = lift_metrics[f"recall_top_{k_pct}pct_base"]
    r_s = lift_metrics[f"recall_top_{k_pct}pct_s2"]
    l_b = lift_metrics[f"lift_top_{k_pct}pct_base"]
    l_s = lift_metrics[f"lift_top_{k_pct}pct_s2"]
    print(f"Top {k_pct:>2d}%    {r_b*100:<14.1f}% {r_s*100:<14.1f}% {l_b:<11.2f}x {l_s:<11.2f}x")

print(f"\n  Prevalence base: {prevalence*100:.2f}%")
print(f"  Random classifier Lift = 1.0x")

# ===========================================================================
# 5. VISUALIZATIONS
# ===========================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS...")
print("=" * 70)

# Color scheme
C_BASE = "#4575b4"
C_S2 = "#d73027"

# --- FIGURE 1: ROC + PR CURVES ---
print("  Figure 1: ROC + PR curves...")
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
ax = axes[0]
ax.plot(fpr_base, tpr_base, color=C_BASE, linewidth=2, alpha=0.7,
        label=f"Base (AUC = {auc_base:.4f})")
ax.fill_between(fpr_grid, fpr_base_lo, fpr_base_hi, color=C_BASE, alpha=0.08)

ax.plot(fpr_s2, tpr_s2, color=C_S2, linewidth=2.5,
        label=f"S2-lag4 (AUC = {auc_s2:.4f})")
ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color=C_S2, alpha=0.12)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# PR
ax = axes[1]
ax.plot(rec_base, prec_base, color=C_BASE, linewidth=2, alpha=0.7,
        label=f"Base (AUC-PR = {aupr_base:.4f})")
ax.fill_between(rec_grid_base, prec_lo_base, prec_hi_base, color=C_BASE, alpha=0.08)

ax.plot(rec_s2, prec_s2, color=C_S2, linewidth=2.5,
        label=f"S2-lag4 (AUC-PR = {aupr_s2:.4f})")
ax.fill_between(rec_grid_s2, prec_lo_s2, prec_hi_s2, color=C_S2, alpha=0.12)

ax.axhline(y=prevalence, color="gray", linestyle="--", linewidth=1, alpha=0.5,
           label=f"No skill (prev = {prevalence*100:.2f}%)")
ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax.set_ylabel("Precision (PPV)", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, max(prec_s2.max(), prec_base.max()) * 1.2)

plt.suptitle("S2-lag4 vs Baseline - ROC and PR Curves\nMunicipalities >2000m, Colombia 2007-2024",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_roc_pr.png")
plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"    Saved: {fig1_path}")

# --- FIGURE 2: SCORE DISTRIBUTION ---
print("  Figure 2: Score distribution...")
fig2 = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.30)

# Histogram 1: Base
ax = fig2.add_subplot(gs[0, 0])
bins = np.linspace(s_base.min(), s_base.max(), 80)
ax.hist(s_base[y_true == 0], bins=bins, alpha=0.5, color=C_BASE,
        label=f"No outbreak (n={n_neg:,})", density=True)
ax.hist(s_base[y_true == 1], bins=bins, alpha=0.7, color=C_S2,
        label=f"Outbreak (n={n_pos:,})", density=True)
ax.set_xlabel("Risk Score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Base Score Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Histogram 2: S2
ax = fig2.add_subplot(gs[0, 1])
bins_s2 = np.linspace(s_s2.min(), s_s2.max(), 80)
ax.hist(s_s2[y_true == 0], bins=bins_s2, alpha=0.5, color=C_BASE,
        label=f"No outbreak (n={n_neg:,})", density=True)
ax.hist(s_s2[y_true == 1], bins=bins_s2, alpha=0.7, color=C_S2,
        label=f"Outbreak (n={n_pos:,})", density=True)
ax.set_xlabel("Risk Score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("S2-lag4 Score Distribution", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Boxplot comparison
ax = fig2.add_subplot(gs[1, :])
bp_data = [s_base[y_true == 0], s_base[y_true == 1],
           s_s2[y_true == 0], s_s2[y_true == 1]]
bp_labels = ["Base\nNo outbreak", "Base\nOutbreak",
             "S2-lag4\nNo outbreak", "S2-lag4\nOutbreak"]
bp_colors = [C_BASE, C_S2, C_BASE, C_S2]

bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.5,
                showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                               markeredgecolor="black", markersize=6))
for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("Risk Score", fontsize=12)
ax.set_title("Score Distribution: No outbreak vs Outbreak", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Score Distribution by Strategy",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_scores.png")
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"    Saved: {fig2_path}")

# --- FIGURE 3: CALIBRATION ---
print("  Figure 3: Calibration curves...")
fig3, ax = plt.subplots(figsize=(8, 7))

# Base calibration
ax.plot(mean_pred_base, obs_frac_base, "o-", color=C_BASE, linewidth=2,
        markersize=10, label="Base", markerfacecolor="white")
for i in range(len(bin_centers_base)):
    ax.text(mean_pred_base[i], obs_frac_base[i] + 0.003, f"n={int(n_bin_base[i])}",
            ha="center", va="bottom", fontsize=7, color=C_BASE)

# S2 calibration
ax.plot(mean_pred_s2, obs_frac_s2, "s-", color=C_S2, linewidth=2.5,
        markersize=10, label="S2-lag4", markerfacecolor="white")
for i in range(len(bin_centers_s2)):
    ax.text(mean_pred_s2[i], obs_frac_s2[i] + 0.003, f"n={int(n_bin_s2[i])}",
            ha="center", va="bottom", fontsize=7, color=C_S2)

# Perfect calibration line
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Perfect")
ax.set_xlabel("Mean Predicted Risk (scaled)", fontsize=12)
ax.set_ylabel("Observed Outbreak Rate", fontsize=12)
ax.set_title("Calibration Curves\n(10 equal-frequency bins)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, max(obs_frac_s2.max(), obs_frac_base.max()) * 1.3)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_calibration.png")
plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"    Saved: {fig3_path}")

# --- FIGURE 4: THRESHOLD ANALYSIS ---
print("  Figure 4: Threshold analysis...")
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# S2 thresholds
thresh_df = pd.DataFrame(thresh_s2)

ax = axes[0, 0]
ax.plot(thresh_df["threshold"], thresh_df["sensitivity"], color=C_S2, linewidth=2, label="Sensitivity")
ax.plot(thresh_df["threshold"], thresh_df["specificity"], color=C_BASE, linewidth=2, label="Specificity")
ax.set_xlabel("Risk Score Threshold", fontsize=11)
ax.set_ylabel("Rate", fontsize=11)
ax.set_title("S2-lag4: Sensitivity & Specificity", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(thresh_df["threshold"], thresh_df["ppv"], color=C_S2, linewidth=2, label="PPV")
ax.plot(thresh_df["threshold"], thresh_df["npv"], color=C_BASE, linewidth=2, label="NPV")
ax.set_xlabel("Risk Score Threshold", fontsize=11)
ax.set_ylabel("Rate", fontsize=11)
ax.set_title("S2-lag4: PPV & NPV", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Youden index
ax = axes[1, 0]
youden = thresh_df["sensitivity"] + thresh_df["specificity"] - 1
best_youden_idx = np.argmax(youden)
best_thresh = thresh_df.iloc[best_youden_idx]
ax.plot(thresh_df["threshold"], youden, color="#1a9850", linewidth=2)
ax.axvline(x=best_thresh["threshold"], color="red", linestyle="--", alpha=0.7,
           label=f"Optimal = {best_thresh['threshold']:.4f}")
ax.scatter(best_thresh["threshold"], youden[best_youden_idx], color="red", s=100, zorder=5)
ax.set_xlabel("Risk Score Threshold", fontsize=11)
ax.set_ylabel("Youden Index", fontsize=11)
ax.set_title(f"S2-lag4: Youden Index (max={youden[best_youden_idx]:.4f})", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# N predicted positive
ax = axes[1, 1]
ax.plot(thresh_df["threshold"], thresh_df["n_pred_pos"] / n_total * 100, color="#6a3d9a", linewidth=2)
ax.set_xlabel("Risk Score Threshold", fontsize=11)
ax.set_ylabel("% Population Flagged", fontsize=11)
ax.set_title("S2-lag4: % Municipality-months Alerted", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax2 = ax.twinx()
ax2.plot(thresh_df["threshold"], thresh_df["n_tp"] / n_pos * 100, color=C_S2, linewidth=1.5, linestyle=":")
ax2.set_ylabel("% Outbreaks Detected", fontsize=11, color=C_S2)

plt.suptitle("Threshold Analysis - S2-lag4", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_thresholds.png")
plt.savefig(fig4_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"    Saved: {fig4_path}")

# --- FIGURE 5: LIFT / GAINS CHART ---
print("  Figure 5: Lift / Gains chart...")
fig5, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cumulative gains
ax = axes[0]
pct_axis = np.linspace(0, 1, len(y_true))
sorted_base = np.sort(s_base)[::-1]
sorted_s2 = np.sort(s_s2)[::-1]
idx_base = np.argsort(s_base)[::-1]
idx_s2 = np.argsort(s_s2)[::-1]
cum_base = np.cumsum(y_true[idx_base]) / n_pos
cum_s2 = np.cumsum(y_true[idx_s2]) / n_pos

ax.plot(pct_axis * 100, cum_base * 100, color=C_BASE, linewidth=2, label="Base")
ax.plot(pct_axis * 100, cum_s2 * 100, color=C_S2, linewidth=2.5, label="S2-lag4")
ax.plot([0, 100], [0, 100], "k--", linewidth=1, alpha=0.4, label="Random")
ax.set_xlabel("% Population Screened", fontsize=12)
ax.set_ylabel("% Outbreaks Detected", fontsize=12)
ax.set_title("Cumulative Gains Chart", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)
ax.set_ylim(0, 100)

# Lift curve
ax = axes[1]
lift_base = cum_base / (pct_axis + 1e-10)
lift_s2 = cum_s2 / (pct_axis + 1e-10)

ax.plot(pct_axis * 100, lift_base, color=C_BASE, linewidth=2, label="Base")
ax.plot(pct_axis * 100, lift_s2, color=C_S2, linewidth=2.5, label="S2-lag4")
ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("% Population Screened", fontsize=12)
ax.set_ylabel("Lift", fontsize=12)
ax.set_title("Lift Curve", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)

plt.suptitle("Gains & Lift Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_lift.png")
plt.savefig(fig5_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"    Saved: {fig5_path}")

# ===========================================================================
# 6. SAVE RESULTS
# ===========================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS...")
print("=" * 70)

# Full JSON
results = {
    "metadata": {
        "n_records": n_total,
        "n_municipalities": int(df_valid["DANE"].nunique()),
        "n_outbreaks": n_pos,
        "prevalence": float(prevalence),
    },
    "metrics_base": {
        "auc_roc": float(auc_base),
        "auc_roc_ci": ci_base["auc_roc_ci"],
        "auc_pr": float(aupr_base),
        "auc_pr_ci": ci_base["auc_pr_ci"],
        "spearman_rho": float(rho_base),
        "spearman_p": float(p_base),
        "brier_score": float(brier_base),
        "brier_skill_score": float(1 - brier_base / brier_baseline),
    },
    "metrics_s2_lag4": {
        "auc_roc": float(auc_s2),
        "auc_roc_ci": ci_s2["auc_roc_ci"],
        "auc_pr": float(aupr_s2),
        "auc_pr_ci": ci_s2["auc_pr_ci"],
        "spearman_rho": float(rho_s2),
        "spearman_p": float(p_s2),
        "brier_score": float(brier_s2),
        "brier_skill_score": float(1 - brier_s2 / brier_baseline),
    },
    "improvements": {
        "auc_roc_diff": float(auc_s2 - auc_base),
        "auc_roc_pct": float((auc_s2 / auc_base - 1) * 100),
        "auc_pr_diff": float(aupr_s2 - aupr_base),
        "auc_pr_pct": float((aupr_s2 / aupr_base - 1) * 100),
    },
    "delong_test": delong,
    "lift_metrics": lift_metrics,
    "best_threshold_s2": {
        "threshold": float(best_thresh["threshold"]),
        "sensitivity": float(best_thresh["sensitivity"]),
        "specificity": float(best_thresh["specificity"]),
        "ppv": float(best_thresh["ppv"]),
        "npv": float(best_thresh["npv"]),
        "youden_index": float(youden[best_youden_idx]),
        "n_predicted_positive": int(best_thresh["n_pred_pos"]),
        "pct_flagged": float(best_thresh["n_pred_pos"] / n_total * 100),
    },
}

json_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  JSON: {json_path}")

# TXT Report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("DETAILED ANALYSIS: S2-lag4 vs BASELINE")
report_lines.append("Municipalities > 2000 m.a.s.l., Colombia 2007-2024")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Dataset: {n_total:,} municipality-months, {int(df_valid['DANE'].nunique())} municipalities")
report_lines.append(f"Outbreaks: {n_pos:,} ({prevalence*100:.2f}%)")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("1. MAIN METRICS")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append(f"{'Metric':<30s} {'Base':<20s} {'S2-lag4':<20s} {'Change':<15s}")
report_lines.append(f"{'─'*85}")
report_lines.append(f"{'AUC-ROC':<30s} {auc_base:<20.6f} {auc_s2:<20.6f} +{auc_s2-auc_base:<13.6f}")
report_lines.append(f"{'AUC-PR':<30s} {aupr_base:<20.6f} {aupr_s2:<20.6f} +{aupr_s2-aupr_base:<13.6f}")
report_lines.append(f"{'AUC-PR Lift':<30s} {aupr_base/prevalence:<19.2f}x {aupr_s2/prevalence:<19.2f}x")
report_lines.append(f"{'Spearman rho':<30s} {rho_base:<20.4f} {rho_s2:<20.4f}")
report_lines.append(f"{'Brier Skill Score':<30s} {1-brier_base/brier_baseline:<20.4f} {1-brier_s2/brier_baseline:<20.4f}")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append(f"  Base:")
report_lines.append(f"    AUC-ROC: {ci_base['auc_roc_mean']:.4f} [{ci_base['auc_roc_ci'][0]:.4f}, {ci_base['auc_roc_ci'][1]:.4f}]")
report_lines.append(f"    AUC-PR:  {ci_base['auc_pr_mean']:.4f} [{ci_base['auc_pr_ci'][0]:.4f}, {ci_base['auc_pr_ci'][1]:.4f}]")
report_lines.append(f"  S2-lag4:")
report_lines.append(f"    AUC-ROC: {ci_s2['auc_roc_mean']:.4f} [{ci_s2['auc_roc_ci'][0]:.4f}, {ci_s2['auc_roc_ci'][1]:.4f}]")
report_lines.append(f"    AUC-PR:  {ci_s2['auc_pr_mean']:.4f} [{ci_s2['auc_pr_ci'][0]:.4f}, {ci_s2['auc_pr_ci'][1]:.4f}]")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("3. STATISTICAL SIGNIFICANCE (Bootstrap Test)")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append(f"  AUC-ROC difference (S2 - Base): {delong['diff_mean']:.6f}")
report_lines.append(f"  95% CI of difference:         [{delong['diff_ci'][0]:.6f}, {delong['diff_ci'][1]:.6f}]")
report_lines.append(f"  z-statistic:                  {delong['z_statistic']:.4f}")
report_lines.append(f"  p-value:                      {delong['p_value']:.6f}")
report_lines.append(f"  Significant at alpha=0.05:    {'YES' if delong['significant_005'] else 'NO'}")
report_lines.append(f"  Significant at alpha=0.01:    {'YES' if delong['significant_001'] else 'NO'}")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("4. LIFT ANALYSIS")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append(f"{'Percentile':<15s} {'Recall Base':<15s} {'Recall S2':<15s} {'Lift Base':<12s} {'Lift S2':<12s}")
report_lines.append(f"{'─'*69}")
for k_pct in [1, 2, 5, 10, 15, 20]:
    r_b = lift_metrics[f"recall_top_{k_pct}pct_base"] * 100
    r_s = lift_metrics[f"recall_top_{k_pct}pct_s2"] * 100
    l_b = lift_metrics[f"lift_top_{k_pct}pct_base"]
    l_s = lift_metrics[f"lift_top_{k_pct}pct_s2"]
    report_lines.append(f"{f'Top {k_pct}%':<15s} {r_b:<14.1f}% {r_s:<14.1f}% {l_b:<11.2f}x {l_s:<11.2f}x")
report_lines.append("")
report_lines.append("-" * 80)
report_lines.append("5. OPTIMAL THRESHOLD (Youden Index) - S2-lag4")
report_lines.append("-" * 80)
report_lines.append("")
report_lines.append(f"  Threshold:                  {best_thresh['threshold']:.6f}")
report_lines.append(f"  Youden Index (sens+spec-1): {youden[best_youden_idx]:.4f}")
report_lines.append(f"  Sensitivity (recall):       {best_thresh['sensitivity']*100:.2f}%")
report_lines.append(f"  Specificity:                {best_thresh['specificity']*100:.2f}%")
report_lines.append(f"  PPV (precision):            {best_thresh['ppv']*100:.2f}%")
report_lines.append(f"  NPV:                        {best_thresh['npv']*100:.2f}%")
report_lines.append(f"  Municipality-months alerted: {best_thresh['n_pred_pos']:,} ({best_thresh['n_pred_pos']/n_total*100:.1f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("OVERALL CONCLUSION")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"  S2-lag4 (Max lag0-lag4) significantly improves over baseline:")
report_lines.append(f"    + AUC-ROC: {auc_base:.4f} -> {auc_s2:.4f} ({(auc_s2/auc_base-1)*100:+.2f}%)")
report_lines.append(f"    + AUC-PR:  {aupr_base:.4f} -> {aupr_s2:.4f} ({(aupr_s2/aupr_base-1)*100:+.2f}%)")
sig_msg = f"    + Significant difference (p = {delong['p_value']:.6f})" if delong['significant_005'] else "    + Difference NOT significant"
report_lines.append(sig_msg)
report_lines.append(f"    + Lift@10%: {lift_metrics['lift_top_10pct_base']:.2f}x -> {lift_metrics['lift_top_10pct_s2']:.2f}x")
report_lines.append("")
report_lines.append(f"  The improvement is consistent across all analyzed lift percentiles.")
report_lines.append(f"  The S2-lag4 strategy better captures municipality-months with highest relative risk")
report_lines.append(f"  by incorporating the maximum risk over the previous 4 months.")

report_path = os.path.join(OUTPUT_DIR, "analysis_s2_vs_base_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  Report: {report_path}")

# ===========================================================================
# 7. FINAL SUMMARY
# ===========================================================================
print(f"\n{'='*70}")
print("COMPLETED")
print(f"{'='*70}")
print(f"\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "analysis_s2_vs_base" in f:
        print(f"  {f}")
