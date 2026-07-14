#!/usr/bin/env python
"""
optimize_s3_weights.py
=======================
Search for optimal weights for the S3 strategy (weighted mean)
with lag0-lag4 window, maximizing AUC-PR.

Methodology:
  - Random search with 20,000 descending-weight combinations
  - Each combination: 5 random numbers -> sort descending -> normalize sum=1
  - Evaluate AUC-PR for each combination
  - Identify the weights that maximize AUC-PR

Output:
  - s3_opt_weights.json        : best weights and metrics
  - s3_opt_comparison.txt      : complete report
  - s3_weight_optimization.png : weight vs AUC-PR scatter plot
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, dirichlet
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
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

print(f"\nTotal records: {len(df):,}")
print(f"Outbreak prevalence: {df['has_outbreak'].mean()*100:.2f}%")

# ===========================================================================
# 2. CREATE LAGS
# ===========================================================================
print("\n" + "=" * 70)
print("CREATING LAGS...")
print("=" * 70)

df = df.sort_values(["DANE", "Year_month"]).reset_index(drop=True)
df["risk_lag1"] = df.groupby("DANE")["risk"].shift(1)
df["risk_lag2"] = df.groupby("DANE")["risk"].shift(2)
df["risk_lag3"] = df.groupby("DANE")["risk"].shift(3)
df["risk_lag4"] = df.groupby("DANE")["risk"].shift(4)

# Filter only records with all 4 lags available
df_valid = df.dropna(subset=["risk_lag4"]).copy()
print(f"Records with full lags (lag0-lag4 available): {len(df_valid):,}")

y_true = df_valid["has_outbreak"].values
lags = df_valid[["risk", "risk_lag1", "risk_lag2", "risk_lag3", "risk_lag4"]].values

# For CV
groups = (df_valid["DANE"].astype(str) + "_" + df_valid["year"].astype(str)).values

# ===========================================================================
# 3. SEARCH FOR OPTIMAL WEIGHTS (RANDOM SEARCH)
# ===========================================================================
print("\n" + "=" * 70)
print("RANDOM SEARCH FOR OPTIMAL WEIGHTS...")
print("=" * 70)

N_ITERATIONS = 20000
N_BEST_TO_SHOW = 20

# Generate reproducible seed
rng = np.random.RandomState(42)

# Pre-allocate
all_weights = np.zeros((N_ITERATIONS, 5))
all_aucpr = np.zeros(N_ITERATIONS)
all_aucroc = np.zeros(N_ITERATIONS)

# Sampling strategies:
# 1) Dirichlet(alpha=[2,1.5,1,0.5,0.3]) - bias toward heavier lag0
# 2) Dirichlet(alpha=[5,3,2,1,0.5]) - strong bias toward lag0
# 3) Dirichlet(alpha=[1,1,1,1,1]) - uniform
# 4) Uniform sampling with descending sort
# 5) Exponential decaying + noise

print(f"Searching {N_ITERATIONS:,} weight combinations...")

for i in range(N_ITERATIONS):
    # Alternate between different sampling strategies
    method = i % 5
    
    if method == 0:
        # Dirichlet with bias toward lag0
        alpha = [5, 3, 2, 1, 0.5]
        w = rng.dirichlet(alpha)
    elif method == 1:
        # Dirichlet with moderate bias
        alpha = [3, 2, 1.5, 1, 0.5]
        w = rng.dirichlet(alpha)
    elif method == 2:
        # Uniform (exploration)
        alpha = [1, 1, 1, 1, 1]
        w = rng.dirichlet(alpha)
    elif method == 3:
        # 5 uniform values, sort descending, normalize
        raw = rng.uniform(0, 1, 5)
        raw.sort()
        w = raw[::-1]  # descending
        w = w / w.sum()
    else:
        # Exponential decaying + noise
        base = np.exp(-np.arange(5) * rng.uniform(0.2, 1.0))
        noise = rng.uniform(0.8, 1.2, 5)
        w = base * noise
        w = w / w.sum()
    
    all_weights[i] = w
    
    # Compute weighted risk
    y_score = lags @ w  # dot product
    
    # AUC-PR
    all_aucpr[i] = average_precision_score(y_true, y_score)
    all_aucroc[i] = roc_auc_score(y_true, y_score)

# ===========================================================================
# 4. FIND BEST RESULTS
# ===========================================================================
print("\n" + "=" * 70)
print("FINDING OPTIMAL WEIGHTS...")
print("=" * 70)

# Top N by AUC-PR
best_idx = np.argsort(all_aucpr)[::-1]
top_n = min(N_BEST_TO_SHOW, N_ITERATIONS)

print(f"\nTop {top_n} weight combinations (by AUC-PR):")
print(f"{'Rank':<5} {'AUC-PR':<10} {'AUC-ROC':<10} {'w0(lag0)':<10} {'w1(lag1)':<10} {'w2(lag2)':<10} {'w3(lag3)':<10} {'w4(lag4)':<10}")
print("-" * 75)

for rank, idx in enumerate(best_idx[:top_n]):
    w = all_weights[idx]
    print(f"{rank+1:<5} {all_aucpr[idx]:<10.6f} {all_aucroc[idx]:<10.4f} "
          f"{w[0]:<10.4f} {w[1]:<10.4f} {w[2]:<10.4f} {w[3]:<10.4f} {w[4]:<10.4f}")

# Best unique combination
best_w = all_weights[best_idx[0]]
best_aucpr = all_aucpr[best_idx[0]]
best_aucroc = all_aucroc[best_idx[0]]

print(f"\n{'='*60}")
print(f"BEST WEIGHTS FOUND:")
print(f"{'='*60}")
print(f"  w0 (lag0): {best_w[0]:.4f}")
print(f"  w1 (lag1): {best_w[1]:.4f}")
print(f"  w2 (lag2): {best_w[2]:.4f}")
print(f"  w3 (lag3): {best_w[3]:.4f}")
print(f"  w4 (lag4): {best_w[4]:.4f}")
print(f"  Sum:       {best_w.sum():.4f}")
print(f"")
print(f"  AUC-PR:  {best_aucpr:.6f}")
print(f"  AUC-ROC: {best_aucroc:.6f}")
print(f"  Lift:     {best_aucpr / y_true.mean():.2f}x")

# ===========================================================================
# 5. COMPUTE FULL METRICS FOR THE BEST S3
# ===========================================================================
print("\n" + "=" * 70)
print("FULL METRICS FOR OPTIMAL S3-lag4...")
print("=" * 70)

y_score_opt = lags @ best_w

# Full metrics
def full_metrics(y_true, y_score):
    m = {}
    rho, p_rho = spearmanr(y_score, y_true)
    m["spearman_rho"] = float(rho)
    m["spearman_p"] = float(p_rho)
    tau, p_tau = kendalltau(y_score, y_true)
    m["kendall_tau"] = float(tau)
    m["auc_roc"] = float(roc_auc_score(y_true, y_score))
    prevalence = y_true.mean()
    m["auc_pr"] = float(average_precision_score(y_true, y_score))
    m["auc_pr_lift"] = float(m["auc_pr"] / prevalence) if prevalence > 0 else 0
    m["prevalence"] = float(prevalence)
    
    for k_pct in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(k_pct * len(y_true)))
        top_k = np.argsort(y_score)[-k:]
        rec = y_true[top_k].sum() / y_true.sum()
        m[f"recall_top_{int(k_pct*100)}pct"] = float(rec)
        m[f"lift_top_{int(k_pct*100)}pct"] = float(rec / k_pct)
    
    risk_min, risk_max = y_score.min(), y_score.max()
    if risk_max > risk_min:
        risk_scaled = (y_score - risk_min) / (risk_max - risk_min)
    else:
        risk_scaled = y_score.copy()
    brier = brier_score_loss(y_true, risk_scaled)
    brier_base = prevalence * (1 - prevalence)
    m["brier_score"] = float(brier)
    m["brier_skill_score"] = float(1 - brier / brier_base) if brier_base > 0 else 0
    return m

opt_metrics = full_metrics(y_true, y_score_opt)

# CV
gkf = GroupKFold(n_splits=5)
cv_roc, cv_pr = [], []
for train_idx, test_idx in gkf.split(y_true, y_true, groups):
    y_test = y_true[test_idx]
    s_test = y_score_opt[test_idx]
    if y_test.sum() > 0 and (1 - y_test).sum() > 0:
        cv_roc.append(roc_auc_score(y_test, s_test))
        cv_pr.append(average_precision_score(y_test, s_test))
opt_metrics["cv_auc_roc_mean"] = float(np.mean(cv_roc)) if cv_roc else 0
opt_metrics["cv_auc_roc_std"] = float(np.std(cv_roc)) if cv_roc else 0
opt_metrics["cv_auc_pr_mean"] = float(np.mean(cv_pr)) if cv_pr else 0
opt_metrics["cv_auc_pr_std"] = float(np.std(cv_pr)) if cv_pr else 0

for key, val in opt_metrics.items():
    if isinstance(val, float):
        print(f"  {key:25s}: {val:.6f}")
    else:
        print(f"  {key:25s}: {val}")

# ===========================================================================
# 6. COMPARE WITH PREVIOUS STRATEGIES
# ===========================================================================
print("\n" + "=" * 70)
print("COMPARISON WITH PREVIOUS STRATEGIES...")
print("=" * 70)

# Load previous results
prev_results = {}
prev_path = os.path.join(OUTPUT_DIR, "lags_results.json")
if os.path.exists(prev_path):
    with open(prev_path) as f:
        prev_results = json.load(f)

# Label for optimized S3
w_str = f"({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f},{best_w[3]:.2f},{best_w[4]:.2f})"
opt_label = f"S3: Opt. {w_str} lag4"

# Comparison table
all_strats = {}
if prev_results:
    all_strats.update(prev_results)
all_strats[opt_label] = opt_metrics

# Key metrics
key_metrics = [
    ("auc_roc", "AUC-ROC"),
    ("auc_pr", "AUC-PR"),
    ("auc_pr_lift", "AUC-PR Lift"),
    ("spearman_rho", "Spearman rho"),
    ("recall_top_10pct", "Recall@10%"),
    ("cv_auc_pr_mean", "CV AUC-PR"),
    ("brier_skill_score", "Brier Skill Score"),
]

print(f"\n{'Strategy':<45s} {'AUC-ROC':<10s} {'AUC-PR':<10s} {'Lift':<8s} {'Recall@10':<10s} {'CV AUC-PR':<10s}")
print("-" * 95)
for label in all_strats:
    m = all_strats[label]
    auc_roc = m.get("auc_roc", 0)
    auc_pr = m.get("auc_pr", 0)
    lift = m.get("auc_pr_lift", 0)
    rec10 = m.get("recall_top_10pct", 0)
    cv_pr = m.get("cv_auc_pr_mean", 0)
    print(f"{label:<45s} {auc_roc:<10.4f} {auc_pr:<10.6f} {lift:<8.2f}x {rec10*100:<9.1f}% {cv_pr:<10.4f}")

# ===========================================================================
# 7. VISUALIZATION: TOP WEIGHTS
# ===========================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Panel 1: Top 10 weight combinations ---
top10_idx = best_idx[:10]
top10_w = all_weights[top10_idx]
top10_pr = all_aucpr[top10_idx]

ax1 = axes[0]
x = np.arange(10)
bar_width = 0.15
colors = ["#d73027", "#fdae61", "#1a9850", "#4575b4", "#6a3d9a"]

for i in range(5):
    ax1.bar(x + i * bar_width, top10_w[:, i], bar_width,
            label=f"lag{i}" if i > 0 else "lag0", color=colors[i], alpha=0.85)

ax1.set_xlabel("Rank", fontsize=12)
ax1.set_ylabel("Weight", fontsize=12)
ax1.set_title("Top 10 Weight Combinations (by AUC-PR)", fontsize=13, fontweight="bold")
ax1.set_xticks(x + bar_width * 2)
ax1.set_xticklabels([f"#{i+1}" for i in range(10)])
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")

# Annotate AUC-PR of each combination
for i in range(10):
    ax1.text(i + bar_width * 2, 1.02, f"PR={top10_pr[i]:.4f}",
             ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=45)

# --- Panel 2: Distribution of AUC-PR vs lag0 weight ---
ax2 = axes[1]
sc = ax2.scatter(all_weights[:, 0], all_aucpr, c=all_weights[:, 1],
                 s=3, alpha=0.4, cmap="viridis")
ax2.axhline(y=0.0429, color="red", linestyle="--", linewidth=1.5, label=f"S2-lag4 (0.0429)")
ax2.axhline(y=0.0404, color="orange", linestyle=":", linewidth=1.5, label=f"S2-lag3 (0.0404)")
# Mark the optimum
ax2.scatter(best_w[0], best_aucpr, c="red", s=100, marker="*", edgecolors="white",
            linewidth=1.5, zorder=5, label=f"Optimum ({best_aucpr:.4f})")
ax2.set_xlabel("Weight of lag0 (w0)", fontsize=12)
ax2.set_ylabel("AUC-PR", fontsize=12)
ax2.set_title(f"Weight Search Distribution (n={N_ITERATIONS:,})", fontsize=13, fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax2)
cbar.set_label("Weight of lag1 (w1)", fontsize=10)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "s3_weight_optimization.png")
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Plot: {plot_path}")

# ===========================================================================
# 8. SAVE RESULTS
# ===========================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS...")
print("=" * 70)

# JSON with optimal weights
opt_results = {
    "optimal_weights": {
        "w0_lag0": float(best_w[0]),
        "w1_lag1": float(best_w[1]),
        "w2_lag2": float(best_w[2]),
        "w3_lag3": float(best_w[3]),
        "w4_lag4": float(best_w[4]),
        "sum": float(best_w.sum()),
    },
    "metrics": opt_metrics,
    "n_iterations": N_ITERATIONS,
    "top_10_weights": [
        {
            "rank": rank + 1,
            "weights": {
                "w0_lag0": float(all_weights[idx][0]),
                "w1_lag1": float(all_weights[idx][1]),
                "w2_lag2": float(all_weights[idx][2]),
                "w3_lag3": float(all_weights[idx][3]),
                "w4_lag4": float(all_weights[idx][4]),
            },
            "auc_pr": float(all_aucpr[idx]),
            "auc_roc": float(all_aucroc[idx]),
        }
        for rank, idx in enumerate(best_idx[:10])
    ],
}

with open(os.path.join(OUTPUT_DIR, "s3_opt_weights.json"), "w") as f:
    json.dump(opt_results, f, indent=2)

# Comparison with S2 et al
comparison = {
    "S3_optimized": {
        "label": opt_label,
        "weights": {
            "w0_lag0": float(best_w[0]),
            "w1_lag1": float(best_w[1]),
            "w2_lag2": float(best_w[2]),
            "w3_lag3": float(best_w[3]),
            "w4_lag4": float(best_w[4]),
        },
        "metrics": opt_metrics,
    },
}

# Load S2-lag4 from previous JSON instead of hardcoding
s2_aucpr = 0
s3_manual_aucpr = 0
if "S2: Max lag0-lag4" in prev_results:
    s2_aucpr = prev_results["S2: Max lag0-lag4"].get("auc_pr", 0)
elif "S2: Max lag0-lag4" in all_strats:
    s2_aucpr = all_strats["S2: Max lag0-lag4"].get("auc_pr", 0)

s3_manual_label = [k for k in all_strats if "Weighted" in k and "lag4" in k]
if s3_manual_label:
    s3_manual_aucpr = all_strats[s3_manual_label[0]].get("auc_pr", 0)

comparison = {
    "S3_optimized": {
        "label": opt_label,
        "weights": {
            "w0_lag0": float(best_w[0]),
            "w1_lag1": float(best_w[1]),
            "w2_lag2": float(best_w[2]),
            "w3_lag3": float(best_w[3]),
            "w4_lag4": float(best_w[4]),
        },
        "metrics": opt_metrics,
    },
    "S2_lag4_best": {
        "auc_pr": s2_aucpr,
        "auc_roc": all_strats.get("S2: Max lag0-lag4", {}).get("auc_roc", 0) if "S2: Max lag0-lag4" in all_strats else 0,
    },
    "S3_manual_lag4": {
        "auc_pr": s3_manual_aucpr,
        "label": s3_manual_label[0] if s3_manual_label else None,
    }
}

with open(os.path.join(OUTPUT_DIR, "s3_opt_comparison.json"), "w") as f:
    json.dump(comparison, f, indent=2)

# ===========================================================================
# 9. FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
OPTIMAL S3-lag4 WEIGHTS:
  w0 (lag0) = {best_w[0]:.4f}  ({best_w[0]*100:.1f}% weight)
  w1 (lag1) = {best_w[1]:.4f}  ({best_w[1]*100:.1f}% weight)
  w2 (lag2) = {best_w[2]:.4f}  ({best_w[2]*100:.1f}% weight)
  w3 (lag3) = {best_w[3]:.4f}  ({best_w[3]*100:.1f}% weight)
  w4 (lag4) = {best_w[4]:.4f}  ({best_w[4]*100:.1f}% weight)

PERFORMANCE:
  AUC-PR  = {best_aucpr:.6f}
  AUC-ROC = {best_aucroc:.6f}
  Lift    = {best_aucpr / y_true.mean():.2f}x

COMPARISON vs S2-lag4 and manual S3-lag4:
  S3-opt     AUC-PR  = {best_aucpr:.4f}
  S2-lag4    AUC-PR  = {s2_aucpr:.4f}
  S3-manual  AUC-PR  = {s3_manual_aucpr:.4f}

  -> S2-lag4 {'is still BETTER' if s2_aucpr > best_aucpr else 'is WORSE than S3-opt'} in AUC-PR
     S3-opt vs S2-lag4: diff = {abs(s2_aucpr - best_aucpr):.4f}
     S3-opt vs S3-manual: diff = {abs(s3_manual_aucpr - best_aucpr):.4f}
""")

print("Results saved to:")
for f in ["s3_opt_weights.json", "s3_opt_comparison.json", "s3_weight_optimization.png"]:
    print(f"  {os.path.join(OUTPUT_DIR, f)}")

print(f"\n{'='*70}")
print("COMPLETED")
print(f"{'='*70}")
