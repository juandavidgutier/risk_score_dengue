#!/usr/bin/env python
"""
test_lags_component_opt.py
===========================
Evaluation of individual lags per component:
  - potential_intro (Pintro) with lag from 0 to 4 months
  - R0_rel with lag from 0 to 4 months

For each combination (lag_pintro, lag_r0rel), the risk is recomputed as:
  Risk = Pintro_s[t - lag_pintro] x R0_rel[t - lag_r0rel] x S[t] x D[t]

Where S = susceptibility and D = scaled population density (no lag).
The 25 combinations are then compared against existing strategies
(Base, S1, S2, S3, S4/DLNM).

Output in lags/:
  - component_lag_results.json   : metrics for all 25 combinations
  - component_lag_comparison.txt : readable comparison report
  - component_lag_heatmap.png    : heatmap of AUC-PR for the 25 combinations
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIG
# ===========================================================================
BASE_DIR = "C:/"
OUTPUT_DIR = os.path.join(BASE_DIR, "lags")
EPS = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# 1. LOAD DATA (similar to compute_final_risk.py)
# ===========================================================================
print("=" * 70)
print("COMPONENT-LEVEL LAG OPTIMIZATION")
print("=" * 70)
print("\n[1/5] Loading data...")

# --- potential_intro ---
intro = pd.read_csv(
    os.path.join(BASE_DIR, "potential_introduction", "potential_intro_monthly.csv.gz"),
    compression="gzip"
)
intro["Year_month"] = intro["Year_month"].astype(str)
print(f"  potential_intro: {len(intro):,} rec, {intro['DANE'].nunique()} munis")

# --- R0_rel ---
r0 = pd.read_csv(os.path.join(BASE_DIR, "R0", "R0_rel_monthly.csv"))
r0["Year_month"] = r0["Year_month"].astype(str)
print(f"  R0_rel: {len(r0):,} rec, {r0['DANE'].nunique()} munis")

# --- susceptibility ---
sus = pd.read_csv(os.path.join(BASE_DIR, "proxy_susceptibility", "susceptibility_monthly.csv"))
sus["Year_month"] = sus["Year_month"].astype(str)
print(f"  susceptibility: {len(sus):,} rec, {sus['DANE'].nunique()} munis")

# --- annual population ---
pop_annual = pd.read_csv(os.path.join(BASE_DIR, "radiation", "population_per_municipality.csv"))
pop_long = pop_annual.melt(
    id_vars=["DANE"],
    value_vars=[f"pop_{y}" for y in range(2007, 2025)],
    var_name="year_str", value_name="population"
)
pop_long["year"] = pop_long["year_str"].str.replace("pop_", "").astype(int)

# --- shapefile for area and names ---
import geopandas as gpd
shp = gpd.read_file(os.path.join(BASE_DIR, "map", "map_Colombia.shp"))
shp["DANE"] = shp["DANE"].astype(int)
shp_proj = shp.to_crs("EPSG:3857")
shp["area_km2"] = shp_proj.geometry.area / 1_000_000
print(f"  shapefile: {len(shp)} munis")

# --- altitude ---
alt_df = pd.read_csv(os.path.join(BASE_DIR, "map", "altitude.csv"))
alt_df["DANE"] = alt_df["DANE"].astype(int)

# --- cases ---
cases_df = pd.read_csv(os.path.join(BASE_DIR, "cases", "cases_dengue.csv"))
cases_df["Year_month"] = cases_df["Year_month"].astype(str)

# ===========================================================================
# 2. MERGE COMPONENTS (same steps as compute_final_risk.py)
# ===========================================================================
print("\n[2/5] Merging components...")

# Merge intro + R0
merged = intro.merge(r0[["DANE", "Year_month", "R0_rel"]], on=["DANE", "Year_month"], how="left")
print(f"  intro+R0: {len(merged):,} rec")

# Merge susceptibility
merged = merged.merge(sus[["DANE", "Year_month", "susceptibility"]],
                       on=["DANE", "Year_month"], how="left")
merged["susceptibility"] = merged["susceptibility"].fillna(1.0)
print(f"  +susceptibility: {len(merged):,} rec")

# Merge population
merged["year"] = merged["Year_month"].apply(lambda x: int(x.split("-")[0]))
merged = merged.merge(pop_long[["DANE", "year", "population"]], on=["DANE", "year"], how="left")
n_missing_pop = merged["population"].isna().sum()
if n_missing_pop > 0:
    med_pop = merged.groupby("DANE")["population"].transform("median")
    merged["population"] = merged["population"].fillna(med_pop)
print(f"  +population: {len(merged):,} rec ({n_missing_pop} NaN filled)")

# Merge shapefile
merged = merged.merge(shp[["DANE", "MPIO_CNMBR", "area_km2"]], on="DANE", how="left")

# Merge altitude
merged = merged.merge(alt_df[["DANE", "altitude_min"]], on="DANE", how="left")

# ===========================================================================
# 3. COMPUTE SCALED COMPONENTS
# ===========================================================================
print("\n[3/5] Computing scaled components...")

def minmax_floor(series, eps=EPS):
    """Rescales a series to [eps, 1] via min-max."""
    s = series.astype(np.float64)
    s_min, s_max = np.nanmin(s), np.nanmax(s)
    rng = s_max - s_min
    if rng == 0 or np.isnan(rng):
        return pd.Series(np.ones(len(s)), index=s.index)
    s_norm = (s - s_min) / rng
    return eps + (1.0 - eps) * s_norm

# potential_intro_s: min-max scaling with floor
merged["potential_intro_s"] = minmax_floor(merged["potential_intro"])

# density: population / area
merged["density"] = merged["population"] / merged["area_km2"]
merged["density"] = merged["density"].replace([np.inf, -np.inf], np.nan)
med_density = merged.groupby("DANE")["density"].transform("median")
merged["density"] = merged["density"].fillna(med_density)
merged["density_s"] = minmax_floor(merged["density"])

# R0_rel and susceptibility in native scale
merged["R0_rel_use"] = merged["R0_rel"].clip(lower=0.0).fillna(0.0)
merged["susceptibility_use"] = merged["susceptibility"].clip(lower=0.0, upper=1.0).fillna(1.0)

print(f"  potential_intro_s: [{merged['potential_intro_s'].min():.6f}, {merged['potential_intro_s'].max():.6f}]")
print(f"  R0_rel_use:        [{merged['R0_rel_use'].min():.6f}, {merged['R0_rel_use'].max():.6f}]")
print(f"  susceptibility_use: [{merged['susceptibility_use'].min():.6f}, {merged['susceptibility_use'].max():.6f}]")
print(f"  density_s:         [{merged['density_s'].min():.6f}, {merged['density_s'].max():.6f}]")

# Base risk (no lags)
merged["risk_base"] = (merged["potential_intro_s"] *
                       merged["R0_rel_use"] *
                       merged["susceptibility_use"] *
                       merged["density_s"])

# ===========================================================================
# 4. FILTER >2000m AND CREATE COMPONENT LAGS
# ===========================================================================
print("\n[4/5] Filtering >2000m and creating component lags...")

destinations = merged[merged["altitude_min"] > 2000].copy()
n_destinations = destinations["DANE"].nunique()
print(f"  Destinations >2000m: {n_destinations} munis, {len(destinations):,} rec")

# Sort by DANE and Year_month
destinations = destinations.sort_values(["DANE", "Year_month"]).reset_index(drop=True)

# Create lags for potential_intro_s (1 to 4)
destinations["pintro_s_lag1"] = destinations.groupby("DANE")["potential_intro_s"].shift(1)
destinations["pintro_s_lag2"] = destinations.groupby("DANE")["potential_intro_s"].shift(2)
destinations["pintro_s_lag3"] = destinations.groupby("DANE")["potential_intro_s"].shift(3)
destinations["pintro_s_lag4"] = destinations.groupby("DANE")["potential_intro_s"].shift(4)

# Create lags for R0_rel_use (1 to 4)
destinations["r0_lag1"] = destinations.groupby("DANE")["R0_rel_use"].shift(1)
destinations["r0_lag2"] = destinations.groupby("DANE")["R0_rel_use"].shift(2)
destinations["r0_lag3"] = destinations.groupby("DANE")["R0_rel_use"].shift(3)
destinations["r0_lag4"] = destinations.groupby("DANE")["R0_rel_use"].shift(4)

# ===========================================================================
# 5. METRIC FUNCTIONS
# ===========================================================================
def compute_metrics(y_true, y_score):
    """Calculate metrics for binary scoring."""
    metrics = {}
    
    rho, p_rho = spearmanr(y_score, y_true)
    metrics["spearman_rho"] = float(rho)
    metrics["spearman_p"] = float(p_rho)
    
    tau, _ = kendalltau(y_score, y_true)
    metrics["kendall_tau"] = float(tau)
    
    auc_roc = roc_auc_score(y_true, y_score)
    metrics["auc_roc"] = float(auc_roc)
    
    prevalence = y_true.mean()
    auc_pr = average_precision_score(y_true, y_score)
    metrics["auc_pr"] = float(auc_pr)
    metrics["auc_pr_lift"] = float(auc_pr / prevalence) if prevalence > 0 else 0
    metrics["prevalence"] = float(prevalence)
    
    # Recall@k
    for k_pct in [0.01, 0.02, 0.05, 0.10, 0.20]:
        k = max(1, int(k_pct * len(y_true)))
        top_k_idx = np.argsort(y_score)[-k:]
        recall_k = y_true[top_k_idx].sum() / y_true.sum()
        metrics[f"recall_top_{int(k_pct*100)}pct"] = float(recall_k)
        metrics[f"lift_top_{int(k_pct*100)}pct"] = float(recall_k / k_pct)
    
    # Brier
    risk_min, risk_max = y_score.min(), y_score.max()
    risk_scaled = (y_score - risk_min) / (risk_max - risk_min) if risk_max > risk_min else y_score.copy()
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
# 6. EVALUATE ALL COMBINATIONS (lag_pintro x lag_r0rel)
# ===========================================================================
print("\n[5/5] Evaluating all component lag combinations...")

# Dictionary mapping lag to column name
pintro_cols = {
    0: "potential_intro_s",
    1: "pintro_s_lag1",
    2: "pintro_s_lag2",
    3: "pintro_s_lag3",
    4: "pintro_s_lag4"
}
r0_cols = {
    0: "R0_rel_use",
    1: "r0_lag1",
    2: "r0_lag2",
    3: "r0_lag3",
    4: "r0_lag4"
}

y_true_base = None
groups_base = None
all_combos = {}
best_combo = None
best_auc_pr = -1

for pi_lag in range(5):
    for r0_lag in range(5):
        pcol = pintro_cols[pi_lag]
        rcol = r0_cols[r0_lag]
        
        label = f"Pintro_lag{pi_lag}_R0rel_lag{r0_lag}"
        
        # Compute risk with lagged components
        risk_opt = (destinations[pcol] * 
                    destinations[rcol] * 
                    destinations["susceptibility_use"] * 
                    destinations["density_s"])
        
        # Merge with cases to obtain outcome variable
        df_temp = destinations[[pcol, rcol, "susceptibility_use", "density_s",
                                "DANE", "Year_month", "year", "population"]].copy()
        df_temp["risk_opt"] = risk_opt
        
        df_temp = df_temp.merge(cases_df[["DANE", "Year_month", "cases"]],
                                on=["DANE", "Year_month"], how="left")
        df_temp["cases"] = df_temp["cases"].fillna(0).astype(int)
        df_temp["incidence_per_10k"] = (df_temp["cases"] / df_temp["population"]) * 10000
        df_temp["has_outbreak"] = (df_temp["incidence_per_10k"] > 0).astype(int)
        
        # Filter NaN (first months without complete lags)
        df_valid = df_temp.dropna(subset=["risk_opt", "has_outbreak"]).copy()
        
        if len(df_valid) < 100:
            print(f"  {label}: only {len(df_valid)} valid records, skipping")
            continue
        
        y_true = df_valid["has_outbreak"].values
        y_score = df_valid["risk_opt"].values
        groups = (df_valid["DANE"].astype(str) + "_" + df_valid["year"].astype(str)).values
        
        # Save base for comparison
        if pi_lag == 0 and r0_lag == 0:
            y_true_base = y_true
            groups_base = groups
        
        metrics = compute_metrics(y_true, y_score)
        cv = cv_metrics(y_true, y_score, groups, n_splits=5)
        metrics.update(cv)
        
        all_combos[label] = {
            "lag_pintro": pi_lag,
            "lag_r0rel": r0_lag,
            "n_records": len(df_valid),
            **metrics
        }
        
        auc_pr = metrics["auc_pr"]
        marker = " <<< BEST" if auc_pr > best_auc_pr else ""
        if auc_pr > best_auc_pr:
            best_auc_pr = auc_pr
            best_combo = label
        
        print(f"  {label:25s}: AUC-ROC={metrics['auc_roc']:.4f}  AUC-PR={auc_pr:.4f}  Lift={metrics['auc_pr_lift']:.2f}x{marker}")

print(f"\n  {'='*50}")
print(f"  BEST COMBINATION: {best_combo} (AUC-PR = {best_auc_pr:.6f})")
print(f"  {'='*50}")

# ===========================================================================
# 7. COMPARE WITH EXISTING STRATEGIES
# ===========================================================================
print("\n" + "=" * 70)
print("COMPARISON WITH EXISTING STRATEGIES")
print("=" * 70)

# Load existing results
lags_results_path = os.path.join(OUTPUT_DIR, "lags_results.json")
existing_results = {}
if os.path.exists(lags_results_path):
    with open(lags_results_path) as f:
        existing_results = json.load(f)
    print(f"  Loaded {len(existing_results)} existing strategies from lags_results.json")

# Re-evaluate base (no lags) and S2-lag4 on same subset for fair comparison
print("\n  Re-evaluating on same subset for fair comparison...")

# Base (no lags)
df_base = destinations[["DANE", "Year_month", "year", "population", "risk_base"]].copy()
df_base = df_base.merge(cases_df[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
df_base["cases"] = df_base["cases"].fillna(0).astype(int)
df_base["incidence_per_10k"] = (df_base["cases"] / df_base["population"]) * 10000
df_base["has_outbreak"] = (df_base["incidence_per_10k"] > 0).astype(int)
df_base = df_base.dropna(subset=["risk_base"])

y_true_ref = df_base["has_outbreak"].values
groups_ref = (df_base["DANE"].astype(str) + "_" + df_base["year"].astype(str)).values

metrics_base = compute_metrics(y_true_ref, df_base["risk_base"].values)
cv_base = cv_metrics(y_true_ref, df_base["risk_base"].values, groups_ref)
metrics_base.update(cv_base)

# Evaluate S2-lag4 on the same dataset
destinations["risk_lag1_base"] = destinations.groupby("DANE")["risk_base"].shift(1)
destinations["risk_lag2_base"] = destinations.groupby("DANE")["risk_base"].shift(2)
destinations["risk_lag3_base"] = destinations.groupby("DANE")["risk_base"].shift(3)
destinations["risk_lag4_base"] = destinations.groupby("DANE")["risk_base"].shift(4)
destinations["risk_s2_lag4"] = destinations[["risk_base", "risk_lag1_base", "risk_lag2_base", "risk_lag3_base", "risk_lag4_base"]].max(axis=1)

df_s2 = destinations[["DANE", "Year_month", "year", "population", "risk_s2_lag4"]].copy()
df_s2 = df_s2.merge(cases_df[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
df_s2["cases"] = df_s2["cases"].fillna(0).astype(int)
df_s2["incidence_per_10k"] = (df_s2["cases"] / df_s2["population"]) * 10000
df_s2["has_outbreak"] = (df_s2["incidence_per_10k"] > 0).astype(int)
df_s2 = df_s2.dropna(subset=["risk_s2_lag4"])

metrics_s2 = compute_metrics(df_s2["has_outbreak"].values, df_s2["risk_s2_lag4"].values)
cv_s2 = cv_metrics(df_s2["has_outbreak"].values, df_s2["risk_s2_lag4"].values,
                   (df_s2["DANE"].astype(str) + "_" + df_s2["year"].astype(str)).values)
metrics_s2.update(cv_s2)

# ===========================================================================
# 8. FINAL COMPARISON TABLE
# ===========================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

comparison = []

# Base (re-evaluated)
comparison.append({
    "Strategy": "Base (no lags)",
    "Description": "Current risk, no lags (re-evaluated)",
    "AUC-ROC": metrics_base["auc_roc"],
    "AUC-PR": metrics_base["auc_pr"],
    "AUC-PR Lift": metrics_base["auc_pr_lift"],
    "Spearman rho": metrics_base["spearman_rho"],
    "Recall@10%": metrics_base["recall_top_10pct"],
    "CV AUC-PR": metrics_base["cv_auc_pr_mean"],
})

# S2-lag4 re-evaluated (same pipeline as component strategy)
comparison.append({
    "Strategy": "S2-lag4: Risk max",
    "Description": "Max of risk over lag0-lag4",
    "AUC-ROC": metrics_s2["auc_roc"],
    "AUC-PR": metrics_s2["auc_pr"],
    "AUC-PR Lift": metrics_s2["auc_pr_lift"],
    "Spearman rho": metrics_s2["spearman_rho"],
    "Recall@10%": metrics_s2["recall_top_10pct"],
    "CV AUC-PR": metrics_s2["cv_auc_pr_mean"],
})

# Best component-level combination
best_data = all_combos.get(best_combo, {})
if best_data:
    comparison.append({
        "Strategy": f"Component-opt: Pintro_lag{best_data['lag_pintro']} + R0rel_lag{best_data['lag_r0rel']}",
        "Description": f"Pintro lag {best_data['lag_pintro']}, R0_rel lag {best_data['lag_r0rel']}",
        "AUC-ROC": best_data["auc_roc"],
        "AUC-PR": best_data["auc_pr"],
        "AUC-PR Lift": best_data["auc_pr_lift"],
        "Spearman rho": best_data["spearman_rho"],
        "Recall@10%": best_data["recall_top_10pct"],
        "CV AUC-PR": best_data["cv_auc_pr_mean"],
    })

# Also add the original S2-max-lag4 from lags_results.json for reference (not re-evaluated)
if "S2: Max lag0-lag4" in existing_results:
    m = existing_results["S2: Max lag0-lag4"]
    comparison.append({
        "Strategy": "S2-lag4 (original report)",
        "Description": "From original lags_results.json report (slightly different dataset)",
        "AUC-ROC": m.get("auc_roc", 0),
        "AUC-PR": m.get("auc_pr", 0),
        "AUC-PR Lift": m.get("auc_pr_lift", 0),
        "Spearman rho": m.get("spearman_rho", 0),
        "Recall@10%": m.get("recall_top_10pct", 0),
        "CV AUC-PR": m.get("cv_auc_pr_mean", 0),
    })

# Sort by AUC-PR descending
comparison.sort(key=lambda x: x["AUC-PR"], reverse=True)

# Print table
print(f"\n{'Strategy':45s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'Lift':>6s} {'Spear.':>7s} {'Rec@10%':>8s} {'CV PR':>7s}")
print("-" * 100)
for c in comparison:
    desc = c["Description"]
    label = c["Strategy"]
    if desc:
        label = f"{label} ({desc})"
    print(f"{label:45s} {c['AUC-ROC']:8.4f} {c['AUC-PR']:8.5f} {c['AUC-PR Lift']:6.2f}x {c['Spearman rho']:7.4f} {c['Recall@10%']:8.4f} {c['CV AUC-PR']:7.5f}")

# ===========================================================================
# 9. SAVE RESULTS
# ===========================================================================
print("\n\nSaving results...")

# JSON with all combinations
with open(os.path.join(OUTPUT_DIR, "component_lag_results.json"), "w") as f:
    json.dump(all_combos, f, indent=2, default=str)

# JSON with complete ranking of all strategies
ranking_export = []
for c in comparison:
    ranking_export.append({
        "rank": comparison.index(c) + 1,
        "strategy": c["Strategy"],
        "description": c["Description"],
        "auc_roc": c["AUC-ROC"],
        "auc_pr": c["AUC-PR"],
        "auc_pr_lift": c["AUC-PR Lift"],
        "spearman_rho": c["Spearman rho"],
        "recall_top_10pct": c["Recall@10%"],
        "cv_auc_pr_mean": c["CV AUC-PR"],
    })
with open(os.path.join(OUTPUT_DIR, "component_lag_ranking.json"), "w") as f:
    json.dump(ranking_export, f, indent=2)

# TXT Report
lines = []
lines.append("=" * 80)
lines.append("COMPONENT-LEVEL LAG OPTIMIZATION RESULTS")
lines.append("Municipalities > 2000 m.a.s.l., Colombia 2007-2024")
lines.append("=" * 80)
lines.append("")
lines.append(f"Best combination: {best_combo}")
lines.append(f"  Pintro lag = {best_data['lag_pintro']}")
lines.append(f"  R0_rel lag = {best_data['lag_r0rel']}")
lines.append(f"  AUC-PR = {best_data['auc_pr']:.6f}")
lines.append(f"  AUC-ROC = {best_data['auc_roc']:.6f}")
lines.append("")

lines.append("-" * 80)
lines.append("ALL 25 COMBINATIONS (sorted by AUC-PR)")
lines.append("-" * 80)
lines.append("")
header = f"{'Lag Pintro':>12s} {'Lag R0_rel':>12s} {'AUC-ROC':>10s} {'AUC-PR':>10s} {'Lift':>8s} {'Spearman':>10s} {'Rec@10%':>10s}"
lines.append(header)
lines.append("-" * 72)

# Sort all combos by AUC-PR
sorted_combos = sorted(all_combos.items(), key=lambda x: x[1]["auc_pr"], reverse=True)
for label, m in sorted_combos:
    marker = " <<<" if label == best_combo else ""
    lines.append(f"{m['lag_pintro']:>12d} {m['lag_r0rel']:>12d} {m['auc_roc']:>10.4f} {m['auc_pr']:>10.5f} {m['auc_pr_lift']:>7.2f}x {m['spearman_rho']:>10.4f} {m['recall_top_10pct']:>10.4f}{marker}")

lines.append("")
lines.append("-" * 80)
lines.append("COMPARATIVE RANKING (all strategies)")
lines.append("-" * 80)
lines.append("")
for i, c in enumerate(comparison, 1):
    desc = c["Description"]
    label = c["Strategy"]
    if desc:
        label = f"{label} ({desc})"
    lines.append(f"{i:2d}. {label:50s} AUC-PR={c['AUC-PR']:.5f}  AUC-ROC={c['AUC-ROC']:.4f}  Lift={c['AUC-PR Lift']:.2f}x")

lines.append("")
lines.append("=" * 80)
lines.append("CONCLUSION")
lines.append("=" * 80)
lines.append("")

# Best strategy overall
best_overall = comparison[0]
lines.append(f"Best overall strategy: {best_overall['Strategy']}")
lines.append(f"  Description: {best_overall['Description']}")
lines.append(f"  AUC-PR: {best_overall['AUC-PR']:.5f}")
lines.append(f"  AUC-ROC: {best_overall['AUC-ROC']:.4f}")
lines.append(f"  AUC-PR Lift: {best_overall['AUC-PR Lift']:.2f}x")

report_path = os.path.join(OUTPUT_DIR, "component_lag_comparison.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  Report: {report_path}")

# ===========================================================================
# 10. HEATMAP OF THE 25 COMBINATIONS
# ===========================================================================
print("\nCreating heatmap...")

# Matrix 5x5 of AUC-PR
auc_pr_matrix = np.full((5, 5), np.nan)
auc_roc_matrix = np.full((5, 5), np.nan)

for label, m in all_combos.items():
    i = m["lag_pintro"]
    j = m["lag_r0rel"]
    auc_pr_matrix[i, j] = m["auc_pr"]
    auc_roc_matrix[i, j] = m["auc_roc"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap AUC-PR
ax = axes[0]
vmin_pr = np.nanmin(auc_pr_matrix)
vmax_pr = np.nanmax(auc_pr_matrix)
im = ax.imshow(auc_pr_matrix, cmap="RdYlGn", aspect="auto",
               vmin=vmin_pr, vmax=vmax_pr)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([f"R0_lag={i}" for i in range(5)])
ax.set_yticklabels([f"Pintro_lag={i}" for i in range(5)])
ax.set_title("AUC-PR by component lag", fontsize=13, fontweight="bold")

# Annotate
for i in range(5):
    for j in range(5):
        val = auc_pr_matrix[i, j]
        if not np.isnan(val):
            text_color = "white" if val > (vmin_pr + vmax_pr) / 2 else "black"
            ax.text(j, i, f"{val:.5f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

# Mark best
bi, bj = best_data["lag_pintro"], best_data["lag_r0rel"]
ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                            fill=False, edgecolor="blue", linewidth=3, linestyle="--"))

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Heatmap AUC-ROC
ax = axes[1]
vmin_roc = np.nanmin(auc_roc_matrix)
vmax_roc = np.nanmax(auc_roc_matrix)
im = ax.imshow(auc_roc_matrix, cmap="RdYlGn", aspect="auto",
               vmin=vmin_roc, vmax=vmax_roc)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([f"R0_lag={i}" for i in range(5)])
ax.set_yticklabels([f"Pintro_lag={i}" for i in range(5)])
ax.set_title("AUC-ROC by component lag", fontsize=13, fontweight="bold")

for i in range(5):
    for j in range(5):
        val = auc_roc_matrix[i, j]
        if not np.isnan(val):
            text_color = "white" if val > (vmin_roc + vmax_roc) / 2 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                            fill=False, edgecolor="blue", linewidth=3, linestyle="--"))

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "component_lag_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Heatmap: {heatmap_path}")

# ===========================================================================
# Final
# ===========================================================================
print(f"\n{'='*70}")
print("COMPLETED")
print(f"{'='*70}")
print(f"\nBest component lag combination: {best_combo}")
print(f"  Pintro lag = {best_data['lag_pintro']} month(s)")
print(f"  R0_rel lag = {best_data['lag_r0rel']} month(s)")
print(f"  AUC-PR = {best_data['auc_pr']:.5f}")
print(f"  AUC-ROC = {best_data['auc_roc']:.4f}")
print(f"\nResults saved to {OUTPUT_DIR}/")
print(f"  component_lag_results.json")
print(f"  component_lag_comparison.txt")
print(f"  component_lag_heatmap.png")
