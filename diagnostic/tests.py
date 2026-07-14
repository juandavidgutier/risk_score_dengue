"""
tests.py — Consistency Diagnostic: Estimated Risk vs Observed Incidence
====================================================================================
Scale: risk < 6e-5 (dimensionless relative index) vs incidence < 1.3/10,000 (absolute rate)
Metrics selected: scale-invariant, operationally relevant, tail diagnostics.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss
)
from sklearn.model_selection import GroupKFold
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.proportion import proportion_effectsize
    STATSMODELS_AVAILABLE = True
except Exception as e:
    print(f"  [WARN] statsmodels import failed: {e}")
    print(f"  [WARN] Poisson, NegBin, ZINB, QuantReg, and logit models will be skipped.")
    sm = None
    smf = None
    proportion_effectsize = None
    STATSMODELS_AVAILABLE = False
import json

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = "C:/"
RISK_PATH = os.path.join(BASE_DIR, "risk_score", "risk_2000m.csv.gz")
CASES_PATH = os.path.join(BASE_DIR, "cases", "cases_dengue.csv")
POP_PATH = os.path.join(BASE_DIR, "radiation", "population_per_municipality.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. DATA LOADING AND PREPARATION
# ──────────────────────────────────────────────────────────────────────
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

# Population annual (wide -> long)
pop_wide = pd.read_csv(POP_PATH)
pop_long = pop_wide.melt(
    id_vars=["DANE"],
    value_vars=[f"pop_{y}" for y in range(2007, 2025)],
    var_name="year_str", value_name="population"
)
pop_long["year"] = pop_long["year_str"].str.replace("pop_", "").astype(int)
print(f"Population: {len(pop_long):,} municipality-year records")

# Merge risk + cases
df = risk.merge(cases[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
df["cases"] = df["cases"].fillna(0).astype(int)

# Population ALREADY INCLUDED in the risk file (population column)
# Verify it exists
if "population" not in df.columns:
    # Fallback: merge with population_per_municipality.csv
    pop_wide = pd.read_csv(POP_PATH)
    pop_long = pop_wide.melt(
        id_vars=["DANE"],
        value_vars=[f"pop_{y}" for y in range(2007, 2025)],
        var_name="year_str", value_name="population"
    )
    pop_long["year"] = pop_long["year_str"].str.replace("pop_", "").astype(int)
    df["year"] = df["Year_month"].apply(lambda x: int(x.split("-")[0]))
    df = df.merge(pop_long[["DANE", "year", "population"]], on=["DANE", "year"], how="left")
    missing_pop = df["population"].isna().sum()
    if missing_pop > 0:
        med_pop = df.groupby("DANE")["population"].transform("median")
        df["population"] = df["population"].fillna(med_pop)
        print(f"  {missing_pop} NaN population -> municipal median")
else:
    print(f"  Population already present in risk data: {df['population'].notna().sum():,} records")
    # Ensure year column
    if "year" not in df.columns:
        df["year"] = df["Year_month"].apply(lambda x: int(x.split("-")[0]))

# ──────────────────────────────────────────────────────────────────────
# 2. CALCULATE INCIDENCE PER 10,000 INHABITANTS
# ──────────────────────────────────────────────────────────────────────
df["incidence_per_10k"] = (df["cases"] / df["population"]) * 10000
df["has_outbreak"] = (df["incidence_per_10k"] > 0).astype(int)

df = df.sort_values(["DANE", "Year_month"]).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────
# 2b. APPLY S2 LAG STRATEGY (MAX over lag0-lag4)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2b. APPLYING S2 LAG STRATEGY (max lag0-lag4)...")
print("=" * 70)

df["risk_lag1"] = df.groupby("DANE")["risk"].shift(1)
df["risk_lag2"] = df.groupby("DANE")["risk"].shift(2)
df["risk_lag3"] = df.groupby("DANE")["risk"].shift(3)
df["risk_lag4"] = df.groupby("DANE")["risk"].shift(4)

df["risk_s2_lag4"] = df[["risk", "risk_lag1", "risk_lag2", "risk_lag3", "risk_lag4"]].max(axis=1)

# For first 4 months per municipality where lags are not available, fall back to risk
df["risk_s2_lag4"] = df["risk_s2_lag4"].fillna(df["risk"])

# Replace original risk with lag-optimized version for all downstream analyses
df["risk"] = df["risk_s2_lag4"]

print(f"  Risk range (S2-lag4): min={df['risk'].min():.2e}, max={df['risk'].max():.2e}, mean={df['risk'].mean():.2e}")

print(f"\nBasic statistics (after S2-lag4):")
print(f"  Risk:        min={df['risk'].min():.2e}, max={df['risk'].max():.2e}, mean={df['risk'].mean():.2e}")
print(f"  Incidence:   min={df['incidence_per_10k'].min():.4f}, max={df['incidence_per_10k'].max():.4f}, mean={df['incidence_per_10k'].mean():.4f}")
print(f"  Cases > 0:   {df['has_outbreak'].sum():,} / {len(df):,} ({df['has_outbreak'].mean()*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# 3. METRIC FUNCTIONS AND INTERPRETATION
# ──────────────────────────────────────────────────────────────────────
results = {}

def add_result(name, value, interpretation, details=None):
    if value is not None and not isinstance(value, (str, dict, list)):
        val = float(value)
    else:
        val = value
    results[name] = {
        "value": val,
        "interpretation": interpretation,
        "details": details or {}
    }

# ──────────────────────────────────────────────────────────────────────
# 3.1 RANK CORRELATIONS (scale-invariant)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.1 RANK CORRELATIONS")
print("=" * 70)

rho, p_rho = spearmanr(df["risk"], df["incidence_per_10k"])
add_result("spearman_rho", rho,
    f"Monotonic rank correlation: rho={rho:.4f} (p={p_rho:.2e}). "
    f"{'Positive and significant' if rho > 0 and p_rho < 0.05 else 'Not significant'}. "
    f"Indicates whether the relative ordering of risk coincides with that of incidence.",
    {"p_value": float(p_rho), "n": len(df)})

tau, p_tau = kendalltau(df["risk"], df["incidence_per_10k"])
add_result("kendall_tau", tau,
    f"Pairwise concordance (Kendall tau): tau={tau:.4f} (p={p_tau:.2e}). "
    f"More robust than Spearman for large n; tau approx 2/pi * arcsin(rho) under normality.",
    {"p_value": float(p_tau), "n": len(df)})

# Outbreak months only (conditional)
df_pos = df[df["has_outbreak"] == 1]
if len(df_pos) > 10:
    rho_pos, p_pos = spearmanr(df_pos["risk"], df_pos["incidence_per_10k"])
    add_result("spearman_rho_conditional", rho_pos,
        f"Spearman only in months with incidence>0 (n={len(df_pos)}): rho={rho_pos:.4f}. "
        f"Does risk discriminate outbreak magnitude once it occurs?",
        {"p_value": float(p_pos), "n": len(df_pos)})

# ──────────────────────────────────────────────────────────────────────
# 3.2 BINARY DISCRIMINATION (AUC-ROC, AUC-PR) — MAIN METRIC
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.2 BINARY DISCRIMINATION (OUTBREAK vs NO OUTBREAK)")
print("=" * 70)

y_true = df["has_outbreak"].values
y_score = df["risk"].values

auc_roc = roc_auc_score(y_true, y_score)
auc_pr = average_precision_score(y_true, y_score)
# PR baseline = prevalence
baseline_pr = y_true.mean()

add_result("auc_roc", auc_roc,
    f"AUC-ROC = {auc_roc:.4f} (baseline 0.5). "
    f"{'Good' if auc_roc > 0.7 else 'Moderate' if auc_roc > 0.6 else 'Weak'}. "
    f"Measures overall outbreak/non-outbreak separation; insensitive to class imbalance.",
    {"baseline": 0.5, "prevalence": float(baseline_pr)})

add_result("auc_pr", auc_pr,
    f"AUC-PR = {auc_pr:.4f} (baseline prevalence = {baseline_pr:.4f}). "
    f"{'Good' if auc_pr > 2*baseline_pr else 'Moderate' if auc_pr > 1.5*baseline_pr else 'Weak'}. "
    f"Main metric for rare events: does the ranking concentrate outbreaks at high scores?",
    {"baseline_prevalence": float(baseline_pr), "lift": float(auc_pr / baseline_pr)})

# Precision-Recall curve for plotting
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)

# Operational top-k recall
for k_pct in [0.01, 0.05, 0.10, 0.20]:
    k = int(k_pct * len(df))
    top_k_idx = np.argsort(y_score)[-k:]
    recall_k = y_true[top_k_idx].sum() / y_true.sum()
    precision_k = y_true[top_k_idx].mean()
    add_result(f"recall_top_{int(k_pct*100)}pct", recall_k,
        f"Top {k_pct*100:.0f}% risk captures {recall_k*100:.1f}% of outbreaks "
        f"(precision={precision_k*100:.1f}%). "
        f"Lift vs random: {recall_k/k_pct:.1f}x",
        {"k": k, "precision": float(precision_k), "lift": float(recall_k/k_pct)})

# ──────────────────────────────────────────────────────────────────────
# 3.3 OPERATIONAL BINNING (RISK DECILES)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.3 OPERATIONAL BINNING (RISK DECILES)")
print("=" * 70)

n_bins = 10
df["risk_bin"] = pd.qcut(df["risk"], q=n_bins, labels=False, duplicates="drop")
bin_stats = df.groupby("risk_bin", observed=True).agg(
    mean_risk=("risk", "mean"),
    median_risk=("risk", "median"),
    mean_incidence=("incidence_per_10k", "mean"),
    median_incidence=("incidence_per_10k", "median"),
    n=("risk", "size"),
    n_outbreaks=("has_outbreak", "sum"),
    pct_outbreaks=("has_outbreak", "mean"),
    total_cases=("cases", "sum"),
    total_pop=("population", "sum")
).reset_index()

bin_stats["expected_incidence_per_bin"] = (bin_stats["total_cases"] / bin_stats["total_pop"]) * 10000
bin_stats["lift"] = bin_stats["pct_outbreaks"] / df["has_outbreak"].mean()

print(bin_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Cochran-Armitage trend test (approximated via ordinal logistic regression)
if STATSMODELS_AVAILABLE and smf is not None:
    try:
        model_ca = smf.logit("has_outbreak ~ risk_bin", data=df).fit(disp=0)
        p_trend = model_ca.pvalues["risk_bin"]
        add_result("cochran_armitage_trend", float(model_ca.params["risk_bin"]),
            f"Linear trend in log-odds of outbreak vs risk bin: "
            f"coef={model_ca.params['risk_bin']:.4f} (p={p_trend:.2e}). "
            f"{'Significant' if p_trend < 0.05 else 'Not significant'}. "
            f"Confirms monotonically increasing outbreak probability with risk.",
            {"p_value": float(p_trend), "coef": float(model_ca.params["risk_bin"])})
    except Exception as e:
        add_result("cochran_armitage_trend", None, f"Could not compute: {e}")
else:
    add_result("cochran_armitage_trend", None, "statsmodels not available")

# Spearman monotonicity across bins
rho_bins, p_bins = spearmanr(bin_stats["mean_risk"], bin_stats["expected_incidence_per_bin"])
add_result("spearman_bins", rho_bins,
    f"Correlation mean risk vs mean incidence per decile: rho={rho_bins:.4f} (p={p_bins:.2e}). "
    f"Operational calibration curve: does incidence increase with risk?",
    {"p_value": float(p_bins), "n_bins": len(bin_stats)})

# Save bin table
bin_stats.to_csv(os.path.join(OUTPUT_DIR, "bin_calibration.csv"), index=False)

# ──────────────────────────────────────────────────────────────────────
# 3.4 COUNT REGRESSION (CASES ~ RISK + LOG(POP) OFFSET)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.4 COUNT REGRESSION (CASES ~ RISK + LOG(POP) OFFSET)")
print("=" * 70)

# Prepare data for GLM
glm_df = df.copy()
glm_df["log_pop"] = np.log(glm_df["population"].clip(lower=1))

# Poisson
if STATSMODELS_AVAILABLE and smf is not None:
    try:
        pois = smf.glm("cases ~ risk", offset=glm_df["log_pop"],
                       family=sm.families.Poisson(), data=glm_df).fit()
        add_result("poisson_coef", float(pois.params["risk"]),
            f"Poisson: log(rate ratio) per unit risk = {pois.params['risk']:.4f} "
            f"(p={pois.pvalues['risk']:.2e}). "
            f"IRR = {np.exp(pois.params['risk']):.4f}. "
            f"Deviance={pois.deviance:.1f}, AIC={pois.aic:.1f}. "
            f"{'Overdispersion likely' if pois.deviance / pois.df_resid > 2 else 'OK'}",
            {"p_value": float(pois.pvalues["risk"]), "irr": float(np.exp(pois.params["risk"])),
             "deviance": float(pois.deviance), "aic": float(pois.aic),
             "dispersion": float(pois.deviance / pois.df_resid)})
    except Exception as e:
        add_result("poisson_coef", None, f"Error: {e}")
else:
    add_result("poisson_coef", None, "statsmodels not available")

# Negative Binomial (handles overdispersion)
if STATSMODELS_AVAILABLE and smf is not None:
    try:
        nb = smf.glm("cases ~ risk", offset=glm_df["log_pop"],
                     family=sm.families.NegativeBinomial(alpha=1.0), data=glm_df).fit()
        add_result("negbin_coef", float(nb.params["risk"]),
            f"Negative Binomial: log(rate ratio) = {nb.params['risk']:.4f} "
            f"(p={nb.pvalues['risk']:.2e}). IRR = {np.exp(nb.params['risk']):.4f}. "
            f"AIC={nb.aic:.1f}. Alpha={nb.params.get('alpha', 'N/A')}. "
            f"Better than Poisson if overdispersion.",
            {"p_value": float(nb.pvalues["risk"]), "irr": float(np.exp(nb.params["risk"])),
             "aic": float(nb.aic), "alpha": float(nb.params.get("alpha", 0))})
    except Exception as e:
        add_result("negbin_coef", None, f"Error: {e}")
else:
    add_result("negbin_coef", None, "statsmodels not available")

# Zero-Inflated Negative Binomial (ZINB) - separates structural zeros
if STATSMODELS_AVAILABLE and sm is not None:
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
        zinb = ZeroInflatedNegativeBinomialP(
            glm_df["cases"], 
            sm.add_constant(glm_df["risk"]),  # count part
            exog_infl=sm.add_constant(glm_df["risk"]),  # zero inflation part
            offset=glm_df["log_pop"]
        ).fit(disp=0, maxiter=200)
        add_result("zinb_coef_count", float(zinb.params["risk"]),
            f"ZINB (count part): coef={zinb.params['risk']:.4f} (p={zinb.pvalues['risk']:.2e}). "
            f"Zero inflation part: risk_coef={zinb.params.get('inflate_risk', 'N/A'):.4f}. "
            f"AIC={zinb.aic:.1f}. More realistic model for excess zeros.",
            {"p_value_count": float(zinb.pvalues["risk"]), "aic": float(zinb.aic)})
    except Exception as e:
        add_result("zinb_coef_count", None, f"Error/Not available: {e}")
else:
    add_result("zinb_coef_count", None, "statsmodels not available")

# ──────────────────────────────────────────────────────────────────────
# 3.5 TAIL ANALYSIS (QUANTILE REGRESSION / Q-Q)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.5 TAIL ANALYSIS (QUANTILE REGRESSION)")
print("=" * 70)

if STATSMODELS_AVAILABLE and sm is not None:
    try:
        from statsmodels.regression.quantile_regression import QuantReg
        for tau in [0.5, 0.75, 0.90, 0.95, 0.99]:
            qr = QuantReg(df["incidence_per_10k"], sm.add_constant(df["risk"])).fit(q=tau)
            add_result(f"quantreg_tau_{int(tau*100)}", float(qr.params["risk"]),
                f"Quantile regression tau={tau}: risk_coef={qr.params['risk']:.6f} "
                f"(p={qr.pvalues['risk']:.2e}). "
                f"Does risk predict high percentiles of incidence?",
                {"p_value": float(qr.pvalues["risk"]), "tau": tau})
    except Exception as e:
        add_result("quantreg", None, f"Error: {e}")
else:
    for tau in [0.5, 0.75, 0.90, 0.95, 0.99]:
        add_result(f"quantreg_tau_{int(tau*100)}", None, "statsmodels not available")
    add_result("quantreg", None, "statsmodels not available")

# ──────────────────────────────────────────────────────────────────────
# 3.6 SPATIO-TEMPORAL VALIDATION (GROUP K-FOLD)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.6 SPATIO-TEMPORAL VALIDATION (GROUP K-FOLD)")
print("=" * 70)

# Create groups: (municipality, year) to block space-time
df["group_st"] = df["DANE"].astype(str) + "_" + df["year"].astype(str)
groups = df["group_st"].values

# Use GroupKFold with 5 folds
gkf = GroupKFold(n_splits=5)
cv_auc_roc = []
cv_auc_pr = []
cv_recall_10 = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(df, df["has_outbreak"], groups)):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    
    # Train: we only use the risk ranking (no model to train)
    # Test: evaluate discrimination
    y_test = test["has_outbreak"].values
    s_test = test["risk"].values
    
    if y_test.sum() > 0 and (1 - y_test).sum() > 0:
        cv_auc_roc.append(roc_auc_score(y_test, s_test))
        cv_auc_pr.append(average_precision_score(y_test, s_test))
        
        # Recall@10%
        k = int(0.10 * len(test))
        top_k = np.argsort(s_test)[-k:]
        rec = y_test[top_k].sum() / y_test.sum()
        cv_recall_10.append(rec)

add_result("cv_auc_roc_mean", np.mean(cv_auc_roc),
    f"Spatio-Temporal CV (5 folds): AUC-ROC = {np.mean(cv_auc_roc):.4f} +/- {np.std(cv_auc_roc):.4f}. "
    f"Generalization to unseen municipalities/years.",
    {"std": float(np.std(cv_auc_roc)), "folds": cv_auc_roc})

add_result("cv_auc_pr_mean", np.mean(cv_auc_pr),
    f"Spatio-Temporal CV: AUC-PR = {np.mean(cv_auc_pr):.4f} +/- {np.std(cv_auc_pr):.4f} "
    f"(baseline prevalence={baseline_pr:.4f}).",
    {"std": float(np.std(cv_auc_pr)), "folds": cv_auc_pr, "baseline": float(baseline_pr)})

add_result("cv_recall_10pct_mean", np.mean(cv_recall_10),
    f"Spatio-Temporal CV: Recall@10% = {np.mean(cv_recall_10):.4f} +/- {np.std(cv_recall_10):.4f}. "
    f"Lift = {np.mean(cv_recall_10)/0.10:.1f}x.",
    {"std": float(np.std(cv_recall_10)), "folds": cv_recall_10})

# ──────────────────────────────────────────────────────────────────────
# 3.7 BRIER SCORE (ONLY IF WE RESCALE RISK TO [0,1])
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3.7 BRIER SCORE (RISK MIN-MAX RESCALED TO [0,1])")
print("=" * 70)

risk_scaled = (df["risk"] - df["risk"].min()) / (df["risk"].max() - df["risk"].min())
brier = brier_score_loss(y_true, risk_scaled)
brier_baseline = y_true.mean() * (1 - y_true.mean())  # Brier of constant predictor = prevalence
bss = 1 - brier / brier_baseline  # Brier Skill Score

add_result("brier_score", brier,
    f"Brier Score (risk min-max [0,1]): {brier:.6f} "
    f"(constant baseline={brier_baseline:.6f}). "
    f"BSS = {bss:.4f} ({'improves' if bss > 0 else 'worsens'} vs baseline). "
    f"Note: original risk is not a calibrated probability; min-max is arbitrary.",
    {"brier_baseline": float(brier_baseline), "bss": float(bss)})

# ──────────────────────────────────────────────────────────────────────
# 4. GENERATE KEY FIGURES
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. GENERATING FIGURES...")
print("=" * 70)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})

# Color palette
colors = {
    "primary": "#2c7bb6",
    "secondary": "#d7191c",
    "accent": "#fdae61",
    "neutral": "#7570b3",
    "gray": "#999999"
}

# ─── FIGURE 1: Precision-Recall Curve ───
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, color=colors["primary"], linewidth=2, label=f"Model (AUC-PR = {auc_pr:.3f})")
ax.axhline(y=baseline_pr, color=colors["gray"], linestyle="--", linewidth=1.5, label=f"Baseline (prevalence = {baseline_pr:.3f})")
ax.fill_between(recall, precision, baseline_pr, alpha=0.15, color=colors["primary"])
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision-Recall Curve\n(Outbreak vs Non-outbreak Discrimination)")
ax.set_xlim([0, 1])
ax.set_ylim([0, max(1.05 * max(precision), 1.05 * baseline_pr)])
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"))
plt.close()

# ─── FIGURE 2: ROC Curve ───
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=colors["primary"], linewidth=2, label=f"Model (AUC-ROC = {auc_roc:.3f})")
ax.plot([0, 1], [0, 1], color=colors["gray"], linestyle="--", linewidth=1.5, label="Random (AUC = 0.5)")
ax.fill_between(fpr, tpr, fpr, alpha=0.15, color=colors["primary"])
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curve")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

# ─── FIGURE 3: Binning Calibration (ScoreCard) ───
fig, ax = plt.subplots(figsize=(8, 5))
x = bin_stats["mean_risk"]
y = bin_stats["expected_incidence_per_bin"]
ax.scatter(x, y, s=80, color=colors["primary"], zorder=3, label="Observed deciles")
# Trend line
z = np.polyfit(x, y, 1)
ax.plot(x, np.polyval(z, x), color=colors["secondary"], linestyle="--", linewidth=1.5, 
        label=f"Linear trend (rho={rho_bins:.3f})")
# Scaled identity line (reference)
ax.plot([x.min(), x.max()], [y.min(), y.max()], color=colors["gray"], linestyle=":", linewidth=1, alpha=0.5)

# Annotate bins
for i, row in bin_stats.iterrows():
    ax.annotate(f"D{int(row['risk_bin'])+1}", (row["mean_risk"], row["expected_incidence_per_bin"]),
                xytext=(3, 3), textcoords="offset points", fontsize=8, color=colors["gray"])

ax.set_xlabel("Mean risk per decile (relative index)")
ax.set_ylabel("Observed incidence per 10,000 pop")
ax.set_title("Operational Calibration: Risk vs Incidence by Decile\n(Scorecard-style Calibration Curve)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bin_calibration.png"))
plt.close()

# ─── FIGURE 4: Scatter Risk vs Incidence (log-log) ───
fig, ax = plt.subplots(figsize=(6, 5))
# Subsample for visualization
n_plot = min(5000, len(df))
df_plot = df.sample(n=n_plot, random_state=42)
sc = ax.scatter(df_plot["risk"], df_plot["incidence_per_10k"], 
                c=df_plot["has_outbreak"], cmap="RdYlBu_r", 
                s=8, alpha=0.5, edgecolors="none")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Estimated risk (log scale)")
ax.set_ylabel("Incidence per 10,000 pop (log scale)")
ax.set_title(f"Risk vs Incidence (sample n={n_plot:,})\nBlue=no outbreak, Red=outbreak")
plt.colorbar(sc, ax=ax, label="Outbreak (0/1)")
ax.grid(True, alpha=0.3, linestyle=":", which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_loglog.png"))
plt.close()

# ─── FIGURE 5: Q-Q plot tail (risk vs incidence in outbreaks) ───
fig, ax = plt.subplots(figsize=(6, 5))
df_pos = df[df["has_outbreak"] == 1]
if len(df_pos) > 10:
    # Empirical quantiles
    n_pos = len(df_pos)
    q_risk = df_pos["risk"].rank(method="average") / (n_pos + 1)
    q_inc = df_pos["incidence_per_10k"].rank(method="average") / (n_pos + 1)
    ax.scatter(q_risk, q_inc, s=10, alpha=0.6, color=colors["primary"])
    ax.plot([0, 1], [0, 1], color=colors["gray"], linestyle="--", linewidth=1.5, label="Identity")
    ax.set_xlabel("Empirical quantile of Risk (conditional on outbreak)")
    ax.set_ylabel("Empirical quantile of Incidence (conditional on outbreak)")
    ax.set_title("Tail Q-Q Plot\n(Does the risk tail map to the incidence tail?)")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "qq_tail.png"))
plt.close()

# ─── FIGURE 6: Top-k Recall / Lift Curve ───
fig, ax = plt.subplots(figsize=(7, 5))
k_pcts = np.linspace(0.01, 0.5, 50)
recalls = []
lifts = []
for kp in k_pcts:
    k = int(kp * len(df))
    top_k_idx = np.argsort(y_score)[-k:]
    rec = y_true[top_k].sum() / y_true.sum()
    recalls.append(rec)
    lifts.append(rec / kp if kp > 0 else 0)

ax.plot(k_pcts * 100, recalls, color=colors["primary"], linewidth=2, label="Recall (fraction of outbreaks captured)")
ax.plot(k_pcts * 100, k_pcts, color=colors["gray"], linestyle="--", linewidth=1.5, label="Random (recall = % selected)")
ax.fill_between(k_pcts * 100, recalls, k_pcts, alpha=0.15, color=colors["primary"])
ax.set_xlabel("Percentage of municipality-months selected by highest risk (%)")
ax.set_ylabel("Recall (fraction of total outbreaks captured)")
ax.set_title("Lift / Recall@k Curve\n(Operational: how much by monitoring the top X%?)")
ax.set_xlim([0, 50])
ax.set_ylim([0, 1.05])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle=":")
# Annotate key points
for kp in [0.01, 0.05, 0.10, 0.20]:
    k = int(kp * len(df))
    top_k_idx = np.argsort(y_score)[-k:]
    rec = y_true[top_k].sum() / y_true.sum()
    ax.annotate(f"{rec*100:.0f}%", (kp*100, rec), xytext=(5, 5), textcoords="offset points", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lift_curve.png"))
plt.close()

# ─── FIGURE 7: CV Results Boxplot ───
fig, ax = plt.subplots(figsize=(8, 5))
cv_data = [cv_auc_roc, cv_auc_pr, cv_recall_10]
cv_labels = [f"AUC-ROC\n(mean={np.mean(cv_auc_roc):.3f})",
             f"AUC-PR\n(mean={np.mean(cv_auc_pr):.3f})",
             f"Recall@10%\n(mean={np.mean(cv_recall_10):.3f})"]
bp = ax.boxplot(cv_data, tick_labels=cv_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], [colors["primary"], colors["secondary"], colors["accent"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for median in bp["medians"]:
    median.set_color("white")
    median.set_linewidth(2)
ax.axhline(y=baseline_pr, color=colors["gray"], linestyle=":", linewidth=1.5, label=f"Baseline PR ({baseline_pr:.3f})")
ax.axhline(y=0.5, color=colors["gray"], linestyle=":", linewidth=1.5)
ax.set_ylabel("Metric")
ax.set_title("Spatio-Temporal Validation (5 folds, blocked by municipality-year)\nGeneralization to unseen municipalities/years")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle=":", axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cv_boxplot.png"))
plt.close()

# ─── FIGURE 8: Risk distribution by outbreak status ───
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df[df["has_outbreak"]==0]["risk"], bins=50, alpha=0.5, label="No outbreak (0)", 
        color=colors["primary"], density=True, edgecolor="none")
ax.hist(df[df["has_outbreak"]==1]["risk"], bins=50, alpha=0.7, label="Outbreak (1)", 
        color=colors["secondary"], density=True, edgecolor="none")
ax.set_xlabel("Estimated risk")
ax.set_ylabel("Density")
ax.set_title("Risk Distribution: Outbreak vs No Outbreak\n(Separation of the two populations)")
ax.legend()
ax.grid(True, alpha=0.3, linestyle=":", axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "risk_distribution.png"))
plt.close()

# ─── FIGURE 9: Monthly heatmap (years vs months) - Recall@10% ───
df["month"] = df["Year_month"].apply(lambda x: int(x.split("-")[1]))
monthly_recall = df.groupby(["year", "month"]).apply(
    lambda g: g.nlargest(max(1, int(0.1*len(g))), "risk")["has_outbreak"].mean() if g["has_outbreak"].sum() > 0 else np.nan
).reset_index(name="recall_10pct")
monthly_recall["year_month"] = monthly_recall["year"] + (monthly_recall["month"] - 1) / 12

fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(monthly_recall["year_month"], monthly_recall["month"], 
                c=monthly_recall["recall_10pct"], cmap="RdYlGn", s=80, vmin=0, vmax=1, edgecolors="white", linewidth=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Month")
ax.set_title("Recall@10% by Month-Year\n(In which months does the top 10% risk capture more outbreaks?)")
ax.set_yticks(range(1, 13))
ax.set_yticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
plt.colorbar(sc, ax=ax, label="Recall@10%")
ax.grid(True, alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_recall_heatmap.png"))
plt.close()

# ─── FIGURE 10: Quantile Regression Coefficients ───
fig, ax = plt.subplots(figsize=(6, 5))
taus = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
qr_coefs = []
qr_pvals = []
for tau in taus:
    key = f"quantreg_tau_{int(tau*100)}"
    if key in results and results[key]["value"] is not None:
        qr_coefs.append(results[key]["value"])
        qr_pvals.append(results[key]["details"].get("p_value", 1))
    else:
        qr_coefs.append(np.nan)
        qr_pvals.append(np.nan)

ax.plot(taus, qr_coefs, "o-", color=colors["primary"], linewidth=2, markersize=8)
# Mark significant ones
sig = np.array(qr_pvals) < 0.05
ax.scatter(np.array(taus)[sig], np.array(qr_coefs)[sig], s=120, color=colors["secondary"], zorder=5, label="Significant (p<0.05)")
ax.axhline(y=0, color=colors["gray"], linestyle="--", linewidth=1)
ax.set_xlabel("Quantile (tau)")
ax.set_ylabel("Risk coefficient")
ax.set_title("Quantile Regression: Effect of risk on different incidence quantiles\n(Does it predict the tail better than the median?)")
ax.legend()
ax.grid(True, alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "quantreg_coefs.png"))
plt.close()

# ──────────────────────────────────────────────────────────────────────
# 5. SAVE RESULTS JSON + TEXT REPORT
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. SAVING RESULTS...")
print("=" * 70)

# JSON
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

# Readable text report
report_path = os.path.join(OUTPUT_DIR, "REPORT.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("CONSISTENCY DIAGNOSTIC: ESTIMATED RISK vs OBSERVED INCIDENCE\n")
    f.write("Municipalities > 2000 m.a.s.l., Colombia 2007-2024\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SCALES:\n")
    f.write(f"  Estimated risk (relative index):  min={df['risk'].min():.2e}, max={df['risk'].max():.2e}, mean={df['risk'].mean():.2e}\n")
    f.write(f"  Observed incidence (cases/10k): min={df['incidence_per_10k'].min():.4f}, max={df['incidence_per_10k'].max():.4f}, mean={df['incidence_per_10k'].mean():.4f}\n")
    f.write(f"  Total records: {len(df):,} | Municipalities: {df['DANE'].nunique()} | Months: {df['Year_month'].nunique()}\n")
    f.write(f"  Outbreak prevalence (incidence>0): {df['has_outbreak'].mean()*100:.1f}%\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 80 + "\n")
    
    # Global interpretation
    auc_pr_val = results.get("auc_pr", {}).get("value", 0)
    lift_pr = results.get("auc_pr", {}).get("details", {}).get("lift", 0)
    recall_10 = results.get("recall_top_10pct", {}).get("value", 0)
    cv_pr = results.get("cv_auc_pr_mean", {}).get("value", 0)
    
    f.write(f"""
The estimated risk (relative index < 6e-5) is NOT a calibrated probability.
The observed incidence is an absolute rate (< 1.3/10,000).

KEY METRICS:
  * AUC-PR = {auc_pr_val:.4f} (baseline prevalence = {baseline_pr:.4f}) -> Lift = {lift_pr:.1f}x
  * Recall@10% = {recall_10*100:.1f}% of outbreaks captured by monitoring top 10% risk -> Lift = {recall_10/0.10:.1f}x
  * Spatio-Temporal CV AUC-PR = {cv_pr:.4f} +/- {results.get('cv_auc_pr_mean',{}).get('details',{}).get('std',0):.4f}

INTERPRETATION:
""")
    
    if auc_pr_val > 2 * baseline_pr:
        f.write("  YES Risk ranking DISCRIMINATES well between outbreaks and non-outbreaks (AUC-PR > 2x baseline).\n")
    elif auc_pr_val > 1.5 * baseline_pr:
        f.write("  ~ Risk ranking has MODERATE discrimination (AUC-PR 1.5-2x baseline).\n")
    else:
        f.write("  X Risk ranking has WEAK discrimination (AUC-PR ~= baseline).\n")
    
    if recall_10 > 0.3:
        f.write(f"  YES Monitoring the top 10% risk captures {recall_10*100:.0f}% of outbreaks (operationally useful).\n")
    elif recall_10 > 0.15:
        f.write(f"  ~ Monitoring the top 10% captures {recall_10*100:.0f}% of outbreaks (moderate).\n")
    else:
        f.write(f"  X Monitoring the top 10% captures only {recall_10*100:.0f}% of outbreaks (limited utility).\n")
    
    if cv_pr > 1.5 * baseline_pr:
        f.write("  YES Discrimination GENERALIZES to unseen municipalities/years (CV > 1.5x baseline).\n")
    else:
        f.write("  ! Discrimination does NOT generalize well (CV ~= baseline). Possible spatio-temporal overfitting.\n")
    
    f.write("\n" + "-" * 80 + "\n")
    f.write("METRIC DETAILS\n")
    f.write("-" * 80 + "\n\n")
    
    for name, res in results.items():
        if res["value"] is not None:
            f.write(f"{name}:\n")
            f.write(f"  Value: {res['value']}\n")
            f.write(f"  Interpretation: {res['interpretation']}\n")
            if res["details"]:
                for k, v in res["details"].items():
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("  - metrics.json")
print("  - REPORT.txt")
print("  - bin_calibration.csv")
print("  - Figures: pr_curve.png, roc_curve.png, bin_calibration.png, scatter_loglog.png,")
print("              qq_tail.png, lift_curve.png, cv_boxplot.png, risk_distribution.png,")
print("              monthly_recall_heatmap.png, quantreg_coefs.png")

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)
