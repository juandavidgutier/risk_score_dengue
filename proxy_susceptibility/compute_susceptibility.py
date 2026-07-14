"""
compute_susceptibility.py
=========================
Computes the population susceptibility proxy based on the methodology
from Figure S9 of Childs et al. (2025) "Climate warming is expanding dengue
burden in the Americas and Asia".

Methodology:
  1. For each municipality and month t, the average incidence in the
     window [t-18, t-7] (7 to 18 months prior) is calculated.
     This is the "immunity proxy": high past incidence -> more immunity.

  2. The susceptibility proxy is the inverse of the immunity proxy:
     - If past incidence is low -> high susceptibility (naive population)
     - If past incidence is high -> low susceptibility (immune population)

  3. Susceptibility is normalized to range [0, 1] where:
     - 1 = maximum susceptibility (no past cases)
     - 0 = minimum susceptibility (many past cases)

Output:
  - susceptibility_monthly.csv.gz   (long format)
  - susceptibility_monthly.rds       (for R)
  - susceptibility_matrix.rds        (matrix 1122 x 216 for R)
  - susceptibility_summary_stats.csv (monthly statistics)

Reference:
  Childs, M.L. et al. (2025). "Climate warming is expanding dengue burden
  in the Americas and Asia." medRxiv. Figure S9.
"""

import pandas as pd
import numpy as np
import os
import warnings
import gc

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "C:/"
CASES_FILE = os.path.join(BASE_DIR, "cases", "cases_dengue.csv")
POP_FILE = os.path.join(BASE_DIR, "radiation", "population_per_municipality.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "proxy_susceptibility")

# Window of 7-18 months before each month (inclusive)
LAG_MIN = 7   # start 7 months before
LAG_MAX = 18  # up to 18 months before

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("POPULATION SUSCEPTIBILITY PROXY")
print("Window: [t-{}, t-{}] months".format(LAG_MAX, LAG_MIN))
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")

# --- Monthly cases ---
print("  Reading cases...")
cases = pd.read_csv(CASES_FILE)
cases["Year_month"] = cases["Year_month"].astype(str)
print(f"  Cases: {len(cases):,} records, {cases['DANE'].nunique()} municipalities")
print(f"  Date range: {cases['Year_month'].min()} to {cases['Year_month'].max()}")

# --- Annual population ---
print("  Reading population...")
pop = pd.read_csv(POP_FILE)
pop_annual = pop.melt(
    id_vars=["DANE"],
    value_vars=[f"pop_{y}" for y in range(2007, 2025)],
    var_name="year",
    value_name="population"
)
pop_annual["year"] = pop_annual["year"].str.replace("pop_", "").astype(int)
pop_annual["DANE"] = pop_annual["DANE"].astype(int)
print(f"  Population: {len(pop_annual):,} records, {pop_annual['DANE'].nunique()} municipalities")

# ============================================================
# 2. PREPARE DATA
# ============================================================
print("\n[2/5] Preparing data...")

# Create all DANE x Year_month combinations
all_danes = sorted(cases["DANE"].unique())
all_year_months = sorted(cases["Year_month"].unique())

print(f"  Municipalities: {len(all_danes)}")
print(f"  Months: {len(all_year_months)}")

# Convert Year_month to numeric year and month
def parse_ym(ym_str):
    parts = ym_str.split("-")
    return int(parts[0]), int(parts[1])

# Create complete grid
print("  Creating complete grid...")
grid = pd.MultiIndex.from_product(
    [all_danes, all_year_months],
    names=["DANE", "Year_month"]
).to_frame(index=False)

# Merge with cases
grid = grid.merge(cases[["DANE", "Year_month", "cases"]], on=["DANE", "Year_month"], how="left")
grid["cases"] = grid["cases"].fillna(0).astype(int)

# Merge with annual population
grid["year"] = grid["Year_month"].apply(lambda x: parse_ym(x)[0])
grid = grid.merge(pop_annual, on=["DANE", "year"], how="left")

# Check for missing population values
n_missing_pop = grid["population"].isna().sum()
if n_missing_pop > 0:
    print(f"  [WARN] {n_missing_pop} records without population. Using median.")
    median_pop = grid.groupby("DANE")["population"].transform("median")
    grid["population"] = grid["population"].fillna(median_pop)

# Calculate incidence (cases per 100,000 inhabitants)
grid["incidence"] = (grid["cases"] / grid["population"]) * 100000

print(f"  Complete grid: {len(grid):,} records")
print(f"  Total cases: {grid['cases'].sum():,}")
print(f"  Mean incidence: {grid['incidence'].mean():.4f} per 100,000 pop")

# ============================================================
# 3. COMPUTE IMMUNITY PROXY (7-18 month window)
# ============================================================
print("\n[3/5] Computing immunity proxy...")

# Sort by DANE and date
grid = grid.sort_values(["DANE", "Year_month"]).reset_index(drop=True)

# For each municipality, calculate the moving average in window [t-18, t-7]
# Using lags of 7 to 18 months

immunity_proxy_list = []
idx_map = {}  # to map position in grid to (DANE, Year_month)

print("  Creating lag windows...")

# Strategy: for each month t, extract incidence at t-7, t-8, ..., t-18
# and average non-NaN values

n_munis = len(all_danes)
n_months = len(all_year_months)

# Create 2D array: municipalities x months
muni_to_idx = {m: i for i, m in enumerate(all_danes)}
month_to_idx = {m: j for j, m in enumerate(all_year_months)}

incidence_mat = np.full((n_munis, n_months), np.nan, dtype=np.float64)

# Fill the matrix
for _, row in grid.iterrows():
    i = muni_to_idx.get(row["DANE"])
    j = month_to_idx.get(row["Year_month"])
    if i is not None and j is not None:
        incidence_mat[i, j] = row["incidence"]

print(f"  Incidence matrix: {incidence_mat.shape}")
print(f"  NaN in matrix: {np.isnan(incidence_mat).sum():,}")

# Compute immunity proxy for each cell
print("  Computing averages in 7-18 month window...")
immunity_mat = np.full_like(incidence_mat, np.nan)

for j in range(n_months):
    if j % 24 == 0:
        print(f"    Month {j+1}/{n_months}")

    # Indices from window: from j-LAG_MAX to j-LAG_MIN
    start = j - LAG_MAX
    end = j - LAG_MIN + 1  # inclusive

    if start < 0:
        # Partial window: use only available months
        start = 0
        if end <= 0:
            # Not enough data in the window
            continue

    if start >= j:
        # Window is in the future (should not happen with LAG_MIN=7)
        continue

    # Window: [start, end) where end = j - LAG_MIN + 1
    window = incidence_mat[:, start:end]

    # Calculate mean ignoring NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        immunity_mat[:, j] = np.nanmean(window, axis=1)

print("  Immunity proxy computed.")

# ============================================================
# 4. CONVERT TO SUSCEPTIBILITY PROXY
# ============================================================
print("\n[4/5] Converting to susceptibility proxy...")

# --- Methodology from Figure S9 (Childs et al. 2025) ---
# The paper uses quantiles of the immunity proxy:
#   1) Q1 = exactly zero incidence (typically >25% of observations)
#   2) Q2, Q3, Q4 = terciles of non-zero values
#
# Susceptibility is defined as the inverse:
#   Q1 (0 past cases) -> susceptibility = 1.0 (maximally naive)
#   Q4 (many past cases) -> susceptibility = 0.0 (maximally immune)

# Immunity proxy statistics
imm_valid = immunity_mat[~np.isnan(immunity_mat)]
imm_mean = np.mean(imm_valid)
imm_median = np.median(imm_valid)
prop_zero = (imm_valid == 0).mean() * 100

print(f"\n  Immunity proxy statistics (mean incidence in 7-18m window):")
print(f"    Min:    {np.min(imm_valid):.6f}")
print(f"    Mean:   {imm_mean:.6f}")
print(f"    Median: {imm_median:.6f}")
print(f"    Max:    {np.max(imm_valid):.6f}")
print(f"    Proportion zero: {prop_zero:.1f}%")

# --- Step 1: Compute immunity proxy quantiles ---
# Q1 = exactly 0
# Q2, Q3, Q4 = terciles of values > 0

# Non-zero values
imm_nonzero = imm_valid[imm_valid > 0]

if len(imm_nonzero) > 0:
    # Terciles of non-zero values
    terciles = np.percentile(imm_nonzero, [33.33, 66.67])
    t1, t2 = terciles[0], terciles[1]

    print(f"\n  Immunity proxy quantiles (Figure S9):")
    print(f"    Q1 (zero):            0.000")
    print(f"    Q2 (low immunity):    0.000 - {t1:.4f}")
    print(f"    Q3 (medium immunity): {t1:.4f} - {t2:.4f}")
    print(f"    Q4 (high immunity):   > {t2:.4f}")
else:
    t1, t2 = 0, 0
    print("  [WARN] No non-zero values in immunity proxy")

# --- Step 2: Assign susceptibility by quantile ---
# Map: Q1 -> 1.0, Q2 -> 0.67, Q3 -> 0.33, Q4 -> 0.0

susceptibility_mat = np.full_like(immunity_mat, np.nan)
susceptibility_quantile = np.full_like(immunity_mat, np.nan, dtype=np.float64)

for i in range(n_munis):
    for j in range(n_months):
        imm_val = immunity_mat[i, j]
        if np.isnan(imm_val):
            continue

        # Assign quantile
        if imm_val == 0:
            sus_quantile = 1  # Q1: no past cases
            sus_value = 1.0
        elif imm_val <= t1:
            sus_quantile = 2  # Q2: low immunity
            sus_value = 2.0 / 3.0  # 0.667
        elif imm_val <= t2:
            sus_quantile = 3  # Q3: medium immunity
            sus_value = 1.0 / 3.0  # 0.333
        else:
            sus_quantile = 4  # Q4: high immunity
            sus_value = 0.0

        susceptibility_mat[i, j] = sus_value
        susceptibility_quantile[i, j] = sus_quantile

# For months without sufficient data (first 6-7 months), assign susceptibility = 1.0
# This is reasonable: no past case data, assume naive population
n_early = np.isnan(susceptibility_mat).sum()
susceptibility_mat[np.isnan(susceptibility_mat)] = 1.0
susceptibility_quantile[np.isnan(susceptibility_quantile)] = 1.0

if n_early > 0:
    print(f"\n  Early months without complete window: {n_early} assigned to Q1 (susceptibility=1.0)")

print(f"\n  Susceptibility proxy statistics (quantiles):")
sus_valid = susceptibility_mat[~np.isnan(susceptibility_mat)]
print(f"    Mean:   {np.mean(sus_valid):.4f}")
print(f"    Median: {np.median(sus_valid):.4f}")
print(f"    Min:    {np.min(sus_valid):.4f}")
print(f"    Max:    {np.max(sus_valid):.4f}")

# Susceptibility quantile distribution
print(f"\n  Susceptibility quantile distribution:")
q_counts = {1: 0, 2: 0, 3: 0, 4: 0}
q_valid = susceptibility_quantile[~np.isnan(susceptibility_quantile)]
for q in [1, 2, 3, 4]:
    pct = (q_valid == q).mean() * 100
    print(f"    Q{q} (sus={[1.0, 0.667, 0.333, 0.0][q-1]:.3f}): {pct:.1f}%")

# ============================================================
# 5. SAVE RESULTS
# ============================================================
print("\n[5/5] Saving results...")

# --- Long format: DANE, Year_month, susceptibility ---
records = []
for i, muni in enumerate(all_danes):
    for j, ym in enumerate(all_year_months):
        sus_val = susceptibility_mat[i, j]
        if not np.isnan(sus_val):
            records.append({
                "DANE": muni,
                "Year_month": ym,
                "susceptibility": float(sus_val)
            })

sus_df = pd.DataFrame(records)
sus_df = sus_df.sort_values(["DANE", "Year_month"]).reset_index(drop=True)

print(f"\n  Final DataFrame: {len(sus_df):,} records")
print(f"  Municipalities: {sus_df['DANE'].nunique()}")
print(f"  Months: {sus_df['Year_month'].nunique()}")

# Save compressed CSV
csv_gz = os.path.join(OUTPUT_DIR, "susceptibility_monthly.csv.gz")
sus_df.to_csv(csv_gz, index=False, compression="gzip")
print(f"  Saved: {csv_gz}")

# Save uncompressed CSV (for quick inspection)
csv_path = os.path.join(OUTPUT_DIR, "susceptibility_monthly.csv")
sus_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# Save RDS (for loading in R)
try:
    import pyreadr
    pyreadr.write_rds(os.path.join(OUTPUT_DIR, "susceptibility_monthly.rds"), sus_df)
    print(f"  Saved: susceptibility_monthly.rds")
except ImportError:
    print("  [WARN] pyreadr not available. Skipping RDS.")
except Exception as e:
    print(f"  [WARN] Error saving RDS: {e}")

# --- Matrix for R ---
# Convert to matrix: rows = municipalities, columns = months
sus_matrix = susceptibility_mat.copy()
sus_matrix_clean = np.where(np.isnan(sus_matrix), 0, sus_matrix)

# Save as CSV
mat_df = pd.DataFrame(
    sus_matrix_clean,
    index=[int(d) for d in all_danes],
    columns=all_year_months
)
mat_df.index.name = "DANE"
mat_df.to_csv(os.path.join(OUTPUT_DIR, "susceptibility_matrix.csv"))
print(f"  Saved: susceptibility_matrix.csv ({mat_df.shape})")

# Save as RDS
try:
    pyreadr.write_rds(os.path.join(OUTPUT_DIR, "susceptibility_matrix.rds"), mat_df)
    print(f"  Saved: susceptibility_matrix.rds")
except:
    pass

# --- Monthly statistics ---
monthly_stats = []
for j, ym in enumerate(all_year_months):
    vals = sus_valid = susceptibility_mat[:, j]
    vals = vals[~np.isnan(vals)]
    if len(vals) > 0:
        monthly_stats.append({
            "Year_month": ym,
            "mean": np.mean(vals),
            "median": np.median(vals),
            "std": np.std(vals),
            "min": np.min(vals),
            "max": np.max(vals),
            "p25": np.percentile(vals, 25),
            "p75": np.percentile(vals, 75),
        })

stats_df = pd.DataFrame(monthly_stats)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "susceptibility_summary_stats.csv"), index=False)
print(f"  Saved: susceptibility_summary_stats.csv")

# --- Comparison with R0_rel (if exists) ---
r0_file = os.path.join(BASE_DIR, "R0", "R0_rel_monthly.csv.gz")
if os.path.exists(r0_file):
    print("\n  Comparing with R0_rel...")
    r0 = pd.read_csv(r0_file, compression="gzip")
    if "Year_month" in r0.columns and "R0_rel" in r0.columns:
        # Merge
        merged = sus_df.merge(r0, on=["DANE", "Year_month"], how="inner")
        print(f"    Combined records: {len(merged):,}")
        corr = merged["susceptibility"].corr(merged["R0_rel"])
        print(f"    Correlation susceptibility vs R0_rel: {corr:.4f}")
        # Save combined file (optional)
        merged.to_csv(os.path.join(OUTPUT_DIR, "susceptibility_vs_R0rel.csv"), index=False)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\nFiles in {OUTPUT_DIR}/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    if size > 1_000_000:
        print(f"  {f:45s} {size/1e6:.1f} MB")
    elif size > 1_000:
        print(f"  {f:45s} {size/1e3:.1f} KB")
    else:
        print(f"  {f:45s} {size} B")
