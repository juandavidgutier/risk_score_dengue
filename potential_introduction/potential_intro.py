#!/usr/bin/env python
"""
compute_potential_intro.py
=====================
Compute monthly introduction risk (riesgo de introducción) for dengue in Colombia,
following the methodology from the YF study (OJO_fiebre_amarilla.pdf).

The core formula (from risk_calc_functions.R in the YF study):
    potential_intro = inci %*% mov

Where:
    inci : vector of cases (number of infected individuals) in source municipalities
    mov  : radiation probability matrix M_ij = probability of moving from i to j

For each month t:
    R_intro(d, t) = Σ_{f} C(f, t) x M(f -> d)

This gives the expected number of infected individuals arriving at each destination
municipality d from all source municipalities f in month t.

Inputs:
  - ../radiation/radiation_matrix.npy  : 1122 x 1122 radiation matrix
  - ../cases/cases_dengue.csv          : monthly dengue cases per municipality
  - ../radiation/municipality_data.csv : DANE code mapping for matrix rows/cols

Outputs (saved to risk_introduction/):
  - potential_intro_monthly.csv.gz          : compressed CSV: DANE_dest, Year_month, potential_intro
  - potential_intro_matrix.rds              : matrix TxN (time x municipality) for R
  - potential_intro_monthly.rds             : long-format data frame for R
  - monthly_cases_vector.csv.gz        : cases vectors per month (for verification)
  - summary_stats.csv                  : basic summary statistics

Reference:
  Simini et al. (2012) Nature 484, 96-100
  make_risk_df() / main_risk_function() in YF_WHO_risk_reports/shared/risk_calc_functions.R
"""

import numpy as np
import pandas as pd
import os, gzip, csv
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAD_DIR    = os.path.join(BASE_DIR, "radiation")
CASES_FILE = os.path.join(BASE_DIR, "cases", "cases_dengue.csv")
MUNI_FILE  = os.path.join(RAD_DIR, "municipality_data.csv")
RAD_FILE   = os.path.join(RAD_DIR, "radiation_matrix.npy")
OUT_DIR    = os.path.join(BASE_DIR, "potential_introduction")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[{datetime.now().isoformat()}] Loading radiation matrix...")
M = np.load(RAD_FILE)
n = M.shape[0]
print(f"  Radiation matrix: {M.shape[0]} x {M.shape[1]}")

print(f"[{datetime.now().isoformat()}] Loading municipality data...")
muni_df = pd.read_csv(MUNI_FILE)
dane_codes = muni_df["DANE"].astype(int).values
print(f"  DANE codes from shapefile: {len(dane_codes)}")

# Create mapping from DANE code -> matrix index
dane_to_idx = {dane: i for i, dane in enumerate(dane_codes)}
idx_to_dane = {i: dane for i, dane in enumerate(dane_codes)}

print(f"[{datetime.now().isoformat()}] Loading cases data...")
cases_df = pd.read_csv(CASES_FILE)
cases_df["DANE"] = cases_df["DANE"].astype(int)
print(f"  Cases rows: {len(cases_df)}")
print(f"  Unique DANE codes in cases: {cases_df['DANE'].nunique()}")
print(f"  Total cases reported: {cases_df['cases'].sum()}")
print(f"  Date range: {cases_df['Year_month'].min()} to {cases_df['Year_month'].max()}")
print(f"  Unique months: {cases_df['Year_month'].nunique()}")

# ============================================================
# STEP 1: Pivot cases into a matrix [municipalities x months]
# ============================================================
print(f"\n[{datetime.now().isoformat()}] Pivoting cases to municipalityxmonth matrix...")

# Get all unique months sorted chronologically
# Note: Year_month is like "2007-1", "2007-10", etc. 
# We need to sort them properly (by year, then month)
months = sorted(cases_df["Year_month"].unique(),
                key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1])))
n_months = len(months)
print(f"  Months: {n_months} ({months[0]} -> {months[-1]})")

# Create pivot table: each row is a municipality, each column is a month
# Using pandas pivot for efficiency
pivot = cases_df.pivot_table(
    index="DANE", 
    columns="Year_month", 
    values="cases",
    fill_value=0,
    aggfunc="sum"
)

# Ensure all months are present (even if some have no data)
pivot = pivot.reindex(columns=months, fill_value=0)

# Reorder columns to chronological order
pivot = pivot[months]

# Now align with the radiation matrix (all 1122 municipalities)
# Initialize a full matrix of zeros
cases_matrix = np.zeros((n, n_months), dtype=np.float64)

# Map DANE codes to matrix indices
for dane_val in pivot.index:
    if dane_val in dane_to_idx:
        idx = dane_to_idx[dane_val]
        cases_matrix[idx, :] = pivot.loc[dane_val].values

print(f"  Cases matrix shape: {cases_matrix.shape} (municipalities x months)")

# Verify totals
total_from_cases = cases_df["cases"].sum()
total_in_matrix = cases_matrix.sum()
print(f"  Total cases in raw data: {total_from_cases}")
print(f"  Total cases in aligned matrix: {total_in_matrix:.0f}")
assert abs(total_in_matrix - total_from_cases) < 1, "Case total mismatch!"

# ============================================================
# STEP 2: Compute monthly potential introduction
# ============================================================
print(f"\n[{datetime.now().isoformat()}] Computing monthly introduction risk...")
print(f"  Formula: R_intro = cases_vector %*% M")
print(f"    where cases_vector is (1 x {n}) and M is ({n} x {n})")
print(f"    yielding a (1 x {n}) risk vector per month")

# potential_intro[t, d] = Σ_f cases[f, t] x M[f, d]
# For each month t, compute: potential_intro_t[d] = Σ_f cases[f,t] * M[f,d]
#
# Vectorized: transposed multiplication
# cases_matrix is (n_munis x n_months)
# M is (n_munis x n_munis)
# potential_intro = M.T @ cases_matrix  (or equivalently (cases_matrix.T @ M).T)
#
# Actually: potential_intro[t, d] = Σ_f cases[f, t] * M[f, d]
# So: potential_intro = cases_matrix.T @ M
# Where cases_matrix.T is (n_months x n_munis)
# Result: (n_months x n_munis)

potential_intro = cases_matrix.T @ M  # shape: (n_months, n_munis)

print(f"  Introduction risk matrix shape: {potential_intro.shape[0]} x {potential_intro.shape[1]}")

# Check range
print(f"  Min intro risk: {potential_intro.min():.6f}")
print(f"  Max intro risk: {potential_intro.max():.6f}")
print(f"  Mean intro risk: {potential_intro.mean():.6f}")
print(f"  Monthly total intro risk (cases 'redistributed'): {potential_intro.sum(axis=1).mean():.2f}")

# ============================================================
# STEP 3: Save results
# ============================================================
print(f"\n[{datetime.now().isoformat()}] Saving results...")

# --- 3a: Long-format CSV (one row per municipality-month) ---
print("  Saving potential_intro_monthly.csv.gz (long format)...")
out_csv = os.path.join(OUT_DIR, "potential_intro_monthly.csv.gz")
with gzip.open(out_csv, "wt", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DANE", "Year_month", "potential_intro", "altitude_min_msnm"])
    
    for mi, month in enumerate(months):
        for muni_idx in range(n):
            dane = int(idx_to_dane[muni_idx])
            risk_val = potential_intro[mi, muni_idx]
            if risk_val > 1e-12:  # skip numerical zeros to save space
                writer.writerow([
                    dane,
                    month,
                    f"{risk_val:.8f}",
                    int(muni_df.iloc[muni_idx]["min_msnm"])
                ])

# --- 3b: Wide matrix format as CSV (months x municipalities) ---
print("  Saving potential_intro_matrix.csv (wide format, months x DANE)...")
intro_df = pd.DataFrame(
    potential_intro,
    index=months,
    columns=[f"DANE_{int(d)}" for d in dane_codes]
)
intro_df.index.name = "Year_month"
intro_df.to_csv(os.path.join(OUT_DIR, "potential_intro_matrix.csv"))
print(f"    Shape: {intro_df.shape}")

# --- 3c: Save as .rds for R ---
print("  Saving potential_intro_monthly.rds...")
try:
    import pyreadr
    
    # Long format for R
    records = []
    for mi, month in enumerate(months):
        for muni_idx in range(n):
            dane = int(idx_to_dane[muni_idx])
            risk_val = potential_intro[mi, muni_idx]
            records.append({
                "DANE": dane,
                "Year_month": month,
                "potential_intro": risk_val,
                "min_msnm": int(muni_df.iloc[muni_idx]["min_msnm"]),
                "name": muni_df.iloc[muni_idx]["name"]
            })
    
    long_df = pd.DataFrame(records)
    
    # Also save a summary per month (sum across destinations)
    monthly_summary = long_df.groupby("Year_month")["potential_intro"].agg(["sum", "mean", "max"]).reset_index()
    monthly_summary.columns = ["Year_month", "total_potential_intro", "mean_potential_intro", "max_potential_intro"]
    
    # Add cases total per month for comparison
    monthly_cases = cases_df.groupby("Year_month")["cases"].sum().reset_index()
    monthly_cases.columns = ["Year_month", "total_cases"]
    monthly_summary = monthly_summary.merge(monthly_cases, on="Year_month", how="left")
    
    pyreadr.write_rds(
        os.path.join(OUT_DIR, "potential_intro_monthly.rds"), 
        long_df
    )
    pyreadr.write_rds(
        os.path.join(OUT_DIR, "potential_intro_matrix.rds"),
        intro_df
    )
    pyreadr.write_rds(
        os.path.join(OUT_DIR, "summary_monthly.rds"),
        monthly_summary
    )
    print("    Saved: potential_intro_monthly.rds, potential_intro_matrix.rds, summary_monthly.rds")
except ImportError:
    print("    pyreadr not available. Install with: pip install pyreadr")
    print("    Results saved as CSV instead.")

# --- 3d: Summary statistics ---
print("  Saving summary_stats.csv...")
stats = []
for mi, month in enumerate(months):
    stats.append({
        "Year_month": month,
        "year": int(month.split("-")[0]),
        "month": int(month.split("-")[1]),
        "total_cases_in_sources": float(cases_matrix[:, mi].sum()),
        "total_potential_intro": float(potential_intro[mi, :].sum()),
        "mean_potential_intro": float(potential_intro[mi, :].mean()),
        "max_potential_intro": float(potential_intro[mi, :].max()),
        "risk_destinations_above_zero": int((potential_intro[mi, :] > 0).sum()),
        "pct_destinations_reached": 100 * (potential_intro[mi, :] > 0).sum() / n
    })

stats_df = pd.DataFrame(stats)
stats_df.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"), index=False)
print(f"    Saved: summary_stats.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"MONTHLY INTRODUCTION RISK COMPUTATION COMPLETE")
print(f"{'='*60}")
print(f"  Municipalities:                    {n}")
print(f"  Months computed:                   {n_months}")
print(f"  Period:                            {months[0]} -> {months[-1]}")
print(f"  Total cases in period:             {total_from_cases}")
print(f"  Mean monthly total intro risk:     {potential_intro.sum(axis=1).mean():.2f}")
print(f"  Mean monthly intro risk/dest:      {potential_intro.mean():.6f}")
print(f"  Avg % destinations reached/month:  {stats_df['pct_destinations_reached'].mean():.1f}%")
print(f"")
print(f"  Output files in {OUT_DIR}/:")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fname:40s} {size_mb:.2f} MB")
print(f"{'='*60}")
