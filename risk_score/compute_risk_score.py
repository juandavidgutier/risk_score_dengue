"""
compute_risk_score.py
=====================
Computes the dengue introduction risk score for destination municipalities
with minimum altitude > 2000 m a.s.l. in Colombia.

Risk formula:

    Risk = potential_intro x R0_rel x susceptibility x density

  Where each factor is constructed as follows:
  - potential_intro : expected imported cases (radiation model),
                     rescaled by min-max with floor to range [eps, 1].
  - R0_rel            : relative thermal suitability for transmission using
                     DEGREE-DAYS (Mordecai 2017, degree-days > T_base_PDR=17.33C,
                     normalized to [0,1]). Used in its native scale, WITHOUT floor:
                     its zero is biological (no thermal viability -> risk 0),
                     so it is preserved and not rescaled.
  - susceptibility    : population susceptibility proxy (Fig S9, Childs
                     2025, cumulative incidence in window 7-18 months). Already
                     bounded to (0,1] and never exactly 0, so it is used in
                     native scale, WITHOUT floor.
  - density         : population density (people/km2) of the municipality
                     (WorldPop 1km raster aggregated by municipality, shapefile area),
                     rescaled by min-max with floor to range [eps, 1].

STANDARDIZATION (min-max with eps floor):
  In the multiplicative product, a single factor at its minimum value (or equal to
  0) would collapse the entire score to 0, even if the other factors indicate
  high risk. To avoid artificial NUMERIC zeros (e.g., a municipality with
  potential_intro = 0 in a month with no source cases, or a "typical" municipality
  in z-score = 0), unbounded factors are rescaled with a floor:

      X_s = eps + (1 - eps) * (X - X_min) / (X_max - X_min)   -> range [eps, 1]

  with eps = 0.01. This guarantees X_s >= eps > 0. The floor is NOT applied to R0_rel
  (its zero is biological and desirable) nor to susceptibility (already bounded to (0,1]).
  The normalization is computed over the ENTIRE domain (all municipalities,
  all months) so that the scale is stable and comparable between municipalities.

Note: Population density (people/km2) is used instead of total population
      (or log_population) to capture demographic concentration per unit area.
      This avoids overestimating large but sparsely populated municipalities
      (typical of high mountains) compared to small but densely populated ones.
      Density is rescaled with min-max + eps floor to range [0.01, 1] in the
      multiplicative product.

Note IMPORTANT: As of this version, the R0_rel component is computed
      using the DEGREE-DAYS approach (not the direct monthly temperature
      estimation). Degree-days estimate cumulative heat above the
      PDR threshold (~17.33 C) assuming Normal intra-monthly distribution
      with sigma = 2.5 + 0.0008 * altitude. This replaces the previous estimation
      that underestimated thermal suitability in high areas.

Output in risk_score/:
  - risk_monthly.csv.gz          : Complete monthly risk (long format)
  - risk_monthly.rds             : For loading in R
  - risk_matrix.rds              : Matrix 1122 x 216
  - risk_summary_stats.csv       : Monthly risk statistics
  - risk_2000m.csv.gz   : Only municipalities >2000m (long format)
  - risk_2000m.rds      : For R
  - risk_2000m_matrix.rds : Matrix only for destinations >2000m
  - risk_2000m_summary.csv : Summary per destination municipality
  - risk_map_mean.png            : Mean risk map (2007-2024) >2000m
  - risk_map_trend.png           : Trend map (2007-2014 vs 2015-2024)
  - risk_component_potential_intro.png    : Mean potential_intro map >2000m
  - risk_component_R0_rel.png        : Mean R0_rel map (degree-days) >2000m
  - risk_component_susceptibility.png: Mean susceptibility map >2000m
  - risk_component_density.png       : Mean population density (scaled) >2000m
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "C:/"
OUTPUT_DIR = os.path.join(BASE_DIR, "risk_score")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DENGUE RISK IN COLOMBIA")
print("Filter: destination municipalities with altitude > 2000 m a.s.l.")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/9] Loading data...")

# --- potential_intro ---
print("  potential_intro...", end=" ")
intro = pd.read_csv(
    os.path.join(BASE_DIR, "potential_introduction", "potential_intro_monthly.csv.gz"),
    compression="gzip"
)
intro["Year_month"] = intro["Year_month"].astype(str)
print(f"{len(intro):,} records, {intro['DANE'].nunique()} municipalities")

# --- R0_rel ---
print("  R0_rel...", end=" ")
r0 = pd.read_csv(os.path.join(BASE_DIR, "R0", "R0_rel_monthly.csv"))
r0["Year_month"] = r0["Year_month"].astype(str)
print(f"{len(r0):,} records, {r0['DANE'].nunique()} municipalities")

# --- susceptibility ---
print("  susceptibility...", end=" ")
sus = pd.read_csv(os.path.join(BASE_DIR, "proxy_susceptibility", "susceptibility_monthly.csv"))
sus["Year_month"] = sus["Year_month"].astype(str)
print(f"{len(sus):,} records, {sus['DANE'].nunique()} municipalities")

# --- annual population per municipality (from WorldPop 1km rasters) ---
print("  annual population (from pop/ rasters)...", end=" ")
pop_annual = pd.read_csv(os.path.join(BASE_DIR, "radiation", "population_per_municipality.csv"))
print(f"{len(pop_annual)} municipalities, columns: {[c for c in pop_annual.columns if c.startswith('pop_')]}")

# --- dengue cases ---
print("  dengue cases...", end=" ")
cases_df = pd.read_csv(os.path.join(BASE_DIR, "cases", "cases_dengue.csv"))
cases_df["Year_month"] = cases_df["Year_month"].astype(str)
print(f"{len(cases_df):,} records, {cases_df['DANE'].nunique()} municipalities")

# --- shapefile with names and geometry ---
print("  shapefile...", end=" ")
import geopandas as gpd
shp = gpd.read_file(os.path.join(BASE_DIR, "map", "map_Colombia.shp"))
shp["DANE"] = shp["DANE"].astype(int)
print(f"{len(shp)} municipalities")

# --- area in km2 from geometry (projected to Web Mercator for area calculation) ---
print("  area_km2 from geometry...", end=" ")
shp_proj = shp.to_crs("EPSG:3857")
shp["area_km2"] = shp_proj.geometry.area / 1_000_000
print(f"range [{shp['area_km2'].min():.2f}, {shp['area_km2'].max():.2f}] km2")

# --- altitude.csv with altitudes ---
print("  altitude.csv (altitudes)...", end=" ")
alt_df = pd.read_csv(os.path.join(BASE_DIR, "map", "altitude.csv"))
alt_df["DANE"] = alt_df["DANE"].astype(int)
print(f"{len(alt_df)} municipalities, altitude_min: {alt_df['altitude_min'].min()}-{alt_df['altitude_min'].max()} m")

# ============================================================
# 2. MERGE COMPONENTS
# ============================================================
print("\n[2/9] Merging components...")

# Merge intro + R0
print("  Merge potential_intro + R0_rel...", end=" ")
merged = intro.merge(r0[["DANE", "Year_month", "R0_rel"]], on=["DANE", "Year_month"], how="left")
print(f"{len(merged):,} records")

# Merge susceptibility (1057 municipalities with cases; 65 without cases -> susceptibility=1.0)
print("  Merge susceptibility...", end=" ")
merged = merged.merge(sus[["DANE", "Year_month", "susceptibility"]],
                       on=["DANE", "Year_month"], how="left")
n_missing_sus = merged["susceptibility"].isna().sum()
merged["susceptibility"] = merged["susceptibility"].fillna(1.0)
print(f"{n_missing_sus} NaN -> susceptibility=1.0")

# Merge annual population
# Convert pop_annual to long format
print("  Merge annual population...", end=" ")
pop_long = pop_annual.melt(
    id_vars=["DANE"],
    value_vars=[f"pop_{y}" for y in range(2007, 2025)],
    var_name="year_str",
    value_name="population"
)
pop_long["year"] = pop_long["year_str"].str.replace("pop_", "").astype(int)

merged["year"] = merged["Year_month"].apply(lambda x: int(x.split("-")[0]))
n_before = len(merged)
merged = merged.merge(pop_long[["DANE", "year", "population"]], on=["DANE", "year"], how="left")
n_missing_pop = merged["population"].isna().sum()
if n_missing_pop > 0:
    med_pop = merged.groupby("DANE")["population"].transform("median")
    merged["population"] = merged["population"].fillna(med_pop)
    print(f"{n_missing_pop} NaN -> municipal median")
else:
    print("OK")

# Merge shapefile for name and geometry
print("  Merge shapefile (name+geometry)...", end=" ")
merged = merged.merge(
    shp[["DANE", "MPIO_CNMBR", "area_km2", "geometry"]],
    on="DANE", how="left"
)
print("OK")

# Merge altitude.csv for actual altitude
print("  Merge altitude.csv (altitude_min)...", end=" ")
merged = merged.merge(
    alt_df[["DANE", "altitude_min", "altitude_max", "altitude_mean", "gradient_altitude", "altitude_centroid"]],
    on="DANE", how="left"
)
print("OK")

# ============================================================
# 3. COMPUTE RISK SCORE
# ============================================================
print("\n[3/9] Computing risk score...")
print("  Formula: Risk = potential_intro x R0_rel x susceptibility x density")
print("          (min-max standardization with eps=0.01 floor on potential_intro and density)")

# ------------------------------------------------------------------
# Standardization: min-max with eps floor in [eps, 1]
# ------------------------------------------------------------------
# In the multiplicative product, a factor at its minimum value (or at 0)
# would collapse the score to 0. To avoid artificial NUMERIC zeros, the
# unbounded factors are rescaled with an eps floor. The floor is NOT applied to
# R0_rel (biological zero) nor to susceptibility (already bounded to (0,1]).
EPS = 0.01

def minmax_floor(series, eps=EPS):
    """Rescales a series to [eps, 1] via min-max. Uses observed min/max.
    If all values are equal (range 0), returns 1.0 so as not to nullify the product."""
    s = series.astype(np.float64)
    s_min, s_max = np.nanmin(s), np.nanmax(s)
    rng = s_max - s_min
    if rng == 0 or np.isnan(rng):
        return pd.Series(np.ones(len(s)), index=s.index)
    s_norm = (s - s_min) / rng          # [0, 1]
    return eps + (1.0 - eps) * s_norm   # [eps, 1]

# Rescaled components (over the entire domain: all municipalities, all months)
merged["potential_intro_s"] = minmax_floor(merged["potential_intro"])

# Population density (people/km2), then rescaled to [eps, 1]
# Density is used instead of log_pop to capture population concentration
# (higher density -> higher concentration of susceptible people per km2).
merged["density"] = merged["population"] / merged["area_km2"]
merged["density"] = merged["density"].replace([np.inf, -np.inf], np.nan)
med_density = merged.groupby("DANE")["density"].transform("median")
merged["density"] = merged["density"].fillna(med_density)
merged["density_s"] = minmax_floor(merged["density"])

# R0_rel and susceptibility are used in native scale:
#   - R0_rel ~ U[0,1]: zero is biological (no thermal viability), preserved.
#   - susceptibility ~ U(0,1]: already bounded and never 0, no floor needed.
merged["R0_rel_use"]        = merged["R0_rel"].clip(lower=0.0).fillna(0.0)
merged["susceptibility_use"] = merged["susceptibility"].clip(lower=0.0, upper=1.0).fillna(1.0)

print(f"\n  Ranges after standardization (all municipalities, all months):")
print(f"    potential_intro_s : [{merged['potential_intro_s'].min():.6f}, {merged['potential_intro_s'].max():.6f}]")
print(f"    R0_rel_use        : [{merged['R0_rel_use'].min():.6f}, {merged['R0_rel_use'].max():.6f}]")
print(f"    susceptibility_use: [{merged['susceptibility_use'].min():.6f}, {merged['susceptibility_use'].max():.6f}]")
print(f"    density_s         : [{merged['density_s'].min():.6f}, {merged['density_s'].max():.6f}]")
print(f"    density (raw)     : [{merged['density'].min():.2f}, {merged['density'].max():.2f}] hab/km2")

# Final score (multiplicative product)
merged["risk"] = (merged["potential_intro_s"] *
                  merged["R0_rel_use"] *
                  merged["susceptibility_use"] *
                  merged["density_s"])

print(f"\n  Global risk statistics (all municipalities, all months):")
for stat in ["min", "mean", "median", "max", "std"]:
    val = getattr(merged["risk"], stat)()
    print(f"    {stat.capitalize():6s}: {val:.6f}")

print(f"\n  Global distribution:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p:02d}: {merged['risk'].quantile(p/100):.6f}")

# -----------------------------------------------------------------
# 3b. COMPUTE S2-lag4 (MAXIMUM lag0-lag4 OF RISK) DIRECTLY ON merged
# -----------------------------------------------------------------
print("\n[3b/9] Computing S2-lag4 (max lag0-lag4)...")

# Sort by DANE and Year_month to create lags correctly
merged_sorted = merged.sort_values(["DANE", "Year_month"]).copy()

# Create risk lags using groupby shift
merged_sorted["risk_lag1"] = merged_sorted.groupby("DANE")["risk"].shift(1)
merged_sorted["risk_lag2"] = merged_sorted.groupby("DANE")["risk"].shift(2)
merged_sorted["risk_lag3"] = merged_sorted.groupby("DANE")["risk"].shift(3)
merged_sorted["risk_lag4"] = merged_sorted.groupby("DANE")["risk"].shift(4)

# S2: maximum of lag0-lag4
merged_sorted["risk_s2_lag4"] = merged_sorted[["risk", "risk_lag1", "risk_lag2", "risk_lag3", "risk_lag4"]].max(axis=1)

# Fill NaN (first 4 months of each municipality) with current risk
merged_sorted["risk_s2_lag4"] = merged_sorted["risk_s2_lag4"].fillna(merged_sorted["risk"])

# Transfer column to merged (aligned by original index)
merged = merged.merge(
    merged_sorted[["DANE", "Year_month", "risk_s2_lag4"]],
    on=["DANE", "Year_month"], how="left"
)

print(f"  risk_s2_lag4 range: [{merged['risk_s2_lag4'].min():.6e}, {merged['risk_s2_lag4'].max():.6e}]")
print(f"  NaN values: {merged['risk_s2_lag4'].isna().sum()}")

# Also create risk_s2_lag4 for destinations since destinations is a subset of merged
# destinations is created in section 4, so merged_s2_lag4 will be available
del merged_sorted  # free memory

# ============================================================
# 4. FILTER DESTINATIONS > 2000 m
# ============================================================
print("\n[4/9] Filtering destination municipalities (>2000 m)...")

n_missing_alt = merged["altitude_min"].isna().sum()
if n_missing_alt > 0:
    print(f"  [WARN] {n_missing_alt} municipalities without altitude")

destinations = merged[merged["altitude_min"] > 2000].copy()
n_destinations = destinations["DANE"].nunique()
print(f"  Municipalities > 2000 m: {n_destinations}")
print(f"  Records: {len(destinations):,}")

if n_destinations == 0:
    print("  [ERROR] No municipalities > 2000 m. Check altitude column.")
    exit(1)

# TOP 10 highest
dest_uniq = destinations[["DANE", "MPIO_CNMBR", "altitude_min"]].drop_duplicates()
print(f"\n  TOP 10 highest municipalities:")
for _, r in dest_uniq.sort_values("altitude_min", ascending=False).head(10).iterrows():
    print(f"    {r['MPIO_CNMBR']:25s} DANE={r['DANE']:6d}  altitude={r['altitude_min']} m")

# Risk statistics in destinations
print(f"\n  Risk in destinations >2000 m (S2-lag4):")
d_risk = destinations["risk_s2_lag4"]
for stat in ["min", "mean", "median", "max", "std"]:
    val = getattr(d_risk, stat)()
    print(f"    {stat.capitalize():6s}: {val:.6f}")

# TOP 10 mean risk
risk_mean_muni = destinations.groupby(["DANE", "MPIO_CNMBR", "altitude_min"])["risk_s2_lag4"].mean().reset_index()
risk_mean_muni.columns = ["DANE", "municipio", "altitude", "risk_mean"]
print(f"\n  TOP 10 destinations with highest mean risk (S2-lag4):")
for _, r in risk_mean_muni.sort_values("risk_mean", ascending=False).head(10).iterrows():
    print(f"    {r['municipio']:25s} alt={r['altitude']:4d}m  risk_mean={r['risk_mean']:.6f}")

# ============================================================
# 5. TEMPORAL TRENDS
# ============================================================
print("\n[5/9] Computing temporal trends...")

destinations["year"] = destinations["Year_month"].apply(lambda x: int(x.split("-")[0]))
risk_annual = destinations.groupby(["DANE", "year"])["risk_s2_lag4"].mean().reset_index()
risk_annual.columns = ["DANE", "year", "risk_annual"]

early = risk_annual[risk_annual["year"] <= 2014].groupby("DANE")["risk_annual"].mean().reset_index()
early.columns = ["DANE", "risk_early"]
late = risk_annual[risk_annual["year"] >= 2015].groupby("DANE")["risk_annual"].mean().reset_index()
late.columns = ["DANE", "risk_late"]

trend = early.merge(late, on="DANE", how="outer").fillna(0)
trend["risk_change_pct"] = np.where(
    trend["risk_early"] > 0,
    (trend["risk_late"] - trend["risk_early"]) / trend["risk_early"] * 100,
    0
)
trend["risk_trend"] = np.where(trend["risk_change_pct"] > 25, "Increase",
                      np.where(trend["risk_change_pct"] < -25, "Decrease",
                               "Stable"))

n_inc = (trend["risk_trend"] == "Increase").sum()
n_dec = (trend["risk_trend"] == "Decrease").sum()
n_stab = (trend["risk_trend"] == "Stable").sum()
print(f"  Trends (>2000m):")
print(f"    Increase (>25%): {n_inc}")
print(f"    Stable:           {n_stab}")
print(f"    Decrease (<-25%): {n_dec}")

# GeoDataFrames for maps
dest_danes = destinations["DANE"].unique()

risk_mean_all = destinations.groupby("DANE")["risk_s2_lag4"].mean().reset_index()
risk_mean_gdf = shp[shp["DANE"].isin(dest_danes)].merge(risk_mean_all, on="DANE", how="left")

risk_early_gdf = shp[shp["DANE"].isin(dest_danes)].merge(early, on="DANE", how="left")
risk_late_gdf = shp[shp["DANE"].isin(dest_danes)].merge(late, on="DANE", how="left")
risk_trend_gdf = shp[shp["DANE"].isin(dest_danes)].merge(trend, on="DANE", how="left")


# ============================================================
# 6. COMPONENT RISK MAPS
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colombia = shp.copy()
colombia_outline = colombia.dissolve()

def base_map(ax):
    colombia_outline.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.8, zorder=1)
    colombia.plot(ax=ax, color="#e8e8e8", edgecolor="#dddddd", linewidth=0.3, zorder=2)
    ax.set_xlim([-80, -66]); ax.set_ylim([-5, 13])
    ax.set_aspect("equal"); ax.axis("off")

print("\n[7/9] Creating maps of each risk component...")

# Prepare aggregated data for each component (mean per municipality >2000m)
# All components are now displayed in 0-1 scale
component_configs = [
    {
        "col": "potential_intro_s",
        "title": "Fig 2a",
        "filename": "risk_component_potential_intro.png",
        "cmap": "OrRd",
        "fmt": ".4f"
    },
    {
        "col": "R0_rel_use",
        "title": "Fig 2b",
        "filename": "risk_component_R0_rel.png",
        "cmap": "RdYlGn_r",
        "fmt": ".4f"
    },
    {
        "col": "susceptibility_use",
        "title": "Fig 2c)",
        "filename": "risk_component_susceptibility.png",
        "cmap": "PuRd",
        "fmt": ".4f"
    },
    {
        "col": "density_s",
        "title": "Fig 2d",
        "filename": "risk_component_density.png",
        "cmap": "YlOrBr",
        "fmt": ".4f"
    }
]

for n_cfg, cfg in enumerate(component_configs, 1):
    col = cfg["col"]
    print(f"  Component map {n_cfg}/4: {col}...", end=" ")

    # Mean over entire period 2007-2024 for destinations >2000m
    comp_mean = destinations.groupby("DANE")[col].mean().reset_index()
    comp_mean.columns = ["DANE", "value"]

    # Merge with shapefile for mapping
    comp_gdf = shp[shp["DANE"].isin(dest_danes)].merge(comp_mean, on="DANE", how="left")

    # Fixed scale 0-1 for all component maps
    vmin, vmax = 0.0, 1.0

    cmap = plt.colormaps[cfg["cmap"]]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    base_map(ax)

    comp_gdf.sort_values("value").plot(
        ax=ax, column="value", cmap=cmap, norm=norm,
        edgecolor="#555555", linewidth=0.6, alpha=0.85, zorder=3
    )

    cax = fig.add_axes([0.15, 0.08, 0.70, 0.025])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cax, orientation="horizontal")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(os.path.join(OUTPUT_DIR, cfg["filename"]), dpi=200, bbox_inches="tight")
    plt.close(fig)

    obs_min = comp_gdf["value"].min()
    obs_max = comp_gdf["value"].max()
    print(f"obs. range [{obs_min:{cfg['fmt']}}, {obs_max:{cfg['fmt']}}] | fixed colorbar [0, 1]")

print("  Component maps saved to risk_score/ (all in 0-1 scale)")


# ============================================================
# 7. RISK MAPS
# ============================================================
print("\n[6/9] Creating risk maps...")



plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 8

cmap_risk = plt.colormaps["RdYlGn_r"]
cmap_diff = plt.colormaps["RdBu_r"]


# --- Fig 3a: Mean risk ---
print("  Map 1/2: Mean risk...")
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
base_map(ax)


vmin, vmax = risk_mean_gdf["risk_s2_lag4"].min(), risk_mean_gdf["risk_s2_lag4"].max()
if vmin == vmax: vmax = vmin + 0.001
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

risk_mean_gdf.sort_values("risk_s2_lag4").plot(
    ax=ax, column="risk_s2_lag4", cmap=cmap_risk, norm=norm,
    edgecolor="#555555", linewidth=0.6, alpha=0.85, zorder=3
)
ax.set_title("Average risk 2007-2024",
             fontsize=13, fontweight="bold", pad=20)

cax = fig.add_axes([0.15, 0.08, 0.70, 0.025])
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_risk),
             cax=cax, orientation="horizontal")
#             label="Average risk (potential_intro_s x R0_rel x susceptibility x density_s)")
#fig.text(0.5, 0.04, f"{len(risk_mean_gdf)} municipalities > 2000 m a.s.l. | 2007-2024",
#         ha="center", fontsize=9, color="#666666", fontstyle="italic")
plt.tight_layout(); fig.subplots_adjust(bottom=0.12)
fig.savefig(os.path.join(OUTPUT_DIR, "risk_map_mean.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# --- Fig 3b: Change between periods ---
print("  Map 2/2: Change between periods...")

trend_map = trend.merge(risk_mean_all, on="DANE", how="left")
trend_map["risk_diff"] = trend_map["risk_late"] - trend_map["risk_early"]
trend_gdf = shp[shp["DANE"].isin(dest_danes)].merge(trend_map, on="DANE", how="left")

q05, q95 = trend_gdf["risk_diff"].quantile(0.05), trend_gdf["risk_diff"].quantile(0.95)
v_max = max(abs(q05), abs(q95))

fig, ax = plt.subplots(1, 1, figsize=(10, 12))
base_map(ax)


norm_d = mcolors.TwoSlopeNorm(vmin=-v_max, vcenter=0, vmax=v_max)
trend_gdf.sort_values("risk_diff").plot(
    ax=ax, column="risk_diff", cmap=cmap_diff, norm=norm_d,
    edgecolor="#555555", linewidth=0.6, alpha=0.85, zorder=3
)
ax.set_title("Change in risk between \n2007-2014  vs  2015-2024",
             fontsize=13, fontweight="bold", pad=20)

cax = fig.add_axes([0.15, 0.08, 0.70, 0.025])
fig.colorbar(plt.cm.ScalarMappable(norm=norm_d, cmap=cmap_diff),
             cax=cax, orientation="horizontal")

n_inc_map = len(trend_gdf[trend_gdf["risk_diff"] > 0])
n_dec_map = len(trend_gdf[trend_gdf["risk_diff"] < 0])
#fig.text(0.5, 0.04,
#         f"{n_inc_map} municipalities with increase | {n_dec_map} with decrease",
#         ha="center", fontsize=9, color="#666666", fontstyle="italic")
plt.tight_layout(); fig.subplots_adjust(bottom=0.12)
fig.savefig(os.path.join(OUTPUT_DIR, "risk_map_trend.png"), dpi=200, bbox_inches="tight")
plt.close(fig)



# ============================================================
# 8. SAVE RESULTS
# ============================================================
all_year_months = sorted(merged['Year_month'].unique())
print("\n[9/9] Saving results...")


columns_to_save = ["DANE", "MPIO_CNMBR", "Year_month", "year", "altitude_min",
                    "potential_intro", "R0_rel", "susceptibility", "population", "risk",
                    "risk_s2_lag4"]

# Full
risk_save = merged[columns_to_save]
risk_save.to_csv(os.path.join(OUTPUT_DIR, "risk_monthly.csv.gz"), index=False, compression="gzip")
print(f"  risk_monthly.csv.gz: {len(risk_save):,} rec | {merged['DANE'].nunique()} munis")

# Destinations >2000m
dest_save = destinations[columns_to_save]
dest_save.to_csv(os.path.join(OUTPUT_DIR, "risk_2000m.csv.gz"), index=False, compression="gzip")
print(f"  risk_2000m.csv.gz: {len(dest_save):,} rec | {n_destinations} munis")

# RDS
try:
    import pyreadr
    pyreadr.write_rds(os.path.join(OUTPUT_DIR, "risk_monthly.rds"), risk_save)
    pyreadr.write_rds(os.path.join(OUTPUT_DIR, "risk_destinos_2000m.rds"), dest_save)

    risk_mat = risk_save.pivot_table(index="DANE", columns="Year_month", values="risk_s2_lag4", fill_value=0)
    risk_mat.index = risk_mat.index.astype(int); risk_mat.index.name = "DANE"
    pyreadr.write_rds(os.path.join(OUTPUT_DIR, "risk_matrix.rds"), risk_mat)

    risk_dm = dest_save.pivot_table(index="DANE", columns="Year_month", values="risk_s2_lag4", fill_value=0)
    risk_dm.index = risk_dm.index.astype(int); risk_dm.index.name = "DANE"
    pyreadr.write_rds(os.path.join(OUTPUT_DIR, "risk_destinos_2000m_matrix.rds"), risk_dm)
    print("  RDS files OK")
except Exception as e:
    print(f"  [WARN] RDS: {e}")

# Monthly statistics
monthly_stats = merged.groupby("Year_month").agg(
    mean_risk=("risk_s2_lag4", "mean"), median_risk=("risk_s2_lag4", "median"),
    std_risk=("risk_s2_lag4", "std"), min_risk=("risk_s2_lag4", "min"), max_risk=("risk_s2_lag4", "max"),
    p25_risk=("risk_s2_lag4", lambda x: x.quantile(0.25)),
    p75_risk=("risk_s2_lag4", lambda x: x.quantile(0.75))
).reset_index()
monthly_stats.to_csv(os.path.join(OUTPUT_DIR, "risk_summary_stats.csv"), index=False)
print(f"  risk_summary_stats.csv")
# Summary per destination municipality
risk_muni = dest_save.groupby(["DANE", "MPIO_CNMBR", "altitude_min"])["risk_s2_lag4"].agg(
    ["mean", "median", "max", "std"]
).reset_index()
risk_muni.columns = ["DANE", "municipio", "altitude_m", "risk_mean", "risk_median", "risk_max", "risk_std"]
risk_muni = risk_muni.merge(trend[["DANE", "risk_change_pct", "risk_trend", "risk_early", "risk_late"]],
                             on="DANE", how="left")
risk_muni.to_csv(os.path.join(OUTPUT_DIR, "risk_destinos_2000m_summary.csv"), index=False)
print(f"  risk_destinos_2000m_summary.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
FORMULA:
  Risk = potential_intro_s x R0_rel x susceptibility x density_s

  Where each component represents:
  1) potential_intro_s: Expected imported cases (radiation model),
                      rescaled with min-max + floor to [eps, 1]
  2) R0_rel         : Relative thermal suitability with DEGREE-DAYS for transmission
                      (Mordecai 2017, degree-days > T_base=17.33C,
                      sigma = 2.5 + 0.0008*altitude, normalized [0,1])
  3) susceptibility: Fraction of susceptible population (immunity proxy
                      by past incidence, Childs 2025 Fig S9, [0,1])
  4) density_s      : Population density (people/km2) of the destination municipality
                      (WorldPop 1km aggregated per municipality / shapefile area),
                      rescaled with min-max + floor to [eps, 1]

  Density is used instead of total population to capture demographic
  concentration. Small but densely populated municipalities are weighted
  appropriately compared to large, sparsely populated ones.

COVERAGE:
  - {merged['DANE'].nunique()} municipalities total
  - {n_destinations} destination municipalities with altitude > 2000 m
  - {len(all_year_months)} months (2007-2024)

RISK IN DESTINATIONS >2000 m:
  - Mean:  {destinations['risk_s2_lag4'].mean():.6f}
  - Median: {destinations['risk_s2_lag4'].median():.6f}
  - Max:    {destinations['risk_s2_lag4'].max():.6f}
  - P95:    {destinations['risk_s2_lag4'].quantile(0.95):.6f}

TREND (2007-2014 vs 2015-2024):
  - {n_inc} municipalities with increase >25%
  - {n_stab} stable municipalities
  - {n_dec} municipalities with decrease >25%
""")

print("Files generated in risk_score/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    unit = "MB" if size > 1e6 else ("KB" if size > 1e3 else "B")
    val = size / (1e6 if size > 1e6 else (1e3 if size > 1e3 else 1))
    print(f"  {f:45s} {val:.1f} {unit}")

print("\nMaps:")
print("  risk_map_mean.png            - Mean risk 2007-2024 in municipalities >2000m")
print("  risk_map_trend.png           - Change between 2007-2014 and 2015-2024 periods")
print("")
print("\nCOMPLETE")
