#!/usr/bin/env python
"""
radiation_model.py
==================
Compute the radiation model matrix (Simini et al. 2012) for Colombian municipalities.

Inputs:
  - map/map_Colombia.shp          : shapefile with DANE code, centroid Lat/Lon, min_msnm
  - population/col_ppp_YYYY_1km_Aggregated.tif : WorldPop rasters (1 km² resolution) for 2007-2024

Outputs (saved to radiation/):
  - municipality_data.csv         : DANE code, name, coordinates, min altitude, avg population
  - radiation_matrix.npy          : numpy array (n×n) of M_ij probabilities
  - radiation_matrix.rds          : R-compatible format (.rds)
  - radiation_matrix.csv.gz       : compressed CSV (only upper triangle, for inspection)
  - s_ij_matrix.npy               : the intermediate s_ij matrix (population within radius)
  - population_per_municipality.csv : population by year for each municipality

Reference: Simini et al. (2012) Nature 484, 96-100
           https://doi.org/10.1038/nature10856
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
import os, glob, warnings, gzip, csv
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_SHP      = os.path.join(PROJECT_ROOT, "map", "map_Colombia.shp")
POP_DIR      = os.path.join(PROJECT_ROOT, "population")
OUT_DIR      = os.path.join(PROJECT_ROOT, "radiation")

YEARS = list(range(2007, 2025))  # 2007 – 2024

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[{datetime.now().isoformat()}] Loading shapefile...")
gdf = gpd.read_file(MAP_SHP)

# Ensure CRS is WGS84 lat/lon
if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# Ensure we have the required columns
expected_cols = {"DANE", "Lat", "Lon", "min_msnm", "MPIO_CNMBR"}
missing = expected_cols - set(gdf.columns)
if missing:
    raise KeyError(f"Missing columns in shapefile: {missing}")

n = len(gdf)
print(f"  Municipalities loaded: {n}")
print(f"  CRS: {gdf.crs}")

# ============================================================
# STEP 1: Extract population from WorldPop rasters for each municipality
# ============================================================
print(f"\n[{datetime.now().isoformat()}] Extracting population from WorldPop rasters (2007-2024)...")

# Prepare population matrix: rows = municipalities, cols = years
pop_matrix = np.zeros((n, len(YEARS)), dtype=np.float64)

# Find all raster files
raster_files = sorted(glob.glob(os.path.join(POP_DIR, "col_ppp_*_1km_Aggregated.tif")))
print(f"  Found {len(raster_files)} raster files")

# For GeoPandas operations, we need the geometry in the CRS matching the raster
# WorldPop rasters are in EPSG:4326
# Let's first test one raster to see its properties
with rasterio.open(raster_files[0]) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds
    raster_transform = src.transform
    raster_width = src.width
    raster_height = src.height
    print(f"  Raster CRS: {raster_crs}")
    print(f"  Raster bounds: {raster_bounds}")
    print(f"  Raster size: {raster_width} x {raster_height} px")

# Process each year
for yi, raster_path in enumerate(raster_files):
    year = YEARS[yi]
    print(f"  Processing year {year}... ", end="", flush=True)
    
    with rasterio.open(raster_path) as src:
        # The raster must be in the same CRS as the shapefile
        assert src.crs == gdf.crs, f"CRS mismatch for {raster_path}: {src.crs} vs {gdf.crs}"
        
        # For each municipality, extract the population values within its polygon
        for i in range(n):
            geom = [mapping(gdf.geometry.iloc[i])]
            try:
                out_image, out_transform = rio_mask(src, geom, crop=True, all_touched=True, filled=True)
                # out_image shape: (1, height, width)
                data = out_image[0]
                # Sum of population within the polygon
                pop_val = data[data > 0].sum()
                pop_matrix[i, yi] = pop_val
            except Exception as e:
                print(f"  Error for municipality {gdf['DANE'].iloc[i]}: {e}")
                pop_matrix[i, yi] = 0.0
    
    print(f"done (total pop: {pop_matrix[:, yi].sum():.0f})")

# Compute average population per municipality across all years
avg_pop = pop_matrix.mean(axis=1)

# Create municipality data table
municipality_df = gdf[["DANE", "MPIO_CNMBR", "Lat", "Lon", "min_msnm"]].copy()
municipality_df.columns = ["DANE", "name", "Lat", "Lon", "min_msnm"]
municipality_df["avg_pop"] = avg_pop

# Save population per year
pop_by_year_df = municipality_df[["DANE", "name"]].copy()
for yi, year in enumerate(YEARS):
    pop_by_year_df[f"pop_{year}"] = pop_matrix[:, yi]
pop_by_year_df.to_csv(os.path.join(OUT_DIR, "population_per_municipality.csv"), index=False)

municipality_df.to_csv(os.path.join(OUT_DIR, "municipality_data.csv"), index=False)
print(f"\n  Population data saved to {OUT_DIR}/")

# ============================================================
# STEP 2: Compute the radiation model matrix
# ============================================================
print(f"\n[{datetime.now().isoformat()}] Computing radiation model matrix...")

# Extract coordinates (centroids) in radians for distance computation
# Colombia is near the equator, so we can use a simple approximation:
# 1° latitude  ≈ 111.32 km
# 1° longitude ≈ 111.32 * cos(lat_mean) km
lat_rad = np.radians(gdf["Lat"].values)
lon_rad = np.radians(gdf["Lon"].values)

# Mean latitude for Colombia (approximately 4°N)
mean_lat_rad = np.radians(4.0)

# Conversion factors: degrees to km
km_per_deg_lat = 111.32  # constant
km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)  # ~111.0 km at 4°N

# Coordinates in km from a reference point
coords_km = np.column_stack([
    gdf["Lon"].values * km_per_deg_lon,
    gdf["Lat"].values * km_per_deg_lat
])

# Compute pairwise Euclidean distance matrix (in km)
print("  Computing distance matrix...")
dist_km = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    diff = coords_km - coords_km[i:i+1, :]
    dist_km[i, :] = np.sqrt((diff ** 2).sum(axis=1))

# Exclude self-distances (set to infinity so s_ii is not calculated for self)
np.fill_diagonal(dist_km, np.inf)

print(f"  Distance matrix computed: {n} × {n}")

# Population vector
p = avg_pop.copy()

# Compute s_ij: total population within radius d_ij of i, excluding i and j
# s_ij = sum(p_k for k where dist[i,k] <= dist[i,j] AND k != i AND k != j)
#
# Optimization: For each origin i, sort destinations by distance,
# then s_ij is the cumulative sum up to j, minus p_i and p_j

print("  Computing s_ij matrix (this may take a while)...")
s_ij = np.zeros((n, n), dtype=np.float64)

for i in range(n):
    if i % 100 == 0 and i > 0:
        print(f"    Progress: {i}/{n}")
    
    # Distances from i to all destinations
    dists_i = dist_km[i, :]
    
    # Get sorted indices (ascending distance)
    sorted_idx = np.argsort(dists_i)
    
    # Cumulative population sum as we move outward
    cum_pop = 0.0
    for rank, j in enumerate(sorted_idx):
        if np.isinf(dists_i[j]):
            break  # no more valid destinations
        # s_ij = cumulative population of all municipalities closer than j
        # (excluding i and j themselves)
        s_ij[i, j] = cum_pop
        cum_pop += p[j]

print("  s_ij matrix computed.")

# Save s_ij matrix
np.save(os.path.join(OUT_DIR, "s_ij_matrix.npy"), s_ij)

# ============================================================
# STEP 3: Calculate M_ij
# ============================================================
print("  Computing M_ij matrix...")

# To avoid division by zero: for pairs where p_i + s_ij == 0 or p_i + p_j + s_ij == 0
# (i.e., origin with zero population), set M_ij = 0

# M_ij = p_i * p_j / ((p_i + s_ij) * (p_i + p_j + s_ij))
p_i = p[:, np.newaxis]  # shape (n, 1)
p_j = p[np.newaxis, :]  # shape (1, n)

denom1 = p_i + s_ij
denom2 = p_i + p_j + s_ij

# Avoid division by zero
nonzero_mask = (denom1 > 0) & (denom2 > 0)

M = np.zeros((n, n), dtype=np.float64)
M[nonzero_mask] = (p_i * p_j)[nonzero_mask] / (denom1[nonzero_mask] * denom2[nonzero_mask])

# Set diagonal to 0 (no self-movement)
np.fill_diagonal(M, 0.0)

print(f"  M_ij matrix computed: {n} × {n}")
print(f"  Non-zero entries: {np.count_nonzero(M)} / {n * n}")
print(f"  Row sums (should be < 1 for each origin):")
row_sums = M.sum(axis=1)
print(f"    Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}, Mean: {row_sums.mean():.6f}")

# ============================================================
# STEP 4: Save results
# ============================================================
print(f"\n[{datetime.now().isoformat()}] Saving results...")

# Save as numpy .npy
np.save(os.path.join(OUT_DIR, "radiation_matrix.npy"), M)
print(f"  Saved: radiation_matrix.npy")

# Save as compressed CSV for inspection (only upper triangle + diagonal)
print("  Saving compressed CSV...")
out_csv = os.path.join(OUT_DIR, "radiation_matrix.csv.gz")
with gzip.open(out_csv, "wt", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["origin_DANE", "dest_DANE", "M_ij", "dist_km"])
    dane_codes = gdf["DANE"].values
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] > 0:  # Skip zeros to save space
                writer.writerow([dane_codes[i], dane_codes[j], 
                                f"{M[i, j]:.10f}", f"{dist_km[i, j]:.2f}"])
print(f"  Saved: radiation_matrix.csv.gz")

# Save as .rds for R (using rds library if available, otherwise pyreadr)
try:
    import pyreadr
    # Create a dict with the matrix and metadata
    r_data = {
        "radiation_matrix": M,
        "municipality_data": municipality_df,
        "s_ij_matrix": s_ij,
        "population_by_year": pop_by_year_df
    }
    # pyreadr expects DataFrames, so convert numpy arrays
    M_df = pd.DataFrame(M, index=gdf["DANE"].values, columns=gdf["DANE"].values)
    M_df.index.name = "DANE_origin"
    M_df.columns.name = "DANE_dest"
    
    s_df = pd.DataFrame(s_ij, index=gdf["DANE"].values, columns=gdf["DANE"].values)
    s_df.index.name = "DANE_origin"
    s_df.columns.name = "DANE_dest"
    
    pyreadr.write_rds(os.path.join(OUT_DIR, "radiation_matrix.rds"), M_df)
    pyreadr.write_rds(os.path.join(OUT_DIR, "s_ij_matrix.rds"), s_df)
    print(f"  Saved: radiation_matrix.rds, s_ij_matrix.rds")
except ImportError:
    print("  pyreadr not available, skipping .rds export. Install with: pip install pyreadr")
    print("  Results saved as .npy and .csv.gz formats.")

# Save distance matrix
dist_df = pd.DataFrame(dist_km, index=gdf["DANE"].values, columns=gdf["DANE"].values)
dist_df.index.name = "DANE_origin"
dist_df.columns.name = "DANE_dest"
dist_df.to_csv(os.path.join(OUT_DIR, "distance_matrix_km.csv"), index=True)
print(f"  Saved: distance_matrix_km.csv")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"RADIATION MODEL COMPUTATION COMPLETE")
print(f"{'='*60}")
print(f"  Municipalities: {n}")
print(f"  Years averaged: {YEARS[0]}–{YEARS[-1]}")
print(f"")
print(f"  Matrix shape: {M.shape}")
print(f"  Non-zero entries: {np.count_nonzero(M)} / {n * n} ({100 * np.count_nonzero(M) / (n * n):.1f}%)")
print(f"  Row sum min/max/mean: {row_sums.min():.4f} / {row_sums.max():.4f} / {row_sums.mean():.4f}")
print(f"")
print(f"  Output files:")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"    {fname:40s} {size_mb:.2f} MB")
print(f"{'='*60}")
