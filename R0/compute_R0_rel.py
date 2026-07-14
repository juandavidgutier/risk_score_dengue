#!/usr/bin/env python
"""
compute_R0_rel.py
==================
Estimation of R0_rel using the REAL standard deviation
of daily temperature (sigma_real) obtained from ERA5 daily data
for ALL terrestrial pixels of Colombia.

Methodology:
  1) Load shapefile and monthly ERA5 NetCDF
  2) Download daily ERA5 data (12:00 UTC) for the entire domain
  3) Calculate sigma_real = standard deviation of daily means
     within each month, for each pixel
  4) Regrid sigma_real from 0.25° to 0.1°
  5) Calculate degree-days and Z(T_eff) for the entire grid
  6) Normalize: R0_rel = Z / Z_max
  7) Aggregate by municipality (population-weighted)
  8) Save results

Output in R0/:
  - R0_rel_monthly.csv.gz   : Monthly R0_rel with real ERA5 sigma (compressed)
  - R0_rel_monthly.csv      : Monthly R0_rel (uncompressed, for consumer scripts)
  - R0_rel_monthly.rds      : for R
  - R0_rel_summary_stats.csv: monthly statistics

(daily ERA5 data and sigma_real_monthly_full.nc saved in R0/)

References:
  - Mordecai et al. (2017, 2019): Briere + Quadratic parameters for Aedes
  - ERA5: Copernicus Climate Change Service (C3S) ERA5 reanalysis
"""

import os, sys, warnings, gc, time
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from affine import Affine
from datetime import datetime
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIG
# =============================================================================
# File is in R0/compute_R0_rel.py -> up 2 levels to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_NC = os.path.join(BASE_DIR, 'temperature', 'data_stream-moda.nc')
SHP_PATH = os.path.join(BASE_DIR, 'map', 'map_Colombia.shp')
POP_DIR = os.path.join(BASE_DIR, 'population')
OUT_DIR = os.path.join(BASE_DIR, 'R0')                     # main output and intermediate data
DAILY_NC = os.path.join(OUT_DIR, 'era5_daily_t2m.nc')

os.makedirs(OUT_DIR, exist_ok=True)

# Mordecai parameters (Aedes aegypti)
PARAM_A     = np.array([ 2.248952, 40.13383,     0.000271964])  # Briere
PARAM_MU    = np.array([12.71508,  38.04809469,  -0.757869    ])  # Quadratic
PARAM_PDR   = np.array([17.33263,  42.19592,     0.000135891])  # Briere
T_BASE_PDR  = PARAM_PDR[0]  # ~17.33 C

# =============================================================================
# 1. THERMAL SUITABILITY FUNCTIONS (Mordecai)
# =============================================================================

def briere(T, parm):
    T_safe = np.maximum(T, parm[0] + 0.001)
    T_safe = np.minimum(T_safe, parm[1] - 0.001)
    result = parm[2] * T_safe * (T_safe - parm[0]) * np.sqrt(np.maximum(parm[1] - T_safe, 0))
    return np.maximum(result, 0)

def quad(T, parm):
    result = -parm[2] * (T - parm[0]) * (parm[1] - T)
    return np.maximum(result, 0)

def temp_suitability(T_celsius):
    T = np.asarray(T_celsius, dtype=np.float64)
    a = briere(T, PARAM_A)
    lf = quad(T, PARAM_MU)
    mu = np.where(lf <= 0, 1.0, 1.0 / lf)
    PDR = briere(T, PARAM_PDR)
    PDR = np.maximum(PDR, 1e-10)
    Z = (a**2 * np.exp(-mu / PDR)) / np.maximum(mu, 1e-10)
    return np.maximum(Z, 0)

# =============================================================================
# 2. DEGREE-DAYS FUNCTIONS
# =============================================================================

def expected_degree_days_vectorized(T_mean, sigma, days=30):
    result = np.zeros_like(T_mean, dtype=np.float64)
    valid = ~np.isnan(T_mean) & ~np.isnan(sigma) & (sigma > 0)
    if not valid.any():
        return result
    high = valid & (T_mean >= T_BASE_PDR + 3 * sigma)
    result[high] = np.maximum(T_mean[high] - T_BASE_PDR, 0) * days
    mid = valid & (~high) & (T_mean > T_BASE_PDR - 4 * sigma)
    z = (T_BASE_PDR - T_mean[mid]) / sigma[mid]
    dd_per_day = sigma[mid] * (
        scipy_stats.norm.pdf(z) - z * (1 - scipy_stats.norm.cdf(z))
    )
    result[mid] = np.maximum(dd_per_day, 0) * days
    return result

# =============================================================================
# 3. LOAD SPATIAL DATA AND MONTHLY TEMPERATURE
# =============================================================================
t0 = time.time()
print(f"[{datetime.now().isoformat()}] Loading shapefile and monthly NetCDF...")

gdf = gpd.read_file(SHP_PATH)
gdf = gdf.to_crs('EPSG:4326')
gdf['DANE_int'] = gdf['DANE'].astype(int)
n_munis = len(gdf)
print(f"   {n_munis} municipalities loaded")

# Load monthly NetCDF
ds = xr.open_dataset(TEMP_NC)
lats = ds.latitude.values
lons = ds.longitude.values
t2m = ds.t2m.values.astype(np.float64)
ds.close()

# Identify months in NetCDF
n_months, n_lat, n_lon = t2m.shape
months_list = [f"{2007 + m//12}-{(m%12)+1}" for m in range(n_months)]
print(f"   Monthly temp shape: {t2m.shape}")

pixel_lon = (lons[-1] - lons[0]) / (n_lon - 1)
pixel_lat = (lats[0] - lats[-1]) / (n_lat - 1)
transform_temp = Affine(pixel_lon, 0, lons[0] - pixel_lon/2,
                        0, -pixel_lat, lats[0] + pixel_lat/2)

# =============================================================================
# 4. DOWNLOAD / LOAD DAILY ERA5 DATA (12:00 UTC)
# =============================================================================
print(f"[{datetime.now().isoformat()}] Retrieving daily ERA5 data...")

if os.path.exists(DAILY_NC):
    print(f"   Loading existing daily data: {DAILY_NC}")
    ds_daily = xr.open_dataset(DAILY_NC)
else:
    print("   Downloading daily ERA5 data (12:00 UTC) via CDS API...")
    print("   One request per year, approx 40-80 MB/year, ~1-2 min/year")
    try:
        import cdsapi
        c = cdsapi.Client()

        all_years = list(range(2007, 2025))
        for year in all_years:
            year_nc = os.path.join(OUT_DIR, f'era5_daily_{year}.nc')
            if os.path.exists(year_nc):
                print(f"     Year {year}: already exists ({os.path.getsize(year_nc)/1e6:.0f} MB)")
                continue
            print(f"     Downloading {year}...", end=' ', flush=True)
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': '2m_temperature',
                    'year': str(year),
                    'month': [f'{m:02d}' for m in range(1, 13)],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': '12:00',
                    'format': 'netcdf',
                    'area': [lats[0], lons[0], lats[-1], lons[-1]],
                },
                year_nc
            )
            print(f"OK ({os.path.getsize(year_nc)/1e6:.0f} MB)")
            time.sleep(1)

        print("   Loading downloaded years...")
        yearly_files = sorted([os.path.join(OUT_DIR, f'era5_daily_{y}.nc') for y in all_years])
        datasets = [xr.open_dataset(f, chunks='auto') for f in yearly_files]
        ds_combined = xr.concat(datasets, dim='valid_time')
        ds_combined.to_netcdf(DAILY_NC)
        print(f"   Saved: {DAILY_NC} ({os.path.getsize(DAILY_NC)/1e6:.0f} MB)")
        for d in datasets:
            d.close()
        ds_daily = xr.open_dataset(DAILY_NC)
    except Exception as e:
        print(f"   [ERROR] Could not download daily data: {e}")
        raise

# =============================================================================
# 5. CALCULATE SIGMA_REAL FROM DAILY DATA (0.25°) AND REGRID TO 0.1°
# =============================================================================
print(f"[{datetime.now().isoformat()}] Calculating sigma_real from daily ERA5 data...")

t2m_daily_kelvin = ds_daily.t2m.values.astype(np.float64)
daily_times = ds_daily.valid_time.values

# Convert to Celsius
t2m_daily = t2m_daily_kelvin - 273.15

# Daily grid coordinates (0.25°)
daily_lats = ds_daily.latitude.values
daily_lons = ds_daily.longitude.values
n_daily_lat, n_daily_lon = len(daily_lats), len(daily_lons)
print(f"   Daily grid: {n_daily_lat} lat x {n_daily_lon} lon ({n_daily_lat*n_daily_lon:,} pixels)")

times_dt = pd.to_datetime(daily_times)
daily_ym = times_dt.strftime('%Y-%m').tolist()
unique_ym = sorted(set(daily_ym))
print(f"   {len(unique_ym)} unique months in daily data")

# --- Step 1: Calculate sigma_real on daily grid (0.25°) ---
print(f"   Step 1: computing sigma_real on 0.25° grid...")

month_indices = {}
for i, ym in enumerate(daily_ym):
    if ym not in month_indices:
        month_indices[ym] = []
    month_indices[ym].append(i)

sigma_real_daily = np.full((len(unique_ym), n_daily_lat, n_daily_lon), np.nan, dtype=np.float64)

for mi, ym in enumerate(unique_ym):
    if mi % 12 == 0:
        print(f"     Month {mi+1}/{len(unique_ym)}")
    idx_list = month_indices[ym]
    t2m_month = t2m_daily[idx_list]
    if t2m_month.shape[0] >= 3:
        sigma_real_daily[mi] = np.nanstd(t2m_month, axis=0, ddof=1)
    else:
        sigma_real_daily[mi] = 0.0

sigma_real_daily = np.nan_to_num(sigma_real_daily, nan=0.0)
print(f"   sigma_real (0.25°): {sigma_real_daily.min():.2f}C - {sigma_real_daily.max():.2f}C")

# --- Step 2: Regrid from 0.25° → 0.1° using RegularGridInterpolation ---
print(f"   Step 2: regridding from 0.25° ({n_daily_lat}x{n_daily_lon}) to 0.1° ({n_lat}x{n_lon})...")

from scipy.interpolate import RegularGridInterpolator

mesh_lon, mesh_lat = np.meshgrid(lons, lats)
points_target = np.column_stack([mesh_lat.ravel(), mesh_lon.ravel()])

sigma_real_grid = np.full((len(unique_ym), n_lat, n_lon), np.nan, dtype=np.float64)

for mi in range(len(unique_ym)):
    if mi % 36 == 0:
        print(f"     Regrid {mi+1}/{len(unique_ym)}")

    interp = RegularGridInterpolator(
        (daily_lats, daily_lons),
        sigma_real_daily[mi],
        bounds_error=False,
        fill_value=np.nan,
        method='linear'
    )
    sigma_regrid = interp(points_target)
    sigma_regrid = sigma_regrid.reshape(n_lat, n_lon)

    # Where interpolation is not valid (edges), use nearest value
    nan_mask = np.isnan(sigma_regrid)
    if nan_mask.any():
        interp_nearest = RegularGridInterpolator(
            (daily_lats, daily_lons),
            sigma_real_daily[mi],
            bounds_error=False,
            fill_value=np.nan,
            method='nearest'
        )
        sigma_regrid[nan_mask] = interp_nearest(points_target).reshape(n_lat, n_lon)[nan_mask]

    sigma_real_grid[mi] = np.nan_to_num(sigma_regrid, nan=0.0)

print(f"   sigma_real (0.1° regrid): {np.nanmin(sigma_real_grid):.2f}C - {np.nanmax(sigma_real_grid):.2f}C")

# --- Step 3: Save sigma_real to NetCDF ---
time_coords = pd.to_datetime([f'{ym}-01' for ym in unique_ym])
sigma_ds = xr.Dataset(
    {'sigma_real': (['valid_time', 'latitude', 'longitude'], sigma_real_grid.astype(np.float32))},
    coords={
        'valid_time': time_coords,
        'latitude': lats,
        'longitude': lons,
    }
)
sigma_ds.to_netcdf(os.path.join(OUT_DIR, 'sigma_real_monthly_full.nc'))
print(f"   [OK] sigma_real_monthly_full.nc saved ({len(unique_ym)} months)")
sigma_ds.close()
ds_daily.close()

# =============================================================================
# 6. CALCULATE R0_rel USING SIGMA_REAL
# =============================================================================
print(f"[{datetime.now().isoformat()}] Calculating degree-days and Z(T) with real ERA5 sigma...")

# Align temporal indices
if len(unique_ym) != n_months:
    print(f"   Warning: unique_ym={len(unique_ym)} vs n_months={n_months}. Using months_list.")
    common_ym = months_list
    n_months_eff = n_months
else:
    n_months_eff = n_months
    common_ym = months_list

t2m_celsius = t2m - 273.15

# Calculate Z for all months and pixels using sigma_real
Z_all = np.full((n_months_eff, n_lat, n_lon), np.nan, dtype=np.float64)
for mi in range(n_months_eff):
    T_month = t2m_celsius[mi]
    sigma_m = sigma_real_grid[mi]

    dd = expected_degree_days_vectorized(T_month, sigma_m, days=30)

    T_eff = np.copy(T_month)
    mask_dd = dd > 0
    T_eff[mask_dd] = np.minimum(
        T_BASE_PDR + dd[mask_dd] / 30,
        PARAM_PDR[1] - 0.1
    )

    Z = temp_suitability(T_eff)
    Z[np.isnan(T_month)] = np.nan
    Z_all[mi] = Z

    if (mi + 1) % 36 == 0:
        zr = Z_all[mi][~np.isnan(Z_all[mi])]
        pct0 = 100 * np.sum(zr < 1e-10) / len(zr) if len(zr) > 0 else 0
        print(f"   Month {mi+1}/{n_months_eff}: Z mean={np.nanmean(zr):.4f}  max={np.nanmax(zr):.4f}  Z=0: {pct0:.0f}%")

Z_max = np.nanmax(Z_all)
print(f"   Global Z_max = {Z_max:.6f}")

R0_rel = np.where(Z_max > 0, Z_all / Z_max, Z_all).astype(np.float32)
R0_rel[np.isnan(Z_all)] = np.nan
print(f"   R0_rel range: [{np.nanmin(R0_rel):.8f}, {np.nanmax(R0_rel):.6f}]")

del Z_all, t2m_celsius
gc.collect()

# =============================================================================
# 7. POPULATION: 1km -> THERMAL GRID AGGREGATION
# =============================================================================
print(f"[{datetime.now().isoformat()}] Aggregating 1km population to thermal grid...")

years_range = list(range(2007, 2025))
pop_grid = np.zeros((n_lat, n_lon), dtype=np.float64)
n_pop_files = 0

for year in years_range:
    pop_file = os.path.join(POP_DIR, f'col_ppp_{year}_1km_Aggregated.tif')
    if not os.path.exists(pop_file):
        continue
    with rasterio.open(pop_file) as src:
        src_band = src.read(1).astype(np.float64)
        src_band[src_band < 0] = 0
        pop_temp = np.zeros((n_lat, n_lon), dtype=np.float64)
        reproject(
            source=src_band,
            destination=pop_temp,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_temp,
            dst_crs='EPSG:4326',
            resampling=Resampling.sum
        )
        pop_grid += pop_temp
        n_pop_files += 1

pop_grid /= max(n_pop_files, 1)
pop_weights = np.maximum(pop_grid, 1e-10)
R0_flat = R0_rel.reshape(n_months_eff, -1)

# =============================================================================
# 8. AGGREGATE BY MUNICIPALITY (ALL MUNICIPALITIES)
# =============================================================================
print(f"[{datetime.now().isoformat()}] Aggregating R0_rel by municipality...")

results_list = []
n_processed = 0

for idx_muni, (_, row) in enumerate(gdf.iterrows()):
    dane = row['DANE_int']
    geom = row.geometry

    mask = geometry_mask(
        [geom],
        out_shape=(n_lat, n_lon),
        transform=transform_temp,
        all_touched=True,
        invert=True
    )

    pix_indices = np.where(mask)
    n_pix = len(pix_indices[0])

    if n_pix == 0:
        mask = geometry_mask(
            [geom],
            out_shape=(n_lat, n_lon),
            transform=transform_temp,
            all_touched=False,
            invert=True
        )
        pix_indices = np.where(mask)
        n_pix = len(pix_indices[0])

    if n_pix == 0:
        lat_c, lon_c = row.geometry.centroid.y, row.geometry.centroid.x
        lat_idx = int((lats[0] - lat_c) / pixel_lat)
        lon_idx = int((lon_c - lons[0]) / pixel_lon)
        lat_idx = np.clip(lat_idx, 0, n_lat - 1)
        lon_idx = np.clip(lon_idx, 0, n_lon - 1)
        pix_indices = (np.array([lat_idx]), np.array([lon_idx]))
        n_pix = 1

    idx_flat = np.ravel_multi_index(pix_indices, (n_lat, n_lon))
    pop_local = pop_weights.flatten()[idx_flat]
    pop_total = pop_local.sum()

    if pop_total <= 0:
        w = np.ones(n_pix)
    else:
        w = pop_local

    r0_muni = R0_flat[:, idx_flat]

    for m in range(n_months_eff):
        r0_vals = r0_muni[m]
        valid = ~np.isnan(r0_vals)
        if not valid.any():
            wr0 = 0.0
        elif pop_total <= 0:
            wr0 = float(np.mean(r0_vals[valid]))
        else:
            wv = w[valid]
            if wv.sum() <= 0:
                wr0 = float(np.mean(r0_vals[valid]))
            else:
                wr0 = float(np.average(r0_vals[valid], weights=wv))
        results_list.append({'DANE': dane, 'Year_month': common_ym[m], 'R0_rel': wr0})

    n_processed += 1
    if n_processed % 100 == 0:
        print(f"   {n_processed}/{n_munis} municipalities processed")

print(f"   {n_processed}/{n_munis} municipalities processed")

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================
print(f"[{datetime.now().isoformat()}] Saving results...")

df_daily = pd.DataFrame(results_list)
df_daily[['Year', 'Month']] = df_daily['Year_month'].str.split('-', expand=True).astype(int)
df_daily = df_daily.sort_values(['DANE', 'Year_month']).reset_index(drop=True)

print(f"   Shape: {df_daily.shape}")
print(f"   Municipalities: {df_daily['DANE'].nunique()}")
print(f"   Range: {df_daily['Year_month'].min()} -> {df_daily['Year_month'].max()}")
print(f"   R0_rel: [{df_daily['R0_rel'].min():.8f}, {df_daily['R0_rel'].max():.6f}]")

# --- Main output in R0/ (standard project format) ---
df_out = df_daily  # columns: DANE, Year_month, Year, Month, R0_rel

# Compressed CSV (for scripts using compression='gzip')
df_out.to_csv(os.path.join(OUT_DIR, 'R0_rel_monthly.csv.gz'),
              index=False, compression='gzip')
print(f"   [OK] R0_rel_monthly.csv.gz")

# Uncompressed CSV (for scripts using simple pd.read_csv)
df_out.to_csv(os.path.join(OUT_DIR, 'R0_rel_monthly.csv'), index=False)
print(f"   [OK] R0_rel_monthly.csv")

# RDS
try:
    import pyreadr
    pyreadr.write_rds(os.path.join(OUT_DIR, 'R0_rel_monthly.rds'), df_out)
    print(f"   [OK] R0_rel_monthly.rds")
except:
    pass

# Statistics
stats = df_out.groupby('Year_month').agg(
    mean_R0_rel=('R0_rel', 'mean'),
    min_R0_rel=('R0_rel', 'min'),
    max_R0_rel=('R0_rel', 'max'),
    std_R0_rel=('R0_rel', 'std')
).reset_index()
stats.to_csv(os.path.join(OUT_DIR, 'R0_rel_summary_stats.csv'), index=False)
print(f"   [OK] R0_rel_summary_stats.csv")

# =============================================================================
# FINAL
# =============================================================================
elapsed = time.time() - t0
print(f"\n[{datetime.now().isoformat()}] COMPLETE in {elapsed/60:.1f} min")
print(f"   Files in {OUT_DIR}/:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"     {f:50s} {size_mb:.2f} MB")


print(f"\n[OK] COMPLETE")
