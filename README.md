# risk_score_dengue

**Prioritizing surveillance where dengue is rare: a monthly risk score to detect outbreak-prone municipalities above 2,000 m in Colombia**

**Author**: Juan D. Gutiérrez¹

¹ Universidad de Santander, Facultad de Ciencias Médicas y de la Salud, Instituto de Investigación Masira, Bucaramanga, Colombia

**Contact**: jdgutierrez@udes.edu.co

---

## Overview

This repository contains the complete analysis pipeline for a study that developed and validated a **multi-component risk score** to estimate the monthly occurrence of dengue in Colombian municipalities located above 2,000 meters above sea level from 2007 to 2024.

The risk score is computed as the product of four components:

1. **Case importation potential (Pintro)** — Expected number of infected individuals arriving at each high-altitude municipality, estimated via a parameter-free **radiation human mobility model** [Simini et al., 2012] combined with monthly dengue case data from source municipalities.
2. **Thermal suitability for transmission (R0-rel)** — Temperature-dependent relative reproductive number with a **degree-days correction** to account for intra-monthly temperature fluctuations, based on the mechanistic framework of Mordecai et al. [2017, 2019].
3. **Population susceptibility proxy (S)** — A dynamic proxy based on historical dengue incidence over a 7-to-18-month rolling window, following the methodology of Childs et al. [2025].
4. **Population density (D)** — Annual population per municipality from WorldPop gridded datasets, aggregated to municipal boundaries.

**Key findings**:
- Among 116 municipalities above 2,000 m, 81 reported at least one dengue case (875 cases total, 2.3% of municipality-months)
- AUC-PR of 0.042 — 1.9 times higher than random expectation
- AUC-ROC of 0.62
- Targeting the top 10% of risk scores captures approximately **25% of outbreak months**
- Spatio-temporal cross-validation confirmed mean recall of 24.0% (SD 3.8%) at the top decile
- Granada exhibited the highest average risk; Bogotá showed substantially lower risk despite higher absolute case numbers
- A negative Brier Skill Score (−0.05) confirms the score is a relative prioritization tool, not a calibrated probabilistic forecast

**Study period**: 2007–2024 (18 years)
**Geographic scope**: 116 municipalities above 2,000 m in Colombia
**Software**: Python 3.11, R v4.2.1

---

## Repository Structure

```
risk_score_dengue/
│
├── README.md                          # This file
│
├── cases/                             # Epidemiological input data
│   └── cases_dengue.csv               # Laboratory-confirmed dengue cases (SIVIGILA)
│
├── population/                        # WorldPop population rasters (1 km)
│   └── col_ppp_2007_1km_Aggregated.tif … col_ppp_2024_1km_Aggregated.tif
│
├── radiation/                         # Human mobility model (radiation model)
│   ├── radiation_model.py             # Computes radiation model probabilities Mij
│   ├── municipality_data.csv          # Municipality population and coordinates
│   ├── distance_matrix_km.csv         # Pairwise inter-municipality distances
│   ├── population_per_municipality.csv
│   ├── radiation_matrix.npy / .rds    # Output: Mij matrices
│   └── s_ij_matrix.npy / .rds         # Output: intervening opportunities matrix
│
├── potential_introduction/            # Case importation potential component
│   ├── potential_intro.py             # Computes Pintro for each municipality-month
│   ├── potential_intro_matrix.csv / .rds
│   ├── potential_intro_monthly.csv.gz / .rds
│   ├── summary_monthly.rds
│   └── summary_stats.csv
│
├── R0/                                # Thermal suitability component (R0-rel)
│   ├── compute_R0_rel.py              # Computes degree-days corrected R0-rel
│   ├── era5_daily_2007.nc … era5_daily_2024.nc  # ERA5 daily temperature data
│   ├── R0_rel_monthly.csv / .rds
│   └── R0_rel_summary_stats.csv
│
├── temperature/                       # ERA5 temperature-derived metrics
│   └── (auxiliary temperature grids and indices)
│
├── proxy_susceptibility/              # Population susceptibility component
│   ├── compute_susceptibility.py      # Computes S proxy from historical incidence
│   ├── susceptibility_matrix.csv / .rds
│   ├── susceptibility_monthly.csv / .rds
│   ├── susceptibility_summary_stats.csv
│   └── susceptibility_vs_R0rel.csv
│
├── risk_score/                        # Final risk score computation
│   ├── compute_risk_score.py          # Combines all 4 components into final score
│   ├── risk_2000m.csv.gz              # Output: final risk scores
│   ├── risk_destinos_2000m.rds / .csv
│   ├── risk_destinos_2000m_matrix.rds
│   ├── risk_destinos_2000m_summary.csv
│   ├── risk_matrix.rds
│   ├── risk_monthly.rds
│   └── risk_summary_stats.csv
│
├── diagnostic/                        # Model diagnostics and validation
│   ├── tests.py                       # Performance metrics (AUC-PR, AUC-ROC, recall, Brier, etc.)
│   ├── metrics.json                   # Computed metrics
│   ├── REPORT.txt                     # Diagnostic report
│   ├── bin_calibration.csv / .png
│   ├── pr_curve.png / roc_curve.png / lift_curve.png
│   ├── cv_boxplot.png
│   ├── monthly_recall_heatmap.png
│   ├── risk_distribution.png
│   └── ... (additional diagnostic plots)
│
├── lags/                              # Temporal lag optimization
│   ├── test_lags.py                   # Tests S1, S2, S3 lag strategies
│   ├── test_lags_component_opt.py     # Component-level lag optimization
│   ├── optimize_s3_weights.py         # Weight optimization for S3 strategy
│   ├── lags_detailed_analysis.py      # Detailed lag performance analysis
│   ├── detailed_analysis_s2_vs_base.py # Compares S2 (max lag) vs base model
│   ├── test_lags_dlnm.R               # Distributed Lag Non-linear Model (R)
│   ├── lags_results.json / .txt
│   ├── component_lag_results.json / .txt
│   ├── dlnm_results.json
│   └── ... (comparison plots)
│
├── map/                               # Geographic data
   ├── MGN_MPIO_POLITICO_wgs84.shp / .shx / .dbf / .prj / .qmd  # Municipality boundaries
   ├── map_Colombia.shp / .shx / .dbf / .prj / .qmd             # Simplified country outline
   └── altitude.csv                   # Minimum municipal elevation



```

---

## Setup

### 1. Python environment

It is recommended to use the pre-built virtual environment in `radiation_env/`. To activate it:

```bash
# On Windows (cmd):
radiation_env\Scripts\activate

# On Windows (PowerShell):
radiation_env\Scripts\Activate.ps1
```

> **Note:** The provided virtual environment (`radiation_env/`) is Windows-specific (Python 3.11). On Linux/macOS, create a new environment and install the required packages as described below.

Alternatively, create a new environment with the required packages:```bash

```bash
pip install numpy pandas scipy xarray netCDF4 rasterio geopandas shapely \
            matplotlib pyreadr pypdf seaborn scikit-learn
```

### 2. R environment (for DLNM analysis)

```r
install.packages(c("dlnm", "splines"))
```

### 3. Input data

The analysis requires the following data sources (some are already included in the repository):

- **Dengue case data**: `cases/cases_dengue.csv` — laboratory-confirmed notifications from SIVIGILA (2007–2024)
- **ERA5 temperature data**: Daily near-surface air temperature NetCDF files in `R0/` (2007–2024)
- **WorldPop population rasters**: 1 km resolution annual rasters in `population/` (2007–2024)
- **Municipality shapefile**: Colombian municipal boundaries from IGAC in `map/`
- **Municipality altitudes**: Minimum municipal elevation in `map/altitude.csv`

---

## Pipeline execution order

The analysis pipeline should be run in the following order:

### Step 1: Radiation model (human mobility)

```bash
cd radiation
python radiation_model.py
```
Computes the radiation model probability matrix Mij for all municipality pairs.

### Steps 2–4: Component estimation (can be run in parallel)

Steps 2, 3, and 4 are computationally independent and can be executed concurrently to save time. The R0 computation (Step 3) is the most expensive step, processing ~18 years of daily ERA5 temperature data.

### Step 2: Case importation potential (Pintro)

```bash
cd potential_introduction
python potential_intro.py
```
Estimates expected monthly imported cases for each destination municipality.

### Step 3: Thermal suitability (R0-rel)

```bash
cd R0
python compute_R0_rel.py
```
Computes the degree-days corrected relative reproductive number using ERA5 temperature data.

### Step 4: Population susceptibility proxy (S)

```bash
cd proxy_susceptibility
python compute_susceptibility.py
```
Computes the population susceptibility proxy from historical incidence.

### Step 5: Final risk score

```bash
cd risk_score
python compute_risk_score.py
```
Combines all four components into the monthly risk score for each municipality above 2,000 m.

### Step 6: Model diagnostics

```bash
cd diagnostic
python tests.py
```
Computes performance metrics: AUC-PR, AUC-ROC, recall@k%, Brier score, lift curves, and cross-validation.

### Step 7: Temporal lag optimization

```bash
cd lags
python test_lags.py
python test_lags_component_opt.py
python optimize_s3_weights.py
python lags_detailed_analysis.py
python detailed_analysis_s2_vs_base.py
Rscript test_lags_dlnm.R
```
Evaluates temporal carry-over effects through four lag strategies (S1–S3 heuristic + DLNM).

---

## Dataset and privacy

This dataset has been processed to ensure complete anonymization and contains no personally identifiable information (PII). All data has been:
- Aggregated at appropriate spatial/temporal scales
- Stripped of any individual identifiers
- Processed to remove direct or indirect identifying elements

The dataset is suitable for public sharing and complies with data privacy standards.

**What is NOT included:**
- Names, addresses, or contact information
- Individual-level identifiers
- Any data that could be used to re-identify individuals

Dengue surveillance data were retrieved from the National Surveillance System (SIVIGILA) and were fully de-identified prior to any analysis. The study received ethical approval from the Bioethics Committee of Universidad de Santander (Record No. 002, February 13, 2023).

---

## Results summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Prevalence | 2.3% | Rare events (875 case-months / 37,584 municipality-months) |
| AUC-PR | 0.042 | 1.9× random expectation |
| AUC-ROC | 0.62 | Moderate discrimination |
| Recall@10% | ~25% | Top decile captures ~1/4 of outbreak months |
| CV recall@10% | 24.0% (SD 3.8%) | Stable performance across folds |
| Spearman ρ | −0.10 | Weak negative correlation with outbreak magnitude |
| Brier Skill Score | −0.05 | Not a calibrated probability forecast |
| Cochran–Armitage | p < 0.001 | Significant monotonic trend across risk deciles |

---

## References

- Simini F, González MC, Maritan A, Barabási A-L. A universal model for mobility and migration patterns. *Nature*. 2012;484:96–100. doi:10.1038/nature10856
- Mordecai EA, Cohen JM, Evans MV, et al. Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models. *PLoS Negl Trop Dis*. 2017;11:e0005568. doi:10.1371/journal.pntd.0005568
- Mordecai EA, Caldwell JM, Grossman MK, et al. Thermal biology of mosquito-borne disease. *Ecol Lett*. 2019;22:1690–1708. doi:10.1111/ele.13335
- Childs ML, Lyberger K, Harris MJ, Burke M, Mordecai EA. Climate warming is expanding dengue burden in the Americas and Asia. *Proc Natl Acad Sci USA*. 2025;122:e2512350122. doi:10.1073/pnas.2512350122
- Tjaden NB, Thomas SM, Fischer D, Beierkuhnlein C. Extrinsic Incubation Period of Dengue: Knowledge, Backlog, and Applications of Temperature Dependence. *PLoS Negl Trop Dis*. 2013;7:e2207. doi:10.1371/journal.pntd.0002207
- Muñoz-Sabater J, Dutra E, Agustí-Panareda A, et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth Syst Sci Data*. 2021;13:4349–4383. doi:10.5194/essd-13-4349-2021
- WorldPop. WorldPop Hub [Internet]. 2020. Available: https://hub.worldpop.org/
