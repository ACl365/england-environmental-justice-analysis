# Data Acquisition Guide

This document provides detailed instructions for acquiring and setting up the datasets required for the Environmental Justice Analysis in England project.

## Required Datasets

The following datasets are required to run the analysis scripts:

### 1. Unified Air Quality and Deprivation Dataset

- **Filename:** `unified_dataset_with_air_quality.csv`
- **Description:** Contains air quality metrics (NO2, O3, PM10, PM2.5), Index of Multiple Deprivation (IMD) scores, and environmental justice indices at LSOA level
- **Where to place:** `data/processed/unified_datasets/unified_dataset_with_air_quality.csv`
- **How to obtain:**
  - Download raw data from DEFRA UK-AIR: https://uk-air.defra.gov.uk/data/data-selector
    - Select "Annual Mean" for NO2, PM2.5, PM10, and O3
    - Select the most recent complete year
    - Download as CSV format
  - Download IMD 2019 data from DLUHC: https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
    - Specifically download "File 7: all ranks, deciles and scores for the indices of deprivation"
  - Process these files using the methodology described in the Data Processing Pipeline section below

### 2. Health Indicators Dataset

- **Filename:** `health_indicators_by_lad.csv`
- **Description:** Contains respiratory health indicators at Local Authority District (LAD) level
- **Where to place:** `data/raw/health/health_indicators_by_lad.csv`
- **How to obtain:**
  - Access the NHS Outcomes Framework: https://digital.nhs.uk/data-and-information/publications/statistical/nhs-outcomes-framework
  - Download indicators 2.3.i and 2.3.ii (emergency admissions for respiratory conditions)
  - Download as CSV format

### 3. LSOA Boundaries

- **Filename:** `Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.geojson`
- **Description:** Geospatial data for LSOA boundaries in England and Wales (December 2021)
- **Where to place:** `data/raw/geographic/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4.geojson`
- **How to obtain:**
  - Download directly from ONS Open Geography Portal: https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-1/about
  - Select GeoJSON format for download

### 4. LAD Boundaries

- **Filename:** `LAD_Dec_2021_GB_BFC_2022.geojson`
- **Description:** Geospatial data for LAD boundaries in Great Britain (December 2021)
- **Where to place:** `data/raw/geographic/LAD_Dec_2021_GB_BFC_2022.geojson`
- **How to obtain:**
  - Download directly from ONS Open Geography Portal: https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-gb-bfc/about
  - Select GeoJSON format for download

### 5. Population Data

- **Filename:** `ons_lad_population_estimates.csv`
- **Description:** Population estimates for each Local Authority District
- **Where to place:** `data/raw/population/ons_lad_population_estimates.csv`
- **How to obtain:**
  - Download from ONS: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates
  - Select "Mid-year population estimates" for the most recent year
  - Download as CSV format
- **Note:** If this file is not available, the code creates a placeholder with realistic values based on ONS 2021 estimates

## Sample Data

For convenience, a small sample dataset is provided in the `data/sample/` directory. This contains:

- `sample_unified_dataset.csv`: 10 rows from the unified dataset
- `sample_health_indicators.csv`: 5 rows from the health indicators dataset
- `sample_lsoa_boundaries.geojson`: A simplified version with 10 LSOA boundaries

These sample files allow you to test parts of the code without downloading the full datasets, though complete analysis requires the full datasets.

## Data Organization

The data directory is organized as follows:

```
data/
├── raw/                  # Original, immutable data
│   ├── air_quality/      # Air quality measurements from DEFRA
│   ├── census/           # Census 2021 data from ONS
│   ├── deprivation/      # IMD 2019 data from DLUHC
│   ├── geographic/       # Geographic boundary files
│   ├── health/           # Health outcome data from NHS
│   └── population/       # Population estimates from ONS
├── processed/            # Cleaned, transformed data ready for analysis
│   ├── unified_datasets/ # Combined datasets
│   └── spatial/          # Processed spatial data
├── sample/               # Small sample datasets for testing
└── external/             # Data from third-party sources
```

## Data Processing Pipeline

The creation of the unified dataset involves several preprocessing steps:

1. **Loading raw data** from multiple sources (IMD, air quality, etc.)
2. **Cleaning and standardising** variable names and formats
3. **Merging datasets** based on LSOA or LAD codes
4. **Calculating derived metrics** such as the environmental justice index
5. **Normalising variables** for consistent scales across different measures

While the preprocessing code is not explicitly included in the repository, the analysis scripts in the `src/` directory contain the logic for loading and using these processed datasets.

## Data Preparation Checklist

Before running the analysis, ensure you have:

- [ ] Downloaded all required datasets
- [ ] Placed files in the correct directory structure
- [ ] Verified file formats (CSV, GeoJSON)
- [ ] Created necessary directories if they don't exist
- [ ] Checked sample data to understand expected format

## Troubleshooting Common Data Issues

- **Missing geographic codes**: Ensure LSOA/LAD codes are consistent across datasets
- **Projection issues**: Geographic data should use EPSG:27700 (British National Grid)
- **Encoding problems**: CSV files should use UTF-8 encoding
- **Date format inconsistencies**: Standardise date formats before merging datasets