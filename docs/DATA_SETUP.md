# Data Setup Instructions for UK Environmental Justice Analysis

This project relies on publicly available UK data. Due to size and licensing, these files are not included in the repository and must be downloaded manually from the sources below. This document outlines the necessary raw datasets, their sources, and the expected directory structure required to run the analyses in this project.

## Expected Directory Structure

All raw data should be downloaded and placed within the `data/raw/` directory, organised into subdirectories as follows:

```
UK_ENV/
└── data/
    └── raw/
        ├── air_quality/      # Air quality measurements from DEFRA
        ├── census/           # Census 2021 data from ONS
        ├── deprivation/      # IMD 2019 data from DLUHC
        ├── geographic/       # Geographic boundary files from ONS
        ├── health/           # Health outcome data from NHS
        └── population/       # Population estimates from ONS
```

## Required Raw Datasets

Please download the following datasets and place them in the corresponding subdirectories within `data/raw/`.

### 1. Air Quality Data

*   **Source:** DEFRA UK-AIR Database
*   **URL:** [https://uk-air.defra.gov.uk/](https://uk-air.defra.gov.uk/)
*   **Files Used in Original Analysis:**
    *   `NO2_Tables_2023(1).ods` - Nitrogen dioxide measurements (Year 2023).
    *   `PM25_Tables_2023(1).ods` - PM2.5 measurements (Year 2023).
    *   **Note:** The original analysis also used "Various CSV files with hourly and annual measurements" according to `data/README.md`. The specific pollutants, years (likely 2023 or the analysis period), and aggregation level (e.g., station data vs pre-aggregated LSOA/LAD tables) for these additional CSVs are not detailed in the README. If re-running preprocessing, you may need to investigate UK-AIR for relevant hourly/annual data for pollutants like NO2, PM2.5, O3, and PM10 for the desired time period. Using the specific tables listed above is recommended for consistency if possible. Later versions of these tables may work but have not been tested with this project's scripts.
*   **Location:** `data/raw/air_quality/`

### 2. Deprivation Data (IMD 2019)

*   **Source:** Department for Levelling Up, Housing and Communities (DLUHC)
*   **URL:** [https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
*   **Files:**
    *   `File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx`
    *   `File_5_-_IoD2019_Scores.xlsx`
    *   `File_6_-_IoD2019_Population_Denominators.xlsx`
    *   `File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv`
*   **Location:** `data/raw/deprivation/`

### 3. Geographic Boundaries

*   **Source:** ONS Open Geography Portal
*   **Files & URLs (December 2021 versions used):**
    *   **LSOA Boundaries:** `Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_*.geojson` (Download the GeoJSON version)
        *   URL: [https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-1/about](https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-1/about)
    *   **LAD Boundaries:** `LAD_Dec_2021_GB_BFC_2022_*.geojson` (Download the GeoJSON version)
        *   URL: [https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-gb-bfc/about](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-gb-bfc/about)
*   **Location:** `data/raw/geographic/`
    *   *Note: Ensure you download the specific December 2021 boundary files referenced or their direct equivalents.*

### 4. Health Data

*   **Source:** NHS Outcomes Framework (Now potentially NHS England Indicator Library)
*   **URL:** The original link pointed to the 2024/25 framework: [https://www.gov.uk/government/publications/nhs-outcomes-framework-2024-to-2025](https://www.gov.uk/government/publications/nhs-outcomes-framework-2024-to-2025). You may need to search the NHS England website for the specific indicators.
*   **Files:**
    *   `NHS OF Indicators - Data Source Links v2.1.csv` (or latest equivalent metadata file, if available). This file, mentioned in `data/README.md`, contains metadata about the indicators used.
    *   **Specific Indicator CSVs:** The `data/README.md` mentions "Various NHSOF CSV files with specific health indicators" were used. Unfortunately, it does not list the exact filenames of these indicator CSVs. To replicate the setup, you would need to:
        1.  Consult the `NHS OF Indicators - Data Source Links v2.1.csv` metadata file (if you can locate it or a similar file).
        2.  Alternatively, examine the analysis scripts (e.g., in `src/`) to identify which specific health indicators are loaded.
        3.  Search the NHS indicator library/website for the data corresponding to those indicators (likely requiring individual downloads).
*   **Location:** `data/raw/health/`

### 5. Population Data

*   **Source:** ONS Mid-Year Population Estimates or Census 2021
*   **URL:** Search the ONS website. The specific table ID used for the original analysis is not listed in `data/README.md`. Look for Local Authority District level population estimates (e.g., search for "MYE" or "SAPE" tables). The year used likely corresponds to the analysis period (e.g., 2021 or 2023).
*   **File:** The analysis expects a file named `ons_lad_population_estimates.csv`. You may need to rename the downloaded ONS file or adjust the loading code.
*   **Location:** `data/raw/population/`
    *   *Note: The analysis script `src/causal_inference_analysis.py` might generate a placeholder if this file is missing, but using actual ONS data is recommended.*

### 6. Census Data (Optional - Context for Preprocessing)

*   **Source:** ONS Census 2021
*   **URL:** Search the ONS website for Census 2021 datasets.
*   **Files:** The original preprocessing likely used various tables. Examples mentioned in `data/README.md` include `RM058`, `RM121`, `TS006`. These are primarily needed if attempting to fully replicate the preprocessing steps that generated the unified datasets.
*   **Location:** `data/raw/census/`

---

Once all required raw data is downloaded and placed in the correct directories, the analysis scripts should be able to locate and use them (or the processed versions derived from them). Details on the processed files generated by the analysis scripts can be found in the project's main README or Data Dictionary, if available.