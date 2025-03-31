# Data Directory
This directory is intended for storing data files used in the England Environmental Justice Analysis project. The data files themselves are not included in the Git repository due to size constraints and potential licensing restrictions.

## Required Input Files

The analysis scripts in this project require the following processed datasets:

1. **Unified Air Quality and Deprivation Dataset**
   - **Filename**: `unified_dataset_with_air_quality.csv`
   - **Description**: Contains air quality metrics (NO2, O3, PM10, PM2.5), Index of Multiple Deprivation (IMD) scores, and an environmental justice index at the Lower Super Output Area (LSOA) level.
   - **Location**: `data/processed/unified_datasets/unified_dataset_with_air_quality.csv`
   - **Source**: This file is derived from multiple raw data sources (see Raw Data Sources section below)

2. **Health Indicators Dataset**
   - **Filename**: `health_indicators_by_lad.csv`
   - **Description**: Contains respiratory health indicators at the Local Authority District (LAD) level derived from NHS Outcomes Framework Indicators.
   - **Location**: `data/raw/health/health_indicators_by_lad.csv`
   - **Source**: NHS Outcomes Framework Indicators - see `NHS OF Indicators - Data Source Links v2.1.csv` for specific indicators used

3. **LSOA Boundaries**
   - **Filename**: `Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-4299016806856585929.geojson`
   - **Description**: Geospatial data for Lower Super Output Areas (LSOA) boundaries in England and Wales, December 2021.
   - **Location**: Root of data directory
   - **Source**: [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-1/about)

4. **LAD Boundaries**
   - **Filename**: `LAD_Dec_2021_GB_BFC_2022_-8975151699474964544.geojson`
   - **Description**: Geospatial data for Local Authority District (LAD) boundaries in Great Britain, December 2021.
   - **Location**: Root of data directory
   - **Source**: [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2021-gb-bfc/about)

5. **Population Data**
   - **Filename**: Referenced in `src/causal_inference_analysis.py` as `ons_lad_population_estimates.csv`
   - **Description**: Population estimates for each Local Authority District, used in the `quantify_impact` function.
   - **Location**: `data/raw/population/ons_lad_population_estimates.csv`
   - **Source**: ONS Mid-Year Population Estimates or Census 2021 data
   - **Note**: If this file is not available, the code creates a placeholder with realistic values based on ONS 2021 estimates.

## Raw Data Sources

The processed datasets were derived from the following raw data sources:

### Air Quality Data
- **Source**: DEFRA's UK-AIR database ([https://uk-air.defra.gov.uk/](https://uk-air.defra.gov.uk/))
- **Files**:
  - `NO2_Tables_2023(1).ods` - Nitrogen dioxide measurements
  - `PM25_Tables_2023(1).ods` - PM2.5 measurements
  - Various CSV files with hourly and annual measurements

### Deprivation Data
- **Source**: Department for Levelling Up, Housing and Communities (DLUHC) ([https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019))
- **Files**:
  - `File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx` - Main IMD scores
  - `File_5_-_IoD2019_Scores.xlsx` - Detailed domain scores
  - `File_6_-_IoD2019_Population_Denominators.xlsx` - Population data by LSOA
  - `File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv` - Combined IMD data

### Census Data
- **Source**: Office for National Statistics (ONS) Census 2021
- **Files**: Various Excel files with census data (RM058, RM121, TS006, etc.)

### Health Data
- **Source**: NHS Outcomes Framework ([https://www.gov.uk/government/publications/nhs-outcomes-framework-2024-to-2025](https://www.gov.uk/government/publications/nhs-outcomes-framework-2024-to-2025))
- **Files**:
  - `NHS OF Indicators - Data Source Links v2.1.csv` - Metadata about NHS indicators
  - Various NHSOF CSV files with specific health indicators

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
└── external/             # Data from third-party sources
```

## Data Processing Pipeline

The creation of the unified dataset involves several preprocessing steps:

1. **Loading raw data** from multiple sources (IMD, air quality, etc.)
2. **Cleaning and standardizing** variable names and formats
3. **Merging datasets** based on LSOA or LAD codes
4. **Calculating derived metrics** such as the environmental justice index
5. **Normalizing variables** for consistent scales across different measures

While the preprocessing code is not explicitly included in the repository, the analysis scripts in the `src/` directory contain the logic for loading and using these processed datasets.

## Notes

- Large data files (`.csv`, `.parquet`, `.geojson`, `.xlsx`, `.ods`) are excluded from version control via `.gitignore`.
- If you need to recreate the processed datasets, you will need to obtain the raw data from the sources listed above and follow the general preprocessing steps outlined.
- The `data/raw/health/NHS OF Indicators - Data Source Links v2.1.csv` file contains metadata about the specific NHS indicators used in the analysis.