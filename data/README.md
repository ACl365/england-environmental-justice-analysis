# Data Directory

This directory is intended for storing data files used in the UK Environmental Justice Analysis project. The data files themselves are not included in the Git repository due to size constraints and potential licensing restrictions.

## Required Datasets

The project requires the following datasets:

1. **Unified Air Quality and Deprivation Dataset**
   - Description: Contains air quality metrics, Index of Multiple Deprivation (IMD) scores, and an environmental justice index at the Lower Super Output Area (LSOA) level.
   - Source: [Add source information or download instructions here]
   - Recommended filename: `unified_air_quality_deprivation.csv`

2. **Health Indicators Dataset**
   - Description: Contains respiratory health indicators at the Local Authority District (LAD) level derived from NHS Outcomes Framework Indicators.
   - Source: NHS Digital - [Add specific link or download instructions]
   - Recommended filename: `health_indicators.csv`

3. **UK Ward Boundaries**
   - Description: Geospatial data for UK ward boundaries.
   - Source: Ordnance Survey or ONS Geography Portal
   - Filename: `Wards_December_2024_Boundaries_UK_BFC_7247148252775165514.geojson`

## Data Organization

Once obtained, place the datasets in this directory with the following structure:

```
data/
├── raw/                  # Original, immutable data
│   ├── air_quality/      # Air quality measurements
│   ├── deprivation/      # IMD and domain data
│   └── health/           # Health outcome data
├── processed/            # Cleaned, transformed data ready for analysis
│   ├── unified_datasets/ # Combined datasets
│   └── spatial/          # Processed spatial data
└── external/             # Data from third-party sources
```

## Data Processing

The data processing scripts in the `src/` directory expect these datasets to be available in this directory. If you place the data files in different locations or use different filenames, you may need to update the file paths in the scripts.

## Notes

- Large data files (`.csv`, `.parquet`, `.geojson`, `.xlsx`) are excluded from version control via `.gitignore`.
- For reproducibility, consider documenting the exact data sources, versions, and any preprocessing steps in your analysis notebooks.