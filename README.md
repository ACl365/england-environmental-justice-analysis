# Environmental Justice Analysis in England: Addressing Health Inequalities Through Data-Driven Insights

<!-- [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

**📊 Live Demo / Project Showcase:** [View the interactive project summary](https://ACl365.github.io/england-environmental-justice-analysis/)

## Introduction: The Challenge & Project Aim

Environmental injustice poses a significant public health challenge in England, where socioeconomically disadvantaged communities frequently experience disproportionately high levels of air pollution (e.g., NO₂, PM2.5). This unequal burden is strongly linked to adverse health outcomes, particularly respiratory conditions, deepening existing health inequalities.

This project moves beyond simple correlations to provide a rigorous, multi-method analysis of this critical nexus. **The aim is to identify specific areas and populations most vulnerable to the combined impacts of pollution and deprivation, quantify the associational link with respiratory health, and provide data-driven evidence to inform targeted, equitable policy interventions.**

## Key Findings Summary

This comprehensive analysis revealed several crucial insights into environmental justice and health inequalities in England:

*   **Geographic Concentration:** Environmental injustice (combined high pollution and deprivation) is not randomly distributed but exhibits significant spatial clustering, particularly in major urban centres and specific post-industrial regions. (LISA/Gi* analysis).
*   **NO₂ Health Association:** After controlling for area-level deprivation using Propensity Score Matching, higher NO₂ exposure was found to have a statistically significant negative association with respiratory health outcomes (ATT ≈ -0.039, p<0.05).
*   **Distinct Area Profiles:** Local Authority Districts cluster into distinct typologies facing different combinations of environmental, social, and health challenges, highlighting the need for tailored, place-based policy rather than one-size-fits-all approaches (KMeans Clustering & SHAP).
*   **Policy Simulation Potential:** Predictive modelling indicated that targeted reductions in NO₂, particularly in identified high-impact areas (e.g., Derby, Trafford), offer substantial potential for improving average respiratory health indicators (GBR Simulation).
*   **Deprivation Nuances:** Specific deprivation domains, notably 'Living Environment Deprivation' and 'Barriers to Housing & Services', showed stronger correlations with pollution than income or employment alone, suggesting key areas for integrated intervention.

**➡️ Explore the detailed findings and visualisations in the [Project Showcase](https://ACl365.github.io/england-environmental-justice-analysis/).**

## Analytical Approach Highlights

An integrated framework combining advanced techniques was employed:

*   **Spatial Statistics (PySAL, GeoPandas):** Moran's I (Global/Local), LISA, Getis-Ord Gi* to identify spatial patterns and statistically significant clusters/hotspots. Queen contiguity weights used for administrative boundaries.
*   **Machine Learning (Scikit-learn, SHAP):** Unsupervised Clustering (KMeans) to identify area typologies, Predictive Modelling (Gradient Boosting Regressors) for policy simulation, and SHAP for model interpretability.
*   **Causal Inference (Associational - Statsmodels):** Propensity Score Matching (PSM) with rigorous diagnostics (SMD < 0.1, Rosenbaum Bounds) to estimate the association between pollution and health while controlling for observed socioeconomic confounders.
*   **Spatial Econometrics (spreg):** OLS and Spatial Lag models (ML_Lag) to investigate relationships at the LAD level, accounting for spatial dependence (neighbourhood effects).
*   **Data Integration & Geospatial Handling:** Meticulous merging, cleaning, and validation of complex multi-source datasets across LSOA and LAD scales, ensuring geospatial integrity (EPSG:27700).

## Technology Stack

Python | Pandas | GeoPandas | NumPy | Scikit-learn | Statsmodels | PySAL (libpysal, esda, spreg) | SHAP | Matplotlib | Seaborn | Plotly

## Repository Structure

```
UK_ENV/
├── assets/ # (Not gitignored) Web assets (key images, CSS) for GitHub Pages
│ └── images/
├── data/ # (Gitignored - requires manual download) Raw & Processed Data
│ ├── raw/
│ ├── processed/
│ └── geographies/
├── docs/ # Supplementary documentation (e.g., DATA_SETUP.md)
├── notebooks/ # Exploratory Jupyter notebooks (optional)
├── src/ # Main analysis source code and runner scripts
├── tests/ # Unit and integration tests
├── .gitignore
├── LICENSE # Project License
├── README.md # This file
├── requirements.txt
├── outputs/ # (Gitignored) Full set of generated outputs (figures, tables, etc.)
└── index.html # HTML file for GitHub Pages showcase
```

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ACl365/england-environmental-justice-analysis.git
    cd england-environmental-justice-analysis
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Acquire Data:** Download the required datasets as detailed in `docs/DATA_SETUP.md` (you will need to create this file with the detailed instructions) and place them in the `data/` directory structure.

## Usage

Run specific analysis modules using the scripts in `src/`. Execute from the project root directory:

```bash
# Example: Run the fixed spatial analysis
python src/run_fixed_spatial_analysis.py

# Example: Run the causal inference analysis
python src/run_causal_analysis.py

# Add commands for other run_*.py scripts as needed
```
(Ensure your virtual environment is activated)

## Testing Strategy

This project employs unit and integration tests using Python's unittest framework and coverage for measuring test coverage.

Goal: Ensure code reliability, validate calculations, and verify data processing steps.

Coverage: Aiming for high coverage (>80%) on core analytical functions.

To run all tests and generate a coverage report:

```bash
# Run tests with coverage
coverage run -m unittest discover tests

# View coverage report
coverage report -m

# Optional: Generate HTML report
coverage html -d coverage_html
```

Refer to tests/run_tests.py for more options.

## Data Requirements

This analysis relies on publicly available UK datasets which are not included in this repository due to size. Users must download the required data files independently.

Required Data: IMD 2019 (England), DEFRA Air Quality estimates, NHS Health Indicators (England), ONS LSOA/LAD Geographic Boundaries (England & Wales / Great Britain).

Placement: Data should be placed within the data/ directory structure (data/raw/, data/geographies/).

Detailed Instructions: Please refer to docs/DATA_SETUP.md for specific file names, download links, and placement instructions. (You need to create this file).

Sample Data: Small sample files may be available in data/sample/ for basic testing (check repository).

## Advanced Considerations & Future Directions

Ethical Considerations & Bias Mitigation: Analysis acknowledges potential biases (monitoring placement, aggregation, temporal) and includes mitigation strategies where possible (multi-scale analysis, spatial models, reporting uncertainty). See code/docs for details. Interpretation must consider these limitations to ensure equitable application.

MLOps & Deployment Strategy: The modular structure facilitates integration into an MLOps pipeline (e.g., using Airflow, MLflow, Docker) for automated data ingestion, retraining, monitoring, and potential API deployment for dynamic reporting.

Generative AI Integration Potential: LLMs could enhance analysis by synthesising policy documents, analysing community sentiment from text data, generating automated report summaries, or providing conversational interfaces for querying results.

Future Work: Potential extensions include longitudinal analysis (requiring time-series data), incorporating more granular health/demographic data, exploring advanced spatial models (GWR, GNNs), integrating qualitative research, and expanding the scope to include other environmental factors or UK nations.

## Contributing

Contributions are welcome. Please refer to CONTRIBUTING.md and adhere to the CODE_OF_CONDUCT.md.

## Licence

This project is licensed under the MIT Licence - see the LICENSE file for details.

## Author

Alexander Clarke

