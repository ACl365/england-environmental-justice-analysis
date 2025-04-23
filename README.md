# Environmental Justice Analysis in England: Addressing Health Inequalities Through Data-Driven Insights

<!-- Optional: Add badges for status, license, etc. -->
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15147442.svg)](https://doi.org/10.5281/zenodo.15147442)


**📊 Live Demo / Project Showcase:** [View the interactive project summary](https://ACl365.github.io/england-environmental-justice-analysis/)


## Introduction: The Challenge & Project Aim

Environmental injustice poses a significant public health challenge in England, where socioeconomically disadvantaged communities frequently experience disproportionately high levels of air pollution (e.g., Nitrogen Dioxide (NO₂), Particulate Matter < 2.5 micrometers (PM2.5)). This unequal burden is strongly linked to adverse health outcomes, particularly respiratory conditions, deepening existing health inequalities.

This project moves beyond simple correlations to provide a rigorous, multi-method analysis of this critical nexus. **The aim is to identify specific areas and populations most vulnerable to the combined impacts of pollution and deprivation, quantify the associational link with respiratory health, and provide data-driven evidence to inform targeted, equitable policy interventions.**

## Key Findings Summary

This comprehensive analysis revealed several crucial insights enabling targeted action:

*   **Geographic Concentration:** Environmental injustice (high pollution/deprivation) exhibits significant spatial clustering (Local Indicators of Spatial Association (LISA)/Getis-Ord Gi* (Gi*)), allowing precise targeting beyond broad approaches.
*   **PM2.5 Health Association:** Higher PM2.5 exposure is significantly associated with poorer respiratory health outcomes, even after controlling for observed deprivation via Propensity Score Matching (PSM) Average Treatment Effect on the Treated (ATT) ≈ -0.0399, p ≈ 0.027). This suggests reducing PM2.5 is associated with a ~4.0% relative improvement in the respiratory health index for matched Local Authority Districts (LADs).
*   **Distinct Area Profiles:** LADs cluster into distinct typologies (KMeans) with unique challenge combinations (e.g., 'Urban Deprived/Polluted'), necessitating tailored policies.
*   **Quantified Policy Impact:** Policy simulations (Gradient Boosting Regressor (GBR)) estimate measurable average improvements (~0.0013) in the respiratory health index from targeted PM2.5 reductions in high-priority LADs (e.g., Southend-on-Sea, Wigan, Bury, Leeds, Westminster, Eastbourne, Portsmouth, Salford, Manchester, Blaby). Preliminary quantification suggests potentially benefiting ~1.4M residents in areas with above-median improvement for a 20% reduction (requires validation with official Office for National Statistics (ONS) data).
*   **Deprivation Nuances:** 'Living Environment' and 'Barriers to Housing & Services' deprivation domains (from the Index of Multiple Deprivation (IMD)) are key correlates of pollution, highlighting specific areas for integrated interventions beyond just income/employment.

**➡️ Explore the detailed findings, visualisations, and actionable recommendations in the [Project Showcase](https://ACl365.github.io/england-environmental-justice-analysis/).**

## Analytical Approach & Method Justification

An integrated framework combining spatial, machine learning, and quasi-causal techniques was employed, with careful methodological choices:

*   **Spatial Statistics (PySAL, GeoPandas):** Used Moran's I, Local Indicators of Spatial Association (LISA), Getis-Ord Gi* (Gi*) to identify spatial patterns. **Justification:** Essential for geographic data where observations aren't independent; identifies statistically significant local clusters (hotspots/coldspots) crucial for targeting. Queen contiguity weights chosen for administrative areas.
*   **Spatial Econometrics (spreg):** Employed Ordinary Least Squares (OLS) and Spatial Lag models (Maximum Likelihood Spatial Lag (ML-Lag)). **Justification:** Addresses spatial dependence (neighbourhood effects) often present in geographic data, which can bias standard OLS regression. The spatial lag model explicitly accounts for this (significance of rho coefficient indicates spatial dependence).
*   **Machine Learning (Scikit-learn, SHapley Additive exPlanations (SHAP)):**
    *   *KMeans Clustering:* Used to identify LAD typologies. **Justification:** Chosen for efficiency and interpretability (via centroids) on this dataset. Alternatives (DBSCAN, GMM - see `src/advanced_cluster_analysis.py`) were considered, but KMeans provided clearer separation into policy-relevant groups here. Silhouette scores guided cluster selection.
    *   *Gradient Boosting Regressor (GBR):* Used for policy simulation. **Justification:** Strong predictive performance, handles non-linearities/interactions, robust to outliers. Allows quantitative estimation of intervention impacts (e.g., PM2.5 reduction on health index). Validated with cross-validation (R²/MSE).
    *   *SHAP:* Used for interpreting ML models (esp. GBR), identifying key drivers and their impact.
*   **Causal Inference (Associational - Statsmodels):** Applied Propensity Score Matching (PSM). **Justification:** Pragmatic approach for observational data to estimate the association between pollutants (PM2.5) and health while controlling for *observed* confounders (IMD domains), approximating a quasi-experiment. The analysis found a statistically significant association for PM2.5 (p<0.05), providing stronger evidence for its link to respiratory health independent of measured deprivation. Acknowledges limitations (unobserved confounders) but uses rigorous diagnostics (Standardised Mean Difference (SMD) < 0.1, Rosenbaum bounds) to assess robustness.
*   **Data Integration & Index Construction (Pandas, GeoPandas):** Merged multi-source datasets (ONS, Department for Environment, Food & Rural Affairs (DEFRA), Department for Levelling Up, Housing and Communities (DLUHC)-IMD, National Health Service (NHS)). Developed custom indices (`env_justice_index`, `respiratory_health_index`) with specific rationales (see `DATA_DICTIONARY.md`) to capture 'double burden' and health concepts effectively. Ensured geospatial integrity (EPSG:27700).

## Technology Stack

Python | Pandas | GeoPandas | NumPy | Scikit-learn | Statsmodels | PySAL (libpysal, esda, spreg) | SHAP | Matplotlib | Seaborn | Plotly

## Repository Structure
```text
england-environmental-justice-analysis/
├── assets/ # (Not gitignored) Web assets (key images, CSS) for GitHub Pages
│   └── images/
├── data/ # (Gitignored - requires manual download) Raw & Processed Data
│   ├── raw/
│   ├── processed/
│   └── geographies/
├── docs/ # Supplementary documentation (e.g., DATA_SETUP.md, Full_Technical_Report.pdf/md)
├── notebooks/ # Exploratory Jupyter notebooks (optional)
├── src/ # Main analysis source code and runner scripts
├── tests/ # Unit and integration tests
├── .gitignore
├── LICENSE.txt # Project License
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
4.  **Acquire Data:** Download the required datasets as detailed in `docs/DATA_SETUP.md` and place them in the correct `data/` directory structure.

## Usage

Run specific analysis modules using the scripts in `src/`. Execute from the project root directory:

```bash
# Example: Run the fixed spatial analysis
python src/run_fixed_spatial_analysis.py

# Example: Run the causal inference analysis
python src/run_causal_analysis.py

# Example: Run the advanced clustering analysis
python src/run_advanced_analysis.py

# Add commands for other run_*.py scripts as needed
Use code with caution.
(Ensure your virtual environment is activated)

Testing Strategy
This project employs unit and integration tests using Python's unittest framework and coverage for measuring test coverage.

Goal: Ensure code reliability, validate calculations, and verify data processing steps.

Coverage: Aiming for high coverage (>80%) on core analytical functions.

To run all tests and generate a coverage report:

# Run tests with coverage
coverage run -m unittest discover tests

# View coverage report
coverage report -m

# Optional: Generate HTML report
coverage html -d coverage_html
Use code with caution.
Bash
Refer to tests/run_tests.py for more options.

Data Requirements
This analysis relies on publicly available UK datasets which are not included in this repository due to size and licensing restrictions. Users must download the required data files independently.

Required Data: IMD 2019 (England), DEFRA Air Quality estimates, NHS Health Indicators (England), ONS Lower Layer Super Output Area (LSOA)/LAD Geographic Boundaries (England & Wales / Great Britain).

Placement: Data should be placed within the data/ directory structure (e.g., data/raw/, data/geographies/).

Detailed Instructions: Please refer to `docs/DATA_SETUP.md` for detailed instructions on data acquisition, including specific filenames, download links, and required processing steps.

## Advanced Considerations, MLOps, & Future Directions

### Ethical Considerations & Responsible AI
*   **Bias Awareness:** Acknowledges potential biases in source data (e.g., pollution monitor placement, health reporting variations) and models.
*   **Mitigation:** Used validated indices (IMD), performed sensitivity checks (PSM Rosenbaum bounds), analysed multiple scales (LSOA/LAD), and employed spatial models less prone to aggregation bias than simple averages.
*   **Interpretation:** Emphasises associational findings (PSM) and ecological fallacy limitations. Stresses the need for careful interpretation to avoid reinforcing inequities when translating findings into interventions. Responsible AI principles were considered throughout.

### MLOps & Productionisation Strategy (Conceptual)
*   **Scalability:** Analysis designed with modular Python scripts. For larger datasets (e.g., full UK, finer time scales), bottlenecks in spatial weight calculation or PSM might require optimisation or distributed computing frameworks (e.g., Dask, Spark).
*   **Monitoring & Retraining:** A production system would require monitoring for:
    *   *Data Drift:* Changes in input data distributions (pollution levels, demographics). Tools like Evidently AI could be used.
    *   *Model Drift:* Changes in the relationships learned by models (e.g., GBR, PSM). Tools like MLflow could track performance.
    *   *Retraining:* Triggered by significant drift or on a regular schedule (e.g., annually with new data releases), managed via an orchestration tool.
*   **Automation:** The `src/run_*.py` scripts form the basis of an automated pipeline (e.g., using Airflow, Prefect, Kubeflow Pipelines) for data ingestion, processing, analysis, and reporting. Docker containerisation (see `Dockerfile`) supports consistent deployment.

### Generative AI / LLM Integration Potential
*   **Contextual Enrichment:** Use LLMs to synthesise relevant policy documents, academic literature, or news reports related to specific LADs or clusters.
*   **Sentiment Analysis:** Analyse local news or social media (with ethical considerations) to understand community perspectives on environmental issues.
*   **Automated Reporting:** Generate tailored summaries of findings for different audiences (policymakers, community groups).
*   **Conversational Interfaces:** Develop chatbots allowing users to query analysis results (e.g., "Show me high-risk LSOAs in Cluster 3").

### Future Work & Enhancements
*   **Longitudinal Analysis:** Incorporate time-series data for trend analysis and stronger quasi-experimental designs (Difference-in-Differences, Interrupted Time Series).
*   **Advanced Spatial/ML Models:** Explore Geographically Weighted Regression (GWR), Graph Neural Networks (GNNs), or deep learning for more complex spatial patterns and interactions.
*   **Granularity & Qualitative Data:** Integrate finer-grained data (hyperlocal sensors, individual surveys - if feasible/ethical) and complement with qualitative research (community interviews) for richer context.
*   **Broaden Scope:** Include other environmental factors (noise, green space access), health outcomes (mental health), or expand geographically (other UK nations).
*   **Economic Impact Analysis:** Conduct a more formal health economic assessment of potential NHS cost savings from interventions.
*   **Responsible AI Deep Dive:** Further investigate algorithmic fairness and bias, exploring advanced mitigation techniques.

Contributing
Contributions are welcome. Please refer to CONTRIBUTING.md and adhere to the CODE_OF_CONDUCT.md.

License
This project is licensed under the MIT License - see the LICENSE.txt file for details. (Ensure you add a LICENSE file)

Author
Alexander Clarke
