# Environmental Justice Analysis in England: Addressing Health Inequalities Through Data-Driven Insights

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15147442.svg)](https://doi.org/10.5281/zenodo.15147442)

**📊 Live Demo / Project Showcase:** [**View the interactive project summary**](https://ACl365.github.io/england-environmental-justice-analysis/)

## Introduction: The Challenge & Project Aim

Environmental injustice poses a significant public health challenge in England. Socioeconomically disadvantaged communities frequently experience disproportionately high levels of air pollution (e.g., Nitrogen Dioxide (NO₂), Particulate Matter < 2.5 micrometers (PM2.5)). This unequal burden is strongly linked to adverse health outcomes, particularly respiratory conditions, deepening existing health inequalities.

This project moves beyond simple correlations to provide a rigorous, multi-method analysis of this critical nexus. **The aim is to identify specific areas and populations most vulnerable to the combined impacts of pollution and deprivation, quantify the associational link with respiratory health, and provide data-driven evidence to inform targeted, equitable policy interventions.**

## Key Findings Summary

This comprehensive analysis revealed several crucial insights enabling targeted action:

*   **Geographic Concentration:** Environmental injustice (high pollution/deprivation) exhibits significant spatial clustering (LISA/Gi*), allowing precise targeting beyond broad approaches.
*   **PM2.5 Health Association:** Higher PM2.5 exposure is significantly associated with poorer respiratory health outcomes, even after controlling for observed deprivation via Propensity Score Matching (PSM ATT ≈ -0.0399, p ≈ 0.027). This suggests reducing PM2.5 is associated with a ~4.0% relative improvement in the respiratory health index for matched Local Authority Districts (LADs).
*   **Distinct Area Profiles:** LADs cluster into distinct typologies (KMeans) with unique challenge combinations (e.g., 'Urban Deprived/Polluted'), necessitating tailored policies.
*   **Quantified Policy Impact:** Policy simulations (GBR) estimate measurable average improvements (~0.0013) in the respiratory health index from targeted PM2.5 reductions in high-priority LADs (e.g., Southend-on-Sea, Wigan, Bury, Leeds, Westminster). **Note:** Preliminary impact estimates require validation with official ONS data but indicate substantial potential (e.g., ~1.4M residents benefiting, ~£60M NHS savings from a 20% PM2.5 reduction).
*   **Deprivation Nuances:** 'Living Environment' and 'Barriers to Housing & Services' IMD domains are key correlates of pollution, highlighting specific areas for integrated interventions beyond just income/employment.

**➡️ Explore the detailed findings, visualisations, and actionable recommendations in the [Project Showcase](https://ACl365.github.io/england-environmental-justice-analysis/).**

## Analytical Approach & Method Justification

An integrated framework combining spatial, machine learning, and quasi-causal techniques was employed:

*   **Spatial Statistics (`PySAL`, `GeoPandas`):** Moran's I, LISA, Getis-Ord Gi* identify statistically significant spatial clusters (hotspots/coldspots), essential for geographic targeting. Queen contiguity weights used.
*   **Spatial Econometrics (`spreg`):** OLS and Spatial Lag models (ML-Lag) address spatial dependence that biases standard regression, providing more reliable estimates.
*   **Machine Learning (`Scikit-learn`, `SHAP`):**
    *   *`KMeans` Clustering:* Identified distinct LAD typologies efficiently. Silhouette scores guided cluster selection. Alternatives considered (see `src/advanced_cluster_analysis.py`).
    *   *Gradient Boosting Regressor (`GBR`):* Provided strong predictive performance for policy simulation, estimating intervention impacts (validated with cross-validation).
    *   *`SHAP`:* Interpreted ML models, identifying key drivers.
*   **Causal Inference (Associational - `Statsmodels`):** Propensity Score Matching (PSM) estimated the association between PM2.5 and health while controlling for *observed* confounders (IMD), approximating a quasi-experiment. Significant findings (p<0.05) with robustness checks (SMD < 0.1, Rosenbaum bounds) strengthen the evidence link. Limitations (unobserved confounders) acknowledged.
*   **Data Integration & Index Construction (`Pandas`, `GeoPandas`):** Merged multi-source datasets (ONS, DEFRA, DLUHC-IMD, NHS). Developed custom indices (`env_justice_index`, `respiratory_health_index`) to capture key concepts (rationale in `DATA_DICTIONARY.md`). Ensured geospatial integrity (EPSG:27700).

## Technology Stack

Python | Pandas | GeoPandas | NumPy | Scikit-learn | Statsmodels | PySAL (libpysal, esda, spreg) | SHAP | Matplotlib | Seaborn | Plotly

## Repository Structure

```text
england-environmental-justice-analysis/
├── assets/         # Web assets (images, CSS) for GitHub Pages showcase
├── data/           # (Gitignored - Requires manual download) Raw & Processed Data
│   ├── raw/
│   ├── processed/
│   └── geographies/
├── docs/           # Documentation (DATA_SETUP.md, Full_Technical_Report.pdf/md)
├── notebooks/      # Exploratory Jupyter notebooks (optional)
├── src/            # Main analysis source code and runner scripts
├── tests/          # Unit and integration tests
├── outputs/        # (Gitignored) Generated figures, tables, model outputs, etc.
├── .gitignore
├── CITATION.cff    # Citation information for this project
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE.txt     # Project License (MIT)
├── README.md       # This file
├── requirements.txt# Python package dependencies
└── index.html      # Root file for GitHub Pages showcase
```

## Getting Started

### Prerequisites

*   Python >= 3.10 (3.13 recommended)
*   `git` for cloning the repository
*   Ability to create a Python virtual environment (e.g., using `venv`)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ACl365/england-environmental-justice-analysis.git
    cd england-environmental-justice-analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows (Command Prompt/PowerShell):
    # venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Requirements

**⚠️ Important:** Due to size and licensing restrictions, the required datasets are **not included** in this repository. You must download them manually.

*   **Required Data:** See `docs/DATA_SETUP.md` for a detailed list, including specific sources (IMD 2019, DEFRA Air Quality, NHS Health Indicators, ONS Geographies) and required filenames.
*   **Download & Placement:** Follow the instructions in `docs/DATA_SETUP.md` carefully to download the data and place it into the correct subdirectories within the `data/` folder (e.g., `data/raw/`, `data/geographies/`).
*   **Preprocessing:** Some initial preprocessing steps outlined in `docs/DATA_SETUP.md` might be necessary before running the analysis scripts.

## Usage

Once the environment is set up and the data is correctly placed, you can run the analysis modules. Execute the runner scripts located in the `src/` directory from the **project root directory**:

```bash
# Ensure your virtual environment ('venv') is activated first!

# Example: Run the fixed spatial analysis module
python src/run_fixed_spatial_analysis.py

# Example: Run the causal inference (PSM) module
python src/run_causal_analysis.py

# Example: Run the advanced clustering and policy simulation module
python src/run_advanced_analysis.py

# Add commands for other run_*.py scripts as needed
```

Generated outputs (figures, tables, model results) will typically be saved in the `outputs/` directory (which is gitignored).

## Testing

This project uses Python's `unittest` framework for unit and integration tests and `coverage` for measuring test coverage.

*   **Goal:** Ensure code reliability, validate calculations, and verify data processing steps.
*   **Target:** Aiming for high test coverage (>80%) on core analytical functions.

To run all tests and view a coverage report:

```bash
# Ensure your virtual environment is activated

# Run tests and collect coverage data
coverage run -m unittest discover tests

# Display coverage report in the terminal
coverage report -m

# Optional: Generate an HTML report for detailed exploration
coverage html -d coverage_html
```

Refer to `tests/README.md` or individual test files for more details if needed.

## Ethical Considerations & Responsible AI

*   **Bias Awareness:** We acknowledge potential biases in source data (e.g., pollution monitor placement, health reporting variations) and models. Efforts were made to mitigate this through validated indices (IMD), sensitivity checks (PSM Rosenbaum bounds), multi-scale analysis (LSOA/LAD), and using spatial models less prone to aggregation bias.
*   **Interpretation:** Findings, particularly from PSM, represent associations controlled for *observed* confounders, not definitive causal proof due to potential unobserved factors. The ecological fallacy (drawing individual inferences from aggregate data) is a key limitation.
*   **Responsible Use:** Results should be interpreted carefully to inform equitable interventions and avoid reinforcing existing disparities. Responsible AI principles guided the analysis design and interpretation.

## MLOps & Productionisation (Conceptual)

While currently a research project, scaling this analysis for production would involve:

*   **Scalability:** Modular Python scripts aid scalability. For larger datasets or higher frequency updates, optimizing spatial weight calculations or using distributed frameworks (e.g., Dask, Spark with GeoSpark) might be necessary.
*   **Monitoring:** Implement monitoring for:
    *   *Data Drift:* Changes in input data distributions (e.g., using tools like Evidently AI).
    *   *Model Drift:* Degradation in model performance over time (e.g., tracked via MLflow).
*   **Retraining:** Establish triggers (e.g., performance decay, scheduled updates) and automate retraining pipelines using orchestration tools (e.g., Airflow, Prefect, Kubeflow Pipelines).
*   **Automation & Deployment:** Package the analysis pipeline (data ingestion, processing, modeling, reporting) potentially using Docker for consistent environments.

## Potential Generative AI / LLM Integration

Future explorations could leverage LLMs for:

*   **Contextual Enrichment:** Synthesizing policy documents or local news relevant to identified high-risk areas.
*   **Sentiment Analysis:** Analyzing public discourse on environmental issues in specific LADs (requires careful ethical consideration).
*   **Automated Reporting:** Generating tailored summaries for diverse stakeholders.
*   **Conversational Interfaces:** Creating tools for querying results interactively.

## Future Work & Potential Enhancements

*   **Longitudinal Analysis:** Incorporate time-series data to analyze trends and employ stronger quasi-experimental designs (e.g., Difference-in-Differences).
*   **Advanced Models:** Explore Geographically Weighted Regression (GWR), Graph Neural Networks (GNNs), or deep learning for spatial patterns.
*   **Granularity & Qualitative Data:** Integrate finer-grained data (e.g., hyperlocal sensors) or qualitative insights (e.g., community interviews) for richer context.
*   **Broader Scope:** Include other environmental factors (noise, green space access), health outcomes (mental health), or expand geographically.
*   **Economic Impact:** Conduct a formal health economic assessment of intervention benefits.
*   **Responsible AI Deep Dive:** Further investigate algorithmic fairness and bias mitigation techniques.

## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and adhere to the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use this project or findings in your research, please cite it using the DOI or the `CITATION.cff` file.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15147442.svg)](https://doi.org/10.5281/zenodo.15147442)

```
@software{alexander_clarke_2024_15147442,
  author       = {Alexander Clarke},
  title        = {{ACl365/england-environmental-justice-analysis:
                   Environmental Justice Analysis in England v1.0.0}},
  month        = aug,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15147442},
  url          = {https://doi.org/10.5281/zenodo.15147442}
}
```

## Author

*   **Alexander Clarke** - [ACl365](https://github.com/ACl365)
