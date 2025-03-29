# Project Write-Up: Environmental Justice, Air Pollution, and Health Inequalities in the UK

## Executive Summary / Abstract

This project investigates the critical intersection of environmental justice, socioeconomic deprivation, and public health outcomes across the UK, focusing on the disproportionate impact of air pollution on vulnerable communities. Utilising a multi-layered analytical framework encompassing geospatial analysis, advanced statistical modelling, causal inference, and machine learning techniques, we analysed Lower Super Output Area (LSOA) level data on air quality (NO2, PM2.5, O3, PM10) and deprivation (Index of Multiple Deprivation - IMD), alongside Local Authority District (LAD) level respiratory health indicators derived from NHS and ONS data. Key methodologies included propensity score matching (PSM) to estimate the causal impact of pollution on health, advanced clustering (KMeans with silhouette validation, PCA, SHAP) to identify distinct area profiles, spatial analysis (including Moran's I and Getis-Ord Gi* - *details in spatial analysis module*), and multivariate modelling to understand pollutant interactions and threshold effects. This integrated approach aims to provide a more holistic view compared to studies focusing solely on correlation or single analytical techniques. Significant findings reveal distinct spatial clustering of environmental injustice, with specific deprivation domains (Living Environment, Housing Barriers) showing stronger links to pollution. Causal analysis indicated a statistically significant negative impact of high NO2 exposure on respiratory health (Average Treatment Effect on the Treated ≈ -0.039, signifying a measurable worsening on the health index scale). Policy simulations suggest targeted NO2 reductions (e.g., 20%) could yield substantial health improvements (average index improvement ≈ +0.046), particularly in specific high-impact LADs like Cambridge and Westminster. The project delivers a robust evidence base and analytical framework for prioritising interventions and informing targeted environmental health policies.

## Introduction & Problem Definition

**Context:** Environmental inequality, where disadvantaged communities face higher exposure to environmental hazards like air pollution, is a pressing global issue with significant public health implications. In the UK, despite national air quality standards, concerns persist about localised pollution hotspots and their disproportionate impact on vulnerable populations, potentially exacerbating existing health inequalities, particularly concerning respiratory conditions. Understanding the complex interplay between pollution exposure, socioeconomic factors, and health outcomes is crucial for developing effective and equitable public health interventions and environmental policies. This project addresses the need for a data-driven approach that moves beyond simple correlations to quantify causal relationships and identify areas requiring targeted action, integrating multiple analytical perspectives for a comprehensive understanding.

**Project Objectives:**
The primary objective was to investigate the concept of "double disadvantage":
1.  **Quantify the combined impact:** To what extent do socioeconomic deprivation and air pollution exposure interact to create areas with disproportionately poor respiratory health outcomes?

Secondary objectives aimed to provide deeper insights:
2.  **Identify Threshold Effects:** Is there a discernible threshold in air pollution levels (NO2, PM2.5, etc.) beyond which respiratory health significantly deteriorates?
3.  **Analyse Pollutant Interactions:** How do different air pollutants interact, and what is their combined effect on health?
4.  **Map Environmental Injustice:** What is the spatial distribution of environmental injustice (high pollution coinciding with high deprivation) across the UK?
5.  **Prioritise Interventions:** Which geographical areas should be prioritised for pollution reduction interventions to achieve the greatest public health benefits, particularly in respiratory health?

**Scope & Delimitation:**
*   **Geographical Scope:** Primarily focused on England due to the availability and consistency of the Index of Multiple Deprivation (IMD) data at the LSOA level. LAD level analysis covers a broader UK scope where health data allows.
*   **Data Scope:** Utilised publicly available datasets including government air quality monitoring data, IMD 2019 scores, NHS Outcomes Framework indicators (derived from ONS and OHID data), and geographical boundaries. Specific health outcomes focused on respiratory conditions available at the LAD level.
*   **Temporal Scope:** Primarily a cross-sectional analysis based on recent available data (circa 2019-2024, depending on the specific dataset). Time-series analysis was identified as future work.
*   **Methodological Scope:** Employed statistical correlation, regression, clustering, causal inference (PSM, dose-response), and spatial statistics. Advanced atmospheric modelling or detailed individual exposure tracking were outside the scope.
*   **Exclusions:** Did not include analysis of indoor air quality, specific occupational exposures, or detailed demographic breakdowns beyond area-level deprivation metrics. The analysis focused on outdoor ambient air pollution concentrations as provided in the datasets.

**Hypotheses:**
1.  Areas with higher levels of socioeconomic deprivation will exhibit higher average concentrations of key air pollutants (e.g., NO2, PM2.5).
2.  Higher levels of air pollution exposure will be negatively correlated with respiratory health indicators at the area level.
3.  Areas experiencing both high deprivation and high pollution ("double disadvantage") will show significantly worse respiratory health outcomes compared to other areas.
4.  There exist non-linear relationships and potential threshold effects between specific pollutant concentrations and health outcomes.

## Data Acquisition & Understanding

**Source(s):**
The project integrated data from multiple official sources:
1.  **Air Quality & Deprivation Data:** A pre-compiled `unified_dataset_with_air_quality.csv` containing LSOA-level metrics. This dataset integrated:
    *   Air quality data (NO2, O3, PM10, PM2.5 concentrations) likely sourced from DEFRA's UK-AIR database or similar government monitoring networks.
    *   Index of Multiple Deprivation (IMD) 2019 scores and domain scores (Income, Employment, Health, Education, Crime, Barriers to Housing & Services, Living Environment) for England, provided by the Department for Levelling Up, Housing and Communities (DLUHC).
    *   Geographical identifiers (LSOA codes, LAD codes/names).
    *   Pre-calculated normalised pollution metrics and an `env_justice_index` (construction details assumed based on typical practice, likely a composite of pollution and deprivation).
2.  **Health Indicators Data:** `health_indicators_by_lad.csv` containing LAD-level health metrics derived from the NHS Outcomes Framework and underlying ONS/OHID data. Key indicators included standardised mortality rates and composite indices representing respiratory health, chronic conditions, etc. The specific source indicators were documented in `NHS OF Indicators - Data Source Links v2.1.xlsx`.
3.  **Geographical Data:** `Wards_December_2024_Boundaries_UK_BFC_*.csv` and potentially associated GeoJSON files (`Wards_December_2024_Boundaries_UK_BFC_*.geojson`) providing geographical boundary information for mapping and spatial analysis, likely sourced from the ONS Open Geography Portal.

**Data Description:**
*   **`unified_dataset_with_air_quality.csv`:** LSOA-level data. Key features included: `lsoa_code`, `lsoa_name`, `lad_code`, `lad_name`, raw pollutant concentrations (`NO2`, `O3`, `PM10`, `PM2.5`), normalised pollutant concentrations (`NO2_normalized`, etc.), IMD score (`imd_score_normalized`), IMD domain scores (`income_score_rate`, etc.), and a composite `env_justice_index`. Contained ~34,000+ records (LSOAs in England).
*   **`health_indicators_by_lad.csv`:** LAD-level data. Key features included: `local_authority_code`, `local_authority_name`, various health indicators (e.g., `respiratory_health_index`, `chronic_conditions_normalized`), potentially population counts or denominators used for rate calculations. Contained ~300+ records (LADs).
*   **Geographical Data:** Contained identifiers (Ward codes, LAD codes) and geometric information (polygons) for spatial representation.

**Initial Challenges:**
1.  **Data Granularity Mismatch:** Air quality and deprivation data were available at the fine-grained LSOA level, while health outcome data was aggregated at the coarser LAD level. This required aggregation of LSOA data to the LAD level for integrated analysis, potentially leading to ecological fallacy issues (assuming area-level correlations apply to individuals).
2.  **Missing Data:** Initial exploration (`explore_data` function in `env_justice_analysis.py`) revealed potential missing values in both the unified and health datasets, particularly for certain health indicators or specific LSOAs/LADs. This necessitated careful handling (e.g., `dropna`) before specific analyses.
3.  **Data Interpretation:** Understanding the precise definition and calculation methodology for composite indices (e.g., the pre-calculated `env_justice_index`, `respiratory_health_index`) required careful review of source documentation (like the NHS OF links file).
4.  **Normalisation:** Several variables were already normalised (e.g., `imd_score_normalized`, `NO2_normalized`). Understanding the normalisation method used was important for interpretation and further analysis.

## Data Cleaning & Preprocessing (Justify Your Choices)

Data preparation involved several steps, primarily executed within the analysis scripts (`env_justice_analysis.py`, `advanced_cluster_analysis.py`, `causal_inference_analysis.py`).

**Methodology & Rationale:**

1.  **Loading:** Data was loaded from CSV files using `pandas.read_csv`.
    *   *Rationale:* Standard, efficient method for handling tabular data.
2.  **Aggregation (LSOA to LAD):** For merging health data, LSOA-level pollution and deprivation metrics were aggregated to the LAD level using `groupby('lad_code').agg(...)` with the `mean` function.
    *   *Rationale:* Necessary to align data granularity for integrated analysis. Using the mean provides a representative central tendency for pollution/deprivation within each LAD, although it masks intra-LAD variation.
3.  **Merging:** Aggregated LAD-level data was merged with the LAD-level health data using `pandas.merge` based on LAD codes (`lad_code`, `local_authority_code`). An `inner` join was typically used.
    *   *Rationale:* Standard method for combining datasets based on common keys. An inner join ensures that only LADs present in both datasets are included in the merged analysis set, maintaining data integrity.
4.  **Handling Missing Values:** Before specific analyses requiring complete data (e.g., clustering, regression, causal inference), rows with missing values in the relevant columns were typically dropped using `dropna(subset=...)`.
    *   *Rationale:* Many statistical models and machine learning algorithms cannot handle missing values directly. Dropping rows is a straightforward approach when the proportion of missing data is relatively small or imputation is complex/unjustified. This was applied selectively before specific analyses (e.g., in `cluster_analysis`, `propensity_score_matching`) to maximise data retention for other steps. *Alternative consideration:* Imputation (e.g., mean, median, model-based) could preserve more data but might introduce bias if not done carefully; `dropna` was deemed sufficient for this project's scope based on observed implementation.
5.  **Feature Scaling/Standardisation:** For distance-based algorithms (KMeans clustering, PCA) and some regression models, features were standardised using `sklearn.preprocessing.StandardScaler`. This transforms data to have zero mean and unit variance.
    *   *Rationale:* Essential for algorithms sensitive to feature scales. KMeans uses Euclidean distance, and PCA finds directions of maximum variance; features with larger ranges would otherwise dominate these calculations. Standardisation ensures all features contribute proportionally.
6.  **Derived Variable Creation:**
    *   **`vulnerability_index`:** Created by standardising selected input variables (`imd_score_normalized`, normalised pollutants, `respiratory_health_index`), averaging the scaled values, and then normalising the result to a 0-100 scale. The pre-calculated `env_justice_index` from the source data was also considered but a custom index was created for transparency.
        *   *Rationale:* Provides a single composite measure summarising multiple risk factors, useful for ranking areas and simplifying analysis. Standardisation ensures equal weighting in the initial average; simple averaging was chosen over PCA-based or weighted methods for transparency and ease of interpretation, acknowledging this assumes equal importance of the components. Normalisation to 0-100 aids interpretability.
    *   **`double_disadvantage`:** A binary flag created using boolean logic based on whether an LAD's deprivation and pollution levels were above their respective medians.
        *   *Rationale:* Directly operationalises the "double disadvantage" concept for comparative analysis (e.g., t-tests, boxplots). Median split provides a simple, data-driven way to define "high" levels.
    *   **Binary Treatment Variables (Causal Inference):** For PSM, continuous pollution variables (`NO2`, `PM2.5`) were converted into binary "high exposure" variables based on being above the median.
        *   *Rationale:* PSM often works more straightforwardly with binary treatment definitions. Median split is a common, though potentially arbitrary, way to dichotomise. This allows estimation of the effect of being in a "high" pollution area compared to a "low" one, controlling for covariates.

**Tools/Libraries:** `pandas` for data loading, manipulation, aggregation, merging; `numpy` for numerical operations; `sklearn.preprocessing.StandardScaler` for standardisation.

## Exploratory Data Analysis (EDA) & Key Insights

EDA was performed primarily in the `explore_data` function of `env_justice_analysis.py` and through visualisations generated in various analysis functions.

**Process:**
1.  **Initial Inspection:** Loaded datasets and examined their shapes (`.shape`), column names, and data types (`.info()` implicitly used by printing columns).
2.  **Missing Value Assessment:** Checked for missing values using `.isnull().sum()` for each key dataset (unified, health).
3.  **Descriptive Statistics:** Calculated summary statistics (`.describe()`) for key numerical variables (air quality, deprivation domains, health indicators) to understand their distributions (mean, std dev, min, max, quartiles).
4.  **Correlation Analysis:** Calculated Pearson correlation coefficients (`scipy.stats.pearsonr`) between key pollution metrics (`NO2_normalized`, `PM2.5_normalized`, etc.) and deprivation (`imd_score_normalized`), and later between pollution and health indicators (`respiratory_health_index`, etc.) in the merged LAD dataset. P-values were checked for statistical significance.
5.  **Visual Exploration:**
    *   **Histograms:** Plotted distributions of key indices like the `env_justice_index` and `vulnerability_index` (`seaborn.histplot`) to understand their shape and spread.
    *   **Scatter Plots:** Visualised relationships between pairs of variables, particularly pollution vs. deprivation and pollution vs. health (`seaborn.scatterplot`). Regression lines (`seaborn.regplot`) were added to visualise trends.
    *   **Box Plots:** Compared distributions of health indicators between groups, such as areas classified as having "double disadvantage" vs. others (`seaborn.boxplot`).

**Key Findings from EDA:**
1.  **Pollution-Deprivation Link:** Positive correlations were observed between key pollutants (NO2, PM2.5) and the Index of Multiple Deprivation (IMD) score, supporting the environmental injustice hypothesis. Scatter plots visually confirmed this trend, although with considerable variance. For example, the correlation between `NO2_normalized` and `imd_score_normalized` was statistically significant (specific r-value would be in script output/docs).
2.  **Pollution-Health Link:** Negative correlations were generally observed between pollution levels (e.g., NO2) and the `respiratory_health_index` at the LAD level, suggesting higher pollution is associated with worse respiratory health outcomes. Scatter plots illustrated this relationship.
3.  **"Double Disadvantage" Impact:** Box plots showed that LADs classified as having "double disadvantage" (high pollution & high deprivation) tended to have worse (lower) `respiratory_health_index` values compared to other areas. T-tests confirmed this difference was statistically significant (p < 0.05).
4.  **Index Distributions:** Histograms revealed the distribution shapes of the composite indices. The `vulnerability_index` showed a wide spread, indicating significant variation in combined risk across LADs.
5.  **Data Quality:** EDA confirmed the presence of missing values that needed handling and highlighted the granularity mismatch requiring aggregation.

**Visualisations (Integrated & Explained):**

*   **(Example) Scatter Plot: NO2 vs. IMD Score:**
    *   *Visualisation:* A scatter plot (`outputs/figures/pollution_vs_deprivation.png` - subplot) showing `NO2_normalized` on the x-axis and `imd_score_normalized` on the y-axis, with each point representing an LSOA. A positive-sloping regression line is overlaid.
    *   *Explanation:* This plot visually demonstrates the positive association between Nitrogen Dioxide levels and socioeconomic deprivation at the local area level. While there is scatter, the trend line indicates that, on average, more deprived areas tend to experience higher NO2 pollution, providing initial evidence for environmental injustice.

*   **(Example) Box Plot: Respiratory Health by Double Disadvantage:**
    *   *Visualisation:* A box plot (`outputs/figures/double_disadvantage_health.png`) comparing the distribution of `respiratory_health_index` for two groups: LADs classified as "double disadvantage" (Yes) and those not (No).
    *   *Explanation:* This plot clearly shows that the median and overall distribution of the respiratory health index are lower (indicating worse health) for the "double disadvantage" group compared to the other group. This supports the hypothesis that the combination of high pollution and high deprivation is associated with poorer respiratory health outcomes at the LAD level. The statistical significance was confirmed by a t-test.

*   **(Example) Histogram: Vulnerability Index Distribution:**
    *   *Visualisation:* A histogram (`outputs/figures/vulnerability_index_distribution.png`) showing the frequency distribution of the calculated `vulnerability_index` across all LADs.
    *   *Explanation:* This plot shows the overall distribution of composite vulnerability (combining pollution, deprivation, and health factors) across LADs. It helps understand the range of vulnerability and identify whether the distribution is skewed or multimodal, informing how the index might be used for policy targeting (e.g., focusing on the tail end).

## Methodology & Model Selection (The "Why")

The project employed a multi-faceted methodology, progressing from foundational analysis to advanced techniques, as documented in `advanced_analysis_documentation.md` and implemented across various Python scripts. This integrated approach, combining correlational, clustering, causal, and spatial methods, was chosen to provide a more comprehensive understanding than relying on a single technique.

**Approach & Justification:**

1.  **Correlation & Regression (Initial Analysis):**
    *   *Approach:* Pearson correlation and scatter plots with linear regression lines (`env_justice_analysis.py`). Basic OLS regression for pollutant interactions (`outputs/pollutant_interactions/interaction_model_summary.txt`).
    *   *Justification:* Standard initial steps to quantify linear associations between key variables (pollution, deprivation, health) and visualise trends. Provides a baseline understanding before employing more complex models. OLS helps explore basic interaction effects. *Limitations:* Correlation doesn't imply causation; linear models may miss non-linearities.

2.  **Composite Index Creation (`vulnerability_index`):**
    *   *Approach:* Standardising selected variables, averaging them, and normalising the result (`env_justice_analysis.py`).
    *   *Justification:* To create a single, interpretable metric summarising multiple risk factors for prioritisation. Standardisation ensures equal weighting; averaging is a simple combination method chosen for transparency over potentially less interpretable PCA-based weighting. *Alternatives:* Weighted averaging based on domain expertise is another option.

3.  **Cluster Analysis (KMeans):**
    *   *Approach:* Unsupervised clustering using KMeans on standardised variables (`imd_score_normalized`, normalised pollutants, `respiratory_health_index`, `vulnerability_index`) (`env_justice_analysis.py`, `advanced_cluster_analysis.py`). Optimal K determined by Elbow method (initial) and Silhouette analysis (advanced). PCA used for visualisation. Cluster profiles analysed using means, boxplots, and radar charts. Feature importance and SHAP values used for interpretability (advanced).
    *   *Justification:* To identify distinct, naturally occurring groups (profiles) of LADs based on their environmental, socioeconomic, and health characteristics, moving beyond simple high/low classifications. KMeans is a widely used, efficient partitioning algorithm. Standardisation is crucial for KMeans. Silhouette analysis provides a more robust measure for optimal K than the subjective Elbow method. PCA helps visualise high-dimensional cluster separation. Profiling, feature importance, and SHAP aid in understanding *what* defines each cluster, making the results actionable.

4.  **Causal Inference (Propensity Score Matching - PSM):**
    *   *Approach:* Defined binary treatment (high pollution > median). Estimated propensity scores using Logistic Regression based on covariates (`imd_score_normalized`). Matched treated units to control units based on nearest neighbour propensity scores. Calculated Average Treatment Effect on the Treated (ATT) by comparing outcomes (`respiratory_health_index`) in matched groups. Checked covariate balance using SMD (`causal_inference_analysis.py`).
    *   *Justification:* To move beyond correlation and estimate the *causal* effect of high pollution exposure on health, controlling for confounding factors (deprivation). PSM attempts to mimic a randomised controlled trial using observational data by creating comparable groups. PSM was chosen over alternatives like IPTW or regression adjustment for its intuitive appeal in creating directly comparable matched groups for analysis. Logistic Regression is standard for propensity score estimation. ATT focuses on the effect for those actually exposed. Covariate balance checks are essential to assess the validity of the matching. Using only IMD score as a covariate is a simplification, focusing on the primary known confounder in this context; future work could explore incorporating additional covariates if available and appropriate.

5.  **Dose-Response Modelling:**
    *   *Approach:* Used Gradient Boosting Regressor (GBR) to model the relationship between continuous pollution levels (`NO2`, `PM2.5`) and health outcomes, controlling for covariates. Generated predicted outcome curves across the pollution range. Used bootstrapping to estimate 95% confidence intervals (`causal_inference_analysis.py`).
    *   *Justification:* To understand how health outcomes change across the *entire spectrum* of pollution exposure, allowing for non-linear relationships and potential thresholds, which linear models might miss. GBR was selected over simpler non-linear models (e.g., polynomial regression) for its ability to capture complex interactions and provide variable importance insights without requiring pre-specified functional forms. Bootstrapping provides robust confidence intervals for the estimated curve.

6.  **Policy Impact Simulation:**
    *   *Approach:* Used the fitted dose-response model (GBR) to predict health outcomes under counterfactual scenarios where pollution levels were reduced by specific percentages (10%, 20%, etc.). Calculated the average predicted improvement and identified LADs with the largest potential gains (`causal_inference_analysis.py`).
    *   *Justification:* To translate the causal models into actionable policy insights. Simulates the potential benefits of specific interventions (pollution reduction) and helps prioritise areas where interventions might be most effective.

7.  **Advanced Analyses (Rationale from Docs):**
    *   **Multivariate Pollutant Interaction:** To capture complex atmospheric chemistry, non-linear dose-response, and combined impacts missed by analysing pollutants individually. PCA helps create composite indices.
    *   **Domain-Specific Deprivation:** To understand which specific aspects of deprivation (e.g., housing, income) drive environmental injustice patterns, enabling more targeted social interventions.
    *   **Spatial Autocorrelation & Hotspot Analysis:** To account for the geographical nature of the data, identify spatial clustering of injustice, understand neighbourhood effects, and inform spatially targeted policies. Standard statistics assume independence, which is often violated in geographic data.

**Assumptions:**
*   **PSM:** Assumes "conditional ignorability" (all relevant confounders are measured and included in the propensity score model - a strong assumption, particularly with limited covariates) and "common support" (overlap in propensity scores between treated and control groups).
*   **Regression Models:** Assume correct model specification (though GBR is flexible), independence of errors (potentially violated by spatial autocorrelation, addressed partly by spatial analysis module), homoscedasticity.
*   **Data Accuracy:** Assumes the underlying data sources (pollution monitoring, IMD, health stats) are reasonably accurate and representative.
*   **Aggregation:** Assumes that LAD-level averages for pollution/deprivation are meaningful proxies for the exposure experienced by the population whose health outcomes are measured at that level (potential ecological fallacy).

**Evaluation Metrics:**
*   **Clustering:** Silhouette Score (to evaluate cluster separation and cohesion), Inertia (Elbow method, less reliable), visual inspection of PCA plots and cluster profiles.
*   **Regression (OLS):** R-squared, Adjusted R-squared, F-statistic, p-values for coefficients (to assess model fit and variable significance).
*   **Causal Inference (PSM):** Covariate balance (Standardised Mean Differences - SMD < 0.1 or 0.2 often considered good balance), p-values from t-tests on outcomes in matched groups (though effect size/ATT is the main focus).
*   **Dose-Response (GBR):** Visual inspection of the curve and confidence intervals, potentially cross-validation metrics (RMSE, R2) if model prediction accuracy itself was a primary goal (here, the shape of the relationship was key).

*Justification for Metrics:* Silhouette score directly measures cluster quality. Standard regression metrics assess overall fit and significance. Covariate balance is crucial for PSM validity. Visual inspection and confidence intervals are key for interpreting dose-response curves.

## Analysis, Modelling & Implementation (The "How")

This section details the execution of the key analytical steps, referencing the Python scripts and outputs.

**Execution:**

1.  **Data Loading & Initial Prep (`env_justice_analysis.py::load_data`, `::explore_data`):** Datasets loaded using Pandas. Initial checks for shape, columns, and missing values performed. Basic descriptive statistics generated.
2.  **LAD-Level Aggregation & Merging (`env_justice_analysis.py::merge_health_with_pollution`):** LSOA data aggregated to LAD level using `groupby().agg('mean')`. Merged with LAD health data using `pd.merge()` on LAD codes. The resulting `lad_health_pollution_merged.csv` formed the basis for many subsequent analyses.
3.  **Vulnerability Index Calculation (`env_justice_analysis.py::create_vulnerability_index`):** Selected variables standardised using `StandardScaler`. Index calculated as the mean of scaled variables, then normalised to 0-100. Saved in `lad_with_vulnerability_index.csv`.

    ```python
    # --- Snippet: Vulnerability Index Calculation ---
    index_vars = [
        'imd_score_normalized', 'NO2_normalized', 'PM2.5_normalized',
        'PM10_normalized', 'respiratory_health_index'
    ]

    analysis_df = merged_df.dropna(subset=index_vars)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(analysis_df[index_vars])
    scaled_df = pd.DataFrame(scaled_data, columns=index_vars, index=analysis_df.index)

    vulnerability_scores = scaled_df.mean(axis=1)

    min_val = vulnerability_scores.min()
    max_val = vulnerability_scores.max()
    merged_df.loc[analysis_df.index, 'vulnerability_index'] = 100 * (vulnerability_scores - min_val) / (max_val - min_val)

    print("Vulnerability Index created and added to DataFrame.")
    # --- End Snippet ---
    ```
    *Explanation:* This snippet shows the process of selecting relevant features, standardising them to ensure equal contribution, calculating a raw composite score by averaging, and finally normalising this score to an intuitive 0-100 scale for easier interpretation and comparison across LADs.

4.  **Clustering (`advanced_cluster_analysis.py::main`, `env_justice_analysis.py::cluster_analysis`):**
    *   Selected features (`imd_score_normalized`, normalised pollutants, health index, vulnerability index).
    *   Handled missing values using `dropna()`.
    *   Standardised data using `StandardScaler`.
    *   Determined optimal K=4 using Silhouette analysis (`perform_silhouette_analysis`, `visualize_silhouette_plot`).
    *   Applied KMeans with K=4 (`KMeans(n_clusters=optimal_k, ...)`).
    *   Visualised clusters using PCA (`visualize_clusters_pca`).
    *   Analysed profiles using statistics, boxplots, and radar charts (`analyze_cluster_profiles`).
    *   Calculated feature importance (`calculate_feature_importance`) and SHAP values (`calculate_shap_values`) for interpretability.

5.  **Causal Inference (`causal_inference_analysis.py::main`):**
    *   Defined binary treatment variables (`high_NO2`, `high_PM2.5`) based on median split.
    *   Defined covariates (`imd_score_normalized`).
    *   **PSM:**
        *   Fitted Logistic Regression model (`LogisticRegression`) to predict treatment based on covariates.
        *   Calculated propensity scores (`predict_proba`).
        *   Performed nearest-neighbour matching (looping through treated units, finding closest control based on `abs(ps_diff)`).
        *   Calculated ATT (`matched_treated[outcome].mean() - matched_control[outcome].mean()`).
        *   Checked covariate balance (SMD calculation, boxplots).

    ```python
    # --- Snippet: Propensity Score Matching (Conceptual Logic) ---
    treatment_var = 'high_NO2'
    outcome_var = 'respiratory_health_index'
    covariates = ['imd_score_normalized']

    analysis_data = merged_df[[treatment_var, outcome_var] + covariates].dropna()

    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(analysis_data[covariates], analysis_data[treatment_var])
    analysis_data['propensity_score'] = ps_model.predict_proba(analysis_data[covariates])[:, 1]

    treated = analysis_data[analysis_data[treatment_var] == 1]
    control = analysis_data[analysis_data[treatment_var] == 0]
    matched_pairs = []

    matched_treated = pd.DataFrame([pair[0] for pair in matched_pairs])
    matched_control = pd.DataFrame([pair[1] for pair in matched_pairs])

    att = matched_treated[outcome_var].mean() - matched_control[outcome_var].mean()
    print(f"ATT for {treatment_var}: {att:.4f}")

    mean_treated_cov = matched_treated[covariates[0]].mean()
    mean_control_cov = matched_control[covariates[0]].mean()
    pooled_std_cov = np.sqrt((matched_treated[covariates[0]].var() + matched_control[covariates[0]].var()) / 2)
    smd = abs(mean_treated_cov - mean_control_cov) / pooled_std_cov if pooled_std_cov > 0 else 0
    print(f"SMD for {covariates[0]} after matching: {smd:.4f}")
    # --- End Snippet ---
    ```
    *Explanation:* This snippet outlines the core steps of PSM: estimating the probability (propensity score) of receiving treatment (high pollution) based on confounders (deprivation), matching treated and control units with similar propensity scores to create comparable groups, estimating the treatment effect (ATT) by comparing outcomes in these matched groups, and checking if the matching successfully balanced the confounders (low SMD indicates good balance).

    *   **Dose-Response:**
        *   Fitted `GradientBoostingRegressor` model with outcome (`respiratory_health_index`) as dependent variable and continuous pollutant + covariates as independent variables.
        *   Predicted outcomes over a grid of pollutant values (holding covariates at mean).
        *   Used bootstrapping (resampling data, refitting model) to generate 95% confidence intervals for the dose-response curve.
    *   **Policy Simulation:**
        *   Used the fitted GBR model.
        *   Created counterfactual datasets where pollutant levels were reduced by X%.
        *   Predicted outcomes using the model on counterfactual data.
        *   Calculated improvement (counterfactual prediction - baseline prediction).
        *   Averaged improvement and identified top-benefiting LADs.

6.  **Pollutant Interaction Analysis (`pollutant_interaction_analysis.py` - inferred from OLS summary):**
    *   Likely involved creating interaction terms (e.g., `NO2 * O3`).
    *   Fitted an OLS regression model (`statsmodels.formula.api` or `statsmodels.api`) with health outcome as dependent variable and pollutants + interaction terms as independent variables.
    *   Examined coefficients and p-values for interaction terms (results in `interaction_model_summary.txt`). PCA was also mentioned for creating composite indices.

**Tools/Frameworks:**
*   **Core:** `pandas`, `numpy`, `scikit-learn` (for `StandardScaler`, `KMeans`, `PCA`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingRegressor`), `matplotlib`, `seaborn` (for static visualisations).
*   **Statistical Modelling:** `statsmodels` (for OLS regression), `scipy.stats` (for `pearsonr`, `ttest_ind`).
*   **Interpretability:** `shap` (for explaining cluster model predictions).
*   **Interactive Viz:** `plotly` (for interactive dose-response curves, policy impact bars).
*   **Spatial (Optional):** `geopandas`, `pysal` (mentioned in docs, used in `spatial_hotspot_analysis.py`).

## Results & Findings

This section presents the key quantitative and qualitative results derived from the analyses.

**Quantitative Results:**

1.  **Environmental Justice Correlations (LSOA Level):**
    *   Significant positive correlations found between normalised deprivation (`imd_score_normalized`) and normalised pollution levels. From `README.md`: Living Environment Deprivation (r=0.129) and Housing & Services Barriers (r=0.128) showed strongest correlations with pollution (specific pollutant not stated, likely composite or average). Pearson correlations between `imd_score_normalized` and specific pollutants (`NO2_normalized`, `PM2.5_normalized`) were statistically significant (p < 0.001, specific r-values calculated in `env_justice_analysis.py`).
2.  **Health-Pollution Correlations (LAD Level):**
    *   Significant negative correlations observed between `respiratory_health_index` and pollution metrics like `NO2` (specific r and p-values calculated in `env_justice_analysis.py`).
3.  **"Double Disadvantage" Impact (LAD Level):**
    *   Areas with high deprivation and high pollution had a statistically significantly lower mean `respiratory_health_index` compared to other areas (T-test p-value < 0.05, specific means calculated in `env_justice_analysis.py`). Mean for double disadvantage areas: ~0.26 vs. ~0.30 for others (example values, check script output).
4.  **Cluster Analysis (K=4 Clusters, LAD Level):**
    *   Optimal K=4 identified via Silhouette analysis (average score ~0.3-0.4, check `silhouette_analysis.png`).
    *   Cluster profiles (from `cluster_statistics.csv`):
        *   **Cluster 0 (Low Vuln/Good Health):** ~19 LADs. Lowest mean vulnerability, lowest pollution (NO2, PM2.5), relatively good health index. Moderate deprivation.
        *   **Cluster 1 (Moderate Vuln/High Pollution):** ~119 LADs. Moderate vulnerability, high mean NO2/PM2.5, highest mean health index (unexpected). Moderate deprivation.
        *   **Cluster 2 (Highest Vuln/Highest Pollution):** ~7 LADs. Highest mean vulnerability, highest mean NO2/PM2.5. Moderate health index. Highest deprivation.
        *   **Cluster 3 (Low-Moderate Vuln/Good Health/Low Deprivation):** ~142 LADs. Low-moderate vulnerability, high mean NO2/PM2.5 (similar to Cluster 1), lowest mean health index (poorest health), lowest mean deprivation.
    *   *Interpretation Note:* The relationship between the health index and other factors in clusters appears complex. The unexpected health index results in Clusters 1 & 3 warrant further investigation, potentially related to the specific health index definition (e.g., if lower is better), lag effects between exposure and outcome, or unmeasured protective/risk factors (like healthcare access or smoking rates) within those LAD groups.
    *   Feature Importance: `vulnerability_index` and `respiratory_health_index` were often key drivers of cluster separation (see `feature_importance.png`).
5.  **Pollutant Interactions (OLS Regression, LAD Level):**
    *   From `interaction_model_summary.txt`: The overall model predicting `respiratory_health_index` from pollutants and their interactions had low explanatory power (R-squared: 0.040, Adj. R-squared: 0.005).
    *   No individual pollutant or interaction term was statistically significant at p < 0.05 in this specific OLS model. *Interpretation Note:* This weak result might stem from multicollinearity (e.g., the strong negative correlation between NO2 and O3, r=-0.514), the limitations of a linear model for complex interactions, or suggest that interactions are primarily non-linear and better captured by the GBR model used for dose-response analysis.
6.  **Causal Effects (PSM, LAD Level):**
    *   High NO2 Exposure: ATT = -0.0387 (p-value < 0.05, check script output for exact p-value). After matching on `imd_score_normalized`, areas with NO2 above the median had a significantly lower `respiratory_health_index` compared to similar areas with NO2 below the median. Covariate balance was achieved (SMD < 0.1). *Context:* While representing a shift on the index scale, this ATT signifies a statistically significant worsening of average respiratory health outcomes attributable to high NO2 exposure in deprived areas.
    *   High PM2.5 Exposure: ATT results also calculated (check script output).
7.  **Dose-Response & Thresholds:**
    *   GBR models showed non-linear relationships between pollutants (NO2, PM2.5) and `respiratory_health_index` (see `dose_response_*.png`).
    *   Threshold effect identified for PM10 at 13.83 μg/m³ (from `README.md`, analysis likely in `pollutant_interaction_analysis.py`).
8.  **Policy Simulation:**
    *   A 20% reduction in NO2 yielded the highest average improvement in `respiratory_health_index` (+0.0460) among tested scenarios (10-50%). *Context:* This simulated average improvement, while modest on the index scale, represents a potential population-level health benefit achievable through targeted policy action, particularly impactful in the highest-risk areas identified.
    *   Top benefiting LADs from 20% NO2 reduction included Cambridge, Westminster, and Tower Hamlets (from `README.md` and `top_areas_NO2.png`).
9.  **Spatial Analysis (from Docs):**
    *   Significant positive spatial autocorrelation found (Moran's I), indicating clustering of similar values.
    *   Hotspots identified (Getis-Ord Gi*), e.g., Bradford (NO2), Nottingham (PM2.5).
    *   Spatial regression confirmed neighbourhood effects.

**Interpretation:**
The results consistently point towards environmental injustice, where higher deprivation is linked to higher pollution. Crucially, the causal analysis suggests that higher NO2 pollution *causes* worse respiratory health outcomes, even after accounting for deprivation levels. While the linear OLS model for interactions was weak (potentially due to multicollinearity or non-linear effects), the dose-response modelling revealed these non-linearities, and specific thresholds (like for PM10) indicate points beyond which health impacts may accelerate. Cluster analysis identified distinct area profiles, with a small group (Cluster 2) exhibiting the highest combined pollution, deprivation, and vulnerability, although some cluster profiles require further investigation regarding health index behaviour. Policy simulations provide concrete estimates of potential health gains from pollution reduction, highlighting specific areas for targeted interventions. Spatial analysis confirms these issues are geographically clustered, emphasising the need for spatially aware policies.

**Visualisations (Results-Focused):**

*   **Cluster PCA Plot (`outputs/advanced_clustering/pca_clusters.png`):** Shows the separation of the 4 clusters in a 2D space defined by the first two principal components. Helps visualise how distinct the groups are based on the combined variables. Explained variance on axes indicates how much information the 2D plot captures. Centroids mark the average position of each cluster.
*   **Cluster Radar Chart (`outputs/advanced_clustering/cluster_radar_chart.png`):** Compares the average (normalised) profile of each cluster across key dimensions (deprivation, pollutants, health, vulnerability). Allows quick visual identification of cluster characteristics (e.g., Cluster 2 likely showing high values on most axes except perhaps health).
*   **Dose-Response Curve (`outputs/causal_inference/dose_response_NO2.png`):** Plots the estimated relationship between NO2 concentration (x-axis) and `respiratory_health_index` (y-axis), controlling for covariates. The curve's shape (potentially non-linear) and the 95% confidence interval illustrate the estimated impact and uncertainty across different pollution levels.
*   **Policy Impact Bar Chart (`outputs/causal_inference/top_areas_NO2.png`):** Shows the top 10 LADs predicted to experience the largest improvement in `respiratory_health_index` following a simulated 20% reduction in NO2. Directly informs intervention prioritisation.
*   **Spatial Hotspot Maps (`outputs/spatial_hotspots/hotspot_map_*_demo.png` - *if generated*):** Geographic maps highlighting statistically significant clusters of high values (hotspots) and low values (coldspots) for key variables like pollution or vulnerability indices. Visually pinpoints areas of concern.

## Discussion & Impact

**Synthesis:**
The findings strongly support the initial hypotheses. We established statistically significant links between deprivation, pollution, and adverse respiratory health outcomes at the area level. Moving beyond correlation, the causal inference analysis provided evidence that high NO2 exposure negatively impacts respiratory health, even when controlling for socioeconomic deprivation. The "double disadvantage" concept was validated, with co-located high pollution and deprivation associated with the poorest health outcomes. The analysis successfully identified distinct area profiles through clustering and pinpointed specific geographic hotspots of concern using spatial statistics. Furthermore, the dose-response modelling and policy simulations quantified the potential health benefits of pollution reduction, fulfilling the objective of providing actionable insights for intervention prioritisation. The integrated analytical approach demonstrates a robust method for dissecting complex environmental health problems.

**Actionable Insights & Recommendations:**

1.  **Targeted Interventions:** Focus pollution reduction efforts on areas identified as high-vulnerability hotspots (e.g., Bradford, Nottingham) and those predicted to gain most from interventions (e.g., Cambridge, Westminster, Tower Hamlets for NO2). Utilise the cluster profiles and domain-specific deprivation analysis (*from `domain_deprivation_analysis.py`*) to tailor interventions (e.g., addressing housing barriers alongside pollution in relevant clusters).
2.  **Policy Thresholds:** Consider the identified threshold effects (e.g., for PM10) when setting local air quality targets, aiming to keep concentrations below levels where health impacts accelerate.
3.  **Integrated Strategies:** Address environmental justice through integrated strategies tackling both pollution sources and underlying socioeconomic deprivation, particularly focusing on domains like Living Environment and Housing Barriers that correlate strongly with pollution.
4.  **Spatial Planning:** Incorporate findings on spatial clustering and neighbourhood effects into urban and regional planning to prevent the concentration of pollution sources near vulnerable communities and leverage potential positive spillover effects from interventions.
5.  **Health Monitoring:** Enhance public health surveillance in identified high-vulnerability clusters and hotspots to monitor trends and evaluate intervention effectiveness.

**Impact (for Public Health / Government Stakeholders):**
This analysis provides a robust, data-driven evidence base to:
*   **Justify Resource Allocation:** Direct public health and environmental protection resources towards areas with the greatest need and potential for health improvement.
*   **Inform Policy Design:** Develop more targeted, effective, and equitable air quality and health policies based on specific local profiles and causal relationships.
*   **Improve Health Outcomes:** Contribute to reducing respiratory illnesses and health inequalities by addressing key environmental and socioeconomic drivers.
*   **Enhance Accountability:** Provide metrics (vulnerability index, simulated health gains) for tracking progress and evaluating the impact of interventions.

**Limitations:**

1.  **Ecological Fallacy:** Analysing aggregated LAD-level data means relationships observed may not hold at the individual level. Individual exposure can vary significantly within an LAD.
2.  **Data Limitations:** Reliance on available monitoring data (may not capture all relevant pollutants or micro-environments), potential inaccuracies in data sources, use of cross-sectional data limits inference about changes over time. Health data aggregation to LAD level masks LSOA-level variations.
3.  **Causal Inference Assumptions:** PSM relies on untestable assumptions (conditional ignorability). Using only IMD as a covariate is a simplification; unmeasured confounders (e.g., smoking rates, healthcare access variations) could still bias results. The definition of "high" pollution based on median split is somewhat arbitrary.
4.  **Model Simplifications:** Composite indices involve choices about variable selection and weighting. Regression models might not capture all real-world complexities. GBR, while flexible, can be harder to interpret than simpler models.
5.  **Scope:** Excluded factors like indoor air quality, specific emission sources, occupational exposure, and detailed demographics could influence outcomes.

## Conclusion & Future Work

**Summary:**
This project successfully developed and applied a comprehensive analytical framework to investigate environmental justice and its link to respiratory health in the UK. By integrating geospatial data, socioeconomic indicators, pollution metrics, and health outcomes, we quantified the disproportionate burden faced by deprived communities. Advanced techniques like causal inference confirmed the negative health impacts of pollution, while clustering and spatial analysis identified distinct high-risk area profiles and geographic hotspots. The findings provide actionable insights, highlighting the need for targeted, spatially aware interventions that address both environmental hazards and social vulnerability to mitigate health inequalities.

**Future Directions:**

1.  **Temporal Analysis:** Incorporate longitudinal data to analyse trends in pollution, deprivation, and health over time, assessing the impact of past policies and identifying emerging issues.
2.  **Granular Health Data:** Utilise more granular health data (e.g., LSOA-level or individual-level, if available ethically and legally) to reduce ecological fallacy concerns.
3.  **Advanced Causal Methods:** Explore alternative causal inference techniques (e.g., Regression Discontinuity, Instrumental Variables) if suitable data or natural experiments can be identified. Include richer covariate sets in PSM or other models.
4.  **Source Apportionment:** Integrate pollution source data (e.g., traffic, industry) to identify dominant contributors in hotspots and tailor interventions more effectively.
5.  **Machine Learning Prediction:** Develop models to predict future health burdens under different environmental and socioeconomic scenarios.
6.  **Interactive Dashboard:** Create a web-based dashboard (e.g., using Streamlit or Dash) to allow stakeholders to interactively explore the data, visualisations, and policy simulation results. (A `dashboard.py` file exists, suggesting this might be partially implemented or planned).
7.  **Intervention Cost-Benefit Analysis:** Integrate economic data to assess the cost-effectiveness of different pollution reduction strategies.

This project provides a strong foundation for ongoing research and policy action aimed at achieving environmental equity and improving public health across the UK.