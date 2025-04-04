"""
Environmental Justice and Health Inequalities: A Geospatial Analysis of Air Pollution,
Deprivation, and Respiratory Health in England

This script performs data loading, preprocessing, and initial analysis of the relationship
between air pollution, socioeconomic deprivation, and respiratory health outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import os

# Try to import optional packages
try:
    import geopandas as gpd

    GEOPANDAS_AVAILABLE = True
except ImportError:
    print("Warning: geopandas not available. Spatial analysis features will be limited.")
    GEOPANDAS_AVAILABLE = False

# Set the plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# File paths
UNIFIED_DATA_PATH = "data/processed/unified_datasets/unified_dataset_with_air_quality.csv"

HEALTH_DATA_PATH = "data/raw/health/health_indicators_by_lad.csv"

# Create output directories if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)


def load_data():
    """
    Load and perform initial preprocessing of the datasets.

    Returns:
        tuple: Processed dataframes (unified_df, health_df, wards_df)
    """
    print("Loading datasets...")

    # Load the unified dataset with air quality and deprivation metrics
    unified_df = pd.read_csv(UNIFIED_DATA_PATH)

    # Load the health indicators dataset
    health_df = pd.read_csv(HEALTH_DATA_PATH)

    # Load the wards dataset
    # wards_df = pd.read_csv(WARDS_DATA_PATH) # Commented out due to undefined WARDS_DATA_PATH
    wards_df = pd.DataFrame()  # Assign empty DataFrame to avoid later errors if referenced

    print(
        f"Loaded {len(unified_df)} LSOA records and {len(health_df)} LAD health records."
    )  # Removed wards_df reference

    # Count missing values before dropping
    print("\nMissing values in unified dataset before dropping:")
    print(unified_df.isnull().sum().sum())
    print("\nMissing values in health dataset before dropping:")
    print(health_df.isnull().sum().sum())

    return unified_df, health_df, wards_df


def explore_data(unified_df, health_df, wards_df):
    """
    Perform exploratory data analysis on the datasets.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
        health_df (pd.DataFrame): Health indicators dataset
        wards_df (pd.DataFrame): Wards geographical dataset
    """
    print("\n--- Exploratory Data Analysis ---")

    # Unified dataset summary
    print("\nUnified Dataset Summary:")
    print(f"Shape: {unified_df.shape}")
    print("\nColumns:")
    for col in unified_df.columns:
        print(f"- {col}")

    # Check for missing values
    print("\nMissing values in unified dataset:")
    missing_unified = unified_df.isnull().sum()
    print(missing_unified[missing_unified > 0])

    # Health dataset summary
    print("\nHealth Dataset Summary:")
    print(f"Shape: {health_df.shape}")
    print("\nColumns:")
    for col in health_df.columns:
        print(f"- {col}")

    # Check for missing values
    print("\nMissing values in health dataset:")
    missing_health = health_df.isnull().sum()
    print(missing_health[missing_health > 0])

    # Basic statistics for key variables
    print("\nAir Quality Statistics:")
    air_quality_cols = ["NO2", "O3", "PM10", "PM2.5"]
    print(unified_df[air_quality_cols].describe())

    print("\nDeprivation Statistics:")
    deprivation_cols = [
        "imd_score_normalized",
        "income_score_rate",
        "employment_score_rate",
        "health_deprivation_and_disability_score",
    ]
    print(unified_df[deprivation_cols].describe())

    print("\nHealth Indicators Statistics:")
    health_cols = [
        "chronic_conditions_normalized",
        "asthma_diabetes_epilepsy_normalized",
        "lrti_children_normalized",
        "respiratory_health_index",
    ]
    print(health_df[health_cols].describe())


def calculate_pollution_deprivation_correlation(unified_df):
    """
    Calculate the correlation between air pollution and deprivation.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
    """
    print("\nCalculating correlation between air pollution and deprivation:")
    pollution_cols = ["NO2_normalized", "PM2.5_normalized", "PM10_normalized", "O3"]
    deprivation_cols = ["imd_score_normalized"]

    for p_col in pollution_cols:
        for d_col in deprivation_cols:
            corr, p_value = pearsonr(unified_df[p_col], unified_df[d_col])
            print(f"{p_col} vs {d_col}: r = {corr:.3f}, p = {p_value:.6f}")


def plot_pollution_deprivation_scatter(unified_df):
    """
    Create scatter plots of air pollution vs. deprivation.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
    """
    print("\nCreating scatter plots of air pollution vs. deprivation:")
    pollution_cols = ["NO2_normalized", "PM2.5_normalized", "PM10_normalized", "O3"]

    plt.figure(figsize=(16, 12))

    for i, p_col in enumerate(pollution_cols):
        plt.subplot(2, 2, i + 1)
        sns.scatterplot(
            x=unified_df[p_col], y=unified_df["imd_score_normalized"], alpha=0.5, edgecolor=None
        )

        # Add trend line
        sns.regplot(
            x=unified_df[p_col],
            y=unified_df["imd_score_normalized"],
            scatter=False,
            line_kws={"color": "red"},
        )

        plt.title(f"{p_col} vs IMD Score")
        plt.xlabel(p_col)
        plt.ylabel("IMD Score (Normalized)")

    plt.tight_layout()
    plt.savefig("outputs/figures/pollution_vs_deprivation.png", dpi=300)


def analyze_environmental_justice_index(unified_df):
    """
    Analyze the distribution of the environmental justice index.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
    """
    print("\nAnalyzing the distribution of the environmental justice index:")
    plt.figure(figsize=(10, 6))
    sns.histplot(unified_df["env_justice_index"], bins=50, kde=True)
    plt.title("Distribution of Environmental Justice Index")
    plt.xlabel("Environmental Justice Index")
    plt.ylabel("Frequency")
    plt.savefig("outputs/figures/env_justice_distribution.png", dpi=300)


def identify_high_injustice_areas(unified_df):
    """
    Identify areas with high environmental injustice.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
    """
    print("\nIdentifying areas with high environmental injustice:")
    high_injustice = unified_df.sort_values("env_justice_index", ascending=False).head(20)
    print("\nTop 20 areas with highest environmental injustice:")
    print(
        high_injustice[
            [
                "lsoa_code",
                "lsoa_name",
                "lad_name",
                "imd_score_normalized",
                "NO2_normalized",
                "PM2.5_normalized",
                "env_justice_index",
            ]
        ]
    )

    # Save to CSV
    high_injustice.to_csv("outputs/data/high_injustice_areas.csv", index=False)


def analyze_environmental_justice(unified_df):
    """
    Analyze environmental justice by examining the relationship between
    air pollution and deprivation.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
    """
    print("\n--- Environmental Justice Analysis ---")

    calculate_pollution_deprivation_correlation(unified_df)
    plot_pollution_deprivation_scatter(unified_df)
    analyze_environmental_justice_index(unified_df)
    identify_high_injustice_areas(unified_df)


def merge_health_with_pollution(unified_df, health_df):
    """
    Merge health indicators with pollution and deprivation data at the LAD level.

    Args:
        unified_df (pd.DataFrame): Unified dataset with air quality and deprivation
        health_df (pd.DataFrame): Health indicators dataset
    """
    print("\n--- Merging Health Data with Pollution Data ---")

    # Aggregate unified data to LAD level
    lad_aggregated = (
        unified_df.groupby("lad_code")
        .agg(
            {
                "lad_name": "first",
                "imd_score_normalized": "mean",
                "NO2": "mean",
                "O3": "mean",
                "PM10": "mean",
                "PM2.5": "mean",
                "NO2_normalized": "mean",
                "PM2.5_normalized": "mean",
                "PM10_normalized": "mean",
                "env_justice_index": "mean",
            }
        )
        .reset_index()
    )

    # Merge with health data
    merged_df = pd.merge(
        lad_aggregated, health_df, left_on="lad_code", right_on="local_authority_code", how="inner"
    )

    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Number of LADs in merged dataset: {len(merged_df)}")

    # Save merged dataset
    merged_df.to_csv("outputs/data/lad_health_pollution_merged.csv", index=False)

    return merged_df


def analyze_health_pollution_relationship(merged_df):
    """
    Analyze the relationship between air pollution and health outcomes.

    Args:
        merged_df (pd.DataFrame): Merged dataset with health and pollution data
    """
    print("\n--- Health-Pollution Relationship Analysis ---")

    # Calculate correlations
    pollution_cols = ["NO2", "O3", "PM10", "PM2.5"]
    health_cols = [
        "respiratory_health_index",
        "chronic_conditions_normalized",
        "asthma_diabetes_epilepsy_normalized",
        "lrti_children_normalized",
    ]

    print("\nCorrelation between air pollution and health outcomes:")
    for p_col in pollution_cols:
        for h_col in health_cols:
            # Drop NaN values for this specific pair of columns
            valid_data = merged_df[[p_col, h_col]].dropna()
            if len(valid_data) > 5:  # Only calculate if we have enough data points
                corr, p_value = pearsonr(valid_data[p_col], valid_data[h_col])
                print(f"{p_col} vs {h_col}: r = {corr:.3f}, p = {p_value:.6f}")
            else:
                print(f"{p_col} vs {h_col}: Insufficient data for correlation")

    # Create scatter plots
    plt.figure(figsize=(16, 12))

    for i, p_col in enumerate(pollution_cols):
        plt.subplot(2, 2, i + 1)
        sns.scatterplot(
            x=merged_df[p_col], y=merged_df["respiratory_health_index"], alpha=0.6, edgecolor=None
        )

        # Add trend line
        sns.regplot(
            x=merged_df[p_col],
            y=merged_df["respiratory_health_index"],
            scatter=False,
            line_kws={"color": "red"},
        )

        plt.title(f"{p_col} vs Respiratory Health Index")
        plt.xlabel(p_col)
        plt.ylabel("Respiratory Health Index")

    plt.tight_layout()
    plt.savefig("outputs/figures/pollution_vs_health.png", dpi=300)

    # Analyze double disadvantage (high pollution + high deprivation)
    merged_df["double_disadvantage"] = (
        merged_df["imd_score_normalized"] > merged_df["imd_score_normalized"].median()
    ) & (
        (merged_df["NO2_normalized"] > merged_df["NO2_normalized"].median())
        | (merged_df["PM2.5_normalized"] > merged_df["PM2.5_normalized"].median())
    )

    # Compare health outcomes between double disadvantage and other areas
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="double_disadvantage", y="respiratory_health_index", data=merged_df)
    plt.title("Respiratory Health Index by Double Disadvantage Status")
    plt.xlabel("Double Disadvantage (High Pollution + High Deprivation)")
    plt.ylabel("Respiratory Health Index")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.savefig("outputs/figures/double_disadvantage_health.png", dpi=300)

    # T-test to compare means
    from scipy.stats import ttest_ind

    double_disadv = merged_df[merged_df["double_disadvantage"]].dropna(
        subset=["respiratory_health_index"]
    )["respiratory_health_index"]
    others = merged_df[~merged_df["double_disadvantage"]].dropna(
        subset=["respiratory_health_index"]
    )["respiratory_health_index"]

    print(
        f"\nT-test comparing respiratory health index between double disadvantage areas and others:"
    )
    if len(double_disadv) > 5 and len(others) > 5:
        t_stat, p_val = ttest_ind(double_disadv, others)
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")
    else:
        print("Insufficient data for t-test")

    print(f"Mean for double disadvantage areas: {double_disadv.mean():.3f}")
    print(f"Mean for other areas: {others.mean():.3f}")
    print(f"Number of double disadvantage areas with data: {len(double_disadv)}")
    print(f"Number of other areas with data: {len(others)}")


def create_vulnerability_index(merged_df):
    """
    Create a composite vulnerability index combining pollution, deprivation, and health factors.

    Args:
        merged_df (pd.DataFrame): Merged dataset with health and pollution data
    """
    print("\n--- Creating Vulnerability Index ---")

    # Select variables for the index
    index_vars = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "PM10_normalized",
        "respiratory_health_index",
    ]

    # Drop rows with missing values in the selected columns
    merged_df_subset = merged_df.dropna(subset=index_vars).copy()

    # Calculate percentage of data lost
    initial_shape = merged_df.shape[0]
    percentage_loss = ((initial_shape - merged_df_subset.shape[0]) / initial_shape) * 100
    print(
        f"\nPercentage of data lost after dropping missing values in create_vulnerability_index: {percentage_loss:.2f}%"
    )

    # Standardize variables
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_df_subset[index_vars])
    scaled_df = pd.DataFrame(scaled_data, columns=index_vars, index=merged_df_subset.index)

    # Create composite index (simple average)
    merged_df["vulnerability_index"] = scaled_df.mean(axis=1)

    # Normalize to 0-100 scale
    min_val = merged_df["vulnerability_index"].min()
    max_val = merged_df["vulnerability_index"].max()
    merged_df["vulnerability_index"] = (
        100 * (merged_df["vulnerability_index"] - min_val) / (max_val - min_val)
    )

    # Identify high vulnerability areas
    high_vulnerability = merged_df.sort_values("vulnerability_index", ascending=False).head(20)
    print("\nTop 20 areas with highest vulnerability:")
    print(
        high_vulnerability[
            [
                "lad_name",
                "vulnerability_index",
                "imd_score_normalized",
                "NO2_normalized",
                "PM2.5_normalized",
                "respiratory_health_index",
            ]
        ]
    )

    # Save to CSV
    high_vulnerability.to_csv("outputs/data/high_vulnerability_areas.csv", index=False)

    # Plot vulnerability index distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df["vulnerability_index"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Vulnerability Index")
    plt.xlabel("Vulnerability Index (0-100)")
    plt.ylabel("Frequency")
    plt.savefig("outputs/figures/vulnerability_index_distribution.png", dpi=300)

    # Save the full dataset with vulnerability index
    merged_df.to_csv("outputs/data/lad_with_vulnerability_index.csv", index=False)

    # Conduct sensitivity analysis
    default_weights = {
        "imd_score_normalized": 0.2,
        "NO2_normalized": 0.2,
        "PM2.5_normalized": 0.2,
        "PM10_normalized": 0.2,
        "respiratory_health_index": 0.2,
    }

    # Test different weighting schemes
    print("\n--- Sensitivity Analysis ---")
    print("\nDefault weights:")
    sensitivity_analysis(merged_df.copy(), default_weights, scaled_df, index_vars)

    # Increase weight on IMD
    imd_weights = default_weights.copy()
    imd_weights["imd_score_normalized"] = 0.6
    imd_weights["NO2_normalized"] = 0.1
    imd_weights["PM2.5_normalized"] = 0.1
    imd_weights["PM10_normalized"] = 0.1
    imd_weights["respiratory_health_index"] = 0.1
    print("\nIncreased weight on IMD:")
    sensitivity_analysis(merged_df.copy(), imd_weights, scaled_df, index_vars)

    # Increase weight on health
    health_weights = default_weights.copy()
    health_weights["imd_score_normalized"] = 0.1
    health_weights["NO2_normalized"] = 0.1
    health_weights["PM2.5_normalized"] = 0.1
    health_weights["PM10_normalized"] = 0.1
    health_weights["respiratory_health_index"] = 0.6
    print("\nIncreased weight on health:")
    sensitivity_analysis(merged_df.copy(), health_weights, scaled_df, index_vars)

    return merged_df


def sensitivity_analysis(merged_df, weights, scaled_df, index_vars):
    """
    Conduct sensitivity analysis on component weights for the vulnerability index.

    Args:
        merged_df (pd.DataFrame): Merged dataset with health and pollution data
        weights (dict): Dictionary of weights for each component of the vulnerability index
        scaled_df (pd.DataFrame): Standardized data
        index_vars (list): List of index variables

    Returns:
        float: Spearman correlation between the weighted vulnerability index and the IMD
    """
    print("\n--- Conducting Sensitivity Analysis ---")

    # Create weighted vulnerability index
    merged_df["weighted_vulnerability_index"] = (
        weights["imd_score_normalized"] * scaled_df["imd_score_normalized"]
        + weights["NO2_normalized"] * scaled_df["NO2_normalized"]
        + weights["PM2.5_normalized"] * scaled_df["PM2.5_normalized"]
        + weights["PM10_normalized"] * scaled_df["PM10_normalized"]
        + weights["respiratory_health_index"] * scaled_df["respiratory_health_index"]
    )

    # Normalize to 0-100 scale
    min_val = merged_df["weighted_vulnerability_index"].min()
    max_val = merged_df["weighted_vulnerability_index"].max()
    # Add a small constant to the denominator to prevent division by zero
    epsilon = 1e-6
    merged_df["weighted_vulnerability_index"] = (
        100 * (merged_df["weighted_vulnerability_index"] - min_val) / (max_val - min_val + epsilon)
    )

    # Calculate Spearman correlation between weighted vulnerability_index and imd_score_normalized
    valid_data = merged_df[["weighted_vulnerability_index", "imd_score_normalized"]].dropna()
    corr, p_value = spearmanr(
        valid_data["weighted_vulnerability_index"], valid_data["imd_score_normalized"]
    )
    print(
        f"Spearman correlation between weighted vulnerability_index and imd_score_normalized: r = {corr:.3f}, p = {p_value:.6f}"
    )

    return corr


def cluster_analysis(merged_df):
    """
    Perform cluster analysis to identify patterns in the data.

    Args:
        merged_df (pd.DataFrame): Merged dataset with vulnerability index
    """
    print("\n--- Cluster Analysis ---")

    # Select variables for clustering
    cluster_vars = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
        "vulnerability_index",
    ]

    # Handle missing values by dropping rows with NaN values
    print(f"Original dataset shape: {merged_df.shape}")
    df_for_clustering = merged_df.dropna(subset=cluster_vars)
    print(f"Dataset shape after dropping NaN values: {df_for_clustering.shape}")

    # Standardize variables
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_for_clustering[cluster_vars])

    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, "o-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.savefig("outputs/figures/elbow_curve.png", dpi=300)

    # Choose k=4 clusters (this can be adjusted based on the elbow curve)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_for_clustering.loc[:, "cluster"] = kmeans.fit_predict(scaled_data)

    # Merge cluster assignments back to the original dataframe
    merged_df = merged_df.reset_index(drop=True)
    df_for_clustering = df_for_clustering.reset_index(drop=True)

    # Create a mapping from index to cluster
    cluster_mapping = dict(zip(df_for_clustering.index, df_for_clustering["cluster"]))

    # Apply the mapping to the original dataframe
    merged_df["cluster"] = merged_df.index.map(lambda x: cluster_mapping.get(x, np.nan))

    # Analyze clusters
    cluster_summary = (
        merged_df.groupby("cluster")
        .agg(
            {
                "lad_name": "count",
                "imd_score_normalized": "mean",
                "NO2_normalized": "mean",
                "PM2.5_normalized": "mean",
                "respiratory_health_index": "mean",
                "vulnerability_index": "mean",
            }
        )
        .rename(columns={"lad_name": "count"})
    )

    print("\nCluster Summary:")
    print(cluster_summary)

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    plt.figure(figsize=(12, 8))
    for i in range(k):
        # Use the df_for_clustering dataframe which has the same indices as pca_result
        mask = df_for_clustering["cluster"] == i
        if mask.any():
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=f"Cluster {i}")

    plt.title("PCA Visualization of Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig("outputs/figures/cluster_pca.png", dpi=300)

    # Save clustered data
    merged_df.to_csv("outputs/data/lad_clustered.csv", index=False)


def main():
    """Main function to execute the analysis pipeline."""
    np.random.seed(42) # Set seed for reproducibility
    print("Starting Environmental Justice Analysis...")

    # Load data
    unified_df, health_df, wards_df = load_data()

    # Explore data
    explore_data(unified_df, health_df, wards_df)

    # Analyze environmental justice
    analyze_environmental_justice(unified_df)

    # Merge health data with pollution data
    merged_df = merge_health_with_pollution(unified_df, health_df)

    # Analyze health-pollution relationship
    analyze_health_pollution_relationship(merged_df)

    # Create vulnerability index
    merged_df = create_vulnerability_index(merged_df)

    # Perform cluster analysis
    cluster_analysis(merged_df)

    print("\nAnalysis complete. Results saved to the 'outputs' directory.")


if __name__ == "__main__":
    main()
