"""
Advanced Cluster Analysis for Environmental Justice Project

This script implements advanced cluster analysis techniques including:
- Silhouette analysis for optimal cluster validation
- Feature importance for each cluster
- SHAP values to explain cluster assignments
- Enhanced visualizations

This builds on the basic cluster analysis in env_justice_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import shap
from sklearn.ensemble import RandomForestClassifier
import os

# Set the plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Create output directories if they don't exist
os.makedirs("outputs/advanced_clustering", exist_ok=True)


def load_data():
    """
    Load the merged dataset with vulnerability index.

    Returns:
        pd.DataFrame: Merged dataset with vulnerability index
    """
    # Try to load the processed data if it exists
    if os.path.exists("outputs/data/lad_with_vulnerability_index.csv"):
        merged_df = pd.read_csv("outputs/data/lad_with_vulnerability_index.csv")
        print(f"Loaded processed data with {len(merged_df)} LADs")
    else:
        # If processed data doesn't exist, load and merge raw data
        print("Processed data not found. Loading and merging raw data...")
        unified_df = pd.read_csv("unified_dataset_with_air_quality.csv")
        health_df = pd.read_csv("health_indicators_by_lad.csv")

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
            lad_aggregated,
            health_df,
            left_on="lad_code",
            right_on="local_authority_code",
            how="inner",
        )

        print(f"Merged raw data with {len(merged_df)} LADs")

    return merged_df


def perform_silhouette_analysis(data, max_clusters=10):
    """
    Perform silhouette analysis to determine the optimal number of clusters.

    Args:
        data (np.ndarray): Standardized data for clustering
        max_clusters (int): Maximum number of clusters to evaluate

    Returns:
        int: Optimal number of clusters
    """
    print("\nPerforming silhouette analysis to determine optimal number of clusters...")

    silhouette_avg_scores = []
    k_range = range(2, max_clusters + 1)

    for k in k_range:
        # Initialize KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the average silhouette score is {silhouette_avg:.3f}")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_avg_scores, "o-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Optimal k")
    plt.grid(True)
    plt.savefig("outputs/advanced_clustering/silhouette_analysis.png", dpi=300)

    # Find optimal number of clusters (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_avg_scores)]
    print(f"Optimal number of clusters based on silhouette analysis: {optimal_k}")

    return optimal_k


def visualize_silhouette_plot(data, optimal_k):
    """
    Create a detailed silhouette plot for the optimal number of clusters.

    Args:
        data (np.ndarray): Standardized data for clustering
        optimal_k (int): Optimal number of clusters
    """
    print(f"\nCreating detailed silhouette plot for k={optimal_k}...")

    # Initialize KMeans with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    # Calculate silhouette scores for each sample
    silhouette_vals = silhouette_samples(data, cluster_labels)

    # Create silhouette plot
    plt.figure(figsize=(12, 8))

    y_lower = 10
    for i in range(optimal_k):
        # Get silhouette scores for samples in cluster i
        ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        ith_cluster_silhouette_vals.sort()

        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / optimal_k)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with cluster numbers
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute new y_lower for next plot
        y_lower = y_upper + 10

    # Get average silhouette score
    avg_score = np.mean(silhouette_vals)

    # Plot average silhouette score
    plt.axvline(x=avg_score, color="red", linestyle="--")
    plt.text(
        avg_score + 0.02,
        plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
        f"Average: {avg_score:.3f}",
        color="red",
    )

    plt.title(f"Silhouette Plot for k={optimal_k} Clusters")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    plt.yticks([])  # Clear y-axis labels
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/silhouette_plot.png", dpi=300)


def calculate_feature_importance(data, cluster_labels, feature_names):
    """
    Calculate feature importance for each cluster using a Random Forest classifier.

    Args:
        data (np.ndarray): Standardized data for clustering
        cluster_labels (np.ndarray): Cluster assignments
        feature_names (list): Names of features

    Returns:
        pd.DataFrame: Feature importance for each cluster
    """
    print("\nCalculating feature importance for each cluster...")

    # Train a Random Forest classifier to predict cluster labels
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(data, cluster_labels)

    # Get feature importance
    importances = rf.feature_importances_

    # Create DataFrame with feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance for Cluster Assignment")
    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/feature_importance.png", dpi=300)

    # Calculate feature importance for each cluster
    cluster_importance = {}
    for cluster in np.unique(cluster_labels):
        # Create binary classification problem: this cluster vs. others
        binary_labels = (cluster_labels == cluster).astype(int)

        # Train a Random Forest for this specific cluster
        rf_cluster = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_cluster.fit(data, binary_labels)

        # Get feature importance for this cluster
        cluster_importance[f"Cluster {cluster}"] = rf_cluster.feature_importances_

    # Create DataFrame with cluster-specific feature importances
    cluster_importance_df = pd.DataFrame(cluster_importance, index=feature_names)

    # Plot cluster-specific feature importance
    plt.figure(figsize=(14, 10))
    cluster_importance_df.plot(kind="bar", figsize=(14, 10))
    plt.title("Feature Importance by Cluster")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/cluster_feature_importance.png", dpi=300)

    return feature_importance_df, cluster_importance_df


def calculate_shap_values(data, cluster_labels, feature_names):
    """
    Calculate SHAP values to explain cluster assignments.

    Args:
        data (np.ndarray): Standardized data for clustering
        cluster_labels (np.ndarray): Cluster assignments
        feature_names (list): Names of features
    """
    print("\nCalculating SHAP values to explain cluster assignments...")

    # Train a Random Forest classifier to predict cluster labels
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(data, cluster_labels)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(data)

    # Plot summary of SHAP values
    plt.figure(figsize=(12, 10))

    # Check if shap_values is a list (one array per class) or a single array
    if isinstance(shap_values, list):
        # For multi-class, use all classes combined
        shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    else:
        # For single output, use the array directly
        shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)

    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/shap_summary.png", dpi=300)

    # Plot SHAP values for each cluster
    unique_clusters = np.unique(cluster_labels)

    # Only attempt to plot individual cluster SHAP values if the structure is appropriate
    if isinstance(shap_values, list) and len(shap_values) == len(unique_clusters):
        for i, cluster in enumerate(unique_clusters):
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[i], data, feature_names=feature_names, show=False)
            plt.title(f"SHAP Values for Cluster {cluster}")
            plt.tight_layout()
            plt.savefig(f"outputs/advanced_clustering/shap_cluster_{cluster}.png", dpi=300)
            plt.close()
    else:
        print(
            "Warning: SHAP values structure doesn't match the expected format for per-cluster plots."
        )
        print("Skipping individual cluster SHAP plots.")


def visualize_clusters_pca(data, cluster_labels, feature_names):
    """
    Visualize clusters using PCA with enhanced visualization.

    Args:
        data (np.ndarray): Standardized data for clustering
        cluster_labels (np.ndarray): Cluster assignments
        feature_names (list): Names of features
    """
    print("\nVisualizing clusters using PCA...")

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_ * 100

    # Create DataFrame for plotting
    pca_df = pd.DataFrame(
        {"PC1": pca_result[:, 0], "PC2": pca_result[:, 1], "Cluster": cluster_labels}
    )

    # Plot PCA results
    plt.figure(figsize=(12, 10))

    # Create scatter plot with larger points and improved aesthetics
    scatter = sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="viridis",
        data=pca_df,
        s=100,  # Larger point size
        alpha=0.7,  # Slight transparency
        edgecolor="w",  # White edge for better visibility
        linewidth=0.5,
    )

    # Add title and labels with variance explained
    plt.title("PCA Visualization of Clusters", fontsize=16)
    plt.xlabel(f"Principal Component 1 ({variance_explained[0]:.1f}% Variance)", fontsize=14)
    plt.ylabel(f"Principal Component 2 ({variance_explained[1]:.1f}% Variance)", fontsize=14)

    # Improve legend
    plt.legend(
        title="Cluster", title_fontsize=14, fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Calculate and plot cluster centroids
    for cluster in np.unique(cluster_labels):
        centroid = pca_df[pca_df["Cluster"] == cluster][["PC1", "PC2"]].mean()
        plt.scatter(
            centroid["PC1"],
            centroid["PC2"],
            s=200,
            marker="X",
            color="black",
            edgecolor="white",
            linewidth=2,
            label=f"Centroid {cluster}" if cluster == 0 else "",
        )
        plt.annotate(
            f"Cluster {cluster}",
            (centroid["PC1"], centroid["PC2"]),
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="bottom",
            xytext=(0, 10),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/pca_clusters.png", dpi=300)

    # Calculate PCA loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Create loadings DataFrame
    loadings_df = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=feature_names)

    # Plot PCA loadings
    plt.figure(figsize=(12, 10))

    # Plot arrows for each feature
    for i, feature in enumerate(feature_names):
        plt.arrow(
            0,
            0,
            loadings[i, 0],
            loadings[i, 1],
            head_width=0.05,
            head_length=0.05,
            fc="blue",
            ec="blue",
            alpha=0.5,
        )
        plt.text(
            loadings[i, 0] * 1.15,
            loadings[i, 1] * 1.15,
            feature,
            color="blue",
            ha="center",
            va="center",
            fontsize=12,
        )

    # Add circle
    circle = plt.Circle((0, 0), 1, fc="white", ec="black", linestyle="--", alpha=0.1)
    plt.gca().add_patch(circle)

    # Set plot limits and labels
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel(f"Principal Component 1 ({variance_explained[0]:.1f}% Variance)", fontsize=14)
    plt.ylabel(f"Principal Component 2 ({variance_explained[1]:.1f}% Variance)", fontsize=14)
    plt.title("PCA Loadings (Feature Contributions)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/pca_loadings.png", dpi=300)

    return pca_df, loadings_df


def analyze_cluster_profiles(merged_df, cluster_labels):
    """
    Analyze and visualize the profiles of each cluster.

    Args:
        merged_df (pd.DataFrame): Merged dataset with all features
        cluster_labels (np.ndarray): Cluster assignments
    """
    print("\nAnalyzing cluster profiles...")

    # Add cluster labels to the DataFrame
    merged_df = merged_df.copy()
    merged_df["cluster"] = cluster_labels

    # Calculate cluster statistics
    cluster_stats = merged_df.groupby("cluster").agg(
        {
            "lad_name": "count",
            "imd_score_normalized": ["mean", "std"],
            "NO2_normalized": ["mean", "std"],
            "PM2.5_normalized": ["mean", "std"],
            "respiratory_health_index": ["mean", "std"],
            "vulnerability_index": ["mean", "std"],
        }
    )

    # Flatten MultiIndex columns
    cluster_stats.columns = ["_".join(col).strip() for col in cluster_stats.columns.values]

    # Rename count column
    cluster_stats = cluster_stats.rename(columns={"lad_name_count": "count"})

    # Save cluster statistics
    cluster_stats.to_csv("outputs/advanced_clustering/cluster_statistics.csv")

    # Create radar chart for cluster profiles
    # Select features for radar chart
    radar_features = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
        "vulnerability_index",
    ]

    # Calculate mean values for each cluster
    radar_data = merged_df.groupby("cluster")[radar_features].mean()

    # Normalize data for radar chart (0-1 scale)
    radar_data_normalized = radar_data.copy()
    for feature in radar_features:
        min_val = merged_df[feature].min()
        max_val = merged_df[feature].max()
        radar_data_normalized[feature] = (radar_data[feature] - min_val) / (max_val - min_val)

    # Create radar chart
    plt.figure(figsize=(12, 10))

    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up plot
    ax = plt.subplot(111, polar=True)

    # Add feature labels
    plt.xticks(angles[:-1], radar_features, size=12)

    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)

    # Plot each cluster
    for cluster in radar_data_normalized.index:
        values = radar_data_normalized.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title("Cluster Profiles", size=16)
    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/cluster_radar_chart.png", dpi=300)

    # Create boxplots for each feature by cluster
    for feature in radar_features:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="cluster", y=feature, data=merged_df)
        plt.title(f"Distribution of {feature} by Cluster", size=16)
        plt.xlabel("Cluster", size=14)
        plt.ylabel(feature, size=14)
        plt.tight_layout()
        plt.savefig(f"outputs/advanced_clustering/boxplot_{feature}.png", dpi=300)

    return cluster_stats, radar_data_normalized

    # Calculate cluster statistics
    cluster_stats = merged_df.groupby("cluster").agg(
        {
            "lad_name": "count",
            "imd_score_normalized": ["mean", "std"],
            "NO2_normalized": ["mean", "std"],
            "PM2.5_normalized": ["mean", "std"],
            "respiratory_health_index": ["mean", "std"],
            "vulnerability_index": ["mean", "std"],
        }
    )

    # Flatten MultiIndex columns
    cluster_stats.columns = ["_".join(col).strip() for col in cluster_stats.columns.values]

    # Rename count column
    cluster_stats = cluster_stats.rename(columns={"lad_name_count": "count"})

    # Save cluster statistics
    cluster_stats.to_csv("outputs/advanced_clustering/cluster_statistics.csv")

    # Create radar chart for cluster profiles
    # Select features for radar chart
    radar_features = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
        "vulnerability_index",
    ]

    # Calculate mean values for each cluster
    radar_data = merged_df.groupby("cluster")[radar_features].mean()

    # Normalize data for radar chart (0-1 scale)
    radar_data_normalized = radar_data.copy()
    for feature in radar_features:
        min_val = merged_df[feature].min()
        max_val = merged_df[feature].max()
        radar_data_normalized[feature] = (radar_data[feature] - min_val) / (max_val - min_val)

    # Create radar chart
    plt.figure(figsize=(12, 10))

    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up plot
    ax = plt.subplot(111, polar=True)

    # Add feature labels
    plt.xticks(angles[:-1], radar_features, size=12)

    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)

    # Plot each cluster
    for cluster in radar_data_normalized.index:
        values = radar_data_normalized.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title("Cluster Profiles", size=16)
    plt.tight_layout()
    plt.savefig("outputs/advanced_clustering/cluster_radar_chart.png", dpi=300)

    # Create boxplots for each feature by cluster
    for feature in radar_features:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="cluster", y=feature, data=merged_df)
        plt.title(f"Distribution of {feature} by Cluster", size=16)
        plt.xlabel("Cluster", size=14)
        plt.ylabel(feature, size=14)
        plt.tight_layout()
        plt.savefig(f"outputs/advanced_clustering/boxplot_{feature}.png", dpi=300)

    return cluster_stats, radar_data_normalized


def main():
    """Main function to execute the advanced cluster analysis."""
    print("Starting Advanced Cluster Analysis...")

    # Load data
    merged_df = load_data()

    # Select variables for clustering
    cluster_vars = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
        "vulnerability_index",
    ]

    # Handle missing values
    print(f"\nOriginal dataset shape: {merged_df.shape}")
    df_for_clustering = merged_df.dropna(subset=cluster_vars)
    print(f"Dataset shape after dropping NaN values: {df_for_clustering.shape}")

    # Standardize variables
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_for_clustering[cluster_vars])

    # Perform silhouette analysis to determine optimal number of clusters
    optimal_k = perform_silhouette_analysis(scaled_data)

    # Create detailed silhouette plot
    visualize_silhouette_plot(scaled_data, optimal_k)

    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Calculate feature importance
    feature_importance, cluster_importance = calculate_feature_importance(
        scaled_data, cluster_labels, cluster_vars
    )

    # Calculate SHAP values
    try:
        calculate_shap_values(scaled_data, cluster_labels, cluster_vars)
    except Exception as e:
        print(f"Warning: SHAP analysis failed with error: {str(e)}")
        print("Continuing with other analyses...")

    # Visualize clusters using PCA
    pca_df, loadings_df = visualize_clusters_pca(scaled_data, cluster_labels, cluster_vars)

    # Analyze cluster profiles
    cluster_stats, radar_data = analyze_cluster_profiles(df_for_clustering, cluster_labels)

    # Print cluster summary
    print("\nCluster Summary:")
    print(
        cluster_stats[
            [
                "count",
                "imd_score_normalized_mean",
                "NO2_normalized_mean",
                "PM2.5_normalized_mean",
                "respiratory_health_index_mean",
                "vulnerability_index_mean",
            ]
        ]
    )

    # Save cluster assignments
    df_for_clustering.loc[:, "cluster"] = cluster_labels
    df_for_clustering.to_csv(
        "outputs/advanced_clustering/lad_with_advanced_clusters.csv", index=False
    )

    print(
        "\nAdvanced cluster analysis complete. Results saved to the 'outputs/advanced_clustering' directory."
    )

    # Perform clustering without respiratory_health_index (KMeans)
    print("\nPerforming clustering without respiratory_health_index (KMeans)...")
    cluster_vars_no_health = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "vulnerability_index",
    ]
    df_for_clustering_no_health = merged_df.dropna(subset=cluster_vars_no_health).copy()
    scaled_data_no_health = scaler.fit_transform(
        df_for_clustering_no_health[cluster_vars_no_health]
    )
    optimal_k_no_health = perform_silhouette_analysis(scaled_data_no_health)
    kmeans_no_health = KMeans(n_clusters=optimal_k_no_health, random_state=42, n_init=10)
    cluster_labels_no_health = kmeans_no_health.fit_predict(scaled_data_no_health)
    df_for_clustering_no_health["cluster_kmeans_no_health"] = cluster_labels_no_health

    # Analyze cluster profiles (KMeans without respiratory_health_index)
    print("\nAnalyzing KMeans cluster profiles (without respiratory_health_index)...")
    cluster_stats_kmeans_no_health, radar_data_kmeans_no_health = analyze_cluster_profiles(
        df_for_clustering_no_health, cluster_labels_no_health
    )

    # Print cluster summary (KMeans without respiratory_health_index)
    print("\nKMeans Cluster Summary (without respiratory_health_index):")
    print(
        cluster_stats_kmeans_no_health[
            [
                "count",
                "imd_score_normalized_mean",
                "NO2_normalized_mean",
                "PM2.5_normalized_mean",
                "respiratory_health_index_mean",
                "vulnerability_index_mean",
            ]
        ]
    )

    df_for_clustering_no_health.to_csv(
        "outputs/advanced_clustering/lad_with_kmeans_clusters_no_health.csv", index=False
    )

    # Implement DBSCAN clustering
    print("\nPerforming DBSCAN clustering...")
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels_dbscan = dbscan.fit_predict(scaled_data)
    df_for_clustering["cluster_dbscan"] = cluster_labels_dbscan

    # Analyze cluster profiles (DBSCAN)
    print("\nAnalyzing DBSCAN cluster profiles...")
    cluster_stats_dbscan, radar_data_dbscan = analyze_cluster_profiles(
        df_for_clustering, cluster_labels_dbscan
    )

    # Print cluster summary (DBSCAN)
    print("\nDBSCAN Cluster Summary:")
    print(
        cluster_stats_dbscan[
            [
                "count",
                "imd_score_normalized_mean",
                "NO2_normalized_mean",
                "PM2.5_normalized_mean",
                "respiratory_health_index_mean",
                "vulnerability_index_mean",
            ]
        ]
    )

    df_for_clustering.to_csv(
        "outputs/advanced_clustering/lad_with_dbscan_clusters.csv", index=False
    )

    # Implement Gaussian Mixture Model (GMM) clustering
    print("\nPerforming Gaussian Mixture Model (GMM) clustering...")
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    cluster_labels_gmm = gmm.fit_predict(scaled_data)
    df_for_clustering["cluster_gmm"] = cluster_labels_gmm

    # Analyze cluster profiles (GMM)
    print("\nAnalyzing GMM cluster profiles...")
    cluster_stats_gmm, radar_data_gmm = analyze_cluster_profiles(
        df_for_clustering, cluster_labels_gmm
    )

    # Print cluster summary (GMM)
    print("\nGMM Cluster Summary:")
    print(
        cluster_stats_gmm[
            [
                "count",
                "imd_score_normalized_mean",
                "NO2_normalized_mean",
                "PM2.5_normalized_mean",
                "respiratory_health_index_mean",
                "vulnerability_index_mean",
            ]
        ]
    )

    df_for_clustering.to_csv("outputs/advanced_clustering/lad_with_gmm_clusters.csv", index=False)

    print(
        "\nAdvanced cluster analysis complete. Results saved to the 'outputs/advanced_clustering' directory."
    )


if __name__ == "__main__":
    main()
