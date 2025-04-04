"""
Environmental Justice and Health Inequalities Dashboard

This Streamlit application provides an interactive visualization of the relationship
between air pollution, socioeconomic deprivation, and respiratory health outcomes in England.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="Environmental Justice Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define paths
OUTPUT_DATA_PATH = "outputs/data/"
UNIFIED_DATA_PATH = "unified_dataset_with_air_quality.csv"
HEALTH_DATA_PATH = "health_indicators_by_lad.csv"

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #005cb2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
        margin-bottom: 1rem;
        color: #333; /* Ensure text is visible */
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load the datasets for the dashboard."""
    # Check if processed data exists
    if os.path.exists(os.path.join(OUTPUT_DATA_PATH, "lad_clustered.csv")):
        # Load processed data
        lad_data = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, "lad_clustered.csv"))
        lsoa_data = pd.read_csv(UNIFIED_DATA_PATH)
        high_vulnerability = pd.read_csv(
            os.path.join(OUTPUT_DATA_PATH, "high_vulnerability_areas.csv")
        )
        high_injustice = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, "high_injustice_areas.csv"))
    else:
        # If processed data doesn't exist, load raw data
        lsoa_data = pd.read_csv(UNIFIED_DATA_PATH)
        health_data = pd.read_csv(HEALTH_DATA_PATH)

        # Basic aggregation to LAD level
        lad_data = (
            lsoa_data.groupby("lad_code")
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
        lad_data = pd.merge(
            lad_data, health_data, left_on="lad_code", right_on="local_authority_code", how="inner"
        )

        # Create placeholder for high vulnerability and injustice areas
        high_vulnerability = lad_data.sort_values("respiratory_health_index", ascending=False).head(
            20
        )
        high_injustice = lsoa_data.sort_values("env_justice_index", ascending=False).head(20)

    return lad_data, lsoa_data, high_vulnerability, high_injustice


def main():
    """Main function to run the dashboard."""
    # Load data
    lad_data, lsoa_data, high_vulnerability, high_injustice = load_data()

    # Header
    # st.title("Environmental Justice and Health Inequalities in England")
    st.markdown(
        '<div class="main-header">Environmental Justice and Health Inequalities in England</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="highlight">
    This dashboard explores the relationship between air pollution, socioeconomic deprivation,
    and respiratory health outcomes across England. It identifies areas of environmental injustice
    and provides insights for targeted interventions.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        [
            "Overview",
            "Environmental Justice Analysis",
            "Health Outcomes",
            "Vulnerability Index",
            "Cluster Analysis",
            "Intervention Priorities",
        ],
    )

    # Filter options in sidebar
    st.sidebar.markdown("---")
    st.sidebar.title("Filters")

    # Filter by region if available
    if "region" in lad_data.columns:
        selected_region = st.sidebar.multiselect(
            "Select Region(s):", options=sorted(lad_data["region"].unique()), default=[]
        )
    else:
        selected_region = []

    # Filter by pollution level
    pollution_metric = st.sidebar.selectbox("Pollution Metric:", ["NO2", "PM2.5", "PM10", "O3"])

    pollution_threshold = st.sidebar.slider(
        f"{pollution_metric} Threshold:",
        min_value=float(lad_data[pollution_metric].min()),
        max_value=float(lad_data[pollution_metric].max()),
        value=float(lad_data[pollution_metric].median()),
    )

    # Apply filters
    filtered_lad_data = lad_data.copy()
    if selected_region:
        if "region" in filtered_lad_data.columns:
            filtered_lad_data = filtered_lad_data[filtered_lad_data["region"].isin(selected_region)]

    filtered_lad_data = filtered_lad_data[
        filtered_lad_data[pollution_metric] >= pollution_threshold
    ]

    # Display different pages based on selection
    if page == "Overview":
        display_overview(filtered_lad_data, lsoa_data)
    elif page == "Environmental Justice Analysis":
        display_environmental_justice(filtered_lad_data, lsoa_data, high_injustice)
    elif page == "Health Outcomes":
        display_health_outcomes(filtered_lad_data)
    elif page == "Vulnerability Index":
        display_vulnerability_index(filtered_lad_data, high_vulnerability)
    elif page == "Cluster Analysis":
        display_cluster_analysis(filtered_lad_data)
    elif page == "Intervention Priorities":
        display_intervention_priorities(filtered_lad_data, high_vulnerability)


def display_overview(lad_data, lsoa_data):
    """Display the overview page."""
    st.markdown('<div class="sub-header">Project Overview</div>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Local Authorities</div>
        </div>
        """.format(
                len(lad_data)
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">LSOAs</div>
        </div>
        """.format(
                len(lsoa_data)
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Avg NO₂ (μg/m³)</div>
        </div>
        """.format(
                lad_data["NO2"].mean()
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Avg PM2.5 (μg/m³)</div>
        </div>
        """.format(
                lad_data["PM2.5"].mean()
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Research Questions</div>', unsafe_allow_html=True)

    st.markdown(
        """
    1. **Primary Research Question**: To what extent do socioeconomic deprivation and air pollution exposure combine to create "double disadvantage" areas with disproportionately poor respiratory health outcomes?

    2. **Secondary Research Questions**:
       - Is there a threshold effect in air pollution exposure that correlates with significant deterioration in respiratory health?
       - How do different pollutants (NO2, PM2.5, O3) interact to affect health outcomes?
       - What is the spatial distribution of environmental injustice across England?
       - Which areas should be prioritized for pollution reduction interventions to maximize health benefits?
    """
    )



def display_environmental_justice(lad_data, lsoa_data, high_injustice):
    """Display the environmental justice analysis page."""
    st.markdown(
        '<div class="sub-header">Environmental Justice Analysis</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="highlight">
    Environmental justice examines how environmental burdens and benefits are distributed across different 
    socioeconomic groups. This analysis explores the relationship between air pollution exposure and
    socioeconomic deprivation across England.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Pollution vs Deprivation Scatter Plot
    st.markdown(
        '<div class="section-header">Relationship Between Air Pollution and Deprivation</div>',
        unsafe_allow_html=True,
    )

    pollution_metric = st.selectbox(
        "Select Pollution Metric:", ["NO2", "PM2.5", "PM10", "O3"], key="pollution_scatter"
    )

    fig = px.scatter(
        lsoa_data,
        x=pollution_metric,
        y="imd_score_normalized",
        hover_name="lsoa_name",
        hover_data=["lad_name"],
        trendline="ols",
        title=f"{pollution_metric} vs. Index of Multiple Deprivation",
        labels={
            pollution_metric: f"{pollution_metric} Concentration",
            "imd_score_normalized": "IMD Score (Normalized)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate correlation
    correlation = lsoa_data[[pollution_metric, "imd_score_normalized"]].corr().iloc[0, 1]
    st.markdown(f"**Correlation Coefficient:** {correlation:.3f}")

    # Environmental Justice Index Distribution
    st.markdown(
        '<div class="section-header">Environmental Justice Index Distribution</div>',
        unsafe_allow_html=True,
    )

    fig = px.histogram(
        lsoa_data,
        x="env_justice_index",
        nbins=50,
        title="Distribution of Environmental Justice Index",
        labels={"env_justice_index": "Environmental Justice Index"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Areas with highest environmental injustice
    st.markdown(
        '<div class="section-header">Areas with Highest Environmental Injustice</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
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


def display_health_outcomes(lad_data):
    """Display the health outcomes analysis page."""
    st.markdown('<div class="sub-header">Health Outcomes Analysis</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="highlight">
    This section examines the relationship between air pollution and respiratory health outcomes
    at the Local Authority District (LAD) level. It explores how different pollutants correlate
    with various health indicators and identifies potential threshold effects.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Pollution vs Health Scatter Plot
    st.markdown(
        '<div class="section-header">Relationship Between Air Pollution and Health Outcomes</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        pollution_metric = st.selectbox(
            "Select Pollution Metric:", ["NO2", "PM2.5", "PM10", "O3"], key="health_pollution"
        )

    with col2:
        health_metric = st.selectbox(
            "Select Health Metric:",
            [
                "respiratory_health_index",
                "chronic_conditions_normalized",
                "asthma_diabetes_epilepsy_normalized",
                "lrti_children_normalized",
            ],
            key="health_metric",
        )

    health_labels = {
        "respiratory_health_index": "Respiratory Health Index",
        "chronic_conditions_normalized": "Chronic Conditions (Normalized)",
        "asthma_diabetes_epilepsy_normalized": "Asthma, Diabetes & Epilepsy (Normalized)",
        "lrti_children_normalized": "Lower Respiratory Tract Infections in Children (Normalized)",
    }

    fig = px.scatter(
        lad_data,
        x=pollution_metric,
        y=health_metric,
        hover_name="lad_name",
        trendline="ols",
        title=f"{pollution_metric} vs. {health_labels[health_metric]}",
        labels={
            pollution_metric: f"{pollution_metric} Concentration",
            health_metric: health_labels[health_metric],
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate correlation
    correlation = lad_data[[pollution_metric, health_metric]].corr().iloc[0, 1]
    st.markdown(f"**Correlation Coefficient:** {correlation:.3f}")

    # Double Disadvantage Analysis
    st.markdown(
        '<div class="section-header">Double Disadvantage Analysis</div>', unsafe_allow_html=True
    )

    # Create double disadvantage indicator
    lad_data["double_disadvantage"] = (
        lad_data["imd_score_normalized"] > lad_data["imd_score_normalized"].median()
    ) & (
        (lad_data["NO2_normalized"] > lad_data["NO2_normalized"].median())
        | (lad_data["PM2.5_normalized"] > lad_data["PM2.5_normalized"].median())
    )

    # Convert to categorical for better visualization
    lad_data["double_disadvantage_cat"] = lad_data["double_disadvantage"].map(
        {True: "Double Disadvantage", False: "Other Areas"}
    )

    fig = px.box(
        lad_data,
        x="double_disadvantage_cat",
        y="respiratory_health_index",
        color="double_disadvantage_cat",
        title="Respiratory Health Index by Double Disadvantage Status",
        labels={
            "double_disadvantage_cat": "",
            "respiratory_health_index": "Respiratory Health Index",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate statistics
    double_disadv = lad_data[lad_data["double_disadvantage"]]["respiratory_health_index"]
    others = lad_data[~lad_data["double_disadvantage"]]["respiratory_health_index"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Double Disadvantage Areas Mean:** {double_disadv.mean():.3f}")
        st.markdown(f"**Other Areas Mean:** {others.mean():.3f}")

    with col2:
        from scipy.stats import ttest_ind

        t_stat, p_val = ttest_ind(double_disadv, others)
        st.markdown(f"**T-test Statistic:** {t_stat:.3f}")
        st.markdown(f"**P-value:** {p_val:.6f}")
        if p_val < 0.05:
            st.markdown("**Result:** Statistically significant difference")
        else:
            st.markdown("**Result:** No statistically significant difference")


def display_vulnerability_index(lad_data, high_vulnerability):
    """Display the vulnerability index page."""
    st.markdown('<div class="sub-header">Vulnerability Index</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="highlight">
    The Vulnerability Index is a composite measure that combines air pollution exposure, 
    socioeconomic deprivation, and respiratory health indicators. It helps identify areas 
    that are most vulnerable to the combined effects of these factors.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if vulnerability index exists
    if "vulnerability_index" not in lad_data.columns:
        st.warning(
            "Vulnerability Index not found in the data. Please run the full analysis script to generate this index."
        )

        # Create a simple placeholder vulnerability index
        st.info("Creating a simple placeholder vulnerability index for demonstration purposes.")

        # Select variables for the index
        index_vars = [
            "imd_score_normalized",
            "NO2_normalized",
            "PM2.5_normalized",
            "respiratory_health_index",
        ]

        # Check if all required columns exist
        missing_cols = [col for col in index_vars if col not in lad_data.columns]
        if missing_cols:
            st.error(f"Missing columns for vulnerability index: {', '.join(missing_cols)}")
            return

        # Create a simple average-based index
        lad_data["vulnerability_index"] = lad_data[index_vars].mean(axis=1)

        # Normalize to 0-100 scale
        min_val = lad_data["vulnerability_index"].min()
        max_val = lad_data["vulnerability_index"].max()
        lad_data["vulnerability_index"] = (
            100 * (lad_data["vulnerability_index"] - min_val) / (max_val - min_val)
        )

        # Update high vulnerability areas
        high_vulnerability = lad_data.sort_values("vulnerability_index", ascending=False).head(20)

    # Vulnerability Index Distribution
    st.markdown(
        '<div class="section-header">Vulnerability Index Distribution</div>', unsafe_allow_html=True
    )

    fig = px.histogram(
        lad_data,
        x="vulnerability_index",
        nbins=30,
        title="Distribution of Vulnerability Index",
        labels={"vulnerability_index": "Vulnerability Index (0-100)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Areas with highest vulnerability
    st.markdown(
        '<div class="section-header">Areas with Highest Vulnerability</div>', unsafe_allow_html=True
    )

    # Determine which columns to display
    display_cols = ["lad_name", "vulnerability_index"]
    for col in [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
    ]:
        if col in high_vulnerability.columns:
            display_cols.append(col)

    st.dataframe(high_vulnerability[display_cols])

    # Vulnerability components analysis
    st.markdown(
        '<div class="section-header">Vulnerability Components Analysis</div>',
        unsafe_allow_html=True,
    )

    # Check if we have the necessary columns
    component_cols = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
    ]
    missing_cols = [col for col in component_cols if col not in lad_data.columns]

    if not missing_cols:
        # Create radar chart for top 5 most vulnerable areas
        top_areas = high_vulnerability.head(5)

        fig = go.Figure()

        for i, row in top_areas.iterrows():
            fig.add_trace(
                go.Scatterpolar(
                    r=[
                        row["imd_score_normalized"],
                        row["NO2_normalized"],
                        row["PM2.5_normalized"],
                        row["respiratory_health_index"],
                    ],
                    theta=["IMD Score", "NO2", "PM2.5", "Respiratory Health"],
                    fill="toself",
                    name=row["lad_name"],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Vulnerability Components for Top 5 Most Vulnerable Areas",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Missing columns for vulnerability components analysis: {', '.join(missing_cols)}")


def display_cluster_analysis(lad_data):
    """Display the cluster analysis page."""
    st.markdown('<div class="sub-header">Cluster Analysis</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="highlight">
    Cluster analysis groups areas with similar characteristics together, helping to identify
    patterns and typologies of environmental justice and health outcomes. This can inform
    targeted policy interventions for different types of areas.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if cluster column exists
    if "cluster" not in lad_data.columns:
        st.warning(
            "Cluster assignments not found in the data. Please run the full analysis script to generate clusters."
        )

        # Create simple clusters for demonstration
        st.info("Creating simple clusters for demonstration purposes.")

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Select variables for clustering
        cluster_vars = [
            "imd_score_normalized",
            "NO2_normalized",
            "PM2.5_normalized",
            "respiratory_health_index",
        ]

        # Check if all required columns exist
        missing_cols = [col for col in cluster_vars if col not in lad_data.columns]
        if missing_cols:
            st.error(f"Missing columns for clustering: {', '.join(missing_cols)}")
            return

        # Standardize variables
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(lad_data[cluster_vars])

        # Perform clustering
        k = 4  # Number of clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        lad_data["cluster"] = kmeans.fit_predict(scaled_data)

    # Cluster Summary
    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)

    # Calculate cluster statistics
    cluster_summary = (
        lad_data.groupby("cluster")
        .agg(
            {
                "lad_name": "count",
            }
        )
        .rename(columns={"lad_name": "count"})
    )

    # Add mean values for key variables if they exist
    for col in [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
        "vulnerability_index",
    ]:
        if col in lad_data.columns:
            cluster_summary[col] = lad_data.groupby("cluster")[col].mean()

    st.dataframe(cluster_summary)

    # Cluster Visualization
    st.markdown('<div class="section-header">Cluster Visualization</div>', unsafe_allow_html=True)

    # Check if we have the necessary columns for visualization
    viz_cols = [
        "imd_score_normalized",
        "NO2_normalized",
        "PM2.5_normalized",
        "respiratory_health_index",
    ]
    missing_cols = [col for col in viz_cols if col not in lad_data.columns]

    if not missing_cols:
        # Create scatter plot matrix
        fig = px.scatter_matrix(
            lad_data,
            dimensions=viz_cols,
            color="cluster",
            title="Cluster Visualization - Scatter Plot Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            lad_data,
            dimensions=viz_cols,
            color="cluster",
            title="Cluster Visualization - Parallel Coordinates",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Missing columns for cluster visualization: {', '.join(missing_cols)}")


def display_intervention_priorities(lad_data, high_vulnerability):
    """Display the intervention priorities page."""
    st.markdown('<div class="sub-header">Intervention Priorities</div>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="highlight">
    This section identifies priority areas for pollution reduction interventions based on the
    vulnerability index and potential health benefits. It provides a framework for targeting
    resources to maximize health improvements and address environmental injustice.
    </div>
    """,
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()
