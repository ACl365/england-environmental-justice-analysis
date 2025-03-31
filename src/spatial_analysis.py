"""
Consolidated Spatial Analysis for Environmental Justice Project

This module performs comprehensive spatial analysis to identify patterns and relationships
in environmental justice indicators across England. It implements:

- LSOA-level Moran's I (Global and Local) using validated ONS boundaries to identify
  statistically significant spatial clusters and outliers.
- LSOA-level Getis-Ord Gi* hotspot analysis to detect areas with high or low values
  surrounded by similar values.
- LAD-level Spatial Regression (OLS and Spatial Lag) to account for spatial dependence
  and neighborhood effects in environmental justice relationships.
- Robust spatial weights creation with appropriate validation and diagnostics.
- Publication-quality visualizations of spatial patterns and relationships.

The analysis addresses the ecological fallacy by conducting analysis at multiple scales
and explicitly modeling spatial dependence through spatial regression techniques.

References:
- Anselin, L. (1995). Local indicators of spatial association—LISA. Geographical Analysis, 27(2), 93-115.
- Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics.
  Geographical Analysis, 24(3), 189-206.
- LeSage, J., & Pace, R. K. (2009). Introduction to spatial econometrics. CRC press.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from shapely.geometry import Polygon, MultiPolygon
import warnings
import traceback

# --- Dependency Check ---
print("--- Checking Dependencies ---")
requirements = {
    "geopandas": False,
    "libpysal": False,
    "esda": False,
    "spreg": False,
    "matplotlib": False,
    "seaborn": False,
    "numpy": False,
    "pandas": False,
    "shapely": False,
}
try:
    import pandas

    requirements["pandas"] = True
except ImportError:
    print("✗ pandas not found")
try:
    import numpy

    requirements["numpy"] = True
except ImportError:
    print("✗ numpy not found")
try:
    import matplotlib

    requirements["matplotlib"] = True
except ImportError:
    print("✗ matplotlib not found")
try:
    import seaborn

    requirements["seaborn"] = True
except ImportError:
    print("✗ seaborn not found")
try:
    import geopandas

    requirements["geopandas"] = True
except ImportError:
    print("✗ geopandas not found")
try:
    import shapely

    requirements["shapely"] = True
except ImportError:
    print("✗ shapely not found")
try:
    import libpysal as lps
    from libpysal.weights import Queen, KNN, W

    requirements["libpysal"] = True
except ImportError:
    print("✗ libpysal not found")
try:
    import esda
    from esda.moran import Moran, Moran_Local
    from esda.getisord import G_Local

    requirements["esda"] = True
except ImportError:
    print("✗ esda not found")
try:
    import spreg
    from spreg import OLS, ML_Lag

    requirements["spreg"] = True
except ImportError:
    print("✗ spreg not found")

PYSAL_AVAILABLE = requirements["libpysal"]
ESDA_AVAILABLE = requirements["esda"]
SPREG_AVAILABLE = requirements["spreg"]
GEOPANDAS_AVAILABLE = requirements["geopandas"]

if not all(requirements.values()):
    print("\nError: Missing critical dependencies. Please install required packages.")
    # Consider exiting if critical ones like pandas/geopandas/libpysal are missing
    # import sys
    # sys.exit(1)
else:
    print("\n✓ All major dependencies seem available.")
print("--------------------------\n")


# --- Constants and Setup ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # Assumes script is in src/
DATA_DIR = os.path.join(BASE_DIR, "data")
GEO_DIR = os.path.join(DATA_DIR)  # Assumes GeoJSONs are here
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "spatial_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File Paths (adjust if your structure is different)
UNIFIED_CSV_PATH = os.path.join(
    DATA_DIR, "processed", "unified_datasets", "unified_dataset_with_air_quality.csv"
)
# VALID LSOA Boundaries File - Using ONS December 2021 boundaries (most recent validated version)
# Source: Office for National Statistics, 2021 Census Geography
LSOA_GEOJSON_PATH = os.path.join(
    GEO_DIR,
    "processed/spatial/LSOA_Dec_2021_Boundaries_EW_BGC_V2.geojson",
)
# VALID LAD Boundaries File - Using ONS December 2021 boundaries (most recent validated version)
# Source: Office for National Statistics, 2021 Census Geography
LAD_GEOJSON_PATH = os.path.join(GEO_DIR, "processed/spatial/LAD_Dec_2021_GB_BGC_V2.geojson")

# Fallback paths if processed files don't exist
LSOA_FALLBACK_PATH = os.path.join(
    GEO_DIR,
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-4299016806856585929.geojson",
)
LAD_FALLBACK_PATH = os.path.join(GEO_DIR, "LAD_Dec_2021_GB_BFC_2022_-8975151699474964544.geojson")

TARGET_CRS = "EPSG:27700"  # British National Grid

# Variables for Analysis
LSOA_VARIABLES = [
    "env_justice_index",
    "air_pollution_index",
    "imd_score_normalized",
    "NO2",
    "PM2.5",
]
LAD_REGRESSION_Y = "env_justice_index"  # Example: using aggregated env_justice_index
LAD_REGRESSION_X = ["imd_score_normalized", "NO2", "PM2.5"]  # Aggregated versions

# Plotting Style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("default")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# --- Helper Functions ---


def fix_geometries(gdf, buffer_dist=0):
    """Fix invalid geometries using buffer(0) or other methods."""
    if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
        return gdf

    initial_invalid = (~gdf.geometry.is_valid).sum()
    if initial_invalid == 0:
        # print("  All geometries are valid.")
        return gdf

    print(f"  Found {initial_invalid} invalid geometries. Attempting fix...")
    # Use buffer(0) - a common trick to fix self-intersections and other topological errors
    gdf["geometry"] = gdf.geometry.buffer(buffer_dist)

    remaining_invalid = (~gdf.geometry.is_valid).sum()
    if remaining_invalid > 0:
        print(
            f"  Warning: {remaining_invalid} geometries still invalid after buffer({buffer_dist})."
        )
        # Add make_valid as a secondary attempt if needed and shapely version supports it
    else:
        print(f"  Fix successful for {initial_invalid} geometries.")
    return gdf


def load_geo_data(path, default_crs="EPSG:4326"):
    """Load geospatial data with error handling and basic checks."""
    if not GEOPANDAS_AVAILABLE:
        return None
    try:
        gdf = gpd.read_file(path)
        print(f"  Successfully loaded: {os.path.basename(path)} ({len(gdf)} features)")
        if gdf.crs is None:
            print(f"  Warning: CRS is missing. Assuming {default_crs}.")
            gdf.set_crs(default_crs, inplace=True)

        # Basic geometry check
        if "geometry" not in gdf.columns or gdf.geometry.isnull().all():
            print(" Error: No valid geometry column found.")
            return None
        if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            print(
                f" Warning: Contains non-polygon geometries: {gdf.geom_type.unique()}. May affect contiguity weights."
            )

        gdf = fix_geometries(gdf)  # Attempt to fix invalid geometries
        return gdf
    except Exception as e:
        print(f" Error loading {os.path.basename(path)}: {e}")
        return None


def create_spatial_weights(gdf, method="queen", id_column=None, **kwargs):
    """
    Create spatial weights matrix with comprehensive error handling and validation.
    
    This function creates a spatial weights matrix that defines the neighborhood
    structure for spatial analysis. The choice of spatial weights is critical as
    it directly affects the results of spatial autocorrelation statistics and
    spatial regression models.
    
    Spatial weights options:
    
    1. Queen contiguity (default): Areas sharing any boundary point (edge or vertex)
       are considered neighbors. This is the most common choice for irregular polygons
       like administrative boundaries, as it ensures a more connected graph and reduces
       the number of islands. Recommended for LSOA/LAD analysis as it captures the
       complex boundary relationships between administrative areas.
       
    2. KNN (k-nearest neighbors): Each area has exactly k neighbors based on centroid
       distance. This ensures no islands but may create asymmetric relationships.
       Useful when dealing with highly irregular polygons or when you want to ensure
       a minimum number of neighbors for each area. May be appropriate for rural areas
       where contiguity-based weights might result in isolated areas.
    
    The weights are row-standardized by default, meaning each row sums to 1.
    This is standard practice for spatial analysis as it makes the spatial lag
    a weighted average of neighboring values.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing geometries
        method (str): Spatial weights method ('queen' or 'knn')
        id_column (str, optional): Column to use as ID for weights
        **kwargs: Additional arguments for specific weight types
                 (e.g., 'k' for KNN weights)
    
    Returns:
        libpysal.weights.W: Spatial weights object or None if creation fails
    """
    if not PYSAL_AVAILABLE:
        return None
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
        print(" Error: Invalid GeoDataFrame for weight creation.")
        return None

    print(f"\nCreating {method.upper()} spatial weights...")
    
    # Justify the choice of weights method
    if method.lower() == "queen":
        print("  Using Queen contiguity weights: areas sharing any boundary point are neighbors.")
        print("  This is appropriate for administrative boundaries like LSOAs/LADs.")
    elif method.lower() == "knn":
        k = kwargs.get("k", 5)
        print(f"  Using K-nearest neighbors (k={k}): each area has exactly {k} neighbors.")
        print("  This ensures no islands but may create asymmetric relationships.")

    # Ensure unique index or ID column
    if id_column and id_column in gdf.columns:
        if not gdf[id_column].is_unique:
            print(f" Warning: ID column '{id_column}' is not unique. Using index.")
            gdf = gdf.reset_index(drop=True)
            id_variable_arg = None
            use_index_arg = True
        else:
            gdf = gdf.set_index(id_column)
            id_variable_arg = gdf.index.name  # Use index name
            use_index_arg = True  # Use the index we just set
    else:
        # Use default index if no id_column provided or it's missing/invalid
        if not gdf.index.is_unique:
            print(f" Warning: Default index is not unique. Resetting index.")
            gdf = gdf.reset_index(drop=True)
        id_variable_arg = None
        use_index_arg = True

    try:
        if method.lower() == "queen":
            # Queen contiguity: considers areas sharing a border or a vertex as neighbors
            w = Queen.from_dataframe(
                gdf, use_index=use_index_arg, ids=gdf.index.tolist() if use_index_arg else None
            )
        elif method.lower() == "knn":
            k = kwargs.get("k", 5)
            # KNN needs coordinates - use centroids
            centroids = gdf.geometry.centroid
            # Create a temporary GDF with centroids for KNN function
            temp_gdf_for_knn = gpd.GeoDataFrame(
                gdf.drop(columns=["geometry"]), geometry=centroids, crs=gdf.crs
            )
            w = KNN.from_dataframe(
                temp_gdf_for_knn,
                k=k,
                use_index=use_index_arg,
                ids=temp_gdf_for_knn.index.tolist() if use_index_arg else None,
            )
        else:
            print(f" Error: Unsupported weights method '{method}'.")
            return None

        # Row-standardize (common practice)
        w.transform = "R"
        print(
            f"  Created {method.upper()} weights for {w.n} areas (Avg Neighbors: {w.mean_neighbors:.2f})."
        )
        if w.islands:
            print(f" Warning: Found {len(w.islands)} islands (areas with no neighbors).")
        return w

    except Exception as e:
        print(f" Error creating {method.upper()} weights: {e}")
        traceback.print_exc()
        return None


# --- Spatial Stats Functions (Adapted for GDFs & Error Handling) ---
def calculate_morans_i(gdf, variable, weights):
    """Calculate Global Moran's I with error handling."""
    if not ESDA_AVAILABLE:
        return None, None
    if gdf is None or variable not in gdf.columns or weights is None:
        print(f" Error: Invalid inputs for Moran's I calculation.")
        return None, None

    print(f"\nCalculating Global Moran's I for {variable}...")

    # Handle NaNs
    y = gdf[variable].copy()
    if y.isnull().any():
        nan_count = y.isnull().sum()
        print(f" Warning: {nan_count} NaN values found in {variable}. Filling with median.")
        y = y.fillna(y.median())

    try:
        moran = Moran(y, weights)
        print(f" Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")
        return moran.I, moran.p_sim
    except Exception as e:
        print(f" Error calculating Moran's I: {e}")
        return None, None


def calculate_local_morans(gdf, variable, weights):
    """Calculate Local Moran's I (LISA) with error handling."""
    if not ESDA_AVAILABLE:
        return None
    if gdf is None or variable not in gdf.columns or weights is None:
        print(f" Error: Invalid inputs for Local Moran's I calculation.")
        return None

    print(f"\nCalculating Local Moran's I for {variable}...")

    # Handle NaNs
    y = gdf[variable].copy()
    if y.isnull().any():
        nan_count = y.isnull().sum()
        print(f" Warning: {nan_count} NaN values found in {variable}. Filling with median.")
        y = y.fillna(y.median())

    try:
        local_moran = Moran_Local(y, weights)

        # Add results to GDF
        result_gdf = gdf.copy()
        result_gdf[f"{variable}_lisa"] = local_moran.Is
        result_gdf[f"{variable}_lisa_p"] = local_moran.p_sim
        result_gdf[f"{variable}_lisa_q"] = local_moran.q

        # Add cluster type
        result_gdf[f"{variable}_lisa_cluster"] = "Not Significant"
        sig_mask = result_gdf[f"{variable}_lisa_p"] < 0.05

        # HH: High-High (hot spots)
        result_gdf.loc[
            sig_mask & (result_gdf[f"{variable}_lisa_q"] == 1), f"{variable}_lisa_cluster"
        ] = "High-High"
        # LL: Low-Low (cold spots)
        result_gdf.loc[
            sig_mask & (result_gdf[f"{variable}_lisa_q"] == 3), f"{variable}_lisa_cluster"
        ] = "Low-Low"
        # HL: High-Low (spatial outliers)
        result_gdf.loc[
            sig_mask & (result_gdf[f"{variable}_lisa_q"] == 4), f"{variable}_lisa_cluster"
        ] = "High-Low"
        # LH: Low-High (spatial outliers)
        result_gdf.loc[
            sig_mask & (result_gdf[f"{variable}_lisa_q"] == 2), f"{variable}_lisa_cluster"
        ] = "Low-High"

        print(f" Local Moran's I calculated for {len(result_gdf)} areas.")
        return result_gdf
    except Exception as e:
        print(f" Error calculating Local Moran's I: {e}")
        return None


def calculate_getis_ord(gdf, variable, weights):
    """Calculate Getis-Ord Gi* with error handling."""
    if not ESDA_AVAILABLE:
        return None
    if gdf is None or variable not in gdf.columns or weights is None:
        print(f" Error: Invalid inputs for Getis-Ord Gi* calculation.")
        return None

    print(f"\nCalculating Getis-Ord Gi* for {variable}...")

    # Handle NaNs
    y = gdf[variable].copy()
    if y.isnull().any():
        nan_count = y.isnull().sum()
        print(f" Warning: {nan_count} NaN values found in {variable}. Filling with median.")
        y = y.fillna(y.median())

    try:
        g_local = G_Local(y, weights)

        # Add results to GDF
        result_gdf = gdf.copy()
        result_gdf[f"{variable}_gi"] = g_local.Gs
        result_gdf[f"{variable}_gi_z"] = g_local.Zs
        result_gdf[f"{variable}_gi_p"] = g_local.p_sim

        # Add hotspot/coldspot classification
        result_gdf[f"{variable}_gi_cluster"] = "Not Significant"
        sig_mask = result_gdf[f"{variable}_gi_p"] < 0.05

        # Hotspots (high values)
        result_gdf.loc[
            sig_mask & (result_gdf[f"{variable}_gi_z"] > 0), f"{variable}_gi_cluster"
        ] = "Hotspot"
        # Coldspots (low values)
        result_gdf.loc[
            sig_mask & (result_gdf[f"{variable}_gi_z"] < 0), f"{variable}_gi_cluster"
        ] = "Coldspot"

        print(f" Getis-Ord Gi* calculated for {len(result_gdf)} areas.")
        return result_gdf
    except Exception as e:
        print(f" Error calculating Getis-Ord Gi*: {e}")
        return None


def visualize_lisa(gdf, variable):
    """Visualize LISA clusters."""
    cluster_col = f"{variable}_lisa_cluster"
    if cluster_col not in gdf.columns:
        print(f" Error: LISA cluster column not found.")
        return

    print(f"\nVisualizing LISA clusters for {variable}...")

    try:
        # Create color map
        color_map = {
            "High-High": "#d7191c",  # Red
            "Low-Low": "#2c7bb6",  # Blue
            "High-Low": "#fdae61",  # Orange
            "Low-High": "#abd9e9",  # Light Blue
            "Not Significant": "#efefef",  # Light Grey
        }

        # Plot map
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            column=cluster_col,
            categorical=True,
            cmap="viridis",
            legend=True,
            ax=ax,
            legend_kwds={"title": "LISA Clusters"},
        )
        ax.set_title(f"Local Moran's I Clusters for {variable}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"lisa_clusters_{variable}.png"), dpi=300)
        plt.close()

        # Plot count of each cluster type
        plt.figure(figsize=(10, 6))
        cluster_counts = gdf[cluster_col].value_counts()
        cluster_counts.plot(
            kind="bar", color=[color_map.get(c, "#efefef") for c in cluster_counts.index]
        )
        plt.title(f"LISA Cluster Types for {variable}")
        plt.xlabel("Cluster Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"lisa_cluster_counts_{variable}.png"), dpi=300)
        plt.close()

        print(f" LISA visualizations saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f" Error visualizing LISA clusters: {e}")


def visualize_hotspots(gdf, variable):
    """Visualize Getis-Ord Gi* hotspots."""
    cluster_col = f"{variable}_gi_cluster"
    if cluster_col not in gdf.columns:
        print(f" Error: Gi* cluster column not found.")
        return

    print(f"\nVisualizing Gi* hotspots for {variable}...")

    try:
        # Create color map
        color_map = {
            "Hotspot": "#d7191c",  # Red
            "Coldspot": "#2c7bb6",  # Blue
            "Not Significant": "#efefef",  # Light Grey
        }

        # Plot map
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(
            column=cluster_col,
            categorical=True,
            cmap="coolwarm",
            legend=True,
            ax=ax,
            legend_kwds={"title": "Gi* Clusters"},
        )
        ax.set_title(f"Getis-Ord Gi* Hotspots for {variable}")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"gi_hotspots_{variable}.png"), dpi=300)
        plt.close()

        # Plot count of each hotspot type
        plt.figure(figsize=(10, 6))
        cluster_counts = gdf[cluster_col].value_counts()
        cluster_counts.plot(
            kind="bar", color=[color_map.get(c, "#efefef") for c in cluster_counts.index]
        )
        plt.title(f"Gi* Hotspot Types for {variable}")
        plt.xlabel("Hotspot Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"gi_hotspot_counts_{variable}.png"), dpi=300)
        plt.close()

        print(f" Gi* visualizations saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f" Error visualizing Gi* hotspots: {e}")


def aggregate_to_lad(lsoa_gdf, lad_code_col="lad_code"):
    """Aggregate LSOA data to LAD level for spatial regression."""
    if lsoa_gdf is None or lad_code_col not in lsoa_gdf.columns:
        print(f" Error: Cannot aggregate to LAD level. Missing LAD code column.")
        return None

    print("\nAggregating data to LAD level...")

    try:
        # Define columns to aggregate
        numeric_cols = [
            col
            for col in lsoa_gdf.columns
            if col != lad_code_col
            and col != "geometry"
            and pd.api.types.is_numeric_dtype(lsoa_gdf[col])
        ]

        # Create aggregation dictionary
        agg_dict = {col: "mean" for col in numeric_cols}

        # Aggregate
        lad_df = lsoa_gdf.dissolve(by=lad_code_col, aggfunc=agg_dict)
        print(f" Aggregated {len(lsoa_gdf)} LSOAs to {len(lad_df)} LADs.")
        return lad_df
    except Exception as e:
        print(f" Error aggregating to LAD level: {e}")
        return None


def run_spatial_regression(gdf, y_var, x_vars, weights):
    """
    Run OLS and Spatial Lag regression models with comprehensive diagnostics.
    
    This function implements two regression models:
    
    1. Ordinary Least Squares (OLS): Standard regression that assumes independence
       of observations. In spatial data, this assumption is often violated due to
       spatial autocorrelation, which can lead to biased coefficient estimates
       and inflated significance levels.
       
    2. Spatial Lag Model (ML_Lag): Accounts for spatial dependence by including a
       spatially lagged dependent variable (Wy). This model explicitly incorporates
       neighborhood effects, recognizing that an area's outcome is influenced by
       outcomes in neighboring areas. The spatial autoregressive coefficient (rho)
       quantifies the strength of these neighborhood effects.
    
    The function performs Moran's I test on OLS residuals to detect spatial
    autocorrelation. Significant spatial autocorrelation in residuals indicates
    that OLS results may be unreliable and that a spatial model is more appropriate.
    
    Important Note on Ecological Fallacy:
    These models use aggregated data at the LAD level, which introduces the risk
    of ecological fallacy - inferring individual-level relationships from group-level
    data. Results should be interpreted as area-level associations, not individual-level
    effects. Cross-scale validation with LSOA-level analysis helps mitigate this
    limitation but cannot eliminate it entirely.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing dependent and independent variables
        y_var (str): Name of dependent variable
        x_vars (list): List of independent variable names
        weights (libpysal.weights.W): Spatial weights matrix
        
    Returns:
        dict: Dictionary containing model objects or None if models fail
    """
    if not SPREG_AVAILABLE:
        return None
    if gdf is None or y_var not in gdf.columns or weights is None:
        print(f" Error: Invalid inputs for spatial regression.")
        return None

    # Check if all X variables exist
    missing_x = [x for x in x_vars if x not in gdf.columns]
    if missing_x:
        print(f" Error: Missing X variables: {missing_x}")
        return None

    print(f"\nRunning spatial regression for {y_var}...")
    print(f" Independent variables: {x_vars}")

    # Handle NaNs
    reg_data = gdf[[y_var] + x_vars].copy()
    if reg_data.isnull().any().any():
        print(f" Warning: NaN values found in regression data. Dropping rows with NaNs.")
        reg_data = reg_data.dropna()
        if len(reg_data) < len(x_vars) + 2:  # Need more observations than parameters
            print(f" Error: Not enough observations after dropping NaNs.")
            return None

    try:
        # Prepare data
        y = reg_data[y_var].values
        X = reg_data[x_vars].values

        # --- Run OLS ---
        ols = OLS(y, X, name_y=y_var, name_x=x_vars)
        print("\nOLS Results:")
        print(f" R-squared: {ols.r2:.4f}")
        print(" Coefficients:")
        for i, var in enumerate(x_vars):
            coef = ols.betas[i + 1]
            std_err = ols.std_err[i + 1]
            coef_val = coef.item() if isinstance(coef, np.ndarray) else coef
            std_err_val = std_err.item() if isinstance(std_err, np.ndarray) else std_err
            print(f"  {var}: {float(coef_val):.4f} (std_err={float(std_err_val):.4f})")

        # Capture OLS summary and residuals
        ols_summary_text = ols.summary
        gdf_with_residuals = gdf.copy()  # Use original GDF passed to function
        # Add residuals, aligning by index from reg_data (which might have dropped NaNs)
        gdf_with_residuals.loc[reg_data.index, "ols_residuals"] = ols.u

        # --- Interpret Moran's I for Residuals ---
        moran_i_res, p_val_res = calculate_morans_i(gdf_with_residuals, "ols_residuals", weights)
        if moran_i_res is not None:
            moran_res_text = (
                f"Moran's I for OLS Residuals: {moran_i_res:.4f} (p-value: {p_val_res:.4f})\n"
            )
            print(f" {moran_res_text.strip()}")
            if p_val_res < 0.05:
                print(
                    "  Significant spatial autocorrelation in OLS residuals detected. OLS results may be unreliable (inflated significance)."
                )
            else:
                print("  No significant spatial autocorrelation detected in OLS residuals.")
        else:
            moran_res_text = "Moran's I for OLS Residuals: Calculation Failed.\n"

        models_dict = {"ols": ols}  # Initialize with OLS result
        spatial_lag_summary_text = "Spatial Lag Model Failed to Run or Was Skipped.\n"

        # --- Run Spatial Lag model ---
        if SPREG_AVAILABLE:  # Check again just in case
            try:
                spatial_lag = ML_Lag(y, X, weights, name_y=y_var, name_x=x_vars)
                print("\nSpatial Lag Model Results:")
                print(f" Pseudo R-squared: {spatial_lag.pr2:.4f}")
                print(f" Spatial Autoregressive Coefficient (rho): {spatial_lag.rho:.4f}")
                print(" Coefficients:")
                for i, var in enumerate(x_vars):
                    coef = spatial_lag.betas[i + 1]
                    std_err = spatial_lag.std_err[i + 1]
                    coef_val = coef.item() if isinstance(coef, np.ndarray) else coef
                    std_err_val = std_err.item() if isinstance(std_err, np.ndarray) else std_err
                    print(f"  {var}: {float(coef_val):.4f} (std_err={float(std_err_val):.4f})")

                spatial_lag_summary_text = spatial_lag.summary
                models_dict["spatial_lag"] = spatial_lag  # Add spatial lag result

            except Exception as e_lag:
                print(f" Error running Spatial Lag model: {e_lag}")
                spatial_lag_summary_text = (
                    f"Spatial Lag Model Failed to Run: {e_lag}\n"  # Capture the error
                )

        # --- Save Summaries and Generate Plots/Analysis ---
        summary_path = os.path.join(OUTPUT_DIR, "lad_regression_summary.txt")
        moran_res_text = ""

        # 1. Moran's I for Residuals (Calculate before writing summary)
        try:
            print("\nCalculating Moran's I for OLS Residuals...")
            # Use the helper function; it handles NaNs internally if needed
            moran_i_res, p_val_res = calculate_morans_i(
                gdf_with_residuals, "ols_residuals", weights
            )
            if moran_i_res is not None:
                moran_res_text = (
                    f"Moran's I for OLS Residuals: {moran_i_res:.4f} (p-value: {p_val_res:.4f})\n"
                )
                print(f" {moran_res_text.strip()}")
            else:
                moran_res_text = "Moran's I for OLS Residuals: Calculation Failed.\n"
        except Exception as moran_err:
            print(f" Error calculating Moran's I for residuals: {moran_err}")
            moran_res_text = f"Moran's I for OLS Residuals: Error ({moran_err})\n"

        # 2. Save Summaries to File
        try:
            with open(summary_path, "w") as f:
                f.write("OLS Model Summary:\n")
                f.write("=" * 80 + "\n")
                f.write(ols_summary_text + "\n\n")
                f.write("Spatial Lag Model Summary:\n")
                f.write("=" * 80 + "\n")
                f.write(spatial_lag_summary_text + "\n\n")
                f.write("=" * 80 + "\n")
                f.write(moran_res_text)
            print(f" Regression summaries saved to {summary_path}")
        except Exception as save_err:
            print(f" Error saving regression summaries: {save_err}")

        # 3. Coefficient Plot (Only if Spatial Lag ran successfully)
        if "spatial_lag" in models_dict:
            try:
                ols_coeffs = pd.Series(
                    {var: ols.betas[i + 1].item() for i, var in enumerate(x_vars)}
                )
                ols_stderr = pd.Series(
                    {var: ols.std_err[i + 1].item() for i, var in enumerate(x_vars)}
                )
                lag_coeffs = pd.Series(
                    {var: spatial_lag.betas[i + 1].item() for i, var in enumerate(x_vars)}
                )
                lag_stderr = pd.Series(
                    {var: spatial_lag.std_err[i + 1].item() for i, var in enumerate(x_vars)}
                )
                # Add rho for spatial lag (rho is beta[0] in ML_Lag)
                lag_coeffs["rho"] = spatial_lag.rho.item()  # rho is beta[0]
                lag_stderr["rho"] = spatial_lag.std_err[0].item()  # std_err[0] corresponds to rho

                coeffs_df = pd.DataFrame(
                    {
                        "OLS_Coeff": ols_coeffs,
                        "OLS_StdErr": ols_stderr,
                        "Lag_Coeff": lag_coeffs,
                        "Lag_StdErr": lag_stderr,
                    }
                ).fillna(
                    0
                )  # Fill NaN for rho in OLS columns

                fig, ax = plt.subplots(figsize=(10, 6))
                coeffs_df[["OLS_Coeff", "Lag_Coeff"]].plot(
                    kind="bar",
                    yerr=coeffs_df[["OLS_StdErr", "Lag_StdErr"]].values.T,
                    ax=ax,
                    capsize=4,
                    rot=0,
                )
                ax.set_ylabel("Coefficient Value")
                ax.set_title("LAD Regression Coefficient Comparison (OLS vs Spatial Lag)")
                ax.axhline(0, color="grey", lw=0.8)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUTPUT_DIR, "lad_regression_coefficient_plot.png"), dpi=300
                )
                plt.close(fig)
                print(" Coefficient comparison plot saved.")
            except Exception as plot_err:
                print(f" Error generating coefficient plot: {plot_err}")
        else:
            print(" Skipping coefficient plot (Spatial Lag model did not run successfully).")

        # 4. Residual Map and LISA for Residuals
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            gdf_with_residuals.plot(
                column="ols_residuals",
                cmap="viridis",
                legend=True,
                ax=ax,
                missing_kwds={"color": "lightgrey", "label": "Missing/NaN"},
            )
            ax.set_title("Spatial Distribution of OLS Residuals (LAD Level)")
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "lad_regression_ols_residuals_map.png"), dpi=300)
            plt.close(fig)
            print(" OLS residuals map saved.")

            # Calculate and visualize LISA for residuals
            lisa_residuals_gdf = calculate_local_morans(
                gdf_with_residuals, "ols_residuals", weights
            )
            if lisa_residuals_gdf is not None:
                visualize_lisa(lisa_residuals_gdf, "ols_residuals")
            else:
                print("  Skipping LISA visualization for residuals due to calculation failure.")

        except Exception as map_err:
            print(f" Error generating residuals map: {map_err}")

        return models_dict  # Return dictionary containing model objects

    except Exception as e_reg:
        print(f" Error running regression: {e_reg}")
        return None


def load_and_prep_data():
    """
    Load and prepare all necessary data for spatial analysis.
    
    This function loads the unified dataset containing environmental and socioeconomic
    indicators, along with validated LSOA and LAD boundary files. It performs
    validation checks on the geometries and ensures consistent coordinate reference
    systems for spatial analysis.
    
    Returns:
        tuple: (unified_df, lsoa_gdf, lad_gdf) containing:
            - unified_df: DataFrame with environmental justice indicators
            - lsoa_gdf: GeoDataFrame with LSOA boundaries
            - lad_gdf: GeoDataFrame with LAD boundaries
    """
    print("\n--- Loading Data ---")

    # Load unified dataset
    try:
        unified_df = pd.read_csv(UNIFIED_CSV_PATH)
        print(f"Loaded unified dataset with {len(unified_df)} rows.")
    except Exception as e:
        print(f"Error loading unified dataset: {e}")
        return None, None, None

    # Load LSOA boundaries - try primary path first, then fallback
    lsoa_gdf = None
    if os.path.exists(LSOA_GEOJSON_PATH):
        print(f"Loading LSOA boundaries from primary path...")
        lsoa_gdf = load_geo_data(LSOA_GEOJSON_PATH)
    
    if lsoa_gdf is None and os.path.exists(LSOA_FALLBACK_PATH):
        print(f"Primary LSOA file not found or invalid. Trying fallback path...")
        lsoa_gdf = load_geo_data(LSOA_FALLBACK_PATH)
    
    if lsoa_gdf is None:
        print("Failed to load LSOA boundaries from any path.")
        return unified_df, None, None

    # Load LAD boundaries - try primary path first, then fallback
    lad_gdf = None
    if os.path.exists(LAD_GEOJSON_PATH):
        print(f"Loading LAD boundaries from primary path...")
        lad_gdf = load_geo_data(LAD_GEOJSON_PATH)
    
    if lad_gdf is None and os.path.exists(LAD_FALLBACK_PATH):
        print(f"Primary LAD file not found or invalid. Trying fallback path...")
        lad_gdf = load_geo_data(LAD_FALLBACK_PATH)
    
    if lad_gdf is None:
        print("Failed to load LAD boundaries from any path.")
        return unified_df, lsoa_gdf, None

    # Ensure correct CRS
    if lsoa_gdf.crs != TARGET_CRS:
        print(f"Reprojecting LSOA boundaries to {TARGET_CRS}")
        lsoa_gdf = lsoa_gdf.to_crs(TARGET_CRS)

    if lad_gdf.crs != TARGET_CRS:
        print(f"Reprojecting LAD boundaries to {TARGET_CRS}")
        lad_gdf = lad_gdf.to_crs(TARGET_CRS)
    
    # Validate boundary files
    print("\nValidating boundary files...")
    print(f"LSOA boundaries: {len(lsoa_gdf)} areas")
    print(f"LAD boundaries: {len(lad_gdf)} areas")
    
    # Check for invalid geometries
    invalid_lsoa = (~lsoa_gdf.geometry.is_valid).sum()
    invalid_lad = (~lad_gdf.geometry.is_valid).sum()
    
    if invalid_lsoa > 0:
        print(f"Warning: Found {invalid_lsoa} invalid LSOA geometries. Attempting to fix...")
        lsoa_gdf = fix_geometries(lsoa_gdf)
    
    if invalid_lad > 0:
        print(f"Warning: Found {invalid_lad} invalid LAD geometries. Attempting to fix...")
        lad_gdf = fix_geometries(lad_gdf)
    
    print("Boundary validation complete.")

    return unified_df, lsoa_gdf, lad_gdf


def merge_data(unified_df, lsoa_gdf):
    """Merge unified data with LSOA boundaries."""
    if unified_df is None or lsoa_gdf is None:
        print("Cannot merge data: missing inputs.")
        return None

    print("\n--- Merging Data ---")

    # Identify LSOA code columns
    lsoa_code_cols = [
        col
        for col in lsoa_gdf.columns
        if "LSOA" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
    ]
    if not lsoa_code_cols:
        print("Error: Could not identify LSOA code column in boundaries.")
        return None

    lsoa_code_col_in_gdf = lsoa_code_cols[0]
    print(f"Using '{lsoa_code_col_in_gdf}' as LSOA code column in boundaries.")

    # Identify LSOA code column in unified data
    unified_lsoa_cols = [
        col
        for col in unified_df.columns
        if "LSOA" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
    ]
    if not unified_lsoa_cols:
        print("Error: Could not identify LSOA code column in unified data.")
        return None

    unified_lsoa_col = unified_lsoa_cols[0]
    print(f"Using '{unified_lsoa_col}' as LSOA code column in unified data.")

    # Ensure code columns are string type
    lsoa_gdf[lsoa_code_col_in_gdf] = lsoa_gdf[lsoa_code_col_in_gdf].astype(str)
    unified_df[unified_lsoa_col] = unified_df[unified_lsoa_col].astype(str)

    # Merge data
    try:
        merged_gdf = lsoa_gdf.merge(
            unified_df, left_on=lsoa_code_col_in_gdf, right_on=unified_lsoa_col, how="inner"
        )

        print(f"Merged data has {len(merged_gdf)} rows.")

        # Check for valid geometries
        if "geometry" not in merged_gdf.columns:
            print("Error: Geometry column lost during merge.")
            return None

        # Filter out invalid geometries
        valid_geom_mask = (
            merged_gdf.geometry.notna()
            & merged_gdf.geometry.is_valid
            & ~merged_gdf.geometry.is_empty
        )
        merged_gdf = merged_gdf[valid_geom_mask]

        print(f"After filtering invalid geometries: {len(merged_gdf)} rows.")
        return merged_gdf
    except Exception as e:
        print(f"Error merging data: {e}")
        return None


def merge_lad_data(lad_df, lad_gdf):
    """Merge aggregated LAD data with LAD boundaries."""
    if lad_df is None or lad_gdf is None:
        print("Cannot merge LAD data: missing inputs.")
        return None

    print("\n--- Merging LAD Data ---")

    # Identify LAD code columns in boundaries
    lad_code_cols_gdf = [
        col
        for col in lad_gdf.columns
        if "LAD" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
    ]
    if not lad_code_cols_gdf:
        print("Error: Could not identify LAD code column in boundaries.")
        return None
    lad_code_col_in_gdf = lad_code_cols_gdf[0]
    print(f"Using '{lad_code_col_in_gdf}' as LAD code column in boundaries.")

    # Identify LAD code column in aggregated data (should be the index after dissolve)
    if lad_df.index.name is None or "lad_code" not in lad_df.index.name.lower():
        # If index is not named 'lad_code', try finding a column
        lad_code_cols_df = [
            col
            for col in lad_df.columns
            if "LAD" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
        ]
        if not lad_code_cols_df:
            print("Error: Could not identify LAD code column in aggregated data.")
            return None
        lad_code_col_in_df = lad_code_cols_df[0]
        lad_df_to_merge = lad_df.reset_index()  # Use column for merging
    else:
        lad_code_col_in_df = lad_df.index.name
        lad_df_to_merge = lad_df.reset_index()  # Reset index to use as column

    print(f"Using '{lad_code_col_in_df}' as LAD code column in aggregated data.")

    # Ensure code columns are string type
    lad_gdf[lad_code_col_in_gdf] = lad_gdf[lad_code_col_in_gdf].astype(str)
    lad_df_to_merge[lad_code_col_in_df] = lad_df_to_merge[lad_code_col_in_df].astype(str)

    # Merge data
    try:
        # --- 2. Perform the Merge using GeoDataFrame's method ---
        # Keep only necessary columns from the aggregated data to avoid potential conflicts
        # Ensure LAD_REGRESSION_Y and LAD_REGRESSION_X are defined (likely global constants)
        cols_to_keep_from_agg = [lad_code_col_in_df] + [LAD_REGRESSION_Y] + LAD_REGRESSION_X
        # Check if columns exist in the aggregated dataframe before selecting
        missing_cols = [col for col in cols_to_keep_from_agg if col not in lad_df_to_merge.columns]
        if missing_cols:
            print(f"Error: Required columns missing from aggregated data: {missing_cols}")
            return None
        lad_aggregated_subset = lad_df_to_merge[cols_to_keep_from_agg]

        print(
            f"Merging aggregated data (right DF) with geometry (left GDF) on '{lad_code_col_in_gdf}' and '{lad_code_col_in_df}'..."
        )
        # Perform the merge with the GeoDataFrame on the LEFT
        merged_gdf = lad_gdf[[lad_code_col_in_gdf, "geometry"]].merge(
            lad_aggregated_subset,
            left_on=lad_code_col_in_gdf,
            right_on=lad_code_col_in_df,
            how="inner",  # Keep only LADs present in both datasets
        )

        # --- 3. Verify the Result ---
        print(f"Merged LAD data shape: {merged_gdf.shape}")
        print(f"Is the result a GeoDataFrame? {isinstance(merged_gdf, gpd.GeoDataFrame)}")
        if isinstance(merged_gdf, gpd.GeoDataFrame):
            print(f"Does it have a geometry column? {'geometry' in merged_gdf.columns}")
            if "geometry" not in merged_gdf.columns:
                print("Error: Geometry column lost during merge despite using GeoDataFrame merge.")
                return None

            print(f"Number of null geometries before filtering: {merged_gdf.geometry.isna().sum()}")
            print(f"CRS before check: {merged_gdf.crs}")
            # Check if CRS needs resetting (sometimes merge can drop it)
            if merged_gdf.crs != TARGET_CRS:
                print(f"Warning: CRS mismatch or lost during merge. Resetting CRS to {TARGET_CRS}")
                merged_gdf.set_crs(TARGET_CRS, inplace=True)

            # Filter out invalid geometries AFTER ensuring it's a GDF
            valid_geom_mask = (
                merged_gdf.geometry.notna()
                & merged_gdf.geometry.is_valid
                & ~merged_gdf.geometry.is_empty
            )
            initial_count = len(merged_gdf)
            merged_gdf = merged_gdf[valid_geom_mask]
            filtered_count = len(merged_gdf)
            if initial_count != filtered_count:
                print(
                    f"Filtered out {initial_count - filtered_count} rows with invalid/null geometries."
                )

            print(f"Final LAD GeoDataFrame shape after filtering: {merged_gdf.shape}")
            return merged_gdf

        else:
            print("Error: Merge failed to produce a GeoDataFrame.")
            return None

    except Exception as e:
        print(f"Error merging LAD data: {e}")
        return None


def main():
    """Main function to run the spatial analysis."""
    print("=" * 80)
    print("Consolidated Spatial Analysis for Environmental Justice Project")
    print("=" * 80)

    # Load data
    unified_df, lsoa_gdf, lad_gdf = load_and_prep_data()
    if unified_df is None:
        print("Error: Failed to load unified dataset. Exiting.")
        return

    # Merge LSOA data
    lsoa_analysis_gdf = merge_data(unified_df, lsoa_gdf)
    if lsoa_analysis_gdf is None:
        print("Error: Failed to merge LSOA data. Skipping LSOA analysis.")
        # Decide whether to continue to LAD analysis or exit
    else:
        # Create LSOA spatial weights
        lsoa_weights = create_spatial_weights(lsoa_analysis_gdf, method="queen")
        if lsoa_weights is None:
            print("Error: Failed to create LSOA spatial weights. Skipping LSOA analysis.")
        else:
            # LSOA-level analysis
            print("\n" + "=" * 80)
            print("LSOA-Level Spatial Analysis")
            print("=" * 80)

            # Filter variables that exist in the data
            available_vars = [var for var in LSOA_VARIABLES if var in lsoa_analysis_gdf.columns]
            if not available_vars:
                print("Error: None of the specified LSOA variables found in merged data.")
            else:
                print(f"Analyzing LSOA variables: {available_vars}")
                current_lsoa_gdf = lsoa_analysis_gdf.copy()  # Keep track of results

                for variable in available_vars:
                    print(f"\n--- Analyzing LSOA Variable: {variable} ---")
                    # Global Moran's I
                    calculate_morans_i(current_lsoa_gdf, variable, lsoa_weights)

                    # Local Moran's I
                    temp_gdf = calculate_local_morans(current_lsoa_gdf, variable, lsoa_weights)
                    if temp_gdf is not None:
                        current_lsoa_gdf = temp_gdf  # Update GDF with results
                        visualize_lisa(current_lsoa_gdf, variable)

                    # Getis-Ord Gi*
                    temp_gdf = calculate_getis_ord(current_lsoa_gdf, variable, lsoa_weights)
                    if temp_gdf is not None:
                        current_lsoa_gdf = temp_gdf  # Update GDF with results
                        visualize_hotspots(current_lsoa_gdf, variable)

                # Save final LSOA GDF with results
                try:
                    output_lsoa_gpkg = os.path.join(OUTPUT_DIR, "lsoa_analysis_results.gpkg")
                    current_lsoa_gdf.to_file(output_lsoa_gpkg, driver="GPKG")
                    print(f"\nSaved LSOA results GeoPackage to: {output_lsoa_gpkg}")
                except Exception as e:
                    print(f"\nError saving LSOA results GeoPackage: {e}")

    # LAD-level analysis
    print("\n" + "=" * 80)
    print("LAD-Level Spatial Regression")
    print("=" * 80)

    if lad_gdf is None:
        print("LAD boundaries not loaded. Skipping LAD-level analysis.")
    elif "lad_code" not in unified_df.columns:
        print("LAD code column not found in unified data. Skipping LAD-level analysis.")
    else:
        # Aggregate LSOA data to LAD level (using original merged GDF if available)
        lad_aggregated_data = aggregate_to_lad(
            lsoa_analysis_gdf if lsoa_analysis_gdf is not None else unified_df, "lad_code"
        )

        if lad_aggregated_data is None:
            print("Failed to aggregate data to LAD level. Skipping LAD analysis.")
        else:
            # Merge aggregated data with LAD boundaries
            lad_analysis_gdf = merge_lad_data(lad_aggregated_data, lad_gdf)

            if lad_analysis_gdf is None:
                print("Failed to merge aggregated data with LAD boundaries. Skipping LAD analysis.")
            else:
                # Create LAD spatial weights
                lad_weights = create_spatial_weights(lad_analysis_gdf, method="queen")
                if lad_weights is None:
                    print("Error: Failed to create LAD spatial weights. Skipping LAD regression.")
                else:
                    # Check if regression variables exist
                    available_lad_x = [
                        var for var in LAD_REGRESSION_X if var in lad_analysis_gdf.columns
                    ]
                    if LAD_REGRESSION_Y not in lad_analysis_gdf.columns:
                        print(
                            f"Error: Dependent variable '{LAD_REGRESSION_Y}' not found in LAD data."
                        )
                    elif not available_lad_x:
                        print(
                            "Error: None of the specified independent variables found in LAD data."
                        )
                    else:
                        if len(available_lad_x) < len(LAD_REGRESSION_X):
                            print(
                                f"Warning: Using subset of X variables for LAD regression: {available_lad_x}"
                            )
                        # Run LAD spatial regression
                        run_spatial_regression(
                            lad_analysis_gdf, LAD_REGRESSION_Y, available_lad_x, lad_weights
                        )

                    # Save final LAD GDF with results (if any were added)
                    try:
                        output_lad_gpkg = os.path.join(OUTPUT_DIR, "lad_analysis_results.gpkg")
                        lad_analysis_gdf.to_file(output_lad_gpkg, driver="GPKG")
                        print(f"\nSaved LAD results GeoPackage to: {output_lad_gpkg}")
                    except Exception as e:
                        print(f"\nError saving LAD results GeoPackage: {e}")

    print("\n" + "=" * 80)
    print("Spatial Analysis Complete")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
