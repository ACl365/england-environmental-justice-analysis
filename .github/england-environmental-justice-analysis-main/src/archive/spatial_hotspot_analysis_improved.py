"""
Spatial Autocorrelation and Hotspot Analysis for Environmental Justice Project

This script implements advanced spatial statistics:
- Moran's I for spatial autocorrelation
- Getis-Ord Gi* for hotspot identification
- Spatial regression models accounting for neighborhood effects
- Spatial visualizations of hotspots and clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, norm
import os
from shapely.geometry import Polygon, MultiPolygon

# Removed Plotly imports for simplicity and focus on core spatial stats/matplotlib plots
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import warnings
import traceback


def check_dependencies():
    """Check if required libraries are available."""
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

    # Check core dependencies
    try:
        import pandas

        requirements["pandas"] = True
    except ImportError:
        print("pandas not available - required for data handling")

    try:
        import numpy

        requirements["numpy"] = True
    except ImportError:
        print("numpy not available - required for numerical operations")

    try:
        import matplotlib

        requirements["matplotlib"] = True
    except ImportError:
        print("matplotlib not available - required for visualization")

    try:
        import seaborn

        requirements["seaborn"] = True
    except ImportError:
        print("seaborn not available - optional for enhanced visualization")

    try:
        import geopandas

        requirements["geopandas"] = True
    except ImportError:
        print("geopandas not available - required for spatial data handling")

    try:
        import shapely

        requirements["shapely"] = True
    except ImportError:
        print("shapely not available - required for geometry operations")

    # Check spatial analysis dependencies
    try:
        import libpysal

        requirements["libpysal"] = True
    except ImportError:
        print("libpysal not available - required for spatial weights")

    try:
        import esda

        requirements["esda"] = True
    except ImportError:
        print("esda not available - required for spatial autocorrelation")

    try:
        import spreg

        requirements["spreg"] = True
    except ImportError:
        print("spreg not available - required for spatial regression")

    # Print summary
    print("\nDependency Check Summary:")
    for dep, available in requirements.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"{dep}: {status}")

    return requirements


# Attempt to import specialized spatial statistics libraries
try:
    # Import new-style PySAL subpackages
    import libpysal as lps
    from libpysal.weights import Queen, KNN, W
    import esda
    from esda.moran import Moran, Moran_Local
    from esda.getisord import G_Local
    import spreg
    from spreg import OLS, ML_Lag  # Import specific models

    PYSAL_AVAILABLE = True
    ESDA_AVAILABLE = True
    print("Successfully imported PySAL subpackages (libpysal, esda, spreg)")
except ImportError as e:
    PYSAL_AVAILABLE = False
    ESDA_AVAILABLE = False
    warnings.warn(
        f"PySAL subpackages not available: {str(e)}. Some spatial statistics will be implemented manually or skipped."
    )

# Check all dependencies at startup
dependencies = check_dependencies()

# Set the plotting style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    print("Seaborn v0.8 style not found, using default.")
    plt.style.use("default")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Create output directories if they don't exist
os.makedirs("outputs/spatial_hotspots", exist_ok=True)


def fix_geometries(gdf):
    """Fix invalid geometries in a GeoDataFrame."""
    if gdf is None or not isinstance(gdf, gpd.GeoDataFrame):
        print("Cannot fix geometries: Invalid or None GeoDataFrame")
        return gdf

    # Report initial validity
    invalid_count = (~gdf.geometry.is_valid).sum()
    print(f"Found {invalid_count} invalid geometries. Attempting to fix...")

    if invalid_count == 0:
        print("All geometries are valid.")
        return gdf

    # Make a copy to avoid modifying the original
    gdf_fixed = gdf.copy()

    # Try buffer(0) first - most common fix
    try:
        print("Applying buffer(0) fix...")
        # Check for null geometries first
        null_geoms = gdf_fixed.geometry.isna().sum()
        if null_geoms > 0:
            print(f"Warning: {null_geoms} null geometries found.")

        # Apply buffer(0) only to non-null invalid geometries
        gdf_fixed["geometry"] = gdf_fixed.apply(
            lambda row: (
                row.geometry.buffer(0)
                if row.geometry is not None and not row.geometry.is_valid
                else row.geometry
            ),
            axis=1,
        )

        # Check if all fixed
        still_invalid = (~gdf_fixed.geometry.is_valid).sum()
        if still_invalid > 0:
            print(f"{still_invalid} geometries still invalid after buffer(0).")

            # Try make_valid for remaining invalid geometries (if shapely >= 1.8)
            try:
                print("Trying make_valid for remaining invalid geometries...")
                gdf_fixed["geometry"] = gdf_fixed.apply(
                    lambda row: (
                        row.geometry.make_valid()
                        if hasattr(row.geometry, "make_valid")
                        and row.geometry is not None
                        and not row.geometry.is_valid
                        else row.geometry
                    ),
                    axis=1,
                )

                # Final check
                final_invalid = (~gdf_fixed.geometry.is_valid).sum()
                print(f"After all fixes: {final_invalid} geometries still invalid.")
            except Exception as e:
                print(f"make_valid failed: {e}")
        else:
            print("All geometries fixed successfully!")
    except Exception as e:
        print(f"Error fixing geometries: {e}")
        return gdf  # Return original if fixing failed

    return gdf_fixed


def load_data():
    """
    Load the unified dataset, LSOA boundaries, and ward boundaries.

    Returns:
        tuple: (unified_df, lsoa_gdf, wards_gdf)
    """
    print("\n--- Loading Data ---")
    # Use environment variables or relative paths
    base_dir = os.environ.get("UK_ENV_BASE_DIR", os.path.join("."))
    data_dir = os.environ.get("UK_ENV_DATA_DIR", os.path.join(base_dir, "data"))

    unified_path = os.environ.get(
        "UNIFIED_DATA_PATH", os.path.join(base_dir, "unified_dataset_with_air_quality.csv")
    )
    lsoa_path = os.environ.get(
        "LSOA_PATH", os.path.join(base_dir, "LSOA_DEC_2021_EW_NC_v3_-7589743170352813307.geojson")
    )
    wards_path = os.environ.get(
        "WARDS_PATH",
        os.path.join(data_dir, "Wards_December_2024_Boundaries_UK_BFC_7247148252775165514.geojson"),
    )

    # Load the unified dataset
    try:
        unified_df = pd.read_csv(unified_path)
        print(
            f"Loaded unified dataset with {len(unified_df)} LSOAs. Columns: {unified_df.columns.tolist()}"
        )
    except FileNotFoundError:
        print(f"Error: File not found at {unified_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading unified dataset: {e}")
        return None, None, None

    # Load the LSOA boundaries
    lsoa_gdf = None
    try:
        lsoa_gdf = gpd.read_file(lsoa_path)
        # Check if it loaded as a GeoDataFrame
        if not isinstance(lsoa_gdf, gpd.GeoDataFrame):
            print(f"Warning: Loaded LSOA file {lsoa_path} is not a GeoDataFrame.")
            lsoa_gdf = None  # Treat as failed load
        else:
            print(f"Loaded LSOA boundaries with {len(lsoa_gdf)} areas.")
            print(f"LSOA GeoDataFrame Info:")
            lsoa_gdf.info(verbose=False, memory_usage="deep")  # Show basic info
            lsoa_gdf = fix_geometries(lsoa_gdf)
    except Exception as e:
        print(f"Could not load LSOA boundaries from GeoJSON: {str(e)}")
        lsoa_gdf = None

    # Load the ward boundaries (Optional, used for LAD fallback)
    wards_gdf = None
    try:
        # Attempt GeoJSON first
        wards_gdf = gpd.read_file(wards_path)
        if not isinstance(wards_gdf, gpd.GeoDataFrame):
            print(f"Warning: Loaded Ward file is not a GeoDataFrame.")
            wards_gdf = None
        else:
            print(f"Loaded ward boundaries with {len(wards_gdf)} wards.")
            wards_gdf = fix_geometries(wards_gdf)
    except Exception as e:
        print(f"Could not load ward boundaries from GeoJSON: {str(e)}")
        wards_gdf = None  # Fail silently if wards aren't critical

    return unified_df, lsoa_gdf, wards_gdf


def create_lsoa_spatial_weights(unified_df, lsoa_gdf, lsoa_code_col="lsoa_code"):
    """
    Create accurate spatial weights matrix using LSOA boundaries.
    Filters out invalid geometries before weight creation.

    Args:
        unified_df (pd.DataFrame): DataFrame with LSOA data
        lsoa_gdf (gpd.GeoDataFrame): GeoDataFrame with LSOA boundaries
        lsoa_code_col (str): Column name for LSOA code in unified_df

    Returns:
        tuple: (weights_matrix, analysis_gdf) or (None, None) on failure
    """
    print("\n--- Creating Accurate LSOA Spatial Weights ---")

    if lsoa_gdf is None or not isinstance(lsoa_gdf, gpd.GeoDataFrame):
        print(
            "LSOA boundaries GeoDataFrame not available or invalid. Cannot create accurate spatial weights."
        )
        return None, None

    if not PYSAL_AVAILABLE:
        print("PySAL not available. Cannot create accurate spatial weights.")
        return None, None

    try:
        # Identify the LSOA code column in the GeoJSON
        lsoa_code_cols = [
            col
            for col in lsoa_gdf.columns
            if "LSOA" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
        ]
        if not lsoa_code_cols:
            print("Error: Could not identify LSOA code column in GeoJSON. Available columns:")
            print(lsoa_gdf.columns.tolist())
            return None, None

        lsoa_code_col_in_gdf = lsoa_code_cols[0]
        print(f"Using '{lsoa_code_col_in_gdf}' as LSOA code column in GeoJSON")

        # Check if the LSOA code column exists in unified_df
        if lsoa_code_col not in unified_df.columns:
            print(f"Error: LSOA code column '{lsoa_code_col}' not found in the unified dataset.")
            print(f"Available columns: {unified_df.columns.tolist()}")
            return None, None

        # Ensure the CRS is appropriate for England spatial analysis (British National Grid)
        target_crs = "EPSG:27700"
        if lsoa_gdf.crs is None:
            print("Warning: LSOA GeoDataFrame has no CRS. Assuming WGS84 (EPSG:4326).")
            lsoa_gdf.set_crs(epsg=4326, inplace=True)

        if str(lsoa_gdf.crs).upper() != target_crs:
            print(f"Reprojecting LSOA boundaries from {lsoa_gdf.crs} to {target_crs}")
            try:
                # Ensure geometry column exists before reprojection
                if "geometry" not in lsoa_gdf.columns or not hasattr(lsoa_gdf, "geometry"):
                    print(
                        "Error: 'geometry' column not found or invalid in LSOA boundaries GeoDataFrame."
                    )
                    return None, None

                # Check for invalid geometries BEFORE reprojection (sometimes helps)
                invalid_geom_count_before = lsoa_gdf[~lsoa_gdf.geometry.is_valid].shape[0]
                if invalid_geom_count_before > 0:
                    print(
                        f"Warning: Found {invalid_geom_count_before} invalid geometries in LSOA boundaries *before* reprojection."
                    )
                    # If ALL are invalid, buffer(0) is unlikely to fix it and might hide a deeper problem
                    if invalid_geom_count_before == len(lsoa_gdf):
                        print(
                            "Warning: ALL geometries are invalid. The buffer(0) fix might be very slow or fail. Check the source GeoJSON."
                        )
                    print(
                        "Attempting to fix invalid geometries using buffer(0)... (This may take time)"
                    )
                    # Filter out null geometries *before* buffering
                    lsoa_gdf = lsoa_gdf[lsoa_gdf.geometry.notna()]
                    if lsoa_gdf.empty:
                        print("Error: All geometries were null before buffering.")
                        return None, None
                    # Apply buffer(0) - use apply for robustness
                    try:
                        lsoa_gdf["geometry"] = lsoa_gdf["geometry"].apply(
                            lambda geom: geom.buffer(0) if not geom.is_valid else geom
                        )
                        fixed_count = (
                            invalid_geom_count_before
                            - lsoa_gdf[~lsoa_gdf.geometry.is_valid].shape[0]
                        )
                        print(f"Attempted fix: {fixed_count} geometries potentially corrected.")
                    except Exception as buffer_err:
                        print(
                            f"Error during buffer(0) fix: {buffer_err}. Proceeding with remaining valid geometries."
                        )

                lsoa_gdf = lsoa_gdf.to_crs(target_crs)
                print("Reprojection successful.")
            except Exception as e:
                print(f"Error during LSOA re-projection: {e}")
                traceback.print_exc()
                return None, None

        # Prepare the geometry data (select only needed columns)
        lsoa_geom_df = lsoa_gdf[[lsoa_code_col_in_gdf, "geometry"]].copy()

        # Diagnostic: Check geometry before merge
        print(f"LSOA Geometry Info Before Merge:")
        print(f"  Total rows: {len(lsoa_geom_df)}")
        print(f"  Non-null geometries: {lsoa_geom_df.geometry.notna().sum()}")
        print(f"  Valid geometries: {lsoa_geom_df.geometry.is_valid.sum()}")
        print(f"  CRS: {lsoa_geom_df.crs}")

        # Perform the merge using GeoPandas' merge
        print(
            f"Merging unified data with LSOA geometries on '{lsoa_code_col}' and '{lsoa_code_col_in_gdf}'..."
        )
        try:
            # Ensure unified_df has the LSOA code column before making it GDF temporarily
            if lsoa_code_col not in unified_df.columns:
                raise ValueError(f"LSOA code column '{lsoa_code_col}' missing from unified_df")

            # Perform merge. Keep only rows present in unified_df.
            analysis_gdf = lsoa_geom_df.merge(
                unified_df,
                left_on=lsoa_code_col_in_gdf,
                right_on=lsoa_code_col,
                how="inner",  # Use inner join to keep only matching LSOAs
            )

            # Check if the result is a GeoDataFrame and has geometry
            if not isinstance(analysis_gdf, gpd.GeoDataFrame):
                print("Warning: Merge resulted in a pandas DataFrame. Reconstructing GeoDataFrame.")
                if "geometry" not in analysis_gdf.columns:
                    print(f"Error: Geometry column 'geometry' lost during merge.")
                    return None, None
                analysis_gdf = gpd.GeoDataFrame(analysis_gdf, geometry="geometry", crs=target_crs)
            elif analysis_gdf.crs != target_crs:
                print(
                    f"Warning: CRS mismatch after merge ({analysis_gdf.crs}). Resetting to {target_crs}."
                )
                analysis_gdf.set_crs(target_crs, inplace=True)

            print(
                f"Merged dataset has {len(analysis_gdf)} rows (matched {len(analysis_gdf)} of {len(unified_df)} LSOAs)"
            )
            if analysis_gdf.empty:
                print(
                    "Error: Merged GeoDataFrame is empty. Check LSOA codes match and data validity."
                )
                return None, None

        except Exception as e:
            print(f"Error during merge operation: {e}")
            traceback.print_exc()
            return None, None

        # Diagnostic: Check geometry right after merge
        print(f"GeoDataFrame Info After Merge:")
        print(f"  Is GeoDataFrame: {isinstance(analysis_gdf, gpd.GeoDataFrame)}")
        if isinstance(analysis_gdf, gpd.GeoDataFrame):
            print(f"  Geometry column present: {'geometry' in analysis_gdf.columns}")
            print(f"  CRS: {analysis_gdf.crs}")
            print(f"  Non-null geometries: {analysis_gdf.geometry.notna().sum()}")
            print(f"  Valid geometries: {analysis_gdf.geometry.is_valid.sum()}")
            print(f"  Empty geometries: {analysis_gdf.geometry.is_empty.sum()}")
        else:
            print("  Merge failed to produce a GeoDataFrame.")
            return None, None

        # Filter invalid/missing/empty geometries *after* merge
        initial_rows = len(analysis_gdf)
        print(f"\nFiltering geometries in merged data ({initial_rows} rows initial):")

        # 1: Not None/NaN
        analysis_gdf = analysis_gdf[analysis_gdf.geometry.notna()]
        rows_after_notna = len(analysis_gdf)
        print(f"  Removed {initial_rows - rows_after_notna} rows with missing geometries.")
        if rows_after_notna == 0:
            print("Error: No geometries remain after checking for nulls.")
            return None, None

        # 2: Valid geometry objects
        analysis_gdf = analysis_gdf[analysis_gdf.geometry.is_valid]
        rows_after_is_valid = len(analysis_gdf)
        print(f"  Removed {rows_after_notna - rows_after_is_valid} rows with invalid geometries.")
        if rows_after_is_valid == 0:
            print("Error: No valid geometries remain.")
            return None, None

        # 3: Not empty geometry objects
        analysis_gdf = analysis_gdf[~analysis_gdf.geometry.is_empty]
        rows_after_is_empty = len(analysis_gdf)
        print(f"  Removed {rows_after_is_valid - rows_after_is_empty} rows with empty geometries.")

        final_rows = len(analysis_gdf)
        print(f"Proceeding with {final_rows} rows having valid geometries.")
        if final_rows == 0:
            print("Error: No valid geometries remain after filtering. Cannot create weights.")
            return None, None  # Return None for weights, but empty gdf might be useful for debug

        # Set index to LSOA code for potentially better performance and clearer weights object
        # Ensure the index is unique before setting it
        if not analysis_gdf[lsoa_code_col].is_unique:
            print(
                f"Warning: LSOA code column '{lsoa_code_col}' is not unique in the final filtered data. Using default integer index for weights."
            )
            analysis_gdf = analysis_gdf.reset_index(drop=True)  # Ensure clean default index
            use_index = False
            # Keep LSOA code as a column for potential joining later
        else:
            print(f"Setting index to unique LSOA code column '{lsoa_code_col}'.")
            analysis_gdf = analysis_gdf.set_index(lsoa_code_col)
            use_index = True  # Tell Queen to use the DataFrame index

        # Create Queen contiguity weights
        print("\nCreating Queen contiguity weights...")
        try:
            # Ensure it's still a GDF before passing
            if not isinstance(analysis_gdf, gpd.GeoDataFrame):
                print("Error: analysis_gdf is not a GeoDataFrame before weight creation.")
                return None, None

            # Try to create Queen weights
            try:
                w_queen = Queen.from_dataframe(
                    analysis_gdf, use_index=use_index, geom_col="geometry"
                )
                print(
                    f"Created Queen weights with {w_queen.n} areas and average of {w_queen.mean_neighbors:.2f} neighbors per area."
                )
                # Add island check
                if w_queen.islands:
                    print(
                        f"Warning: Found {len(w_queen.islands)} islands (areas with no neighbors) in the weights matrix."
                    )
                return w_queen, analysis_gdf
            except Exception as queen_err:
                print(f"Error creating Queen weights: {queen_err}")

                # Try KNN as fallback
                print("Attempting to create KNN weights as fallback...")
                try:
                    w_knn = KNN.from_dataframe(
                        analysis_gdf, k=5, use_index=use_index, geom_col="geometry"
                    )
                    print(f"Created KNN weights with {w_knn.n} areas and k=5 neighbors per area.")
                    return w_knn, analysis_gdf
                except Exception as knn_err:
                    print(f"Error creating KNN weights: {knn_err}")

                    # Final fallback - use the create_fallback_weights function
                    print("Using create_fallback_weights as final fallback...")
                    w_fallback = create_fallback_weights(analysis_gdf, method="mock", k=5)
                    if w_fallback is not None:
                        return w_fallback, analysis_gdf

                    # If all methods fail, return None for weights
                    return None, analysis_gdf
        except Exception as e:
            print(f"Error in weights creation: {e}")
            traceback.print_exc()
            return None, analysis_gdf  # Return GDF even if weights failed, maybe useful

    except Exception as e:
        print(f"An unexpected error occurred in create_lsoa_spatial_weights: {str(e)}")
        traceback.print_exc()
        return None, None


def aggregate_to_lad(unified_df):
    """
    Aggregate the unified dataset to LAD level for spatial regression.

    Args:
        unified_df (pd.DataFrame): DataFrame with LSOA data

    Returns:
        pd.DataFrame: Aggregated data at LAD level, or None if error
    """
    print("\n--- Aggregating Data to LAD Level ---")
    if "lad_code" not in unified_df.columns:
        print("Error: 'lad_code' column missing from unified_df. Cannot aggregate.")
        return None

    # Define aggregation dictionary dynamically based on available columns
    agg_dict = {"lad_name": "first"}
    cols_to_agg = [
        "imd_score_normalized",
        "NO2",
        "O3",
        "PM10",
        "PM2.5",
        "NO2_normalized",
        "PM2.5_normalized",
        "PM10_normalized",
        "env_justice_index",
        "air_pollution_index",
    ]
    for col in cols_to_agg:
        if col in unified_df.columns:
            agg_dict[col] = "mean"
        else:
            print(f"  Warning: Column '{col}' not found in unified_df, skipping aggregation.")

    if len(agg_dict) <= 1:  # Only lad_name was found
        print("Error: No data columns found to aggregate.")
        return None

    try:
        lad_aggregated = unified_df.groupby("lad_code").agg(agg_dict).reset_index()
        print(f"Aggregated to {len(lad_aggregated)} LADs")
        return lad_aggregated
    except Exception as e:
        print(f"Error during LAD aggregation: {e}")
        traceback.print_exc()
        return None


def create_lad_spatial_weights(lad_df, lad_gdf=None):
    """
    Create spatial weights matrix for LAD level analysis.
    Uses LAD boundaries if available, otherwise falls back to KNN based on codes/index.

    Args:
        lad_df (pd.DataFrame): DataFrame with aggregated LAD data (must include 'lad_code')
        lad_gdf (gpd.GeoDataFrame): GeoDataFrame with LAD boundaries (optional)

    Returns:
        tuple: (weights_matrix, analysis_df_or_gdf) or (None, None) on failure
    """
    print("\n--- Creating LAD Spatial Weights ---")

    if not PYSAL_AVAILABLE:
        print("PySAL not available. Cannot create LAD spatial weights.")
        return None, None

    if lad_df is None or lad_df.empty:
        print("Error: LAD DataFrame is empty or None. Cannot create weights.")
        return None, None

    try:
        # If we have LAD boundaries, use them
        if lad_gdf is not None and isinstance(lad_gdf, gpd.GeoDataFrame):
            print("LAD boundaries GeoDataFrame provided. Creating Queen weights.")
            # Identify the LAD code column in the GeoJSON
            lad_code_cols = [
                col
                for col in lad_gdf.columns
                if "LAD" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
            ]
            if not lad_code_cols:
                print(
                    "Error: Could not identify LAD code column in LAD GeoJSON. Available columns:"
                )
                print(lad_gdf.columns.tolist())
                return None, None
            lad_code_col_in_gdf = lad_code_cols[0]
            print(f"Using '{lad_code_col_in_gdf}' as LAD code column in GeoJSON")

            # Ensure CRS is suitable (e.g., BNG)
            target_crs = "EPSG:27700"
            if lad_gdf.crs is None:
                lad_gdf.set_crs(epsg=4326, inplace=True)  # Assume WGS84 if missing
            if str(lad_gdf.crs).upper() != target_crs:
                print(f"Reprojecting LAD boundaries from {lad_gdf.crs} to {target_crs}")
                try:
                    lad_gdf = lad_gdf.to_crs(target_crs)
                except Exception as e:
                    print(f"Error reprojecting LAD boundaries: {e}. Cannot create weights.")
                    return None, None

            # Merge the LAD dataset with the LAD boundaries
            print("Merging aggregated LAD data with LAD boundaries...")
            if "lad_code" not in lad_df.columns:
                print("Error: 'lad_code' column missing from aggregated LAD data.")
                return None, None

            lad_analysis_gdf = lad_gdf.merge(
                lad_df,
                left_on=lad_code_col_in_gdf,
                right_on="lad_code",
                how="inner",  # Keep only matching LADs
            )

            print(
                f"Merged LAD dataset has {len(lad_analysis_gdf)} rows (matched {len(lad_analysis_gdf)} of {len(lad_df)} LADs)"
            )
            if lad_analysis_gdf.empty:
                print("Error: Merged LAD GeoDataFrame is empty. Check code matching.")
                return None, None

            # Filter invalid geometries
            lad_analysis_gdf = lad_analysis_gdf[
                lad_analysis_gdf.geometry.notna()
                & lad_analysis_gdf.geometry.is_valid
                & ~lad_analysis_gdf.geometry.is_empty
            ]
            if lad_analysis_gdf.empty:
                print("Error: No valid geometries remain in merged LAD data.")
                return None, None
            print(f"Proceeding with {len(lad_analysis_gdf)} LADs having valid geometries.")

            # Create Queen contiguity weights for LADs
            print("Creating Queen contiguity weights for LADs...")
            # Use index if lad_code is unique and set as index, otherwise default
            if lad_analysis_gdf["lad_code"].is_unique:
                lad_analysis_gdf = lad_analysis_gdf.set_index("lad_code")
                use_index = True
            else:
                print("Warning: LAD codes are not unique in merged GDF. Using default index.")
                lad_analysis_gdf = lad_analysis_gdf.reset_index(drop=True)
                use_index = False

            w_queen = Queen.from_dataframe(lad_analysis_gdf, use_index=use_index)
            print(
                f"Created LAD Queen weights with {w_queen.n} areas and avg {w_queen.mean_neighbors:.2f} neighbors."
            )
            if w_queen.islands:
                print(f"Warning: Found {len(w_queen.islands)} islands in LAD weights.")
            return w_queen, lad_analysis_gdf  # Return weights and the GeoDataFrame

        # Fallback: No LAD boundaries, create mock KNN weights based on index/order
        else:
            print(
                "LAD boundaries not available or invalid. Creating mock KNN weights based on DataFrame index/order."
            )
            k_neighbors = min(5, len(lad_df) - 1)
            if k_neighbors <= 0:
                print("Error: Not enough LADs to create KNN weights.")
                return None, None

            # PySAL KNN requires coordinates. We don't have them here.
            # Create simple sequential neighbors as a mock fallback
            # This assumes some spatial proximity in the DataFrame's order, which is unlikely!
            # A better fallback might use centroids if available, or just skip spatial regression.
            print(
                f"Warning: Creating highly simplified sequential KNN={k_neighbors} weights. These are NOT geographically accurate."
            )
            ids = lad_df.index.tolist()  # Use the DataFrame's index
            n = len(ids)
            neighbors = {}
            weights = {}  # KNN weights are typically binary 0/1

            for i in range(n):
                current_id = ids[i]
                neighbor_ids = []
                # Connect to next k_neighbors areas (wrapping around)
                for j in range(1, k_neighbors + 1):
                    neighbor_ids.append(ids[(i + j) % n])
                neighbors[current_id] = neighbor_ids
                weights[current_id] = [1.0] * len(neighbor_ids)  # Binary weights

            try:
                # Create a W object directly from the neighbors dict
                w_knn_mock = W(neighbors, weights)  # Pass weights too
                w_knn_mock.transform = "R"  # Row-standardize
                print(
                    f"Created mock sequential KNN weights for {w_knn_mock.n} LADs with k={k_neighbors}"
                )
                # IMPORTANT: Return the weights object and the ORIGINAL lad_df
                return w_knn_mock, lad_df
            except Exception as e:
                print(f"Error creating mock W object: {e}")
                return None, None

    except Exception as e:
        print(f"Error creating LAD spatial weights: {str(e)}")
        traceback.print_exc()
        return None, None


# Moran's I and Getis-Ord functions remain largely the same, but add NaN checks


def calculate_morans_i(df_or_gdf, variable, weights_matrix):
    """
    Calculate Moran's I spatial autocorrelation statistic. Handles potential NaNs.

    Args:
        df_or_gdf (pd.DataFrame or gpd.GeoDataFrame): Data
        variable (str): Variable to analyze
        weights_matrix: PySAL spatial weights matrix

    Returns:
        tuple: (Moran's I, p-value) or (None, None) on failure
    """
    print(f"\nCalculating Global Moran's I for {variable}...")
    if variable not in df_or_gdf.columns:
        print(f"  Error: Variable '{variable}' not found.")
        return None, None
    if weights_matrix is None:
        print(f"  Error: Weights matrix is None. Cannot calculate Moran's I.")
        return None, None

    # Extract data series, aligning with weights index if possible
    y = df_or_gdf[variable].copy()
    if hasattr(weights_matrix, "id_order") and df_or_gdf.index.equals(
        pd.Index(weights_matrix.id_order)
    ):
        print(f"  Data index matches weights ID order.")
    elif hasattr(weights_matrix, "id_order"):
        print(f"  Reindexing data to match weights ID order.")
        try:
            # Ensure the index of df_or_gdf matches the type expected by id_order
            if df_or_gdf.index.dtype != type(weights_matrix.id_order[0]):
                print(
                    f"  Warning: Index dtype mismatch ({df_or_gdf.index.dtype} vs {type(weights_matrix.id_order[0])}). Attempting conversion."
                )
                try:
                    df_or_gdf.index = df_or_gdf.index.astype(type(weights_matrix.id_order[0]))
                except Exception as e:
                    print(f"  Error converting index type: {e}. Proceeding without reindex.")
                    # Fall through to use original y, hoping PySAL handles it or errors informatively

            y = y.reindex(weights_matrix.id_order)
        except Exception as e:
            print(f"  Error reindexing data: {e}. Calculation might be incorrect or fail.")
            # Proceed with original y, PySAL might handle alignment based on position

    # Check for NaNs *after* potential reindexing
    if y.isnull().any():
        nan_count = y.isnull().sum()
        median_val = y.median()
        print(
            f"  Warning: Found {nan_count} NaN(s) in '{variable}' data after alignment/extraction. Filling with median ({median_val:.4f})."
        )
        y.fillna(median_val, inplace=True)
        if y.isnull().any():  # Check if median filling failed (e.g., all NaNs)
            print(f"  Error: Failed to fill NaNs in '{variable}'. Cannot calculate Moran's I.")
            return None, None

    if not ESDA_AVAILABLE:
        print("  ESDA not available. Cannot calculate Moran's I using PySAL.")
        return None, None  # Manual calculation removed for brevity and PySAL focus

    try:
        moran = Moran(y, weights_matrix)
        # Check for potential issues in results
        moran_i = moran.I if np.isfinite(moran.I) else None
        p_sim = moran.p_sim if hasattr(moran, "p_sim") and np.isfinite(moran.p_sim) else None

        if moran_i is None or p_sim is None:
            print(
                f"  Moran's I calculation resulted in non-finite values. I={moran.I}, p={moran.p_sim}"
            )
            return None, None

        print(
            f"  Global Moran's I: {moran_i:.4f}, p-value: {p_sim:.4f} (based on {moran.permutations} permutations)"
        )
        return moran_i, p_sim
    except Exception as e:
        print(f"  Error calculating Moran's I for {variable}: {e}")
        traceback.print_exc()
        return None, None


def calculate_local_morans(gdf, variable, weights_matrix):
    """
    Calculate Local Moran's I statistics. Requires GeoDataFrame for merging results.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with data and geometry
        variable (str): Variable to analyze
        weights_matrix: PySAL spatial weights matrix

    Returns:
        gpd.GeoDataFrame: Input GeoDataFrame with Local Moran's results added, or None on failure
    """
    print(f"\nCalculating Local Moran's I for {variable}...")
    if not isinstance(gdf, gpd.GeoDataFrame):
        print("  Error: Input must be a GeoDataFrame for Local Moran's I processing.")
        return None
    if variable not in gdf.columns:
        print(f"  Error: Variable '{variable}' not found.")
        return None
    if weights_matrix is None:
        print(f"  Error: Weights matrix is None.")
        return None
    if not ESDA_AVAILABLE:
        print("  ESDA not available. Cannot calculate Local Moran's I.")
        return None

    # Align data `y` with weights order, handle NaNs (similar to global Moran's I)
    y = gdf[variable].copy()
    if hasattr(weights_matrix, "id_order"):
        print(f"  Reindexing data to match weights ID order for Local Moran's.")
        try:
            # Ensure index compatibility before reindexing
            if gdf.index.dtype != type(weights_matrix.id_order[0]):
                print(
                    f"  Warning: Index dtype mismatch in GDF for Local Moran's. Attempting conversion."
                )
                try:
                    original_index = gdf.index  # Keep original index if conversion fails
                    gdf.index = gdf.index.astype(type(weights_matrix.id_order[0]))
                except Exception as e:
                    print(f"  Error converting index type: {e}. Using original index.")
                    gdf.index = original_index  # Restore original index

            y = y.reindex(weights_matrix.id_order)
        except Exception as e:
            print(f"  Error reindexing data for Local Moran's: {e}.")
            # Continue with potentially unaligned y, PySAL might handle it or error

    if y.isnull().any():
        nan_count = y.isnull().sum()
        median_val = y.median()
        print(
            f"  Warning: Found {nan_count} NaN(s) in '{variable}' after alignment. Filling with median ({median_val:.4f})."
        )
        y.fillna(median_val, inplace=True)
        if y.isnull().any():
            print(f"  Error: Failed to fill NaNs in '{variable}'. Cannot calculate Local Moran's.")
            return None

    try:
        local_moran = Moran_Local(y, weights_matrix)

        # Create DataFrame with results, ensuring index matches weights order
        results_df = pd.DataFrame(index=weights_matrix.id_order)  # Use weights ID order for index
        results_df[f"{variable}_local_I"] = local_moran.Is
        results_df[f"{variable}_pval"] = local_moran.p_sim
        results_df[f"{variable}_q"] = local_moran.q  # Quadrant codes

        # Define cluster types based on quadrant and significance
        sig_level = 0.05
        results_df[f"{variable}_significant"] = results_df[f"{variable}_pval"] < sig_level
        results_df[f"{variable}_cluster_type"] = "Not significant"

        # Assign cluster labels based on quadrant for significant results
        # Quadrants: 1=HH, 2=LH, 3=LL, 4=HL
        results_df.loc[
            (results_df[f"{variable}_q"] == 1) & results_df[f"{variable}_significant"],
            f"{variable}_cluster_type",
        ] = "HH (High-High)"
        results_df.loc[
            (results_df[f"{variable}_q"] == 2) & results_df[f"{variable}_significant"],
            f"{variable}_cluster_type",
        ] = "LH (Low-High)"
        results_df.loc[
            (results_df[f"{variable}_q"] == 3) & results_df[f"{variable}_significant"],
            f"{variable}_cluster_type",
        ] = "LL (Low-Low)"
        results_df.loc[
            (results_df[f"{variable}_q"] == 4) & results_df[f"{variable}_significant"],
            f"{variable}_cluster_type",
        ] = "HL (High-Low)"

        print(f"  Calculated Local Moran's I for {len(results_df)} areas.")

        # Merge results back into the original GeoDataFrame using the index
        # Ensure gdf index is compatible with results_df index
        if gdf.index.equals(results_df.index):
            gdf_with_results = gdf.join(results_df)
        else:
            print(
                f"  Warning: GeoDataFrame index does not match results index. Attempting merge on index."
            )
            # This might fail if indexes aren't compatible types or values
            try:
                # Ensure gdf index is suitable type if results_df index is numeric/string from id_order
                if results_df.index.dtype != gdf.index.dtype:
                    gdf.index = gdf.index.astype(results_df.index.dtype)
                gdf_with_results = gdf.join(
                    results_df, how="left"
                )  # Left join to keep all original geometries
                # Check if join worked by looking for NaNs in added columns where they shouldn't be
                if (
                    gdf_with_results[f"{variable}_local_I"].isnull().sum() > nan_count
                ):  # Allow original NaNs
                    print(
                        f"  Warning: Potential issue joining Local Moran results back to GeoDataFrame."
                    )
            except Exception as join_err:
                print(
                    f"  Error joining Local Moran results to GeoDataFrame: {join_err}. Returning GDF without results."
                )
                return gdf  # Return original gdf

        return gdf_with_results

    except Exception as e:
        print(f"  Error calculating Local Moran's I for {variable}: {e}")
        traceback.print_exc()
        return None


def calculate_getis_ord(gdf, variable, weights_matrix):
    """
    Calculate Getis-Ord Gi* statistics. Requires GeoDataFrame for merging results.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with data and geometry
        variable (str): Variable to analyze
        weights_matrix: PySAL spatial weights matrix

    Returns:
        gpd.GeoDataFrame: Input GeoDataFrame with Getis-Ord Gi* results added, or None on failure
    """
    print(f"\nCalculating Getis-Ord Gi* for {variable}...")
    if not isinstance(gdf, gpd.GeoDataFrame):
        print("  Error: Input must be a GeoDataFrame.")
        return None
    if variable not in gdf.columns:
        print(f"  Error: Variable '{variable}' not found.")
        return None
    if weights_matrix is None:
        print(f"  Error: Weights matrix is None.")
        return None
    if not ESDA_AVAILABLE:
        print("  ESDA not available. Cannot calculate Getis-Ord Gi*.")
        return None

    # Align data `y` with weights order, handle NaNs
    y = gdf[variable].copy()
    if hasattr(weights_matrix, "id_order"):
        print(f"  Reindexing data to match weights ID order for Getis-Ord.")
        try:
            if gdf.index.dtype != type(weights_matrix.id_order[0]):
                print(
                    f"  Warning: Index dtype mismatch in GDF for Getis-Ord. Attempting conversion."
                )
                try:
                    original_index = gdf.index
                    gdf.index = gdf.index.astype(type(weights_matrix.id_order[0]))
                except Exception as e:
                    print(f"  Error converting index type: {e}. Using original index.")
                    gdf.index = original_index

            y = y.reindex(weights_matrix.id_order)
        except Exception as e:
            print(f"  Error reindexing data for Getis-Ord: {e}.")

    if y.isnull().any():
        nan_count = y.isnull().sum()
        median_val = y.median()
        print(
            f"  Warning: Found {nan_count} NaN(s) in '{variable}' after alignment. Filling with median ({median_val:.4f})."
        )
        y.fillna(median_val, inplace=True)
        if y.isnull().any():
            print(f"  Error: Failed to fill NaNs in '{variable}'. Cannot calculate Getis-Ord.")
            return None

    try:
        g_local = G_Local(y, weights_matrix)

        # Create DataFrame with results, ensuring index matches weights order
        results_df = pd.DataFrame(index=weights_matrix.id_order)
        results_df[f"{variable}_gi_star_z"] = g_local.Zs  # Z-scores are generally used for Gi*
        results_df[f"{variable}_gi_pval"] = g_local.p_sim

        # Determine hotspot/coldspot type based on Z-score and significance
        sig_level = 0.05
        results_df[f"{variable}_hotspot_type"] = "Not significant"
        # Hotspot: High Z-score, low p-value
        results_df.loc[
            (results_df[f"{variable}_gi_star_z"] > norm.ppf(1 - sig_level / 2))
            & (results_df[f"{variable}_gi_pval"] < sig_level),
            f"{variable}_hotspot_type",
        ] = "Hotspot (95%)"
        # Coldspot: Low Z-score, low p-value
        results_df.loc[
            (results_df[f"{variable}_gi_star_z"] < norm.ppf(sig_level / 2))
            & (results_df[f"{variable}_gi_pval"] < sig_level),
            f"{variable}_hotspot_type",
        ] = "Coldspot (95%)"
        # Optionally add more confidence levels (e.g., 99%, 90%)
        sig_level_99 = 0.01
        results_df.loc[
            (results_df[f"{variable}_gi_star_z"] > norm.ppf(1 - sig_level_99 / 2))
            & (results_df[f"{variable}_gi_pval"] < sig_level_99),
            f"{variable}_hotspot_type",
        ] = "Hotspot (99%)"
        results_df.loc[
            (results_df[f"{variable}_gi_star_z"] < norm.ppf(sig_level_99 / 2))
            & (results_df[f"{variable}_gi_pval"] < sig_level_99),
            f"{variable}_hotspot_type",
        ] = "Coldspot (99%)"

        print(f"  Calculated Getis-Ord Gi* for {len(results_df)} areas.")

        # Merge results back into the original GeoDataFrame using the index
        if gdf.index.equals(results_df.index):
            gdf_with_results = gdf.join(results_df)
        else:
            print(
                f"  Warning: GeoDataFrame index does not match results index. Attempting merge on index."
            )
            try:
                if results_df.index.dtype != gdf.index.dtype:
                    gdf.index = gdf.index.astype(results_df.index.dtype)
                gdf_with_results = gdf.join(results_df, how="left")
                if gdf_with_results[f"{variable}_gi_star_z"].isnull().sum() > nan_count:
                    print(
                        f"  Warning: Potential issue joining Getis-Ord results back to GeoDataFrame."
                    )
            except Exception as join_err:
                print(
                    f"  Error joining Getis-Ord results to GeoDataFrame: {join_err}. Returning GDF without results."
                )
                return gdf

        return gdf_with_results

    except Exception as e:
        print(f"  Error calculating Getis-Ord Gi* for {variable}: {e}")
        traceback.print_exc()
        return None


def visualize_spatial_clusters(gdf_with_results, variable):
    """
    Visualize spatial clusters and outliers from Local Moran's I analysis.
    Uses Matplotlib.

    Args:
        gdf_with_results (gpd.GeoDataFrame): GeoDataFrame with geometries AND Local Moran results.
        variable (str): Variable analyzed (used for column names and titles).
    """
    print(f"\nVisualizing Local Moran's I spatial clusters for {variable}...")
    cluster_col = f"{variable}_cluster_type"

    if not isinstance(gdf_with_results, gpd.GeoDataFrame):
        print("  Error: Input is not a GeoDataFrame. Cannot create map.")
        return
    if cluster_col not in gdf_with_results.columns:
        print(f"  Error: Cluster type column '{cluster_col}' not found in GeoDataFrame.")
        return
    if "geometry" not in gdf_with_results.columns or gdf_with_results.geometry.isnull().all():
        print("  Error: No valid geometries found in GeoDataFrame. Cannot create map.")
        return

    # --- Bar Plot of Cluster Counts ---
    plt.figure(figsize=(10, 6))
    cluster_counts = gdf_with_results[cluster_col].value_counts()
    if cluster_counts.empty:
        print("  No cluster types found to plot.")
    else:
        try:
            sns.barplot(
                x=cluster_counts.index, y=cluster_counts.values, palette="viridis"
            )  # Use seaborn for nice barplot
            plt.title(f"Counts of Spatial Cluster Types (Local Moran's I) for {variable}")
            plt.xlabel("Cluster Type")
            plt.ylabel("Number of LSOAs")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"outputs/spatial_hotspots/LISA_cluster_counts_{variable}.png", dpi=300)
            print(
                f"  Saved cluster counts plot to 'outputs/spatial_hotspots/LISA_cluster_counts_{variable}.png'"
            )
            plt.close()  # Close the figure to free memory
        except Exception as e:
            print(f"  Error creating/saving cluster counts plot: {e}")

    # --- Map of Clusters ---
    try:
        # Define a robust color map
        color_map = {
            "HH (High-High)": "#d7191c",  # Red
            "LL (Low-Low)": "#2c7bb6",  # Blue
            "LH (Low-High)": "#abd9e9",  # Light Blue
            "HL (High-Low)": "#fdae61",  # Orange
            "Not significant": "#d3d3d3",  # Light Grey
        }
        default_color = "#808080"  # Darker grey for unexpected values

        # Create a list of colors for plotting, handling missing types
        color_list = [color_map.get(ct, default_color) for ct in gdf_with_results[cluster_col]]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot geometries with assigned colors
        gdf_with_results.plot(
            color=color_list, ax=ax, edgecolor="grey", linewidth=0.1  # Add faint edges for clarity
        )

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = []
        # Add legend elements only for types present in the data
        present_types = gdf_with_results[cluster_col].unique()
        for label, color in color_map.items():
            if label in present_types:
                legend_elements.append(Patch(facecolor=color, edgecolor="grey", label=label))
        # Add legend entry if default color was used
        if default_color in color_list and "Other/Unknown" not in [
            p.get_label() for p in legend_elements
        ]:
            legend_elements.append(
                Patch(facecolor=default_color, edgecolor="grey", label="Other/Unknown")
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                title="LISA Cluster Types",
                loc="upper left",
                fontsize="small",
            )
        else:
            print("  No legend elements to display.")

        ax.set_title(f"Local Moran's I Spatial Clusters for {variable}", fontsize=16)
        ax.set_axis_off()  # Turn off axis labels/ticks for maps
        plt.tight_layout()
        plt.savefig(
            f"outputs/spatial_hotspots/LISA_cluster_map_{variable}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  Saved cluster map to 'outputs/spatial_hotspots/LISA_cluster_map_{variable}.png'")
        plt.close()  # Close the figure

    except Exception as e:
        print(f"  Error creating/saving spatial cluster map: {str(e)}")
        traceback.print_exc()
        plt.close()  # Ensure plot is closed even if error occurs


def visualize_hotspots(gdf_with_results, variable):
    """
    Visualize hotspots and coldspots from Getis-Ord Gi* analysis.
    Uses Matplotlib.

    Args:
        gdf_with_results (gpd.GeoDataFrame): GeoDataFrame with geometries AND Getis-Ord results.
        variable (str): Variable analyzed (used for column names and titles).
    """
    print(f"\nVisualizing Getis-Ord Gi* hotspots/coldspots for {variable}...")
    hotspot_col = f"{variable}_hotspot_type"

    if not isinstance(gdf_with_results, gpd.GeoDataFrame):
        print("  Error: Input is not a GeoDataFrame. Cannot create map.")
        return
    if hotspot_col not in gdf_with_results.columns:
        print(f"  Error: Hotspot type column '{hotspot_col}' not found.")
        return
    if "geometry" not in gdf_with_results.columns or gdf_with_results.geometry.isnull().all():
        print("  Error: No valid geometries found. Cannot create map.")
        return

    # --- Bar Plot of Hotspot Type Counts ---
    plt.figure(figsize=(10, 6))
    hotspot_counts = gdf_with_results[hotspot_col].value_counts()
    if hotspot_counts.empty:
        print("  No hotspot types found to plot.")
    else:
        try:
            sns.barplot(
                x=hotspot_counts.index, y=hotspot_counts.values, palette="coolwarm_r"
            )  # Use coolwarm palette
            plt.title(f"Counts of Hotspot Types (Getis-Ord Gi*) for {variable}")
            plt.xlabel("Hotspot Type")
            plt.ylabel("Number of LSOAs")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"outputs/spatial_hotspots/Hotspot_type_counts_{variable}.png", dpi=300)
            print(
                f"  Saved hotspot counts plot to 'outputs/spatial_hotspots/Hotspot_type_counts_{variable}.png'"
            )
            plt.close()
        except Exception as e:
            print(f"  Error creating/saving hotspot counts plot: {e}")

    # --- Map of Hotspots/Coldspots ---
    try:
        # Define a robust color map for hotspots/coldspots
        color_map = {
            "Hotspot (99%)": "#d73027",  # Dark Red
            "Hotspot (95%)": "#fc8d59",  # Lighter Red/Orange
            "Coldspot (99%)": "#4575b4",  # Dark Blue
            "Coldspot (95%)": "#91bfdb",  # Lighter Blue
            "Not significant": "#d3d3d3",  # Light Grey
        }
        default_color = "#808080"  # Darker grey for unexpected values

        # Create a list of colors for plotting
        color_list = [color_map.get(ht, default_color) for ht in gdf_with_results[hotspot_col]]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot geometries with assigned colors
        gdf_with_results.plot(color=color_list, ax=ax, edgecolor="grey", linewidth=0.1)

        # Create custom legend
        from matplotlib.patches import Patch

        legend_elements = []
        present_types = gdf_with_results[hotspot_col].unique()
        # Define order for legend if desired
        legend_order = [
            "Hotspot (99%)",
            "Hotspot (95%)",
            "Not significant",
            "Coldspot (95%)",
            "Coldspot (99%)",
        ]
        for label in legend_order:
            if label in present_types:
                legend_elements.append(
                    Patch(facecolor=color_map.get(label), edgecolor="grey", label=label)
                )
        # Add other present types not in the predefined order
        for label in present_types:
            if label not in legend_order and label in color_map:
                legend_elements.append(
                    Patch(facecolor=color_map.get(label), edgecolor="grey", label=label)
                )

        if default_color in color_list and "Other/Unknown" not in [
            p.get_label() for p in legend_elements
        ]:
            legend_elements.append(
                Patch(facecolor=default_color, edgecolor="grey", label="Other/Unknown")
            )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                title="Getis-Ord Gi* Hotspot Type",
                loc="upper left",
                fontsize="small",
            )
        else:
            print("  No legend elements to display.")

        ax.set_title(f"Getis-Ord Gi* Hotspots and Coldspots for {variable}", fontsize=16)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(
            f"outputs/spatial_hotspots/Hotspot_map_{variable}.png", dpi=300, bbox_inches="tight"
        )
        print(f"  Saved hotspot map to 'outputs/spatial_hotspots/Hotspot_map_{variable}.png'")
        plt.close()

    except Exception as e:
        print(f"  Error creating/saving spatial hotspot map: {str(e)}")
        traceback.print_exc()
        plt.close()


def run_spatial_regression(df_or_gdf, y_variable, x_variables, weights_matrix):
    """
    Run OLS and Spatial Lag regression models.

    Args:
        df_or_gdf (pd.DataFrame or gpd.GeoDataFrame): Data.
        y_variable (str): Dependent variable name.
        x_variables (list): List of independent variable names.
        weights_matrix: PySAL spatial weights matrix.

    Returns:
        dict: Dictionary containing OLS and Spatial Lag model results, or None on failure.
    """
    print(f"\n--- Running Spatial Regression for {y_variable} ---")
    print(f"Independent variables: {x_variables}")

    if not PYSAL_AVAILABLE:
        print("  spreg module not available (part of PySAL). Cannot run spatial regression.")
        return None
    if weights_matrix is None:
        print("  Error: Weights matrix is None. Cannot run spatial regression.")
        return None

    # Check if variables exist
    all_vars = [y_variable] + x_variables
    missing_vars = [var for var in all_vars if var not in df_or_gdf.columns]
    if missing_vars:
        print(f"  Error: Missing required variables: {missing_vars}")
        return None

    # Prepare data, aligning with weights index and handling NaNs
    data_for_reg = df_or_gdf[all_vars].copy()

    # Align index with weights
    if hasattr(weights_matrix, "id_order"):
        print("  Aligning regression data index with weights ID order.")
        try:
            if data_for_reg.index.dtype != type(weights_matrix.id_order[0]):
                data_for_reg.index = data_for_reg.index.astype(type(weights_matrix.id_order[0]))
            data_for_reg = data_for_reg.reindex(weights_matrix.id_order)
        except Exception as e:
            print(
                f"  Warning: Error reindexing data for regression: {e}. Proceeding with original data."
            )

    # Handle NaNs row-wise (listwise deletion is common in regression)
    initial_n = len(data_for_reg)
    data_for_reg.dropna(inplace=True)
    final_n = len(data_for_reg)
    if final_n < initial_n:
        print(f"  Removed {initial_n - final_n} rows with NaN values for regression variables.")

    if final_n < len(x_variables) + 1:  # Need more observations than parameters
        print(f"  Error: Not enough valid observations ({final_n}) to run regression.")
        return None

    # Ensure weights matrix corresponds to the filtered data
    # This requires weights to be filterable based on the remaining IDs.
    # PySAL weights can often be subsetted.
    try:
        # Get the index (IDs) of the rows remaining in data_for_reg
        valid_ids = data_for_reg.index.tolist()
        # Filter the weights matrix to include only these IDs and their relationships
        try:
            w_filtered = weights_matrix.subset(valid_ids, drop_islands=True)
            if w_filtered.n != final_n:
                print(
                    f"  Warning: Filtered weights matrix size ({w_filtered.n}) does not match filtered data size ({final_n}). Regression might be unreliable."
                )
                # Decide whether to proceed or return None
                # return None
            print(f"  Filtered weights matrix to {w_filtered.n} observations.")
        except AttributeError as e:
            print(
                f"  Error filtering weights matrix for regression: {e}. Using original weights matrix."
            )
            w_filtered = weights_matrix
    except Exception as e:
        print(f"Error during weights filtering: {e}")

    # Prepare Y and X numpy arrays from the filtered data
    y = data_for_reg[y_variable].values.reshape(-1, 1)  # Reshape for spreg
    X = data_for_reg[x_variables].values

    results = {}

    # --- OLS Regression ---
    try:
        print("\nRunning OLS Regression...")
        ols = OLS(
            y,
            X,
            w=w_filtered,
            name_y=y_variable,
            name_x=x_variables,
            name_ds="LAD_Data",
            spat_diag=True,
        )  # Add spatial diagnostics
        print(ols.summary)
        results["ols"] = ols
    except Exception as e:
        print(f"  Error running OLS: {e}")
        traceback.print_exc()
        # Decide if we can continue to spatial lag without OLS results

    # --- Spatial Lag Model (ML Estimation) ---
    try:
        print("\nRunning Spatial Lag (ML) Regression...")
        # Ensure w_filtered is valid
        if w_filtered is None or w_filtered.n == 0:
            raise ValueError("Filtered weights matrix is invalid for Spatial Lag model.")

        spatial_lag = ML_Lag(
            y, X, w=w_filtered, name_y=y_variable, name_x=x_variables, name_ds="LAD_Data"
        )
        print(spatial_lag.summary)
        results["spatial_lag"] = spatial_lag
    except np.linalg.LinAlgError as la_err:
        print(f"  Error running Spatial Lag model (Linear Algebra Error): {la_err}")
        print("  This often indicates multicollinearity or issues with the data/weights matrix.")
        traceback.print_exc()
    except Exception as e:
        print(f"  Error running Spatial Lag model: {e}")
        traceback.print_exc()

    if not results:  # If neither model ran successfully
        return None

    return results


# Manual/Fallback weights creation - Keep for robustness if needed, but prioritize accurate weights
def create_mock_spatial_weights(df, k_neighbors=5):
    """
    Create mock spatial weights matrix based on DataFrame index order (KNN-like).
    This is a fallback method and NOT geographically accurate.

    Args:
        df (pd.DataFrame): DataFrame with data
        k_neighbors (int): Number of 'neighbors' based on index order

    Returns:
        libpysal.weights.W: Mock spatial weights matrix or None on failure
    """
    print("\n--- Creating Mock Spatial Weights (Fallback) ---")
    print(
        f"Warning: Using simplified sequential KNN={k_neighbors} weights based on index order. Not geographically accurate."
    )

    if not PYSAL_AVAILABLE:
        print("  PySAL not available. Cannot create mock weights.")
        return None

    n = len(df)
    if n <= k_neighbors:
        print(
            f"  Error: Number of observations ({n}) is less than or equal to k ({k_neighbors}). Cannot create KNN weights."
        )
        return None

    ids = df.index.tolist()  # Use the actual index values as IDs

    neighbors = {}
    weights_dict = {}  # Use dict for W constructor

    for i in range(n):
        current_id = ids[i]
        neighbor_ids = []
        # Connect to next k_neighbors areas (wrapping around)
        for j in range(1, k_neighbors + 1):
            neighbor_ids.append(ids[(i + j) % n])

        neighbors[current_id] = neighbor_ids
        weights_dict[current_id] = [1.0] * len(neighbor_ids)  # Binary weights

    try:
        w_mock = W(neighbors, weights=weights_dict, id_order=ids)  # Provide id_order explicitly
        w_mock.transform = "R"  # Row-standardize is common
        print(f"Created mock sequential KNN weights for {w_mock.n} areas with k={k_neighbors}")
        return w_mock
    except Exception as e:
        print(f"  Error creating mock W object: {e}")
        traceback.print_exc()
        return None


def create_fallback_weights(df, method="knn", k=5):
    """
    Create fallback weights when proper boundaries aren't available.

    Args:
        df (pd.DataFrame or gpd.GeoDataFrame): DataFrame with data
        method (str): Method to use for creating weights ('knn', 'queen', 'mock')
        k (int): Number of neighbors for KNN weights

    Returns:
        libpysal.weights.W: Spatial weights matrix or None on failure
    """
    print(f"Creating fallback weights using {method} method...")

    if not PYSAL_AVAILABLE:
        print("  PySAL not available. Cannot create weights.")
        return None

    if df is None or len(df) == 0:
        print("  Error: Empty or None DataFrame provided.")
        return None

    # If it's a GeoDataFrame with valid geometries, try spatial methods
    if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
        # Check if geometries are valid
        valid_geoms = df.geometry.is_valid.sum()
        if valid_geoms > 0:
            if method == "knn" and valid_geoms == len(df):
                try:
                    print(f"  Creating KNN weights from {valid_geoms} valid geometries...")
                    knn = weights.KNN.from_dataframe(df, k=k)
                    print(f"  Created KNN weights with {knn.n} areas and k={k}")
                    return knn
                except Exception as e:
                    print(f"  Error creating KNN weights: {e}")

            if method == "queen" and valid_geoms == len(df):
                try:
                    print(f"  Creating Queen weights from {valid_geoms} valid geometries...")
                    queen = weights.Queen.from_dataframe(df)
                    print(f"  Created Queen weights with {queen.n} areas")
                    return queen
                except Exception as e:
                    print(f"  Error creating Queen weights: {e}")

    # If lat/lon columns exist, use them for KNN
    if "latitude" in df.columns and "longitude" in df.columns:
        try:
            print("  Creating KNN weights from latitude/longitude columns...")
            coords = df[["longitude", "latitude"]].values
            knn = weights.KNN.from_array(coords, k=k)
            print(f"  Created KNN weights with {knn.n} areas and k={k}")
            return knn
        except Exception as e:
            print(f"  Error creating KNN weights from coordinates: {e}")

    # Final fallback - mock weights based on index order
    print("  Falling back to mock sequential weights...")
    return create_mock_spatial_weights(df, k_neighbors=k)


def main():
    """Main function to execute the spatial autocorrelation and hotspot analysis."""
    print("======== Starting Spatial Analysis Script ========")

    # --- 1. Load Data ---
    unified_df, lsoa_gdf, wards_gdf = load_data()
    if unified_df is None:
        print("Failed to load unified dataset. Exiting.")
        return

    # --- 2. Prepare for LSOA Level Analysis ---
    env_justice_vars = ["env_justice_index", "air_pollution_index", "imd_score_normalized"]
    pollution_vars = ["NO2", "PM2.5", "PM10", "O3"]  # Add others if available
    all_lsoa_vars = env_justice_vars + pollution_vars

    # Filter variables that actually exist in the loaded dataframe
    available_lsoa_vars = [v for v in all_lsoa_vars if v in unified_df.columns]
    print(f"\nVariables available for LSOA analysis: {available_lsoa_vars}")
    if not available_lsoa_vars:
        print("No variables found for LSOA analysis. Check column names in unified dataset.")
        # Decide whether to proceed to LAD analysis or exit
        # return # Exit if no LSOA vars

    # --- 3. Create LSOA Spatial Weights ---
    lsoa_weights, lsoa_analysis_gdf = create_lsoa_spatial_weights(unified_df, lsoa_gdf)

    analysis_successful = lsoa_weights is not None and lsoa_analysis_gdf is not None
    lsoa_results = {}  # Store global Moran's I results

    # --- 4. LSOA Level Spatial Autocorrelation & Hotspots ---
    if analysis_successful:
        print("\n======== LSOA Level Analysis (Using Accurate Weights) ========")
        # Ensure the analysis GDF has the needed vars after merge/filtering
        available_lsoa_vars_in_gdf = [
            v for v in available_lsoa_vars if v in lsoa_analysis_gdf.columns
        ]
        if not available_lsoa_vars_in_gdf:
            print(
                "Error: None of the target variables are present in the final LSOA GeoDataFrame after merging/filtering."
            )
        else:
            print(f"Analyzing LSOA variables in GeoDataFrame: {available_lsoa_vars_in_gdf}")
            current_gdf_state = lsoa_analysis_gdf.copy()  # Keep track of GDF with results

            for variable in available_lsoa_vars_in_gdf:
                print(f"\n--- Analyzing LSOA Variable: {variable} ---")
                # Global Moran's I
                moran_i, p_val = calculate_morans_i(current_gdf_state, variable, lsoa_weights)
                if moran_i is not None:
                    lsoa_results[variable] = {"morans_i": moran_i, "p_value": p_val}

                if ESDA_AVAILABLE:
                    # Local Moran's I (updates the GDF)
                    temp_gdf = calculate_local_morans(current_gdf_state, variable, lsoa_weights)
                    if temp_gdf is not None:
                        current_gdf_state = temp_gdf  # Update GDF with results
                        visualize_spatial_clusters(current_gdf_state, variable)
                    else:
                        print(
                            f"  Skipping Local Moran visualization for {variable} due to calculation error."
                        )

                    # Getis-Ord Gi* (updates the GDF)
                    temp_gdf = calculate_getis_ord(current_gdf_state, variable, lsoa_weights)
                    if temp_gdf is not None:
                        current_gdf_state = temp_gdf  # Update GDF with results
                        visualize_hotspots(current_gdf_state, variable)
                    else:
                        print(
                            f"  Skipping Getis-Ord visualization for {variable} due to calculation error."
                        )

            # Optionally save the final GDF with all results
            try:
                output_gdf_path = "outputs/spatial_hotspots/lsoa_analysis_results.gpkg"  # GeoPackage is often better than Shapefile
                current_gdf_state.to_file(output_gdf_path, driver="GPKG")
                print(f"\nSaved LSOA GeoDataFrame with all results to {output_gdf_path}")
            except Exception as e:
                print(f"\nError saving final LSOA GeoDataFrame: {e}")

    else:
        print("\n======== LSOA Level Analysis (Using Mock Weights - Fallback) ========")
        print("Warning: Accurate LSOA weights could not be created. Using simplified mock weights.")
        print(
            "Warning: Spatial analysis results and visualizations will be based on index order, not geography."
        )

        lsoa_mock_weights = create_mock_spatial_weights(unified_df)

        if lsoa_mock_weights is not None:
            # Note: We use unified_df here, which lacks geometry. Visualization functions will fail.
            for variable in available_lsoa_vars:
                print(f"\n--- Analyzing LSOA Variable (Mock): {variable} ---")
                # Global Moran's I
                moran_i, p_val = calculate_morans_i(unified_df, variable, lsoa_mock_weights)
                if moran_i is not None:
                    lsoa_results[variable] = {"morans_i": moran_i, "p_value": p_val}

                # Cannot run Local Moran/Getis-Ord meaningfully without geometry for visualization
                print(
                    "  Skipping Local Moran / Getis-Ord analysis and visualization (requires geometry)."
                )

        else:
            print("Error: Could not create mock LSOA weights. Skipping LSOA analysis.")

    # --- 5. LAD Level Spatial Regression ---
    if PYSAL_AVAILABLE:
        print("\n======== LAD Level Spatial Regression ========")
        # Aggregate data first
        lad_df = aggregate_to_lad(unified_df)

        if lad_df is not None:
            # Define variables for regression (use aggregated names)
            lad_y_variable = "env_justice_index"
            lad_x_variables = ["imd_score_normalized", "NO2", "PM2.5"]  # Use aggregated means

            # Check if regression variables exist in the aggregated df
            required_reg_vars = [lad_y_variable] + lad_x_variables
            available_reg_vars = [v for v in required_reg_vars if v in lad_df.columns]

            if set(available_reg_vars) == set(required_reg_vars):
                # Create LAD weights (try with actual boundaries if available, else mock)
                # Pass wards_gdf as potential LAD boundaries if appropriate (check schema)
                # NOTE: Ward boundaries might not be LAD boundaries! Need actual LAD boundaries file usually.
                # For now, assume wards_gdf is NOT LAD boundaries, triggering fallback.
                print(
                    "Attempting to create LAD weights (using fallback mock KNN as LAD boundaries not provided/matched)."
                )
                lad_weights, lad_analysis_df_or_gdf = create_lad_spatial_weights(
                    lad_df, lad_gdf=None
                )  # Explicitly pass None for lad_gdf

                if lad_weights is not None and lad_analysis_df_or_gdf is not None:
                    # Run spatial regression
                    regression_results = run_spatial_regression(
                        lad_analysis_df_or_gdf, lad_y_variable, lad_x_variables, lad_weights
                    )

                    if regression_results:
                        print("\nSpatial Regression Summary:")
                        if "ols" in regression_results:
                            print("--- OLS Results ---")
                            # Access key OLS stats if needed, summary already printed in function
                            print(f"OLS R-squared: {regression_results['ols'].r2:.4f}")
                            print(f"OLS Adj. R-squared: {regression_results['ols'].ar2:.4f}")
                        if "spatial_lag" in regression_results:
                            print("\n--- Spatial Lag Results ---")
                            # Access key Spatial Lag stats
                            print(
                                f"Spatial Lag Pseudo R-squared: {regression_results['spatial_lag'].pr2:.4f}"
                            )
                            print(
                                f"Spatial Lag Rho (Spatial Autoregressive Coeff): {regression_results['spatial_lag'].rho:.4f}"
                            )
                            # Summary already printed in function
                    else:
                        print("Spatial regression failed to produce results.")
                else:
                    print("Could not create LAD spatial weights. Skipping spatial regression.")
            else:
                missing = set(required_reg_vars) - set(available_reg_vars)
                print(f"Cannot run LAD spatial regression. Missing aggregated variables: {missing}")
        else:
            print("Could not aggregate data to LAD level. Skipping spatial regression.")
    else:
        print("\nPySAL (spreg) not available. Skipping LAD Level Spatial Regression.")

    # --- 6. Final Summary ---
    print("\n======== Analysis Summary ========")
    if lsoa_results:
        print("Global Moran's I Results (LSOA Level):")
        for var, res in lsoa_results.items():
            print(f"  {var}: Moran's I = {res['morans_i']:.4f}, p-value = {res['p_value']:.4f}")
    else:
        print("No Global Moran's I results calculated.")

    print("\nSpatial analysis script finished.")
    print("Outputs saved to the 'outputs/spatial_hotspots' directory.")


if __name__ == "__main__":
    # Add basic error handling for the main execution
    try:
        main()
    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"An unexpected error occurred during main execution: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
