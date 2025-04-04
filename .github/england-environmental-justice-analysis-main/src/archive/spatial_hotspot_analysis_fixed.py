"""
Fixed Spatial Autocorrelation and Hotspot Analysis for Environmental Justice Project

This script implements advanced spatial statistics:
- Moran's I for spatial autocorrelation
- Getis-Ord Gi* for hotspot identification
- Spatial regression models accounting for neighborhood effects
- Spatial visualizations of hotspots and clusters

FIXED: Now using a valid LSOA boundaries GeoJSON file with proper geometries
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
import warnings
import traceback

# Check for spatial analysis libraries
try:
    import libpysal as lps
    from libpysal.weights import Queen, KNN, W

    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("Warning: libpysal not available. Spatial weights will be limited.")

try:
    import esda

    ESDA_AVAILABLE = True
except ImportError:
    ESDA_AVAILABLE = False
    print("Warning: esda not available. Spatial autocorrelation measures will be unavailable.")

try:
    from spreg import OLS, ML_Lag

    SPREG_AVAILABLE = True
except ImportError:
    SPREG_AVAILABLE = False
    print("Warning: spreg not available. Spatial regression will be unavailable.")


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
        print(f"  {dep}: {'Available' if available else 'Not Available'}")


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
    base_dir = os.environ.get("UK_ENV_BASE_DIR", os.path.join(".."))
    data_dir = os.environ.get("UK_ENV_DATA_DIR", os.path.join(base_dir, "data"))

    unified_path = os.environ.get(
        "UNIFIED_DATA_PATH",
        os.path.join(
            "data", "processed", "unified_datasets", "unified_dataset_with_air_quality.csv"
        ),
    )

    # FIXED: Use the new GeoJSON file with valid geometries
    lsoa_path = os.environ.get(
        "LSOA_PATH",
        "data/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-4299016806856585929.geojson",
    )

    wards_path = os.environ.get(
        "WARDS_PATH", "data/Wards_December_2024_Boundaries_UK_BFC_7247148252775165514.geojson"
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
        print(f"Attempting to load LSOA boundaries from: {lsoa_path}")
        lsoa_gdf = gpd.read_file(lsoa_path)
        # Check if it loaded as a GeoDataFrame
        if not isinstance(lsoa_gdf, gpd.GeoDataFrame):
            print(f"Warning: Loaded LSOA file {lsoa_path} is not a GeoDataFrame.")
            lsoa_gdf = None  # Treat as failed load
        else:
            print(f"Loaded LSOA boundaries with {len(lsoa_gdf)} areas.")
            print(f"LSOA GeoDataFrame Info:")
            print(f"  Null geometries: {lsoa_gdf.geometry.isna().sum()}")
            print(f"  Valid geometries: {lsoa_gdf.geometry.is_valid.sum()}")
            print(f"  CRS: {lsoa_gdf.crs}")

            # Fix any invalid geometries
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
            print(f"Loaded Ward boundaries with {len(wards_gdf)} areas.")
            wards_gdf = fix_geometries(wards_gdf)
    except Exception as e:
        print(f"Could not load Ward boundaries: {str(e)}")
        wards_gdf = None

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
        # FIXED: Use the known LSOA code column from the new GeoJSON file
        lsoa_code_col_in_gdf = "LSOA21CD"
        print(f"Using '{lsoa_code_col_in_gdf}' as LSOA code column in GeoJSON")

        # Check if the LSOA code column exists in unified_df
        if lsoa_code_col not in unified_df.columns:
            print(f"Error: LSOA code column '{lsoa_code_col}' not found in the unified dataset.")
            print(f"Available columns: {unified_df.columns.tolist()}")
            return None, None

        # Ensure the CRS is appropriate for UK spatial analysis (British National Grid)
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


# Rest of the code remains the same as in the original script
# ...


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

    # Continue with the rest of the analysis...
    # (The rest of the main function remains the same)


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
