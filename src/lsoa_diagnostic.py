"""
LSOA GeoJSON Diagnostic Script

This script performs detailed diagnostics on the LSOA GeoJSON file and the merging process
to identify issues preventing proper spatial weight creation.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import traceback


def diagnose_geojson(file_path):
    """Diagnose issues with a GeoJSON file."""
    print(f"\n{'='*80}\nDIAGNOSING GEOJSON FILE: {file_path}\n{'='*80}")

    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist at path: {file_path}")
        return None

    print(f"File exists and has size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")

    # 2. Try to load the file
    try:
        gdf = gpd.read_file(file_path)
        print(f"Successfully loaded file as GeoDataFrame with {len(gdf)} rows")
    except Exception as e:
        print(f"ERROR loading file: {e}")
        traceback.print_exc()
        return None

    # 3. Check if it's a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        print(f"ERROR: Loaded object is not a GeoDataFrame, but a {type(gdf)}")
        return None

    # 4. Check for geometry column
    if "geometry" not in gdf.columns:
        print("ERROR: No 'geometry' column found in GeoDataFrame")
        print(f"Available columns: {gdf.columns.tolist()}")
        return None

    # 5. Check geometry types
    geom_types = gdf.geometry.type.unique()
    print(f"Geometry types in file: {geom_types}")

    # 6. Check for null geometries
    null_geoms = gdf.geometry.isna().sum()
    print(f"Null geometries: {null_geoms} ({null_geoms/len(gdf)*100:.2f}%)")

    # 7. Check for valid geometries
    valid_geoms = gdf.geometry.is_valid.sum()
    print(f"Valid geometries: {valid_geoms} ({valid_geoms/len(gdf)*100:.2f}%)")

    # 8. Check for empty geometries
    empty_geoms = gdf.geometry.is_empty.sum()
    print(f"Empty geometries: {empty_geoms} ({empty_geoms/len(gdf)*100:.2f}%)")

    # 9. Check CRS
    print(f"CRS: {gdf.crs}")

    # 10. Identify potential LSOA code columns
    lsoa_code_cols = [
        col
        for col in gdf.columns
        if "LSOA" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
    ]
    print(f"Potential LSOA code columns: {lsoa_code_cols}")

    if not lsoa_code_cols:
        print("WARNING: Could not identify LSOA code column. Showing all columns:")
        for col in gdf.columns:
            print(f"  - {col}")
    else:
        # Check uniqueness of LSOA code
        for col in lsoa_code_cols:
            unique_count = gdf[col].nunique()
            print(f"Column '{col}' has {unique_count} unique values out of {len(gdf)} rows")
            print(f"Sample values: {gdf[col].head(3).tolist()}")

    return gdf


def diagnose_merge(lsoa_gdf, unified_df, lsoa_code_col_in_gdf, lsoa_code_col_in_df="lsoa_code"):
    """Diagnose issues with merging LSOA GeoDataFrame with unified data."""
    print(f"\n{'='*80}\nDIAGNOSING MERGE PROCESS\n{'='*80}")

    # 1. Check if both dataframes exist
    if lsoa_gdf is None:
        print("ERROR: LSOA GeoDataFrame is None")
        return None

    if unified_df is None:
        print("ERROR: Unified DataFrame is None")
        return None

    # 2. Check if LSOA code columns exist
    if lsoa_code_col_in_gdf not in lsoa_gdf.columns:
        print(f"ERROR: LSOA code column '{lsoa_code_col_in_gdf}' not found in LSOA GeoDataFrame")
        return None

    if lsoa_code_col_in_df not in unified_df.columns:
        print(f"ERROR: LSOA code column '{lsoa_code_col_in_df}' not found in unified DataFrame")
        return None

    # 3. Check data types of LSOA code columns
    lsoa_gdf_type = lsoa_gdf[lsoa_code_col_in_gdf].dtype
    unified_df_type = unified_df[lsoa_code_col_in_df].dtype

    print(f"LSOA GeoDataFrame code column type: {lsoa_gdf_type}")
    print(f"Unified DataFrame code column type: {unified_df_type}")

    if lsoa_gdf_type != unified_df_type:
        print(f"WARNING: Data type mismatch between LSOA code columns")
        print("Converting both to string for comparison...")
        lsoa_codes_gdf = lsoa_gdf[lsoa_code_col_in_gdf].astype(str)
        lsoa_codes_df = unified_df[lsoa_code_col_in_df].astype(str)
    else:
        lsoa_codes_gdf = lsoa_gdf[lsoa_code_col_in_gdf]
        lsoa_codes_df = unified_df[lsoa_code_col_in_df]

    # 4. Check for matching codes
    common_codes = set(lsoa_codes_gdf).intersection(set(lsoa_codes_df))
    print(
        f"Common LSOA codes: {len(common_codes)} out of {len(lsoa_gdf)} in GeoDataFrame and {len(unified_df)} in DataFrame"
    )

    if len(common_codes) == 0:
        print("ERROR: No matching LSOA codes between datasets!")
        print(f"Sample LSOA codes from GeoDataFrame: {lsoa_codes_gdf.head(3).tolist()}")
        print(f"Sample LSOA codes from DataFrame: {lsoa_codes_df.head(3).tolist()}")
        return None

    # 5. Try the merge
    try:
        print("\nAttempting merge...")
        merged_gdf = lsoa_gdf.merge(
            unified_df, left_on=lsoa_code_col_in_gdf, right_on=lsoa_code_col_in_df, how="inner"
        )

        print(f"Merge successful: {len(merged_gdf)} rows")

        # Check if result is a GeoDataFrame
        if not isinstance(merged_gdf, gpd.GeoDataFrame):
            print(f"WARNING: Merge result is not a GeoDataFrame, but a {type(merged_gdf)}")
            if "geometry" in merged_gdf.columns:
                print("Attempting to convert to GeoDataFrame...")
                merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry="geometry", crs=lsoa_gdf.crs)

        # Check geometry column
        if "geometry" not in merged_gdf.columns:
            print("ERROR: Geometry column lost during merge")
            return None

        # Check for null geometries after merge
        null_geoms = merged_gdf.geometry.isna().sum()
        print(f"Null geometries after merge: {null_geoms} ({null_geoms/len(merged_gdf)*100:.2f}%)")

        # Check for valid geometries after merge
        valid_geoms = merged_gdf.geometry.is_valid.sum()
        print(
            f"Valid geometries after merge: {valid_geoms} ({valid_geoms/len(merged_gdf)*100:.2f}%)"
        )

        return merged_gdf

    except Exception as e:
        print(f"ERROR during merge: {e}")
        traceback.print_exc()
        return None


def diagnose_weights_creation(gdf):
    """Diagnose issues with creating spatial weights."""
    print(f"\n{'='*80}\nDIAGNOSING WEIGHTS CREATION\n{'='*80}")

    if gdf is None:
        print("ERROR: GeoDataFrame is None")
        return

    # 1. Check if it's a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        print(f"ERROR: Input is not a GeoDataFrame, but a {type(gdf)}")
        return

    # 2. Check for geometry column
    if "geometry" not in gdf.columns:
        print("ERROR: No 'geometry' column found in GeoDataFrame")
        return

    # 3. Check for null geometries
    null_geoms = gdf.geometry.isna().sum()
    print(f"Null geometries: {null_geoms} ({null_geoms/len(gdf)*100:.2f}%)")

    if null_geoms > 0:
        print("WARNING: Removing null geometries for weights creation test")
        gdf = gdf[gdf.geometry.notna()]

    # 4. Check for valid geometries
    valid_geoms = gdf.geometry.is_valid.sum()
    print(f"Valid geometries: {valid_geoms} ({valid_geoms/len(gdf)*100:.2f}%)")

    if valid_geoms < len(gdf):
        print("WARNING: Removing invalid geometries for weights creation test")
        gdf = gdf[gdf.geometry.is_valid]

    # 5. Check for empty geometries
    empty_geoms = gdf.geometry.is_empty.sum()
    print(f"Empty geometries: {empty_geoms} ({empty_geoms/len(gdf)*100:.2f}%)")

    if empty_geoms > 0:
        print("WARNING: Removing empty geometries for weights creation test")
        gdf = gdf[~gdf.geometry.is_empty]

    # 6. Check CRS
    print(f"CRS: {gdf.crs}")

    # 7. Try to create weights
    try:
        from libpysal.weights import Queen

        print("\nAttempting to create Queen weights...")
        w = Queen.from_dataframe(gdf)
        print(
            f"SUCCESS: Created Queen weights with {w.n} areas and average of {w.mean_neighbors:.2f} neighbors per area"
        )

        # Check for islands
        if w.islands:
            print(f"WARNING: Found {len(w.islands)} islands (areas with no neighbors)")

        return w
    except Exception as e:
        print(f"ERROR creating Queen weights: {e}")
        traceback.print_exc()

        try:
            from libpysal.weights import KNN

            print("\nAttempting to create KNN weights as fallback...")
            w = KNN.from_dataframe(gdf, k=5)
            print(f"SUCCESS: Created KNN weights with {w.n} areas and k=5 neighbors per area")
            return w
        except Exception as e:
            print(f"ERROR creating KNN weights: {e}")
            traceback.print_exc()
            return None


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


def main():
    """Main function to run diagnostics."""
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    lsoa_path = os.path.join(base_dir, "LSOA_DEC_2021_EW_NC_v3_-7589743170352813307.geojson")
    unified_path = os.path.join(base_dir, "unified_dataset_with_air_quality.csv")

    print(f"Base directory: {base_dir}")
    print(f"LSOA GeoJSON path: {lsoa_path}")
    print(f"Unified dataset path: {unified_path}")

    # Diagnose LSOA GeoJSON
    lsoa_gdf = diagnose_geojson(lsoa_path)

    if lsoa_gdf is None:
        print("\nCannot proceed with merge diagnosis due to LSOA GeoJSON issues.")
        return

    # Fix geometries if needed
    if (~lsoa_gdf.geometry.is_valid).sum() > 0:
        print("\nAttempting to fix invalid geometries...")
        lsoa_gdf = fix_geometries(lsoa_gdf)

    # Load unified dataset
    try:
        unified_df = pd.read_csv(unified_path)
        print(f"\nLoaded unified dataset with {len(unified_df)} rows")
    except Exception as e:
        print(f"\nERROR loading unified dataset: {e}")
        return

    # Identify LSOA code column in GeoJSON
    lsoa_code_cols = [
        col
        for col in lsoa_gdf.columns
        if "LSOA" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
    ]
    if not lsoa_code_cols:
        print("\nCould not identify LSOA code column in GeoJSON. Please specify manually.")
        return

    lsoa_code_col_in_gdf = lsoa_code_cols[0]

    # Diagnose merge
    merged_gdf = diagnose_merge(lsoa_gdf, unified_df, lsoa_code_col_in_gdf)

    if merged_gdf is None:
        print("\nCannot proceed with weights diagnosis due to merge issues.")
        return

    # Diagnose weights creation
    diagnose_weights_creation(merged_gdf)

    print("\nDiagnostics complete. See above for issues and recommendations.")


if __name__ == "__main__":
    main()
