"""
Quick LSOA GeoJSON diagnostic script
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import traceback


def main():
    # Define path to LSOA GeoJSON
    lsoa_path = "LSOA_DEC_2021_EW_NC_v3_-7589743170352813307.geojson"

    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking LSOA file: {lsoa_path}")

    # Check if file exists
    if not os.path.exists(lsoa_path):
        print(f"ERROR: File does not exist at path: {lsoa_path}")
        return

    print(f"File exists and has size: {os.path.getsize(lsoa_path) / (1024*1024):.2f} MB")

    # Try to load the file
    try:
        print("Loading GeoJSON file...")
        gdf = gpd.read_file(lsoa_path)
        print(f"Successfully loaded file as GeoDataFrame with {len(gdf)} rows")

        # Check if it's a GeoDataFrame
        print(f"Is GeoDataFrame: {isinstance(gdf, gpd.GeoDataFrame)}")

        # Check for geometry column
        if "geometry" in gdf.columns:
            print("Geometry column exists")

            # Check geometry types
            geom_types = gdf.geometry.type.unique()
            print(f"Geometry types: {geom_types}")

            # Check for null geometries
            null_geoms = gdf.geometry.isna().sum()
            print(f"Null geometries: {null_geoms} ({null_geoms/len(gdf)*100:.2f}%)")

            # Check for valid geometries
            valid_geoms = gdf.geometry.is_valid.sum()
            print(f"Valid geometries: {valid_geoms} ({valid_geoms/len(gdf)*100:.2f}%)")

            # Check for empty geometries
            empty_geoms = gdf.geometry.is_empty.sum()
            print(f"Empty geometries: {empty_geoms} ({empty_geoms/len(gdf)*100:.2f}%)")
        else:
            print("ERROR: No 'geometry' column found in GeoDataFrame")
            print(f"Available columns: {gdf.columns.tolist()}")

        # Check CRS
        print(f"CRS: {gdf.crs}")

        # Identify potential LSOA code columns
        lsoa_code_cols = [
            col
            for col in gdf.columns
            if "LSOA" in col.upper() and ("CODE" in col.upper() or "CD" in col.upper())
        ]
        print(f"Potential LSOA code columns: {lsoa_code_cols}")

        # Show all columns if no LSOA code column found
        if not lsoa_code_cols:
            print("WARNING: Could not identify LSOA code column. Showing all columns:")
            for col in gdf.columns:
                print(f"  - {col}")

        # Try to load unified dataset
        unified_path = "unified_dataset_with_air_quality.csv"
        if os.path.exists(unified_path):
            print(f"\nChecking unified dataset: {unified_path}")
            unified_df = pd.read_csv(unified_path)
            print(f"Loaded unified dataset with {len(unified_df)} rows")

            # Check for LSOA code column
            if "lsoa_code" in unified_df.columns:
                print("'lsoa_code' column exists in unified dataset")

                # Check for matching codes if LSOA code column found in GeoJSON
                if lsoa_code_cols:
                    lsoa_code_col_in_gdf = lsoa_code_cols[0]
                    lsoa_codes_gdf = gdf[lsoa_code_col_in_gdf].astype(str)
                    lsoa_codes_df = unified_df["lsoa_code"].astype(str)

                    common_codes = set(lsoa_codes_gdf).intersection(set(lsoa_codes_df))
                    print(
                        f"Common LSOA codes: {len(common_codes)} out of {len(gdf)} in GeoDataFrame and {len(unified_df)} in DataFrame"
                    )

                    if len(common_codes) == 0:
                        print("ERROR: No matching LSOA codes between datasets!")
                        print(
                            f"Sample LSOA codes from GeoDataFrame: {lsoa_codes_gdf.head(3).tolist()}"
                        )
                        print(f"Sample LSOA codes from DataFrame: {lsoa_codes_df.head(3).tolist()}")
            else:
                print("ERROR: 'lsoa_code' column not found in unified dataset")
                print(f"Available columns: {unified_df.columns.tolist()}")
        else:
            print(f"\nUnified dataset not found at: {unified_path}")

    except Exception as e:
        print(f"ERROR loading or processing file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
