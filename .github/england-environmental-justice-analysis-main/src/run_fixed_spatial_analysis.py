"""
Run the fixed spatial hotspot analysis script.

This script executes the fixed spatial_hotspot_analysis_fixed.py script
which uses the valid LSOA boundaries GeoJSON file.
"""

import os
import sys
import importlib.util
import traceback


def main():
    """Run the fixed spatial hotspot analysis."""
    print("Running fixed spatial hotspot analysis...")

    # Get the absolute path to the script
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "spatial_hotspot_analysis_fixed.py")
    )

    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return

    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location("spatial_hotspot_analysis_fixed", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Run the main function
        module.main()

        print("Fixed spatial analysis completed successfully.")
    except Exception as e:
        print(f"Error running fixed spatial analysis: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
