"""
Run Spatial Autocorrelation and Hotspot Analysis

This script runs the spatial autocorrelation and hotspot analysis for the Environmental Justice project.
"""

import os
import sys
import subprocess
import time

def main():
    """Run the spatial autocorrelation and hotspot analysis script and handle any errors."""
    print("=" * 80)
    print("Spatial Autocorrelation and Hotspot Analysis for Environmental Justice Project")
    print("=" * 80)
    
    # Check if outputs directory exists, create if not
    if not os.path.exists('outputs/spatial_hotspots'):
        print("Creating outputs directory...")
        os.makedirs('outputs/spatial_hotspots', exist_ok=True)
    
    # Run the spatial analysis script
    print("\nRunning spatial autocorrelation and hotspot analysis...")
    start_time = time.time()
    
    try:
        import spatial_hotspot_analysis
        spatial_hotspot_analysis.main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nSpatial analysis completed successfully in {duration:.2f} seconds.")
        print(f"Results saved to the 'outputs/spatial_hotspots' directory.")
        
        # List generated outputs
        print("\nGenerated outputs:")
        
        if os.path.exists('outputs/spatial_hotspots'):
            print("\nSpatial hotspot outputs:")
            for file in os.listdir('outputs/spatial_hotspots'):
                print(f"  - outputs/spatial_hotspots/{file}")
        
    except Exception as e:
        print(f"\nError running spatial analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())