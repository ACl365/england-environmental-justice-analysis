"""
Run Advanced Cluster Analysis

This script runs the advanced cluster analysis for the Environmental Justice project.
"""

import os
import sys
import subprocess
import time


def main():
    """Run the advanced cluster analysis script and handle any errors."""
    print("=" * 80)
    print("Advanced Cluster Analysis for Environmental Justice Project")
    print("=" * 80)

    # Check if outputs directory exists, create if not
    if not os.path.exists("outputs/advanced_clustering"):
        print("Creating outputs directory...")
        os.makedirs("outputs/advanced_clustering", exist_ok=True)

    # Run the advanced analysis script
    print("\nRunning advanced cluster analysis...")
    start_time = time.time()

    try:
        import advanced_cluster_analysis

        advanced_cluster_analysis.main()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nAdvanced analysis completed successfully in {duration:.2f} seconds.")
        print(f"Results saved to the 'outputs/advanced_clustering' directory.")

        # List generated outputs
        print("\nGenerated outputs:")

        if os.path.exists("outputs/advanced_clustering"):
            print("\nAdvanced clustering outputs:")
            for file in os.listdir("outputs/advanced_clustering"):
                print(f"  - outputs/advanced_clustering/{file}")

    except Exception as e:
        print(f"\nError running advanced analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
