"""
Run Pollutant Interaction Analysis

This script runs the multivariate pollutant interaction analysis for the Environmental Justice project.
"""

import os
import sys
import subprocess
import time


def main():
    """Run the pollutant interaction analysis script and handle any errors."""
    print("=" * 80)
    print("Multivariate Pollutant Interaction Analysis for Environmental Justice Project")
    print("=" * 80)

    # Check if outputs directory exists, create if not
    if not os.path.exists("outputs/pollutant_interactions"):
        print("Creating outputs directory...")
        os.makedirs("outputs/pollutant_interactions", exist_ok=True)

    # Run the pollutant analysis script
    print("\nRunning pollutant interaction analysis...")
    start_time = time.time()

    try:
        import pollutant_interaction_analysis

        pollutant_interaction_analysis.main()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nPollutant interaction analysis completed successfully in {duration:.2f} seconds.")
        print(f"Results saved to the 'outputs/pollutant_interactions' directory.")

        # List generated outputs
        print("\nGenerated outputs:")

        if os.path.exists("outputs/pollutant_interactions"):
            print("\nPollutant interaction outputs:")
            for file in os.listdir("outputs/pollutant_interactions"):
                print(f"  - outputs/pollutant_interactions/{file}")

    except Exception as e:
        print(f"\nError running pollutant interaction analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
