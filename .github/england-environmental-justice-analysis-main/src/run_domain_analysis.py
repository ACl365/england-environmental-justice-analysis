"""
Run Domain-Specific Deprivation Analysis

This script runs the domain-specific deprivation analysis for the Environmental Justice project.
"""

import os
import sys
import subprocess
import time


def main():
    """Run the domain-specific deprivation analysis script and handle any errors."""
    print("=" * 80)
    print("Domain-Specific Deprivation Analysis for Environmental Justice Project")
    print("=" * 80)

    # Check if outputs directory exists, create if not
    if not os.path.exists("outputs/domain_deprivation"):
        print("Creating outputs directory...")
        os.makedirs("outputs/domain_deprivation", exist_ok=True)

    # Run the domain analysis script
    print("\nRunning domain-specific deprivation analysis...")
    start_time = time.time()

    try:
        import domain_deprivation_analysis

        domain_deprivation_analysis.main()

        end_time = time.time()
        duration = end_time - start_time

        print(
            f"\nDomain-specific deprivation analysis completed successfully in {duration:.2f} seconds."
        )
        print(f"Results saved to the 'outputs/domain_deprivation' directory.")

        # List generated outputs
        print("\nGenerated outputs:")

        if os.path.exists("outputs/domain_deprivation"):
            print("\nDomain deprivation outputs:")
            for file in os.listdir("outputs/domain_deprivation"):
                print(f"  - outputs/domain_deprivation/{file}")

    except Exception as e:
        print(f"\nError running domain-specific deprivation analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
