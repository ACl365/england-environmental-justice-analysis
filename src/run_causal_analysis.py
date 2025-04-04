"""
Run Causal Inference Analysis

This script runs the causal inference analysis for policy impact assessment in the Environmental Justice project.
"""

import os
import sys
import subprocess
import time


def main():
    """Run the causal inference analysis script and handle any errors."""
    print("=" * 80)
    print("Causal Inference Analysis for Policy Impact Assessment")
    print("=" * 80)

    # Check if outputs directory exists, create if not
    if not os.path.exists("outputs/causal_inference"):
        print("Creating outputs directory...")
        os.makedirs("outputs/causal_inference", exist_ok=True)

    # Run the causal inference analysis script
    print("\nRunning causal inference analysis...")
    start_time = time.time()

    try:
        # Add project root to Python path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import causal_inference_analysis

        causal_inference_analysis.main()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nCausal inference analysis completed successfully in {duration:.2f} seconds.")
        print(f"Results saved to the 'outputs/causal_inference' directory.")

        # List generated outputs
        print("\nGenerated outputs:")

        if os.path.exists("outputs/causal_inference"):
            print("\nCausal inference outputs:")
            for file in os.listdir("outputs/causal_inference"):
                print(f"  - outputs/causal_inference/{file}")

    except Exception as e:
        print(f"\nError running causal inference analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
