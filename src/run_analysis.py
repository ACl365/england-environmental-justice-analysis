"""
Run Environmental Justice Analysis

This script runs the main environmental justice analysis and generates outputs.
"""

import os
import sys
import subprocess
import time

def main():
    """Run the main analysis script and handle any errors."""
    print("=" * 80)
    print("Environmental Justice and Health Inequalities Analysis")
    print("=" * 80)
    
    # Check if outputs directory exists, create if not
    if not os.path.exists('outputs'):
        print("Creating outputs directory...")
        os.makedirs('outputs/figures', exist_ok=True)
        os.makedirs('outputs/data', exist_ok=True)
    
    # Run the main analysis script
    print("\nRunning analysis...")
    start_time = time.time()
    
    try:
        import env_justice_analysis
        env_justice_analysis.main()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nAnalysis completed successfully in {duration:.2f} seconds.")
        print(f"Results saved to the 'outputs' directory.")
        
        # List generated outputs
        print("\nGenerated outputs:")
        
        if os.path.exists('outputs/figures'):
            print("\nFigures:")
            for file in os.listdir('outputs/figures'):
                print(f"  - outputs/figures/{file}")
        
        if os.path.exists('outputs/data'):
            print("\nData files:")
            for file in os.listdir('outputs/data'):
                print(f"  - outputs/data/{file}")
        
    except Exception as e:
        print(f"\nError running analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())