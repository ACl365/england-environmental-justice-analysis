#!/usr/bin/env python
"""
Comprehensive test runner for the UK Environmental Justice Analysis project.

This script runs all tests in the tests directory and generates a coverage report.
It can be run from the command line with various options to control test execution
and reporting.

Usage:
    python tests/run_tests.py [--verbose] [--coverage] [--html] [--xml]

Options:
    --verbose: Run tests in verbose mode
    --coverage: Generate a coverage report
    --html: Generate an HTML coverage report
    --xml: Generate an XML coverage report for CI tools
"""

import unittest
import sys
import os
import argparse

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tests(verbose=False, coverage=False, html=False, xml=False):
    """
    Run all tests in the tests directory.
    
    Args:
        verbose (bool): Run tests in verbose mode
        coverage (bool): Generate a coverage report
        html (bool): Generate an HTML coverage report
        xml (bool): Generate an XML coverage report
    
    Returns:
        int: Number of test failures
    """
    if coverage:
        try:
            import coverage
            cov = coverage.Coverage(
                source=['src'],
                omit=['*/__pycache__/*', '*/tests/*', '*/venv/*']
            )
            cov.start()
            print("Coverage analysis enabled")
        except ImportError:
            print("Warning: coverage package not installed. Running without coverage analysis.")
            coverage = False
    
    # Discover and run tests
    loader = unittest.TestLoader()
    tests_dir = os.path.abspath(os.path.dirname(__file__))
    suite = loader.discover(tests_dir)
    
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    if coverage:
        cov.stop()
        cov.save()
        
        print("\nCoverage Summary:")
        cov.report()
        
        if html:
            html_dir = os.path.join(os.path.dirname(tests_dir), 'coverage_html')
            os.makedirs(html_dir, exist_ok=True)
            print(f"\nGenerating HTML coverage report in {html_dir}")
            cov.html_report(directory=html_dir)
        
        if xml:
            xml_file = os.path.join(os.path.dirname(tests_dir), 'coverage.xml')
            print(f"\nGenerating XML coverage report: {xml_file}")
            cov.xml_report(outfile=xml_file)
    
    return len(result.failures) + len(result.errors)


def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run tests for the UK Environmental Justice Analysis project')
    parser.add_argument('--verbose', action='store_true', help='Run tests in verbose mode')
    parser.add_argument('--coverage', action='store_true', help='Generate a coverage report')
    parser.add_argument('--html', action='store_true', help='Generate an HTML coverage report')
    parser.add_argument('--xml', action='store_true', help='Generate an XML coverage report')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UK Environmental Justice Analysis - Test Runner")
    print("=" * 80)
    
    # Print test environment information
    print("\nTest Environment:")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Test directory: {os.path.abspath(os.path.dirname(__file__))}")
    
    # Run the tests
    print("\nRunning tests...")
    failures = run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        html=args.html,
        xml=args.xml
    )
    
    # Return non-zero exit code if there were failures
    return failures


if __name__ == '__main__':
    sys.exit(main())