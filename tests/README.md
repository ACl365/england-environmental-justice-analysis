# Testing

This directory contains tests for the UK Environmental Justice Analysis project.

## Overview

The testing framework uses Python's built-in `unittest` module. Tests are organized to verify the functionality of the various analysis modules in the `src/` directory.

## Running Tests

To run all tests, from the project root directory:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests/test_basic.py
```

## Test Structure

- `test_basic.py`: Contains basic tests to verify project setup and environment
- Additional test files should be named with the `test_*.py` pattern
- Each module in `src/` should have corresponding tests

## Writing Tests

When adding new functionality to the project, please also add appropriate tests. Follow these guidelines:

1. Create test files with the naming pattern `test_*.py`
2. Use descriptive test method names that explain what is being tested
3. Include docstrings for test classes and methods
4. Test both expected behavior and edge cases

## Code Coverage

To measure test coverage (requires the `coverage` package):

```bash
# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests

# Generate a coverage report
coverage report -m
```

This helps identify which parts of the codebase need additional testing.