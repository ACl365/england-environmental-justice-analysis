"""
Unit tests for data processing and index calculation functions.

This file contains tests for data loading, preprocessing, and index calculation
functions to ensure data integrity and correct calculations.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import modules to test
try:
    from env_justice_analysis import load_data, explore_data
    ENV_JUSTICE_MODULE_AVAILABLE = True
except ImportError:
    ENV_JUSTICE_MODULE_AVAILABLE = False


class TestDataProcessing(unittest.TestCase):
    """Test class for data processing functions."""

    def setUp(self):
        """Set up test data."""
        # Create sample unified dataset
        self.unified_df = pd.DataFrame({
            'lsoa_code': ['E01000001', 'E01000002', 'E01000003', 'E01000004'],
            'lsoa_name': ['LSOA 1', 'LSOA 2', 'LSOA 3', 'LSOA 4'],
            'lad_code': ['E06000001', 'E06000001', 'E06000002', 'E06000002'],
            'lad_name': ['LAD 1', 'LAD 1', 'LAD 2', 'LAD 2'],
            'imd_score_normalized': [20.5, 35.2, 45.7, 15.3],
            'income_score_rate': [0.15, 0.22, 0.18, 0.12],
            'employment_score_rate': [0.12, 0.18, 0.15, 0.10],
            'health_deprivation_and_disability_score': [0.5, 0.8, 0.6, 0.4],
            'living_environment_score': [25.3, 35.6, 28.9, 20.1],
            'barriers_to_housing_and_services_score': [22.1, 18.5, 24.3, 19.8],
            'crime_score': [0.3, 0.5, 0.4, 0.2],
            'NO2': [35.2, 42.1, 28.5, 22.3],
            'O3': [45.6, 40.2, 50.1, 55.3],
            'PM10': [18.5, 22.3, 16.8, 15.2],
            'PM2.5': [10.2, 12.5, 9.8, 8.5],
            'NO2_normalized': [0.65, 0.85, 0.45, 0.35],
            'PM2.5_normalized': [0.55, 0.75, 0.45, 0.35],
            'PM10_normalized': [0.60, 0.80, 0.50, 0.40],
            'air_pollution_index': [0.62, 0.82, 0.48, 0.38],
            'env_justice_index': [0.58, 0.75, 0.52, 0.32]
        })
        
        # Create sample health dataset
        self.health_df = pd.DataFrame({
            'local_authority_code': ['E06000001', 'E06000002'],
            'local_authority_name': ['LAD 1', 'LAD 2'],
            'chronic_conditions_value': [105.2, 95.6],
            'chronic_conditions_lower_ci': [98.5, 90.2],
            'chronic_conditions_upper_ci': [112.3, 101.5],
            'chronic_conditions_standardised_ratio': [1.05, 0.96],
            'chronic_conditions_name': ['Chronic Conditions', 'Chronic Conditions'],
            'chronic_conditions_description': ['Description 1', 'Description 2'],
            'asthma_diabetes_epilepsy_value': [110.5, 92.3],
            'asthma_diabetes_epilepsy_lower_ci': [105.2, 88.1],
            'asthma_diabetes_epilepsy_upper_ci': [115.8, 96.5],
            'asthma_diabetes_epilepsy_standardised_ratio': [1.10, 0.92],
            'asthma_diabetes_epilepsy_name': ['ADE', 'ADE'],
            'asthma_diabetes_epilepsy_description': ['Description 1', 'Description 2'],
            'lrti_children_value': [108.3, 94.5],
            'lrti_children_lower_ci': [102.1, 90.2],
            'lrti_children_upper_ci': [114.5, 98.8],
            'lrti_children_standardised_ratio': [1.08, 0.94],
            'lrti_children_name': ['LRTI Children', 'LRTI Children'],
            'lrti_children_description': ['Description 1', 'Description 2'],
            'acute_conditions_value': [103.5, 97.2],
            'acute_conditions_lower_ci': [98.2, 92.5],
            'acute_conditions_upper_ci': [108.8, 101.9],
            'acute_conditions_standardised_ratio': [1.03, 0.97],
            'acute_conditions_name': ['Acute Conditions', 'Acute Conditions'],
            'acute_conditions_description': ['Description 1', 'Description 2'],
            'chronic_conditions_normalized': [0.45, 0.65],
            'asthma_diabetes_epilepsy_normalized': [0.40, 0.70],
            'lrti_children_normalized': [0.42, 0.68],
            'acute_conditions_normalized': [0.48, 0.62],
            'respiratory_health_index': [0.43, 0.67],
            'overall_health_index': [0.45, 0.65]
        })
        
        # Create empty wards dataset
        self.wards_df = pd.DataFrame()

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test data loading function."""
        if not ENV_JUSTICE_MODULE_AVAILABLE:
            self.skipTest("env_justice_analysis module not available")
        
        # Mock the read_csv function to return our test dataframes
        mock_read_csv.side_effect = [self.unified_df, self.health_df]
        
        # Call the load_data function
        unified_df, health_df, wards_df = load_data()
        
        # Check that the function returns the expected dataframes
        pd.testing.assert_frame_equal(unified_df, self.unified_df)
        pd.testing.assert_frame_equal(health_df, self.health_df)
        
        # Check that the wards_df is an empty DataFrame
        self.assertTrue(wards_df.empty)
        
        # Check that read_csv was called with the expected paths
        expected_calls = [
            unittest.mock.call("data/processed/unified_datasets/unified_dataset_with_air_quality.csv"),
            unittest.mock.call("data/raw/health/health_indicators_by_lad.csv")
        ]
        mock_read_csv.assert_has_calls(expected_calls)

    def test_air_pollution_index_calculation(self):
        """Test air pollution index calculation."""
        # Calculate air pollution index manually
        calculated_indices = []
        for i in range(len(self.unified_df)):
            no2 = self.unified_df.loc[i, 'NO2_normalized']
            pm25 = self.unified_df.loc[i, 'PM2.5_normalized']
            pm10 = self.unified_df.loc[i, 'PM10_normalized']
            o3 = self.unified_df.loc[i, 'O3']
            
            # Normalize O3 (invert since higher O3 is associated with lower NO2)
            o3_max = self.unified_df['O3'].max()
            o3_normalized = 1 - (o3 / o3_max)
            
            # Calculate air pollution index using the formula from DATA_DICTIONARY.md
            api = 0.4 * no2 + 0.3 * pm25 + 0.2 * pm10 + 0.1 * o3_normalized
            calculated_indices.append(round(api, 2))
        
        # Compare with the values in the dataframe
        for i in range(len(self.unified_df)):
            # Allow for small differences due to rounding
            self.assertAlmostEqual(
                calculated_indices[i],
                self.unified_df.loc[i, 'air_pollution_index'],
                places=1
            )

    def test_env_justice_index_calculation(self):
        """Test environmental justice index calculation."""
        # Calculate environmental justice index manually
        calculated_indices = []
        for i in range(len(self.unified_df)):
            api = self.unified_df.loc[i, 'air_pollution_index']
            imd = self.unified_df.loc[i, 'imd_score_normalized']
            
            # Calculate environmental justice index using the formula from DATA_DICTIONARY.md
            eji = (api * imd) ** 0.5  # Geometric mean
            calculated_indices.append(round(eji, 2))
        
        # Compare with the values in the dataframe
        for i in range(len(self.unified_df)):
            # Allow for small differences due to rounding
            self.assertAlmostEqual(
                calculated_indices[i],
                self.unified_df.loc[i, 'env_justice_index'],
                places=1
            )

    def test_respiratory_health_index_calculation(self):
        """Test respiratory health index calculation."""
        # Calculate respiratory health index manually
        calculated_indices = []
        for i in range(len(self.health_df)):
            chronic = self.health_df.loc[i, 'chronic_conditions_normalized']
            asthma = self.health_df.loc[i, 'asthma_diabetes_epilepsy_normalized']
            lrti = self.health_df.loc[i, 'lrti_children_normalized']
            acute = self.health_df.loc[i, 'acute_conditions_normalized']
            
            # Calculate respiratory health index using the formula from DATA_DICTIONARY.md
            rhi = 0.35 * chronic + 0.35 * asthma + 0.15 * lrti + 0.15 * acute
            calculated_indices.append(round(rhi, 2))
        
        # Compare with the values in the dataframe
        for i in range(len(self.health_df)):
            # Allow for small differences due to rounding
            self.assertAlmostEqual(
                calculated_indices[i],
                self.health_df.loc[i, 'respiratory_health_index'],
                places=1
            )

    @patch('matplotlib.pyplot.figure')  # Mock the figure function
    @patch('seaborn.histplot')          # Mock the histplot function
    @patch('seaborn.scatterplot')       # Mock the scatterplot function
    @patch('seaborn.regplot')           # Mock the regplot function
    def test_explore_data(self, mock_regplot, mock_scatterplot, mock_histplot, mock_figure):
        """Test data exploration function."""
        if not ENV_JUSTICE_MODULE_AVAILABLE:
            self.skipTest("env_justice_analysis module not available")
        
        # Call the explore_data function
        explore_data(self.unified_df, self.health_df, self.wards_df)
        
        # Check that the figure function was called
        mock_figure.assert_called()
        
        # Check that the histplot function was called
        mock_histplot.assert_called()
        
        # Check that the scatterplot function was called
        mock_scatterplot.assert_called()
        
        # Check that the regplot function was called
        mock_regplot.assert_called()


class TestDataIntegrity(unittest.TestCase):
    """Test class for data integrity checks."""

    def setUp(self):
        """Set up test data paths."""
        self.unified_data_path = "data/processed/unified_datasets/unified_dataset_with_air_quality.csv"
        self.health_data_path = "data/raw/health/health_indicators_by_lad.csv"

    def test_unified_data_exists(self):
        """Test that the unified dataset file exists."""
        self.assertTrue(
            os.path.exists(self.unified_data_path),
            f"Unified dataset file not found at {self.unified_data_path}"
        )

    def test_health_data_exists(self):
        """Test that the health dataset file exists."""
        self.assertTrue(
            os.path.exists(self.health_data_path),
            f"Health dataset file not found at {self.health_data_path}"
        )

    @unittest.skipIf(not os.path.exists("data/processed/unified_datasets/unified_dataset_with_air_quality.csv"),
                    "Unified dataset file not found")
    def test_unified_data_structure(self):
        """Test the structure of the unified dataset."""
        try:
            df = pd.read_csv(self.unified_data_path)
            
            # Check that the required columns are present
            required_columns = [
                'lsoa_code', 'lsoa_name', 'lad_code', 'lad_name',
                'imd_score_normalized', 'NO2', 'O3', 'PM10', 'PM2.5',
                'NO2_normalized', 'PM2.5_normalized', 'PM10_normalized',
                'air_pollution_index', 'env_justice_index'
            ]
            
            for col in required_columns:
                self.assertIn(col, df.columns, f"Column {col} not found in unified dataset")
            
            # Check that there are no duplicate LSOA codes
            self.assertEqual(
                len(df['lsoa_code'].unique()),
                len(df),
                "Duplicate LSOA codes found in unified dataset"
            )
            
            # Check that the index columns are within expected ranges
            self.assertTrue(
                (df['air_pollution_index'] >= 0).all() and (df['air_pollution_index'] <= 1).all(),
                "Air pollution index values outside expected range [0, 1]"
            )
            
            self.assertTrue(
                (df['env_justice_index'] >= 0).all() and (df['env_justice_index'] <= 1).all(),
                "Environmental justice index values outside expected range [0, 1]"
            )
            
        except Exception as e:
            self.fail(f"Error reading unified dataset: {e}")

    @unittest.skipIf(not os.path.exists("data/raw/health/health_indicators_by_lad.csv"),
                    "Health dataset file not found")
    def test_health_data_structure(self):
        """Test the structure of the health dataset."""
        try:
            df = pd.read_csv(self.health_data_path)
            
            # Check that the required columns are present
            required_columns = [
                'local_authority_code', 'local_authority_name',
                'chronic_conditions_normalized', 'asthma_diabetes_epilepsy_normalized',
                'lrti_children_normalized', 'acute_conditions_normalized',
                'respiratory_health_index', 'overall_health_index'
            ]
            
            for col in required_columns:
                self.assertIn(col, df.columns, f"Column {col} not found in health dataset")
            
            # Check that there are no duplicate LAD codes
            self.assertEqual(
                len(df['local_authority_code'].unique()),
                len(df),
                "Duplicate LAD codes found in health dataset"
            )
            
            # Check that the health indices are within expected ranges
            self.assertTrue(
                (df['respiratory_health_index'] >= 0).all() and (df['respiratory_health_index'] <= 1).all(),
                "Respiratory health index values outside expected range [0, 1]"
            )
            
            self.assertTrue(
                (df['overall_health_index'] >= 0).all() and (df['overall_health_index'] <= 1).all(),
                "Overall health index values outside expected range [0, 1]"
            )
            
        except Exception as e:
            self.fail(f"Error reading health dataset: {e}")


if __name__ == '__main__':
    unittest.main()