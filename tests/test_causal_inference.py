"""
Unit tests for causal inference analysis module.

This file contains tests for the causal inference analysis functions,
focusing on propensity score matching, dose-response estimation,
and policy impact simulation.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the module to test
from causal_inference_analysis import (
    calculate_smd,
    calculate_variance_ratios,
    rosenbaum_bounds,
    propensity_score_matching,
    dose_response_function,
    policy_impact_simulation,
    quantify_impact
)


class TestCausalInference(unittest.TestCase):
    """Test class for causal inference analysis functions."""

    def setUp(self):
        """Set up test data."""
        # Create a sample dataset for testing
        np.random.seed(42)  # For reproducibility
        n_samples = 100
        
        # Create synthetic data
        self.test_data = pd.DataFrame({
            'lad_code': [f'LAD{i:03d}' for i in range(n_samples)],
            'lad_name': [f'District {i}' for i in range(n_samples)],
            'NO2': np.random.normal(30, 10, n_samples),
            'PM2.5': np.random.normal(15, 5, n_samples),
            'imd_score_normalized': np.random.uniform(0, 100, n_samples),
            'income_score_rate': np.random.uniform(0, 0.3, n_samples),
            'employment_score_rate': np.random.uniform(0, 0.3, n_samples),
            'health_deprivation_and_disability_score': np.random.uniform(0, 2, n_samples),
            'living_environment_score': np.random.uniform(0, 50, n_samples),
            'barriers_to_housing_and_services_score': np.random.uniform(0, 50, n_samples),
            'crime_score': np.random.uniform(0, 2, n_samples),
            'respiratory_health_index': np.random.uniform(0, 1, n_samples)
        })
        
        # Create binary treatment variable
        self.test_data['high_NO2'] = (self.test_data['NO2'] > self.test_data['NO2'].median()).astype(int)
        
        # Create population data
        self.population_data = pd.DataFrame({
            'lad_name': self.test_data['lad_name'].unique(),
            'population': np.random.randint(50000, 200000, len(self.test_data['lad_name'].unique()))
        })
        
        # Define covariates
        self.covariates = [
            'imd_score_normalized',
            'income_score_rate',
            'employment_score_rate'
        ]
        
        # Create output directory for tests
        os.makedirs('outputs/causal_inference', exist_ok=True)

    def test_calculate_smd(self):
        """Test calculation of standardized mean differences."""
        # Calculate SMD
        smd_results = calculate_smd(
            self.test_data, 
            'high_NO2', 
            self.covariates
        )
        
        # Check that we get a result for each covariate
        self.assertEqual(len(smd_results), len(self.covariates))
        
        # Check that SMDs are non-negative
        for covariate, smd in smd_results.items():
            self.assertGreaterEqual(smd, 0)
            
        # Check that SMDs are reasonable values
        for smd in smd_results.values():
            self.assertLess(smd, 10)  # SMDs should not be extremely large

    def test_calculate_variance_ratios(self):
        """Test calculation of variance ratios."""
        # Calculate variance ratios
        variance_ratios = calculate_variance_ratios(
            self.test_data, 
            'high_NO2', 
            self.covariates
        )
        
        # Check that we get a result for each covariate
        self.assertEqual(len(variance_ratios), len(self.covariates))
        
        # Check that variance ratios are non-negative
        for covariate, ratio in variance_ratios.items():
            self.assertGreaterEqual(ratio, 0)

    def test_rosenbaum_bounds(self):
        """Test Rosenbaum bounds sensitivity analysis."""
        # Create matched treated and control groups
        treated_indices = self.test_data[self.test_data['high_NO2'] == 1].index[:10]
        control_indices = self.test_data[self.test_data['high_NO2'] == 0].index[:10]
        
        matched_treated = self.test_data.loc[treated_indices]
        matched_control = self.test_data.loc[control_indices]
        
        # Define gamma range
        gamma_range = [1.0, 1.5, 2.0]
        
        # Calculate Rosenbaum bounds
        bounds_results = rosenbaum_bounds(
            matched_treated, 
            matched_control, 
            'respiratory_health_index', 
            gamma_range
        )
        
        # Check that we get a result for each gamma value
        self.assertEqual(len(bounds_results), len(gamma_range))
        
        # Check that p-values are between 0 and 1
        for gamma, p_val in bounds_results.items():
            self.assertGreaterEqual(p_val, 0)
            self.assertLessEqual(p_val, 1)

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function to avoid creating files
    # @patch('matplotlib.pyplot.figure') # Removed figure mock
    def test_propensity_score_matching(self, mock_savefig): # Removed mock_figure argument
        """Test propensity score matching."""
        # Run propensity score matching with a subset of data for speed
        test_subset = self.test_data.sample(30, random_state=42)
        
        # Mock the plot_covariate_distributions function to avoid creating plots
        with patch('causal_inference_analysis.plot_covariate_distributions'):
            with patch('causal_inference_analysis.plot_common_support'):
                # Run the function
                att, matched_data, smd_results = propensity_score_matching(
                    test_subset,
                    'high_NO2',
                    'respiratory_health_index',
                    self.covariates[:2]  # Use fewer covariates for speed
                )
        
        # Check that ATT is a float
        self.assertIsInstance(att, float)
        
        # Check that matched data has the expected columns
        self.assertIn('matched_group', matched_data.columns)
        self.assertIn('propensity_score', matched_data.columns)
        
        # Check that SMD results are returned for each covariate
        self.assertEqual(len(smd_results), len(self.covariates[:2]))

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function
    # @patch('matplotlib.pyplot.figure') # Removed figure mock
    @patch('plotly.graph_objects.Figure.write_html')  # Mock the write_html function
    def test_dose_response_function(self, mock_write_html, mock_savefig): # Removed mock_figure argument
        """Test dose-response function estimation."""
        # Run dose-response function with a subset of data for speed
        test_subset = self.test_data.sample(30, random_state=42)
        
        # Run the function
        pred_data, ci = dose_response_function(
            test_subset,
            'NO2',
            'respiratory_health_index',
            self.covariates[:2]  # Use fewer covariates for speed
        )
        
        # Check that prediction data is returned
        self.assertIsInstance(pred_data, pd.DataFrame)
        self.assertIn('NO2', pred_data.columns)
        self.assertIn('respiratory_health_index', pred_data.columns)
        
        # Check that confidence intervals are returned
        self.assertEqual(len(ci), 2)
        lower_ci, upper_ci = ci
        self.assertEqual(len(lower_ci), len(pred_data))
        self.assertEqual(len(upper_ci), len(pred_data))

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function
    # @patch('matplotlib.pyplot.figure') # Removed figure mock
    @patch('plotly.express.bar')         # Mock the plotly express bar function
    @patch('plotly.graph_objects.Figure.write_html')  # Mock the write_html function
    def test_policy_impact_simulation(self, mock_write_html, mock_bar, mock_savefig): # Removed mock_figure argument
        """Test policy impact simulation."""
        # Run policy impact simulation with a subset of data for speed
        test_subset = self.test_data.sample(30, random_state=42)
        
        # Define reduction scenarios
        reduction_scenarios = [10, 20]
        
        # Run the function
        results = policy_impact_simulation(
            test_subset,
            'NO2',
            'respiratory_health_index',
            self.covariates[:2],  # Use fewer covariates for speed
            reduction_scenarios
        )
        
        # Check that results are returned for each scenario
        self.assertEqual(len(results), len(reduction_scenarios))
        
        # Check that each result has the expected keys
        for result in results:
            self.assertIn('reduction_pct', result)
            self.assertIn('avg_improvement', result)
            self.assertIn('top_areas', result)
            
            # Check that top_areas is a DataFrame
            self.assertIsInstance(result['top_areas'], pd.DataFrame)
            
            # Check that avg_improvement is a float
            self.assertIsInstance(result['avg_improvement'], float)

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function
    # @patch('matplotlib.pyplot.figure') # Removed figure mock
    def test_quantify_impact(self, mock_savefig): # Removed mock_figure argument
        """Test impact quantification."""
        # Create mock policy results
        policy_results = [
            {
                'reduction_pct': 10,
                'avg_improvement': 0.02,
                'top_areas': self.test_data.head(10).copy()
            },
            {
                'reduction_pct': 20,
                'avg_improvement': 0.04,
                'top_areas': self.test_data.head(10).copy()
            }
        ]
        
        # Add improvement column to top_areas
        policy_results[1]['top_areas']['improvement'] = np.random.uniform(0.01, 0.1, 10)
        
        # Run the function
        impact_metrics = quantify_impact(
            policy_results,
            self.population_data
        )
        
        # Check that impact metrics are returned
        self.assertIsInstance(impact_metrics, dict)
        
        # Check that the expected keys are present
        self.assertIn('population_benefiting', impact_metrics)
        self.assertIn('qaly_gains', impact_metrics)
        self.assertIn('nhs_cost_reduction', impact_metrics)
        
        # Check that the values are of the expected types
        self.assertIsInstance(impact_metrics['population_benefiting'], (int, float, np.number))
        self.assertIsInstance(impact_metrics['qaly_gains'], (float, np.number))
        self.assertIsInstance(impact_metrics['nhs_cost_reduction'], (float, np.number))


if __name__ == '__main__':
    unittest.main()