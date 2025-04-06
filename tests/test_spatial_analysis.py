"""
Unit tests for spatial analysis module.

This file contains tests for the spatial analysis functions,
focusing on spatial weights creation, Moran's I calculation,
Getis-Ord Gi* calculation, and spatial regression.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Polygon

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the module to test
try:
    from spatial_analysis import (
        fix_geometries,
        load_geo_data,
        create_spatial_weights,
        calculate_morans_i,
        calculate_local_morans,
        calculate_getis_ord,
        aggregate_to_lad,
        run_spatial_regression
    )
    SPATIAL_MODULES_AVAILABLE = True
except ImportError:
    SPATIAL_MODULES_AVAILABLE = False


@unittest.skipIf(not SPATIAL_MODULES_AVAILABLE, "Spatial analysis modules not available")
class TestSpatialAnalysis(unittest.TestCase):
    """Test class for spatial analysis functions."""

    def setUp(self):
        """Set up test data."""
        # Create a sample GeoDataFrame for testing
        # Create a 3x3 grid of square polygons
        geometries = []
        for i in range(3):
            for j in range(3):
                geometries.append(Polygon([
                    (i, j), (i+1, j), (i+1, j+1), (i, j+1)
                ]))
        
        # Create GeoDataFrame
        self.test_gdf = gpd.GeoDataFrame({
            'lsoa_code': [f'E0100000{i}' for i in range(9)],
            'lsoa_name': [f'LSOA {i}' for i in range(9)],
            'lad_code': ['E06000001', 'E06000001', 'E06000001',
                         'E06000002', 'E06000002', 'E06000002',
                         'E06000003', 'E06000003', 'E06000003'],
            'lad_name': ['LAD 1', 'LAD 1', 'LAD 1',
                         'LAD 2', 'LAD 2', 'LAD 2',
                         'LAD 3', 'LAD 3', 'LAD 3'],
            'env_justice_index': np.random.uniform(0, 1, 9),
            'air_pollution_index': np.random.uniform(0, 1, 9),
            'imd_score_normalized': np.random.uniform(0, 100, 9),
            'NO2': np.random.uniform(10, 50, 9),
            'PM2.5': np.random.uniform(5, 25, 9),
            'geometry': geometries
        })
        
        # Set CRS
        self.test_gdf.set_crs("EPSG:27700", inplace=True)
        
        # Create output directory for tests
        os.makedirs('outputs/spatial_analysis', exist_ok=True)

    def test_fix_geometries(self):
        """Test fixing invalid geometries."""
        # Create a GeoDataFrame with an invalid geometry
        invalid_geom = Polygon([(0, 0), (1, 1), (0, 1), (1, 0)])  # Self-intersecting polygon
        valid_geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        
        test_gdf = gpd.GeoDataFrame({
            'id': [1, 2],
            'geometry': [invalid_geom, valid_geom]
        })
        
        # Fix geometries
        fixed_gdf = fix_geometries(test_gdf)
        
        # Check that all geometries are now valid
        self.assertTrue(fixed_gdf.geometry.is_valid.all())

    @patch('geopandas.read_file')
    def test_load_geo_data(self, mock_read_file):
        """Test loading geospatial data."""
        # Mock the read_file function to return our test GeoDataFrame
        mock_read_file.return_value = self.test_gdf
        
        # Load geo data
        loaded_gdf = load_geo_data('dummy_path.geojson')
        
        # Check that the function returns a GeoDataFrame
        self.assertIsInstance(loaded_gdf, gpd.GeoDataFrame)
        
        # Check that the GeoDataFrame has the expected columns
        self.assertIn('geometry', loaded_gdf.columns)
        self.assertIn('lsoa_code', loaded_gdf.columns)
        
        # Check that the CRS is set
        self.assertIsNotNone(loaded_gdf.crs)

    def test_create_spatial_weights(self):
        """Test creating spatial weights."""
        # Create spatial weights
        weights = create_spatial_weights(self.test_gdf, method='queen')
        
        # Check that weights are created
        self.assertIsNotNone(weights)
        
        # Check that weights have the expected number of areas
        self.assertEqual(weights.n, len(self.test_gdf))
        
        # Check that the weights are row-standardized
        self.assertEqual(weights.transform, 'R')
        
        # Check that the central cell has 8 neighbors (queen contiguity)
        # The central cell is at index 4 (middle of 3x3 grid)
        central_neighbors = weights.neighbors[4]
        self.assertEqual(len(central_neighbors), 8)
        
        # Check that corner cells have 3 neighbors
        corner_indices = [0, 2, 6, 8]  # Corners of 3x3 grid
        for idx in corner_indices:
            self.assertEqual(len(weights.neighbors[idx]), 3)

    def test_calculate_morans_i(self):
        """Test calculating Global Moran's I."""
        # Create spatial weights
        weights = create_spatial_weights(self.test_gdf, method='queen')
        
        # Calculate Moran's I
        moran_i, p_value = calculate_morans_i(self.test_gdf, 'env_justice_index', weights)
        
        # Check that Moran's I and p-value are returned
        self.assertIsNotNone(moran_i)
        self.assertIsNotNone(p_value)
        
        # Check that Moran's I is between -1 and 1
        self.assertGreaterEqual(moran_i, -1)
        self.assertLessEqual(moran_i, 1)
        
        # Check that p-value is between 0 and 1
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function
    def test_calculate_local_morans(self, mock_savefig):
        """Test calculating Local Moran's I."""
        # Create spatial weights
        weights = create_spatial_weights(self.test_gdf, method='queen')
        
        # Calculate Local Moran's I
        result_gdf = calculate_local_morans(self.test_gdf, 'env_justice_index', weights)
        
        # Check that a GeoDataFrame is returned
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        
        # Check that the result has the expected columns
        self.assertIn('env_justice_index_lisa', result_gdf.columns)
        self.assertIn('env_justice_index_lisa_p', result_gdf.columns)
        self.assertIn('env_justice_index_lisa_cluster', result_gdf.columns)
        
        # Check that the result has the same number of rows as the input
        self.assertEqual(len(result_gdf), len(self.test_gdf))

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function
    def test_calculate_getis_ord(self, mock_savefig):
        """Test calculating Getis-Ord Gi*."""
        # Create spatial weights
        weights = create_spatial_weights(self.test_gdf, method='queen')
        
        # Calculate Getis-Ord Gi*
        result_gdf = calculate_getis_ord(self.test_gdf, 'env_justice_index', weights)
        
        # Check that a GeoDataFrame is returned
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        
        # Check that the result has the expected columns
        self.assertIn('env_justice_index_gi', result_gdf.columns)
        self.assertIn('env_justice_index_gi_z', result_gdf.columns)
        self.assertIn('env_justice_index_gi_p', result_gdf.columns)
        self.assertIn('env_justice_index_gi_cluster', result_gdf.columns)
        
        # Check that the result has the same number of rows as the input
        self.assertEqual(len(result_gdf), len(self.test_gdf))

    def test_aggregate_to_lad(self):
        """Test aggregating LSOA data to LAD level."""
        # Aggregate to LAD level
        lad_df = aggregate_to_lad(self.test_gdf, lad_code_col='lad_code')
        
        # Check that a DataFrame is returned
        self.assertIsInstance(lad_df, gpd.GeoDataFrame)
        
        # Check that the result has the expected number of rows (3 LADs)
        self.assertEqual(len(lad_df), 3)
        
        # Check that the result has the expected columns
        for col in ['env_justice_index', 'air_pollution_index', 'imd_score_normalized', 'NO2', 'PM2.5']:
            self.assertIn(col, lad_df.columns)

    @patch('matplotlib.pyplot.savefig')  # Mock the savefig function
    @patch('matplotlib.pyplot.figure')   # Mock the figure function
    @patch('builtins.open')              # Mock the built-in open function for writing summary
    def test_run_spatial_regression(self, mock_open, mock_figure, mock_savefig):
        """Test running spatial regression."""
        # Create spatial weights for LAD level
        lad_gdf = aggregate_to_lad(self.test_gdf, lad_code_col='lad_code')
        weights = create_spatial_weights(lad_gdf, method='queen')
        
        # Define regression variables
        y_var = 'env_justice_index'
        x_vars = ['imd_score_normalized', 'NO2', 'PM2.5']
        
        # Mock the open function to avoid writing files
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        # Run spatial regression
        models = run_spatial_regression(lad_gdf, y_var, x_vars, weights)
        
        # Check model results based on data size
        if len(lad_gdf) >= 5:  # Check if enough data for regression (adjust threshold if needed)
            self.assertIsNotNone(models, "Expected models dictionary, got None")
            self.assertIn('ols', models, "OLS model missing from results")
            # Optionally check for 'ml_lag' if spreg is expected to work
            # if SPREG_AVAILABLE:
            #     self.assertIn('ml_lag', models, "ML Lag model missing from results")
        else:
            # Expect None if data is insufficient, as run_spatial_regression should handle this
            # Check that models are returned even with small N (spreg seems to handle it)
            self.assertIsNotNone(models, f"Expected models dictionary even with {len(lad_gdf)} data points, but got None.")


if __name__ == '__main__':
    unittest.main()
