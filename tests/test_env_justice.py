import sys
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import pandas as pd
from src import env_justice_analysis

# Sample data for testing
@pytest.fixture
def sample_unified_df():
    data = {'lsoa_code': ['E01000001', 'E01000002'],
            'lsoa_name': ['LSOA1', 'LSOA2'],
            'lad_name': ['LAD1', 'LAD2'],
            'imd_score_normalized': [0.5, 0.7],
            'NO2_normalized': [0.3, 0.4],
            'PM2.5_normalized': [0.2, 0.3],
            'PM10_normalized': [0.1, 0.2],
            'O3': [0.5, 0.6],
            'env_justice_index': [0.6, 0.8]}
    return pd.DataFrame(data)

def test_calculate_pollution_deprivation_correlation(sample_unified_df):
    env_justice_analysis.calculate_pollution_deprivation_correlation(sample_unified_df)

def test_plot_pollution_deprivation_scatter(sample_unified_df):
    env_justice_analysis.plot_pollution_deprivation_scatter(sample_unified_df)

def test_analyze_environmental_justice_index(sample_unified_df):
    env_justice_analysis.analyze_environmental_justice_index(sample_unified_df)

def test_identify_high_injustice_areas(sample_unified_df):
    env_justice_analysis.identify_high_injustice_areas(sample_unified_df)