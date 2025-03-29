"""
Causal Inference Analysis for Environmental Justice Project

This script implements causal inference techniques to assess policy impacts:
- Propensity score matching to compare similar areas with different pollution levels
- Counterfactual analysis for potential policy interventions
- Causal effect estimation of pollution on health outcomes
- Visualization of potential policy impacts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directories if they don't exist
os.makedirs('outputs/causal_inference', exist_ok=True)

def load_data():
    """
    Load the merged dataset with health indicators.
    
    Returns:
        pd.DataFrame: Merged dataset with health indicators
    """
    # Try to load the processed data if it exists
    if os.path.exists('outputs/data/lad_health_pollution_merged.csv'):
        merged_df = pd.read_csv('outputs/data/lad_health_pollution_merged.csv')
        print(f"Loaded processed data with {len(merged_df)} LADs")
    else:
        # If processed data doesn't exist, load and merge raw data
        print("Processed data not found. Loading and merging raw data...")
        unified_df = pd.read_csv('unified_dataset_with_air_quality.csv')
        health_df = pd.read_csv('health_indicators_by_lad.csv')
        
        # Aggregate unified data to LAD level
        lad_aggregated = unified_df.groupby('lad_code').agg({
            'lad_name': 'first',
            'imd_score_normalized': 'mean',
            'NO2': 'mean',
            'O3': 'mean',
            'PM10': 'mean',
            'PM2.5': 'mean',
            'NO2_normalized': 'mean',
            'PM2.5_normalized': 'mean',
            'PM10_normalized': 'mean',
            'env_justice_index': 'mean'
        }).reset_index()
        
        # Merge with health data
        merged_df = pd.merge(
            lad_aggregated,
            health_df,
            left_on='lad_code',
            right_on='local_authority_code',
            how='inner'
        )
        
        print(f"Merged raw data with {len(merged_df)} LADs")
    
    return merged_df

def propensity_score_matching(merged_df, treatment_var, outcome_var, covariates):
    """
    Perform propensity score matching to estimate causal effects.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
        treatment_var (str): Name of the treatment variable (e.g., 'high_NO2')
        outcome_var (str): Name of the outcome variable (e.g., 'respiratory_health_index')
        covariates (list): List of covariate variables to control for
        
    Returns:
        tuple: (ATT, matched_data)
    """
    print(f"\nPerforming propensity score matching for {treatment_var} on {outcome_var}...")
    
    # Drop rows with missing values
    analysis_data = merged_df[[treatment_var, outcome_var] + covariates].dropna()
    
    # Fit propensity score model
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(analysis_data[covariates], analysis_data[treatment_var])
    
    # Calculate propensity scores
    propensity_scores = ps_model.predict_proba(analysis_data[covariates])[:, 1]
    analysis_data['propensity_score'] = propensity_scores
    
    # Plot propensity score distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=analysis_data, x='propensity_score', hue=treatment_var,
        bins=30, alpha=0.6, element='step', common_norm=False
    )
    plt.title(f'Propensity Score Distribution for {treatment_var}')
    plt.xlabel('Propensity Score')
    plt.ylabel('Count')
    plt.savefig(f'outputs/causal_inference/propensity_scores_{treatment_var}.png', dpi=300)
    
    # Perform matching
    treated = analysis_data[analysis_data[treatment_var] == 1]
    control = analysis_data[analysis_data[treatment_var] == 0]
    
    # Match each treated unit to the nearest control unit
    matched_pairs = []
    for _, treated_unit in treated.iterrows():
        ps_treated = treated_unit['propensity_score']
        
        # Find closest control unit by propensity score
        control.loc[:, 'ps_diff'] = abs(control['propensity_score'] - ps_treated)
        closest_control = control.loc[control['ps_diff'].idxmin()]
        
        # Add to matched pairs
        matched_pairs.append((treated_unit, closest_control))
    
    # Create matched dataset
    matched_treated = pd.DataFrame([pair[0] for pair in matched_pairs])
    matched_control = pd.DataFrame([pair[1] for pair in matched_pairs])
    
    # Calculate Average Treatment Effect on the Treated (ATT)
    att = matched_treated[outcome_var].mean() - matched_control[outcome_var].mean()
    
    # Perform t-test on matched data
    t_stat, p_val = ttest_ind(
        matched_treated[outcome_var],
        matched_control[outcome_var]
    )
    
    print(f"Average Treatment Effect on the Treated (ATT): {att:.4f}")
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    
    # Create matched dataset for visualization
    matched_data = pd.concat([matched_treated, matched_control])
    matched_data['matched_group'] = ['Treated'] * len(matched_treated) + ['Control'] * len(matched_control)
    
    # Plot outcome comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='matched_group', y=outcome_var, data=matched_data)
    plt.title(f'Comparison of {outcome_var} After Matching')
    plt.xlabel('Group')
    plt.ylabel(outcome_var)
    plt.savefig(f'outputs/causal_inference/matched_outcome_{treatment_var}.png', dpi=300)
    
    # Check covariate balance
    balance_fig, balance_axes = plt.subplots(len(covariates), 1, figsize=(12, 4 * len(covariates)))
    
    if len(covariates) == 1:
        balance_axes = [balance_axes]
    
    for i, covariate in enumerate(covariates):
        sns.boxplot(x='matched_group', y=covariate, data=matched_data, ax=balance_axes[i])
        balance_axes[i].set_title(f'Balance Check: {covariate}')
        balance_axes[i].set_xlabel('Group')
        balance_axes[i].set_ylabel(covariate)
    
    plt.tight_layout()
    plt.savefig(f'outputs/causal_inference/covariate_balance_{treatment_var}.png', dpi=300)
    
    # Calculate standardized mean differences for covariates
    smd_results = {}
    for covariate in covariates:
        mean_treated = matched_treated[covariate].mean()
        mean_control = matched_control[covariate].mean()
        pooled_std = np.sqrt((matched_treated[covariate].var() + matched_control[covariate].var()) / 2)
        
        if pooled_std > 0:
            smd = abs(mean_treated - mean_control) / pooled_std
        else:
            smd = 0
            
        smd_results[covariate] = smd
    
    print("\nStandardized Mean Differences after matching:")
    for covariate, smd in smd_results.items():
        print(f"{covariate}: {smd:.4f}")
    
    # Plot SMD
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(smd_results.keys()), y=list(smd_results.values()))
    plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    plt.title('Standardized Mean Differences After Matching')
    plt.xlabel('Covariate')
    plt.ylabel('Standardized Mean Difference')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'outputs/causal_inference/smd_{treatment_var}.png', dpi=300)
    
    return att, matched_data, smd_results

def dose_response_function(merged_df, treatment_var, outcome_var, covariates):
    """
    Estimate dose-response function using a flexible regression approach.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
        treatment_var (str): Name of the treatment variable (e.g., 'NO2')
        outcome_var (str): Name of the outcome variable (e.g., 'respiratory_health_index')
        covariates (list): List of covariate variables to control for
    """
    print(f"\nEstimating dose-response function for {treatment_var} on {outcome_var}...")
    
    # Drop rows with missing values
    analysis_data = merged_df[[treatment_var, outcome_var] + covariates].dropna()
    
    # Standardize covariates
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(analysis_data[covariates])
    covariates_scaled_df = pd.DataFrame(covariates_scaled, columns=covariates, index=analysis_data.index)
    
    # Combine with treatment and outcome
    analysis_data_scaled = pd.concat([
        analysis_data[[treatment_var, outcome_var]],
        covariates_scaled_df
    ], axis=1)
    
    # Fit flexible model (Gradient Boosting)
    X = analysis_data_scaled[[treatment_var] + covariates]
    y = analysis_data_scaled[outcome_var]
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Create grid of treatment values
    treatment_grid = np.linspace(
        analysis_data[treatment_var].min(),
        analysis_data[treatment_var].max(),
        100
    )
    
    # Create prediction dataset with mean values for covariates
    pred_data = pd.DataFrame({treatment_var: treatment_grid})
    for covariate in covariates:
        pred_data[covariate] = 0  # Use mean of standardized covariates (which is 0)
    
    # Predict outcomes
    pred_data[outcome_var] = model.predict(pred_data)
    
    # Plot dose-response curve
    plt.figure(figsize=(12, 8))
    
    # Plot predicted dose-response curve
    plt.plot(pred_data[treatment_var], pred_data[outcome_var], 'b-', linewidth=2, label='Estimated Dose-Response')
    
    # Add confidence intervals (using bootstrap)
    n_bootstrap = 100
    bootstrap_predictions = np.zeros((n_bootstrap, len(treatment_grid)))
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(len(analysis_data), len(analysis_data), replace=True)
        bootstrap_data = analysis_data_scaled.iloc[bootstrap_indices]
        
        # Fit model on bootstrap sample
        bootstrap_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=i)
        bootstrap_model.fit(bootstrap_data[[treatment_var] + covariates], bootstrap_data[outcome_var])
        
        # Predict on grid
        bootstrap_predictions[i, :] = bootstrap_model.predict(pred_data[[treatment_var] + covariates])
    
    # Calculate confidence intervals
    lower_ci = np.percentile(bootstrap_predictions, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_predictions, 97.5, axis=0)
    
    # Plot confidence intervals
    plt.fill_between(treatment_grid, lower_ci, upper_ci, alpha=0.2, color='b', label='95% CI')
    
    # Add scatter plot of actual data
    plt.scatter(analysis_data[treatment_var], analysis_data[outcome_var], alpha=0.3, color='gray', label='Observed Data')
    
    # Add labels and title
    plt.xlabel(treatment_var, fontsize=14)
    plt.ylabel(outcome_var, fontsize=14)
    plt.title(f'Dose-Response Function: Effect of {treatment_var} on {outcome_var}', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/causal_inference/dose_response_{treatment_var}.png', dpi=300)
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    # Add dose-response curve
    fig.add_trace(go.Scatter(
        x=pred_data[treatment_var],
        y=pred_data[outcome_var],
        mode='lines',
        name='Estimated Dose-Response',
        line=dict(color='blue', width=3)
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=np.concatenate([treatment_grid, treatment_grid[::-1]]),
        y=np.concatenate([upper_ci, lower_ci[::-1]]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add observed data
    fig.add_trace(go.Scatter(
        x=analysis_data[treatment_var],
        y=analysis_data[outcome_var],
        mode='markers',
        name='Observed Data',
        marker=dict(color='gray', size=8, opacity=0.5)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Dose-Response Function: Effect of {treatment_var} on {outcome_var}',
        xaxis_title=treatment_var,
        yaxis_title=outcome_var,
        legend=dict(x=0.01, y=0.99, bordercolor='Black', borderwidth=1),
        width=900,
        height=600
    )
    
    # Save as HTML
    fig.write_html(f'outputs/causal_inference/interactive_dose_response_{treatment_var}.html')
    
    return pred_data, (lower_ci, upper_ci)

def policy_impact_simulation(merged_df, treatment_var, outcome_var, covariates, reduction_scenarios):
    """
    Simulate the impact of pollution reduction policies on health outcomes.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
        treatment_var (str): Name of the treatment variable (e.g., 'NO2')
        outcome_var (str): Name of the outcome variable (e.g., 'respiratory_health_index')
        covariates (list): List of covariate variables to control for
        reduction_scenarios (list): List of reduction percentages to simulate
    """
    print(f"\nSimulating policy impacts for reducing {treatment_var}...")
    
    # Drop rows with missing values
    analysis_data = merged_df[[treatment_var, outcome_var, 'lad_name'] + covariates].dropna()
    
    # Fit causal model
    X = analysis_data[[treatment_var] + covariates]
    y = analysis_data[outcome_var]
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Baseline predictions
    baseline_predictions = model.predict(X)
    analysis_data['baseline_prediction'] = baseline_predictions
    
    # Simulate reduction scenarios
    results = []
    
    for reduction_pct in reduction_scenarios:
        # Create counterfactual dataset with reduced pollution
        counterfactual_data = analysis_data.copy()
        counterfactual_data[f'{treatment_var}_reduced'] = counterfactual_data[treatment_var] * (1 - reduction_pct/100)
        
        # Predict counterfactual outcomes
        X_cf = counterfactual_data[[f'{treatment_var}_reduced'] + covariates].rename(
            columns={f'{treatment_var}_reduced': treatment_var}
        )
        counterfactual_predictions = model.predict(X_cf)
        counterfactual_data['counterfactual_prediction'] = counterfactual_predictions
        
        # Calculate improvements
        counterfactual_data['improvement'] = counterfactual_data['counterfactual_prediction'] - counterfactual_data['baseline_prediction']
        
        # Calculate average improvement
        avg_improvement = counterfactual_data['improvement'].mean()
        
        # Identify areas with largest improvements
        top_areas = counterfactual_data.sort_values('improvement', ascending=False).head(10)
        
        results.append({
            'reduction_pct': reduction_pct,
            'avg_improvement': avg_improvement,
            'top_areas': top_areas
        })
        
        print(f"\nScenario: {reduction_pct}% reduction in {treatment_var}")
        print(f"Average improvement in {outcome_var}: {avg_improvement:.4f}")
        print("Top 10 areas with largest improvements:")
        print(top_areas[['lad_name', 'improvement']].to_string(index=False))
    
    # Plot average improvements by reduction scenario
    plt.figure(figsize=(10, 6))
    reduction_pcts = [r['reduction_pct'] for r in results]
    avg_improvements = [r['avg_improvement'] for r in results]
    
    plt.plot(reduction_pcts, avg_improvements, 'o-', linewidth=2)
    plt.xlabel(f'Reduction in {treatment_var} (%)', fontsize=14)
    plt.ylabel(f'Average Improvement in {outcome_var}', fontsize=14)
    plt.title(f'Simulated Policy Impact: Reducing {treatment_var}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/causal_inference/policy_impact_{treatment_var}.png', dpi=300)
    
    # Create interactive visualization of policy impacts across areas
    # Select a specific scenario (e.g., 20% reduction)
    scenario_idx = reduction_scenarios.index(20) if 20 in reduction_scenarios else 0
    scenario_data = results[scenario_idx]['top_areas']
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='lad_name', y='improvement', data=scenario_data)
    plt.xlabel('Local Authority District', fontsize=14)
    plt.ylabel(f'Improvement in {outcome_var}', fontsize=14)
    plt.title(f'Top 10 Areas Benefiting from {reduction_scenarios[scenario_idx]}% Reduction in {treatment_var}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'outputs/causal_inference/top_areas_{treatment_var}.png', dpi=300)
    
    # Create interactive visualization with Plotly
    fig = px.bar(
        scenario_data, x='lad_name', y='improvement',
        title=f'Top 10 Areas Benefiting from {reduction_scenarios[scenario_idx]}% Reduction in {treatment_var}',
        labels={'lad_name': 'Local Authority District', 'improvement': f'Improvement in {outcome_var}'},
        color='improvement',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        width=900,
        height=600
    )
    
    # Save as HTML
    fig.write_html(f'outputs/causal_inference/interactive_top_areas_{treatment_var}.html')
    
    return results

def main():
    """Main function to execute the causal inference analysis."""
    print("Starting Causal Inference Analysis for Policy Impact Assessment...")
    
    # Load data
    merged_df = load_data()
    
    # Create binary treatment variables for high pollution levels
    pollutant_vars = ['NO2', 'PM2.5']
    health_var = 'respiratory_health_index'
    
    for pollutant in pollutant_vars:
        # Create binary treatment variable (high pollution = above median)
        median_value = merged_df[pollutant].median()
        treatment_var = f'high_{pollutant}'
        merged_df[treatment_var] = (merged_df[pollutant] > median_value).astype(int)
        
        # Define covariates (factors that might affect both pollution and health)
        covariates = ['imd_score_normalized']
        
        # Perform propensity score matching
        att, matched_data, smd_results = propensity_score_matching(
            merged_df, treatment_var, health_var, covariates
        )
        
        # Estimate dose-response function
        pred_data, ci = dose_response_function(
            merged_df, pollutant, health_var, covariates
        )
        
        # Simulate policy impacts
        reduction_scenarios = [10, 20, 30, 40, 50]
        policy_results = policy_impact_simulation(
            merged_df, pollutant, health_var, covariates, reduction_scenarios
        )
    
    print("\nCausal inference analysis complete. Results saved to the 'outputs/causal_inference' directory.")

if __name__ == "__main__":
    main()