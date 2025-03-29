"""
Multivariate Pollutant Interaction Analysis for Environmental Justice Project

This script implements advanced techniques to analyze interactions between pollutants:
- Interaction terms in regression models
- Principal component analysis for pollution indices
- Non-linear models to capture threshold effects
- 3D visualizations of interaction surfaces
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
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
os.makedirs('outputs/pollutant_interactions', exist_ok=True)

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

def correlation_analysis(merged_df):
    """
    Perform correlation analysis between pollutants and health outcomes.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
    """
    print("\nPerforming correlation analysis between pollutants and health outcomes...")
    
    # Select pollutant and health variables
    pollutant_vars = ['NO2', 'O3', 'PM10', 'PM2.5']
    health_vars = ['respiratory_health_index', 'chronic_conditions_normalized', 
                  'asthma_diabetes_epilepsy_normalized', 'lrti_children_normalized']
    
    # Create correlation matrix
    corr_vars = pollutant_vars + health_vars
    corr_data = merged_df[corr_vars].dropna()
    corr_matrix = corr_data.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title("Correlation Matrix: Pollutants and Health Outcomes", fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/pollutant_interactions/correlation_matrix.png', dpi=300)
    
    # Calculate and print correlations with p-values
    print("\nCorrelations between pollutants and health outcomes (with p-values):")
    for p_var in pollutant_vars:
        for h_var in health_vars:
            valid_data = merged_df[[p_var, h_var]].dropna()
            if len(valid_data) > 5:
                corr, p_value = pearsonr(valid_data[p_var], valid_data[h_var])
                print(f"{p_var} vs {h_var}: r = {corr:.3f}, p = {p_value:.6f}")
    
    # Calculate correlations between pollutants
    print("\nCorrelations between pollutants (with p-values):")
    for i, p1 in enumerate(pollutant_vars):
        for p2 in pollutant_vars[i+1:]:
            valid_data = merged_df[[p1, p2]].dropna()
            if len(valid_data) > 5:
                corr, p_value = pearsonr(valid_data[p1], valid_data[p2])
                print(f"{p1} vs {p2}: r = {corr:.3f}, p = {p_value:.6f}")
    
    return corr_matrix

def create_pollution_index_pca(merged_df):
    """
    Create a pollution index using Principal Component Analysis (PCA).
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
        
    Returns:
        pd.DataFrame: DataFrame with pollution index
    """
    print("\nCreating pollution index using PCA...")
    
    # Select pollutant variables
    pollutant_vars = ['NO2', 'O3', 'PM10', 'PM2.5']
    
    # Drop rows with missing values
    pollution_data = merged_df[pollutant_vars].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pollution_data)
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(len(pollutant_vars))],
        index=pollution_data.index
    )
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # Print explained variance
    print("\nExplained variance by principal components:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.2f}%")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.tight_layout()
    plt.savefig('outputs/pollutant_interactions/pca_explained_variance.png', dpi=300)
    
    # Plot PCA loadings
    plt.figure(figsize=(12, 10))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Create loadings DataFrame
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(len(pollutant_vars))],
        index=pollutant_vars
    )
    
    # Plot loadings
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0)
    plt.title('PCA Loadings: Contribution of Pollutants to Principal Components')
    plt.tight_layout()
    plt.savefig('outputs/pollutant_interactions/pca_loadings.png', dpi=300)
    
    # Create pollution index using first principal component
    merged_df_with_pca = merged_df.copy()
    merged_df_with_pca.loc[pollution_data.index, 'pollution_index_pca'] = pca_result[:, 0]
    
    # Normalize to 0-100 scale for interpretability
    min_val = merged_df_with_pca['pollution_index_pca'].min()
    max_val = merged_df_with_pca['pollution_index_pca'].max()
    merged_df_with_pca['pollution_index_pca_normalized'] = 100 * (merged_df_with_pca['pollution_index_pca'] - min_val) / (max_val - min_val)
    
    # Plot histogram of pollution index
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df_with_pca['pollution_index_pca_normalized'].dropna(), bins=30, kde=True)
    plt.title('Distribution of PCA-based Pollution Index')
    plt.xlabel('Pollution Index (0-100)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('outputs/pollutant_interactions/pollution_index_distribution.png', dpi=300)
    
    # Save the pollution index
    merged_df_with_pca.to_csv('outputs/pollutant_interactions/lad_with_pollution_index.csv', index=False)
    
    return merged_df_with_pca, loadings_df

def analyze_interaction_terms(merged_df):
    """
    Analyze interaction terms between pollutants in regression models.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
    """
    print("\nAnalyzing interaction terms between pollutants...")
    
    # Select variables for analysis
    pollutant_vars = ['NO2', 'O3', 'PM10', 'PM2.5']
    health_var = 'respiratory_health_index'
    
    # Drop rows with missing values
    analysis_data = merged_df[pollutant_vars + [health_var]].dropna()
    
    # Standardize pollutant variables
    scaler = StandardScaler()
    pollutants_scaled = scaler.fit_transform(analysis_data[pollutant_vars])
    pollutants_scaled_df = pd.DataFrame(pollutants_scaled, columns=pollutant_vars, index=analysis_data.index)
    
    # Create interaction terms
    interaction_pairs = []
    for i, p1 in enumerate(pollutant_vars):
        for p2 in pollutant_vars[i+1:]:
            interaction_name = f"{p1}_{p2}"
            pollutants_scaled_df[interaction_name] = pollutants_scaled_df[p1] * pollutants_scaled_df[p2]
            interaction_pairs.append((p1, p2, interaction_name))
    
    # Combine with health variable
    analysis_data_with_interactions = pd.concat([
        pollutants_scaled_df,
        analysis_data[health_var]
    ], axis=1)
    
    # Fit regression model with interaction terms
    X = sm.add_constant(pollutants_scaled_df)
    y = analysis_data[health_var]
    model = sm.OLS(y, X).fit()
    
    # Print model summary
    print("\nRegression Model with Interaction Terms:")
    print(model.summary().tables[1])
    
    # Save model results
    with open('outputs/pollutant_interactions/interaction_model_summary.txt', 'w') as f:
        f.write(model.summary().as_text())
    
    # Identify significant interactions
    significant_interactions = []
    for p1, p2, interaction_name in interaction_pairs:
        p_value = model.pvalues.get(interaction_name, 1.0)
        if p_value < 0.1:  # Using 0.1 as threshold for potential significance
            significant_interactions.append((p1, p2, interaction_name, model.params[interaction_name], p_value))
    
    # Print significant interactions
    print("\nPotentially significant interactions (p < 0.1):")
    for p1, p2, interaction_name, coef, p_value in significant_interactions:
        print(f"{p1} × {p2}: coefficient = {coef:.4f}, p-value = {p_value:.4f}")
    
    # Visualize significant interactions
    for p1, p2, interaction_name, coef, p_value in significant_interactions:
        visualize_interaction(analysis_data_with_interactions, p1, p2, health_var)
    
    # If no significant interactions, visualize the most interesting pair
    if not significant_interactions:
        print("\nNo significant interactions found. Visualizing NO2 × PM2.5 interaction as an example.")
        visualize_interaction(analysis_data_with_interactions, 'NO2', 'PM2.5', health_var)
    
    return model, significant_interactions

def visualize_interaction(data, var1, var2, response_var):
    """
    Create 3D visualization of interaction between two pollutants and health outcome.
    
    Args:
        data (pd.DataFrame): Data with pollutant and health variables
        var1 (str): First pollutant variable
        var2 (str): Second pollutant variable
        response_var (str): Health outcome variable
    """
    print(f"\nCreating 3D visualization for {var1} × {var2} interaction...")
    
    # Create meshgrid for 3D surface
    x_range = np.linspace(data[var1].min(), data[var1].max(), 100)
    y_range = np.linspace(data[var2].min(), data[var2].max(), 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Fit polynomial regression model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data[[var1, var2]])
    poly_reg = LinearRegression()
    poly_reg.fit(poly_features, data[response_var])
    
    # Predict for meshgrid
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    grid_poly = poly.transform(grid_points)
    Z = poly_reg.predict(grid_poly).reshape(X.shape)
    
    # Create 3D surface plot with matplotlib
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Plot actual data points
    scatter = ax.scatter(data[var1], data[var2], data[response_var], 
                         c=data[response_var], cmap='coolwarm', s=50, edgecolor='k')
    
    # Add labels and title
    ax.set_xlabel(var1, fontsize=14)
    ax.set_ylabel(var2, fontsize=14)
    ax.set_zlabel(response_var, fontsize=14)
    ax.set_title(f'Interaction Effect of {var1} and {var2} on {response_var}', fontsize=16)
    
    # Add colorbar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label=response_var)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'outputs/pollutant_interactions/interaction_3d_{var1}_{var2}.png', dpi=300)
    
    # Create interactive 3D plot with Plotly
    fig = go.Figure(data=[
        go.Surface(z=Z, x=x_range, y=y_range, colorscale='Viridis', opacity=0.8),
        go.Scatter3d(
            x=data[var1], y=data[var2], z=data[response_var],
            mode='markers',
            marker=dict(
                size=5,
                color=data[response_var],
                colorscale='Inferno',
                opacity=0.8
            )
        )
    ])
    
    fig.update_layout(
        title=f'Interaction Effect of {var1} and {var2} on {response_var}',
        scene=dict(
            xaxis_title=var1,
            yaxis_title=var2,
            zaxis_title=response_var
        ),
        width=900,
        height=700
    )
    
    # Save as HTML
    fig.write_html(f'outputs/pollutant_interactions/interactive_3d_{var1}_{var2}.html')

def threshold_effect_analysis(merged_df):
    """
    Analyze threshold effects in the relationship between pollutants and health outcomes.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset with health indicators
    """
    print("\nAnalyzing threshold effects in pollutant-health relationships...")
    
    # Select variables for analysis
    pollutant_vars = ['NO2', 'O3', 'PM10', 'PM2.5']
    health_var = 'respiratory_health_index'
    
    # Create subplots for threshold analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Analyze each pollutant
    for i, pollutant in enumerate(pollutant_vars):
        # Drop rows with missing values
        valid_data = merged_df[[pollutant, health_var]].dropna()
        
        # Sort by pollutant level
        sorted_data = valid_data.sort_values(by=pollutant)
        
        # Calculate rolling average of health outcome
        window_size = max(5, len(sorted_data) // 20)  # Adaptive window size
        sorted_data['rolling_health'] = sorted_data[health_var].rolling(window=window_size, center=True).mean()
        
        # Plot data and rolling average
        ax = axes[i]
        ax.scatter(sorted_data[pollutant], sorted_data[health_var], alpha=0.6, label='Data points')
        ax.plot(sorted_data[pollutant], sorted_data['rolling_health'], 'r-', linewidth=2, label=f'Rolling avg (window={window_size})')
        
        # Fit piecewise linear regression (simple approach)
        # Try different potential threshold points
        thresholds = np.percentile(sorted_data[pollutant], np.arange(20, 81, 5))
        best_threshold = None
        best_r2 = -np.inf
        
        for threshold in thresholds:
            # Create indicator for above threshold
            sorted_data['above_threshold'] = (sorted_data[pollutant] > threshold).astype(int)
            sorted_data['pollutant_above'] = sorted_data['above_threshold'] * (sorted_data[pollutant] - threshold)
            
            # Fit model
            X = sm.add_constant(sorted_data[[pollutant, 'pollutant_above']])
            model = sm.OLS(sorted_data[health_var], X).fit()
            
            # Check if better than previous
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_threshold = threshold
                best_model = model
        
        # Plot threshold effect if found
        if best_threshold is not None:
            # Create prediction data
            pred_x = np.linspace(sorted_data[pollutant].min(), sorted_data[pollutant].max(), 100)
            pred_data = pd.DataFrame({pollutant: pred_x})
            pred_data['above_threshold'] = (pred_data[pollutant] > best_threshold).astype(int)
            pred_data['pollutant_above'] = pred_data['above_threshold'] * (pred_data[pollutant] - best_threshold)
            
            # Predict
            pred_data_with_const = sm.add_constant(pred_data[[pollutant, 'pollutant_above']])
            pred_y = best_model.predict(pred_data_with_const)
            
            # Plot prediction
            ax.plot(pred_x, pred_y, 'g-', linewidth=2, label=f'Threshold model (t={best_threshold:.2f})')
            ax.axvline(x=best_threshold, color='k', linestyle='--', alpha=0.5, label='Threshold')
            
            # Print model results
            print(f"\nThreshold analysis for {pollutant}:")
            print(f"Best threshold: {best_threshold:.2f}")
            print(f"R-squared: {best_r2:.4f}")
            print(f"Coefficient before threshold: {best_model.params[pollutant]:.4f}")
            print(f"Additional effect after threshold: {best_model.params['pollutant_above']:.4f}")
            print(f"Total effect after threshold: {best_model.params[pollutant] + best_model.params['pollutant_above']:.4f}")
        
        # Add labels and title
        ax.set_xlabel(pollutant, fontsize=12)
        ax.set_ylabel(health_var, fontsize=12)
        ax.set_title(f'Threshold Effect Analysis: {pollutant} vs {health_var}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/pollutant_interactions/threshold_effects.png', dpi=300)
    
    # Create interactive threshold visualization with Plotly
    for pollutant in pollutant_vars:
        # Drop rows with missing values
        valid_data = merged_df[[pollutant, health_var, 'lad_name']].dropna()
        
        # Sort by pollutant level
        sorted_data = valid_data.sort_values(by=pollutant)
        
        # Calculate rolling average of health outcome
        window_size = max(5, len(sorted_data) // 20)  # Adaptive window size
        sorted_data['rolling_health'] = sorted_data[health_var].rolling(window=window_size, center=True).mean()
        
        # Find threshold (simplified approach)
        threshold = np.percentile(sorted_data[pollutant], 60)  # Example threshold at 60th percentile
        
        # Create interactive plot
        fig = px.scatter(
            sorted_data, x=pollutant, y=health_var, 
            hover_name='lad_name',
            title=f'Threshold Effect Analysis: {pollutant} vs {health_var}',
            labels={pollutant: pollutant, health_var: health_var},
            color=pollutant,
            color_continuous_scale='Viridis'
        )
        
        # Add rolling average
        fig.add_trace(
            go.Scatter(
                x=sorted_data[pollutant], 
                y=sorted_data['rolling_health'],
                mode='lines',
                name=f'Rolling avg (window={window_size})',
                line=dict(color='red', width=3)
            )
        )
        
        # Add threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                     annotation_text=f"Potential threshold: {threshold:.2f}")
        
        # Update layout
        fig.update_layout(
            width=900,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save as HTML
        fig.write_html(f'outputs/pollutant_interactions/interactive_threshold_{pollutant}.html')

def main():
    """Main function to execute the pollutant interaction analysis."""
    print("Starting Multivariate Pollutant Interaction Analysis...")
    
    # Load data
    merged_df = load_data()
    
    # Perform correlation analysis
    corr_matrix = correlation_analysis(merged_df)
    
    # Create pollution index using PCA
    merged_df_with_pca, loadings_df = create_pollution_index_pca(merged_df)
    
    # Analyze interaction terms
    model, significant_interactions = analyze_interaction_terms(merged_df)
    
    # Analyze threshold effects
    threshold_effect_analysis(merged_df)
    
    print("\nMultivariate pollutant interaction analysis complete. Results saved to the 'outputs/pollutant_interactions' directory.")

if __name__ == "__main__":
    main()