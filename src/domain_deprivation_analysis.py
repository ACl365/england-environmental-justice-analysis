"""
Domain-Specific Deprivation Analysis for Environmental Justice Project

This script analyzes the relationships between specific deprivation domains and pollution:
- Breaking down IMD into component domains
- Analyzing which aspects of deprivation are most strongly associated with pollution
- Creating domain-specific vulnerability indices
- Visualizing multidimensional deprivation profiles by area
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
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
os.makedirs('outputs/domain_deprivation', exist_ok=True)

def load_data():
    """
    Load the unified dataset with deprivation domains.
    
    Returns:
        pd.DataFrame: Unified dataset with deprivation domains
    """
    # Load the unified dataset
    unified_df = pd.read_csv('unified_dataset_with_air_quality.csv')
    print(f"Loaded unified dataset with {len(unified_df)} LSOAs")
    
    return unified_df

def identify_deprivation_domains(unified_df):
    """
    Identify and extract deprivation domain variables from the dataset.
    
    Args:
        unified_df (pd.DataFrame): Unified dataset with deprivation domains
        
    Returns:
        tuple: (domain_vars, domain_names)
    """
    print("\nIdentifying deprivation domain variables...")
    
    # Define the main deprivation domains based on IMD structure
    domain_mapping = {
        'income_score_rate': 'Income Deprivation',
        'employment_score_rate': 'Employment Deprivation',
        'education_skills_and_training_score': 'Education Deprivation',
        'health_deprivation_and_disability_score': 'Health Deprivation',
        'crime_score': 'Crime Deprivation',
        'barriers_to_housing_and_services_score': 'Housing & Services Barriers',
        'living_environment_score': 'Living Environment Deprivation'
    }
    
    # Check which domains are available in the dataset
    available_domains = [col for col in domain_mapping.keys() if col in unified_df.columns]
    
    if not available_domains:
        print("Warning: No standard deprivation domains found in the dataset.")
        # Try to identify potential domain variables based on column names
        potential_domains = [col for col in unified_df.columns if 'deprivation' in col.lower() or 'score' in col.lower()]
        print(f"Potential domain variables identified: {potential_domains}")
        available_domains = potential_domains
    
    # Create mapping of available domains
    domain_vars = available_domains
    domain_names = [domain_mapping.get(var, var) for var in domain_vars]
    
    print(f"Identified {len(domain_vars)} deprivation domains:")
    for var, name in zip(domain_vars, domain_names):
        print(f"- {name} ({var})")
    
    return domain_vars, domain_names

def analyze_domain_correlations(unified_df, domain_vars, domain_names):
    """
    Analyze correlations between deprivation domains and pollution variables.
    
    Args:
        unified_df (pd.DataFrame): Unified dataset with deprivation domains
        domain_vars (list): List of deprivation domain variables
        domain_names (list): List of deprivation domain names
    """
    print("\nAnalyzing correlations between deprivation domains and pollution...")
    
    # Define pollution variables
    pollution_vars = ['NO2', 'O3', 'PM10', 'PM2.5']
    
    # Create correlation matrix
    corr_vars = domain_vars + pollution_vars
    corr_data = unified_df[corr_vars].dropna()
    corr_matrix = corr_data.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Use domain names for better readability
    corr_matrix_labeled = corr_matrix.copy()
    corr_matrix_labeled = corr_matrix_labeled.rename(
        index=dict(zip(domain_vars, domain_names)),
        columns=dict(zip(domain_vars, domain_names))
    )
    
    sns.heatmap(corr_matrix_labeled, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title("Correlation Matrix: Deprivation Domains and Pollution", fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/domain_deprivation/domain_pollution_correlation.png', dpi=300)
    
    # Calculate and print correlations with p-values
    print("\nCorrelations between deprivation domains and pollution (with p-values):")
    
    # Create a DataFrame to store correlation results
    corr_results = []
    
    for d_var, d_name in zip(domain_vars, domain_names):
        for p_var in pollution_vars:
            valid_data = unified_df[[d_var, p_var]].dropna()
            if len(valid_data) > 5:
                corr, p_value = pearsonr(valid_data[d_var], valid_data[p_var])
                print(f"{d_name} vs {p_var}: r = {corr:.3f}, p = {p_value:.6f}")
                
                corr_results.append({
                    'Domain': d_name,
                    'Pollutant': p_var,
                    'Correlation': corr,
                    'P-value': p_value,
                    'Significant': p_value < 0.05
                })
    
    # Create DataFrame with correlation results
    corr_df = pd.DataFrame(corr_results)
    
    # Plot correlation strengths
    plt.figure(figsize=(14, 10))
    
    # Create a pivot table for the heatmap
    corr_pivot = corr_df.pivot(index='Domain', columns='Pollutant', values='Correlation')
    
    # Plot heatmap
    sns.heatmap(corr_pivot, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, center=0, linewidths=.5)
    plt.title("Correlation Strength: Deprivation Domains vs. Pollution", fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/domain_deprivation/domain_correlation_heatmap.png', dpi=300)
    
    # Create bar plot of correlation strengths for each pollutant
    plt.figure(figsize=(16, 12))
    
    # Create subplots for each pollutant
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, pollutant in enumerate(pollution_vars):
        pollutant_corrs = corr_df[corr_df['Pollutant'] == pollutant].sort_values('Correlation')
        
        # Create bar plot
        ax = axes[i]
        bars = ax.barh(pollutant_corrs['Domain'], pollutant_corrs['Correlation'])
        
        # Color bars by significance
        for j, bar in enumerate(bars):
            if pollutant_corrs.iloc[j]['Significant']:
                bar.set_color('darkblue')
            else:
                bar.set_color('lightblue')
        
        # Add labels and title
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title(f'Correlation with {pollutant}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values as text
        for j, v in enumerate(pollutant_corrs['Correlation']):
            ax.text(v + (0.02 if v >= 0 else -0.08), j, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/domain_deprivation/domain_correlation_bars.png', dpi=300)
    
    return corr_df

def create_domain_vulnerability_indices(unified_df, domain_vars, domain_names):
    """
    Create domain-specific vulnerability indices combining pollution and specific deprivation domains.
    
    Args:
        unified_df (pd.DataFrame): Unified dataset with deprivation domains
        domain_vars (list): List of deprivation domain variables
        domain_names (list): List of deprivation domain names
        
    Returns:
        pd.DataFrame: DataFrame with domain-specific vulnerability indices
    """
    print("\nCreating domain-specific vulnerability indices...")
    
    # Define pollution variables
    pollution_vars = ['NO2_normalized', 'PM2.5_normalized']
    
    # Create a copy of the dataset for adding indices
    df_with_indices = unified_df.copy()
    
    # Create domain-specific vulnerability indices
    for d_var, d_name in zip(domain_vars, domain_names):
        # Normalize domain score (if not already normalized)
        if not d_var.endswith('_normalized'):
            # Check if the domain scores need to be inverted (if higher is better)
            if 'rank' in d_var and not 'where_1_is_most_deprived' in d_var:
                # For ranks, higher is better, so invert
                domain_normalized = 1 - (unified_df[d_var] - unified_df[d_var].min()) / (unified_df[d_var].max() - unified_df[d_var].min())
            else:
                # For scores, higher is worse, so normalize directly
                domain_normalized = (unified_df[d_var] - unified_df[d_var].min()) / (unified_df[d_var].max() - unified_df[d_var].min())
            
            # Add normalized domain to dataframe
            norm_var_name = f"{d_var}_normalized"
            df_with_indices[norm_var_name] = domain_normalized
        else:
            # Already normalized
            norm_var_name = d_var
        
        # Create domain-specific vulnerability index for each pollution variable
        for p_var in pollution_vars:
            # Simple average of normalized domain and pollution
            index_name = f"{d_name.lower().replace(' ', '_')}_{p_var.split('_')[0]}_vulnerability"
            df_with_indices[index_name] = (df_with_indices[norm_var_name] + df_with_indices[p_var]) / 2
            
            # Scale to 0-100 for interpretability
            min_val = df_with_indices[index_name].min()
            max_val = df_with_indices[index_name].max()
            df_with_indices[index_name] = 100 * (df_with_indices[index_name] - min_val) / (max_val - min_val)
            
            print(f"Created {index_name} (mean: {df_with_indices[index_name].mean():.2f}, std: {df_with_indices[index_name].std():.2f})")
    
    # Save the dataset with domain-specific indices
    df_with_indices.to_csv('outputs/domain_deprivation/lsoa_with_domain_indices.csv', index=False)
    
    return df_with_indices

def identify_domain_hotspots(df_with_indices, domain_names):
    """
    Identify hotspots for each domain-specific vulnerability index.
    
    Args:
        df_with_indices (pd.DataFrame): DataFrame with domain-specific vulnerability indices
        domain_names (list): List of deprivation domain names
    """
    print("\nIdentifying domain-specific vulnerability hotspots...")
    
    # Define pollution variables
    pollution_types = ['NO2', 'PM2.5']
    
    # Create a figure for plotting hotspot distributions
    plt.figure(figsize=(16, 12))
    
    # Calculate number of rows and columns for subplots
    n_domains = len(domain_names)
    n_cols = 2  # One for each pollution type
    n_rows = (n_domains + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    # Flatten axes if multiple rows
    if n_rows > 1:
        axes = axes.flatten()
    
    # Identify hotspots for each domain and pollution combination
    hotspot_results = []
    
    for i, d_name in enumerate(domain_names):
        for j, p_type in enumerate(pollution_types):
            # Construct index name
            index_name = f"{d_name.lower().replace(' ', '_')}_{p_type}_vulnerability"
            
            # Check if index exists
            if index_name not in df_with_indices.columns:
                continue
            
            # Identify top 5% as hotspots
            threshold = df_with_indices[index_name].quantile(0.95)
            hotspots = df_with_indices[df_with_indices[index_name] >= threshold]
            
            # Store results
            hotspot_results.append({
                'Domain': d_name,
                'Pollution': p_type,
                'Index': index_name,
                'Threshold': threshold,
                'Hotspot_Count': len(hotspots),
                'Hotspots': hotspots
            })
            
            # Plot distribution with threshold
            ax_idx = i * n_cols + j
            if ax_idx < len(axes):
                ax = axes[ax_idx]
                sns.histplot(df_with_indices[index_name], bins=30, kde=True, ax=ax)
                ax.axvline(x=threshold, color='red', linestyle='--', label=f'95th Percentile ({threshold:.1f})')
                ax.set_title(f'{d_name} × {p_type} Vulnerability')
                ax.set_xlabel('Vulnerability Index (0-100)')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/domain_deprivation/domain_hotspot_distributions.png', dpi=300)
    
    # Create summary of hotspots
    print("\nDomain-specific vulnerability hotspots:")
    for result in hotspot_results:
        print(f"\n{result['Domain']} × {result['Pollution']} Vulnerability:")
        print(f"- Threshold (95th percentile): {result['Threshold']:.2f}")
        print(f"- Number of hotspot areas: {result['Hotspot_Count']}")
        
        # Get top 5 hotspots
        top_hotspots = result['Hotspots'].sort_values(result['Index'], ascending=False).head(5)
        print("- Top 5 hotspot areas:")
        for _, row in top_hotspots.iterrows():
            print(f"  * {row['lsoa_name']} ({row['lad_name']}): {row[result['Index']]:.2f}")
    
    # Create a summary table of hotspot counts by LAD
    hotspot_summary = []
    
    for result in hotspot_results:
        # Count hotspots by LAD
        lad_counts = result['Hotspots']['lad_name'].value_counts().reset_index()
        lad_counts.columns = ['LAD', 'Count']
        lad_counts['Domain'] = result['Domain']
        lad_counts['Pollution'] = result['Pollution']
        
        # Add to summary
        hotspot_summary.append(lad_counts)
    
    # Combine all summaries
    if hotspot_summary:
        hotspot_summary_df = pd.concat(hotspot_summary)
        
        # Save summary
        hotspot_summary_df.to_csv('outputs/domain_deprivation/hotspot_summary_by_lad.csv', index=False)
        
        # Create pivot table for visualization
        pivot_table = hotspot_summary_df.pivot_table(
            index='LAD', 
            columns=['Domain', 'Pollution'], 
            values='Count',
            fill_value=0
        )
        
        # Save pivot table
        pivot_table.to_csv('outputs/domain_deprivation/hotspot_pivot_table.csv')
    
    return hotspot_results

def create_radar_charts(df_with_indices, domain_vars, domain_names):
    """
    Create radar charts to visualize multidimensional deprivation profiles by area.
    
    Args:
        df_with_indices (pd.DataFrame): DataFrame with domain-specific vulnerability indices
        domain_vars (list): List of deprivation domain variables
        domain_names (list): List of deprivation domain names
    """
    print("\nCreating radar charts for multidimensional deprivation profiles...")
    
    # Aggregate to LAD level for better visualization
    lad_aggregated = df_with_indices.groupby('lad_code').agg({
        'lad_name': 'first',
        **{var: 'mean' for var in domain_vars}
    }).reset_index()
    
    # Normalize domain variables for radar chart
    domain_normalized = {}
    for var in domain_vars:
        # Check if the domain scores need to be inverted (if higher is better)
        if 'rank' in var and not 'where_1_is_most_deprived' in var:
            # For ranks, higher is better, so invert
            domain_normalized[var] = 1 - (lad_aggregated[var] - lad_aggregated[var].min()) / (lad_aggregated[var].max() - lad_aggregated[var].min())
        else:
            # For scores, higher is worse, so normalize directly
            domain_normalized[var] = (lad_aggregated[var] - lad_aggregated[var].min()) / (lad_aggregated[var].max() - lad_aggregated[var].min())
    
    # Add normalized values to dataframe
    for var, values in domain_normalized.items():
        lad_aggregated[f"{var}_normalized"] = values
    
    # Identify areas with different deprivation profiles
    # Use clustering to find representative areas
    
    # Prepare data for clustering
    cluster_data = np.array([domain_normalized[var] for var in domain_vars]).T
    
    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    lad_aggregated['cluster'] = kmeans.fit_predict(cluster_data)
    
    # Find representative area for each cluster (closest to centroid)
    representative_areas = []
    
    for cluster_id in range(5):
        cluster_members = lad_aggregated[lad_aggregated['cluster'] == cluster_id]
        
        if len(cluster_members) == 0:
            continue
        
        # Calculate distance to centroid for each member
        centroid = kmeans.cluster_centers_[cluster_id]
        
        # Calculate distances
        distances = []
        for idx, row in cluster_members.iterrows():
            profile = np.array([domain_normalized[var][idx] for var in domain_vars])
            distance = np.sqrt(np.sum((profile - centroid) ** 2))
            distances.append((idx, distance))
        
        # Find closest member
        closest_idx = min(distances, key=lambda x: x[1])[0]
        representative_areas.append(cluster_members.loc[closest_idx])
    
    # Create radar charts for representative areas
    for i, area in enumerate(representative_areas):
        # Prepare data for radar chart
        categories = domain_names
        values = [domain_normalized[var][area.name] for var in domain_vars]
        
        # Close the loop for the radar chart
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Create radar chart
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # Plot radar chart
        ax.plot(np.linspace(0, 2*np.pi, len(categories)), values, 'o-', linewidth=2)
        ax.fill(np.linspace(0, 2*np.pi, len(categories)), values, alpha=0.25)
        
        # Set category labels
        ax.set_xticks(np.linspace(0, 2*np.pi, len(categories)-1))
        ax.set_xticklabels(categories[:-1])
        
        # Set radial limits
        ax.set_ylim(0, 1)
        
        # Set title
        plt.title(f"Deprivation Profile: {area['lad_name']} (Cluster {i})", size=15)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'outputs/domain_deprivation/radar_cluster_{i}.png', dpi=300)
    
    # Create interactive radar chart with Plotly
    # Select top 5 areas with highest overall deprivation
    lad_aggregated['overall_deprivation'] = np.mean([domain_normalized[var] for var in domain_vars], axis=0)
    top_deprived = lad_aggregated.sort_values('overall_deprivation', ascending=False).head(5)
    
    # Create radar chart
    fig = go.Figure()
    
    for idx, row in top_deprived.iterrows():
        values = [domain_normalized[var][idx] for var in domain_vars]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=domain_names,
            fill='toself',
            name=row['lad_name']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Deprivation Profiles of Most Deprived Areas"
    )
    
    # Save as HTML
    fig.write_html('outputs/domain_deprivation/interactive_radar_top_deprived.html')
    
    # Create radar chart comparing areas with different pollution levels
    # Group areas by NO2 level
    try:
        # Try to create quantiles, handling duplicate values
        lad_aggregated['NO2_level'] = pd.qcut(
            df_with_indices.groupby('lad_code')['NO2'].mean(),
            q=3,
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'  # Handle duplicate bin edges
        )
    except ValueError:
        # If that fails, use simple cut with manual bins
        print("Warning: Using manual bins for NO2 levels due to duplicate values")
        no2_values = df_with_indices.groupby('lad_code')['NO2'].mean()
        min_val, max_val = no2_values.min(), no2_values.max()
        bin_edges = [min_val, min_val + (max_val - min_val)/3, min_val + 2*(max_val - min_val)/3, max_val]
        lad_aggregated['NO2_level'] = pd.cut(
            no2_values,
            bins=bin_edges,
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
    
    # Calculate average domain profiles by pollution level
    pollution_profiles = lad_aggregated.groupby('NO2_level', observed=False).agg({
        **{f"{var}_normalized": 'mean' for var in domain_vars}
    }).reset_index()
    
    # Create radar chart
    fig = go.Figure()
    
    for idx, row in pollution_profiles.iterrows():
        values = [row[f"{var}_normalized"] for var in domain_vars]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=domain_names,
            fill='toself',
            name=f"NO2: {row['NO2_level']}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Deprivation Profiles by Pollution Level"
    )
    
    # Save as HTML
    fig.write_html('outputs/domain_deprivation/interactive_radar_by_pollution.html')
    
    return representative_areas

def analyze_domain_importance(unified_df, domain_vars, domain_names):
    """
    Analyze the importance of different deprivation domains for predicting pollution exposure.
    
    Args:
        unified_df (pd.DataFrame): Unified dataset with deprivation domains
        domain_vars (list): List of deprivation domain variables
        domain_names (list): List of deprivation domain names
    """
    print("\nAnalyzing importance of deprivation domains for predicting pollution exposure...")
    
    # Define pollution variables
    pollution_vars = ['NO2', 'PM2.5']
    
    # Prepare data
    analysis_data = unified_df[domain_vars + pollution_vars].dropna()
    
    # Normalize domain variables if needed
    domain_normalized = {}
    for var in domain_vars:
        # Check if the domain scores need to be inverted (if higher is better)
        if 'rank' in var and not 'where_1_is_most_deprived' in var:
            # For ranks, higher is better, so invert
            domain_normalized[var] = 1 - (analysis_data[var] - analysis_data[var].min()) / (analysis_data[var].max() - analysis_data[var].min())
        else:
            # For scores, higher is worse, so normalize directly
            domain_normalized[var] = (analysis_data[var] - analysis_data[var].min()) / (analysis_data[var].max() - analysis_data[var].min())
    
    # Create DataFrame with normalized domains
    normalized_data = pd.DataFrame(domain_normalized)
    
    # Add pollution variables
    for var in pollution_vars:
        normalized_data[var] = analysis_data[var]
    
    # Analyze importance for each pollution variable
    importance_results = []
    
    for p_var in pollution_vars:
        print(f"\nAnalyzing domain importance for {p_var}:")
        
        # Prepare data
        X = normalized_data[domain_vars]
        y = normalized_data[p_var]
        
        # Fit Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importances = rf.feature_importances_
        
        # Create DataFrame with importances
        importance_df = pd.DataFrame({
            'Domain': domain_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Print importance
        print(importance_df)
        
        # Add to results
        importance_df['Pollutant'] = p_var
        importance_results.append(importance_df)
        
        # Plot importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Domain', data=importance_df)
        plt.title(f'Domain Importance for Predicting {p_var}', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Deprivation Domain', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'outputs/domain_deprivation/domain_importance_{p_var}.png', dpi=300)
    
    # Combine all importance results
    all_importance = pd.concat(importance_results)
    
    # Create a heatmap of importance by pollutant
    plt.figure(figsize=(12, 8))
    importance_pivot = all_importance.pivot(index='Domain', columns='Pollutant', values='Importance')
    sns.heatmap(importance_pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title('Domain Importance by Pollutant', fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/domain_deprivation/domain_importance_heatmap.png', dpi=300)
    
    return all_importance

def main():
    """Main function to execute the domain-specific deprivation analysis."""
    print("Starting Domain-Specific Deprivation Analysis...")
    
    # Load data
    unified_df = load_data()
    
    # Identify deprivation domains
    domain_vars, domain_names = identify_deprivation_domains(unified_df)
    
    # Analyze correlations between domains and pollution
    corr_df = analyze_domain_correlations(unified_df, domain_vars, domain_names)
    
    # Create domain-specific vulnerability indices
    df_with_indices = create_domain_vulnerability_indices(unified_df, domain_vars, domain_names)
    
    # Identify domain-specific hotspots
    hotspot_results = identify_domain_hotspots(df_with_indices, domain_names)
    
    # Create radar charts for multidimensional deprivation profiles
    representative_areas = create_radar_charts(df_with_indices, domain_vars, domain_names)
    
    # Analyze domain importance
    importance_df = analyze_domain_importance(unified_df, domain_vars, domain_names)
    
    print("\nDomain-specific deprivation analysis complete. Results saved to the 'outputs/domain_deprivation' directory.")

if __name__ == "__main__":
    main()