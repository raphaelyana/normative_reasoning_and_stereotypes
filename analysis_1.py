import os
import glob
import json
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy.stats import (
    bootstrap, f_oneway, ttest_ind, kruskal, mannwhitneyu, pearsonr
)

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


from analysis_0 import *
from profile_message import get_profile_traits, PROFILE_META_SYSTEMATIC, ProfileMeta



def factorial_analysis_3way_anova(merged_df, group_keys=("gender", "ethnicity", "cognitive_style")):
    """
    Configurable N-way ANOVA: Trait x Trait x Trait -> Performance Metrics
    
    Tests which profile traits (gender, ethnicity, cognitive style) affect:
    - Accuracy
    - Rescue rate  
    - Bias magnitude
    - Extra error rate
    
    Parameters:
    - merged_df: DataFrame containing predictions and true labels
    - group_keys: tuple of metadata fields to extract and analyze

    Returns:
    - dict with performance data, anova results, and significant effects
    """
    
    # Define persona groups with _passive suffix
    profile_cols = [col for col in merged_df.columns if col.startswith("profile") and "_passive" in col]
    
    # Create mapping from profile to traits
    profile_traits = {
        profile: get_profile_traits(profile, group_keys=("gender", "ethnicity", "cognitive_style"))
        for profile in profile_cols
    }
    
    # Calculate performance metrics for each profile
    performance_data = []
    
    # Get rescue stats for reference
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    
    for profile in profile_cols:
        if profile in merged_df.columns:
            
            # Basic accuracy
            accuracy = (merged_df[profile] == merged_df['true_label']).mean()
            
            # Get rescue stats for this profile
            profile_rescue = rescue_stats[rescue_stats['profile'] == profile]
            avg_rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
            avg_extra_error_rate = profile_rescue['extra_err_rate'].mean() if len(profile_rescue) > 0 else 0
            
            # Get bias patterns for this profile  
            profile_bias = bias_patterns[bias_patterns['profile'] == profile]
            avg_bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
            avg_flip_rate = profile_bias['flip_rate'].mean() if len(profile_bias) > 0 else 0
            
            # Append to performance data
            traits = profile_traits[profile]
            performance_data.append({
                'profile': profile,
                'gender': traits['gender'],
                'ethnicity': traits['ethnicity'],
                'cognitive_style': traits['cognitive_style'],
                'accuracy': accuracy,
                'rescue_rate': avg_rescue_rate,
                'extra_error_rate': avg_extra_error_rate,
                'bias_magnitude': avg_bias_magnitude,
                'flip_rate': avg_flip_rate
            })
    
    performance_df = pd.DataFrame(performance_data)
    
    # ========================================================================
    # ANOVA TESTS FOR EACH DEPENDENT VARIABLE
    # ========================================================================
    
    results = {}
    dependent_vars = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude', 'flip_rate']
    
    for dv in dependent_vars:
        
        dv_results = {'dependent_variable': dv}
        
        # Main effects
        print(f"\n{'='*60}")
        print(f"FACTORIAL ANOVA: {dv.upper()}")
        print(f"{'='*60}")
        
        # Gender main effect
        genders = performance_df["gender"].dropna().unique()

        if len(genders) == 2:
            g1, g2 = genders
            g1_vals = performance_df[performance_df['gender'] == g1][dv].values
            g2_vals = performance_df[performance_df['gender'] == g2][dv].values

            if len(g1_vals) > 0 and len(g2_vals) > 0:
                t_stat, p_val = ttest_ind(g1_vals, g2_vals)
                dv_results['gender_main_effect'] = {
                    'group1': g1,
                    'group2': g2,
                    'group1_mean': np.mean(g1_vals),
                    'group2_mean': np.mean(g2_vals),
                    'difference': np.mean(g2_vals) - np.mean(g1_vals),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'effect_size_d': (np.mean(g2_vals) - np.mean(g1_vals)) / np.sqrt((np.var(g1_vals) + np.var(g2_vals)) / 2)
                }
            
            print(f"GENDER MAIN EFFECT ({g1} vs {g2}):")
            print(f"  {g1} mean: {np.mean(g1_vals):.4f}")
            print(f"  {g2} mean: {np.mean(g2_vals):.4f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
            print(f"  Cohen's d: {dv_results['gender_main_effect']['effect_size_d']:.3f}")
        
        # Ethnicity main effect
        ethnicities = performance_df["ethnicity"].dropna().unique()
        ethnicity_groups = [
            performance_df[performance_df["ethnicity"] == eth][dv].values
            for eth in ethnicities
        ]
        
        if all(len(vals)>0 for vals in ethnicity_groups) and len(ethnicity_groups)>=2:
            f_stat, p_val = f_oneway(*ethnicity_groups)
            dv_results['ethnicity_main_effect'] = {
                'ethnicity_means': {
                    eth: np.mean(performance_df[performance_df["ethnicity"] == eth][dv].values)
                    for eth in ethnicities
                },
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
                    
            print(f"\nETHNICITY MAIN EFFECT:")
            for eth in ethnicities:
                mean_val = performance_df[performance_df["ethnicity"] == eth][dv].mean()
                print(f"  {eth} mean: {mean_val:.4f}")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
        

        # Cognitive style main effect
        cognitive_styles = performance_df['cognitive_style'].unique()
        cognitive_vals = []
        cognitive_means = {}
        
        for style in cognitive_styles:
            style_vals = performance_df[performance_df['cognitive_style'] == style][dv].values
            if len(style_vals) > 0:
                cognitive_vals.append(style_vals)
                cognitive_means[style] = np.mean(style_vals)
        
        if len(cognitive_vals) >= 2:
            f_stat, p_val = f_oneway(*cognitive_vals)
            dv_results['cognitive_main_effect'] = {
                'style_means': cognitive_means,
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            
            print(f"\nCOGNITIVE STYLE MAIN EFFECT:")
            for style, mean_val in cognitive_means.items():
                print(f"  {style} mean: {mean_val:.4f}")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
        

        # Two-way interactions (Gender × Ethnicity)
        print(f"\nINTERACTION EFFECTS:")
        interaction_means = {}
        for gender in genders:
            for ethnicity in ethnicities:
                subset = performance_df[
                    (performance_df['gender'] == gender) &
                    (performance_df['ethnicity'] == ethnicity)
                ][dv].values
                if len(subset) > 0:
                    interaction_means[f"{gender}_{ethnicity}"] = np.mean(subset)
                    print(f"  {gender} {ethnicity}: {np.mean(subset):.4f}")
        
        dv_results['interaction_means'] = interaction_means
        
        results[dv] = dv_results
    


    # Summary of significant effects
    print(f"\n{'='*60}")
    print("FACTORIAL ANOVA SUMMARY")
    print(f"{'='*60}")
    
    significant_effects = []
    for dv, dv_results in results.items():
        print(f"\n{dv.upper()}:")
        
        for effect_type in ['gender_main_effect', 'ethnicity_main_effect', 'cognitive_main_effect']:
            if effect_type in dv_results and dv_results[effect_type]['significant']:
                print(f"  ✓ {effect_type}: p={dv_results[effect_type]['p_value']:.4f}")
                significant_effects.append(f"{dv}_{effect_type}")
            elif effect_type in dv_results:
                print(f"    {effect_type}: p={dv_results[effect_type]['p_value']:.4f}")
    
    print(f"\nTotal significant effects found: {len(significant_effects)}")
    
    return {
        'performance_data': performance_df,
        'anova_results': results,
        'significant_effects': significant_effects,
        'profile_traits': profile_traits
    }


def plot_risk_benefit_frontier(merged_df, figsize=(12, 8)):
    """
    Risk-Benefit Frontier Analysis
    
    Plots each profile on:
    - X-axis: Extra error rate (risk)
    - Y-axis: Rescue rate (benefit)
    
    Identifies Pareto frontier of "safely bold" profiles that maximize
    rescue while minimizing extra errors.
    """
    
    # Get rescue stats
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    
    # Calculate overall rescue and extra error rates per profile
    profile_performance = rescue_stats.groupby('profile').agg({
        'rescue_rate': 'mean',
        'extra_err_rate': 'mean', 
        'rescued': 'sum',
        'extra_errors': 'sum'
    }).reset_index()
    
    # Add demographic info for coloring
    profile_traits = {
        profile: get_profile_traits(profile, group_keys=["gender", "ethnicity", "cognitive_style"])
        for profile in profile_performance["profile"]
    }

    # Add trait information to performance data
    group_keys = ["gender", "ethnicity", "cognitive_style"]

    for trait_name in group_keys:
        profile_performance[trait_name] = profile_performance["profile"].map(
            lambda p: profile_traits.get(p, {}).get(trait_name, "Unknown")
        )
    
    profile_performance["intersectional"] = profile_performance.apply(
        lambda row: f"{row['gender']}_{row['ethnicity']}", axis=1
    )
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Risk-Benefit Frontier Analysis: Rescue Rate vs Extra Error Rate', fontsize=16, fontweight='bold')
    
    # Color schemes for different groupings
    color_schemes = {
        'gender': {'Male': '#1f77b4', 'Female': '#ff7f0e'},
        'ethnicity': {'White': '#2ca02c', 'Black': '#d62728', 'Asian': '#9467bd'},
        'cognitive_style': {
            'Expansive': '#8c564b', 'Literal': '#e377c2', 
            'High_Harm': '#7f7f7f', 'Low_Harm': '#bcbd22', 'Balanced': '#17becf'
        }
    }
    
    # Plot 1: Gender
    ax = axes[0, 0]
    for gender in ['Male', 'Female']:
        subset = profile_performance[profile_performance['gender'] == gender]
        ax.scatter(subset['extra_err_rate'], subset['rescue_rate'], 
                  c=color_schemes['gender'][gender], label=gender, alpha=0.7, s=60)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Risk-Benefit by Gender')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ethnicity  
    ax = axes[0, 1]
    for ethnicity in ['White', 'Black', 'Asian']:
        subset = profile_performance[profile_performance['ethnicity'] == ethnicity]
        ax.scatter(subset['extra_err_rate'], subset['rescue_rate'],
                  c=color_schemes['ethnicity'][ethnicity], label=ethnicity, alpha=0.7, s=60)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Risk-Benefit by Ethnicity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cognitive Style
    ax = axes[1, 0]
    for style in color_schemes['cognitive_style'].keys():
        subset = profile_performance[profile_performance['cognitive_style'] == style]
        if len(subset) > 0:
            ax.scatter(subset['extra_err_rate'], subset['rescue_rate'],
                      c=color_schemes['cognitive_style'][style], label=style, alpha=0.7, s=60)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Risk-Benefit by Cognitive Style')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Pareto Frontier Identification
    ax = axes[1, 1]
    
    # Plot all points
    scatter = ax.scatter(profile_performance['extra_err_rate'], profile_performance['rescue_rate'],
                        c='lightblue', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Calculate Pareto frontier
    points = profile_performance[['extra_err_rate', 'rescue_rate']].values
    
    # Find Pareto optimal points (minimize risk, maximize benefit)
    pareto_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        # A point is Pareto optimal if no other point dominates it
        # (lower extra_err_rate AND higher rescue_rate)
        dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[0] <= point[0] and other_point[1] >= point[1] and 
                    (other_point[0] < point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
        
        if not dominated:
            pareto_mask[i] = True
    
    # Highlight Pareto optimal points
    pareto_points = profile_performance[pareto_mask]
    ax.scatter(pareto_points['extra_err_rate'], pareto_points['rescue_rate'],
              c='red', s=100, marker='*', label='Pareto Optimal', zorder=5)
    
    # Add profile labels for Pareto optimal points
    for idx, row in pareto_points.iterrows():
        ax.annotate(row['profile'].replace('_passive', ''), 
                   (row['extra_err_rate'], row['rescue_rate']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Pareto Frontier: "Safely Bold" Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print Pareto optimal profiles
    print("\n" + "="*60)
    print("PARETO FRONTIER ANALYSIS")
    print("="*60)
    print("\nPareto Optimal Profiles (Safely Bold):")
    for idx, row in pareto_points.iterrows():
        traits = profile_traits.get(row['profile'], {})
        print(f"  {row['profile']}: {traits.get('gender', 'Unknown')} {traits.get('ethnicity', 'Unknown')} {traits.get('cognitive_style', 'Unknown')}")
        print(f"    Rescue Rate: {row['rescue_rate']:.3f}, Extra Error Rate: {row['extra_err_rate']:.3f}")
    
    # Risk-Benefit statistics
    print(f"\nRisk-Benefit Statistics:")
    print(f"  Average rescue rate: {profile_performance['rescue_rate'].mean():.3f}")
    print(f"  Average extra error rate: {profile_performance['extra_err_rate'].mean():.3f}")
    print(f"  Best rescue rate: {profile_performance['rescue_rate'].max():.3f}")
    print(f"  Lowest extra error rate: {profile_performance['extra_err_rate'].min():.3f}")
    
    # Identify specific profile archetypes
    print(f"\nProfile Archetypes:")
    
    # Safest (lowest extra error rate)
    safest = profile_performance.loc[profile_performance['extra_err_rate'].idxmin()]
    safest_traits = profile_traits.get(safest['profile'], {})
    print(f"  Safest Profile: {safest['profile']} ({safest_traits.get('gender', '')} {safest_traits.get('ethnicity', '')} {safest_traits.get('cognitive_style', '')})")
    print(f"    Extra Error Rate: {safest['extra_err_rate']:.3f}")
    
    # Most Beneficial (highest rescue rate)
    most_beneficial = profile_performance.loc[profile_performance['rescue_rate'].idxmax()]
    beneficial_traits = profile_traits.get(most_beneficial['profile'], {})
    print(f"  Most Beneficial Profile: {most_beneficial['profile']} ({beneficial_traits.get('gender', '')} {beneficial_traits.get('ethnicity', '')} {beneficial_traits.get('cognitive_style', '')})")
    print(f"    Rescue Rate: {most_beneficial['rescue_rate']:.3f}")
    
    return {
        'performance_data': profile_performance,
        'pareto_optimal': pareto_points,
        'profile_traits': profile_traits,
        'safest_profile': safest,
        'most_beneficial_profile': most_beneficial
    }


def calculate_effect_sizes(performance_df, anova_results):
    """
    Calculate Cohen's d effect sizes directly from ANOVA performance data.
    
    Analyzes profile-level aggregated metrics to determine which demographic
    and cognitive factors have the largest effects on performance.
    
    Effect size interpretation:
    - Small: d = 0.2
    - Medium: d = 0.5  
    - Large: d = 0.8
    """
    
    effect_sizes = {}
    
    print("CALCULATING ANOVA-BASED EFFECT SIZES")
    print("=" * 60)
    
    def calculate_cohens_d(group1_vals, group2_vals, group1_name, group2_name):
        """Calculate Cohen's d effect size between two groups"""
        if len(group1_vals) == 0 or len(group2_vals) == 0:
            return None
            
        mean1, mean2 = np.mean(group1_vals), np.mean(group2_vals)
        var1, var2 = np.var(group1_vals, ddof=1), np.var(group2_vals, ddof=1)
        
        # Handle case where both variances are 0
        if var1 == 0 and var2 == 0:
            cohens_d = 0.0
        else:
            pooled_std = np.sqrt((var1 + var2) / 2)
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        magnitude = 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'
        
        return {
            'cohens_d': cohens_d,
            'magnitude': magnitude,
            f'{group1_name}_mean': mean1,
            f'{group2_name}_mean': mean2,
            'difference': mean1 - mean2,
            f'{group1_name}_n': len(group1_vals),
            f'{group2_name}_n': len(group2_vals)
        }
    
    # Dependent variables to analyze
    dependent_vars = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude']
    
    # Add disagreement_rate if it exists, otherwise flip_rate
    if 'disagreement_rate' in performance_df.columns:
        dependent_vars.append('disagreement_rate')
    elif 'flip_rate' in performance_df.columns:
        dependent_vars.append('flip_rate')
    
    # Main Effects Analysis
    print("\nMAIN EFFECTS FROM ANOVA")
    print("-" * 40)
    
    for dv in dependent_vars:
        if dv not in performance_df.columns:
            continue
            
        dv_results = anova_results.get(dv, {})
        
        # Gender main effect
        if 'gender_main_effect' in dv_results and dv_results['gender_main_effect'].get('significant', False):
            genders = performance_df["gender"].dropna().unique()
            ethnicities = performance_df["ethnicity"].dropna().unique()
            
            effect_size = calculate_cohens_d(male_vals, female_vals, 'male', 'female')
            if effect_size:
                effect_sizes[f'{dv}_gender_effect'] = effect_size
                print(f"  {dv} - Gender: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")
        
        # Ethnicity main effect  
        if 'ethnicity_main_effect' in dv_results and dv_results['ethnicity_main_effect'].get('significant', False):
            white_vals = performance_df[performance_df['ethnicity'] == 'White'][dv].values
            black_vals = performance_df[performance_df['ethnicity'] == 'Black'][dv].values
            asian_vals = performance_df[performance_df['ethnicity'] == 'Asian'][dv].values
            
            # Calculate pairwise effect sizes for ethnicity
            ethnicity_pairs = [
                (white_vals, black_vals, 'white', 'black'),
                (white_vals, asian_vals, 'white', 'asian'),
                (black_vals, asian_vals, 'black', 'asian')
            ]
            
            for vals1, vals2, name1, name2 in ethnicity_pairs:
                effect_size = calculate_cohens_d(vals1, vals2, name1, name2)
                if effect_size:
                    effect_sizes[f'{dv}_ethnicity_{name1}_vs_{name2}'] = effect_size
                    print(f"  {dv} - {name1.title()} vs {name2.title()}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")
        
        # Cognitive style main effect
        if 'cognitive_main_effect' in dv_results and dv_results['cognitive_main_effect'].get('significant', False):
            cognitive_styles = performance_df['cognitive_style'].unique()
            
            # Calculate pairwise effect sizes for cognitive styles
            for i, style1 in enumerate(cognitive_styles):
                for style2 in cognitive_styles[i+1:]:
                    vals1 = performance_df[performance_df['cognitive_style'] == style1][dv].values
                    vals2 = performance_df[performance_df['cognitive_style'] == style2][dv].values
                    
                    if len(vals1) > 0 and len(vals2) > 0:
                        effect_size = calculate_cohens_d(vals1, vals2, style1.lower(), style2.lower())
                        if effect_size and abs(effect_size['cohens_d']) > 0.2:  # Only meaningful effects
                            effect_sizes[f'{dv}_cognitive_{style1.lower()}_vs_{style2.lower()}'] = effect_size
                            print(f"  {dv} - {style1} vs {style2}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")
    
    # Interaction Effects Analysis
    print("\nINTERACTION EFFECTS")
    print("-" * 40)
    
    # Gender × Ethnicity interactions
    for dv in dependent_vars:
        if dv not in performance_df.columns:
            continue
            
        # Test key intersectional comparisons
        intersectional_comparisons = [
            (['Male', 'White'], ['Female', 'White'], 'white_male', 'white_female'),
            (['Male', 'Black'], ['Female', 'Black'], 'black_male', 'black_female'),
            (['Male', 'Asian'], ['Female', 'Asian'], 'asian_male', 'asian_female'),
            (['Male', 'White'], ['Male', 'Black'], 'white_male', 'black_male'),
            (['Female', 'White'], ['Female', 'Black'], 'white_female', 'black_female'),
            (['Male', 'White'], ['Female', 'Black'], 'white_male', 'black_female')
        ]
        
        for criteria1, criteria2, name1, name2 in intersectional_comparisons:
            vals1 = performance_df[
                (performance_df['gender'] == criteria1[0]) & 
                (performance_df['ethnicity'] == criteria1[1])
            ][dv].values
            
            vals2 = performance_df[
                (performance_df['gender'] == criteria2[0]) & 
                (performance_df['ethnicity'] == criteria2[1])
            ][dv].values
            
            if len(vals1) > 0 and len(vals2) > 0:
                effect_size = calculate_cohens_d(vals1, vals2, name1, name2)
                if effect_size and abs(effect_size['cohens_d']) > 0.3:  # Only substantial interactions
                    effect_sizes[f'{dv}_interaction_{name1}_vs_{name2}'] = effect_size
                    print(f"  {dv} - {name1.replace('_', ' ').title()} vs {name2.replace('_', ' ').title()}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")
    
    # Comprehensive Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANOVA EFFECT SIZE ANALYSIS")
    print("=" * 80)
    
    if not effect_sizes:
        print("No significant effect sizes found.")
        return effect_sizes
    
    # Sort by absolute effect size
    sorted_effects = sorted(effect_sizes.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)
    
    # Categorize by magnitude
    large_effects = [(k, v) for k, v in sorted_effects if abs(v['cohens_d']) >= 0.8]
    medium_effects = [(k, v) for k, v in sorted_effects if 0.5 <= abs(v['cohens_d']) < 0.8]
    small_effects = [(k, v) for k, v in sorted_effects if 0.2 <= abs(v['cohens_d']) < 0.5]
    
    print(f"\nEFFECT SIZE SUMMARY:")
    print(f"  Large effects (d >= 0.8): {len(large_effects)}")
    print(f"  Medium effects (0.5 <= d < 0.8): {len(medium_effects)}")
    print(f"  Small effects (0.2 <= d < 0.5): {len(small_effects)}")
    
    # Print top effects in each category
    for category_name, effects_list in [("LARGE", large_effects), ("MEDIUM", medium_effects), ("SMALL", small_effects[:10])]:
        if effects_list:
            print(f"\n{category_name} EFFECTS:")
            print("-" * 50)
            for finding, stats in effects_list:
                # Parse the finding name for better display
                parts = finding.split('_')
                dv = parts[0]
                effect_type = '_'.join(parts[1:])
                
                print(f"{dv.upper()} - {effect_type.replace('_', ' ').title()}:")
                print(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['magnitude']} effect)")
                print(f"  Mean difference: {stats['difference']:.4f}")
                
                # Print group means
                for key, value in stats.items():
                    if key.endswith('_mean'):
                        group_name = key.replace('_mean', '')
                        n_key = f"{group_name}_n"
                        n_val = stats.get(n_key, '?')
                        print(f"  {group_name.title()}: {value:.4f} (n={n_val})")
                print()
    
    # Key Insights
    print("KEY INSIGHTS:")
    print("-" * 30)
    
    # Factor importance ranking
    factor_effects = {}
    for finding, stats in effect_sizes.items():
        if 'gender_effect' in finding:
            factor_effects['Gender'] = factor_effects.get('Gender', []) + [abs(stats['cohens_d'])]
        elif 'ethnicity_' in finding:
            factor_effects['Ethnicity'] = factor_effects.get('Ethnicity', []) + [abs(stats['cohens_d'])]
        elif 'cognitive_' in finding:
            factor_effects['Cognitive Style'] = factor_effects.get('Cognitive Style', []) + [abs(stats['cohens_d'])]
        elif 'interaction_' in finding:
            factor_effects['Interactions'] = factor_effects.get('Interactions', []) + [abs(stats['cohens_d'])]
    
    factor_summary = {}
    for factor, effects in factor_effects.items():
        factor_summary[factor] = {
            'mean_effect': np.mean(effects),
            'max_effect': np.max(effects),
            'count': len(effects)
        }
    
    # Sort factors by average effect size
    sorted_factors = sorted(factor_summary.items(), key=lambda x: x[1]['mean_effect'], reverse=True)
    
    print("Factor importance ranking (by average effect size):")
    for i, (factor, stats) in enumerate(sorted_factors, 1):
        print(f"  {i}. {factor}: avg d = {stats['mean_effect']:.3f}, max d = {stats['max_effect']:.3f} ({stats['count']} effects)")
    
    # Performance metric bias ranking
    dv_effects = {}
    for finding, stats in effect_sizes.items():
        dv = finding.split('_')[0]
        dv_effects[dv] = dv_effects.get(dv, []) + [abs(stats['cohens_d'])]
    
    dv_summary = {dv: np.mean(effects) for dv, effects in dv_effects.items()}
    sorted_dvs = sorted(dv_summary.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nMost biased performance metrics:")
    for i, (dv, avg_effect) in enumerate(sorted_dvs, 1):
        print(f"  {i}. {dv.replace('_', ' ').title()}: avg d = {avg_effect:.3f}")
    
    return effect_sizes






def run_full_tier1_analysis(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the full Tier 1 analysis pipeline:
    - 3-way ANOVA for demographic profile traits
    - Risk-benefit Pareto frontier analysis
    - Effect size computation for key findings
    
    Returns:
    A dictionary with all analysis results.
    """
    print("\nRunning 3-way factorial ANOVA on profile traits...\n")
    anova_results = factorial_analysis_3way_anova(merged_df)

    print("\nRunning Risk-Benefit Pareto Frontier analysis...\n")
    pareto_results = plot_risk_benefit_frontier(merged_df)

    performance_df = anova_results['performance_data']  # The 30-row DataFrame
    anova_stats = anova_results['anova_results']
    print("\nRunning effect size calculations for key findings...\n")
    effect_sizes = calculate_effect_sizes(merged_df, anova_results)

    return {
        "anova_results": anova_results,
        "pareto_results": pareto_results,
        "effect_sizes": effect_sizes
    }
