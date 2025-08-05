import os
import glob
import json
from collections import Counter
from itertools import combinations, product
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum

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
from statsmodels.formula.api import ols
import statsmodels.api as sm

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


from analysis_0 import *
from profiles.profile_message import get_profile_traits, PROFILE_META_SYSTEMATIC, ProfileMeta



def factorial_analysis_nway_anova(merged_df, group_keys=("gender", "ethnicity", "cognitive_style")):
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
    
    # Calculate metrics
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    performance_data = []
    
    for profile in profile_cols:
        if profile in merged_df.columns:
            
            accuracy = (merged_df[profile] == merged_df['true_label']).mean()
            
            profile_rescue = rescue_stats[rescue_stats['profile'] == profile]
            avg_rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
            avg_extra_error_rate = profile_rescue['extra_err_rate'].mean() if len(profile_rescue) > 0 else 0
            
            profile_bias = bias_patterns[bias_patterns['profile'] == profile]
            avg_bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
            avg_flip_rate = profile_bias['flip_rate'].mean() if len(profile_bias) > 0 else 0
            
            traits = profile_traits[profile]
            row = {
                'profile': profile,
                'accuracy': accuracy,
                'rescue_rate': avg_rescue_rate,
                'extra_error_rate': avg_extra_error_rate,
                'bias_magnitude': avg_bias_magnitude,
                'flip_rate': avg_flip_rate
            }
            row.update(traits)
            performance_data.append(row)
    

    performance_df = pd.DataFrame(performance_data)
    
    # ========================================================================
    # ANOVA TESTS FOR EACH DEPENDENT VARIABLE
    # ========================================================================
    
    results = {}
    dependent_vars = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude', 'flip_rate']
    
    for dv in dependent_vars:
        print(f"\n{'='*60}")
        print(f"FACTORIAL ANOVA: {dv.upper()}")
        print(f"{'='*60}")

        dv_results = {'dependent_variable': dv}
        

        for trait in group_keys:
            levels = performance_df[trait].dropna().unique()
            groups = [performance_df[performance_df[trait] == val][dv].values for val in levels]
            if len(groups) == 2:
                g1, g2 = levels
                if len(groups[0]) > 0 and len(groups[1]) > 0:
                    t_stat, p_val = ttest_ind(groups[0], groups[1])
                    dv_results[f"{trait}_main_effect"] = {
                        'group1': g1,
                        'group2': g2,
                        'group1_mean': np.mean(groups[0]),
                        'group2_mean': np.mean(groups[1]),
                        'difference': np.mean(groups[1]) - np.mean(groups[0]),
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'effect_size_d': (np.mean(groups[1]) - np.mean(groups[0])) / np.sqrt((np.var(groups[0]) + np.var(groups[1])) / 2)
                    }
                    print(f"{trait.upper()} MAIN EFFECT ({g1} vs {g2}):")
                    print(f"  {g1} mean: {np.mean(groups[0]):.4f}")
                    print(f"  {g2} mean: {np.mean(groups[1]):.4f}")
                    print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
                    print(f"  Cohen's d: {dv_results[f'{trait}_main_effect']['effect_size_d']:.3f}")
            elif len(groups) > 2 and all(len(g) > 0 for g in groups):
                f_stat, p_val = f_oneway(*groups)
                means = {str(val): np.mean(performance_df[performance_df[trait] == val][dv].values) for val in levels}
                dv_results[f"{trait}_main_effect"] = {
                    'means': means,
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
                print(f"{trait.upper()} MAIN EFFECT:")
                for val, m in means.items():
                    print(f"  {val} mean: {m:.4f}")
                print(f"  F-statistic: {f_stat:.3f}")
                print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")

        print(f"\nINTERACTION EFFECTS ({group_keys[0]} x {group_keys[1]}):")
        levels1 = performance_df[group_keys[0]].dropna().unique()
        levels2 = performance_df[group_keys[1]].dropna().unique()
               
        interaction_means = {}
        for val1, val2 in product(levels1, levels2):
            subset = performance_df[
                (performance_df[group_keys[0]] == val1) &
                (performance_df[group_keys[1]] == val2)
            ][dv].values
            if len(subset) > 0:
                key = f"{val1}_{val2}"
                interaction_means[key] = np.mean(subset)
                print(f"  {val1} x {val2}: {np.mean(subset):.4f}")
   
        dv_results["interaction_means"] = interaction_means

        p_val_interaction = ols_interaction_pvalue(performance_df, dv, group_keys[0], group_keys[1])
        if p_val_interaction is not None:
            print(f"  → OLS p-value for {group_keys[0]} x {group_keys[1]}: {p_val_interaction:.4f}")
            dv_results[f"{group_keys[0]}_x_{group_keys[1]}_interaction_effect"] = {
                'p_value': p_val_interaction,
                'significant': p_val_interaction < 0.05,
                'method': 'OLS'
            }
        else:
            print(f"  → OLS p-value not available for {group_keys[0]} x {group_keys[1]}")

        results[dv] = dv_results
            

    print(f"\n{'='*60}")
    print("FACTORIAL ANOVA SUMMARY")
    print(f"{'='*60}")

    significant_effects = []
    for dv, dv_results in results.items():
        print(f"\n{dv.upper()}:")
        for key, res in dv_results.items():
            if isinstance(res, dict) and res.get("significant", False):
                print(f"  ✓ {key}: p={res['p_value']:.4f}")
                significant_effects.append(f"{dv}_{key}")
            elif isinstance(res, dict):
                if "p_value" in res:
                    print(f"    {key}: p={res['p_value']:.4f}")
                else:
                    formatted_res = {k: f"{v:.4g}" if isinstance(v, float) else v for k, v in res.items()}
                    print(f"    {key}: {formatted_res}")

    print(f"\nTotal significant effects found: {len(significant_effects)}")

    return {
        'performance_data': performance_df,
        'anova_results': results,
        'significant_effects': significant_effects,
        'profile_traits': profile_traits
    }


def plot_risk_benefit_frontier(merged_df, group_keys=("gender", "ethnicity", "cognitive_style"), figsize=(12, 8)):
    """
    Risk-Benefit Frontier Analysis

    Plots each profile on:
    - X-axis: Extra error rate (risk)
    - Y-axis: Rescue rate (benefit)

    Identifies Pareto frontier of "safely bold" profiles that maximize
    rescue while minimizing extra errors.
    """

    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")

    profile_performance = rescue_stats.groupby('profile').agg({
        'rescue_rate': 'mean',
        'extra_err_rate': 'mean', 
        'rescued': 'sum',
        'extra_errors': 'sum'
    }).reset_index()

    profile_traits = {
        profile: get_profile_traits(profile, group_keys=group_keys)
        for profile in profile_performance["profile"]
    }

    for trait_name in group_keys:
        profile_performance[trait_name] = profile_performance["profile"].map(
            lambda p: profile_traits.get(p, {}).get(trait_name, "Unknown")
        )

    profile_performance["intersectional"] = profile_performance.apply(
        lambda row: "_".join(str(row[k]) for k in group_keys if pd.notnull(row[k])), axis=1
    )


    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Risk-Benefit Frontier Analysis: Rescue Rate vs Extra Error Rate', fontsize=16, fontweight='bold')

    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i, trait in enumerate(group_keys[:3]):
        ax = axes[i//2, i%2]
        values = profile_performance[trait].unique()
        for j, val in enumerate(values):
            subset = profile_performance[profile_performance[trait] == val]
            if len(subset) > 0:
                ax.scatter(subset['extra_err_rate'], subset['rescue_rate'], 
                           c=default_colors[j % len(default_colors)], label=val, alpha=0.7, s=60)
        ax.set_xlabel('Extra Error Rate (Risk)')
        ax.set_ylabel('Rescue Rate (Benefit)')
        ax.set_title(f'Risk-Benefit by {trait.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)


    # Last plot: Pareto Frontier identification
    ax = axes[1, 1]
    ax.scatter(profile_performance['extra_err_rate'], profile_performance['rescue_rate'],
               c='lightblue', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    points = profile_performance[['extra_err_rate', 'rescue_rate']].values
    pareto_mask = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[0] <= point[0] and other_point[1] >= point[1] and 
                    (other_point[0] < point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True

    pareto_points = profile_performance[pareto_mask]
    ax.scatter(pareto_points['extra_err_rate'], pareto_points['rescue_rate'],
               c='red', s=100, marker='*', label='Pareto Optimal', zorder=5)

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

    print("\n" + "="*60)
    print("PARETO FRONTIER ANALYSIS")
    print("="*60)
    print("\nPareto Optimal Profiles (Safely Bold):")
    for idx, row in pareto_points.iterrows():
        traits = profile_traits.get(row['profile'], {})
        trait_str = ", ".join(f"{k}: {traits.get(k, 'Unknown')}" for k in group_keys)
        print(f"  {row['profile']}: {trait_str}")
        print(f"    Rescue Rate: {row['rescue_rate']:.3f}, Extra Error Rate: {row['extra_err_rate']:.3f}")

    print(f"\nRisk-Benefit Statistics:")
    print(f"  Average rescue rate: {profile_performance['rescue_rate'].mean():.3f}")
    print(f"  Average extra error rate: {profile_performance['extra_err_rate'].mean():.3f}")
    print(f"  Best rescue rate: {profile_performance['rescue_rate'].max():.3f}")
    print(f"  Lowest extra error rate: {profile_performance['extra_err_rate'].min():.3f}")

    print(f"\nProfile Archetypes:")
    safest = profile_performance.loc[profile_performance['extra_err_rate'].idxmin()]
    most_beneficial = profile_performance.loc[profile_performance['rescue_rate'].idxmax()]

    safest_traits = profile_traits.get(safest['profile'], {})
    safest_str = ", ".join(f"{k}: {safest_traits.get(k, 'Unknown')}" for k in group_keys)
    print(f"  Safest Profile: {safest['profile']} ({safest_str})")
    print(f"    Extra Error Rate: {safest['extra_err_rate']:.3f}")

    beneficial_traits = profile_traits.get(most_beneficial['profile'], {})
    beneficial_str = ", ".join(f"{k}: {beneficial_traits.get(k, 'Unknown')}" for k in group_keys)
    print(f"  Most Beneficial Profile: {most_beneficial['profile']} ({beneficial_str})")
    print(f"    Rescue Rate: {most_beneficial['rescue_rate']:.3f}")


    return {
        'performance_data': profile_performance,
        'pareto_optimal': pareto_points,
        'profile_traits': profile_traits,
        'safest_profile': safest,
        'most_beneficial_profile': most_beneficial
    }





def calculate_effect_sizes(performance_df, anova_results, group_keys=("gender", "ethnicity", "cognitive_style")):
    """
    Calculate Cohen's d effect sizes directly from ANOVA performance data.

    Supports dynamic group comparisons based on group_keys.
    """
    
    effect_sizes = {}
    print("CALCULATING ANOVA-BASED EFFECT SIZES")
    print("=" * 60)

    def calculate_cohens_d(group1_vals, group2_vals, group1_name, group2_name):
        if len(group1_vals) == 0 or len(group2_vals) == 0:
            return None
        mean1, mean2 = np.mean(group1_vals), np.mean(group2_vals)
        var1, var2 = np.var(group1_vals, ddof=1), np.var(group2_vals, ddof=1)
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

    dependent_vars = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude']
    if 'disagreement_rate' in performance_df.columns:
        dependent_vars.append('disagreement_rate')
    elif 'flip_rate' in performance_df.columns:
        dependent_vars.append('flip_rate')

    print("\nMAIN EFFECTS FROM ANOVA")
    print("-" * 40)

    for dv in dependent_vars:
        if dv not in performance_df.columns:
            continue
        dv_results = anova_results.get(dv, {})
        for trait in group_keys:
            main_effect_key = f"{trait}_main_effect"
            if main_effect_key in dv_results and dv_results[main_effect_key].get("significant", False):
                levels = performance_df[trait].dropna().unique()
                for i, val1 in enumerate(levels):
                    for val2 in levels[i+1:]:
                        vals1 = performance_df[performance_df[trait] == val1][dv].values
                        vals2 = performance_df[performance_df[trait] == val2][dv].values
                        effect_size = calculate_cohens_d(vals1, vals2, val1.lower(), val2.lower())
                        if effect_size:
                            key = f"{dv}_{trait}_{val1.lower()}_vs_{val2.lower()}"
                            effect_sizes[key] = effect_size
                            print(f"  {dv} - {val1} vs {val2}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")

    print("\nINTERACTION EFFECTS")
    print("-" * 40)

    for dv in dependent_vars:
        if dv not in performance_df.columns:
            continue
        for i in range(len(group_keys)):
            for j in range(i+1, len(group_keys)):
                g1, g2 = group_keys[i], group_keys[j]
                g1_vals = performance_df[g1].dropna().unique()
                g2_vals = performance_df[g2].dropna().unique()
                for val1 in g1_vals:
                    for val2 in g2_vals:
                        group1_mask = (performance_df[g1] == val1) & (performance_df[g2] == val2)
                        for val3 in g1_vals:
                            for val4 in g2_vals:
                                if val1 == val3 and val2 == val4:
                                    continue
                                group2_mask = (performance_df[g1] == val3) & (performance_df[g2] == val4)
                                vals1 = performance_df[group1_mask][dv].values
                                vals2 = performance_df[group2_mask][dv].values
                                if len(vals1) > 0 and len(vals2) > 0:
                                    name1 = f"{val1}_{val2}".lower()
                                    name2 = f"{val3}_{val4}".lower()
                                    effect_size = calculate_cohens_d(vals1, vals2, name1, name2)
                                    if effect_size and abs(effect_size['cohens_d']) > 0.3:
                                        key = f"{dv}_interaction_{name1}_vs_{name2}"
                                        effect_sizes[key] = effect_size
                                        print(f"  {dv} - {name1} vs {name2}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")

    return effect_sizes



def ols_interaction_pvalue(df, dv, factor1, factor2):
    """
    Run OLS regression with interaction term and return the p-value for the interaction.

    Parameters:
    - df: DataFrame with data
    - dv: dependent variable (e.g. "accuracy")
    - factor1: first factor (e.g. "gender")
    - factor2: second factor (e.g. "ethnicity")

    Returns:
    - interaction_p: float or None
    """
    try:
        formula = f"{dv} ~ C({factor1}) * C({factor2})"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        interaction_term = f"C({factor1}):C({factor2})"
        p_value = anova_table.loc[interaction_term, "PR(>F)"]
        return p_value
    except Exception as e:
        print(f"⚠️ Interaction OLS failed: {e}")
        return None






def run_full_tier1_analysis(merged_df: pd.DataFrame, group_keys=("gender", "ethnicity", "cognitive_style")) -> Dict[str, Any]:
    """
    Run the full Tier 1 analysis pipeline:
    - N-way ANOVA for demographic profile traits
    - Risk-benefit Pareto frontier analysis
    - Effect size computation for key findings
    
    Parameters:
    - merged_df: Merged classification results with 'profile', 'true_label', and predicted labels.
    - group_keys: Traits to include in the group-level analysis (must exist in ProfileMeta)
    
    Returns:
    A dictionary with all analysis results.
    """
    print("\nRunning factorial ANOVA on profile traits...\n")
    anova_results = factorial_analysis_nway_anova(merged_df, group_keys=group_keys)

    print("\nRunning Risk-Benefit Pareto Frontier analysis...\n")
    pareto_results = plot_risk_benefit_frontier(merged_df, group_keys=group_keys)

    performance_df = anova_results['performance_data']
    print("\nRunning effect size calculations for key findings...\n")
    effect_sizes = calculate_effect_sizes(performance_df, anova_results, group_keys=group_keys)

    return {
        "anova_results": anova_results,
        "pareto_results": pareto_results,
        "effect_sizes": effect_sizes
    }
