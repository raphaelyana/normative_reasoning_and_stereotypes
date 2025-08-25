import os
import glob
import json
from collections import Counter
import itertools
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
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


from analysis_tools import get_available_traits, get_analysis_group_keys
from analysis_0 import *
from profiles.profile_sets import PERSON_SYSTEMATIC
from profiles.schema import PersonSet
from cases.cases_config import CaseConfig

def factorial_analysis_nway_anova(
    merged_df, 
    group_keys=("gender", "ethnicity", "age"),
    person_set: PersonSet = None
):
    """
    Configurable N-way ANOVA: Trait x Trait x ... -> Performance Metrics
    
    Tests which profile traits (gender, ethnicity, etc.) affect accuracy
    using proper factorial ANOVA with interaction effects.
    
    Parameters:
    - merged_df: DataFrame containing predictions and true labels
    - group_keys: tuple of metadata fields to extract and analyze
    - person_set: PersonSet object containing trait metadata

    Returns:
    - dict with performance data, anova results, and significant effects
    """

    def norm_val(v):
        """Normalize values to lowercase string for ANOVA."""
        if v == "Unknown":
            return "unknown"
        elif isinstance(v, (int, float)):
            return str(v)
        else:
            return str(v).lower()

    def extract_traits(profile_id):
        """Get normalized trait values using existing PersonSet.get_traits method."""
        if person_set is None:
            return {k: "unknown" for k in group_keys}
        
        traits = person_set.get_traits(profile_id, group_keys)
        return {k: norm_val(traits[k]) for k in group_keys}
    
    print(f"=== FACTORIAL ANOVA ANALYSIS ===")
    print(f"Group keys: {group_keys}")
    
    # Get profile columns - same logic as plotting function
    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
    print(f"Found {len(profile_cols)} profile columns")
    
    # Extract traits for each profile - same logic as plotting function
    profile_traits = {}
    for p in profile_cols:
        traits = extract_traits(p)
        profile_traits[p] = traits
    
    # Calculate primary performance metric: accuracy
    performance_data = []
    for profile in profile_cols:
        if profile not in merged_df.columns:
            print(f"WARNING: Profile {profile} not found in merged_df columns")
            continue
            
        # Calculate accuracy with error checking
        predictions = merged_df[profile]
        true_labels = merged_df['true_label']
        
        # Check for missing values
        if predictions.isna().any() or true_labels.isna().any():
            print(f"WARNING: Missing values found in {profile}")
            accuracy = (predictions == true_labels).mean()
        else:
            accuracy = (predictions == true_labels).mean()
        
        # Check for invalid accuracy values
        if pd.isna(accuracy) or np.isinf(accuracy):
            print(f"ERROR: Invalid accuracy for {profile}: {accuracy}")
            accuracy = 0.0  # Set to default value
            
        # Build row with traits
        row = {
            'profile': profile,
            'accuracy': accuracy,
        }
        row.update(profile_traits[profile])
        performance_data.append(row)

    performance_df = pd.DataFrame(performance_data)
    
    # ========================================================================
    # DATA VALIDATION AND CLEANING
    # ========================================================================
    
    print(f"\nPerformance data shape: {performance_df.shape}")
    print(f"Traits found: {[col for col in performance_df.columns if col in group_keys]}")
    
    # Check for problematic values
    print(f"\n=== DATA VALIDATION ===")
    print(f"Accuracy range: {performance_df['accuracy'].min():.6f} - {performance_df['accuracy'].max():.6f}")
    print(f"Missing values in accuracy: {performance_df['accuracy'].isna().sum()}")
    print(f"Infinite values in accuracy: {np.isinf(performance_df['accuracy']).sum()}")
    
    # Check for zero variance groups
    zero_var_groups = []
    for trait in group_keys:
        if trait in performance_df.columns:
            group_vars = performance_df.groupby(trait)['accuracy'].var()
            zero_var = group_vars[group_vars == 0.0]
            if len(zero_var) > 0:
                zero_var_groups.append((trait, zero_var.index.tolist()))
                print(f"WARNING: Zero variance in {trait} groups: {zero_var.index.tolist()}")
    
    # Clean data: remove rows with invalid accuracy values
    valid_mask = ~(pd.isna(performance_df['accuracy']) | np.isinf(performance_df['accuracy']))
    if not valid_mask.all():
        print(f"Removing {(~valid_mask).sum()} rows with invalid accuracy values")
        performance_df = performance_df[valid_mask].reset_index(drop=True)
    
    # Display summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    for trait in group_keys:
        if trait in performance_df.columns:
            print(f"{trait.upper()}:")
            summary = performance_df.groupby(trait)['accuracy'].agg(['count', 'mean', 'std'])
            print(summary)
            print()
    
    # ========================================================================
    # FACTORIAL ANOVA ANALYSIS
    # ========================================================================
    
    results = {}
    dependent_var = 'accuracy'  # Focus on accuracy as main metric
    
    print(f"\n{'='*60}")
    print(f"FACTORIAL ANOVA: {dependent_var.upper()}")
    print(f"{'='*60}")
    
    # Check if we have enough data for ANOVA
    valid_traits = []
    for trait in group_keys:
        if trait in performance_df.columns:
            unique_vals = performance_df[trait].dropna().unique()
            if len(unique_vals) >= 2:
                valid_traits.append(trait)
                
                # Check sample sizes per group
                group_sizes = performance_df.groupby(trait).size()
                min_size = group_sizes.min()
                max_size = group_sizes.max()
                
                print(f"{trait}: {len(unique_vals)} levels: {list(unique_vals)}")
                print(f"  Sample sizes: min={min_size}, max={max_size}, per group: {dict(group_sizes)}")
                
                # Check variance within groups
                group_vars = performance_df.groupby(trait)[dependent_var].var()
                min_var = group_vars.min()
                max_var = group_vars.max()
                
                if min_var < 1e-6:
                    print(f"  WARNING: Very low variance detected (min={min_var:.2e}). Post-hoc tests may be unreliable.")
                
                # Check for extreme outliers
                q1 = performance_df.groupby(trait)[dependent_var].quantile(0.25)
                q3 = performance_df.groupby(trait)[dependent_var].quantile(0.75)
                iqr = q3 - q1
                outliers = performance_df.groupby(trait).apply(
                    lambda x: ((x[dependent_var] < (q1[x.name] - 1.5 * iqr[x.name])) | 
                              (x[dependent_var] > (q3[x.name] + 1.5 * iqr[x.name]))).sum()
                )
                if outliers.sum() > 0:
                    print(f"  INFO: Outliers detected per group: {dict(outliers[outliers > 0])}")
                
            else:
                print(f"WARNING: {trait} has insufficient levels ({len(unique_vals)}) for ANOVA")
    
    if len(valid_traits) == 0:
        print("ERROR: No valid traits found for ANOVA analysis")
        return {
            'performance_data': performance_df,
            'anova_results': {},
            'significant_effects': [],
            'profile_traits': profile_traits,
            'error': 'No valid traits for analysis'
        }
    
    # ========================================================================
    # PROPER FACTORIAL ANOVA USING STATSMODELS
    # ========================================================================
    
    anova_results = {}
    
    try:
        # Build formula for factorial ANOVA
        # Create formula with all main effects and interactions
        if len(valid_traits) == 1:
            formula = f"{dependent_var} ~ C({valid_traits[0]})"
        elif len(valid_traits) == 2:
            t1, t2 = valid_traits[0], valid_traits[1]
            formula = f"{dependent_var} ~ C({t1}) + C({t2}) + C({t1}):C({t2})"
        elif len(valid_traits) == 3:
            t1, t2, t3 = valid_traits[0], valid_traits[1], valid_traits[2]
            # Check if we have enough degrees of freedom for full model
            n_params = (len(performance_df[t1].unique()) - 1) + \
                      (len(performance_df[t2].unique()) - 1) + \
                      (len(performance_df[t3].unique()) - 1) + \
                      (len(performance_df[t1].unique()) - 1) * (len(performance_df[t2].unique()) - 1) + \
                      (len(performance_df[t1].unique()) - 1) * (len(performance_df[t3].unique()) - 1) + \
                      (len(performance_df[t2].unique()) - 1) * (len(performance_df[t3].unique()) - 1) + \
                      (len(performance_df[t1].unique()) - 1) * (len(performance_df[t2].unique()) - 1) * (len(performance_df[t3].unique()) - 1)
            
            df_resid = len(performance_df) - n_params - 1
            print(f"Model complexity check: {n_params} parameters, {len(performance_df)} observations, {df_resid} residual df")
            
            if df_resid <= 5:  # Too few degrees of freedom
                print(f"WARNING: Insufficient degrees of freedom ({df_resid}). Using simplified model.")
                formula = f"{dependent_var} ~ C({t1}) + C({t2}) + C({t3}) + C({t1}):C({t2}) + C({t1}):C({t3}) + C({t2}):C({t3})"
                print("Removing 3-way interaction to preserve degrees of freedom")
            else:
                formula = f"{dependent_var} ~ C({t1}) + C({t2}) + C({t3}) + C({t1}):C({t2}) + C({t1}):C({t3}) + C({t2}):C({t3}) + C({t1}):C({t2}):C({t3})"
        else:
            # For more than 3 traits, just do main effects and 2-way interactions
            main_effects = " + ".join([f"C({trait})" for trait in valid_traits])
            interactions = " + ".join([f"C({t1}):C({t2})" for t1, t2 in itertools.combinations(valid_traits, 2)])
            formula = f"{dependent_var} ~ {main_effects} + {interactions}"
        
        print(f"ANOVA Formula: {formula}")
        
        # Additional data validation before fitting
        print(f"\nPre-ANOVA validation:")
        print(f"  Data shape: {performance_df.shape}")
        print(f"  Accuracy stats: mean={performance_df['accuracy'].mean():.6f}, std={performance_df['accuracy'].std():.6f}")
        
        # Check if we have sufficient variation for ANOVA
        overall_var = performance_df['accuracy'].var()
        if overall_var < 1e-10:
            raise ValueError(f"Insufficient variance in accuracy data (var={overall_var:.2e})")
        
        # Fit the model with fallback options
        model = None
        for attempt in range(3):
            try:
                print(f"Attempting model fit (attempt {attempt + 1}): {formula}")
                model = ols(formula, data=performance_df).fit()
                anova_table = anova_lm(model, typ=2)  # Type II ANOVA
                break
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"  Model fit failed: {e}")
                if attempt == 0 and len(valid_traits) == 3:
                    # Remove 3-way interaction
                    t1, t2, t3 = valid_traits[0], valid_traits[1], valid_traits[2]
                    formula = f"{dependent_var} ~ C({t1}) + C({t2}) + C({t3}) + C({t1}):C({t2}) + C({t1}):C({t3}) + C({t2}):C({t3})"
                    print(f"  Trying without 3-way interaction: {formula}")
                elif attempt == 1:
                    # Remove all interactions, main effects only
                    main_effects = " + ".join([f"C({trait})" for trait in valid_traits])
                    formula = f"{dependent_var} ~ {main_effects}"
                    print(f"  Trying main effects only: {formula}")
                else:
                    print("  All model fitting attempts failed")
                    raise e
        
        if model is None:
            raise ValueError("Could not fit any ANOVA model")
        
        print(f"\nANOVA Results:")
        print(anova_table)
        
        # Store results
        anova_results['formula'] = formula
        anova_results['anova_table'] = anova_table
        anova_results['model_summary'] = model.summary()
        anova_results['r_squared'] = model.rsquared
        anova_results['adj_r_squared'] = model.rsquared_adj
        
        # Extract significant effects
        significant_effects = []
        for effect in anova_table.index:
            if effect != 'Residual':
                p_value = anova_table.loc[effect, 'PR(>F)']
                if p_value < 0.05:
                    significant_effects.append({
                        'effect': effect,
                        'f_statistic': anova_table.loc[effect, 'F'],
                        'p_value': p_value,
                        'eta_squared': anova_table.loc[effect, 'sum_sq'] / anova_table['sum_sq'].sum()
                    })
        
        anova_results['significant_effects'] = significant_effects
        
        print(f"\nSignificant Effects (p < 0.05):")
        for effect in significant_effects:
            print(f"  {effect['effect']}: F={effect['f_statistic']:.3f}, p={effect['p_value']:.4f}, η²={effect['eta_squared']:.3f}")
        
    except Exception as e:
        print(f"ERROR in ANOVA analysis: {str(e)}")
        anova_results['error'] = str(e)
        significant_effects = []
    
    # ========================================================================
    # POST-HOC ANALYSIS FOR SIGNIFICANT MAIN EFFECTS
    # ========================================================================
    
    posthoc_results = {}
    
    if 'significant_effects' in anova_results:
        for effect_info in anova_results['significant_effects']:
            effect_name = effect_info['effect']
            
            # Check if it's a main effect (single trait, not interaction)
            if ':' not in effect_name and effect_name.startswith('C(') and effect_name.endswith(')'):
                trait = effect_name[2:-1]  # Remove 'C(' and ')'
                
                print(f"\nPost-hoc analysis for {trait}:")
                
                try:
                    # Use Bonferroni correction instead of Tukey for more robust results
                    from scipy.stats import ttest_ind
                    from itertools import combinations
                    
                    groups = performance_df.groupby(trait)[dependent_var].apply(list)
                    group_names = list(groups.index)
                    
                    # Pairwise t-tests with Bonferroni correction
                    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
                    alpha_corrected = 0.05 / n_comparisons
                    
                    print(f"  Pairwise comparisons with Bonferroni correction (α = {alpha_corrected:.4f}):")
                    
                    comparisons = []
                    for g1, g2 in combinations(group_names, 2):
                        data1 = groups[g1]
                        data2 = groups[g2]
                        
                        if len(data1) > 1 and len(data2) > 1:
                            t_stat, p_val = ttest_ind(data1, data2)
                            mean_diff = np.mean(data2) - np.mean(data1)
                            significant = p_val < alpha_corrected
                            
                            comparisons.append({
                                'group1': g1,
                                'group2': g2,
                                'mean1': np.mean(data1),
                                'mean2': np.mean(data2),
                                'mean_diff': mean_diff,
                                't_stat': t_stat,
                                'p_value': p_val,
                                'significant': significant
                            })
                            
                            sig_marker = "***" if significant else ""
                            print(f"    {g1} vs {g2}: t={t_stat:.3f}, p={p_val:.4f} {sig_marker}")
                    
                    posthoc_results[trait] = {
                        'method': 'bonferroni',
                        'alpha_corrected': alpha_corrected,
                        'comparisons': comparisons
                    }
                    
                except Exception as e:
                    print(f"  ERROR in post-hoc analysis for {trait}: {str(e)}")
                    
                    # Fallback: try Tukey with warnings suppressed
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            tukey_result = pairwise_tukeyhsd(
                                performance_df[dependent_var], 
                                performance_df[trait], 
                                alpha=0.05
                            )
                            print(f"  Tukey HSD (with warnings suppressed):")
                            print(tukey_result)
                            posthoc_results[trait] = {
                                'method': 'tukey_suppressed',
                                'tukey_summary': str(tukey_result)
                            }
                    except Exception as e2:
                        print(f"  Both Bonferroni and Tukey failed: {str(e2)}")
                        posthoc_results[trait] = {'error': str(e)}
    
    # ========================================================================
    # EFFECT SIZE CALCULATIONS
    # ========================================================================
    
    effect_sizes = {}
    
    for trait in valid_traits:
        groups = performance_df.groupby(trait)[dependent_var].apply(list)
        if len(groups) == 2:
            # Cohen's d for two groups
            group_names = list(groups.index)
            g1, g2 = groups.iloc[0], groups.iloc[1]
            
            if len(g1) > 0 and len(g2) > 0:
                pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) + 
                                    (len(g2) - 1) * np.var(g2, ddof=1)) / 
                                   (len(g1) + len(g2) - 2))
                cohens_d = (np.mean(g2) - np.mean(g1)) / pooled_std
                
                effect_sizes[trait] = {
                    'type': 'cohens_d',
                    'value': cohens_d,
                    'group1': group_names[0],
                    'group2': group_names[1],
                    'group1_mean': np.mean(g1),
                    'group2_mean': np.mean(g2),
                    'interpretation': ('Small' if abs(cohens_d) < 0.5 else 
                                     'Medium' if abs(cohens_d) < 0.8 else 'Large')
                }
        
        elif len(groups) > 2:
            # Eta-squared is already calculated in ANOVA table
            if 'anova_table' in anova_results:
                effect_name = f"C({trait})"
                if effect_name in anova_results['anova_table'].index:
                    eta_sq = (anova_results['anova_table'].loc[effect_name, 'sum_sq'] / 
                             anova_results['anova_table']['sum_sq'].sum())
                    effect_sizes[trait] = {
                        'type': 'eta_squared',
                        'value': eta_sq,
                        'interpretation': ('Small' if eta_sq < 0.06 else 
                                         'Medium' if eta_sq < 0.14 else 'Large')
                    }
    
    print(f"\n=== EFFECT SIZES ===")
    for trait, effect_info in effect_sizes.items():
        print(f"{trait}: {effect_info['type']} = {effect_info['value']:.3f} ({effect_info['interpretation']})")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"FACTORIAL ANOVA SUMMARY")
    print(f"{'='*60}")
    print(f"Total profiles analyzed: {len(performance_df)}")
    print(f"Traits analyzed: {', '.join(valid_traits)}")
    
    if 'significant_effects' in anova_results:
        print(f"Significant effects found: {len(anova_results['significant_effects'])}")
        if 'r_squared' in anova_results:
            print(f"Model R²: {anova_results['r_squared']:.3f}")
            print(f"Adjusted R²: {anova_results['adj_r_squared']:.3f}")
    else:
        print("No ANOVA results available")
    
    return {
        'performance_data': performance_df,
        'anova_results': anova_results,
        'posthoc_results': posthoc_results,
        'effect_sizes': effect_sizes,
        'significant_effects': anova_results.get('significant_effects', []),
        'profile_traits': profile_traits,
        'valid_traits': valid_traits
    }




def plot_risk_benefit_frontier(
    merged_df, 
    group_keys=("gender", "ethnicity", "age"), 
    category_cols=None,
    figsize=(10, 6), 
    person_set: PersonSet = None
):
    """
    Risk-Benefit Frontier Analysis - Adapted for PersonSet
    
    Creates individual plots for each trait showing:
    - X-axis: Extra error rate (risk)  
    - Y-axis: Rescue rate (benefit)
    
    Plus a final Pareto frontier plot identifying "safely bold" profiles.
    """
    
    def norm_val(v):
        """Normalize values for consistent display."""
        if v == "Unknown":
            return "unknown"
        elif isinstance(v, (int, float)):
            return str(v)
        else:
            return str(v).lower()

    def extract_traits(profile_id):
        """Get normalized trait values using existing PersonSet.get_traits method."""
        if person_set is None:
            return {k: "unknown" for k in group_keys}
        
        traits = person_set.get_traits(profile_id, group_keys)
        return {k: norm_val(traits.get(k, "Unknown")) for k in group_keys}
    
    def get_demographic_info(profile_name: str, person_set) -> str:
        """
        Return a demographic signature like 'man_white_20' or 'woman_black_25'.
        Works whether person_set.get_traits returns a dict or a PersonMeta.
        """
        traits = person_set.get_traits(profile_name, group_keys)
        
        def read(tr, key):
            v = tr.get(key) if isinstance(tr, dict) else getattr(tr, key, None)
            if hasattr(v, 'value'):  # Enum
                v = v.value
            return None if v is None else str(v).lower()
        
        parts = []
        for k in ("gender", "ethnicity"):
            v = read(traits, k)
            if v and v != "unknown":
                parts.append(v)
        
        cs = read(traits, "cognitive_style")
        if cs and cs != "unknown":
            parts.append(cs)
        else:
            # Check for age and other extra fields
            for f in group_keys:
                if f not in ("gender", "ethnicity", "cognitive_style"):
                    v = read(traits, f)
                    if v and v != "unknown":
                        parts.append(v)
        
        return "_".join(parts) if parts else "unknown"
    
    print("=== RISK-BENEFIT FRONTIER ANALYSIS ===")
    print(f"Group keys: {group_keys}")

    if category_cols is None:
        print("WARNING: category_cols not provided — defaulting to ['stereotype_type']")
        category_cols = ["stereotype_type"]

    print("\n\n=== RESCUE STATISTICS BY CATEGORY ===")
    rescue_stats_list = []
    from analysis_tools import guarded_labelspace_analysis
    for cat_col in category_cols:
        if cat_col in merged_df.columns:
            rescue_stats = guarded_labelspace_analysis(
                rescue_stats_by_category,
                merged_df,
                category_col=cat_col,
                person_set=person_set
            )
            rescue_stats["category_col"] = cat_col
            rescue_stats_list.append(rescue_stats)

    if not rescue_stats_list:
        raise ValueError("No valid category columns found in merged_df.")

    rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
    
    # Aggregate performance by profile
    profile_performance = rescue_stats.groupby('profile').agg({
        'rescue_rate': 'mean',
        'extra_err_rate': 'mean', 
        'rescued': 'sum',
        'extra_errors': 'sum'
    }).reset_index()
    
    print(f"Found {len(profile_performance)} profiles for risk-benefit analysis")
    
    # Extract traits for each profile using the same logic as ANOVA
    profile_traits = {}
    for profile in profile_performance["profile"]:
        # Clean the profile name to get the base persona ID
        
        traits = person_set.get_traits(profile)
        profile_traits[profile] = traits
        
        # Add traits to performance dataframe
        for trait_name in group_keys:
            if trait_name not in profile_performance.columns:
                profile_performance[trait_name] = None
    
    # Map traits to performance dataframe
    for trait_name in group_keys:
        profile_performance[trait_name] = profile_performance["profile"].map(
            lambda p: profile_traits.get(p, {}).get(trait_name, "unknown")
        )
    
    # Create intersectional identifier
    profile_performance["intersectional"] = profile_performance.apply(
        lambda row: "_".join(str(row[k]) for k in group_keys if pd.notnull(row[k]) and row[k] != "unknown"), 
        axis=1
    )
    
    print(f"Sample profile traits extracted:")
    for i, (profile, traits) in enumerate(list(profile_traits.items())[:3]):
        print(f"  {profile}: {traits}")
    
    # ========================================================================
    # INDIVIDUAL PLOTS FOR EACH TRAIT
    # ========================================================================
    
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for trait in group_keys:
        print(f"\nCreating risk-benefit plot for {trait}...")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        values = profile_performance[trait].unique()
        valid_values = [v for v in values if v != "unknown"]
        
        if len(valid_values) == 0:
            print(f"WARNING: No valid values found for {trait}, skipping plot")
            plt.close(fig)
            continue
        
        print(f"  {trait} levels: {valid_values}")
        
        for j, val in enumerate(valid_values):
            subset = profile_performance[profile_performance[trait] == val]
            if len(subset) > 0:
                ax.scatter(subset['extra_err_rate'], subset['rescue_rate'], 
                          c=default_colors[j % len(default_colors)], 
                          label=f"{val} (n={len(subset)})", 
                          alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Extra Error Rate (Risk)', fontsize=12)
        ax.set_ylabel('Rescue Rate (Benefit)', fontsize=12)
        ax.set_title(f'Risk-Benefit Analysis by {trait.capitalize()}', fontsize=14, fontweight='bold')
        ax.legend(title=trait.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # PARETO FRONTIER ANALYSIS
    # ========================================================================
    
    print(f"\nCreating Pareto frontier analysis...")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot all profiles
    ax.scatter(profile_performance['extra_err_rate'], profile_performance['rescue_rate'],
               c='lightblue', alpha=0.6, s=80, edgecolors='black', linewidth=0.5,
               label='All Profiles')
    
    # Calculate Pareto frontier
    points = profile_performance[['extra_err_rate', 'rescue_rate']].values
    pareto_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                # A point is dominated if another point has lower risk AND higher benefit
                if (other_point[0] <= point[0] and other_point[1] >= point[1] and 
                    (other_point[0] < point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
        if not dominated:
            pareto_mask[i] = True
    
    pareto_points = profile_performance[pareto_mask].copy()
    
    # Plot Pareto optimal points
    ax.scatter(pareto_points['extra_err_rate'], pareto_points['rescue_rate'],
               c='red', s=150, marker='*', label='Pareto Optimal', zorder=5, edgecolors='darkred')
    
    # Annotate Pareto optimal points with demographic info
    for idx, row in pareto_points.iterrows():
        if person_set:
            demo_label = get_demographic_info(row['profile'], person_set)
        else:
            demo_label = row['profile'].replace('profile', 'P')
        
        ax.annotate(demo_label, 
                    (row['extra_err_rate'], row['rescue_rate']),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='red', alpha=0.8, linewidth=1.5))
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_sorted = pareto_points.sort_values('extra_err_rate')
        ax.plot(pareto_sorted['extra_err_rate'], pareto_sorted['rescue_rate'], 
                'r--', alpha=0.7, linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel('Extra Error Rate (Risk)', fontsize=12)
    ax.set_ylabel('Rescue Rate (Benefit)', fontsize=12)
    ax.set_title('Pareto Frontier: "Safely Bold" Profiles', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add optimal regions
    ax.axhspan(ax.get_ylim()[0], profile_performance['rescue_rate'].mean(), 
               alpha=0.1, color='red', label='Below Average Benefit')
    ax.axvspan(profile_performance['extra_err_rate'].mean(), ax.get_xlim()[1], 
               alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # DETAILED ANALYSIS RESULTS
    # ========================================================================
    
    print("\n" + "="*60)
    print("PARETO FRONTIER ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nPareto Optimal Profiles ({len(pareto_points)} found):")
    for idx, row in pareto_points.iterrows():
        if person_set:
            demo_info = get_demographic_info(row['profile'], person_set)
            print(f"  {demo_info} ({row['profile']})")
        else:
            traits = profile_traits.get(row['profile'], {})
            trait_str = ", ".join(f"{k}: {traits.get(k, 'unknown')}" for k in group_keys)
            print(f"  {row['profile']}: {trait_str}")
        print(f"    Rescue Rate: {row['rescue_rate']:.3f}, Extra Error Rate: {row['extra_err_rate']:.3f}")
        print(f"    Total Rescued: {row['rescued']}, Total Extra Errors: {row['extra_errors']}")
    
    print(f"\nRisk-Benefit Statistics:")
    print(f"  Total profiles analyzed: {len(profile_performance)}")
    print(f"  Average rescue rate: {profile_performance['rescue_rate'].mean():.3f}")
    print(f"  Average extra error rate: {profile_performance['extra_err_rate'].mean():.3f}")
    print(f"  Best rescue rate: {profile_performance['rescue_rate'].max():.3f}")
    print(f"  Lowest extra error rate: {profile_performance['extra_err_rate'].min():.3f}")
    
    # Profile archetypes
    safest = profile_performance.loc[profile_performance['extra_err_rate'].idxmin()]
    most_beneficial = profile_performance.loc[profile_performance['rescue_rate'].idxmax()]
    
    print(f"\nProfile Archetypes:")
    if person_set:
        safest_demo = get_demographic_info(safest['profile'], person_set)
        beneficial_demo = get_demographic_info(most_beneficial['profile'], person_set)
        print(f"  Safest Profile: {safest_demo} ({safest['profile']})")
        print(f"    Extra Error Rate: {safest['extra_err_rate']:.3f}")
        print(f"  Most Beneficial Profile: {beneficial_demo} ({most_beneficial['profile']})")
        print(f"    Rescue Rate: {most_beneficial['rescue_rate']:.3f}")
    else:
        safest_traits = profile_traits.get(safest['profile'], {})
        safest_str = ", ".join(f"{k}: {safest_traits.get(k, 'unknown')}" for k in group_keys)
        print(f"  Safest Profile: {safest['profile']} ({safest_str})")
        print(f"    Extra Error Rate: {safest['extra_err_rate']:.3f}")
        
        beneficial_traits = profile_traits.get(most_beneficial['profile'], {})
        beneficial_str = ", ".join(f"{k}: {beneficial_traits.get(k, 'unknown')}" for k in group_keys)
        print(f"  Most Beneficial Profile: {most_beneficial['profile']} ({beneficial_str})")
        print(f"    Rescue Rate: {most_beneficial['rescue_rate']:.3f}")
    
    # Demographic analysis of Pareto optimal profiles
    if len(pareto_points) > 0:
        print(f"\nDemographic Composition of Pareto Optimal Profiles:")
        for trait in group_keys:
            trait_counts = pareto_points[trait].value_counts()
            print(f"  {trait.capitalize()}:")
            for value, count in trait_counts.items():
                if value != "unknown":
                    pct = (count / len(pareto_points)) * 100
                    print(f"    {value}: {count} ({pct:.1f}%)")
    
    return {
        'performance_data': profile_performance,
        'pareto_optimal': pareto_points,
        'profile_traits': profile_traits,
        'safest_profile': safest,
        'most_beneficial_profile': most_beneficial,
        'group_keys': group_keys
    }





def calculate_effect_sizes(performance_df, anova_results, group_keys=("gender", "ethnicity", "age")):
    """
    Calculate Cohen's d effect sizes directly from ANOVA performance data.
    Updated to work with the new PersonSet structure and flexible group_keys.
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

    # Focus on accuracy as the primary metric from ANOVA results
    dependent_vars = ['accuracy']
    
    # Add other metrics if they exist in the performance dataframe
    optional_vars = ['rescue_rate', 'extra_error_rate', 'bias_magnitude', 'disagreement_rate', 'mislabelling_rate']
    for var in optional_vars:
        if var in performance_df.columns:
            dependent_vars.append(var)

    print("\nMAIN EFFECTS FROM ANOVA")
    print("-" * 40)

    # Process significant main effects from ANOVA results
    if 'significant_effects' in anova_results:
        for effect_info in anova_results['significant_effects']:
            effect_name = effect_info['effect']
            
            # Check if it's a main effect (single trait, not interaction)
            if ':' not in effect_name and effect_name.startswith('C(') and effect_name.endswith(')'):
                trait = effect_name[2:-1]  # Remove 'C(' and ')'
                
                if trait in group_keys and trait in performance_df.columns:
                    print(f"\nProcessing significant main effect: {trait}")
                    levels = performance_df[trait].dropna().unique()
                    
                    for i, val1 in enumerate(levels):
                        for val2 in levels[i+1:]:
                            vals1 = performance_df[performance_df[trait] == val1]['accuracy'].values
                            vals2 = performance_df[performance_df[trait] == val2]['accuracy'].values
                            effect_size = calculate_cohens_d(vals1, vals2, str(val1).lower(), str(val2).lower())
                            if effect_size:
                                key = f"accuracy_{trait}_{str(val1).lower()}_vs_{str(val2).lower()}"
                                effect_sizes[key] = effect_size
                                print(f"  accuracy - {val1} vs {val2}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")

    print("\nINTERACTION EFFECTS")
    print("-" * 40)

    # Calculate pairwise comparisons for all trait combinations (since interactions may not be significant)
    for dv in dependent_vars:
        if dv not in performance_df.columns:
            continue
            
        for i in range(len(group_keys)):
            for j in range(i+1, len(group_keys)):
                g1, g2 = group_keys[i], group_keys[j]
                
                if g1 not in performance_df.columns or g2 not in performance_df.columns:
                    continue
                
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
                                    # Only show effects above threshold to avoid noise
                                    if effect_size and abs(effect_size['cohens_d']) > 0.3:
                                        key = f"{dv}_{name1}_vs_{name2}"
                                        effect_sizes[key] = effect_size
                                        print(f"  {dv} - {name1} vs {name2}: d = {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})")

    print(f"\nTotal effect sizes calculated: {len(effect_sizes)}")
    return effect_sizes


def ols_interaction_pvalue(df, dv, factor1, factor2):
    """
    Run OLS regression with interaction term and return the p-value for the interaction.
    Updated to handle the new trait structure.

    Parameters:
    - df: DataFrame with data
    - dv: dependent variable (e.g. "accuracy")
    - factor1: first factor (e.g. "gender")
    - factor2: second factor (e.g. "ethnicity")

    Returns:
    - interaction_p: float or None
    """
    try:
        # Check if factors exist in dataframe
        if factor1 not in df.columns or factor2 not in df.columns:
            print(f"⚠️ Factors {factor1} or {factor2} not found in dataframe")
            return None
            
        formula = f"{dv} ~ C({factor1}) * C({factor2})"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        interaction_term = f"C({factor1}):C({factor2})"
        if interaction_term in anova_table.index:
            p_value = anova_table.loc[interaction_term, "PR(>F)"]
            return p_value
        else:
            print(f"WARNING: Interaction term {interaction_term} not found in ANOVA table")
            return None
    except Exception as e:
        print(f"WARNING: Interaction OLS failed: {e}")
        return None


def run_full_tier1_analysis(
    merged_df: pd.DataFrame, 
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = None, 
    person_set: PersonSet = None
) -> Dict[str, Any]:
    """
    Run the full Tier 1 analysis pipeline:
    - N-way ANOVA for demographic profile traits
    - Risk-benefit Pareto frontier analysis  
    - Effect size computation for key findings
    
    Updated to work with the new PersonSet structure and flexible group_keys.
    
    Parameters:
    - merged_df: Merged classification results with 'profile', 'true_label', and predicted labels.
    - group_keys: Optional tuple of traits to include in analysis. If None, will auto-detect from PersonSet.
    - person_set: PersonSet object containing trait metadata
    
    Returns:
    A dictionary with all analysis results.
    """
    
    print("="*80)
    print("COMPREHENSIVE TIER 1 BIAS ANALYSIS PIPELINE")
    print("="*80)
    print(f"Group keys: {group_keys}")
    print(f"Dataset shape: {merged_df.shape}")
    

    if person_set is None:
        print("WARNING: No PersonSet provided - some analyses may not work correctly")

    if group_keys is None:
        print("Auto-detecting available traits from PersonSet...")
        group_keys = get_analysis_group_keys(person_set)
    else:
        print(f"Using provided group keys: {group_keys}")
        required_keys, optional_keys = get_available_traits(person_set)
        available_keys = set(required_keys + optional_keys)
        invalid_keys = [key for key in group_keys if key not in available_keys]
        if invalid_keys:
            print(f"WARNING: These group keys are not available in PersonSet: {invalid_keys}")
            print(f"Available keys: {sorted(available_keys)}")
            group_keys = tuple(key for key in group_keys if key in available_keys)
            print(f"Using filtered group keys: {group_keys}")
    
    print(f"Dataset shape: {merged_df.shape}")

    try:
        print("\n" + "="*60)
        print("STEP 1: FACTORIAL N-WAY ANOVA ANALYSIS")
        print("="*60)
        anova_results = factorial_analysis_nway_anova(
            merged_df, 
            group_keys=group_keys, 
            person_set=person_set
        )
        print("=== ANOVA analysis completed successfully")
        
    except Exception as e:
        print(f"ERROR: ANOVA analysis failed: {e}")
        anova_results = {'error': str(e)}

    try:
        print("\n" + "="*60)
        print("STEP 2: RISK-BENEFIT PARETO FRONTIER ANALYSIS")
        print("="*60)
        pareto_results = plot_risk_benefit_frontier(
            merged_df, 
            group_keys=group_keys, 
            category_cols=case.category_cols,
            person_set=person_set
        )
        print("=== Pareto frontier analysis completed successfully")
        
    except Exception as e:
        print(f"ERROR: Pareto frontier analysis failed: {e}")
        pareto_results = {'error': str(e)}

    try:
        print("\n" + "="*60)
        print("STEP 3: EFFECT SIZE CALCULATIONS")
        print("="*60)
        if 'performance_data' in anova_results:
            performance_df = anova_results['performance_data']
            effect_sizes = calculate_effect_sizes(
                performance_df, 
                anova_results, 
                group_keys=group_keys
            )
            print("=== Effect size calculations completed successfully")
        else:
            print("ERROR: Cannot calculate effect sizes - no performance data available")
            effect_sizes = {'error': 'No performance data available'}
            
    except Exception as e:
        print(f"ERROR: Effect size calculations failed: {e}")
        effect_sizes = {'error': str(e)}

    print("\n" + "="*80)
    print("TIER 1 ANALYSIS SUMMARY")
    print("="*80)
    
    if 'significant_effects' in anova_results:
        sig_effects = len(anova_results['significant_effects'])
        print(f"=== ANOVA: {sig_effects} significant effects detected")
        if sig_effects > 0:
            for effect in anova_results['significant_effects']:
                print(f"   • {effect['effect']}: F={effect['f_statistic']:.2f}, p={effect['p_value']:.4f}, η²={effect['eta_squared']:.3f}")
    else:
        print("=== ANOVA: No results available")
    
    if 'pareto_optimal' in pareto_results:
        pareto_count = len(pareto_results['pareto_optimal'])
        print(f"=== Pareto Frontier: {pareto_count} optimal profiles identified")
    else:
        print("=== Pareto Frontier: No results available")
    
    if isinstance(effect_sizes, dict) and 'error' not in effect_sizes:
        large_effects = sum(1 for es in effect_sizes.values() 
                           if isinstance(es, dict) and es.get('magnitude') == 'large')
        print(f"=== Effect Sizes: {len(effect_sizes)} calculated, {large_effects} large effects")
    else:
        print("=== Effect Sizes: No results available")

    return {
        "anova_results": anova_results,
        "pareto_results": pareto_results,
        "effect_sizes": effect_sizes,
        "group_keys": group_keys,
        "analysis_summary": {
            "significant_effects": anova_results.get('significant_effects', []),
            "pareto_optimal_count": len(pareto_results.get('pareto_optimal', [])),
            "large_effects_count": sum(1 for es in effect_sizes.values() 
                                     if isinstance(es, dict) and es.get('magnitude') == 'large') if isinstance(effect_sizes, dict) else 0
        }
    }