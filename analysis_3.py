import os
import glob
import json
import re
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import fields


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import accuracy_score
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
from analysis_tools import get_analysis_group_keys, get_available_traits
from profiles.profile_sets import PERSON_ETHNICS
from profiles.schema import PersonSet, PersonMeta
from cases.cases_config import CaseConfig


def consistency_vs_boldness_analysis(merged_df, 
                                     case: CaseConfig,
                                     n_folds=5,
                                     person_set: PersonSet = PERSON_ETHNICS,
                                     group_keys=("gender", "ethnicity", "age"),
                                     ):
    """
    Consistency vs Boldness Tradeoff Analysis
    
    Tests if *inconsistent profiles* actually make better moral calls by examining:
    - Consistency (inverse of std dev across cross-validation runs)  
    - Boldness (rescue rate, bias magnitude)
    - Moral value (alignment with normative judgments)
    
    Reveals whether *risk-taking* profiles are **normatively valuable** 
    despite being inconsistent across different data samples.
    
    Updated to work with PersonSet structure and flexible group_keys.
    """
    
    print("=" * 80)
    print("CONSISTENCY vs BOLDNESS TRADEOFF ANALYSIS")
    print("=" * 80)
    print(f"Group keys for analysis: {group_keys}")
    
    # ========================================================================
    # STEP 1: IDENTIFY VALID PROFILES USING PERSONSET
    # ========================================================================
    
    def get_valid_profiles(merged_df, person_set):
        """Extract profile columns that exist in both dataframe and PersonSet metadata."""
        all_profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        valid_profiles = []
        
        for col in all_profile_cols:            
            # Check if this profile exists in PersonSet metadata
            if col in person_set.metadata:
                valid_profiles.append(col)
                
        print(f"Found {len(valid_profiles)} valid profiles out of {len(all_profile_cols)} total profile columns")
        return valid_profiles
    
    profile_cols = get_valid_profiles(merged_df, person_set)
    
    if len(profile_cols) == 0:
        print("ERROR: No valid profiles found in PersonSet metadata")
        return {'error': 'No valid profiles found'}

    # ========================================================================
    # STEP 2: CONSISTENCY ANALYSIS ACROSS CV FOLDS
    # ========================================================================
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    consistency_data = {}
    
    print(f"=== Calculating consistency across {n_folds} folds...")
    
    for profile in profile_cols:
        if profile not in merged_df.columns:
            continue
            
        fold_accuracies = []
        fold_rescue_rates = []
        fold_bias_magnitudes = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(merged_df)):
            test_data = merged_df.iloc[test_idx]
            
            # Accuracy on this fold
            acc = accuracy_score(test_data['true_label'], test_data[profile])
            fold_accuracies.append(acc)
            
            # Rescue rate on this fold
            base_correct = (test_data['base_pred'] == test_data['true_label'])
            profile_correct = (test_data[profile] == test_data['true_label'])
            rescued = ((~base_correct) & profile_correct).sum()
            base_errors = (~base_correct).sum()
            rescue_rate = rescued / base_errors if base_errors > 0 else 0
            fold_rescue_rates.append(rescue_rate)
            
            # Bias magnitude on this fold (simplified)
            base_preds = test_data['base_pred']
            profile_preds = test_data[profile]
            
            to_positive = ((base_preds == "no") & (profile_preds == "yes")).sum()
            to_negative = ((base_preds == "yes") & (profile_preds == "no")).sum()
            bias_magnitude = abs(to_positive - to_negative) / len(test_data)
            fold_bias_magnitudes.append(bias_magnitude)
        
        # Calculate consistency metrics (lower std = higher consistency)
        consistency_data[profile] = {
            'accuracy_mean': np.mean(fold_accuracies),
            'accuracy_std': np.std(fold_accuracies),
            'accuracy_cv': np.std(fold_accuracies) / np.mean(fold_accuracies) if np.mean(fold_accuracies) > 0 else np.inf,
            'rescue_rate_mean': np.mean(fold_rescue_rates),
            'rescue_rate_std': np.std(fold_rescue_rates),
            'rescue_rate_cv': np.std(fold_rescue_rates) / np.mean(fold_rescue_rates) if np.mean(fold_rescue_rates) > 0 else np.inf,
            'bias_magnitude_mean': np.mean(fold_bias_magnitudes),
            'bias_magnitude_std': np.std(fold_bias_magnitudes),
            'fold_accuracies': fold_accuracies,
            'fold_rescue_rates': fold_rescue_rates,
            'fold_bias_magnitudes': fold_bias_magnitudes
        }

    # ========================================================================
    # STEP 3: BOLDNESS METRICS CALCULATION
    # ========================================================================
    
    print("=== Calculating boldness metrics...")
    
    # Get rescue stats and bias patterns for boldness calculation
    try:
        category_cols = getattr(case, "category_cols", None)
        if not category_cols:
            print("Error in retrieving category columns - defaulting back to ['stereotype_type']")
            category_cols = ["stereotype_type"]

        rescue_stats_list = []
        bias_patterns_list = []
        for cat_col in category_cols:
            if cat_col in merged_df.columns:
                rs = rescue_stats_by_category(merged_df, category_col=cat_col)
                rs["category_col"] = cat_col
                rescue_stats_list.append(rs)
        
                bp = detect_systematic_biases(merged_df, category_col=cat_col)
                bp["category_col"] = cat_col
                bias_patterns_list.append(bp)
        if not rescue_stats_list:
            raise ValueError("No valid category columns found in merged_df.")
        
        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True)

    except Exception as e:
        print(f"WARNING: Could not calculate rescue/bias stats: {e}")
        print("Creating simplified boldness metrics...")
        
        # Fallback: create simplified rescue and bias stats
        rescue_stats = []
        bias_patterns = []
        
        for profile in profile_cols:
            if profile not in merged_df.columns:
                continue
                
            # Simplified rescue rate
            base_correct = (merged_df['base_pred'] == merged_df['true_label'])
            profile_correct = (merged_df[profile] == merged_df['true_label'])
            rescued = ((~base_correct) & profile_correct).sum()
            base_errors = (~base_correct).sum()
            rescue_rate = rescued / base_errors if base_errors > 0 else 0
            
            rescue_stats.append({
                'profile': profile,
                'rescue_rate': rescue_rate
            })
            
            # Simplified bias magnitude
            base_preds = merged_df['base_pred']
            profile_preds = merged_df[profile]
            to_positive = ((base_preds == "no") & (profile_preds == "yes")).sum()
            to_negative = ((base_preds == "yes") & (profile_preds == "no")).sum()
            bias_magnitude = abs(to_positive - to_negative) / len(merged_df)
            mislabelling_rate = (merged_df[profile] != merged_df['base_pred']).mean()
            
            bias_patterns.append({
                'profile': profile,
                'bias_magnitude': bias_magnitude,
                'mislabelling_rate': mislabelling_rate
            })
        
        rescue_stats = pd.DataFrame(rescue_stats)
        bias_patterns = pd.DataFrame(bias_patterns)
    
    boldness_data = {}
    
    for profile in profile_cols:
        if profile not in consistency_data:
            continue
            
        # Boldness metric 1: Rescue effectiveness
        profile_rescue = rescue_stats[rescue_stats['profile'] == profile]
        avg_rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
        
        # Boldness metric 2: Willingness to disagree with baseline (mislabelling rate)
        profile_bias = bias_patterns[bias_patterns['profile'] == profile]
        avg_mislabelling_rate = profile_bias['mislabelling_rate'].mean() if len(profile_bias) > 0 else 0
        
        # Boldness metric 3: Magnitude of bias (willingness to take strong positions)
        avg_bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
        
        # Composite boldness score
        boldness_score = (
            0.4 * avg_rescue_rate +      # 40% rescue effectiveness
            0.3 * avg_mislabelling_rate + # 30% disagreement rate  
            0.3 * avg_bias_magnitude     # 30% position strength
        )
        
        boldness_data[profile] = {
            'rescue_rate': avg_rescue_rate,
            'mislabelling_rate': avg_mislabelling_rate,
            'bias_magnitude': avg_bias_magnitude,
            'boldness_score': boldness_score
        }

    # ========================================================================
    # STEP 4: DEMOGRAPHIC-AWARE STABILITY-BOLDNESS CORRELATION ANALYSIS
    # ========================================================================
    
    print("=== Analyzing consistency-boldness correlations by demographics...")
    
    # Prepare data for correlation analysis
    analysis_data = []
    
    for profile in profile_cols:
        if profile in consistency_data and profile in boldness_data:
            # Extract traits using PersonSet
            traits = person_set.get_traits(profile, group_keys)
            
            row = {
                'profile': profile,
                'volatility': consistency_data[profile]['accuracy_std'],  # Higher = less consistent
                'consistency': 1 / (1 + consistency_data[profile]['accuracy_std']),  # Higher = more consistent
                'boldness_score': boldness_data[profile]['boldness_score'],
                'rescue_rate': boldness_data[profile]['rescue_rate'],
                'accuracy_mean': consistency_data[profile]['accuracy_mean'],
                'bias_magnitude': boldness_data[profile]['bias_magnitude'],
                'mislabelling_rate': boldness_data[profile]['mislabelling_rate']
            }
            # Add demographic traits
            for trait_name in group_keys:
                row[trait_name] = traits.get(trait_name, "Unknown")
            
            analysis_data.append(row)
    
    analysis_df = pd.DataFrame(analysis_data)
    
    if len(analysis_df) == 0:
        print("ERROR: No data available for correlation analysis")
        return {'error': 'No analysis data available'}
    
    print(f"Analysis dataset shape: {analysis_df.shape}")
    
    # Calculate overall correlations
    correlations = {}
    
    if len(analysis_df) >= 3:  # Need at least 3 points for meaningful correlation
        
        # Volatility vs Boldness (higher volatility = lower consistency)
        corr_vol_bold, p_val_vol_bold = pearsonr(analysis_df['volatility'], analysis_df['boldness_score'])
        correlations['volatility_vs_boldness'] = {
            'correlation': corr_vol_bold,
            'p_value': p_val_vol_bold,
            'significant': p_val_vol_bold < 0.05,
            'interpretation': 'Positive correlation means less consistent profiles are bolder'
        }
        
        # Consistency vs Boldness (inverse relationship)
        corr_cons_bold, p_val_cons_bold = pearsonr(analysis_df['consistency'], analysis_df['boldness_score'])
        correlations['consistency_vs_boldness'] = {
            'correlation': corr_cons_bold,
            'p_value': p_val_cons_bold,
            'significant': p_val_cons_bold < 0.05,
            'interpretation': 'Negative correlation means more consistent profiles are less bold'
        }
        
        # Volatility vs Rescue Rate (moral value proxy)
        corr_vol_rescue, p_val_vol_rescue = pearsonr(analysis_df['volatility'], analysis_df['rescue_rate'])
        correlations['volatility_vs_rescue'] = {
            'correlation': corr_vol_rescue,
            'p_value': p_val_vol_rescue,
            'significant': p_val_vol_rescue < 0.05
        }
        
        # Boldness vs Accuracy (performance tradeoff)
        corr_bold_acc, p_val_bold_acc = pearsonr(analysis_df['boldness_score'], analysis_df['accuracy_mean'])
        correlations['boldness_vs_accuracy'] = {
            'correlation': corr_bold_acc,
            'p_value': p_val_bold_acc,
            'significant': p_val_bold_acc < 0.05
        }
        
        print(f"\n--- OVERALL CORRELATION RESULTS:")
        print(f"  Volatility vs Boldness: r={corr_vol_bold:.3f}, p={p_val_vol_bold:.4f} {'***' if p_val_vol_bold < 0.05 else ''}")
        print(f"    → {correlations['volatility_vs_boldness']['interpretation']}")
        print(f"  Consistency vs Boldness: r={corr_cons_bold:.3f}, p={p_val_cons_bold:.4f} {'***' if p_val_cons_bold < 0.05 else ''}")
        print(f"    → {correlations['consistency_vs_boldness']['interpretation']}")
        print(f"  Volatility vs Rescue Rate: r={corr_vol_rescue:.3f}, p={p_val_vol_rescue:.4f} {'***' if p_val_vol_rescue < 0.05 else ''}")
        print(f"  Boldness vs Accuracy: r={corr_bold_acc:.3f}, p={p_val_bold_acc:.4f} {'***' if p_val_bold_acc < 0.05 else ''}")
    
    # ========================================================================
    # STEP 5: DEMOGRAPHIC-SPECIFIC CORRELATIONS
    # ========================================================================
    
    demographic_correlations = {}
    
    for trait_name in group_keys:
        if trait_name not in analysis_df.columns:
            continue
            
        trait_values = analysis_df[trait_name].dropna().unique()
        valid_values = [v for v in trait_values if v != "Unknown"]
        
        if len(valid_values) == 0:
            continue
            
        print(f"\n--- CORRELATIONS BY {trait_name.upper()}:")
        demographic_correlations[trait_name] = {}
        
        for value in valid_values:
            subset = analysis_df[analysis_df[trait_name] == value]
            
            if len(subset) >= 3:  # Need at least 3 points
                try:
                    corr_vol_bold, p_val = pearsonr(subset['volatility'], subset['boldness_score'])
                    demographic_correlations[trait_name][str(value)] = {
                        'volatility_vs_boldness': {
                            'correlation': corr_vol_bold,
                            'p_value': p_val,
                            'n': len(subset),
                            'significant': p_val < 0.05
                        }
                    }
                    print(f"  {value} (n={len(subset)}): Volatility vs Boldness r={corr_vol_bold:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
                except:
                    print(f"  {value} (n={len(subset)}): Could not calculate correlation")
            else:
                print(f"  {value} (n={len(subset)}): Insufficient data for correlation")

    # ========================================================================
    # STEP 6: PROFILE CLASSIFICATION WITH DEMOGRAPHIC CONTEXT
    # ========================================================================
    
    print(f"\n=== PROFILE CLASSIFICATION BY DEMOGRAPHICS:")
    profile_archetypes = {}
    
    volatility_median = analysis_df['volatility'].median()
    boldness_median = analysis_df['boldness_score'].median()
    
    # Group profiles by demographics and archetypes
    archetype_by_demographics = {}
    
    for trait_name in group_keys:
        if trait_name not in analysis_df.columns:
            continue
        archetype_by_demographics[trait_name] = {}
    
    for _, row in analysis_df.iterrows():
        profile = row['profile']
        vol = row['volatility']
        bold = row['boldness_score']
        rescue = row['rescue_rate']
        acc = row['accuracy_mean']
        
        if vol > volatility_median and bold > boldness_median:
            archetype = "Inconsistent Bold" 
            description = "High volatility, high moral value"
        elif vol < volatility_median and bold > boldness_median:
            archetype = "Consistent Bold"
            description = "Best performance, low volatility"
        elif vol > volatility_median and bold < boldness_median:
            archetype = "Inconsistent Cautious"
            description = "High volatility, low moral value"
        else:
            archetype = "Consistent Cautious"
            description = "Low volatility, predictable"
        
        profile_archetypes[profile] = {
            'archetype': archetype,
            'description': description,
            'volatility': vol,
            'boldness': bold,
            'rescue_rate': rescue,
            'accuracy': acc
        }
        
        # Add to demographic groupings
        for trait_name in group_keys:
            if trait_name in row and row[trait_name] != "Unknown":
                trait_val = str(row[trait_name])
                if trait_val not in archetype_by_demographics[trait_name]:
                    archetype_by_demographics[trait_name][trait_val] = {}
                if archetype not in archetype_by_demographics[trait_name][trait_val]:
                    archetype_by_demographics[trait_name][trait_val][archetype] = []
                archetype_by_demographics[trait_name][trait_val][archetype].append(profile)
        
        # Get trait display string
        traits = {k: row[k] for k in group_keys if k in row and row[k] != "Unknown"}
        trait_str = ", ".join(f"{k}={v}" for k, v in traits.items())
        print(f"  {profile} ({trait_str}): {archetype} – {description}")

    # ========================================================================
    # STEP 7: DEMOGRAPHIC BIAS DETECTION IN ARCHETYPES
    # ========================================================================
    
    print(f"\n=== DEMOGRAPHIC BIAS DETECTION IN ARCHETYPES:")
    
    archetype_bias_analysis = {}
    
    for trait_name in group_keys:
        if trait_name not in archetype_by_demographics:
            continue
            
        print(f"\n--- {trait_name.upper()} BIAS ANALYSIS:")
        archetype_bias_analysis[trait_name] = {}
        
        # Count archetypes by demographic group
        for trait_val, archetypes in archetype_by_demographics[trait_name].items():
            total_profiles = sum(len(profiles) for profiles in archetypes.values())
            print(f"  {trait_val} (n={total_profiles}):")
            
            archetype_bias_analysis[trait_name][trait_val] = {}
            
            for archetype, profiles in archetypes.items():
                percentage = (len(profiles) / total_profiles) * 100 if total_profiles > 0 else 0
                archetype_bias_analysis[trait_name][trait_val][archetype] = {
                    'count': len(profiles),
                    'percentage': percentage,
                    'profiles': profiles
                }
                print(f"    {archetype}: {len(profiles)} ({percentage:.1f}%)")

    # ========================================================================
    # STEP 8: NORMATIVE VALUE ASSESSMENT BY DEMOGRAPHICS
    # ========================================================================
    
    print(f"\n=== NORMATIVE VALUE ASSESSMENT BY DEMOGRAPHICS:")
    
    normative_assessment = {}
    
    # Overall assessment
    high_volatility_profiles = analysis_df[analysis_df['volatility'] > volatility_median]
    low_volatility_profiles = analysis_df[analysis_df['volatility'] <= volatility_median]
    
    if len(high_volatility_profiles) > 0 and len(low_volatility_profiles) > 0:
        high_vol_rescue_mean = high_volatility_profiles['rescue_rate'].mean()
        low_vol_rescue_mean = low_volatility_profiles['rescue_rate'].mean()
        
        t_stat, p_val = ttest_ind(high_volatility_profiles['rescue_rate'], 
                                 low_volatility_profiles['rescue_rate'])
        
        normative_assessment['overall'] = {
            'high_volatility_rescue': high_vol_rescue_mean,
            'low_volatility_rescue': low_vol_rescue_mean,
            'difference': high_vol_rescue_mean - low_vol_rescue_mean,
            'statistical_test': {'t_stat': t_stat, 'p_value': p_val, 'significant': p_val < 0.05}
        }
        
        print(f"  OVERALL:")
        print(f"    Inconsistent Profiles Rescue Rate: {high_vol_rescue_mean:.3f} (n={len(high_volatility_profiles)})")
        print(f"    Consistent Profiles Rescue Rate: {low_vol_rescue_mean:.3f} (n={len(low_volatility_profiles)})")
        print(f"    Difference: {high_vol_rescue_mean - low_vol_rescue_mean:.3f}")
        print(f"    Statistical Test: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
    
    # Demographic-specific assessment
    for trait_name in group_keys:
        if trait_name not in analysis_df.columns:
            continue
            
        trait_values = analysis_df[trait_name].dropna().unique()
        valid_values = [v for v in trait_values if v != "Unknown"]
        
        if len(valid_values) == 0:
            continue
            
        print(f"\n  BY {trait_name.upper()}:")
        normative_assessment[trait_name] = {}
        
        for value in valid_values:
            subset = analysis_df[analysis_df[trait_name] == value]
            
            if len(subset) >= 4:  # Need sufficient data for splitting
                subset_high_vol = subset[subset['volatility'] > subset['volatility'].median()]
                subset_low_vol = subset[subset['volatility'] <= subset['volatility'].median()]
                
                if len(subset_high_vol) > 0 and len(subset_low_vol) > 0:
                    high_rescue = subset_high_vol['rescue_rate'].mean()
                    low_rescue = subset_low_vol['rescue_rate'].mean()
                    
                    try:
                        t_stat, p_val = ttest_ind(subset_high_vol['rescue_rate'], 
                                                 subset_low_vol['rescue_rate'])
                        
                        normative_assessment[trait_name][str(value)] = {
                            'high_volatility_rescue': high_rescue,
                            'low_volatility_rescue': low_rescue,
                            'difference': high_rescue - low_rescue,
                            'statistical_test': {'t_stat': t_stat, 'p_value': p_val, 'significant': p_val < 0.05}
                        }
                        
                        print(f"    {value}: Inconsistent Rescue={high_rescue:.3f} vs Consistent Rescue={low_rescue:.3f}, "
                              f"diff={high_rescue - low_rescue:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
                              
                    except:
                        print(f"    {value}: Could not perform statistical test")
            else:
                print(f"    {value}: Insufficient data (n={len(subset)})")

    return {
        'consistency_data': consistency_data,
        'boldness_data': boldness_data,
        'analysis_df': analysis_df,
        'correlations': correlations,
        'demographic_correlations': demographic_correlations,
        'profile_archetypes': profile_archetypes,
        'archetype_bias_analysis': archetype_bias_analysis,
        'normative_assessment': normative_assessment,
        'group_keys': group_keys,
        'valid_profiles': profile_cols
    }


def plot_consistency_boldness_analysis(consistency_results, 
                                       person_set: PersonSet = None,
                                       group_keys=("gender", "ethnicity", "age"),
                                       figsize=(10, 6)):
    """
    Create comprehensive visualizations for consistency vs boldness analysis.
    
    Generates 4-panel plot:
    1. Consistency vs Boldness Scatter (with demographic coloring)
    2. Volatility vs Rescue Rate  
    3. Profile Archetype Distribution by Demographics
    4. Consistency Trends Across CV Folds
    
    Updated for PersonSet structure and demographic bias detection.
    """
    
    # Extract data
    consistency_data = consistency_results['consistency_data']
    boldness_data = consistency_results['boldness_data']
    archetypes = consistency_results['profile_archetypes']
    analysis_df = consistency_results.get('analysis_df', pd.DataFrame())
    
    # Prepare arrays
    profiles = list(archetypes.keys())
    volatilities = [archetypes[p]['volatility'] for p in profiles]
    boldness_scores = [archetypes[p]['boldness'] for p in profiles]
    rescue_rates = [archetypes[p]['rescue_rate'] for p in profiles]
    accuracies = [archetypes[p]['accuracy'] for p in profiles]
    
    # Archetype colors (updated terminology)
    archetype_colors = {
        'Inconsistent Bold': '#d62728',      # Red - high risk, high value
        'Consistent Bold': '#2ca02c',        # Green - best performance
        'Inconsistent Cautious': '#ff7f0e',  # Orange - high risk, low value
        'Consistent Cautious': '#1f77b4'     # Blue - safe and predictable
    }
    
    colors = [archetype_colors.get(archetypes[p]['archetype'], '#8c564b') for p in profiles]
    
    def _new_fig(title):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle('Consistency vs Boldness Analysis', fontsize=14, fontweight='bold')
        ax.set_title(title)
        return fig, ax
    
    figs = {}

    # Figure A: Consistency vs Boldness
    figA, ax = _new_fig('Consistency vs Boldness Tradeoff')
    if person_set and len(analysis_df) > 0:
        primary_trait = group_keys[0] if group_keys else 'gender'
        if primary_trait in analysis_df.columns:
            trait_values = [v for v in analysis_df[primary_trait].unique() if v != "Unknown"]
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            trait_markers = {val: markers[i % len(markers)] for i, val in enumerate(trait_values)}
            for trait_val in trait_values:
                mask = analysis_df[primary_trait] == trait_val
                trait_subset = analysis_df[mask]
                if len(trait_subset) > 0:
                    vols = trait_subset['volatility'].values
                    bolds = trait_subset['boldness_score'].values
                    trait_colors = [archetype_colors.get(archetypes[p]['archetype'], '#8c564b') 
                                    for p in trait_subset['profile'].values]
                    ax.scatter(vols, bolds, c=trait_colors, s=100, alpha=0.7, 
                               marker=trait_markers[trait_val], edgecolors='black', linewidth=1,
                               label=f'{primary_trait}={trait_val}')
    else:
        ax.scatter(volatilities, boldness_scores, c=colors, s=100, alpha=0.7, edgecolors='black')

    for i, profile in enumerate(profiles):
        if volatilities[i] > np.percentile(volatilities, 80) or boldness_scores[i] > np.percentile(boldness_scores, 80):
            if person_set:
                traits = person_set.get_traits(profile, group_keys)
                demo_label = "_".join(str(traits.get(k, '?'))[:3] for k in group_keys[:2] if traits.get(k) != "Unknown")
                label = demo_label if demo_label else profile.replace('profile', 'P')
            else:
                label = profile.replace('profile', 'P')
            ax.annotate(label, (volatilities[i], boldness_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    if 'volatility_vs_boldness' in consistency_results['correlations']:
        corr_data = consistency_results['correlations']['volatility_vs_boldness']
        if corr_data['significant']:
            z = np.polyfit(volatilities, boldness_scores, 1)
            p = np.poly1d(z)
            ax.plot(sorted(volatilities), p(sorted(volatilities)), "r--", alpha=0.8)
            ax.text(0.05, 0.95, f"r={corr_data['correlation']:.3f}*", 
                    transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    ax.set_xlabel('Volatility (Higher = Less Consistent)')
    ax.set_ylabel('Boldness Score')
    ax.grid(True, alpha=0.3)
    if person_set and len(analysis_df) > 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    figs['consistency_vs_boldness'] = figA

    # Figure B: Volatility vs Rescue
    figB, ax = _new_fig('Consistency vs Moral Value')
    ax.scatter(volatilities, rescue_rates, c=colors, s=100, alpha=0.7, edgecolors='black')
    if 'volatility_vs_rescue' in consistency_results['correlations']:
        corr_data = consistency_results['correlations']['volatility_vs_rescue']
        if corr_data['significant']:
            z = np.polyfit(volatilities, rescue_rates, 1)
            p = np.poly1d(z)
            ax.plot(sorted(volatilities), p(sorted(volatilities)), "g--", alpha=0.8)
            ax.text(0.05, 0.95, f"r={corr_data['correlation']:.3f}*", 
                    transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    ax.set_xlabel('Volatility (Higher = Less Consistent)')
    ax.set_ylabel('Rescue Rate (Moral Value)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs['volatility_vs_rescue'] = figB

    # Figure C: Archetype Distribution
    figC, ax = _new_fig('Profile Archetype Distribution')
    if person_set and len(analysis_df) > 0 and group_keys and group_keys[0] in analysis_df.columns:
        primary_trait = group_keys[0]
        trait_values = [v for v in analysis_df[primary_trait].unique() if v != "Unknown"]
        archetype_by_trait = {}
        for trait_val in trait_values:
            archetype_by_trait[trait_val] = {}
            subset = analysis_df[analysis_df[primary_trait] == trait_val]
            for _, row in subset.iterrows():
                profile = row['profile']
                if profile in archetypes:
                    arch = archetypes[profile]['archetype']
                    archetype_by_trait[trait_val][arch] = archetype_by_trait[trait_val].get(arch, 0) + 1
        archetypes_list = list(archetype_colors.keys())
        trait_positions = np.arange(len(trait_values))
        bottom = np.zeros(len(trait_values))
        for arch in archetypes_list:
            counts = [archetype_by_trait[trait_val].get(arch, 0) for trait_val in trait_values]
            ax.bar(trait_positions, counts, bottom=bottom, 
                   color=archetype_colors[arch], label=arch, alpha=0.8)
            bottom += counts
        ax.set_xticks(trait_positions)
        ax.set_xticklabels(trait_values, rotation=45)
        ax.set_xlabel(primary_trait.capitalize())
        ax.set_ylabel('Number of Profiles')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        archetype_counts = {}
        for profile_data in archetypes.values():
            arch = profile_data['archetype']
            archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        archetype_names = list(archetype_counts.keys())
        counts = list(archetype_counts.values())
        colors_pie = [archetype_colors.get(arch, '#8c564b') for arch in archetype_names]
        ax.pie(counts, labels=archetype_names, colors=colors_pie, autopct='%1.0f%%', startangle=90)
    ax.grid(False)
    plt.tight_layout()
    figs['archetype_by_demo'] = figC

    # Figure D: CV fold trends
    figD, ax = _new_fig('Consistency Across CV Folds')
    representative_profiles = []
    for archetype in archetype_colors.keys():
        for profile, data in archetypes.items():
            if data['archetype'] == archetype and profile not in representative_profiles:
                representative_profiles.append(profile)
                break
    representative_profiles = representative_profiles[:4]
    if consistency_data:
        fold_numbers = list(range(1, len(list(consistency_data.values())[0]['fold_accuracies']) + 1))
        for i, profile in enumerate(representative_profiles):
            if profile in consistency_data:
                fold_accs = consistency_data[profile]['fold_accuracies']
                if fold_accs:
                    archetype = archetypes[profile]['archetype']
                    color = archetype_colors.get(archetype, '#8c564b')
                    if person_set:
                        traits = person_set.get_traits(profile, group_keys)
                        demo_info = "_".join(str(traits.get(k, '?'))[:3] for k in group_keys[:2] if traits.get(k) != "Unknown")
                        legend_label = f"{demo_info} ({archetype})"
                    else:
                        legend_label = f"{profile.replace('profile', 'P')} ({archetype})"
                    ax.plot(fold_numbers, fold_accs, 'o-', color=color, alpha=0.7, label=legend_label)
        ax.set_xlabel('Cross-Validation Fold')
        ax.set_ylabel('Accuracy')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs['cv_trends'] = figD

    plt.show()

    return figs




def simplified_causal_modeling(merged_df, 
                             person_set: PersonSet, 
                             case: CaseConfig,
                             group_keys=("gender", "ethnicity", "age"),
                             consistency_results=None):
    """
    Simplified Causal Modeling for Bias Effects
    
    Models causal influence of profile traits on bias/alignment using:
    - Linear regression with demographic and other traits predictors
    - Path analysis showing direct vs indirect effects
    - Theoretical framework for understanding bias mechanisms
    
    Updated to work with PersonSet structure and flexible group_keys.
    Gives theoretical weight to findings for thesis contribution.
    """
    
    print("=" * 80)
    print("SIMPLIFIED CAUSAL MODELING: PROFILE TRAITS → BIAS OUTCOMES")
    print("=" * 80)
    print(f"Group keys: {group_keys}")
    
    # ========================================================================
    # STEP 1: IDENTIFY VALID PROFILES AND PREPARE CAUSAL VARIABLES
    # ========================================================================
    
    def get_valid_profiles(merged_df, person_set):
        """Extract profile columns that exist in both dataframe and PersonSet metadata."""
        all_profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        valid_profiles = []
        
        for col in all_profile_cols:
            if col in person_set.metadata:
                valid_profiles.append(col)
                
        return valid_profiles
    
    profile_cols = get_valid_profiles(merged_df, person_set)
    
    if len(profile_cols) == 0:
        print("ERROR: No valid profiles found in PersonSet metadata")
        return {'error': 'No valid profiles found'}
    
    print(f"Found {len(profile_cols)} valid profiles for causal modeling")
    
    # Create causal dataset
    causal_data = []

    # Collect all unique trait values across profiles for one-hot encoding
    all_trait_keys = set()
    trait_reference_categories = {}  # Track reference categories to drop
    
    for profile in profile_cols:
        traits = person_set.get_traits(profile, group_keys)
        
        for trait_name, value in traits.items():
            if value != "Unknown":
                # Handle age enum values
                if trait_name == "age" and hasattr(value, 'value'):
                    all_trait_keys.add(f"{trait_name}_{value.value}")
                else:
                    all_trait_keys.add(f"{trait_name}_{value}")
    
    # Identify reference categories (first category for each trait) to avoid multicollinearity
    for trait_name in group_keys:
        trait_values = [key.split('_')[1] for key in all_trait_keys if key.startswith(f"{trait_name}_")]
        if trait_values:
            # Sort to ensure consistent reference category selection
            trait_values_sorted = sorted(trait_values)
            reference_category = f"{trait_name}_{trait_values_sorted[0]}"
            trait_reference_categories[trait_name] = reference_category
            # Remove reference category from trait keys to avoid multicollinearity
            if reference_category in all_trait_keys:
                all_trait_keys.remove(reference_category)
    
    print(f"Identified {len(all_trait_keys)} unique trait-value combinations (after removing reference categories)")
    print(f"Reference categories: {trait_reference_categories}")
    
    # Get performance metrics
    try:
        category_cols = getattr(case, "category_cols", None)
        if not category_cols:
            print("Error in retrieving category columns - defaulting back to ['stereotype_type']")
            category_cols = ["stereotype_type"]

        rescue_stats_list = []
        bias_patterns_list = []
        for cat_col in category_cols:
            if cat_col in merged_df.columns:
                rs = rescue_stats_by_category(merged_df, category_col=cat_col)
                rs["category_col"] = cat_col
                rescue_stats_list.append(rs)
        
                bp = detect_systematic_biases(merged_df, category_col=cat_col)
                bp["category_col"] = cat_col
                bias_patterns_list.append(bp)
        if not rescue_stats_list:
            raise ValueError("No valid category columns found in merged_df.")
        
        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True)
    except Exception as e:
        print(f"WARNING: Could not get rescue/bias stats: {e}")
        print("Creating simplified metrics...")
        
        # Fallback: create simplified stats
        rescue_stats = []
        bias_patterns = []
        
        for profile in profile_cols:
            if profile not in merged_df.columns:
                continue
                
            # Simplified rescue rate
            base_correct = (merged_df['base_pred'] == merged_df['true_label'])
            profile_correct = (merged_df[profile] == merged_df['true_label'])
            rescued = ((~base_correct) & profile_correct).sum()
            base_errors = (~base_correct).sum()
            rescue_rate = rescued / base_errors if base_errors > 0 else 0
            extra_err_rate = ((base_correct) & (~profile_correct)).sum() / len(merged_df)
            
            rescue_stats.append({
                'profile': profile,
                'rescue_rate': rescue_rate,
                'extra_err_rate': extra_err_rate
            })
            
            # Simplified bias metrics
            base_preds = merged_df['base_pred']
            profile_preds = merged_df[profile]
            to_positive = ((base_preds == "no") & (profile_preds == "yes")).sum()
            to_negative = ((base_preds == "yes") & (profile_preds == "no")).sum()
            bias_magnitude = abs(to_positive - to_negative) / len(merged_df)
            mislabelling_rate = (merged_df[profile] != merged_df['base_pred']).mean()
            
            bias_patterns.append({
                'profile': profile,
                'bias_magnitude': bias_magnitude,
                'mislabelling_rate': mislabelling_rate
            })
        
        rescue_stats = pd.DataFrame(rescue_stats)
        bias_patterns = pd.DataFrame(bias_patterns)
    
    # Build causal dataset
    for profile_name in profile_cols:
        if profile_name not in merged_df.columns:
            continue

        traits = person_set.get_traits(profile_name, group_keys)
        
        causal_row = {"profile": profile_name}

        # Add raw trait values (handle age enum properly)
        for trait_name in group_keys:
            raw_value = traits.get(trait_name, "Unknown")
            # Handle age enum - extract the numeric value
            if trait_name == "age" and hasattr(raw_value, 'value'):
                raw_value = raw_value.value
            causal_row[f"raw_{trait_name}"] = raw_value
        
        # Add one-hot encoded traits (exclude reference categories to avoid multicollinearity)
        for trait_name in group_keys:
            value = traits.get(trait_name, "Unknown")
            if value != "Unknown":
                # Handle age enum values
                if trait_name == "age" and hasattr(value, 'value'):
                    trait_key = f"{trait_name}_{value.value}"
                else:
                    trait_key = f"{trait_name}_{value}"
                
                # Only add if not a reference category
                if trait_key in all_trait_keys:
                    causal_row[trait_key] = 1

        # Initialize all possible trait combinations to 0
        for key in all_trait_keys:
            causal_row.setdefault(key, 0)
        
        # Calculate outcome variables
        accuracy = (merged_df[profile_name] == merged_df['true_label']).mean()
        
        # Get rescue and bias metrics
        profile_rescue = rescue_stats[rescue_stats['profile'] == profile_name]
        rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
        extra_error_rate = profile_rescue['extra_err_rate'].mean() if len(profile_rescue) > 0 else 0
        
        profile_bias = bias_patterns[bias_patterns['profile'] == profile_name]
        bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
        mislabelling_rate = profile_bias['mislabelling_rate'].mean() if len(profile_bias) > 0 else 0
        
        # Get consistency metrics if available
        volatility = 0
        if consistency_results and profile_name in consistency_results['consistency_data']:
            volatility = consistency_results['consistency_data'][profile_name]['accuracy_std']
        
        causal_row.update({
            'accuracy': accuracy,
            'rescue_rate': rescue_rate,
            'extra_error_rate': extra_error_rate,
            'bias_magnitude': bias_magnitude,
            'mislabelling_rate': mislabelling_rate,
            'volatility': volatility
        })

        causal_data.append(causal_row)
    
    causal_df = pd.DataFrame(causal_data)
    
    print(f"Causal dataset prepared with {len(causal_df)} profiles")
    print(f"    Trait variables: {len([col for col in causal_df.columns if any(col.startswith(f'{trait}_') for trait in group_keys)])}")
    print(f"    Reference categories excluded: {list(trait_reference_categories.values())}")
    
    # ========================================================================
    # STEP 2: LINEAR REGRESSION MODELS WITH DEMOGRAPHIC VS OTHER TRAITS
    # ========================================================================
    
    print(f"\nCAUSAL PATH ANALYSIS:")
    
    # Define predictor sets based on group_keys
    trait_predictors = sorted([
        col for col in causal_df.columns
        if any(col.startswith(f'{trait}_') for trait in group_keys) 
        and not col.endswith('_Unknown')
    ])
    
    # Separate demographic vs non-demographic predictors
    demographic_traits = ['gender', 'ethnicity']  # Core demographic traits
    demographic_predictors = [col for col in trait_predictors 
                            if any(col.startswith(f'{trait}_') for trait in demographic_traits)]
    other_predictors = [col for col in trait_predictors if col not in demographic_predictors]

    print(f"Demographic predictors ({len(demographic_predictors)}): {demographic_predictors}")
    print(f"Other trait predictors ({len(other_predictors)}): {other_predictors}")
    
    # Full predictor list
    all_predictors = demographic_predictors + other_predictors
    
    # Outcome variables to model
    outcomes = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude']
    if consistency_results:
        outcomes.append('volatility')
    
    causal_results = {}
    
    for outcome in outcomes:
        print(f"\n--- MODELING: {outcome.upper()} ---")
        
        y = causal_df[outcome].values
        
        if len(demographic_predictors) == 0 or len(other_predictors) == 0:
            print(f"  WARNING: Insufficient predictors for full causal analysis")
            continue
        
        # Model 1: Demographics only
        X_demo = causal_df[demographic_predictors].values
        if X_demo.shape[1] > 0:
            model_demo = LinearRegression().fit(X_demo, y)
            r2_demo = model_demo.score(X_demo, y)
        else:
            r2_demo = 0
            model_demo = None
        
        # Model 2: Other traits only
        X_other = causal_df[other_predictors].values
        if X_other.shape[1] > 0:
            model_other = LinearRegression().fit(X_other, y)
            r2_other = model_other.score(X_other, y)
        else:
            r2_other = 0
            model_other = None
        
        # Model 3: Full model (demographics + other traits)
        X_full = causal_df[all_predictors].values
        if X_full.shape[1] > 0:
            model_full = LinearRegression().fit(X_full, y)
            r2_full = model_full.score(X_full, y)
        else:
            r2_full = 0
            model_full = None
        
        # Calculate unique contributions
        demo_unique = max(0, r2_full - r2_other)      # Variance explained by demographics beyond other traits
        other_unique = max(0, r2_full - r2_demo)      # Variance explained by other traits beyond demographics  
        shared = max(0, r2_demo + r2_other - r2_full) # Shared variance
        
        print(f"  Demographics only R²: {r2_demo:.3f}")
        print(f"  Other traits only R²: {r2_other:.3f}")
        print(f"  Full model R²: {r2_full:.3f}")
        print(f"  Demographics unique contribution: {demo_unique:.3f}")
        print(f"  Other traits unique contribution: {other_unique:.3f}")
        print(f"  Shared variance: {shared:.3f}")
        
        # Coefficient analysis for full model
        coefficients = {}
        if model_full is not None:
            for i, predictor in enumerate(all_predictors):
                coef = model_full.coef_[i]
                coefficients[predictor] = coef
                if abs(coef) > 0.01:  # Only report meaningful coefficients
                    print(f"    {predictor}: β={coef:.3f}")
        
        causal_results[outcome] = {
            'r2_demographics': r2_demo,
            'r2_other': r2_other,
            'r2_full': r2_full,
            'demo_unique': demo_unique,
            'other_unique': other_unique,
            'shared_variance': shared,
            'coefficients': coefficients,
            'models': {
                'demographics': model_demo,
                'other': model_other,
                'full': model_full
            }
        }
    
    # ========================================================================
    # STEP 3: DEMOGRAPHIC BIAS DETECTION
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("DEMOGRAPHIC BIAS DETECTION")
    print(f"{'='*60}")
    
    bias_detection = {}
    
    for outcome in outcomes:
        if outcome not in causal_results:
            continue
            
        results = causal_results[outcome]
        demo_unique = results.get('demo_unique', 0)
        other_unique = results.get('other_unique', 0)
        
        # Bias indicators
        demographic_dominance = demo_unique / (other_unique + 0.001)
        total_demographic_effect = results.get('r2_demographics', 0)
        
        # Classification of bias level
        if demographic_dominance > 2.0 and total_demographic_effect > 0.1:
            bias_level = "HIGH BIAS: Demographics strongly predict outcomes"
        elif demographic_dominance > 1.0 and total_demographic_effect > 0.05:
            bias_level = "MODERATE BIAS: Demographics have notable effects"
        elif total_demographic_effect > 0.02:
            bias_level = "LOW BIAS: Demographics have minor effects"
        else:
            bias_level = "NO BIAS: Demographics show minimal predictive power"
        
        # Identify most problematic demographic predictors
        coefficients = results.get('coefficients', {})
        demographic_coeffs = {k: v for k, v in coefficients.items() 
                            if any(k.startswith(f'{trait}_') for trait in demographic_traits)}
        
        strongest_demo_bias = None
        if demographic_coeffs:
            strongest_demo_bias = max(demographic_coeffs.items(), key=lambda x: abs(x[1]))
        
        bias_detection[outcome] = {
            'bias_level': bias_level,
            'demographic_dominance': demographic_dominance,
            'total_demographic_effect': total_demographic_effect,
            'strongest_bias': strongest_demo_bias,
            'demographic_coefficients': demographic_coeffs
        }
        
        print(f"\n{outcome.upper()}:")
        print(f"  {bias_level}")
        print(f"  Demographic dominance ratio: {demographic_dominance:.2f}")
        if strongest_demo_bias:
            print(f"  Strongest demographic effect: {strongest_demo_bias[0]} (β={strongest_demo_bias[1]:.3f})")
    
    # ========================================================================
    # STEP 4: CAUSAL PATH INTERPRETATION
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("CAUSAL PATH INTERPRETATION")
    print(f"{'='*60}")
    
    strongest_predictors = {}
    
    for outcome, results in causal_results.items():
        max_coef = 0
        strongest_predictor = None
    
        for predictor, coef in results['coefficients'].items():
            if abs(coef) > abs(max_coef):
                max_coef = coef
                strongest_predictor = predictor
    
        if strongest_predictor is None and results['coefficients']:
            strongest_predictor = list(results['coefficients'].keys())[0]
            max_coef = results['coefficients'][strongest_predictor]
    
        strongest_predictors[outcome] = {
            'predictor': strongest_predictor,
            'coefficient': max_coef,
            'direction': 'increases' if max_coef > 0 else 'decreases',
            'is_demographic': any(strongest_predictor.startswith(f'{trait}_') for trait in demographic_traits) if strongest_predictor else False
        }
    
        print(f"\n{outcome.upper()}:")
        if strongest_predictor:
            print(f"  Strongest predictor: {strongest_predictor} (β={max_coef:.3f})")
            print(f"  Effect: {strongest_predictor} {strongest_predictors[outcome]['direction']} {outcome}")
            if strongest_predictors[outcome]['is_demographic']:
                print(f"  WARNING: Strongest predictor is a DEMOGRAPHIC trait")
    
        # Compare demographic vs other trait contributions
        demo_unique = results.get('demo_unique', 0)
        other_unique = results.get('other_unique', 0)
    
        if demo_unique > other_unique:
            dominant_factor = "demographic traits"
            dominance_ratio = demo_unique / (other_unique + 0.001)
        else:
            dominant_factor = "non-demographic traits"
            dominance_ratio = other_unique / (demo_unique + 0.001)
    
        print(f"  Dominant factor: {dominant_factor} (ratio: {dominance_ratio:.2f})")
    
    # ========================================================================
    # STEP 5: BIAS MITIGATION RECOMMENDATIONS
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("BIAS MITIGATION RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Identify which traits to manipulate for desired outcomes while minimizing demographic bias
    recommendations = {}
    
    # For maximizing rescue rate while minimizing demographic bias
    rescue_predictors = causal_results.get('rescue_rate', {}).get('coefficients', {})
    non_demo_rescue_predictors = {k: v for k, v in rescue_predictors.items() 
                                 if not any(k.startswith(f'{trait}_') for trait in demographic_traits)}
    best_rescue_traits = sorted(non_demo_rescue_predictors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n=== TO MAXIMIZE RESCUE RATE (Moral Value) WITHOUT DEMOGRAPHIC BIAS:")
    if best_rescue_traits:
        for trait, coef in best_rescue_traits:
            if coef > 0.01:
                print(f"  Select profiles with: {trait} (β={coef:.3f})")
    else:
        print("  No strong non-demographic predictors found")
    
    recommendations['maximize_rescue_unbiased'] = best_rescue_traits
    
    # For minimizing extra errors while avoiding demographic bias
    error_predictors = causal_results.get('extra_error_rate', {}).get('coefficients', {})
    non_demo_error_predictors = {k: v for k, v in error_predictors.items() 
                                if not any(k.startswith(f'{trait}_') for trait in demographic_traits)}
    safest_traits = sorted(non_demo_error_predictors.items(), key=lambda x: x[1])[:3]
    
    print(f"\n=== TO MINIMIZE EXTRA ERRORS (Safety) WITHOUT DEMOGRAPHIC BIAS:")
    if safest_traits:
        for trait, coef in safest_traits:
            if coef < -0.01:
                print(f"  Select profiles with: {trait} (β={coef:.3f})")
            elif coef < 0.01:
                print(f"  Avoid profiles with: {trait} (β={coef:.3f})")
    else:
        print("  No strong non-demographic predictors found")
    
    recommendations['maximize_safety_unbiased'] = safest_traits
    
    # For maximizing overall accuracy without bias
    accuracy_predictors = causal_results.get('accuracy', {}).get('coefficients', {})
    non_demo_accuracy_predictors = {k: v for k, v in accuracy_predictors.items() 
                                   if not any(k.startswith(f'{trait}_') for trait in demographic_traits)}
    best_accuracy_traits = sorted(non_demo_accuracy_predictors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n=== TO MAXIMIZE ACCURACY (Performance) WITHOUT DEMOGRAPHIC BIAS:")
    if best_accuracy_traits:
        for trait, coef in best_accuracy_traits:
            if coef > 0.005:
                print(f"  Select profiles with: {trait} (β={coef:.3f})")
    else:
        print("  No strong non-demographic predictors found")
    
    recommendations['maximize_accuracy_unbiased'] = best_accuracy_traits
    
    # Special analysis for consistency if available
    if consistency_results and 'volatility' in causal_results:
        volatility_predictors = causal_results['volatility']['coefficients']
        non_demo_volatility_predictors = {k: v for k, v in volatility_predictors.items() 
                                         if not any(k.startswith(f'{trait}_') for trait in demographic_traits)}
        consistency_traits = sorted(non_demo_volatility_predictors.items(), key=lambda x: x[1])[:3]
        
        print(f"\n=== TO MAXIMIZE CONSISTENCY (Low Volatility) WITHOUT DEMOGRAPHIC BIAS:")
        if consistency_traits:
            for trait, coef in consistency_traits:
                if coef < -0.01:
                    print(f"  Select profiles with: {trait} (β={coef:.3f}) - reduces volatility")
                elif coef < 0.01:
                    print(f"  Avoid profiles with: {trait} (β={coef:.3f}) - increases volatility")
        else:
            print("  No strong non-demographic predictors found")
        
        recommendations['maximize_consistency_unbiased'] = consistency_traits
    
    # ========================================================================
    # STEP 6: THEORETICAL FRAMEWORK FOR BIAS
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("THEORETICAL CAUSAL FRAMEWORK FOR BIAS")
    print(f"{'='*60}")
    
    print(f"\n=== CAUSAL MECHANISM HYPOTHESIS:")
    print(f"  Profile Demographics → Bias Outcomes")
    print(f"  Profile Other Traits → Performance Outcomes")
    print(f"  Mediation: Other Traits may mediate or suppress demographic effects")
    
    # Test mediation hypothesis: Do other traits mediate demographic effects?
    mediation_evidence = {}

    for outcome in outcomes:
        if outcome not in causal_results:
            continue
            
        demo_direct = causal_results[outcome].get('demo_unique', 0)
        other_contribution = causal_results[outcome].get('other_unique', 0)
        shared_variance = causal_results[outcome].get('shared_variance', 0)
        
        # Mediation analysis
        if demo_direct > 0.001:
            mediation_strength = other_contribution / demo_direct
            suppression_strength = shared_variance / demo_direct
        else:
            mediation_strength = float('inf') if other_contribution > 0.01 else 0
            suppression_strength = 0
        
        if mediation_strength > 2.0:
            mediation_type = "Strong mediation: Other traits largely override demographic effects"
            policy_implication = "Focus on other traits to minimize demographic bias"
        elif mediation_strength > 1.0:
            mediation_type = "Partial mediation: Other traits somewhat offset demographic effects"
            policy_implication = "Balance demographic and other trait considerations"
        elif suppression_strength > 0.5:
            mediation_type = "Suppression: Other traits mask demographic effects"
            policy_implication = "Careful trait selection needed to avoid hidden bias"
        else:
            mediation_type = "Direct demographic effects: Other traits don't mediate bias"
            policy_implication = "Direct intervention needed to address demographic bias"
        
        mediation_evidence[outcome] = {
            'mediation_strength': mediation_strength,
            'suppression_strength': suppression_strength,
            'interpretation': mediation_type,
            'policy_implication': policy_implication
        }
        
        print(f"\n  {outcome.upper()}: {mediation_type}")
        print(f"    Mediation ratio: {mediation_strength:.2f}")
        print(f"    Policy implication: {policy_implication}")
    
    # ========================================================================
    # STEP 7: FINAL BIAS ASSESSMENT AND RECOMMENDATIONS
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("FINAL BIAS ASSESSMENT")
    print(f"{'='*60}")
    
    # Overall bias severity
    high_bias_outcomes = [outcome for outcome, detection in bias_detection.items() 
                         if 'HIGH BIAS' in detection['bias_level']]
    moderate_bias_outcomes = [outcome for outcome, detection in bias_detection.items() 
                             if 'MODERATE BIAS' in detection['bias_level']]
    
    if high_bias_outcomes:
        overall_assessment = "CRITICAL: High demographic bias detected"
        priority = "Immediate intervention required"
    elif moderate_bias_outcomes:
        overall_assessment = "WARNING: Moderate demographic bias detected"
        priority = "Bias mitigation recommended"
    else:
        overall_assessment = "ACCEPTABLE: Low or no demographic bias detected"
        priority = "Continue monitoring"
    
    print(f"\n=== OVERALL ASSESSMENT: {overall_assessment}")
    print(f"=== PRIORITY: {priority}")
    
    if high_bias_outcomes:
        print(f"\n=== HIGH BIAS OUTCOMES: {', '.join(high_bias_outcomes)}")
    if moderate_bias_outcomes:
        print(f"=== MODERATE BIAS OUTCOMES: {', '.join(moderate_bias_outcomes)}")
    
    # Key recommendations summary
    print(f"\n=== KEY RECOMMENDATIONS:")
    
    if high_bias_outcomes or moderate_bias_outcomes:
        print("  1. BIAS MITIGATION PRIORITY:")
        print("     - Use non-demographic traits for profile selection")
        print("     - Monitor demographic representation in selected profiles")
        print("     - Implement bias-aware evaluation metrics")
        
        # Find the most problematic demographic traits
        all_demo_effects = {}
        for outcome, detection in bias_detection.items():
            if detection['strongest_bias']:
                trait, effect = detection['strongest_bias']
                all_demo_effects[trait] = all_demo_effects.get(trait, 0) + abs(effect)
        
        if all_demo_effects:
            most_problematic = max(all_demo_effects.items(), key=lambda x: x[1])
            print(f"     - Pay special attention to {most_problematic[0]} effects")
    
    print("  2. OPTIMAL PROFILE SELECTION:")
    print("     - Prioritize profiles with high rescue rates from non-demographic traits")
    print("     - Balance consistency and boldness based on use case requirements")
    print("     - Regularly audit for emerging bias patterns")
    
    print("  3. ONGOING MONITORING:")
    print("     - Track demographic parity in model outcomes")
    print("     - Monitor for bias amplification over time")
    print("     - Validate bias mitigation effectiveness")
    
    return {
        'causal_data': causal_df,
        'causal_results': causal_results,
        'bias_detection': bias_detection,
        'strongest_predictors': strongest_predictors,
        'mediation_evidence': mediation_evidence,
        'recommendations': recommendations,
        'overall_assessment': {
            'bias_level': overall_assessment,
            'priority': priority,
            'high_bias_outcomes': high_bias_outcomes,
            'moderate_bias_outcomes': moderate_bias_outcomes
        },
        'theoretical_framework': {
            'hypothesis': "Profile Demographics vs Other Traits effects on Bias Outcomes",
            'mediation_support': mediation_evidence,
            'bias_mitigation_strategies': recommendations,
            'policy_implications': [evidence['policy_implication'] for evidence in mediation_evidence.values()]
        }
    }


def visualize_causal_model(causal_results, figsize=(10, 6)):
    """
    Create comprehensive visualizations for causal modeling results.
    
    Generates 4-panel plot:
    1. Variance Decomposition (Demographics vs Other Traits)
    2. Causal Path Strengths (Coefficient Heatmap)
    3. Mediation Analysis 
    4. Causal Network Diagram
    
    Updated to work with PersonSet structure and professional output.
    """
    
    causal_data = causal_results['causal_results']
    outcomes = list(causal_data.keys())

    # Collect predictors
    all_coeffs = list(next(iter(causal_data.values()))['coefficients'].keys()) if causal_data else []
    demographic_predictors = [p for p in all_coeffs if any(x in p for x in ['gender', 'ethnicity'])]
    other_predictors = [p for p in all_coeffs if p not in demographic_predictors]
    all_predictors = demographic_predictors + other_predictors

    def _new_fig(title):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle('Causal Modeling: Profile Traits Effects on Bias Outcomes', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=11)
        return fig, ax

    figs = {}


    # A) Variance Decomposition
    figA, ax = _new_fig('Variance Decomposition: Demographics vs Other Traits')
    demo_contrib = [causal_data[o].get('demo_unique', 0) for o in outcomes]
    other_contrib = [causal_data[o].get('other_unique', 0) for o in outcomes]
    shared_contrib = [causal_data[o].get('shared_variance', 0) for o in outcomes]
    x = np.arange(len(outcomes))
    width = 0.6
    ax.bar(x, demo_contrib, width, label='Demographics', color='#1f77b4', alpha=0.8)
    ax.bar(x, other_contrib, width, bottom=demo_contrib, label='Other Traits', color='#ff7f0e', alpha=0.8)
    ax.bar(x, shared_contrib, width, bottom=np.array(demo_contrib) + np.array(other_contrib), 
           label='Shared Variance', color='#2ca02c', alpha=0.8)
    for i, outcome in enumerate(outcomes):
        r2 = causal_data[outcome].get('r2_full', 0)
        ax.text(i, r2 + 0.01, f'R²={r2:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Variance Explained (R²)')
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figs['variance_decomposition'] = figA


    # B) Coefficient Heatmap
    figB, ax = _new_fig('Causal Path Coefficients (β)')
    if all_predictors and outcomes:
        coef_matrix = np.zeros((len(all_predictors), len(outcomes)))
        for j, outcome in enumerate(outcomes):
            for i, predictor in enumerate(all_predictors):
                coef_matrix[i, j] = causal_data[outcome]['coefficients'].get(predictor, 0)
        vmax = max(0.05, np.abs(coef_matrix).max() * 0.8)
        im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        for i in range(len(all_predictors)):
            for j in range(len(outcomes)):
                value = coef_matrix[i, j]
                text_color = "white" if abs(value) > vmax * 0.5 else "black"
                ax.text(j, i, f'{value:.1e}', ha="center", va="center",
                        color=text_color, fontweight='bold', fontsize=8)
        ax.set_xticks(np.arange(len(outcomes)))
        ax.set_yticks(np.arange(len(all_predictors)))
        ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes],
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([p.replace('_', ' ').title() for p in all_predictors], fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, format='%.1e').set_label('Coefficient Value')
    else:
        ax.text(0.5, 0.5, 'No coefficients available', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    figs['coef_heatmap'] = figB


    # C) Mediation Analysis
    figC, ax = _new_fig('Mediation Analysis: Other Traits vs Demographics')
    mediation = causal_results.get('mediation_evidence', {})
    if mediation:
        ratios = [mediation[o].get('mediation_strength', 0) for o in outcomes if o in mediation]
        valid_outcomes = [o for o in outcomes if o in mediation]
        if ratios:
            colors = ['#2ca02c' if r > 1.5 else '#ff7f0e' if r > 0.8 else '#d62728' for r in ratios]
            bars = ax.bar(range(len(valid_outcomes)), ratios, color=colors, alpha=0.7)
            ax.axhline(1.5, color='green', linestyle='--', label='Strong Mediation (>1.5)', alpha=0.7)
            ax.axhline(0.8, color='orange', linestyle='--', label='Partial Mediation (>0.8)', alpha=0.7)
            for i, (bar, r) in enumerate(zip(bars, ratios)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, f'{r:.2f}',
                        ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel('Mediation Ratio')
            ax.set_xticks(range(len(valid_outcomes)))
            ax.set_xticklabels([o.replace('_', ' ').title() for o in valid_outcomes], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No mediation data available', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No mediation analysis available', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    figs['mediation'] = figC


    # D) Causal Network
    figD, ax = _new_fig('Causal Network Structure')
    outcome_positions = [(0.8, y) for y in (0.9, 0.7, 0.5, 0.3, 0.1)]
    actual_node_pos = {'Demographics': (0.2, 0.8), 'Other Traits': (0.2, 0.4)}
    outcome_names = [o.replace('_', ' ').title() for o in outcomes]
    for i, outcome_name in enumerate(outcome_names):
        if i < len(outcome_positions):
            actual_node_pos[outcome_name] = outcome_positions[i]
    for node, (x, y) in actual_node_pos.items():
        if node == 'Demographics':
            color = '#1f77b4'; size = 1200
        elif node == 'Other Traits':
            color = '#ff7f0e'; size = 1200
        else:
            color = '#2ca02c'; size = 800
        ax.scatter(x, y, s=size, c=color, edgecolors='black', alpha=0.7, linewidth=2)
        ax.text(x, y-0.06, node, ha='center', va='top', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    strongest_predictors = causal_results.get('strongest_predictors', {})
    for outcome, pred_info in strongest_predictors.items():
        if not pred_info or not pred_info.get('predictor'):
            continue
        predictor = pred_info['predictor']
        coefficient = pred_info.get('coefficient', 0)
        outcome_display = outcome.replace('_', ' ').title()
        if outcome_display not in actual_node_pos:
            continue
        start_node = 'Demographics' if any(predictor.startswith(f'{t}_') for t in ['gender','ethnicity']) else 'Other Traits'
        start_pos = actual_node_pos[start_node]; end_pos = actual_node_pos[outcome_display]
        line_width = min(abs(coefficient) * 20, 4) + 0.5
        color = '#2ca02c' if coefficient > 0 else '#d62728'
        ax.annotate('', xy=end_pos, xytext=start_pos, 
                    arrowprops=dict(arrowstyle='->', lw=line_width, color=color, alpha=0.7))
    if 'Demographics' in actual_node_pos and 'Other Traits' in actual_node_pos:
        ax.annotate('', xy=actual_node_pos['Other Traits'], xytext=actual_node_pos['Demographics'],
                    arrowprops=dict(arrowstyle='->', lw=2, color='#1f77b4', alpha=0.5, linestyle='--'))
    ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.02, 0.02, 'Green = Positive Effect, Red = Negative Effect\nLine thickness ∝ Effect size',
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.9), 
            fontsize=8, va='bottom')
    plt.tight_layout()
    figs['causal_network'] = figD

    plt.show()

    return figs




def run_full_tier3_analysis(
    merged_df,
    person_set: PersonSet,
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = None, 
    n_folds=5
):
    """
    Run complete Tier 3 analysis pipeline.

    Executes:
    1. Consistency vs Boldness Analysis
    2. Simplified Causal Modeling with Bias Detection
    3. Visualizations
    4. Theoretical Integration for Research Conclusions
    5. Methodological Assessment

    Returns:
    Integrated results for thesis conclusions.
    """

    print("EXECUTING TIER 3 ANALYSIS PIPELINE")
    print("=" * 80)

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

    print(f"Cross-validation folds: {n_folds}")

    # ================================================================
    # STEP 1: CONSISTENCY vs BOLDNESS ANALYSIS
    # ================================================================
    print("\n=== Running Consistency vs Boldness Analysis ===")
    try:
        consistency_results = consistency_vs_boldness_analysis(
            merged_df,
            case=case,
            n_folds=n_folds,
            person_set=person_set,
            group_keys=group_keys
        )
        print("Consistency analysis completed successfully")
    except Exception as e:
        print(f"ERROR in consistency analysis: {e}")
        consistency_results = {'error': str(e)}

    # ================================================================
    # STEP 2: CAUSAL MODELING WITH BIAS DETECTION
    # ================================================================
    print("\n=== Running Simplified Causal Modeling ===")
    try:
        causal_results = simplified_causal_modeling(
            merged_df,
            person_set=person_set,
            case=case,
            group_keys=group_keys,
            consistency_results=(
                consistency_results if 'error' not in consistency_results else None
            )
        )
        print("Causal modeling completed successfully")
    except Exception as e:
        print(f"ERROR in causal modeling: {e}")
        causal_results = {'error': str(e)}

    # ================================================================
    # STEP 3: VISUALIZATIONS
    # ================================================================
    print("\n=== Creating Consistency Analysis Visualizations ===")
    consistency_viz = None
    if 'error' not in consistency_results:
        try:
            consistency_viz = plot_consistency_boldness_analysis(
                consistency_results,
                person_set=person_set,
                group_keys=group_keys
            )
            print("Consistency visualizations created successfully")
        except Exception as e:
            print(f"WARNING: Could not create consistency visualizations: {e}")

    print("\n=== Creating Causal Model Visualizations ===")
    causal_viz = None
    if 'error' not in causal_results:
        try:
            causal_viz = visualize_causal_model(causal_results)
            print("Causal visualizations created successfully")
        except Exception as e:
            print(f"WARNING: Could not create causal visualizations: {e}")

    # ================================================================
    # STEP 4: THEORETICAL INTEGRATION
    # ================================================================
    print("\n" + "=" * 80)
    print("TIER 3 THEORETICAL INTEGRATION")
    print("=" * 80)

    consistency_insight = (
        consistency_results.get('normative_assessment', {})
        if 'error' not in consistency_results else {}
    )
    high_vol_rescue = consistency_insight.get('overall', {}).get('high_volatility_rescue', 0)
    low_vol_rescue = consistency_insight.get('overall', {}).get('low_volatility_rescue', 0)

    causal_insight = (
        causal_results.get('theoretical_framework', {})
        if 'error' not in causal_results else {}
    )
    strongest_predictors = (
        causal_results.get('strongest_predictors', {})
        if 'error' not in causal_results else {}
    )
    bias_detection = (
        causal_results.get('bias_detection', {})
        if 'error' not in causal_results else {}
    )

    print("\n=== THEORETICAL INSIGHTS:")

    # Insight 1: Consistency-Boldness Tradeoff
    if high_vol_rescue and low_vol_rescue:
        if high_vol_rescue > low_vol_rescue + 0.01:
            volatility_conclusion = (
                "Inconsistent profiles provide higher moral value through increased rescue rates"
            )
            volatility_implication = (
                "Controlled inconsistency in AI systems may be normatively justified"
            )
        else:
            volatility_conclusion = (
                "Consistent profiles match or outperform inconsistent ones in moral value"
            )
            volatility_implication = (
                "Consistency should be prioritized over boldness in AI systems"
            )

        print(f"   - {volatility_conclusion}")
        print(f"   - Implication: {volatility_implication}")
        print(f"   - Evidence: Inconsistent rescue rate: {high_vol_rescue:.3f} vs Consistent: {low_vol_rescue:.3f}")
    else:
        volatility_conclusion = "Insufficient data for consistency-boldness analysis"
        print(f"   - {volatility_conclusion}")

    # Insight 2: Causal Mechanisms and Bias Detection
    if strongest_predictors:
        rescue_predictor = strongest_predictors.get('rescue_rate', {}).get('predictor')
        accuracy_predictor = strongest_predictors.get('accuracy', {}).get('predictor')

        print(f"   - Strongest predictor of moral value (rescue): {rescue_predictor or 'None'}")
        print(f"   - Strongest predictor of performance (accuracy): {accuracy_predictor or 'None'}")

        demographic_traits = ['gender', 'ethnicity']
        if any(
            rescue_predictor and rescue_predictor.startswith(f"{t}_")
            for t in demographic_traits
        ) or any(
            accuracy_predictor and accuracy_predictor.startswith(f"{t}_")
            for t in demographic_traits
        ):
            print("   - WARNING: Demographic traits strongly influence key outcomes")

    # Insight 3: Bias Severity
    if bias_detection:
        high_bias_count = sum(
            1 for d in bias_detection.values()
            if 'HIGH BIAS' in d.get('bias_level', '')
        )
        moderate_bias_count = sum(
            1 for d in bias_detection.values()
            if 'MODERATE BIAS' in d.get('bias_level', '')
        )

        print(f"   - Bias assessment: {high_bias_count} high, {moderate_bias_count} moderate bias cases detected")

    # ================================================================
    # STEP 5: THESIS-LEVEL CONCLUSIONS
    # ================================================================
    print(f"\n{'=' * 60}")
    print("THESIS-LEVEL CONCLUSIONS")
    print(f"{'=' * 60}")

    conclusions = []

    if high_vol_rescue and low_vol_rescue:
        if high_vol_rescue > low_vol_rescue + 0.02:
            conclusions.append({
                'finding': 'Inconsistent profiles provide superior moral value',
                'evidence': f'{high_vol_rescue:.3f} vs {low_vol_rescue:.3f} rescue rates',
                'implication': 'Controlled inconsistency can improve moral outcomes',
                'strength': 'Strong' if high_vol_rescue > low_vol_rescue + 0.05 else 'Moderate'
            })
        else:
            conclusions.append({
                'finding': 'Consistency and inconsistency yield similar moral value',
                'evidence': f'Diff: {abs(high_vol_rescue - low_vol_rescue):.3f}',
                'implication': 'Prefer consistency for predictability',
                'strength': 'Moderate'
            })

    if bias_detection:
        demo_outcomes = len([
            d for d in bias_detection.values()
            if 'HIGH BIAS' in d.get('bias_level', '') or 'MODERATE BIAS' in d.get('bias_level', '')
        ])
        total_outcomes = len(bias_detection)
        if demo_outcomes > total_outcomes / 2:
            conclusions.append({
                'finding': 'Demographics significantly influence AI bias patterns',
                'evidence': f'{demo_outcomes}/{total_outcomes} biased outcomes',
                'implication': 'Diversity and bias mitigation required',
                'strength': 'Strong'
            })

    # ================================================================
    # STEP 6: METHODOLOGICAL ASSESSMENT
    # ================================================================
    print(f"\n{'=' * 60}")
    print("METHODOLOGICAL ASSESSMENT")
    print(f"{'=' * 60}")

    method_assessment = {
        'data_quality': 'Good' if len(merged_df) > 100 else 'Limited',
        'statistical_power': 'Adequate' if 'error' not in consistency_results and 'error' not in causal_results else 'Insufficient',
        'bias_detection_capability': 'Strong' if bias_detection else 'Limited',
        'generalizability': 'Domain-specific' if person_set else 'Unknown'
    }

    for aspect, assessment in method_assessment.items():
        print(f"   - {aspect.replace('_', ' ').title()}: {assessment}")

    valid_profiles = consistency_results.get('valid_profiles', []) if 'error' not in consistency_results else []
    print(f"   - Profile coverage: {len(valid_profiles)} profiles")
    if group_keys:
        print(f"   - Trait coverage: {len(group_keys)} ({', '.join(group_keys)})")

    return {
        'consistency_analysis': consistency_results,
        'causal_analysis': causal_results,
        'visualizations': {
            'consistency_plot': consistency_viz,
            'causal_plot': causal_viz
        },
        'theoretical_integration': {
            'volatility_conclusion': volatility_conclusion if 'volatility_conclusion' in locals() else None,
            'causal_mechanisms': strongest_predictors,
            'bias_assessment': bias_detection
        },
        'thesis_conclusions': conclusions,
        'methodological_assessment': method_assessment,
        'analysis_parameters': {
            'group_keys': group_keys,
            'n_folds': n_folds,
            'dataset_size': len(merged_df)
        }
    }
