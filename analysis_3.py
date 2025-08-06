import os
import glob
import json
import re
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any
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
from profiles.profile_message import get_profile_traits
from profiles.profile_sets import PERSON_SYSTEMATIC
from profiles.schema import PersonSet, PersonMeta




def temporal_stability_vs_boldness_analysis(merged_df, 
                                            n_folds=5,
                                            person_set: PersonSet = PERSON_SYSTEMATIC,
                                            ):
    """
    Temporal Stability vs Boldness Tradeoff Analysis
    
    Tests if *unstable profiles* actually make better moral calls by examining:
    - Volatility (std dev across cross-validation runs)  
    - Boldness (rescue rate, bias magnitude)
    - Moral value (alignment with normative judgments)
    
    Reveals whether *risk-taking* profiles are **normatively valuable** 
    despite being volatile.
    """
    
    print("=" * 80)
    print("TEMPORAL STABILITY vs BOLDNESS TRADEOFF ANALYSIS")
    print("=" * 80)
    
    pattern = re.compile(r"^profile\d+_(passive|active)$")
    profile_cols = [col for col in merged_df.columns if pattern.match(col) and col.replace("_passive", "") in person_set.metadata]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    stability_data = {}
    
    print(f"=== Calculating temporal stability across {n_folds} folds...")
    
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
        
        # Calculate stability metrics
        stability_data[profile] = {
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




    # step 2: Define Boldness Metrics
    
    print("=== Calculating boldness metrics...")
    
    # Get rescue stats and bias patterns for boldness calculation
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    
    boldness_data = {}
    
    for profile in profile_cols:
        if profile not in stability_data:
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
            0.3 * avg_mislabelling_rate +        # 30% disagreement rate  
            0.3 * avg_bias_magnitude     # 30% position strength
        )
        
        boldness_data[profile] = {
            'rescue_rate': avg_rescue_rate,
            'mislabelling_rate': avg_mislabelling_rate,
            'bias_magnitude': avg_bias_magnitude,
            'boldness_score': boldness_score
        }
    



    # step 3: Stability vs Boldness Correlation Analysis
    
    print("=== Analyzing stability-boldness correlations...")
    
    # Prepare data for correlation analysis
    profiles = []
    volatilities = []
    boldness_scores = []
    rescue_rates = []
    accuracy_means = []
    
    for profile in profile_cols:
        if profile in stability_data and profile in boldness_data:
            profiles.append(profile)
            volatilities.append(stability_data[profile]['accuracy_std'])
            boldness_scores.append(boldness_data[profile]['boldness_score'])
            rescue_rates.append(boldness_data[profile]['rescue_rate'])
            accuracy_means.append(stability_data[profile]['accuracy_mean'])
    
    # Calculate correlations
    correlations = {}
    
    if len(volatilities) >= 3:  # Need at least 3 points for meaningful correlation
        
        # Volatility vs Boldness
        corr_vol_bold, p_val_vol_bold = pearsonr(volatilities, boldness_scores)
        correlations['volatility_vs_boldness'] = {
            'correlation': corr_vol_bold,
            'p_value': p_val_vol_bold,
            'significant': p_val_vol_bold < 0.05
        }
        
        # Volatility vs Rescue Rate (moral value proxy)
        corr_vol_rescue, p_val_vol_rescue = pearsonr(volatilities, rescue_rates)
        correlations['volatility_vs_rescue'] = {
            'correlation': corr_vol_rescue,
            'p_value': p_val_vol_rescue,
            'significant': p_val_vol_rescue < 0.05
        }
        
        # Boldness vs Accuracy (performance tradeoff)
        corr_bold_acc, p_val_bold_acc = pearsonr(boldness_scores, accuracy_means)
        correlations['boldness_vs_accuracy'] = {
            'correlation': corr_bold_acc,
            'p_value': p_val_bold_acc,
            'significant': p_val_bold_acc < 0.05
        }
        
        print(f"\n--- CORRELATION RESULTS:")
        print(f"  Volatility vs Boldness: r={corr_vol_bold:.3f}, p={p_val_vol_bold:.4f} {'***' if p_val_vol_bold < 0.05 else ''}")
        print(f"  Volatility vs Rescue Rate: r={corr_vol_rescue:.3f}, p={p_val_vol_rescue:.4f} {'***' if p_val_vol_rescue < 0.05 else ''}")
        print(f"  Boldness vs Accuracy: r={corr_bold_acc:.3f}, p={p_val_bold_acc:.4f} {'***' if p_val_bold_acc < 0.05 else ''}")
    
    # ========================================================================
    # STEP 4: Profile Classification and Insights
    # ========================================================================
    
    print(f"\n=== PROFILE CLASSIFICATION:")
    profile_archetypes = {}
    
    volatility_median = np.median(volatilities) if volatilities else 0
    boldness_median = np.median(boldness_scores) if boldness_scores else 0
    
    for i, profile in enumerate(profiles):
        vol = volatilities[i]
        bold = boldness_scores[i]
        rescue = rescue_rates[i]
        acc = accuracy_means[i]
        
        if vol > volatility_median and bold > boldness_median:
            archetype = "Volatile Bold" 
            description = "High risk, high moral value"
        elif vol < volatility_median and bold > boldness_median:
            archetype = "Stable Bold"
            description = "Best performance, low risk"
        elif vol > volatility_median and bold < boldness_median:
            archetype = "Volatile Cautious"
            description = "High risk, low moral value"
        else:
            archetype = "Stable Cautious"
            description = "Low risk, predictable"
        
        profile_archetypes[profile] = {
            'archetype': archetype,
            'description': description,
            'volatility': vol,
            'boldness': bold,
            'rescue_rate': rescue,
            'accuracy': acc
        }
        
        traits = person_set.get_traits(profile)
        trait_str = ", ".join(f"{k}={v}" for k, v in traits.items())
        print(f"  {profile.replace('_passive', '')} ({trait_str}): {archetype} – {description}")    


    # step 5: Normative Value Assessment
    print(f"\n=== NORMATIVE VALUE ASSESSMENT:")
    
    # Test the key hypothesis: Do volatile profiles provide moral value?
    high_volatility_profiles = [p for p, data in profile_archetypes.items() 
                               if data['volatility'] > volatility_median]
    low_volatility_profiles = [p for p, data in profile_archetypes.items() 
                              if data['volatility'] <= volatility_median]
    
    if high_volatility_profiles and low_volatility_profiles:
        high_vol_rescue = [profile_archetypes[p]['rescue_rate'] for p in high_volatility_profiles]
        low_vol_rescue = [profile_archetypes[p]['rescue_rate'] for p in low_volatility_profiles]
        
        high_vol_rescue_mean = np.mean(high_vol_rescue)
        low_vol_rescue_mean = np.mean(low_vol_rescue)
        
        t_stat, p_val = ttest_ind(high_vol_rescue, low_vol_rescue)
        
        print(f"  High Volatility Profiles Rescue Rate: {high_vol_rescue_mean:.3f}")
        print(f"  Low Volatility Profiles Rescue Rate: {low_vol_rescue_mean:.3f}")
        print(f"  Difference: {high_vol_rescue_mean - low_vol_rescue_mean:.3f}")
        print(f"  Statistical Test: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
        
        if high_vol_rescue_mean > low_vol_rescue_mean and p_val < 0.05:
            conclusion = "Volatile profiles provide significant moral value advantage."
        elif high_vol_rescue_mean > low_vol_rescue_mean:
            conclusion = "Volatile profiles show higher rescue rates but not statistically significant."
        else:
            conclusion = "Volatile profiles do not provide moral value advantage."
        
        print(f"\n=== CONCLUSION: {conclusion}")
    
    return {
        'stability_data': stability_data,
        'boldness_data': boldness_data,
        'correlations': correlations,
        'profile_archetypes': profile_archetypes,
        'normative_assessment': {
            'high_volatility_rescue': high_vol_rescue_mean if 'high_vol_rescue_mean' in locals() else 0,
            'low_volatility_rescue': low_vol_rescue_mean if 'low_vol_rescue_mean' in locals() else 0, 
            'statistical_test': {'t_stat': t_stat, 'p_value': p_val} if 't_stat' in locals() else None
        }
    }


def plot_stability_boldness_analysis(stability_results, figsize=(16, 10)):
    """
    Create comprehensive visualizations for temporal stability vs boldness analysis.
    
    Generates 4-panel plot:
    1. Stability vs Boldness Scatter
    2. Volatility vs Rescue Rate  
    3. Profile Archetype Distribution
    4. Temporal Stability Trends
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Temporal Stability vs Boldness Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    stability_data = stability_results['stability_data']
    boldness_data = stability_results['boldness_data']
    archetypes = stability_results['profile_archetypes']
    
    # Prepare arrays
    profiles = list(archetypes.keys())
    volatilities = [archetypes[p]['volatility'] for p in profiles]
    boldness_scores = [archetypes[p]['boldness'] for p in profiles]
    rescue_rates = [archetypes[p]['rescue_rate'] for p in profiles]
    accuracies = [archetypes[p]['accuracy'] for p in profiles]
    
    # Archetype colors
    archetype_colors = {
        'Volatile Bold': '#d62728',
        'Stable Bold': '#2ca02c',
        'Volatile Cautious': '#ff7f0e',
        'Stable Cautious': '#1f77b4'
    }
    
    colors = [archetype_colors.get(archetypes[p]['archetype'], '#8c564b') for p in profiles]
    
    # ===== PANEL 1: Stability vs Boldness Scatter =====
    ax = axes[0, 0]
    
    scatter = ax.scatter(volatilities, boldness_scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add profile labels for extreme cases
    for i, profile in enumerate(profiles):
        if volatilities[i] > np.percentile(volatilities, 75) or boldness_scores[i] > np.percentile(boldness_scores, 75):
            ax.annotate(profile.replace('profile', 'P').replace('_passive', ''), 
                       (volatilities[i], boldness_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add correlation line if significant
    if 'volatility_vs_boldness' in stability_results['correlations']:
        corr_data = stability_results['correlations']['volatility_vs_boldness']
        if corr_data['significant']:
            z = np.polyfit(volatilities, boldness_scores, 1)
            p = np.poly1d(z)
            ax.plot(sorted(volatilities), p(sorted(volatilities)), "r--", alpha=0.8)
            ax.text(0.05, 0.95, f"r={corr_data['correlation']:.3f}*", 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Volatility (Accuracy Std Dev)')
    ax.set_ylabel('Boldness Score')
    ax.set_title('Stability vs Boldness Tradeoff')
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 2: Volatility vs Rescue Rate =====
    ax = axes[0, 1]
    
    scatter = ax.scatter(volatilities, rescue_rates, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add correlation line if significant
    if 'volatility_vs_rescue' in stability_results['correlations']:
        corr_data = stability_results['correlations']['volatility_vs_rescue']
        if corr_data['significant']:
            z = np.polyfit(volatilities, rescue_rates, 1)
            p = np.poly1d(z)
            ax.plot(sorted(volatilities), p(sorted(volatilities)), "g--", alpha=0.8)
            ax.text(0.05, 0.95, f"r={corr_data['correlation']:.3f}*", 
                   transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Volatility (Accuracy Std Dev)')
    ax.set_ylabel('Rescue Rate (Moral Value)')
    ax.set_title('Volatility vs Moral Value')
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 3: Profile Archetype Distribution =====
    ax = axes[1, 0]
    
    archetype_counts = {}
    for profile_data in archetypes.values():
        arch = profile_data['archetype']
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
    
    archetype_names = list(archetype_counts.keys())
    counts = list(archetype_counts.values())
    colors_pie = [archetype_colors.get(arch, '#8c564b') for arch in archetype_names]
    
    wedges, texts, autotexts = ax.pie(counts, labels=archetype_names, colors=colors_pie, 
                                     autopct='%1.0f%%', startangle=90)
    ax.set_title('Profile Archetype Distribution')
    
    # ===== PANEL 4: Temporal Stability Trends =====
    ax = axes[1, 1]
    
    # Show accuracy trends across folds for a few representative profiles
    representative_profiles = []
    
    # Get one profile from each archetype if possible
    for archetype in archetype_colors.keys():
        for profile, data in archetypes.items():
            if data['archetype'] == archetype and profile not in representative_profiles:
                representative_profiles.append(profile)
                break
    
    # Limit to 4 most interesting profiles
    representative_profiles = representative_profiles[:4]
    
    fold_numbers = list(range(1, len(list(stability_data.values())[0]['fold_accuracies']) + 1))
    
    for i, profile in enumerate(representative_profiles):
        if profile in stability_data:
            fold_accs = stability_data[profile]['fold_accuracies']
            if fold_accs:
                archetype = archetypes[profile]['archetype']
                color = archetype_colors.get(archetype, '#8c564b')
            
                ax.plot(fold_numbers, fold_accs, 'o-', color=color, alpha=0.7, 
                       label=f"{profile.replace('profile', 'P').replace('_passive', '')} ({archetype})")
                    
    ax.set_xlabel('Cross-Validation Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Temporal Stability Across Folds')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def simplified_causal_modeling(merged_df, person_set: PersonSet, stability_results=None):
    """
    Simplified Causal Modeling for Bias Effects
    
    Models causal influence of profile traits on bias/alignment using:
    - Linear regression with demographic and other traits predictors
    - Path analysis showing direct vs indirect effects
    - Theoretical framework for understanding bias mechanisms
    
    Gives theoretical weight to findings for thesis contribution.
    """
    
    print("=" * 80)
    print("SIMPLIFIED CAUSAL MODELING: PROFILE TRAITS → BIAS OUTCOMES")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Prepare Causal Variables
    # ========================================================================
    
    profile_cols = [col for col in merged_df.columns if col.startswith("profile") and "_passive" in col]
    
    # Create causal dataset
    causal_data = []

    # Collect all unique traits across profiles
    all_trait_keys = set()
    for profile in profile_cols:
        traits = person_set.get_traits(profile)
        for trait, value in traits.items():
            if value != "Unknown":
                all_trait_keys.add(f"{trait}_{value}")
    
    # Get performance metrics
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    
    for profile_name in profile_cols:
        if profile_name not in merged_df.columns:
            continue

        traits = person_set.get_traits(profile_name)
        causal_row = {"profile": profile_name}

        for k, v in traits.items():
            causal_row[f"raw_{k}"] = v
        
        for trait, value in traits.items():
            if value != "Unknown":
                causal_row[f"{trait}_{value}"] = 1

        for key in all_trait_keys:
            causal_row.setdefault(key, 0)
        
        accuracy = (merged_df[profile_name] == merged_df['true_label']).mean()
        profile_rescue = rescue_stats[rescue_stats['profile'] == profile_name]
        rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
        extra_error_rate = profile_rescue['extra_err_rate'].mean() if len(profile_rescue) > 0 else 0
        profile_bias = bias_patterns[bias_patterns['profile'] == profile_name]
        bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
        mislabelling_rate = profile_bias['mislabelling_rate'].mean() if len(profile_bias) > 0 else 0
        volatility = stability_results['stability_data'][profile_name]['accuracy_std'] if stability_results and profile_name in stability_results['stability_data'] else 0
        
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
    
    print(f"📊 Causal dataset prepared with {len(causal_df)} profiles")
    
    # ========================================================================
    # STEP 2: Linear Regression Models
    # ========================================================================
    
    print(f"\n🔗 CAUSAL PATH ANALYSIS:")
    
    # Define predictor sets
    trait_keys = [f.name for f in fields(PersonMeta)]
    trait_prefixes = [f"{key}_" for key in trait_keys]

    trait_predictors = sorted([
        col for col in causal_df.columns
        if any(col.startswith(prefix) for prefix in trait_prefixes)
    ])
    
    # Group predictors for causal models
    demographic_keys = ['gender', 'ethnicity']
    demographic_prefixes = [f"{key}_" for key in demographic_keys]
    demographic_predictors = [col for col in trait_predictors if any(col.startswith(prefix) for prefix in demographic_prefixes)]
    other_predictors = [col for col in trait_predictors if col not in demographic_predictors]

    # Full predictor list
    all_predictors = demographic_predictors + other_predictors
    
    # Outcome variables to model
    outcomes = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude']
    
    causal_results = {}
    
    for outcome in outcomes:
        print(f"\n--- MODELING: {outcome.upper()} ---")
        
        y = causal_df[outcome].values
        
        # Model 1: Demographics only
        X_demo = causal_df[demographic_predictors].values
        model_demo = LinearRegression().fit(X_demo, y)
        r2_demo = model_demo.score(X_demo, y)
        
        # Model 2: Other traits than demographics (gender, ethnicity) only
        X_other = causal_df[other_predictors].values
        model_other = LinearRegression().fit(X_other, y)
        r2_other = model_other.score(X_other, y)
        
        # Model 3: Full model (demographics + other traits)
        X_full = causal_df[all_predictors].values
        model_full = LinearRegression().fit(X_full, y)
        r2_full = model_full.score(X_full, y)
        
        # Calculate unique contributions
        demo_unique = r2_full - r2_other       # Variance explained by demographics beyond other traits
        other_unique = r2_full - r2_demo         # Variance explained by other traits beyond demographics  
        shared = r2_demo + r2_other - r2_full  # Shared variance
        
        print(f"  Demographics only R^2: {r2_demo:.3f}")
        print(f"  Other traits only R^2: {r2_other:.3f}")
        print(f"  Full model R^2: {r2_full:.3f}")
        print(f"  Demographics unique contribution: {demo_unique:.3f}")
        print(f"  Other traits unique contribution: {other_unique:.3f}")
        print(f"  Shared variance: {shared:.3f}")
        
        # Coefficient analysis for full model
        coefficients = {}
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
    # STEP 3: Causal Path Interpretation
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
            'direction': 'increases' if max_coef > 0 else 'decreases'
        }
    
        print(f"\n{outcome.upper()}:")
        print(f"  Strongest predictor: {strongest_predictor} (β={max_coef:.3f})")
        print(f"  Effect: {strongest_predictor} {strongest_predictors[outcome]['direction']} {outcome}")
    
        # Generalized comparison of contributions
        demo_unique = results.get('demo_unique', 0)
        other_unique = results.get('other_unique', 0)
    
        if demo_unique > other_unique:
            dominant_factor = "demographic traits"
            dominance_ratio = demo_unique / other_unique if other_unique > 0 else float('inf')
        else:
            dominant_factor = "non-demographic traits"
            dominance_ratio = other_unique / demo_unique if demo_unique > 0 else float('inf')
    
        print(f"  Dominant factor: {dominant_factor} (ratio: {dominance_ratio:.2f})")
    


    # ========================================================================
    # STEP 4: Theoretical Framework
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("THEORETICAL CAUSAL FRAMEWORK")
    print(f"{'='*60}")
    
    print(f"\n=== CAUSAL MECHANISM HYPOTHESIS:")
    print(f"  Profile Demographics and other traits Processing effect on Bias Outcomes")
    
    # Test mediation hypothesis: Do other traits mediate demographic effects?
    mediation_evidence = {}

    for outcome in outcomes:
        demo_direct = causal_results[outcome].get('demo_unique', 0)
        other_contribution = causal_results[outcome].get('other_unique', 0)
        total_variance = causal_results[outcome].get('r2_full', 0)
        
        # Simple mediation indicator: other traits factors explain more than demographics
        mediation_strength = other_contribution / (demo_direct+0.001)
        
        if mediation_strength > 1.5:
            mediation_type = "Strong mediation: Other traits largely mediate demographic effects"
        elif mediation_strength > 0.8:
            mediation_type = "Partial mediation: Both demographics and other traits matter"
        else:
            mediation_type = "Direct effects: Demographics dominate over other traits"
        
        mediation_evidence[outcome] = {
            'mediation_strength': mediation_strength,
            'interpretation': mediation_type
        }
        
        print(f"\n  {outcome.upper()}: {mediation_type}")
        print(f"    Mediation ratio: {mediation_strength:.2f}")
    
    # ========================================================================
    # STEP 5: Causal Recommendations
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("CAUSAL-BASED RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Identify which traits to manipulate for desired outcomes
    recommendations = {}
    
    # For maximizing rescue rate (moral value)
    rescue_predictors = causal_results['rescue_rate']['coefficients']
    best_rescue_traits = sorted(rescue_predictors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n=== TO MAXIMIZE RESCUE RATE (Moral Value):")
    for trait, coef in best_rescue_traits:
        if coef > 0.01:
            print(f"  ✓ Select profiles with: {trait} (β={coef:.3f})")
    
    recommendations['maximize_rescue'] = best_rescue_traits
    
    # For minimizing extra errors (safety)
    error_predictors = causal_results['extra_error_rate']['coefficients']
    safest_traits = sorted(error_predictors.items(), key=lambda x: x[1])[:3]
    
    print(f"\n=== TO MINIMIZE EXTRA ERRORS (Safety):")
    for trait, coef in safest_traits:
        if coef<(-0.01):
            print(f"  ✓ Select profiles with: {trait} (β={coef:.3f})")
        elif coef<0.01:
            print(f"  ✓ Avoid profiles with: {trait} (β={coef:.3f})")
    
    recommendations['maximize_safety'] = safest_traits
    
    # For maximizing overall accuracy
    accuracy_predictors = causal_results['accuracy']['coefficients']
    best_accuracy_traits = sorted(accuracy_predictors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n=== TO MAXIMIZE ACCURACY (Performance):")
    for trait, coef in best_accuracy_traits:
        if coef > 0.005:
            print(f"  ✓ Select profiles with: {trait} (β={coef:.3f})")
    
    recommendations['maximize_accuracy'] = best_accuracy_traits
    
    return {
        'causal_data': causal_df,
        'causal_results': causal_results,
        'strongest_predictors': strongest_predictors,
        'mediation_evidence': mediation_evidence,
        'recommendations': recommendations,
        'theoretical_framework': {
            'hypothesis': "Profile Demographics and Profile Traits effects on Bias Outcomes",
            'mediation_support': mediation_evidence,
            'policy_implications': recommendations
        }
    }


def visualize_causal_model(causal_results, figsize=(16, 12)):
    """
    Create comprehensive visualizations for causal modeling results.
    
    Generates 4-panel plot:
    1. Variance Decomposition (Demographics vs Other Traits)
    2. Causal Path Strengths (Coefficient Heatmap)
    3. Mediation Analysis 
    4. Causal Network Diagram
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Causal Modeling: Profile Traits effect on Bias Outcomes', fontsize=16, fontweight='bold')
    
    causal_data = causal_results['causal_results']
    outcomes = list(causal_data.keys())

    all_coeffs = list(next(iter(causal_data.values()))['coefficients'].keys())
    demographic_predictors = [p for p in all_coeffs if any(x in p for x in ['gender', 'ethnicity'])]
    other_predictors = [p for p in all_coeffs if p not in demographic_predictors]
    all_predictors = demographic_predictors + other_predictors
    
    
    # ===== PANEL 1: Variance Decomposition =====
    ax = axes[0, 0]
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
        ax.text(i, r2 + 0.01, f'R^2={r2:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Variance Explained (R^2)')
    ax.set_title('Variance Decomposition')
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 2: Coefficient Heatmap =====
    ax = axes[0, 1]
    coef_matrix = np.zeros((len(all_predictors), len(outcomes)))

    for j, outcome in enumerate(outcomes):
        for i, predictor in enumerate(all_predictors):
            coef_matrix[i, j] = causal_data[outcome]['coefficients'].get(predictor, 0)

    im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    
    for i in range(len(all_predictors)):
        for j in range(len(outcomes)):
            value = coef_matrix[i, j]
            ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                    color="white" if abs(value) > 0.05 else "black", fontweight='bold')

    ax.set_xticks(np.arange(len(outcomes)))
    ax.set_yticks(np.arange(len(all_predictors)))
    ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=45, ha='right')
    ax.set_yticklabels([p.replace('_', ' ').title() for p in all_predictors])
    ax.set_title('Causal Coefficients (β)')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Coefficient')
    


    # ===== PANEL 3: Mediation Analysis =====
    ax = axes[1, 0]
    mediation = causal_results['mediation_evidence']
    ratios = [mediation[o]['mediation_strength'] for o in outcomes]
    colors = ['#2ca02c' if r > 1.5 else '#ff7f0e' if r > 0.8 else '#d62728' for r in ratios]

    bars = ax.bar(range(len(outcomes)), ratios, color=colors, alpha=0.7)
    ax.axhline(1.5, color='green', linestyle='--', label='Strong Mediation')
    ax.axhline(0.8, color='orange', linestyle='--', label='Partial Mediation')

    for i, (bar, r) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{r:.2f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Mediation Ratio (Other / Demographics)')
    ax.set_title('Mediation Analysis')
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)



    # ===== PANEL 4: Simplified Causal Network =====
    ax = axes[1, 1]
    node_pos = {
        'Demographics': (0.2, 0.8),
        'Other Traits': (0.2, 0.5),
        'Accuracy': (0.8, 0.9),
        'Rescue Rate': (0.8, 0.7),
        'Extra Error Rate': (0.8, 0.5),
        'Bias Magnitude': (0.8, 0.3)
    }

    for node, (x, y) in node_pos.items():
        color = '#1f77b4' if node == 'Demographics' else '#ff7f0e' if node == 'Other Traits' else '#2ca02c'
        size = 1000 if node in ['Demographics', 'Other Traits'] else 800
        ax.scatter(x, y, s=size, c=color, edgecolors='black', alpha=0.7, linewidth=2)
        ax.text(x, y-0.05, node, ha='center', va='top', fontweight='bold', fontsize=9)

    # Arrows for strongest paths
    for outcome, pred_info in causal_results['strongest_predictors'].items():
        predictor = pred_info['predictor']
        if not predictor:
            continue
        outcome_node = node_pos.get(outcome.replace('_', ' ').title())
        start_node = 'Other Traits' if predictor in other_predictors else 'Demographics'
        start_pos = node_pos[start_node]
        line_width = min(abs(pred_info['coefficient']) * 50, 5)
        ax.annotate('', xy=outcome_node, xytext=start_pos, 
                    arrowprops=dict(arrowstyle='->', lw=max(line_width, 0.5),
                                    color='green' if pred_info['coefficient'] > 0 else 'red', alpha=0.6))

    # Static Demographics → Other link
    ax.annotate('', xy=node_pos['Other Traits'], xytext=node_pos['Demographics'],
                arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.5))

    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title('Causal Network')
    ax.text(0.05, 0.05, 'Green = Positive, Red = Negative, Thickness ∝ Effect',
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


def run_full_tier3_analysis(merged_df,
                            person_set: PersonSet,
                            ):
    """
    Run complete Tier 3 analysis pipeline.
    
    Executes:
    1. Temporal Stability vs Boldness Analysis
    2. Simplified Causal Modeling
    3. Comprehensive Visualizations
    4. Theoretical Integration
    
    Returns integrated results for thesis conclusions.
    """
    
    print("🚀 EXECUTING TIER 3 ANALYSIS PIPELINE")
    print("="*80)
    
    # 1. Temporal Stability vs Boldness Analysis
    print("\n === Running Temporal Stability vs Boldness Analysis... === ")
    stability_results = temporal_stability_vs_boldness_analysis(merged_df, n_folds=5, person_set=person_set)
    
    # 2. Simplified Causal Modeling
    print("\n === Running Simplified Causal Modeling... === ")
    causal_results = simplified_causal_modeling(merged_df, person_set, stability_results)
    
    # 3. Generate Visualizations
    print("\n === Creating Temporal Stability Visualizations... ===")
    stability_viz = plot_stability_boldness_analysis(stability_results)
    
    print("\n === Creating Causal Model Visualizations... ===")
    causal_viz = visualize_causal_model(causal_results)

    print("\n" + "="*80)
    print("TIER 3 THEORETICAL INTEGRATION")
    print("="*80)
    
    stability_insight = stability_results['normative_assessment']
    high_vol_rescue = stability_insight.get('high_volatility_rescue', 0)
    low_vol_rescue = stability_insight.get('low_volatility_rescue', 0)
    
    causal_insight = causal_results['theoretical_framework']
    strongest_predictors = causal_results['strongest_predictors']
    
    print(f"\n=== THEORETICAL INSIGHTS:")
    
    # Volatility-Boldness Finding
    if high_vol_rescue > low_vol_rescue:
        volatility_conclusion = "--- Volatile profiles provide higher moral value through increased rescue rates"
        volatility_implication = "Risk-taking in AI annotation may be normatively justified"
    else:
        volatility_conclusion = "--- Stable profiles outperform volatile ones in moral value"
        volatility_implication = "Consistency should be prioritized over boldness in AI systems"
    
    print(f"   - {volatility_conclusion}")
    print(f"   - Implication: {volatility_implication}")
    
    # Causal Mechanism Finding
    rescue_predictor = strongest_predictors.get('rescue_rate', {}).get('predictor', 'Unknown')
    accuracy_predictor = strongest_predictors.get('accuracy', {}).get('predictor', 'Unknown')
    
    # Handle None predictors
    if rescue_predictor is None:
        rescue_predictor = 'No significant predictor'
    if accuracy_predictor is None:
        accuracy_predictor = 'No significant predictor'
    
    print(f"   - Strongest predictor of moral value (rescue): {rescue_predictor}")
    print(f"   - Strongest predictor of performance (accuracy): {accuracy_predictor}")
    
    # Mediation findings
    mediation_summary = []
    for outcome, evidence in causal_results['mediation_evidence'].items():
        if evidence['mediation_strength'] > 1.5:
            mediation_summary.append(f"{outcome}: Strong other traits mediation")
        elif evidence['mediation_strength'] > 0.8:
            mediation_summary.append(f"{outcome}: Partial mediation")
    
    if mediation_summary:
        print(f"   - Mediation effects found: {', '.join(mediation_summary)}")
    
    # ========================================================================
    # STEP 5: Thesis-Level Conclusions
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("THESIS-LEVEL CONCLUSIONS")
    print(f"{'='*60}")
    
    conclusions = []
    
    # Conclusion 1: Stability-Boldness Tradeoff
    if high_vol_rescue>low_vol_rescue+0.02:
        conclusions.append({
            'finding': 'Volatile profiles provide superior moral value',
            'evidence': f'High-volatility rescue rate: {high_vol_rescue:.3f} vs Low-volatility: {low_vol_rescue:.3f}',
            'implication': 'AI systems should incorporate controlled risk-taking for better moral outcomes'
        })
    
    # Conclusion 2: Causal Mechanisms
    demo_dominance = []
    other_dominance = []
    
    for outcome, results in causal_results['causal_results'].items():
        if results['demo_unique'] > results['other_unique']:
            demo_dominance.append(outcome)
        else:
            other_dominance.append(outcome)
    
    if len(other_dominance) > len(demo_dominance):
        conclusions.append({
            'finding': 'Other traits styles dominate over demographics in bias formation',
            'evidence': f'Other traits factors dominate in {len(other_dominance)}/{len(causal_results["causal_results"])} outcomes',
            'implication': 'Bias mitigation should focus on other traits framing rather than demographic representation'
        })
    else:
        conclusions.append({
            'finding': 'Demographics remain primary drivers of bias',
            'evidence': f'Demographic factors dominate in {len(demo_dominance)}/{len(causal_results["causal_results"])} outcomes',
            'implication': 'Diverse demographic representation is crucial for bias mitigation'
        })
    
    # Conclusion 3: System Design Recommendations
    recommendations = causal_results['recommendations']
    
    # Find the most recommended trait across all objectives
    trait_counts = {}
    for objective, traits in recommendations.items():
        for trait, coef in traits:
            if trait is not None:
                trait_counts[trait] = trait_counts.get(trait, 0) + 1
    
    most_important_trait = max(trait_counts, key=trait_counts.get) if trait_counts else 'No clear pattern'
    
    conclusions.append({
        'finding': f'{most_important_trait} is the most critical trait for system design',
        'evidence': f'Appears in {trait_counts.get(most_important_trait, 0)}/3 optimization objectives',
        'implication': f'Systems should prioritize profiles with {most_important_trait} characteristics'
    })
    
    # Print conclusions
    for i, conclusion in enumerate(conclusions, 1):
        print(f"\n=== CONCLUSION {i}: {conclusion['finding']}")
        print(f"   Evidence: {conclusion['evidence']}")
        print(f"   Implication: {conclusion['implication']}")
    
    return {
        'stability_analysis': stability_results,
        'causal_analysis': causal_results,
        'visualizations': {
            'stability_plot': stability_viz,
            'causal_plot': causal_viz
        },
        'theoretical_integration': {
            'volatility_conclusion': volatility_conclusion,
            'causal_mechanisms': strongest_predictors,
            'mediation_evidence': causal_results['mediation_evidence']
        },
        'thesis_conclusions': conclusions
    }