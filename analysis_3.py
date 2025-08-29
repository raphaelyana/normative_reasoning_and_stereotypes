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
from analysis_tools import get_analysis_group_keys, get_available_traits, resolve_plot_dir
from profiles.profile_sets import PERSON_ETHNICS
from profiles.schema import PersonSet, PersonMeta
from cases.cases_config import CaseConfig



def consistency_vs_boldness_analysis(merged_df, 
                                     case: CaseConfig,
                                     n_folds=5,
                                     person_set: PersonSet = PERSON_ETHNICS,
                                     group_keys=("gender", "ethnicity", "age"),
                                     perf_df: Optional[pd.DataFrame] = None):
    """
    Consistency vs Boldness Tradeoff Analysis — with token/cost metrics
    
    New in this version:
      - Accepts perf_df with per-profile 'tokens_per_sample' and 'cost_per_sample'
      - Carries tokens/cost into the analysis_df
      - Adds efficiency features (rescue/accuracy per 1k tokens and per $)
    """
    print("=" * 80)
    print("CONSISTENCY vs BOLDNESS TRADEOFF ANALYSIS")
    print("=" * 80)
    print(f"Group keys for analysis: {group_keys}")
    
    # helper: valid profiles in PersonSet
    def get_valid_profiles(merged_df, person_set):
        all_profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        valid_profiles = [col for col in all_profile_cols if col in person_set.metadata]
        print(f"Found {len(valid_profiles)} valid profiles out of {len(all_profile_cols)} total profile columns")
        return valid_profiles
    
    profile_cols = get_valid_profiles(merged_df, person_set)
    if len(profile_cols) == 0:
        print("ERROR: No valid profiles found in PersonSet metadata")
        return {'error': 'No valid profiles found'}

    # ------------------------------------------------------------------------
    # Token/cost maps (optional)
    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        perf = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in perf.columns:
            tokens_map = perf["tokens_per_sample"].to_dict()
        if "cost_per_sample" in perf.columns:
            cost_map = perf["cost_per_sample"].to_dict()
        have_tokens = len(tokens_map) > 0
        have_cost   = len(cost_map) > 0
        print(f"Token/cost availability → tokens:{have_tokens} cost:{have_cost}")
    else:
        have_tokens = have_cost = False

    # ========================================================================
    # STEP 2: CONSISTENCY ACROSS CV FOLDS
    # ========================================================================
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    consistency_data = {}
    print(f"=== Calculating consistency across {n_folds} folds...")

    for profile in profile_cols:
        if profile not in merged_df.columns:
            continue
            
        fold_accuracies, fold_rescue_rates, fold_bias_magnitudes = [], [], []
        for _, (_, test_idx) in enumerate(kf.split(merged_df)):
            test_data = merged_df.iloc[test_idx]
            acc = accuracy_score(test_data['true_label'], test_data[profile])
            fold_accuracies.append(acc)
            base_correct = (test_data['base_pred'] == test_data['true_label'])
            profile_correct = (test_data[profile] == test_data['true_label'])
            rescued = ((~base_correct) & profile_correct).sum()
            base_errors = (~base_correct).sum()
            rescue_rate = rescued / base_errors if base_errors > 0 else 0.0
            fold_rescue_rates.append(rescue_rate)
            to_positive = ((test_data['base_pred'] == "no") & (test_data[profile] == "yes")).sum()
            to_negative = ((test_data['base_pred'] == "yes") & (test_data[profile] == "no")).sum()
            bias_magnitude = abs(to_positive - to_negative) / len(test_data)
            fold_bias_magnitudes.append(bias_magnitude)
        
        consistency_data[profile] = {
            'accuracy_mean': float(np.mean(fold_accuracies)),
            'accuracy_std':  float(np.std(fold_accuracies)),
            'accuracy_cv':   float(np.std(fold_accuracies) / np.mean(fold_accuracies)) if np.mean(fold_accuracies) > 0 else np.inf,
            'rescue_rate_mean': float(np.mean(fold_rescue_rates)),
            'rescue_rate_std':  float(np.std(fold_rescue_rates)),
            'rescue_rate_cv':   float(np.std(fold_rescue_rates) / np.mean(fold_rescue_rates)) if np.mean(fold_rescue_rates) > 0 else np.inf,
            'bias_magnitude_mean': float(np.mean(fold_bias_magnitudes)),
            'bias_magnitude_std':  float(np.std(fold_bias_magnitudes)),
            'fold_accuracies': fold_accuracies,
            'fold_rescue_rates': fold_rescue_rates,
            'fold_bias_magnitudes': fold_bias_magnitudes,
            # attach tokens/cost if available (constant per profile)
            'tokens_per_sample': float(tokens_map.get(profile, np.nan)) if have_tokens else np.nan,
            'cost_per_sample':   float(cost_map.get(profile, np.nan)) if have_cost else np.nan,
        }

    # ========================================================================
    # STEP 3: BOLDNESS METRICS
    # ========================================================================
    print("=== Calculating boldness metrics...")
    try:
        category_cols = getattr(case, "category_cols", None) or ["stereotype_type"]
        rescue_stats_list, bias_patterns_list = [], []
        from analysis_tools import guarded_labelspace_analysis
        for cat_col in category_cols:
            if cat_col in merged_df.columns:
                rs = guarded_labelspace_analysis(
                    rescue_stats_by_category,
                    merged_df,
                    case=case,
                    category_col=cat_col,
                    person_set=person_set,
                    group_keys=group_keys
                ); rs["category_col"] = cat_col; rescue_stats_list.append(rs)
                bp = guarded_labelspace_analysis(
                    detect_systematic_biases,
                    merged_df,
                    case=case,
                    category_col=cat_col,
                    person_set=person_set,
                ); bp["category_col"] = cat_col; bias_patterns_list.append(bp)
        if not rescue_stats_list: raise ValueError("No valid category columns found in merged_df.")
        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True)
    except Exception as e:
        print(f"WARNING: Could not calculate rescue/bias stats: {e}")
        rescue_stats, bias_patterns = [], []
        for profile in profile_cols:
            base_correct = (merged_df['base_pred'] == merged_df['true_label'])
            profile_correct = (merged_df[profile] == merged_df['true_label'])
            rescued = ((~base_correct) & profile_correct).sum()
            base_errors = (~base_correct).sum()
            rescue_rate = rescued / base_errors if base_errors > 0 else 0
            rescue_stats.append({'profile': profile, 'rescue_rate': rescue_rate})
            to_positive = ((merged_df['base_pred'] == "no") & (merged_df[profile] == "yes")).sum()
            to_negative = ((merged_df['base_pred'] == "yes") & (merged_df[profile] == "no")).sum()
            bias_magnitude = abs(to_positive - to_negative) / len(merged_df)
            mislabelling_rate = (merged_df[profile] != merged_df['base_pred']).mean()
            bias_patterns.append({'profile': profile, 'bias_magnitude': bias_magnitude, 'mislabelling_rate': mislabelling_rate})
        rescue_stats = pd.DataFrame(rescue_stats)
        bias_patterns = pd.DataFrame(bias_patterns)
    
    boldness_data = {}
    for profile in profile_cols:
        if profile not in consistency_data: continue
        rs = rescue_stats[rescue_stats['profile'] == profile]
        bp = bias_patterns[bias_patterns['profile'] == profile]
        avg_rescue_rate = rs['rescue_rate'].mean() if len(rs) > 0 else 0.0
        avg_mislabelling_rate = bp['mislabelling_rate'].mean() if len(bp) > 0 else 0.0
        avg_bias_magnitude   = bp['bias_magnitude'].mean() if len(bp) > 0 else 0.0
        boldness_score = 0.4*avg_rescue_rate + 0.3*avg_mislabelling_rate + 0.3*avg_bias_magnitude
        # efficiency features if token/cost available
        toks = consistency_data[profile].get('tokens_per_sample', np.nan)
        cost = consistency_data[profile].get('cost_per_sample', np.nan)
        tok_k = toks/1000.0 if np.isfinite(toks) and toks>0 else np.nan
        cost_d = cost if np.isfinite(cost) and cost>0 else np.nan
        rescue_per_1k_tokens = (avg_rescue_rate / tok_k) if np.isfinite(tok_k) else np.nan
        extra_err_rate = max(0.0, consistency_data[profile]['rescue_rate_mean'] - avg_rescue_rate)  # simple proxy if detailed missing
        extra_err_per_1k_tokens = (extra_err_rate / tok_k) if np.isfinite(tok_k) else np.nan
        rescue_per_dollar = (avg_rescue_rate / cost_d) if np.isfinite(cost_d) else np.nan
        boldness_data[profile] = {
            'rescue_rate': avg_rescue_rate,
            'mislabelling_rate': avg_mislabelling_rate,
            'bias_magnitude': avg_bias_magnitude,
            'boldness_score': boldness_score,
            'rescue_per_1k_tokens': rescue_per_1k_tokens,
            'extra_err_per_1k_tokens': extra_err_per_1k_tokens,
            'rescue_per_dollar': rescue_per_dollar
        }

    # ========================================================================
    # STEP 4: DEMOGRAPHIC-AWARE CORRELATION DATASET (now with tokens/cost)
    # ========================================================================
    print("=== Analyzing consistency-boldness correlations by demographics...")
    analysis_data = []
    for profile in profile_cols:
        if profile in consistency_data and profile in boldness_data:
            traits = person_set.get_traits(profile, group_keys)
            toks = consistency_data[profile].get('tokens_per_sample', np.nan)
            cost = consistency_data[profile].get('cost_per_sample', np.nan)
            row = {
                'profile': profile,
                'volatility': consistency_data[profile]['accuracy_std'],
                'consistency': 1 / (1 + consistency_data[profile]['accuracy_std']),
                'boldness_score': boldness_data[profile]['boldness_score'],
                'rescue_rate': boldness_data[profile]['rescue_rate'],
                'accuracy_mean': consistency_data[profile]['accuracy_mean'],
                'bias_magnitude': boldness_data[profile]['bias_magnitude'],
                'mislabelling_rate': boldness_data[profile]['mislabelling_rate'],
                'tokens_per_sample': toks,
                'cost_per_sample': cost,
                'rescue_per_1k_tokens': boldness_data[profile]['rescue_per_1k_tokens'],
                'rescue_per_dollar': boldness_data[profile]['rescue_per_dollar'],
                'extra_err_per_1k_tokens': boldness_data[profile]['extra_err_per_1k_tokens'],
            }
            for trait_name in group_keys:
                row[trait_name] = traits.get(trait_name, "Unknown")
            analysis_data.append(row)
    analysis_df = pd.DataFrame(analysis_data)
    if len(analysis_df) == 0:
        print("ERROR: No data available for correlation analysis")
        return {'error': 'No analysis data available'}
    print(f"Analysis dataset shape: {analysis_df.shape}")

    # overall correlations (add token/cost angles if available)
    correlations = {}
    if len(analysis_df) >= 3:
        corr_vol_bold, p_val_vol_bold = pearsonr(analysis_df['volatility'], analysis_df['boldness_score'])
        correlations['volatility_vs_boldness'] = {'correlation': corr_vol_bold, 'p_value': p_val_vol_bold, 'significant': p_val_vol_bold < 0.05,
                                                  'interpretation': 'Positive correlation means less consistent profiles are bolder'}
        corr_cons_bold, p_val_cons_bold = pearsonr(analysis_df['consistency'], analysis_df['boldness_score'])
        correlations['consistency_vs_boldness'] = {'correlation': corr_cons_bold, 'p_value': p_val_cons_bold, 'significant': p_val_cons_bold < 0.05,
                                                   'interpretation': 'Negative correlation means more consistent profiles are less bold'}
        corr_vol_rescue, p_val_vol_rescue = pearsonr(analysis_df['volatility'], analysis_df['rescue_rate'])
        correlations['volatility_vs_rescue'] = {'correlation': corr_vol_rescue, 'p_value': p_val_vol_rescue, 'significant': p_val_vol_rescue < 0.05}
        corr_bold_acc, p_val_bold_acc = pearsonr(analysis_df['boldness_score'], analysis_df['accuracy_mean'])
        correlations['boldness_vs_accuracy'] = {'correlation': corr_bold_acc, 'p_value': p_val_bold_acc, 'significant': p_val_bold_acc < 0.05}

        if analysis_df['tokens_per_sample'].notna().sum() >= 3:
            r_tok_rescue, p_tok_rescue = pearsonr(analysis_df['tokens_per_sample'].fillna(0), analysis_df['rescue_rate'].fillna(0))
            correlations['tokens_vs_rescue'] = {'correlation': r_tok_rescue, 'p_value': p_tok_rescue, 'significant': p_tok_rescue < 0.05}
            r_tok_acc, p_tok_acc = pearsonr(analysis_df['tokens_per_sample'].fillna(0), analysis_df['accuracy_mean'].fillna(0))
            correlations['tokens_vs_accuracy'] = {'correlation': r_tok_acc, 'p_value': p_tok_acc, 'significant': p_tok_acc < 0.05}
        if analysis_df['cost_per_sample'].notna().sum() >= 3:
            r_cost_rescue, p_cost_rescue = pearsonr(analysis_df['cost_per_sample'].fillna(0), analysis_df['rescue_rate'].fillna(0))
            correlations['cost_vs_rescue'] = {'correlation': r_cost_rescue, 'p_value': p_cost_rescue, 'significant': p_cost_rescue < 0.05}
            r_cost_acc, p_cost_acc = pearsonr(analysis_df['cost_per_sample'].fillna(0), analysis_df['accuracy_mean'].fillna(0))
            correlations['cost_vs_accuracy'] = {'correlation': r_cost_acc, 'p_value': p_cost_acc, 'significant': p_cost_acc < 0.05}

        print(f"\n--- OVERALL CORRELATION RESULTS:")
        print(f"  Volatility vs Boldness: r={corr_vol_bold:.3f}, p={p_val_vol_bold:.4f} {'***' if p_val_vol_bold < 0.05 else ''}")
        print(f"  Consistency vs Boldness: r={corr_cons_bold:.3f}, p={p_val_cons_bold:.4f} {'***' if p_val_cons_bold < 0.05 else ''}")
        print(f"  Volatility vs Rescue Rate: r={corr_vol_rescue:.3f}, p={p_val_vol_rescue:.4f} {'***' if p_val_vol_rescue < 0.05 else ''}")
        print(f"  Boldness vs Accuracy: r={corr_bold_acc:.3f}, p={p_val_bold_acc:.4f} {'***' if p_val_bold_acc < 0.05 else ''}")
        if 'tokens_vs_rescue' in correlations:
            print(f"  Tokens vs Rescue: r={correlations['tokens_vs_rescue']['correlation']:.3f}, p={correlations['tokens_vs_rescue']['p_value']:.4f} {'***' if correlations['tokens_vs_rescue']['p_value'] < 0.05 else ''}")
        if 'cost_vs_rescue' in correlations:
            print(f"  Cost vs Rescue: r={correlations['cost_vs_rescue']['correlation']:.3f}, p={correlations['cost_vs_rescue']['p_value']:.4f} {'***' if correlations['cost_vs_rescue']['p_value'] < 0.05 else ''}")

    # demographic-specific correlations (unchanged)
    demographic_correlations = {}
    for trait_name in group_keys:
        if trait_name not in analysis_df.columns: continue
        valid_values = [v for v in analysis_df[trait_name].dropna().unique() if v != "Unknown"]
        if not valid_values: continue
        print(f"\n--- CORRELATIONS BY {trait_name.upper()}:")
        demographic_correlations[trait_name] = {}
        for value in valid_values:
            subset = analysis_df[analysis_df[trait_name] == value]
            if len(subset) >= 3:
                try:
                    corr_vol_bold, p_val = pearsonr(subset['volatility'], subset['boldness_score'])
                    demographic_correlations[trait_name][str(value)] = {'volatility_vs_boldness': {'correlation': corr_vol_bold, 'p_value': p_val, 'n': len(subset), 'significant': p_val < 0.05}}
                    print(f"  {value} (n={len(subset)}): Volatility vs Boldness r={corr_vol_bold:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
                except:
                    print(f"  {value} (n={len(subset)}): Could not calculate correlation")
            else:
                print(f"  {value} (n={len(subset)}): Insufficient data for correlation")

    # archetypes, bias detection, normative assessment (unchanged except printing)
    print(f"\n=== PROFILE CLASSIFICATION BY DEMOGRAPHICS:")
    profile_archetypes = {}
    volatility_median = analysis_df['volatility'].median()
    boldness_median = analysis_df['boldness_score'].median()
    archetype_by_demographics = {t:{} for t in group_keys if t in analysis_df.columns}

    for _, row in analysis_df.iterrows():
        profile = row['profile']; vol = row['volatility']; bold = row['boldness_score']
        rescue = row['rescue_rate']; acc = row['accuracy_mean']
        if vol > volatility_median and bold > boldness_median:
            archetype, description = "Inconsistent Bold", "High volatility, high moral value"
        elif vol < volatility_median and bold > boldness_median:
            archetype, description = "Consistent Bold", "Best performance, low volatility"
        elif vol > volatility_median and bold < boldness_median:
            archetype, description = "Inconsistent Cautious", "High volatility, low moral value"
        else:
            archetype, description = "Consistent Cautious", "Low volatility, predictable"
        profile_archetypes[profile] = {'archetype': archetype, 'description': description, 'volatility': vol, 'boldness': bold, 'rescue_rate': rescue, 'accuracy': acc}
        for trait_name in group_keys:
            if trait_name in row and row[trait_name] != "Unknown":
                trait_val = str(row[trait_name])
                archetype_by_demographics.setdefault(trait_name, {}).setdefault(trait_val, {}).setdefault(archetype, []).append(profile)
        traits = {k: row[k] for k in group_keys if k in row and row[k] != "Unknown"}
        trait_str = ", ".join(f"{k}={v}" for k, v in traits.items())
        print(f"  {profile} ({trait_str}): {archetype} – {description}")

    print(f"\n=== DEMOGRAPHIC BIAS DETECTION IN ARCHETYPES:")
    archetype_bias_analysis = {}
    for trait_name, groups in archetype_by_demographics.items():
        print(f"\n--- {trait_name.upper()} BIAS ANALYSIS:")
        archetype_bias_analysis[trait_name] = {}
        for trait_val, archetypes in groups.items():
            total_profiles = sum(len(profiles) for profiles in archetypes.values())
            print(f"  {trait_val} (n={total_profiles}):")
            archetype_bias_analysis[trait_name][trait_val] = {}
            for archetype, profiles_list in archetypes.items():
                percentage = (len(profiles_list) / total_profiles) * 100 if total_profiles > 0 else 0
                archetype_bias_analysis[trait_name][trait_val][archetype] = {'count': len(profiles_list), 'percentage': percentage, 'profiles': profiles_list}
                print(f"    {archetype}: {len(profiles_list)} ({percentage:.1f}%)")

    print(f"\n=== NORMATIVE VALUE ASSESSMENT BY DEMOGRAPHICS:")
    normative_assessment = {}
    high_volatility_profiles = analysis_df[analysis_df['volatility'] > volatility_median]
    low_volatility_profiles  = analysis_df[analysis_df['volatility'] <= volatility_median]
    if len(high_volatility_profiles) > 0 and len(low_volatility_profiles) > 0:
        high_vol_rescue_mean = high_volatility_profiles['rescue_rate'].mean()
        low_vol_rescue_mean  = low_volatility_profiles['rescue_rate'].mean()
        t_stat, p_val = ttest_ind(high_volatility_profiles['rescue_rate'], low_volatility_profiles['rescue_rate'])
        normative_assessment['overall'] = {
            'high_volatility_rescue': high_vol_rescue_mean,
            'low_volatility_rescue':  low_vol_rescue_mean,
            'difference': high_vol_rescue_mean - low_vol_rescue_mean,
            'statistical_test': {'t_stat': t_stat, 'p_value': p_val, 'significant': p_val < 0.05}
        }
        print(f"  OVERALL:")
        print(f"    Inconsistent Profiles Rescue Rate: {high_vol_rescue_mean:.3f} (n={len(high_volatility_profiles)})")
        print(f"    Consistent Profiles Rescue Rate: {low_vol_rescue_mean:.3f} (n={len(low_volatility_profiles)})")
        print(f"    Difference: {high_vol_rescue_mean - low_vol_rescue_mean:.3f}")
        print(f"    Statistical Test: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")

    for trait_name in group_keys:
        if trait_name not in analysis_df.columns: continue
        valid_values = [v for v in analysis_df[trait_name].dropna().unique() if v != "Unknown"]
        if not valid_values: continue
        print(f"\n  BY {trait_name.upper()}:")
        normative_assessment[trait_name] = {}
        for value in valid_values:
            subset = analysis_df[analysis_df[trait_name] == value]
            if len(subset) >= 4:
                subset_high = subset[subset['volatility'] > subset['volatility'].median()]
                subset_low  = subset[subset['volatility'] <= subset['volatility'].median()]
                if len(subset_high) > 0 and len(subset_low) > 0:
                    high_rescue = subset_high['rescue_rate'].mean()
                    low_rescue  = subset_low['rescue_rate'].mean()
                    try:
                        t_stat, p_val = ttest_ind(subset_high['rescue_rate'], subset_low['rescue_rate'])
                        normative_assessment[trait_name][str(value)] = {
                            'high_volatility_rescue': high_rescue,
                            'low_volatility_rescue':  low_rescue,
                            'difference': high_rescue - low_rescue,
                            'statistical_test': {'t_stat': t_stat, 'p_value': p_val, 'significant': p_val < 0.05}
                        }
                        print(f"    {value}: Inconsistent Rescue={high_rescue:.3f} vs Consistent Rescue={low_rescue:.3f}, diff={high_rescue - low_rescue:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
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
    Visuals incl. token-aware sizing:
      - Scatter bubbles are sized by tokens_per_sample when available
    """
    consistency_data = consistency_results['consistency_data']
    boldness_data = consistency_results['boldness_data']
    archetypes = consistency_results['profile_archetypes']
    analysis_df = consistency_results.get('analysis_df', pd.DataFrame())
    
    profiles = list(archetypes.keys())
    volatilities = [archetypes[p]['volatility'] for p in profiles]
    boldness_scores = [archetypes[p]['boldness'] for p in profiles]
    rescue_rates = [archetypes[p]['rescue_rate'] for p in profiles]
    accuracies = [archetypes[p]['accuracy'] for p in profiles]
    # bubble size by tokens (fallback to constant)
    sizes = []
    for p in profiles:
        toks = consistency_data.get(p, {}).get('tokens_per_sample', np.nan)
        if np.isfinite(toks) and toks > 0:
            sizes.append(60 + toks * 0.08)  # mild scaling
        else:
            sizes.append(100)

    archetype_colors = {
        'Inconsistent Bold': '#d62728',
        'Consistent Bold': '#2ca02c',
        'Inconsistent Cautious': '#ff7f0e',
        'Consistent Cautious': '#1f77b4'
    }
    colors = [archetype_colors.get(archetypes[p]['archetype'], '#8c564b') for p in profiles]
    
    def _new_fig(title):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle('Consistency vs Boldness Analysis', fontsize=14, fontweight='bold')
        ax.set_title(title)
        return fig, ax
    
    figs = {}

    # A: Consistency vs Boldness
    figA, ax = _new_fig('Consistency vs Boldness (size ∝ tokens/sample)')
    if person_set and len(analysis_df) > 0:
        primary_trait = group_keys[0] if group_keys else 'gender'
        if primary_trait in analysis_df.columns:
            trait_values = [v for v in analysis_df[primary_trait].unique() if v != "Unknown"]
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            trait_markers = {val: markers[i % len(markers)] for i, val in enumerate(trait_values)}
            for trait_val in trait_values:
                mask = analysis_df[primary_trait] == trait_val
                subset = analysis_df[mask]
                if len(subset) > 0:
                    vols = subset['volatility'].values
                    bolds = subset['boldness_score'].values
                    subsizes = []
                    for prof in subset['profile'].values:
                        toks = consistency_data.get(prof, {}).get('tokens_per_sample', np.nan)
                        subsizes.append(60 + toks * 0.08 if np.isfinite(toks) and toks > 0 else 100)
                    trait_colors = [archetype_colors.get(archetypes[p]['archetype'], '#8c564b') for p in subset['profile'].values]
                    ax.scatter(vols, bolds, c=trait_colors, s=subsizes, alpha=0.7, 
                               marker=trait_markers[trait_val], edgecolors='black', linewidth=1,
                               label=f'{primary_trait}={trait_val}')
    else:
        ax.scatter(volatilities, boldness_scores, c=colors, s=sizes, alpha=0.7, edgecolors='black')

    # annotate notable profiles
    for i, profile in enumerate(profiles):
        if volatilities[i] > np.percentile(volatilities, 80) or boldness_scores[i] > np.percentile(boldness_scores, 80):
            label = profile.replace('profile', 'P')
            ax.annotate(label, (volatilities[i], boldness_scores[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)

    if 'volatility_vs_boldness' in consistency_results['correlations'] and consistency_results['correlations']['volatility_vs_boldness']['significant']:
        z = np.polyfit(volatilities, boldness_scores, 1); p = np.poly1d(z)
        ax.plot(sorted(volatilities), p(sorted(volatilities)), "r--", alpha=0.8)
        ax.text(0.05, 0.95, f"r={consistency_results['correlations']['volatility_vs_boldness']['correlation']:.3f}*", 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    ax.set_xlabel('Volatility (Higher = Less Consistent)'); ax.set_ylabel('Boldness Score'); ax.grid(True, alpha=0.3)
    if person_set and len(analysis_df) > 0: ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout(); figs['consistency_vs_boldness'] = figA

    # B: Volatility vs Rescue (bubble size by tokens)
    figB, ax = _new_fig('Consistency vs Moral Value (size ∝ tokens/sample)')
    ax.scatter(volatilities, rescue_rates, c=colors, s=sizes, alpha=0.7, edgecolors='black')
    if 'volatility_vs_rescue' in consistency_results['correlations'] and consistency_results['correlations']['volatility_vs_rescue']['significant']:
        z = np.polyfit(volatilities, rescue_rates, 1); p = np.poly1d(z)
        ax.plot(sorted(volatilities), p(sorted(volatilities)), "g--", alpha=0.8)
        ax.text(0.05, 0.95, f"r={consistency_results['correlations']['volatility_vs_rescue']['correlation']:.3f}*", 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    ax.set_xlabel('Volatility (Higher = Less Consistent)'); ax.set_ylabel('Rescue Rate (Moral Value)'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); figs['volatility_vs_rescue'] = figB

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
                               consistency_results=None,
                               perf_df: Optional[pd.DataFrame] = None,
                               enable_compute_analysis: bool = True):

    print("=" * 80)
    print("SIMPLIFIED CAUSAL MODELING: PROFILE TRAITS → BIAS/EFFICIENCY OUTCOMES")
    print("=" * 80)
    print(f"Group keys: {group_keys}")

    def get_valid_profiles(merged_df, person_set):
        all_profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        return [col for col in all_profile_cols if col in person_set.metadata]
    
    profile_cols = get_valid_profiles(merged_df, person_set)
    if len(profile_cols) == 0:
        print("ERROR: No valid profiles found in PersonSet metadata")
        return {'error': 'No valid profiles found'}
    print(f"Found {len(profile_cols)} valid profiles for causal modeling")

    # token/cost maps
    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        perf = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in perf.columns:
            tokens_map = perf["tokens_per_sample"].to_dict()
        if "cost_per_sample" in perf.columns:
            cost_map = perf["cost_per_sample"].to_dict()
    have_tokens = len(tokens_map) > 0
    have_cost   = len(cost_map) > 0

    # collect unique trait keys and reference categories
    all_trait_keys, trait_reference_categories = set(), {}
    for profile in profile_cols:
        traits = person_set.get_traits(profile, group_keys)
        for trait_name, value in traits.items():
            if value != "Unknown":
                if trait_name == "age" and hasattr(value, 'value'):
                    all_trait_keys.add(f"{trait_name}_{value.value}")
                else:
                    all_trait_keys.add(f"{trait_name}_{value}")
    for trait_name in group_keys:
        trait_values = [key.split('_', 1)[1] for key in all_trait_keys if key.startswith(f"{trait_name}_")]
        if trait_values:
            trait_values_sorted = sorted(trait_values)
            ref = f"{trait_name}_{trait_values_sorted[0]}"
            trait_reference_categories[trait_name] = ref
            if ref in all_trait_keys: all_trait_keys.remove(ref)
    print(f"Identified {len(all_trait_keys)} trait dummies (reference dropped): {trait_reference_categories}")

    # rescue/bias stats (same as before)
    try:
        category_cols = getattr(case, "category_cols", None) or ["stereotype_type"]
        rescue_stats_list, bias_patterns_list = [], []
        from analysis_tools import guarded_labelspace_analysis
        for cat_col in category_cols:
            if cat_col in merged_df.columns:
                rs = guarded_labelspace_analysis(
                    rescue_stats_by_category, merged_df, case=case,
                    category_col=cat_col, person_set=person_set, group_keys=group_keys
                ); rs["category_col"] = cat_col; rescue_stats_list.append(rs)
                bp = guarded_labelspace_analysis(
                    detect_systematic_biases, merged_df, case=case,
                    category_col=cat_col, person_set=person_set
                ); bp["category_col"] = cat_col; bias_patterns_list.append(bp)
        if not rescue_stats_list: raise ValueError("No valid category columns found in merged_df.")
        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True)
    except Exception as e:
        print(f"WARNING: Could not get rescue/bias stats: {e}")
        # fallback simplified
        rescue_stats, bias_patterns = [], []
        for profile in profile_cols:
            base_correct = (merged_df['base_pred'] == merged_df['true_label'])
            profile_correct = (merged_df[profile] == merged_df['true_label'])
            rescued = ((~base_correct) & profile_correct).sum()
            base_errors = (~base_correct).sum()
            rescue_rate = rescued / base_errors if base_errors > 0 else 0.0
            extra_err_rate = ((base_correct) & (~profile_correct)).sum() / len(merged_df)
            rescue_stats.append({'profile': profile, 'rescue_rate': rescue_rate, 'extra_err_rate': extra_err_rate})
            to_positive = ((merged_df['base_pred'] == "no") & (merged_df[profile] == "yes")).sum()
            to_negative = ((merged_df['base_pred'] == "yes") & (merged_df[profile] == "no")).sum()
            bias_magnitude = abs(to_positive - to_negative) / len(merged_df)
            mislabelling_rate = (merged_df[profile] != merged_df['base_pred']).mean()
            bias_patterns.append({'profile': profile, 'bias_magnitude': bias_magnitude, 'mislabelling_rate': mislabelling_rate})
        rescue_stats = pd.DataFrame(rescue_stats); bias_patterns = pd.DataFrame(bias_patterns)

    # build causal dataset per profile (with cost/tokens + efficiencies)
    causal_rows = []
    for profile_name in profile_cols:
        if profile_name not in merged_df.columns: continue
        traits = person_set.get_traits(profile_name, group_keys)
        row = {"profile": profile_name}
        for trait_name in group_keys:
            raw = traits.get(trait_name, "Unknown")
            if trait_name == "age" and hasattr(raw, 'value'): raw = raw.value
            row[f"raw_{trait_name}"] = raw
        for trait_name in group_keys:
            value = traits.get(trait_name, "Unknown")
            if value != "Unknown":
                key = f"{trait_name}_{value.value}" if (trait_name == "age" and hasattr(value, 'value')) else f"{trait_name}_{value}"
                if key in all_trait_keys: row[key] = 1
        for key in all_trait_keys: row.setdefault(key, 0)

        accuracy = float((merged_df[profile_name] == merged_df['true_label']).mean())
        rs = rescue_stats[rescue_stats['profile'] == profile_name]
        rescue_rate = float(rs['rescue_rate'].mean()) if len(rs) > 0 else 0.0
        extra_error_rate = float(rs['extra_err_rate'].mean()) if ('extra_err_rate' in rs.columns and len(rs) > 0) else 0.0
        bp = bias_patterns[bias_patterns['profile'] == profile_name]
        bias_magnitude = float(bp['bias_magnitude'].mean()) if len(bp) > 0 else 0.0
        mislabelling_rate = float(bp['mislabelling_rate'].mean()) if len(bp) > 0 else 0.0
        volatility = 0.0
        if consistency_results and profile_name in consistency_results.get('consistency_data', {}):
            volatility = float(consistency_results['consistency_data'][profile_name]['accuracy_std'])
        toks = float(tokens_map.get(profile_name, np.nan)) if have_tokens else np.nan
        cost = float(cost_map.get(profile_name, np.nan)) if have_cost else np.nan
        tok_k = toks/1000.0 if np.isfinite(toks) and toks>0 else np.nan
        cost_d = cost if np.isfinite(cost) and cost>0 else np.nan

        row.update({
            'accuracy': accuracy,
            'rescue_rate': rescue_rate,
            'extra_error_rate': extra_error_rate,
            'bias_magnitude': bias_magnitude,
            'mislabelling_rate': mislabelling_rate,
            'volatility': volatility,
            'tokens_per_sample': toks if have_tokens else np.nan,
            'cost_per_sample':   cost if have_cost else np.nan,
            'accuracy_per_1k_tokens': (accuracy / tok_k) if np.isfinite(tok_k) else np.nan,
            'rescue_per_1k_tokens':   (rescue_rate / tok_k) if np.isfinite(tok_k) else np.nan,
            'extra_error_per_1k_tokens': (extra_error_rate / tok_k) if np.isfinite(tok_k) else np.nan,
            'accuracy_per_dollar': (accuracy / cost_d) if np.isfinite(cost_d) else np.nan,
            'rescue_per_dollar':   (rescue_rate / cost_d) if np.isfinite(cost_d) else np.nan,
            'extra_error_per_dollar': (extra_error_rate / cost_d) if np.isfinite(cost_d) else np.nan,
        })
        causal_rows.append(row)
    
    causal_df = pd.DataFrame(causal_rows)
    print(f"Causal dataset prepared with {len(causal_df)} profiles")
    print(f"    Trait dummies: {len([c for c in causal_df.columns if any(c.startswith(f'{t}_') for t in group_keys)])}")
    print(f"    Reference categories: {list(trait_reference_categories.values())}")

    # predictor sets
    trait_predictors = sorted([c for c in causal_df.columns if any(c.startswith(f'{t}_') for t in group_keys) and not c.endswith('_Unknown')])
    demographic_traits = ['gender', 'ethnicity']
    demographic_predictors = [c for c in trait_predictors if any(c.startswith(f'{t}_') for t in demographic_traits)]
    other_predictors = [c for c in trait_predictors if c not in demographic_predictors]
    all_predictors = demographic_predictors + other_predictors
    print(f"Demographic predictors ({len(demographic_predictors)}), Other traits ({len(other_predictors)})")

    # outcomes (add tokens/cost/efficiency only if have data)
    outcomes = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude']
    if consistency_results: outcomes.append('volatility')
    if have_tokens: outcomes += ['tokens_per_sample', 'accuracy_per_1k_tokens', 'rescue_per_1k_tokens', 'extra_error_per_1k_tokens']
    if have_cost:   outcomes += ['cost_per_sample', 'accuracy_per_dollar', 'rescue_per_dollar', 'extra_error_per_dollar']

    causal_results = {}
    for outcome in outcomes:
        print(f"\n--- MODELING: {outcome.upper()} ---")
        y = causal_df[outcome].values
        # filter rows with NaNs in outcome for regression stability
        mask = np.isfinite(y)
        if mask.sum() < 3:
            print("  WARNING: insufficient non-NaN points for regression")
            continue
        y_masked = y[mask]
        X_demo = causal_df.loc[mask, demographic_predictors].values if demographic_predictors else np.zeros((mask.sum(), 0))
        X_other = causal_df.loc[mask, other_predictors].values if other_predictors else np.zeros((mask.sum(), 0))
        X_full  = causal_df.loc[mask, all_predictors].values if all_predictors else np.zeros((mask.sum(), 0))

        r2_demo = LinearRegression().fit(X_demo, y_masked).score(X_demo, y_masked) if X_demo.shape[1] > 0 else 0.0
        r2_other = LinearRegression().fit(X_other, y_masked).score(X_other, y_masked) if X_other.shape[1] > 0 else 0.0
        r2_full  = LinearRegression().fit(X_full,  y_masked).score(X_full,  y_masked) if X_full.shape[1]  > 0 else 0.0

        demo_unique = max(0.0, r2_full - r2_other)
        other_unique = max(0.0, r2_full - r2_demo)
        shared = max(0.0, r2_demo + r2_other - r2_full)
        print(f"  Demographics R²: {r2_demo:.3f} | Other traits R²: {r2_other:.3f} | Full R²: {r2_full:.3f}")
        print(f"  Unique: demo={demo_unique:.3f} other={other_unique:.3f} | Shared={shared:.3f}")

        coeffs = {}
        if X_full.shape[1] > 0:
            model_full = LinearRegression().fit(X_full, y_masked)
            for i, predictor in enumerate(all_predictors):
                coeffs[predictor] = float(model_full.coef_[i])
        else:
            model_full = None

        causal_results[outcome] = {
            'r2_demographics': r2_demo,
            'r2_other': r2_other,
            'r2_full': r2_full,
            'demo_unique': demo_unique,
            'other_unique': other_unique,
            'shared_variance': shared,
            'coefficients': coeffs,
            'models': {
                'demographics': None,
                'other': None,
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


def visualize_causal_model(results, figsize=(10, 6)):
    """
    Visualize causal modeling (R² decomposition + coefficients + mediation + network)
    without mixing layout engines. Compatible with Matplotlib 3.8+.
    Accepts either the full dict from simplified_causal_modeling(...) or just the
    outcome->result mapping.
    """
    # normalize input
    if isinstance(results, dict) and 'causal_results' in results:
        causal_data = results['causal_results']
        mediation = results.get('mediation_evidence', {})
        strongest_predictors = results.get('strongest_predictors', {})
    else:
        causal_data = results
        mediation = {}
        strongest_predictors = {}

    outcomes = list(causal_data.keys())

    # predictors
    all_coeffs = list(next(iter(causal_data.values()))['coefficients'].keys()) if outcomes else []
    demographic_predictors = [p for p in all_coeffs if p.startswith('gender_') or p.startswith('ethnicity_')]
    other_predictors = [p for p in all_coeffs if p not in demographic_predictors]
    all_predictors = demographic_predictors + other_predictors

    def _new_fig(title):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # Use ONE layout engine: constrained
        try:
            fig.set_layout_engine('constrained')   # mpl>=3.8
        except Exception:
            try:
                fig.set_constrained_layout(True)   # older mpl
            except Exception:
                pass
        fig.suptitle('Causal Modeling: Traits → Outcomes (incl. Cost/Efficiency)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=11)
        return fig, ax

    figs = {}

    # A) Variance Decomposition
    figA, ax = _new_fig('Variance Decomposition (R²): Demographics vs Other Traits')
    demo_contrib  = [causal_data[o].get('demo_unique', 0) for o in outcomes]
    other_contrib = [causal_data[o].get('other_unique', 0) for o in outcomes]
    shared_contrib= [causal_data[o].get('shared_variance', 0) for o in outcomes]
    r2s           = [causal_data[o].get('r2_full', 0) for o in outcomes]
    x = np.arange(len(outcomes)); width = 0.6
    ax.bar(x, demo_contrib,  width, label='Demographics',  alpha=0.8)
    ax.bar(x, other_contrib, width, bottom=demo_contrib, label='Other Traits', alpha=0.8)
    ax.bar(x, shared_contrib,width, bottom=np.array(demo_contrib)+np.array(other_contrib), label='Shared', alpha=0.8)
    for i, outcome in enumerate(outcomes):
        ax.text(i, r2s[i] + 0.01, f'R²={r2s[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('Outcome Variables'); ax.set_ylabel('Variance Explained (R²)')
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=45, ha='right')
    ax.legend(); ax.grid(True, alpha=0.3)
    figs['variance_decomposition'] = figA

    # B) Coefficient Heatmap
    figB, ax = _new_fig('Causal Path Coefficients (β)')
    if all_predictors and outcomes:
        coef_matrix = np.zeros((len(all_predictors), len(outcomes)))
        for j, outcome in enumerate(outcomes):
            for i, predictor in enumerate(all_predictors):
                coef_matrix[i, j] = causal_data[outcome]['coefficients'].get(predictor, 0.0)

        vmax = max(0.05, float(np.nanmax(np.abs(coef_matrix))) * 0.8)
        im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        # Attach colorbar to FIGURE (not plt), so the layout engine can manage it
        cbar = figB.colorbar(im, ax=ax, shrink=0.8, format='%.1e')
        cbar.set_label('Coefficient Value')

        for i in range(len(all_predictors)):
            for j in range(len(outcomes)):
                value = coef_matrix[i, j]
                ax.text(j, i, f'{value:.1e}', ha="center", va="center",
                        color=("white" if abs(value) > vmax * 0.5 else "black"),
                        fontweight='bold', fontsize=8)
        ax.set_xticks(np.arange(len(outcomes))); ax.set_yticks(np.arange(len(all_predictors)))
        ax.set_xticklabels([o.replace('_', ' ').title() for o in outcomes], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([p.replace('_', ' ').title() for p in all_predictors], fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No coefficients available', ha='center', va='center', transform=ax.transAxes)
    figs['coef_heatmap'] = figB

    # C) Mediation (single version)
    figC, ax = _new_fig('Mediation: Other Traits vs Demographics')
    if mediation:
        valid_outcomes = [o for o in outcomes if o in mediation]
        ratios = [mediation[o].get('mediation_strength', 0) for o in valid_outcomes]
        if ratios:
            bars = ax.bar(range(len(valid_outcomes)), ratios, alpha=0.7)
            ax.axhline(1.5, linestyle='--', alpha=0.7, label='Strong Mediation')
            ax.axhline(0.8, linestyle='--', alpha=0.7, label='Partial Mediation')
            for bar, r in zip(bars, ratios):
                ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.1, f'{r:.2f}', ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel('Mediation Ratio')
            ax.set_xticks(range(len(valid_outcomes)))
            ax.set_xticklabels([o.replace('_', ' ').title() for o in valid_outcomes], rotation=45, ha='right')
            ax.legend(); ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No mediation data available', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No mediation analysis available', ha='center', va='center', transform=ax.transAxes)
    figs['mediation'] = figC

    # D) Causal Network
    figD, ax = _new_fig('Causal Network Structure')
    outcome_positions = [(0.8, y) for y in (0.9, 0.7, 0.5, 0.3, 0.1)]
    actual_node_pos = {'Demographics': (0.2, 0.8), 'Other Traits': (0.2, 0.4)}
    outcome_names = [o.replace('_', ' ').title() for o in outcomes]
    for i, outcome_name in enumerate(outcome_names):
        if i < len(outcome_positions):
            actual_node_pos[outcome_name] = outcome_positions[i]

    for node, (x0, y0) in actual_node_pos.items():
        if node == 'Demographics':
            color = '#1f77b4'; size = 1200
        elif node == 'Other Traits':
            color = '#ff7f0e'; size = 1200
        else:
            color = '#2ca02c'; size = 800
        ax.scatter(x0, y0, s=size, c=color, edgecolors='black', alpha=0.7, linewidth=2)
        ax.text(x0, y0-0.06, node, ha='center', va='top', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    for outcome, pred_info in strongest_predictors.items():
        if not pred_info or not pred_info.get('predictor'): 
            continue
        predictor = pred_info['predictor']
        coef = float(pred_info.get('coefficient', 0))
        outcome_display = outcome.replace('_', ' ').title()
        if outcome_display not in actual_node_pos:
            continue
        start_node = 'Demographics' if (predictor.startswith('gender_') or predictor.startswith('ethnicity_')) else 'Other Traits'
        start_pos = actual_node_pos[start_node]; end_pos = actual_node_pos[outcome_display]
        lw = min(abs(coef) * 20, 4) + 0.5
        color = '#2ca02c' if coef > 0 else '#d62728'
        ax.annotate('', xy=end_pos, xytext=start_pos, arrowprops=dict(arrowstyle='->', lw=lw, color=color, alpha=0.7))

    if 'Demographics' in actual_node_pos and 'Other Traits' in actual_node_pos:
        ax.annotate('', xy=actual_node_pos['Other Traits'], xytext=actual_node_pos['Demographics'],
                    arrowprops=dict(arrowstyle='->', lw=2, alpha=0.5, linestyle='--'))
    ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.02, 0.02, 'Green = Positive Effect, Red = Negative Effect\nLine thickness ∝ Effect size',
            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.9),
            fontsize=8, va='bottom')
    figs['causal_network'] = figD

    return figs



def _dir(case, plots_root=None, strategy=None, stage="tier3", extra=None):
    return resolve_plot_dir(
        case, plots_root=plots_root, strategy=strategy, stage=stage, extra_subdir=extra
    )

def _ensure(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _save(
    fig,
    case,
    plots_root=None,
    strategy=None,
    stage: str = "tier3",
    subdir: str = None,
    filename: str = None,
    fmt: str = "pdf",
    dpi: int = None,
):
    """
    Save a matplotlib Figure to the resolved tier3 directory.
    - If filename is None, uses 'figure.<fmt>'.
    - If filename has no extension, appends .<fmt>.
    """
    out_dir = _ensure(_dir(case, plots_root=plots_root, strategy=strategy, stage=stage, extra=subdir))

    if filename is None:
        filename = f"figure.{fmt}"
    else:
        root, ext = os.path.splitext(filename)
        if not ext: 
            filename = f"{filename}.{fmt}"

    path = os.path.join(out_dir, filename)

    try:
        fig.canvas.draw()
    except Exception:
        pass

    fig.savefig(path, bbox_inches="tight", dpi=dpi, format=fmt)
    try:
        plt.close(fig)
    except Exception:
        pass
    return path


def save_tier3_outputs(
    consistency_results: Dict[str, Any],
    causal_results: Dict[str, Any],
    case: CaseConfig,
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,
    stage: str = "tier3",
    subdirs: Optional[Dict[str, str]] = None,
    *,
    fmt: str = "pdf",
    dpi: int = None,
    person_set: Optional[PersonSet] = None,
    group_keys: Optional[Tuple[str, ...]] = None,
):
    """
    Saves tier3 figures using resolve_plot_dir(...) exactly like analysis_0.
      subdir 'consistency_boldness' → figs from plot_consistency_boldness_analysis(...)
      subdir 'causal'               → figs from visualize_causal_model(...)
    Returns: dict of saved file paths.
    """
    subdirs = subdirs or {}
    dir_consistency = _dir(case, plots_root, strategy, stage, extra=subdirs.get("consistency_boldness", "consistency_boldness"))
    dir_causal      = _dir(case, plots_root, strategy, stage, extra=subdirs.get("causal", "causal"))
    _ensure(dir_consistency); _ensure(dir_causal)

    # Build figs (formatting only; heavy analysis already done)
    try:
        figs_cons = plot_consistency_boldness_analysis(
            consistency_results,
            person_set=person_set,
            group_keys=(group_keys or ("gender", "ethnicity", "age")),
        ) or {}
    except Exception as e:
        print(f"[WARN] Could not build consistency/boldness figs: {e}")
        figs_cons = {}

    try:
        figs_caus = visualize_causal_model(causal_results) or {}
    except Exception as e:
        print(f"[WARN] Could not build causal figs: {e}")
        figs_caus = {}

    saved = {"consistency_boldness": {}, "causal": {}}

    # --- Consistency/Boldness figs ---
    if "consistency_vs_boldness" in figs_cons:
        saved["consistency_boldness"]["consistency_vs_boldness"] = _save(
            figs_cons["consistency_vs_boldness"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="consistency_vs_boldness", fmt=fmt, dpi=dpi
        )
    if "volatility_vs_rescue" in figs_cons:
        saved["consistency_boldness"]["volatility_vs_rescue"] = _save(
            figs_cons["volatility_vs_rescue"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="volatility_vs_rescue", fmt=fmt, dpi=dpi
        )
    if "archetype_by_demo" in figs_cons:
        saved["consistency_boldness"]["archetype_by_demo"] = _save(
            figs_cons["archetype_by_demo"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="archetype_by_demo", fmt=fmt, dpi=dpi
        )
    if "cv_trends" in figs_cons:
        saved["consistency_boldness"]["cv_trends"] = _save(
            figs_cons["cv_trends"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="cv_trends", fmt=fmt, dpi=dpi
        )

    # --- Causal figs ---
    if "variance_decomposition" in figs_caus:
        saved["causal"]["variance_decomposition"] = _save(
            figs_caus["variance_decomposition"], case, plots_root, strategy, stage,
            subdir=subdirs.get("causal","causal"),
            filename="variance_decomposition", fmt=fmt, dpi=dpi
        )
    if "coef_heatmap" in figs_caus:
        saved["causal"]["coef_heatmap"] = _save(
            figs_caus["coef_heatmap"], case, plots_root, strategy, stage,
            subdir=subdirs.get("causal","causal"),
            filename="coef_heatmap", fmt=fmt, dpi=dpi
        )
    if "mediation" in figs_caus:
        saved["causal"]["mediation"] = _save(
            figs_caus["mediation"], case, plots_root, strategy, stage,
            subdir=subdirs.get("causal","causal"),
            filename="mediation", fmt=fmt, dpi=dpi
        )
    if "causal_network" in figs_caus or "network" in figs_caus:
        key = "causal_network" if "causal_network" in figs_caus else "network"
        saved["causal"]["causal_network"] = _save(
            figs_caus[key], case, plots_root, strategy, stage,
            subdir=subdirs.get("causal","causal"),
            filename="causal_network", fmt=fmt, dpi=dpi
        )

    return saved



# at the top of tier3 file, make sure this import is present
from analysis_tools import get_analysis_group_keys, get_available_traits, resolve_plot_dir

def run_full_tier3_analysis(
    merged_df,
    person_set: PersonSet,
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = None, 
    n_folds=5,
    perf_df: Optional[pd.DataFrame] = None,
    plots_root: str = "results/figs",
    strategy: str = "not_defined",
    stage: str = "tier3",
    per_figure_subdirs: Optional[Dict[str, str]] = None,
    save: bool = True,
    fmt: str = "pdf",
    dpi: int = 200,
):
    print("EXECUTING TIER 3 ANALYSIS PIPELINE")
    print("=" * 80)

    if person_set is None:
        print("WARNING: No PersonSet provided - some analyses may not work correctly")

    # Keys / traits
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

    # Default subdirs that match save_tier3_outputs(...) expectations
    if per_figure_subdirs is None:
        per_figure_subdirs = {
            "consistency_boldness": "consistency_boldness",
            "causal": "causal",
        }

    # Base dir (for reporting) — same resolver as analysis_0
    base_dir = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage)
    if save:
        os.makedirs(base_dir, exist_ok=True)

    # 1) Consistency vs Boldness
    print("\n=== Running Consistency vs Boldness Analysis ===")
    try:
        consistency_results = consistency_vs_boldness_analysis(
            merged_df,
            case=case,
            n_folds=n_folds,
            person_set=person_set,
            group_keys=group_keys,
            perf_df=perf_df
        )
        print("Consistency analysis completed successfully")
    except Exception as e:
        print(f"ERROR in consistency analysis: {e}")
        consistency_results = {'error': str(e)}

    # 2) Causal Modeling
    print("\n=== Running Simplified Causal Modeling ===")
    try:
        causal_results = simplified_causal_modeling(
            merged_df,
            person_set=person_set,
            case=case,
            group_keys=group_keys,
            consistency_results=(consistency_results if 'error' not in consistency_results else None),
            perf_df=perf_df
        )
        print("Causal modeling completed successfully")
    except Exception as e:
        print(f"ERROR in causal modeling: {e}")
        causal_results = {'error': str(e)}

    # 3) Optional on-screen figs (not used for saving)
    print("\n=== Creating Consistency Analysis Visualizations ===")
    consistency_viz = None
    if 'error' not in consistency_results:
        try:
            consistency_viz = plot_consistency_boldness_analysis(
                consistency_results, person_set=person_set, group_keys=group_keys
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

    # 4) Centralized saving (single call; matches analysis_0 directory scheme)
    saved_paths = {}
    if save:
        try:
            saved_paths = save_tier3_outputs(
                consistency_results=consistency_results if 'error' not in consistency_results else {},
                causal_results=causal_results if 'error' not in causal_results else {},
                case=case,
                plots_root=plots_root,
                strategy=strategy,
                stage=stage,
                subdirs=per_figure_subdirs,
                fmt=fmt,
                dpi=dpi,
                person_set=person_set,
                group_keys=group_keys,
            )
            print("\nSaved tier3 figures:")
            for group, items in saved_paths.items():
                for name, p in items.items():
                    print(f"  [{group}] {name}: {p}")
        except Exception as e:
            print(f"WARNING: Saving failed: {e}")

    # 5) Theoretical integration & 6) Methodological assessment
    print("\n" + "=" * 80)
    print("TIER 3 THEORETICAL INTEGRATION")
    print("=" * 80)

    consistency_insight = (consistency_results.get('normative_assessment', {}) if 'error' not in consistency_results else {})
    high_vol_rescue = consistency_insight.get('overall', {}).get('high_volatility_rescue', 0)
    low_vol_rescue  = consistency_insight.get('overall', {}).get('low_volatility_rescue', 0)

    strongest_predictors = (causal_results.get('strongest_predictors', {}) if 'error' not in causal_results else {})
    bias_detection = (causal_results.get('bias_detection', {}) if 'error' not in causal_results else {})

    print("\n=== THEORETICAL INSIGHTS:")
    if high_vol_rescue and low_vol_rescue:
        if high_vol_rescue > low_vol_rescue + 0.01:
            vol_conclusion = "Inconsistent profiles provide higher moral value through increased rescue rates"
            vol_implication = "Controlled inconsistency in AI systems may be normatively justified"
        else:
            vol_conclusion = "Consistent profiles match or outperform inconsistent ones in moral value"
            vol_implication = "Consistency should be prioritized over boldness in AI systems"
        print(f"   - {vol_conclusion}\n   - Implication: {vol_implication}\n   - Evidence: {high_vol_rescue:.3f} vs {low_vol_rescue:.3f}")
    else:
        vol_conclusion = "Insufficient data for consistency-boldness analysis"
        print(f"   - {vol_conclusion}")

    if strongest_predictors:
        rescue_pred = strongest_predictors.get('rescue_rate', {}).get('predictor')
        accuracy_pred = strongest_predictors.get('accuracy', {}).get('predictor')
        print(f"   - Strongest rescue predictor: {rescue_pred or 'None'}")
        print(f"   - Strongest accuracy predictor: {accuracy_pred or 'None'}")

    if bias_detection:
        high_bias_count = sum(1 for d in bias_detection.values() if 'HIGH BIAS' in d.get('bias_level', ''))
        moderate_bias_count = sum(1 for d in bias_detection.values() if 'MODERATE BIAS' in d.get('bias_level', ''))
        print(f"   - Bias assessment: {high_bias_count} high, {moderate_bias_count} moderate bias cases detected")

    print(f"\n{'=' * 60}\nTHESIS-LEVEL CONCLUSIONS\n{'=' * 60}")
    conclusions = []
    if high_vol_rescue and low_vol_rescue:
        if high_vol_rescue > low_vol_rescue + 0.02:
            conclusions.append({'finding': 'Inconsistent profiles provide superior moral value',
                                'evidence': f'{high_vol_rescue:.3f} vs {low_vol_rescue:.3f} rescue rates',
                                'implication': 'Controlled inconsistency can improve moral outcomes',
                                'strength': 'Strong' if high_vol_rescue > low_vol_rescue + 0.05 else 'Moderate'})
        else:
            conclusions.append({'finding': 'Consistency and inconsistency yield similar moral value',
                                'evidence': f'Diff: {abs(high_vol_rescue - low_vol_rescue):.3f}',
                                'implication': 'Prefer consistency for predictability',
                                'strength': 'Moderate'})

    print(f"\n{'=' * 60}\nMETHODOLOGICAL ASSESSMENT\n{'=' * 60}")
    method_assessment = {
        'data_quality': 'Good' if len(merged_df) > 100 else 'Limited',
        'statistical_power': 'Adequate' if 'error' not in consistency_results and 'error' not in causal_results else 'Insufficient',
        'bias_detection_capability': 'Strong' if bias_detection else 'Limited',
        'generalizability': 'Domain-specific' if person_set else 'Unknown'
    }
    for k, v in method_assessment.items():
        print(f"   - {k.replace('_',' ').title()}: {v}")

    valid_profiles = consistency_results.get('valid_profiles', []) if 'error' not in consistency_results else []
    if group_keys:
        print(f"   - Profile coverage: {len(valid_profiles)} | Trait coverage: {len(group_keys)} ({', '.join(group_keys)})")

    return {
        'consistency_analysis': consistency_results,
        'causal_analysis': causal_results,
        'visualizations': {'consistency_plot': consistency_viz, 'causal_plot': causal_viz},
        'figure_paths': saved_paths,         
        'saved_to': base_dir if save else None,
        'theoretical_integration': {'volatility_conclusion': vol_conclusion if 'vol_conclusion' in locals() else None,
                                    'causal_mechanisms': strongest_predictors,
                                    'bias_assessment': bias_detection},
        'thesis_conclusions': conclusions,
        'methodological_assessment': method_assessment,
        'analysis_parameters': {'group_keys': group_keys, 'n_folds': n_folds, 'dataset_size': len(merged_df)}
    }
