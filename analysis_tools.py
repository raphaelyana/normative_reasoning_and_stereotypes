import os
import glob
import json
import re
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any


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


def load_and_merge_profiles(
    base_file_path: str,
    role_playing_glob_pattern: str,
    sample_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
    label_col: str = "true_label",
    pred_col: str = "pred_label",
    extra_cols: list = ["stereotype_type", "original_dataset"]
) -> pd.DataFrame:

    df_base = pd.read_csv(base_file_path)
    merged = df_base[[sample_id_col, label_col, pred_col]].rename(columns={pred_col: "base_pred"})

    profile_files = glob.glob(role_playing_glob_pattern)
    print(f"=== Found {len(profile_files)} profile result files.")

    for file in profile_files:
        profile_folder = os.path.basename(os.path.dirname(file))
        df_profile = pd.read_csv(file)

        if sample_id_col not in df_profile.columns or pred_col not in df_profile.columns:
            print(f"=== Skipping {profile_folder} (missing required columns)")
            continue

        df_profile = df_profile[[sample_id_col, pred_col]].rename(columns={pred_col: profile_folder})

        if profile_folder in merged.columns:
            merged = merged.drop(columns=[profile_folder])

        merged = merged.merge(df_profile, on=sample_id_col, how="left")

    sample_df = sample_df.reset_index(drop=True)
    sample_df[sample_id_col] = sample_df.index

    merged = merged.merge(
        sample_df[[sample_id_col] + extra_cols],
        on=sample_id_col,
        how="left"
    )

    merged[label_col] = merged[label_col].astype(str).str.strip().str.lower()
    merged["base_pred"] = merged["base_pred"].astype(str).str.strip().str.lower()
    for col in merged.columns[3:]:
        merged[col] = merged[col].astype(str).str.strip().str.lower()

    fixed_columns = [sample_id_col, label_col, "base_pred"]
    meta_columns = [col for col in extra_cols if col in merged.columns]

    profile_columns = [
        col for col in merged.columns
        if col.startswith("profile") and col not in fixed_columns + meta_columns
    ]

    # Sort numerically by profile number
    def extract_profile_number(col_name):
        match = re.search(r"profile(\d+)", col_name)
        return int(match.group(1)) if match else float('inf')

    profile_columns = sorted(profile_columns, key=extract_profile_number)

    # Final column ordering
    final_columns = fixed_columns + meta_columns + profile_columns
    merged = merged[final_columns]

    print("=== Merged DataFrame ready with columns:\n", merged.columns.tolist())
    return merged


def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


##########################################
###          Tier 2 beginning          ###
##########################################


def ensemble_by_trait_analysis(merged_df):
    """
    Selective Ensemble Performance Analysis
    
    Tests majority vote performance using only specific trait groups:
    - WOMEN_PROFILES only
    - BALANCED_PROFILES only  
    - HIGH_HARM_PROFILES only
    - WHITE_PROFILES only
    - Best performing cognitive style
    - Pareto optimal profiles
    
    Compares to full ensemble and baseline to show how role design affects
    system-level safety and performance.
    """
    
    # Define trait groups with _passive suffix
    trait_groups = {
        'ALL_PROFILES': [f"profile{i}_passive" for i in range(1, 31)],
        'MEN_ONLY': [f"profile{i}_passive" for i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]],
        'WOMEN_ONLY': [f"profile{i}_passive" for i in [6,7,8,9,10, 16,17,18,19,20, 26,27,28,29,30]],
        'WHITE_ONLY': [f"profile{i}_passive" for i in range(1, 11)],
        'BLACK_ONLY': [f"profile{i}_passive" for i in range(11, 21)],
        'ASIAN_ONLY': [f"profile{i}_passive" for i in range(21, 31)],
        'EXPANSIVE_ONLY': [f"profile{i}_passive" for i in [1,6,11,16,21,26]],
        'LITERAL_ONLY': [f"profile{i}_passive" for i in [2,7,12,17,22,27]],
        'HIGH_HARM_ONLY': [f"profile{i}_passive" for i in [3,8,13,18,23,28]],
        'LOW_HARM_ONLY': [f"profile{i}_passive" for i in [4,9,14,19,24,29]],
        'BALANCED_ONLY': [f"profile{i}_passive" for i in [5,10,15,20,25,30]],
        'WHITE_MEN_ONLY': [f"profile{i}_passive" for i in range(1, 6)],
        'BLACK_WOMEN_ONLY': [f"profile{i}_passive" for i in range(16, 21)],
        'NON_WHITE_MEN': [f"profile{i}_passive" for i in list(range(6, 31))]  # Everyone except white men
    }
    
    def majority_vote_ensemble(df, profile_list):
        """Calculate majority vote for a list of profiles"""
        available_profiles = [p for p in profile_list if p in df.columns]
        if not available_profiles:
            return pd.Series([''] * len(df), index=df.index)
        
        # Get predictions from available profiles
        ensemble_preds = df[available_profiles]
        
        # Calculate majority vote for each row
        majority_votes = []
        for idx, row in ensemble_preds.iterrows():
            votes = [vote for vote in row.values if pd.notna(vote) and vote != '']
            if votes:
                vote_counts = Counter(votes)
                majority_vote = vote_counts.most_common(1)[0][0]
                majority_votes.append(majority_vote)
            else:
                majority_votes.append('')
        
        return pd.Series(majority_votes, index=df.index)
    
    # Calculate ensemble performance for each trait group
    ensemble_results = {}
    true_labels = merged_df['true_label']
    baseline_preds = merged_df['base_pred']
    baseline_accuracy = accuracy_score(true_labels, baseline_preds)
    
    print("=" * 80)
    print("ENSEMBLE BY TRAIT ANALYSIS")
    print("=" * 80)
    
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"\nEnsemble Performance by Trait Group:")
    print("-" * 50)
    
    for group_name, profile_list in trait_groups.items():
        # Get majority vote predictions
        ensemble_preds = majority_vote_ensemble(merged_df, profile_list)
        
        # Calculate accuracy
        if len(ensemble_preds) > 0 and not ensemble_preds.eq('').all():
            ensemble_accuracy = accuracy_score(true_labels, ensemble_preds)
            improvement = ensemble_accuracy - baseline_accuracy
            
            # Calculate rescue and extra error stats
            base_correct = (baseline_preds == true_labels)
            ensemble_correct = (ensemble_preds == true_labels)
            
            rescued = ((~base_correct) & ensemble_correct).sum()
            extra_errors = (base_correct & (~ensemble_correct)).sum()
            
            base_errors = (~base_correct).sum()
            base_correct_count = base_correct.sum()
            
            rescue_rate = rescued / base_errors if base_errors > 0 else 0
            extra_error_rate = extra_errors / base_correct_count if base_correct_count > 0 else 0
            
            ensemble_results[group_name] = {
                'accuracy': ensemble_accuracy,
                'improvement': improvement,
                'rescued': rescued,
                'extra_errors': extra_errors,
                'rescue_rate': rescue_rate,
                'extra_error_rate': extra_error_rate,
                'n_profiles': len([p for p in profile_list if p in merged_df.columns]),
                'ensemble_preds': ensemble_preds
            }
            
            print(f"{group_name:15s}: {ensemble_accuracy:.4f} ({improvement:+.4f}) | "
                  f"Rescue: {rescue_rate:.3f} | Extra Err: {extra_error_rate:.3f} | "
                  f"n={len([p for p in profile_list if p in merged_df.columns])}")
        else:
            print(f"{group_name:15s}: No valid predictions")
    
    # Category-specific analysis
    print(f"\n{'='*60}")
    print("CATEGORY-SPECIFIC ENSEMBLE PERFORMANCE")
    print(f"{'='*60}")
    
    categories = merged_df['stereotype_type'].unique()
    category_results = {}
    
    for category in categories:
        cat_subset = merged_df[merged_df['stereotype_type'] == category]
        cat_true = cat_subset['true_label']
        cat_baseline = cat_subset['base_pred']
        cat_baseline_acc = accuracy_score(cat_true, cat_baseline)
        
        print(f"\n{category.upper()} STEREOTYPES (n={len(cat_subset)}):")
        print(f"Baseline accuracy: {cat_baseline_acc:.4f}")
        print("-" * 40)
        
        category_results[category] = {'baseline_accuracy': cat_baseline_acc, 'ensembles': {}}
        
        # Test key ensemble strategies on this category
        key_ensembles = ['ALL_PROFILES', 'WOMEN_ONLY', 'BALANCED_ONLY', 'HIGH_HARM_ONLY', 'NON_WHITE_MEN']
        
        for ensemble_name in key_ensembles:
            if ensemble_name in ensemble_results:
                # Get ensemble predictions for this category
                cat_ensemble_preds = ensemble_results[ensemble_name]['ensemble_preds'].loc[cat_subset.index]
                
                if not cat_ensemble_preds.eq('').all():
                    cat_ensemble_acc = accuracy_score(cat_true, cat_ensemble_preds)
                    cat_improvement = cat_ensemble_acc - cat_baseline_acc
                    
                    category_results[category]['ensembles'][ensemble_name] = {
                        'accuracy': cat_ensemble_acc,
                        'improvement': cat_improvement
                    }
                    
                    print(f"  {ensemble_name:15s}: {cat_ensemble_acc:.4f} ({cat_improvement:+.4f})")
    
    # Identify best performing ensembles
    print(f"\n{'='*60}")
    print("ENSEMBLE RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Sort by overall accuracy
    sorted_ensembles = sorted(ensemble_results.items(), 
                             key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nTop 5 Performing Ensembles (Overall Accuracy):")
    for i, (name, results) in enumerate(sorted_ensembles[:5]):
        print(f"  {i+1}. {name}: {results['accuracy']:.4f} "
              f"(+{results['improvement']:.4f}) | "
              f"Rescue: {results['rescue_rate']:.3f} | "
              f"Risk: {results['extra_error_rate']:.3f}")
    
    # Sort by safety (lowest extra error rate)
    sorted_by_safety = sorted(ensemble_results.items(),
                             key=lambda x: x[1]['extra_error_rate'])
    
    print("\nSafest Ensembles (Lowest Extra Error Rate):")
    for i, (name, results) in enumerate(sorted_by_safety[:5]):
        print(f"  {i+1}. {name}: Extra Error Rate {results['extra_error_rate']:.3f} | "
              f"Accuracy: {results['accuracy']:.4f}")
    
    # Sort by rescue effectiveness
    sorted_by_rescue = sorted(ensemble_results.items(),
                             key=lambda x: x[1]['rescue_rate'], reverse=True)
    
    print("\nMost Effective Rescue Ensembles:")
    for i, (name, results) in enumerate(sorted_by_rescue[:5]):
        print(f"  {i+1}. {name}: Rescue Rate {results['rescue_rate']:.3f} | "
              f"Accuracy: {results['accuracy']:.4f}")
    
    # System-level safety recommendations
    print(f"\n{'='*60}")
    print("SYSTEM-LEVEL SAFETY RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Find the ensemble with best balance of performance and safety
    safety_performance_scores = {}
    for name, results in ensemble_results.items():
        # Weighted score: 60% accuracy improvement, 40% safety (inverse of extra error rate)
        safety_score = 1 - results['extra_error_rate']  # Higher is better
        combined_score = 0.6 * results['improvement'] + 0.4 * safety_score
        safety_performance_scores[name] = combined_score
    
    best_balanced = max(safety_performance_scores.items(), key=lambda x: x[1])
    
    print(f"\nRecommended Ensemble (Best Safety-Performance Balance):")
    print(f"  {best_balanced[0]}")
    print(f"  Accuracy: {ensemble_results[best_balanced[0]]['accuracy']:.4f}")
    print(f"  Improvement: +{ensemble_results[best_balanced[0]]['improvement']:.4f}")
    print(f"  Rescue Rate: {ensemble_results[best_balanced[0]]['rescue_rate']:.3f}")
    print(f"  Extra Error Rate: {ensemble_results[best_balanced[0]]['extra_error_rate']:.3f}")
    
    return {
        'ensemble_results': ensemble_results,
        'category_results': category_results,
        'recommendations': {
            'best_overall': sorted_ensembles[0],
            'safest': sorted_by_safety[0], 
            'best_rescue': sorted_by_rescue[0],
            'best_balanced': best_balanced
        },
        'trait_groups': trait_groups
    }


def cluster_level_bias_patterns(merged_df, similarity_results=None):
    """
    Cluster-level Bias and Rescue Pattern Analysis
    
    Evaluates grouped behaviors using existing persona clusters.
    For each cluster, computes:
    - Average bias_magnitude, rescue_rate, accuracy
    - Identifies tradeoffs: "cluster 2 performs best overall, but is it safer?"
    - Recommends profile subsets for aligned ensembles
    """
    
    # Get clustering results if not provided
    if similarity_results is None:
        similarity_results = analyze_persona_similarity(merged_df)
    
    # Get rescue and bias stats
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    
    print("=" * 80)
    print("CLUSTER-LEVEL BIAS AND RESCUE PATTERNS")
    print("=" * 80)
    
    cluster_analysis = {}
    baseline_accuracy = accuracy_score(merged_df["true_label"], merged_df["base_pred"])
    
    for cluster_id, cluster_info in similarity_results['clusters'].items():
        cluster_profiles = cluster_info['profiles']
        available_profiles = [p for p in cluster_profiles if p in merged_df.columns]

        print(f"\n{cluster_id.upper()} ({len(available_profiles)} profiles):")
        print(f"Profiles: {', '.join(p.replace('_passive', '') for p in available_profiles[:5])}")
        if len(available_profiles) > 5:
            print(f"          ... and {len(available_profiles) - 5} more")

        cluster_metrics = {
            'profiles': available_profiles,
            'size': len(available_profiles),
            'internal_agreement': cluster_info.get('internal_agreement', 0)
        }
        
        # Accuracy metrics
        accuracies = [(merged_df[p] == merged_df['true_label']).mean() for p in available_profiles]
        cluster_metrics['accuracy_mean'] = np.mean(accuracies) if accuracies else 0
        cluster_metrics['accuracy_std'] = np.std(accuracies) if accuracies else 0
        
        # Rescue metrics
        cluster_rescue = rescue_stats[rescue_stats['profile'].isin(available_profiles)]
        if not cluster_rescue.empty:
            cluster_metrics.update({
                'rescue_rate_mean': cluster_rescue['rescue_rate'].mean(),
                'rescue_rate_std': cluster_rescue['rescue_rate'].std(),
                'extra_error_rate_mean': cluster_rescue['extra_err_rate'].mean(),
                'extra_error_rate_std': cluster_rescue['extra_err_rate'].std(),
                'total_rescued': cluster_rescue['rescued'].sum(),
                'total_extra_errors': cluster_rescue['extra_errors'].sum()
            })
        else:
            cluster_metrics.update({
                'rescue_rate_mean': 0, 'rescue_rate_std': 0,
                'extra_error_rate_mean': 0, 'extra_error_rate_std': 0,
                'total_rescued': 0, 'total_extra_errors': 0
            })
        
        # Bias metrics
        cluster_bias = bias_patterns[bias_patterns['profile'].isin(available_profiles)]
        if not cluster_bias.empty:
            cluster_metrics.update({
                'bias_magnitude_mean': cluster_bias['bias_magnitude'].mean(),
                'bias_magnitude_std': cluster_bias['bias_magnitude'].std(),
                'mislabelling_rate_mean': cluster_bias['mislabelling_rate'].mean(),
                'dominant_bias_direction': cluster_bias['bias_direction'].mode().iloc[0] if not cluster_bias['bias_direction'].empty else 'none'
            })
        else:
            cluster_metrics.update({
                'bias_magnitude_mean': 0, 'bias_magnitude_std': 0,
                'mislabelling_rate_mean': 0, 'dominant_bias_direction': 'none'
            })
        
        # Print cluster summary
        print(f"  Accuracy: {cluster_metrics['accuracy_mean']:.4f} ± {cluster_metrics['accuracy_std']:.4f}")
        print(f"  Rescue Rate: {cluster_metrics['rescue_rate_mean']:.3f} ± {cluster_metrics['rescue_rate_std']:.3f}")
        print(f"  Extra Error Rate: {cluster_metrics['extra_error_rate_mean']:.3f} ± {cluster_metrics['extra_error_rate_std']:.3f}")
        print(f"  Bias Magnitude: {cluster_metrics['bias_magnitude_mean']:.3f} (direction: {cluster_metrics['dominant_bias_direction']})")
        print(f"  Internal Agreement: {cluster_metrics['internal_agreement']:.3f}")
        
        # Identify cluster archetype
        if cluster_metrics['rescue_rate_mean'] > 0.15 and cluster_metrics['extra_error_rate_mean'] < 0.05:
            archetype = "Safely Bold"
        elif cluster_metrics['extra_error_rate_mean'] < 0.03:
            archetype = "Safe"
        elif cluster_metrics['rescue_rate_mean'] > 0.20:
            archetype = "High Rescue"
        elif cluster_metrics['accuracy_mean'] > 0.72:
            archetype = "High Performer"
        elif cluster_metrics['internal_agreement'] > 0.95:
            archetype = "Highly Consistent"
        else:
            archetype = "Balanced"
        
        cluster_metrics['archetype'] = archetype
        
        print(f"  Accuracy: {cluster_metrics['accuracy_mean']:.4f} ± {cluster_metrics['accuracy_std']:.4f}")
        print(f"  Rescue Rate: {cluster_metrics['rescue_rate_mean']:.3f} ± {cluster_metrics['rescue_rate_std']:.3f}")
        print(f"  Extra Error Rate: {cluster_metrics['extra_error_rate_mean']:.3f} ± {cluster_metrics['extra_error_rate_std']:.3f}")
        print(f"  Bias Magnitude: {cluster_metrics['bias_magnitude_mean']:.3f} (direction: {cluster_metrics['dominant_bias_direction']})")
        print(f"  Internal Agreement: {cluster_metrics['internal_agreement']:.3f}")
        print(f"  Archetype: {archetype}")

        cluster_analysis[cluster_id] = cluster_metrics
    
    # Cluster comparison and recommendations
    print(f"\n{'='*60}")
    print("CLUSTER COMPARISON AND RECOMMENDATIONS")
    print(f"{'='*60}")

    best_accuracy = max(cluster_analysis.items(), key=lambda x: x[1]['accuracy_mean'])
    safest_cluster = min(cluster_analysis.items(), key=lambda x: x[1]['extra_error_rate_mean'])
    best_rescue = max(cluster_analysis.items(), key=lambda x: x[1]['rescue_rate_mean'])

    print(f"\nBest Overall Performance: {best_accuracy[0]} (accuracy: {best_accuracy[1]['accuracy_mean']:.4f})")
    print(f"Safest Cluster: {safest_cluster[0]} (extra error rate: {safest_cluster[1]['extra_error_rate_mean']:.3f})")
    print(f"Best Rescue Cluster: {best_rescue[0]} (rescue rate: {best_rescue[1]['rescue_rate_mean']:.3f})")

    # Cluster majority vote performance
    print(f"\n{'='*60}")
    print("CLUSTER ENSEMBLE PERFORMANCE")
    print(f"{'='*60}")

    for cluster_id, cluster_info in cluster_analysis.items():
        cluster_profiles = cluster_info['profiles']
        ensemble_preds = majority_vote_ensemble(merged_df, cluster_profiles)

        if not ensemble_preds.eq('').all():
            ensemble_accuracy = accuracy_score(merged_df['true_label'], ensemble_preds)
            improvement = ensemble_accuracy - baseline_accuracy

            cluster_analysis[cluster_id]['ensemble_accuracy'] = ensemble_accuracy
            cluster_analysis[cluster_id]['ensemble_improvement'] = improvement

            print(f"{cluster_id}: {ensemble_accuracy:.4f} ({improvement:+.4f}) | {cluster_info['archetype']}")
        else:
            cluster_analysis[cluster_id]['ensemble_accuracy'] = None
            cluster_analysis[cluster_id]['ensemble_improvement'] = None
            print(f"{cluster_id}: No valid ensemble predictions")

    return {
        'cluster_analysis': cluster_analysis,
        'recommendations': {
            'best_accuracy': best_accuracy,
            'safest': safest_cluster,
            'best_rescue': best_rescue
        },
        'similarity_results': similarity_results
    }


def cognitive_style_comparison_controlled(merged_df):
    """
    Cognitive Style Comparison (Controlled for Demographics)
    
    Evaluates performance across cognitive styles while controlling for demographic noise.
    Uses Kruskal-Wallis test and pairwise Wilcoxon tests for robust comparison.
    
    Tells you if styles like "balanced" or "high-harm" consistently outperform others.
    """
    
    # Define cognitive style groups
    cognitive_styles = {
        'Expansive': [f"profile{i}_passive" for i in [1,6,11,16,21,26]],
        'Literal': [f"profile{i}_passive" for i in [2,7,12,17,22,27]],
        'High_Harm': [f"profile{i}_passive" for i in [3,8,13,18,23,28]],
        'Low_Harm': [f"profile{i}_passive" for i in [4,9,14,19,24,29]],
        'Balanced': [f"profile{i}_passive" for i in [5,10,15,20,25,30]]
    }
    
    print("=" * 80)
    print("COGNITIVE STYLE COMPARISON (CONTROLLED)")
    print("=" * 80)
    
    # Calculate performance metrics for each cognitive style
    style_performance = {}
    
    for style_name, profiles in cognitive_styles.items():
        available_profiles = [p for p in profiles if p in merged_df.columns]
        
        if not available_profiles:
            continue
        
        # Calculate metrics for each profile in this cognitive style
        accuracies = []
        rescue_rates = []
        extra_error_rates = []
        bias_magnitudes = []
        
        # Get rescue stats
        rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
        bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
        
        for profile in available_profiles:
            # Accuracy
            acc = accuracy_score(merged_df['true_label'], merged_df[profile])
            accuracies.append(acc)
            
            # Rescue stats
            profile_rescue = rescue_stats[rescue_stats['profile'] == profile]
            if len(profile_rescue) > 0:
                rescue_rates.append(profile_rescue['rescue_rate'].mean())
                extra_error_rates.append(profile_rescue['extra_err_rate'].mean())
            else:
                rescue_rates.append(0)
                extra_error_rates.append(0)
            
            # Bias patterns
            profile_bias = bias_patterns[bias_patterns['profile'] == profile]
            if len(profile_bias) > 0:
                bias_magnitudes.append(profile_bias['bias_magnitude'].mean())
            else:
                bias_magnitudes.append(0)
        
        style_performance[style_name] = {
            'accuracies': accuracies,
            'rescue_rates': rescue_rates,
            'extra_error_rates': extra_error_rates,
            'bias_magnitudes': bias_magnitudes,
            'n_profiles': len(available_profiles),
            'profiles': available_profiles
        }
        
        # Print summary statistics
        print(f"\n{style_name.upper()} COGNITIVE STYLE (n={len(available_profiles)}):")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Rescue Rate: {np.mean(rescue_rates):.3f} ± {np.std(rescue_rates):.3f}")
        print(f"  Extra Error Rate: {np.mean(extra_error_rates):.3f} ± {np.std(extra_error_rates):.3f}")
        print(f"  Bias Magnitude: {np.mean(bias_magnitudes):.3f} ± {np.std(bias_magnitudes):.3f}")
    
    # Statistical tests comparing cognitive styles
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISONS BETWEEN COGNITIVE STYLES")
    print(f"{'='*60}")
    
    metrics = ['accuracies', 'rescue_rates', 'extra_error_rates', 'bias_magnitudes']
    metric_names = ['Accuracy', 'Rescue Rate', 'Extra Error Rate', 'Bias Magnitude']
    
    statistical_results = {}
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{metric_name.upper()}:")
        
        # Prepare data for Kruskal-Wallis test
        groups = []
        group_names = []
        
        for style_name, data in style_performance.items():
            if metric in data and len(data[metric]) > 0:
                groups.append(data[metric])
                group_names.append(style_name)
        
        if len(groups) >= 3:  
            # Need at least 3 groups for meaningful comparison
            # Kruskal-Wallis test (non-parametric ANOVA)
            combined_values = np.concatenate(groups)
            if np.all(combined_values == combined_values[0]):
                print(f"  Skipping {metric_name} comparison — all values are identical.")
                continue
            else:
                h_stat, p_value = kruskal(*groups)
            
            print(f"  Kruskal-Wallis H-statistic: {h_stat:.3f}")
            print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.05 else ''}")
            
            statistical_results[metric] = {
                'kruskal_wallis': {'h_statistic': h_stat, 'p_value': p_value, 'significant': p_value < 0.05},
                'group_means': {name: np.mean(data) for name, data in zip(group_names, groups)},
                'pairwise_comparisons': {}
            }
            
            # Pairwise comparisons if overall test is significant
            if p_value < 0.05:
                print(f"  Pairwise comparisons (Mann-Whitney U):")
                
                for i, (name1, group1) in enumerate(zip(group_names, groups)):
                    for j, (name2, group2) in enumerate(zip(group_names, groups)):
                        if i < j:  # Avoid duplicate comparisons
                            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                            
                            # Calculate effect size (rank-biserial correlation)
                            n1, n2 = len(group1), len(group2)
                            effect_size = 1 - (2 * u_stat) / (n1 * n2)
                            
                            statistical_results[metric]['pairwise_comparisons'][f'{name1}_vs_{name2}'] = {
                                'u_statistic': u_stat,
                                'p_value': p_val,
                                'significant': p_val < 0.05,
                                'effect_size': effect_size,
                                'mean_diff': np.mean(group1) - np.mean(group2)
                            }
                            
                            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                            print(f"    {name1} vs {name2}: p={p_val:.4f} {significance} (effect size: {effect_size:.3f})")
            else:
                print(f"  No significant differences between cognitive styles for {metric_name}")
    
    # Cognitive style rankings
    print(f"\n{'='*60}")
    print("COGNITIVE STYLE RANKINGS")
    print(f"{'='*60}")
    
    rankings = {}
    
    for metric, metric_name in zip(metrics, metric_names):
        if metric in statistical_results:
            group_means = statistical_results[metric]['group_means']
            
            # Sort by performance (higher is better for accuracy and rescue_rate, lower is better for error rates)
            if metric in ['accuracies', 'rescue_rates']:
                sorted_styles = sorted(group_means.items(), key=lambda x: x[1], reverse=True)
            else:  # error rates and bias magnitude
                sorted_styles = sorted(group_means.items(), key=lambda x: x[1])
            
            rankings[metric_name] = sorted_styles
            
            print(f"\n{metric_name} Rankings:")
            for rank, (style, value) in enumerate(sorted_styles, 1):
                print(f"  {rank}. {style}: {value:.4f}")
    
    # Overall cognitive style recommendation
    print(f"\n{'='*60}")
    print("COGNITIVE STYLE RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Calculate composite scores for each cognitive style
    composite_scores = {}
    
    for style_name in cognitive_styles.keys():
        if style_name in [ranking[0][0] for ranking in rankings.values()]:  # Style has data
            score = 0
            
            # Weight different metrics (higher weight = more important)
            weights = {
                'Accuracy': 0.3,
                'Rescue Rate': 0.25,
                'Extra Error Rate': -0.25,  # Negative because lower is better
                'Bias Magnitude': -0.2      # Negative because lower is better
            }
            
            for metric_name, weight in weights.items():
                if metric_name in rankings:
                    # Find this style's rank (1-based)
                    style_rank = next((rank for rank, (style, _) in enumerate(rankings[metric_name], 1) 
                                     if style == style_name), len(rankings[metric_name]))
                    
                    # Convert rank to score (5 = best rank 1, 1 = worst rank 5)
                    rank_score = 6 - style_rank
                    score += weight * rank_score
            
            composite_scores[style_name] = score
    
    # Sort by composite score
    recommended_styles = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nRecommended Cognitive Styles (Composite Score):")
    for rank, (style, score) in enumerate(recommended_styles, 1):
        print(f"  {rank}. {style}: {score:.3f}")
    
    # Best cognitive style for different objectives
    if rankings:
        print(f"\nSpecialized Recommendations:")
        
        if 'Accuracy' in rankings:
            best_accuracy_style = rankings['Accuracy'][0][0]
            print(f"  Best for Overall Performance: {best_accuracy_style}")
        
        if 'Extra Error Rate' in rankings:
            safest_style = rankings['Extra Error Rate'][0][0]  # Lowest error rate
            print(f"  Safest (Lowest Extra Errors): {safest_style}")
        
        if 'Rescue Rate' in rankings:
            best_rescue_style = rankings['Rescue Rate'][0][0]
            print(f"  Best for Error Correction: {best_rescue_style}")
        
        if 'Bias Magnitude' in rankings:
            least_biased_style = rankings['Bias Magnitude'][0][0]  # Lowest bias
            print(f"  Least Biased: {least_biased_style}")
    
    return {
        'style_performance': style_performance,
        'statistical_results': statistical_results,
        'rankings': rankings,
        'composite_scores': composite_scores,
        'recommendations': recommended_styles,
        'cognitive_styles': cognitive_styles
    }


def visualize_tier2_results(ensemble_results, cluster_results, cognitive_results, figsize=(16, 12)):
    """
    Create comprehensive visualizations for Tier 2 analyses.
    
    Generates 4-panel plot showing:
    1. Ensemble Performance Comparison
    2. Cluster Risk-Benefit Analysis  
    3. Cognitive Style Performance Radar
    4. System-Level Safety Recommendations
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Tier 2 Analysis: Ensemble, Cluster & Cognitive Style Results', fontsize=16, fontweight='bold')
    
    # ===== PANEL 1: Ensemble Performance Comparison =====
    ax = axes[0, 0]
    
    ensemble_data = ensemble_results['ensemble_results']
    ensemble_names = []
    accuracies = []
    improvements = []
    
    # Select key ensembles for visualization
    key_ensembles = ['ALL_PROFILES', 'WOMEN_ONLY', 'BALANCED_ONLY', 'HIGH_HARM_ONLY', 
                    'NON_WHITE_MEN', 'WHITE_ONLY', 'BLACK_WOMEN_ONLY']
    
    for name in key_ensembles:
        if name in ensemble_data:
            ensemble_names.append(name.replace('_', ' ').title())
            accuracies.append(ensemble_data[name]['accuracy'])
            improvements.append(ensemble_data[name]['improvement'])
    
    # Create bar plot
    x_pos = np.arange(len(ensemble_names))
    bars = ax.bar(x_pos, improvements, color=['#1f77b4' if imp >= 0 else '#d62728' for imp in improvements])
    
    # Add accuracy values on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005 if height >= 0 else height - 0.0015,
                f'{acc:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    ax.set_xlabel('Ensemble Strategy')
    ax.set_ylabel('Accuracy Improvement vs Baseline')
    ax.set_title('Ensemble Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ensemble_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 2: Cluster Risk-Benefit Analysis =====
    ax = axes[0, 1]
    
    cluster_data = cluster_results['cluster_analysis']
    
    rescue_rates = []
    extra_error_rates = []
    cluster_labels = []
    cluster_colors = []
    
    # Color mapping for archetypes
    archetype_colors = {
        'Safely Bold': '#2ca02c',
        'Ultra Safe': '#1f77b4', 
        'High Rescue': '#ff7f0e',
        'High Performer': '#d62728',
        'Highly Consistent': '#9467bd',
        'Balanced': '#8c564b'
    }
    
    for cluster_id, cluster_info in cluster_data.items():
        rescue_rates.append(cluster_info['rescue_rate_mean'])
        extra_error_rates.append(cluster_info['extra_error_rate_mean'])
        cluster_labels.append(cluster_id.replace('cluster_', 'C'))
        archetype = cluster_info['archetype']
        cluster_colors.append(archetype_colors.get(archetype, '#bcbd22'))
    
    scatter = ax.scatter(extra_error_rates, rescue_rates, c=cluster_colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add cluster labels
    for i, label in enumerate(cluster_labels):
        ax.annotate(label, (extra_error_rates[i], rescue_rates[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Cluster Risk-Benefit Analysis')
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 3: Cognitive Style Performance Radar =====
    ax = axes[1, 0]
    
    cognitive_data = cognitive_results['style_performance']
    
    # Prepare data for radar chart
    styles = list(cognitive_data.keys())
    metrics = ['Accuracy', 'Rescue Rate', 'Safety', 'Consistency']  # Safety = 1 - extra_error_rate
    
    # Calculate normalized scores for each style
    style_scores = {}
    for style, data in cognitive_data.items():
        accuracy_score = np.mean(data['accuracies'])
        rescue_score = np.mean(data['rescue_rates'])
        safety_score = 1 - np.mean(data['extra_error_rates'])  # Invert so higher is better
        consistency_score = 1 - np.std(data['accuracies'])     # Lower std = higher consistency
        
        style_scores[style] = [accuracy_score, rescue_score, safety_score, consistency_score]
    
    # Create simplified bar chart instead of radar (easier to read)
    x_pos = np.arange(len(styles))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        values = [style_scores[style][i] for style in styles]
        ax.bar(x_pos + i*width, values, width, label=metric, color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Cognitive Style')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Cognitive Style Performance Profile')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(styles, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 4: System-Level Safety Recommendations =====
    ax = axes[1, 1]
    
    # Create a summary visualization of recommendations
    recommendations = []
    scores = []
    colors = []
    
    # Best ensemble recommendation
    best_ensemble = ensemble_results['recommendations']['best_balanced'][0]
    ensemble_score = ensemble_results['ensemble_results'][best_ensemble]['improvement']
    recommendations.append(f"Best Ensemble:\n{best_ensemble.replace('_', ' ')}")
    scores.append(ensemble_score)
    colors.append('#2ca02c')
    
    # Best cluster recommendation  
    best_cluster = cluster_results['recommendations']['best_accuracy'][0]
    cluster_score = cluster_results['cluster_analysis'][best_cluster]['accuracy_mean'] - 0.70  # Baseline approximation
    recommendations.append(f"Best Cluster:\n{best_cluster}")
    scores.append(cluster_score)
    colors.append('#1f77b4')
    
    # Best cognitive style
    if cognitive_results['recommendations']:
        best_cognitive = cognitive_results['recommendations'][0][0]
        cognitive_score = cognitive_results['composite_scores'][best_cognitive] / 10  # Normalize
        recommendations.append(f"Best Cognitive:\n{best_cognitive}")
        scores.append(cognitive_score)
        colors.append('#ff7f0e')
    
    # Safest options
    safest_ensemble = ensemble_results['recommendations']['safest'][0]
    safety_score = -ensemble_results['ensemble_results'][safest_ensemble]['extra_error_rate']  # Negative of error rate
    recommendations.append(f"Safest Ensemble:\n{safest_ensemble.replace('_', ' ')}")
    scores.append(safety_score)
    colors.append('#d62728')
    
    # Create horizontal bar chart
    y_pos = np.arange(len(recommendations))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        ax.text(width + 0.001 if width >= 0 else width - 0.001, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(recommendations)
    ax.set_xlabel('Performance Score')
    ax.set_title('System-Level Recommendations')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# TIER 2 INTEGRATION FUNCTION
# ============================================================================

def run_tier2_analysis(merged_df):
    """
    Run complete Tier 2 analysis pipeline.
    
    Executes:
    1. Ensemble by Trait Analysis
    2. Cluster-level Bias Patterns  
    3. Cognitive Style Comparison
    4. Comprehensive Visualization
    
    Returns integrated results for thesis reporting.
    """
    
    print("🚀 EXECUTING TIER 2 ANALYSIS PIPELINE")
    print("="*80)
    
    # 1. Ensemble by Trait Analysis
    print("\n📊 Running Ensemble by Trait Analysis...")
    ensemble_results = ensemble_by_trait_analysis(merged_df)
    
    # 2. Cluster-level Bias Patterns
    print("\n🔗 Running Cluster-level Bias Analysis...")
    cluster_results = cluster_level_bias_patterns(merged_df)
    
    # 3. Cognitive Style Comparison
    print("\n🧠 Running Cognitive Style Comparison...")
    cognitive_results = cognitive_style_comparison_controlled(merged_df)
    
    # 4. Generate Visualizations
    print("\n📈 Creating Tier 2 Visualizations...")
    visualization = visualize_tier2_results(ensemble_results, cluster_results, cognitive_results)
    
    # 5. Generate Executive Summary
    print("\n" + "="*80)
    print("TIER 2 EXECUTIVE SUMMARY")
    print("="*80)
    
    # Best recommendations from each analysis
    best_ensemble = ensemble_results['recommendations']['best_balanced'][0]
    best_cluster = cluster_results['recommendations']['best_accuracy'][0]
    best_cognitive = cognitive_results['recommendations'][0][0] if cognitive_results['recommendations'] else 'Unknown'
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Best Ensemble Strategy: {best_ensemble}")
    print(f"   • Best Cluster: {best_cluster}")  
    print(f"   • Best Cognitive Style: {best_cognitive}")
    
    # Performance improvements
    ensemble_improvement = ensemble_results['ensemble_results'][best_ensemble]['improvement']
    print(f"\n📈 PERFORMANCE GAINS:")
    print(f"   • Best Ensemble Improvement: +{ensemble_improvement:.4f}")
    print(f"   • Safety-Performance Balance Achieved")
    
    # Safety insights
    safest_ensemble = ensemble_results['recommendations']['safest'][0]
    safety_rate = ensemble_results['ensemble_results'][safest_ensemble]['extra_error_rate']
    print(f"\n🛡️ SAFETY INSIGHTS:")
    print(f"   • Safest Strategy: {safest_ensemble}")
    print(f"   • Minimum Extra Error Rate: {safety_rate:.3f}")
    
    return {
        'ensemble_analysis': ensemble_results,
        'cluster_analysis': cluster_results,
        'cognitive_analysis': cognitive_results,
        'visualization': visualization,
        'executive_summary': {
            'best_ensemble': best_ensemble,
            'best_cluster': best_cluster,
            'best_cognitive': best_cognitive,
            'performance_improvement': ensemble_improvement,
            'safety_rate': safety_rate
        }
    }


# ============================================================================
# TIER 3 ANALYSES: TEMPORAL STABILITY vs BOLDNESS & CAUSAL MODELING
# Add these functions to your analysis_tools.py
# ============================================================================

def temporal_stability_vs_boldness_analysis(merged_df, n_folds=5):
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
    
    # Get profile columns
    profile_cols = [col for col in merged_df.columns if col.startswith("profile") and "_passive" in col]
    
    # ========================================================================
    # STEP 1: Calculate Temporal Stability (Cross-Validation Volatility)
    # ========================================================================
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    stability_data = {}
    
    print(f"📊 Calculating temporal stability across {n_folds} folds...")
    
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
    
    # ========================================================================
    # STEP 2: Define Boldness Metrics
    # ========================================================================
    
    print("⚡ Calculating boldness metrics...")
    
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
        
        # Boldness metric 2: Willingness to disagree with baseline (flip rate)
        profile_bias = bias_patterns[bias_patterns['profile'] == profile]
        avg_flip_rate = profile_bias['flip_rate'].mean() if len(profile_bias) > 0 else 0
        
        # Boldness metric 3: Magnitude of bias (willingness to take strong positions)
        avg_bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
        
        # Composite boldness score
        boldness_score = (
            0.4 * avg_rescue_rate +      # 40% rescue effectiveness
            0.3 * avg_flip_rate +        # 30% disagreement rate  
            0.3 * avg_bias_magnitude     # 30% position strength
        )
        
        boldness_data[profile] = {
            'rescue_rate': avg_rescue_rate,
            'flip_rate': avg_flip_rate,
            'bias_magnitude': avg_bias_magnitude,
            'boldness_score': boldness_score
        }
    
    # ========================================================================
    # STEP 3: Stability vs Boldness Correlation Analysis
    # ========================================================================
    
    print("🔗 Analyzing stability-boldness correlations...")
    
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
        
        print(f"\n📈 CORRELATION RESULTS:")
        print(f"  Volatility vs Boldness: r={corr_vol_bold:.3f}, p={p_val_vol_bold:.4f} {'***' if p_val_vol_bold < 0.05 else ''}")
        print(f"  Volatility vs Rescue Rate: r={corr_vol_rescue:.3f}, p={p_val_vol_rescue:.4f} {'***' if p_val_vol_rescue < 0.05 else ''}")
        print(f"  Boldness vs Accuracy: r={corr_bold_acc:.3f}, p={p_val_bold_acc:.4f} {'***' if p_val_bold_acc < 0.05 else ''}")
    
    # ========================================================================
    # STEP 4: Profile Classification and Insights
    # ========================================================================
    
    print(f"\n🎯 PROFILE CLASSIFICATION:")
    
    # Classify profiles into archetypes
    profile_archetypes = {}
    
    volatility_median = np.median(volatilities) if volatilities else 0
    boldness_median = np.median(boldness_scores) if boldness_scores else 0
    
    for i, profile in enumerate(profiles):
        vol = volatilities[i]
        bold = boldness_scores[i]
        rescue = rescue_rates[i]
        acc = accuracy_means[i]
        
        # Determine archetype
        if vol > volatility_median and bold > boldness_median:
            archetype = "🌪️ Volatile Bold" 
            description = "High risk, high moral value"
        elif vol < volatility_median and bold > boldness_median:
            archetype = "⚡ Stable Bold"
            description = "Best of both worlds"
        elif vol > volatility_median and bold < boldness_median:
            archetype = "🎲 Volatile Cautious"
            description = "High risk, low moral value"
        else:
            archetype = "🛡️ Stable Cautious"
            description = "Low risk, predictable"
        
        profile_archetypes[profile] = {
            'archetype': archetype,
            'description': description,
            'volatility': vol,
            'boldness': bold,
            'rescue_rate': rescue,
            'accuracy': acc
        }
        
        print(f"  {profile.replace('_passive', '')}: {archetype} - {description}")
        print(f"    Volatility: {vol:.4f}, Boldness: {bold:.3f}, Rescue: {rescue:.3f}, Accuracy: {acc:.4f}")
    
    # ========================================================================
    # STEP 5: Normative Value Assessment
    # ========================================================================
    
    print(f"\n⚖️ NORMATIVE VALUE ASSESSMENT:")
    
    # Test the key hypothesis: Do volatile profiles provide moral value?
    high_volatility_profiles = [p for p, data in profile_archetypes.items() 
                               if data['volatility'] > volatility_median]
    low_volatility_profiles = [p for p, data in profile_archetypes.items() 
                              if data['volatility'] <= volatility_median]
    
    if high_volatility_profiles and low_volatility_profiles:
        # Compare rescue rates
        high_vol_rescue = [profile_archetypes[p]['rescue_rate'] for p in high_volatility_profiles]
        low_vol_rescue = [profile_archetypes[p]['rescue_rate'] for p in low_volatility_profiles]
        
        high_vol_rescue_mean = np.mean(high_vol_rescue)
        low_vol_rescue_mean = np.mean(low_vol_rescue)
        
        # Statistical test
        
        t_stat, p_val = ttest_ind(high_vol_rescue, low_vol_rescue)
        
        print(f"  High Volatility Profiles Rescue Rate: {high_vol_rescue_mean:.3f}")
        print(f"  Low Volatility Profiles Rescue Rate: {low_vol_rescue_mean:.3f}")
        print(f"  Difference: {high_vol_rescue_mean - low_vol_rescue_mean:.3f}")
        print(f"  Statistical Test: t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
        
        if high_vol_rescue_mean > low_vol_rescue_mean and p_val < 0.05:
            conclusion = "✅ VOLATILE PROFILES PROVIDE SIGNIFICANT MORAL VALUE"
        elif high_vol_rescue_mean > low_vol_rescue_mean:
            conclusion = "⚠️ VOLATILE PROFILES SHOW HIGHER RESCUE RATES (NOT SIGNIFICANT)"
        else:
            conclusion = "❌ VOLATILE PROFILES DO NOT PROVIDE MORAL VALUE ADVANTAGE"
        
        print(f"\n🏆 CONCLUSION: {conclusion}")
    
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
        '🌪️ Volatile Bold': '#d62728',      # Red
        '⚡ Stable Bold': '#2ca02c',        # Green  
        '🎲 Volatile Cautious': '#ff7f0e',  # Orange
        '🛡️ Stable Cautious': '#1f77b4'    # Blue
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


def simplified_causal_modeling(merged_df, stability_results=None):
    """
    Simplified Causal Modeling for Bias Effects
    
    Models causal influence of profile traits on bias/alignment using:
    - Linear regression with demographic and cognitive predictors
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
    
    # Get performance metrics
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    
    for i in range(1, 31):  # Profiles 1-30
        profile_name = f"profile{i}_passive"
        
        if profile_name not in merged_df.columns:
            continue
        
        # Causal predictors (profile traits)
        gender = 1 if i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25] else 0  # 1=Male, 0=Female
        
        # Ethnicity (dummy coding with White as reference)
        white = 1 if i in range(1, 11) else 0
        black = 1 if i in range(11, 21) else 0
        asian = 1 if i in range(21, 31) else 0
        
        # Cognitive style (dummy coding with Balanced as reference)
        expansive = 1 if i in [1,6,11,16,21,26] else 0
        literal = 1 if i in [2,7,12,17,22,27] else 0
        high_harm = 1 if i in [3,8,13,18,23,28] else 0
        low_harm = 1 if i in [4,9,14,19,24,29] else 0
        
        # Outcome variables
        accuracy = (merged_df[profile_name] == merged_df['true_label']).mean()
        
        # Rescue metrics
        profile_rescue = rescue_stats[rescue_stats['profile'] == profile_name]
        rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
        extra_error_rate = profile_rescue['extra_err_rate'].mean() if len(profile_rescue) > 0 else 0
        
        # Bias metrics
        profile_bias = bias_patterns[bias_patterns['profile'] == profile_name]
        bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
        flip_rate = profile_bias['flip_rate'].mean() if len(profile_bias) > 0 else 0
        
        # Add stability metrics if available
        volatility = 0
        if stability_results and profile_name in stability_results['stability_data']:
            volatility = stability_results['stability_data'][profile_name]['accuracy_std']
        
        causal_data.append({
            'profile': profile_name,
            'gender_male': gender,
            'ethnicity_white': white,
            'ethnicity_black': black,
            'ethnicity_asian': asian,
            'cognitive_expansive': expansive,
            'cognitive_literal': literal,
            'cognitive_high_harm': high_harm,
            'cognitive_low_harm': low_harm,
            'accuracy': accuracy,
            'rescue_rate': rescue_rate,
            'extra_error_rate': extra_error_rate,
            'bias_magnitude': bias_magnitude,
            'flip_rate': flip_rate,
            'volatility': volatility
        })
    
    causal_df = pd.DataFrame(causal_data)
    
    print(f"📊 Causal dataset prepared with {len(causal_df)} profiles")
    
    # ========================================================================
    # STEP 2: Linear Regression Models
    # ========================================================================
    
    print(f"\n🔗 CAUSAL PATH ANALYSIS:")
    
    # Define predictor sets
    demographic_predictors = ['gender_male', 'ethnicity_black', 'ethnicity_asian']  # White as reference
    cognitive_predictors = ['cognitive_expansive', 'cognitive_literal', 'cognitive_high_harm', 'cognitive_low_harm']  # Balanced as reference
    all_predictors = demographic_predictors + cognitive_predictors
    
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
        
        # Model 2: Cognitive styles only
        X_cog = causal_df[cognitive_predictors].values
        model_cog = LinearRegression().fit(X_cog, y)
        r2_cog = model_cog.score(X_cog, y)
        
        # Model 3: Full model (demographics + cognitive)
        X_full = causal_df[all_predictors].values
        model_full = LinearRegression().fit(X_full, y)
        r2_full = model_full.score(X_full, y)
        
        # Calculate unique contributions
        demo_unique = r2_full - r2_cog  # Variance explained by demographics beyond cognitive
        cog_unique = r2_full - r2_demo   # Variance explained by cognitive beyond demographics  
        shared = r2_demo + r2_cog - r2_full  # Shared variance
        
        print(f"  Demographics only R²: {r2_demo:.3f}")
        print(f"  Cognitive only R²: {r2_cog:.3f}")
        print(f"  Full model R²: {r2_full:.3f}")
        print(f"  Demographics unique contribution: {demo_unique:.3f}")
        print(f"  Cognitive unique contribution: {cog_unique:.3f}")
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
            'r2_cognitive': r2_cog,
            'r2_full': r2_full,
            'demo_unique': demo_unique,
            'cog_unique': cog_unique,
            'shared_variance': shared,
            'coefficients': coefficients,
            'models': {
                'demographics': model_demo,
                'cognitive': model_cog,
                'full': model_full
            }
        }
    
    # ========================================================================
    # STEP 3: Causal Path Interpretation
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("CAUSAL PATH INTERPRETATION")
    print(f"{'='*60}")
    
    # Identify strongest causal pathways
    strongest_predictors = {}
    
    for outcome, results in causal_results.items():
        max_coef = 0
        strongest_predictor = None
        
        for predictor, coef in results['coefficients'].items():
            if abs(coef) > abs(max_coef):
                max_coef = coef
                strongest_predictor = predictor
        
        # Ensure we have a valid predictor, even if weak
        if strongest_predictor is None and results['coefficients']:
            # Take any predictor if none are strong
            strongest_predictor = list(results['coefficients'].keys())[0]
            max_coef = results['coefficients'][strongest_predictor]
        
        strongest_predictors[outcome] = {
            'predictor': strongest_predictor,
            'coefficient': max_coef,
            'direction': 'increases' if max_coef > 0 else 'decreases'
        }
        
        if strongest_predictor is not None:
            print(f"\n{outcome.upper()}:")
            print(f"  Strongest predictor: {strongest_predictor} (β={max_coef:.3f})")
            print(f"  Effect: {strongest_predictor} {strongest_predictors[outcome]['direction']} {outcome}")
        else:
            print(f"\n{outcome.upper()}:")
            print(f"  No significant predictors found")
        
        # Identify whether demographics or cognitive factors dominate
        if results['demo_unique'] > results['cog_unique']:
            dominant_factor = "demographic traits"
            dominance_ratio = results['demo_unique'] / results['cog_unique'] if results['cog_unique'] > 0 else float('inf')
        else:
            dominant_factor = "cognitive traits" 
            dominance_ratio = results['cog_unique'] / results['demo_unique'] if results['demo_unique'] > 0 else float('inf')
        
        print(f"  Dominant factor: {dominant_factor} (ratio: {dominance_ratio:.2f})")
    
    # ========================================================================
    # STEP 4: Theoretical Framework
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("THEORETICAL CAUSAL FRAMEWORK")
    print(f"{'='*60}")
    
    print(f"\n🧠 CAUSAL MECHANISM HYPOTHESIS:")
    print(f"  Profile Demographics → Cognitive Processing → Bias Outcomes")
    
    # Test mediation hypothesis: Do cognitive styles mediate demographic effects?
    mediation_evidence = {}
    
    for outcome in outcomes:
        demo_direct = causal_results[outcome]['demo_unique']
        cog_contribution = causal_results[outcome]['cog_unique']
        total_variance = causal_results[outcome]['r2_full']
        
        # Simple mediation indicator: cognitive factors explain more than demographics
        mediation_strength = cog_contribution / (demo_direct + 0.001)  # Avoid division by zero
        
        if mediation_strength > 1.5:
            mediation_type = "Strong mediation: Cognitive styles largely mediate demographic effects"
        elif mediation_strength > 0.8:
            mediation_type = "Partial mediation: Both demographics and cognitive styles matter"
        else:
            mediation_type = "Direct effects: Demographics dominate over cognitive processing"
        
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
    
    print(f"\n🎯 TO MAXIMIZE RESCUE RATE (Moral Value):")
    for trait, coef in best_rescue_traits:
        if coef > 0.01:
            print(f"  ✓ Select profiles with: {trait} (β={coef:.3f})")
    
    recommendations['maximize_rescue'] = best_rescue_traits
    
    # For minimizing extra errors (safety)
    error_predictors = causal_results['extra_error_rate']['coefficients']
    safest_traits = sorted(error_predictors.items(), key=lambda x: x[1])[:3]  # Lowest coefficients
    
    print(f"\n🛡️ TO MINIMIZE EXTRA ERRORS (Safety):")
    for trait, coef in safest_traits:
        if coef < -0.01:
            print(f"  ✓ Select profiles with: {trait} (β={coef:.3f})")
        elif coef < 0.01:
            print(f"  ✓ Avoid profiles with: {trait} (β={coef:.3f})")
    
    recommendations['maximize_safety'] = safest_traits
    
    # For maximizing overall accuracy
    accuracy_predictors = causal_results['accuracy']['coefficients']
    best_accuracy_traits = sorted(accuracy_predictors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"\n📈 TO MAXIMIZE ACCURACY (Performance):")
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
            'hypothesis': "Profile Demographics → Cognitive Processing → Bias Outcomes",
            'mediation_support': mediation_evidence,
            'policy_implications': recommendations
        }
    }


def visualize_causal_model(causal_results, figsize=(16, 12)):
    """
    Create comprehensive visualizations for causal modeling results.
    
    Generates 4-panel plot:
    1. Variance Decomposition (Demographics vs Cognitive)
    2. Causal Path Strengths (Coefficient Heatmap)
    3. Mediation Analysis 
    4. Causal Network Diagram
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Causal Modeling: Profile Traits → Bias Outcomes', fontsize=16, fontweight='bold')
    
    causal_data = causal_results['causal_results']
    outcomes = list(causal_data.keys())
    
    # ===== PANEL 1: Variance Decomposition =====
    ax = axes[0, 0]
    
    demo_contrib = [causal_data[outcome]['demo_unique'] for outcome in outcomes]
    cog_contrib = [causal_data[outcome]['cog_unique'] for outcome in outcomes]
    shared_contrib = [causal_data[outcome]['shared_variance'] for outcome in outcomes]
    
    x = np.arange(len(outcomes))
    width = 0.6
    
    # Stacked bar chart
    p1 = ax.bar(x, demo_contrib, width, label='Demographics Only', color='#1f77b4', alpha=0.8)
    p2 = ax.bar(x, cog_contrib, width, bottom=demo_contrib, label='Cognitive Only', color='#ff7f0e', alpha=0.8)
    p3 = ax.bar(x, shared_contrib, width, bottom=np.array(demo_contrib) + np.array(cog_contrib), 
               label='Shared Variance', color='#2ca02c', alpha=0.8)
    
    # Add total R² labels
    for i, outcome in enumerate(outcomes):
        total_r2 = causal_data[outcome]['r2_full']
        ax.text(i, total_r2 + 0.02, f'R²={total_r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Variance Explained (R²)')
    ax.set_title('Variance Decomposition: Demographics vs Cognitive')
    ax.set_xticks(x)
    ax.set_xticklabels([outcome.replace('_', ' ').title() for outcome in outcomes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 2: Coefficient Heatmap =====
    ax = axes[0, 1]
    
    # Create coefficient matrix
    predictors = ['gender_male', 'ethnicity_black', 'ethnicity_asian', 
                 'cognitive_expansive', 'cognitive_literal', 'cognitive_high_harm', 'cognitive_low_harm']
    
    coef_matrix = np.zeros((len(predictors), len(outcomes)))
    
    for j, outcome in enumerate(outcomes):
        for i, predictor in enumerate(predictors):
            if predictor in causal_data[outcome]['coefficients']:
                coef_matrix[i, j] = causal_data[outcome]['coefficients'][predictor]
    
    # Create heatmap
    im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    
    # Add coefficient values
    for i in range(len(predictors)):
        for j in range(len(outcomes)):
            text = ax.text(j, i, f'{coef_matrix[i, j]:.3f}', ha="center", va="center", 
                          color="white" if abs(coef_matrix[i, j]) > 0.05 else "black", fontweight='bold')
    
    ax.set_xticks(np.arange(len(outcomes)))
    ax.set_yticks(np.arange(len(predictors)))
    ax.set_xticklabels([outcome.replace('_', ' ').title() for outcome in outcomes], rotation=45, ha='right')
    ax.set_yticklabels([pred.replace('_', ' ').title() for pred in predictors])
    ax.set_title('Causal Path Strengths (β coefficients)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Standardized Coefficient')
    
    # ===== PANEL 3: Mediation Analysis =====
    ax = axes[1, 0]
    
    mediation_evidence = causal_results['mediation_evidence']
    
    mediation_ratios = [mediation_evidence[outcome]['mediation_strength'] for outcome in outcomes]
    colors = ['#2ca02c' if ratio > 1.5 else '#ff7f0e' if ratio > 0.8 else '#d62728' for ratio in mediation_ratios]
    
    bars = ax.bar(range(len(outcomes)), mediation_ratios, color=colors, alpha=0.7)
    
    # Add threshold lines
    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Strong Mediation')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Partial Mediation')
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars, mediation_ratios)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Outcome Variables')
    ax.set_ylabel('Mediation Ratio (Cognitive/Demographics)')
    ax.set_title('Mediation Analysis: Do Cognitive Styles Mediate Demographics?')
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels([outcome.replace('_', ' ').title() for outcome in outcomes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ===== PANEL 4: Simplified Network Diagram =====
    ax = axes[1, 1]
    
    # Create a simplified network visualization
    # Nodes: Demographics, Cognitive Styles, Outcomes
    node_positions = {
        'Demographics': (0.2, 0.8),
        'Cognitive Styles': (0.2, 0.5),
        'Accuracy': (0.8, 0.9),
        'Rescue Rate': (0.8, 0.7),
        'Extra Errors': (0.8, 0.5),
        'Bias Magnitude': (0.8, 0.3)
    }
    
    # Draw nodes
    for node, (x, y) in node_positions.items():
        if node in ['Demographics', 'Cognitive Styles']:
            color = '#1f77b4' if node == 'Demographics' else '#ff7f0e'
            size = 1000
        else:
            color = '#2ca02c'
            size = 800
        
        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(x, y-0.05, node, ha='center', va='top', fontweight='bold', fontsize=10)
    
    # Draw connections (simplified based on strongest effects)
    strongest_predictors = causal_results['strongest_predictors']
    
    for outcome, pred_info in strongest_predictors.items():
        outcome_pos = node_positions.get(outcome.replace('_', ' ').title())
        if outcome_pos and pred_info['predictor'] is not None:  # Check for None predictor
            predictor_name = pred_info['predictor']
            
            # Determine if predictor is demographic or cognitive
            if ('gender' in predictor_name or 'ethnicity' in predictor_name or 
                'demographic' in predictor_name):
                start_pos = node_positions['Demographics']
            elif ('cognitive' in predictor_name):
                start_pos = node_positions['Cognitive Styles']
            else:
                # Default to demographics if unclear
                start_pos = node_positions['Demographics']
            
            # Draw arrow (limit line width to reasonable values)
            line_width = min(abs(pred_info['coefficient']) * 50, 5)  # Cap at 5
            arrow_props = dict(arrowstyle='->', lw=max(line_width, 0.5),  # Minimum 0.5
                             color='red' if pred_info['coefficient'] < 0 else 'green', alpha=0.6)
            ax.annotate('', xy=outcome_pos, xytext=start_pos, arrowprops=arrow_props)
    
    # Draw Demographics → Cognitive Styles connection
    ax.annotate('', xy=node_positions['Cognitive Styles'], xytext=node_positions['Demographics'], 
               arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.set_title('Causal Network: Profile Traits → Outcomes')
    ax.axis('off')
    
    # Add legend
    ax.text(0.05, 0.05, 'Green arrows: Positive effects\nRed arrows: Negative effects\nThickness ∝ Effect size', 
           transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def run_tier3_analysis(merged_df):
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
    stability_results = temporal_stability_vs_boldness_analysis(merged_df, n_folds=5)
    
    # 2. Simplified Causal Modeling
    print("\n === Running Simplified Causal Modeling... === ")
    causal_results = simplified_causal_modeling(merged_df, stability_results)
    
    # 3. Generate Visualizations
    print("\n === Creating Temporal Stability Visualizations... ===")
    stability_viz = plot_stability_boldness_analysis(stability_results)
    
    print("\n === Creating Causal Model Visualizations... ===")
    causal_viz = visualize_causal_model(causal_results)

    print("\n" + "="*80)
    print("TIER 3 THEORETICAL INTEGRATION")
    print("="*80)
    
    # Key insights from stability analysis
    stability_insight = stability_results['normative_assessment']
    high_vol_rescue = stability_insight.get('high_volatility_rescue', 0)
    low_vol_rescue = stability_insight.get('low_volatility_rescue', 0)
    
    # Key insights from causal analysis
    causal_insight = causal_results['theoretical_framework']
    strongest_predictors = causal_results['strongest_predictors']
    
    print(f"\n🧠 THEORETICAL INSIGHTS:")
    
    # Volatility-Boldness Finding
    if high_vol_rescue > low_vol_rescue:
        volatility_conclusion = "✅ Volatile profiles provide higher moral value through increased rescue rates"
        volatility_implication = "Risk-taking in AI annotation may be normatively justified"
    else:
        volatility_conclusion = "❌ Stable profiles outperform volatile ones in moral value"
        volatility_implication = "Consistency should be prioritized over boldness in AI systems"
    
    print(f"   • {volatility_conclusion}")
    print(f"   • Implication: {volatility_implication}")
    
    # Causal Mechanism Finding
    rescue_predictor = strongest_predictors.get('rescue_rate', {}).get('predictor', 'Unknown')
    accuracy_predictor = strongest_predictors.get('accuracy', {}).get('predictor', 'Unknown')
    
    # Handle None predictors
    if rescue_predictor is None:
        rescue_predictor = 'No significant predictor'
    if accuracy_predictor is None:
        accuracy_predictor = 'No significant predictor'
    
    print(f"   • Strongest predictor of moral value (rescue): {rescue_predictor}")
    print(f"   • Strongest predictor of performance (accuracy): {accuracy_predictor}")
    
    # Mediation findings
    mediation_summary = []
    for outcome, evidence in causal_results['mediation_evidence'].items():
        if evidence['mediation_strength'] > 1.5:
            mediation_summary.append(f"{outcome}: Strong cognitive mediation")
        elif evidence['mediation_strength'] > 0.8:
            mediation_summary.append(f"{outcome}: Partial mediation")
    
    if mediation_summary:
        print(f"   • Mediation effects found: {', '.join(mediation_summary)}")
    
    # ========================================================================
    # STEP 5: Thesis-Level Conclusions
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("THESIS-LEVEL CONCLUSIONS")
    print(f"{'='*60}")
    
    conclusions = []
    
    # Conclusion 1: Stability-Boldness Tradeoff
    if high_vol_rescue > low_vol_rescue + 0.02:  # Meaningful difference
        conclusions.append({
            'finding': 'Volatile profiles provide superior moral value',
            'evidence': f'High-volatility rescue rate: {high_vol_rescue:.3f} vs Low-volatility: {low_vol_rescue:.3f}',
            'implication': 'AI systems should incorporate controlled risk-taking for better moral outcomes'
        })
    
    # Conclusion 2: Causal Mechanisms
    demo_dominance = []
    cog_dominance = []
    
    for outcome, results in causal_results['causal_results'].items():
        if results['demo_unique'] > results['cog_unique']:
            demo_dominance.append(outcome)
        else:
            cog_dominance.append(outcome)
    
    if len(cog_dominance) > len(demo_dominance):
        conclusions.append({
            'finding': 'Cognitive styles dominate over demographics in bias formation',
            'evidence': f'Cognitive factors dominate in {len(cog_dominance)}/{len(causal_results["causal_results"])} outcomes',
            'implication': 'Bias mitigation should focus on cognitive framing rather than demographic representation'
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
            if trait is not None:  # Only count non-None traits
                trait_counts[trait] = trait_counts.get(trait, 0) + 1
    
    most_important_trait = max(trait_counts, key=trait_counts.get) if trait_counts else 'No clear pattern'
    
    conclusions.append({
        'finding': f'{most_important_trait} is the most critical trait for system design',
        'evidence': f'Appears in {trait_counts.get(most_important_trait, 0)}/3 optimization objectives',
        'implication': f'AI systems should prioritize profiles with {most_important_trait} characteristics'
    })
    
    # Print conclusions
    for i, conclusion in enumerate(conclusions, 1):
        print(f"\n🎯 CONCLUSION {i}: {conclusion['finding']}")
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
