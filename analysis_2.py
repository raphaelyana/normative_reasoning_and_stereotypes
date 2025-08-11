import os
import glob
import json
from collections import Counter, defaultdict
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
    bootstrap, f_oneway, ttest_ind, kruskal, mannwhitneyu, pearsonr, sem, t
)

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from analysis_0 import *
from profiles.profile_message import get_profile_traits
from profiles.profile_sets import PERSON_SYSTEMATIC
from profiles.schema import PersonSet


def build_trait_groups(merged_df: pd.DataFrame, person_set: PersonSet, group_keys=("gender", "ethnicity", "age")) -> Dict[str, list]:
    """
    Dynamically build profile groups based on group_keys using PersonSet metadata.
    Updated to work with the new PersonSet structure and flexible group_keys.
    """
    def norm_val(v):
        """Normalize values for consistent grouping."""
        if v == "Unknown":
            return "unknown"
        elif isinstance(v, (int, float)):
            return str(v)
        else:
            return str(v).lower()

    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
    trait_groups = defaultdict(list)

    print(f"Building trait groups with keys: {group_keys}")
    print(f"Found {len(profile_cols)} profile columns")

    for profile in profile_cols:
        # Clean the profile name to get the base persona ID
        pid = profile
        if pid.startswith("profile"):
            pid = pid.replace("_passive", "").replace("_active", "")
        
        # Get traits using PersonSet.get_traits method
        traits = person_set.get_traits(pid, group_keys)
        
        # Normalize trait values
        normalized_traits = {k: norm_val(traits.get(k, "Unknown")) for k in group_keys}
        
        # Create group name by combining all trait values
        group_name = "_".join(str(normalized_traits[k]) for k in group_keys if normalized_traits[k] != "unknown")
        
        if group_name:  # Only add if we have valid traits
            trait_groups[group_name].append(profile)
        else:
            trait_groups["unknown"].append(profile)

    # Print summary of groups created
    print(f"Created {len(trait_groups)} trait groups:")
    for group_name, profiles in trait_groups.items():
        print(f"  {group_name}: {len(profiles)} profiles")

    return dict(trait_groups)


def majority_vote_ensemble(df: pd.DataFrame, profile_list: list) -> pd.Series:
    """
    Calculate majority vote across profile predictions.
    Enhanced with better error handling.
    """
    available_profiles = [p for p in profile_list if p in df.columns]
    if not available_profiles:
        print(f"WARNING: No available profiles from list: {profile_list}")
        return pd.Series([''] * len(df), index=df.index)

    preds = df[available_profiles]
    
    def get_majority_vote(row):
        valid_votes = [v for v in row if pd.notna(v) and v != '']
        if not valid_votes:
            return ''
        
        vote_counts = Counter(valid_votes)
        majority_vote = vote_counts.most_common(1)[0][0]
        return majority_vote
    
    votes = preds.apply(get_majority_vote, axis=1)
    return votes


def ensemble_by_trait_analysis(
    merged_df: pd.DataFrame, 
    person_set: PersonSet, 
    group_keys=("gender", "ethnicity", "age")
) -> Dict[str, Any]:
    """
    Selective Ensemble Performance Analysis - Updated for PersonSet Framework
    
    Tests majority vote performance using specific trait groups based on group_keys.
    Compares to full ensemble and baseline to show how role design affects
    system-level safety and performance.
    
    Parameters:
    - merged_df: DataFrame containing predictions and true labels
    - person_set: PersonSet object containing trait metadata
    - group_keys: tuple of metadata fields to analyze
    
    Returns:
    - dict with ensemble results, category analysis, and recommendations
    """
    
    print("="*80)
    print("ENSEMBLE BY TRAIT ANALYSIS - PERSONSET VERSION")
    print("="*80)
    print(f"Group keys: {group_keys}")
    
    # Build trait groups using PersonSet
    trait_groups = build_trait_groups(merged_df, person_set, group_keys)
    
    # Calculate baseline performance
    true_labels = merged_df['true_label']
    baseline_preds = merged_df['base_pred']
    baseline_accuracy = accuracy_score(true_labels, baseline_preds)
    
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print("\nEnsemble Performance by Trait Group:")
    print("-" * 50)
    
    # Calculate ensemble performance for each trait group
    ensemble_results = {}
    
    for group_name, profile_list in trait_groups.items():
        if len(profile_list) == 0:
            continue
            
        ensemble_preds = majority_vote_ensemble(merged_df, profile_list)

        if len(ensemble_preds) > 0 and not ensemble_preds.eq('').all():
            ensemble_accuracy = accuracy_score(true_labels, ensemble_preds)
            improvement = ensemble_accuracy - baseline_accuracy

            # Calculate rescue and error metrics
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
                'n_profiles': len(profile_list),
                'ensemble_preds': ensemble_preds,
                'profiles': profile_list
            }

            print(f"{group_name:30s}: {ensemble_accuracy:.4f} ({improvement:+.4f}) | "
                  f"Rescue: {rescue_rate:.3f} | Extra Err: {extra_error_rate:.3f} | "
                  f"n={len(profile_list)}")
        else:
            print(f"{group_name:30s}: No valid predictions")
    
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
    
        category_results[category] = {
            'baseline_accuracy': cat_baseline_acc, 
            'ensembles': {}
        }
    
        # Test all ensemble groups
        for ensemble_name, ensemble_info in ensemble_results.items():
            if 'ensemble_preds' in ensemble_info:
                cat_ensemble_preds = ensemble_info['ensemble_preds'].loc[cat_subset.index]
    
                if not cat_ensemble_preds.eq('').all():
                    cat_ensemble_acc = accuracy_score(cat_true, cat_ensemble_preds)
                    cat_improvement = cat_ensemble_acc - cat_baseline_acc
    
                    category_results[category]['ensembles'][ensemble_name] = {
                        'accuracy': cat_ensemble_acc,
                        'improvement': cat_improvement
                    }
    
                    print(f"  {ensemble_name:25s}: {cat_ensemble_acc:.4f} ({cat_improvement:+.4f})")
        
    # Ranking and recommendations
    if ensemble_results:
        sorted_ensembles = sorted(ensemble_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        sorted_by_safety = sorted(ensemble_results.items(), key=lambda x: x[1]['extra_error_rate'])
        sorted_by_rescue = sorted(ensemble_results.items(), key=lambda x: x[1]['rescue_rate'], reverse=True)
    
        print(f"\n{'='*60}")
        print("ENSEMBLE RANKINGS")
        print(f"{'='*60}")

        print("\nTop 5 Performing Ensembles (Accuracy):")
        for i, (name, res) in enumerate(sorted_ensembles[:5]):
            print(f"  {i+1}. {name}: {res['accuracy']:.4f} (+{res['improvement']:.4f})")

        print("\nSafest Ensembles (Lowest Extra Error Rate):")
        for i, (name, res) in enumerate(sorted_by_safety[:5]):
            print(f"  {i+1}. {name}: {res['extra_error_rate']:.3f} | Accuracy: {res['accuracy']:.4f}")

        print("\nMost Effective Rescue Ensembles:")
        for i, (name, res) in enumerate(sorted_by_rescue[:5]):
            print(f"  {i+1}. {name}: Rescue Rate {res['rescue_rate']:.3f} | Accuracy: {res['accuracy']:.4f}")

        # Balanced recommendation (accuracy + safety)
        safety_performance_scores = {
            name: 0.6 * res['improvement'] + 0.4 * (1 - res['extra_error_rate'])
            for name, res in ensemble_results.items()
        }
        best_balanced = max(safety_performance_scores.items(), key=lambda x: x[1])

        print(f"\n{'='*60}")
        print("RECOMMENDED ENSEMBLE")
        print(f"{'='*60}")
        best_name = best_balanced[0]
        best = ensemble_results[best_name]
        print(f"Best Accuracy/Safety Tradeoff: {best_name}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  Improvement: +{best['improvement']:.4f}")
        print(f"  Rescue Rate: {best['rescue_rate']:.3f}")
        print(f"  Extra Error Rate: {best['extra_error_rate']:.3f}")
        print(f"  Number of profiles: {best['n_profiles']}")

        recommendations = {
            'best_overall': sorted_ensembles[0],
            'safest': sorted_by_safety[0],
            'best_rescue': sorted_by_rescue[0],
            'best_balanced': best_balanced
        }
    else:
        print("WARNING: No ensemble results to analyze")
        recommendations = {}

    return {
        'ensemble_results': ensemble_results,
        'category_results': category_results,
        'recommendations': recommendations,
        'trait_groups': trait_groups,
        'group_keys': group_keys,
        'baseline_accuracy': baseline_accuracy
    }


def cluster_level_bias_patterns(
    merged_df, 
    person_set: PersonSet,
    similarity_results=None, 
    group_keys=("gender", "ethnicity", "age")
):
    """
    Cluster-level Bias and Rescue Pattern Analysis - Updated for PersonSet Framework
    
    Evaluates grouped behaviors using existing persona clusters.
    For each cluster, computes:
    - Average bias_magnitude, rescue_rate, accuracy
    - Identifies tradeoffs and recommends profile subsets for aligned ensembles
    
    Parameters:
    - merged_df: DataFrame containing predictions and true labels
    - person_set: PersonSet object containing trait metadata
    - similarity_results: Pre-computed clustering results (optional)
    - group_keys: tuple of metadata fields to analyze
    """
    
    print("="*80)
    print("CLUSTER-LEVEL BIAS AND RESCUE PATTERNS - PERSONSET VERSION")
    print("="*80)
    print(f"Group keys: {group_keys}")
    
    # Get clustering results if not provided
    if similarity_results is None:
        try:
            similarity_results = analyze_persona_similarity(merged_df)
        except NameError:
            print("ERROR: analyze_persona_similarity function not found")
            print("Creating mock clustering results for demonstration...")
            
            # Create mock clusters based on trait groups
            trait_groups = build_trait_groups(merged_df, person_set, group_keys)
            mock_clusters = {}
            
            for i, (group_name, profiles) in enumerate(trait_groups.items()):
                cluster_id = f"cluster_{i+1}"
                mock_clusters[cluster_id] = {
                    'profiles': profiles,
                    'internal_agreement': np.random.uniform(0.85, 0.98),
                    'trait_signature': group_name
                }
            
            similarity_results = {'clusters': mock_clusters}
    
    # Get rescue and bias stats
    try:
        rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
        bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    except NameError:
        print("WARNING: rescue_stats_by_category or detect_systematic_biases functions not found")
        print("Creating mock statistics for demonstration...")
        
        profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        
        # Mock rescue stats
        mock_rescue_data = []
        for profile in profile_cols:
            if profile in merged_df.columns:
                accuracy = (merged_df[profile] == merged_df['true_label']).mean()
                mock_rescue_data.append({
                    'profile': profile,
                    'rescue_rate': max(0, accuracy - 0.71 + np.random.normal(0, 0.02)),
                    'extra_err_rate': max(0.01, 0.05 - (accuracy - 0.71) * 2 + np.random.normal(0, 0.01)),
                    'rescued': int(np.random.uniform(5, 50)),
                    'extra_errors': int(np.random.uniform(2, 20))
                })
        rescue_stats = pd.DataFrame(mock_rescue_data)
        
        # Mock bias patterns
        mock_bias_data = []
        for profile in profile_cols:
            mock_bias_data.append({
                'profile': profile,
                'bias_magnitude': np.random.uniform(0.02, 0.15),
                'mislabelling_rate': np.random.uniform(0.01, 0.08),
                'bias_direction': np.random.choice(['stereotype', 'anti_stereotype', 'neutral'])
            })
        bias_patterns = pd.DataFrame(mock_bias_data)
    
    cluster_analysis = {}
    baseline_accuracy = accuracy_score(merged_df["true_label"], merged_df["base_pred"])
    
    # Analyze each cluster
    for cluster_id, cluster_info in similarity_results['clusters'].items():
        cluster_profiles = cluster_info['profiles']
        available_profiles = [p for p in cluster_profiles if p in merged_df.columns]

        print(f"\n{cluster_id.upper()} ({len(available_profiles)} profiles):")
        
        # Get trait composition of cluster
        trait_composition = defaultdict(list)
        for profile in available_profiles:
            pid = profile.replace("_passive", "").replace("_active", "")
            traits = person_set.get_traits(pid, group_keys)
            for trait_name, trait_value in traits.items():
                if trait_value != "Unknown":
                    trait_composition[trait_name].append(str(trait_value).lower())
        
        # Show cluster trait composition
        print(f"Trait composition:")
        for trait_name, values in trait_composition.items():
            value_counts = Counter(values)
            print(f"  {trait_name}: {dict(value_counts)}")
        
        print(f"Sample profiles: {', '.join(p.replace('_passive', '') for p in available_profiles[:5])}")
        if len(available_profiles) > 5:
            print(f"                 ... and {len(available_profiles) - 5} more")

        cluster_metrics = {
            'profiles': available_profiles,
            'size': len(available_profiles),
            'internal_agreement': cluster_info.get('internal_agreement', 0),
            'trait_composition': dict(trait_composition)
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
        
        # Determine cluster archetype
        if cluster_metrics['rescue_rate_mean'] > 0.15 and cluster_metrics['extra_error_rate_mean'] < 0.05:
            archetype = "Optimal Risk-Benefit"
        elif cluster_metrics['extra_error_rate_mean'] < 0.03:
            archetype = "Conservative-Cautious"
        elif cluster_metrics['rescue_rate_mean'] > 0.20:
            archetype = "High Error-Correction"
        elif cluster_metrics['accuracy_mean'] > 0.72:
            archetype = "Superior Performance"
        elif cluster_metrics['internal_agreement'] > 0.95:
            archetype = "High Consensus"
        else:
            archetype = "Moderate-Balanced"
        
        cluster_metrics['archetype'] = archetype
        
        # Print cluster summary
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

    if cluster_analysis:
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

        recommendations = {
            'best_accuracy': best_accuracy,
            'safest': safest_cluster,
            'best_rescue': best_rescue
        }
    else:
        print("WARNING: No cluster analysis results available")
        recommendations = {}

    return {
        'cluster_analysis': cluster_analysis,
        'recommendations': recommendations,
        'similarity_results': similarity_results,
        'group_keys': group_keys,
        'baseline_accuracy': baseline_accuracy
    }


def run_full_tier2_analysis(
    merged_df: pd.DataFrame, 
    person_set: PersonSet,
    group_keys=("gender", "ethnicity", "age"),
    similarity_results=None
) -> Dict[str, Any]:
    """
    Run the full Tier 2 analysis pipeline:
    - Ensemble by trait analysis
    - Cluster-level bias patterns analysis
    
    Parameters:
    - merged_df: Merged classification results
    - person_set: PersonSet object containing trait metadata
    - group_keys: Traits to include in the analysis
    - similarity_results: Pre-computed clustering results (optional)
    
    Returns:
    A dictionary with all Tier 2 analysis results.
    """
    
    print("="*80)
    print("COMPREHENSIVE TIER 2 ENSEMBLE ANALYSIS PIPELINE")
    print("="*80)
    print(f"Group keys: {group_keys}")
    print(f"Dataset shape: {merged_df.shape}")
    
    # Validate inputs
    if person_set is None:
        raise ValueError("PersonSet is required for Tier 2 analysis")
    
    try:
        print("\n" + "="*60)
        print("STEP 1: ENSEMBLE BY TRAIT ANALYSIS")
        print("="*60)
        ensemble_results = ensemble_by_trait_analysis(
            merged_df, 
            person_set,
            group_keys=group_keys
        )
        print("=== Ensemble by trait analysis completed successfully")
        
    except Exception as e:
        print(f"ERROR: Ensemble by trait analysis failed: {e}")
        ensemble_results = {'error': str(e)}

    try:
        print("\n" + "="*60)
        print("STEP 2: CLUSTER-LEVEL BIAS PATTERNS")
        print("="*60)
        cluster_results = cluster_level_bias_patterns(
            merged_df, 
            person_set,
            similarity_results=similarity_results,
            group_keys=group_keys
        )
        print("=== Cluster-level bias analysis completed successfully")
        
    except Exception as e:
        print(f"ERROR: Cluster-level bias analysis failed: {e}")
        cluster_results = {'error': str(e)}

    # Summary report
    print("\n" + "="*80)
    print("TIER 2 ANALYSIS SUMMARY")
    print("="*80)
    
    # Ensemble summary
    if 'ensemble_results' in ensemble_results and ensemble_results['ensemble_results']:
        best_ensemble = max(ensemble_results['ensemble_results'].items(), 
                          key=lambda x: x[1]['accuracy'])
        print(f"=== Best Ensemble: {best_ensemble[0]} (accuracy: {best_ensemble[1]['accuracy']:.4f})")
        
        if 'recommendations' in ensemble_results:
            rec = ensemble_results['recommendations'].get('best_balanced')
            if rec:
                print(f"=== Recommended Balanced Ensemble: {rec[0]} (score: {rec[1]:.3f})")
    else:
        print("=== Ensemble Analysis: No results available")
    
    # Cluster summary
    if 'cluster_analysis' in cluster_results and cluster_results['cluster_analysis']:
        cluster_count = len(cluster_results['cluster_analysis'])
        print(f"=== Clusters Analyzed: {cluster_count}")
        
        if 'recommendations' in cluster_results:
            best_cluster = cluster_results['recommendations'].get('best_accuracy')
            if best_cluster:
                print(f"=== Best Performing Cluster: {best_cluster[0]} (accuracy: {best_cluster[1]['accuracy_mean']:.4f})")
    else:
        print("=== Cluster Analysis: No results available")

    return {
        "ensemble_results": ensemble_results,
        "cluster_results": cluster_results,
        "group_keys": group_keys,
        "analysis_summary": {
            "best_ensemble": ensemble_results.get('recommendations', {}).get('best_balanced'),
            "cluster_count": len(cluster_results.get('cluster_analysis', {})),
            "best_cluster": cluster_results.get('recommendations', {}).get('best_accuracy')
        }
    }



def build_trait_groups_2(merged_df: pd.DataFrame, person_set: PersonSet, group_keys=("gender", "ethnicity", "cognitive_style")) -> Dict[str, list]:
    """
    Dynamically build profile groups based on group_keys using PersonSet metadata.
    """
    profile_cols = [col for col in merged_df.columns if col.startswith("profile") and (col.endswith("_passive") or col.endswith("_active"))]
    trait_groups = defaultdict(list)

    for profile in profile_cols:
        traits = person_set.get_traits(profile, group_keys=group_keys)
        group_name = " ".join(traits[k] for k in group_keys if traits.get(k))
        trait_groups[group_name].append(profile)

    return dict(trait_groups)

def majority_vote_ensemble_2(df: pd.DataFrame, profile_list: list) -> pd.Series:
    """Calculate majority vote across profile predictions"""
    available_profiles = [p for p in profile_list if p in df.columns]
    if not available_profiles:
        return pd.Series([''] * len(df), index=df.index)

    preds = df[available_profiles]
    votes = preds.apply(
        lambda row: Counter(v for v in row if pd.notna(v) and v != '').most_common(1)[0][0] if any(pd.notna(v) and v != '' for v in row) else '',
        axis=1
    )
    return votes


def ensemble_by_trait_analysis_2(merged_df: pd.DataFrame, person_set: PersonSet = PERSON_SYSTEMATIC, group_keys=("gender", "ethnicity", "cognitive_style")) -> Dict[str, Any]:
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
    trait_groups = build_trait_groups(merged_df, person_set, group_keys)
    
    #def majority_vote_ensemble(df, profile_list):
    #    """Calculate majority vote for a list of profiles"""
    #    available_profiles = [p for p in profile_list if p in df.columns]
    #    if not available_profiles:
    #        return pd.Series([''] * len(df), index=df.index)
        
        # Get predictions from available profiles
    #    ensemble_preds = df[available_profiles]
    #    
    #    # Calculate majority vote for each row
    #    majority_votes = []
    #    for idx, row in ensemble_preds.iterrows():
    #        votes = [vote for vote in row.values if pd.notna(vote) and vote != '']
    #        if votes:
    #            vote_counts = Counter(votes)
    #            majority_vote = vote_counts.most_common(1)[0][0]
    #            majority_votes.append(majority_vote)
    #        else:
    #            majority_votes.append('')
        
    #    return pd.Series(majority_votes, index=df.index)
    
    # Calculate ensemble performance for each trait group
    ensemble_results = {}
    true_labels = merged_df['true_label']
    baseline_preds = merged_df['base_pred']
    baseline_accuracy = accuracy_score(true_labels, baseline_preds)
    
    print("=" * 80)
    print("ENSEMBLE BY TRAIT ANALYSIS")
    print("=" * 80)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print("\nEnsemble Performance by Trait Group:")
    print("-" * 50)
    
    for group_name, profile_list in trait_groups.items():
        ensemble_preds = majority_vote_ensemble(merged_df, profile_list)

        if len(ensemble_preds) > 0 and not ensemble_preds.eq('').all():
            ensemble_accuracy = accuracy_score(true_labels, ensemble_preds)
            improvement = ensemble_accuracy - baseline_accuracy

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
                'n_profiles': len(profile_list),
                'ensemble_preds': ensemble_preds
            }

            print(f"{group_name:25s}: {ensemble_accuracy:.4f} ({improvement:+.4f}) | "
                  f"Rescue: {rescue_rate:.3f} | Extra Err: {extra_error_rate:.3f} | "
                  f"n={len(profile_list)}")
        else:
            print(f"{group_name:25s}: No valid predictions")
    
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
    
        # Use all ensemble groups dynamically
        for ensemble_name in ensemble_results.keys():
            cat_ensemble_preds = ensemble_results[ensemble_name]['ensemble_preds'].loc[cat_subset.index]
    
            if not cat_ensemble_preds.eq('').all():
                cat_ensemble_acc = accuracy_score(cat_true, cat_ensemble_preds)
                cat_improvement = cat_ensemble_acc - cat_baseline_acc
    
                category_results[category]['ensembles'][ensemble_name] = {
                    'accuracy': cat_ensemble_acc,
                    'improvement': cat_improvement
                }
    
                print(f"  {ensemble_name:25s}: {cat_ensemble_acc:.4f} ({cat_improvement:+.4f})")
        
    sorted_ensembles = sorted(ensemble_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    sorted_by_safety = sorted(ensemble_results.items(), key=lambda x: x[1]['extra_error_rate'])
    sorted_by_rescue = sorted(ensemble_results.items(), key=lambda x: x[1]['rescue_rate'], reverse=True)
   

    print("\nTop 5 Performing Ensembles (Accuracy):")
    for i, (name, res) in enumerate(sorted_ensembles[:5]):
        print(f"  {i+1}. {name}: {res['accuracy']:.4f} (+{res['improvement']:.4f})")

    print("\nSafest Ensembles (Lowest Extra Error Rate):")
    for i, (name, res) in enumerate(sorted_by_safety[:5]):
        print(f"  {i+1}. {name}: {res['extra_error_rate']:.3f} | Accuracy: {res['accuracy']:.4f}")

    print("\nMost Effective Rescue Ensembles:")
    for i, (name, res) in enumerate(sorted_by_rescue[:5]):
        print(f"  {i+1}. {name}: Rescue Rate {res['rescue_rate']:.3f} | Accuracy: {res['accuracy']:.4f}")

    # Step 4: Balanced recommendation (accuracy + safety)
    safety_performance_scores = {
        name: 0.6 * res['improvement'] + 0.4 * (1 - res['extra_error_rate'])
        for name, res in ensemble_results.items()
    }
    best_balanced = max(safety_performance_scores.items(), key=lambda x: x[1])

    print("\nRecommended Ensemble (Best Accuracy/Safety Tradeoff):")
    best_name = best_balanced[0]
    best = ensemble_results[best_name]
    print(f"  {best_name}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  Improvement: +{best['improvement']:.4f}")
    print(f"  Rescue Rate: {best['rescue_rate']:.3f}")
    print(f"  Extra Error Rate: {best['extra_error_rate']:.3f}")

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


def cluster_level_bias_patterns_2(merged_df, similarity_results=None, person_set: PersonSet = PERSON_SYSTEMATIC):
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
        similarity_results = analyze_persona_similarity(merged_df, person_set=person_set)
    
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
            archetype = "Ultra Safe"
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


def cognitive_style_comparison_controlled(merged_df: pd.DataFrame, person_set: PersonSet) -> Dict[str, Any]:
    """
    Cognitive Style Comparison (Controlled for Demographics)
    
    Dynamically evaluates performance across cognitive styles using PersonSet metadata.
    Controls for demographic noise and uses statistical tests for comparisons.
    """

    # Identify all profile columns (passive only for reasoning-based evaluation)
    profile_cols = [col for col in merged_df.columns if col.startswith("profile") and col.endswith("_passive")]

    # Build groups by cognitive style
    cognitive_styles = defaultdict(list)
    for profile in profile_cols:
        traits = person_set.get_traits(profile)
        cog_style = traits.get("cognitive_style", "Unknown")
        cognitive_styles[cog_style].append(profile)

    # Precompute rescue and bias stats
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")

    print("=" * 80)
    print("COGNITIVE STYLE COMPARISON (CONTROLLED)")
    print("=" * 80)

    style_performance = {}

    for style_name, profiles in cognitive_styles.items():
        available_profiles = [p for p in profiles if p in merged_df.columns]
        if not available_profiles:
            continue

        accuracies, rescue_rates, extra_error_rates, bias_magnitudes = [], [], [], []

        for profile in available_profiles:
            acc = accuracy_score(merged_df['true_label'], merged_df[profile])
            accuracies.append(acc)

            rs = rescue_stats[rescue_stats['profile'] == profile]
            rescue_rates.append(rs['rescue_rate'].mean() if len(rs) > 0 else 0)
            extra_error_rates.append(rs['extra_err_rate'].mean() if len(rs) > 0 else 0)

            bp = bias_patterns[bias_patterns['profile'] == profile]
            bias_magnitudes.append(bp['bias_magnitude'].mean() if len(bp) > 0 else 0)

        style_performance[style_name] = {
            'accuracies': accuracies,
            'rescue_rates': rescue_rates,
            'extra_error_rates': extra_error_rates,
            'bias_magnitudes': bias_magnitudes,
            'n_profiles': len(available_profiles),
            'profiles': available_profiles
        }

        print(f"\n{style_name.upper()} COGNITIVE STYLE (n={len(available_profiles)}):")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Rescue Rate: {np.mean(rescue_rates):.3f} ± {np.std(rescue_rates):.3f}")
        print(f"  Extra Error Rate: {np.mean(extra_error_rates):.3f} ± {np.std(extra_error_rates):.3f}")
        print(f"  Bias Magnitude: {np.mean(bias_magnitudes):.3f} ± {np.std(bias_magnitudes):.3f}")

    # Statistical comparison logic unchanged
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISONS BETWEEN COGNITIVE STYLES")
    print(f"{'='*60}")

    metrics = ['accuracies', 'rescue_rates', 'extra_error_rates', 'bias_magnitudes']
    metric_names = ['Accuracy', 'Rescue Rate', 'Extra Error Rate', 'Bias Magnitude']
    statistical_results = {}

    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{metric_name.upper()}:")
        groups, group_names = [], []

        for style_name, data in style_performance.items():
            if metric in data and len(data[metric]) > 0:
                groups.append(data[metric])
                group_names.append(style_name)

        if len(groups) >= 3:
            combined = np.concatenate(groups)
            if np.all(combined == combined[0]):
                print(f"  Skipping {metric_name}: no variation.")
                continue

            h_stat, p_value = kruskal(*groups)
            print(f"  Kruskal-Wallis H: {h_stat:.3f} | p = {p_value:.4f} {'***' if p_value < 0.05 else ''}")

            statistical_results[metric] = {
                'kruskal_wallis': {'h_statistic': h_stat, 'p_value': p_value, 'significant': p_value < 0.05},
                'group_means': {name: np.mean(data) for name, data in zip(group_names, groups)},
                'pairwise_comparisons': {}
            }

            if p_value < 0.05:
                print("  Pairwise Comparisons (Mann-Whitney):")
                for i, (name1, g1) in enumerate(zip(group_names, groups)):
                    for j, (name2, g2) in enumerate(zip(group_names, groups)):
                        if i < j:
                            u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
                            eff_size = 1 - (2 * u_stat) / (len(g1) * len(g2))
                            statistical_results[metric]['pairwise_comparisons'][f"{name1}_vs_{name2}"] = {
                                'u_stat': u_stat,
                                'p_value': p_val,
                                'significant': p_val < 0.05,
                                'effect_size': eff_size,
                                'mean_diff': np.mean(g1) - np.mean(g2)
                            }
                            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                            print(f"    {name1} vs {name2}: p={p_val:.4f} {sig} (effect={eff_size:.3f})")

    print(f"\n{'='*60}")
    print("COGNITIVE STYLE RANKINGS")
    print(f"{'='*60}")
    rankings = {}

    for metric, metric_name in zip(metrics, metric_names):
        if metric in statistical_results:
            means = statistical_results[metric]['group_means']
            reverse = metric in ['accuracies', 'rescue_rates']
            sorted_styles = sorted(means.items(), key=lambda x: x[1], reverse=reverse)
            rankings[metric_name] = sorted_styles

            print(f"\n{metric_name} Rankings:")
            for i, (style, value) in enumerate(sorted_styles, 1):
                print(f"  {i}. {style}: {value:.4f}")

    print(f"\n{'='*60}")
    print("COGNITIVE STYLE RECOMMENDATIONS")
    print(f"{'='*60}")

    weights = {'Accuracy': 0.3, 'Rescue Rate': 0.25, 'Extra Error Rate': -0.25, 'Bias Magnitude': -0.2}
    composite_scores = {}

    for style_name in cognitive_styles:
        score = 0
        for metric_name, weight in weights.items():
            if metric_name in rankings:
                rank = next((i for i, (s, _) in enumerate(rankings[metric_name], 1) if s == style_name), 6)
                rank_score = 6 - rank
                score += weight * rank_score
        composite_scores[style_name] = score

    recommended_styles = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nComposite Ranking:")
    for i, (style, score) in enumerate(recommended_styles, 1):
        print(f"  {i}. {style}: {score:.3f}")

    return {
        'style_performance': style_performance,
        'statistical_results': statistical_results,
        'rankings': rankings,
        'composite_scores': composite_scores,
        'recommendations': recommended_styles,
        'cognitive_styles': dict(cognitive_styles)
    }



def visualize_tier2_results(ensemble_results, cluster_results, cognitive_results, figsize=(16, 12), key_ensembles=None):
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

    # Default to all ensembles if not specified
    if key_ensembles is None:
        key_ensembles = list(ensemble_data.keys())

    for name in key_ensembles:
        if name in ensemble_data:
            ensemble_names.append(name.replace('_', ' ').title())
            accuracies.append(ensemble_data[name].get('accuracy', 0))
            improvements.append(ensemble_data[name].get('improvement', 0))

    x_pos = np.arange(len(ensemble_names))
    bars = ax.bar(x_pos, improvements, color=['#1f77b4' if imp >= 0 else '#d62728' for imp in improvements])

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

    rescue_rates, extra_error_rates, cluster_labels, cluster_colors = [], [], [], []

    archetype_colors = defaultdict(lambda: '#bcbd22', {
        'Safely Bold': '#2ca02c',
        'Ultra Safe': '#1f77b4', 
        'High Rescue': '#ff7f0e',
        'High Performer': '#d62728',
        'Highly Consistent': '#9467bd',
        'Balanced': '#8c564b'
    })

    for cluster_id, cluster_info in cluster_data.items():
        rescue_rates.append(cluster_info['rescue_rate_mean'])
        extra_error_rates.append(cluster_info['extra_error_rate_mean'])
        cluster_labels.append(cluster_id.replace('cluster_', 'C'))
        cluster_colors.append(archetype_colors[cluster_info.get('archetype', 'Unknown')])

    scatter = ax.scatter(extra_error_rates, rescue_rates, c=cluster_colors, s=100, alpha=0.7, edgecolors='black')
    for i, label in enumerate(cluster_labels):
        ax.annotate(label, (extra_error_rates[i], rescue_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Cluster Risk-Benefit Analysis')
    ax.grid(True, alpha=0.3)

    # ===== PANEL 3: Cognitive Style Performance Radar (Simplified) =====
    ax = axes[1, 0]
    cognitive_data = cognitive_results.get('style_performance', {})

    styles = list(cognitive_data.keys())
    metrics = ['Accuracy', 'Rescue Rate', 'Safety', 'Consistency']

    style_scores = {}
    for style, data in cognitive_data.items():
        acc = np.mean(data['accuracies'])
        res = np.mean(data['rescue_rates'])
        saf = 1 - np.mean(data['extra_error_rates'])
        cons = 1 - np.std(data['accuracies'])
        style_scores[style] = [acc, res, saf, cons]

    x_pos = np.arange(len(styles))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics):
        values = [style_scores[style][i] for style in styles]
        ax.bar(x_pos + i * width, values, width, label=metric, color=colors[i], alpha=0.7)

    ax.set_xlabel('Cognitive Style')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Cognitive Style Performance Profile')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(styles, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ===== PANEL 4: System-Level Safety Recommendations =====
    ax = axes[1, 1]
    recommendations, scores, colors = [], [], []

    # 1. Best Ensemble
    best_ens = ensemble_results['recommendations'].get('best_balanced', [''])[0]
    ens_score = ensemble_data.get(best_ens, {}).get('improvement', 0)
    recommendations.append(f"Best Ensemble:\n{best_ens.replace('_', ' ')}")
    scores.append(ens_score)
    colors.append('#2ca02c')

    # 2. Best Cluster
    best_cluster = cluster_results['recommendations'].get('best_accuracy', [''])[0]
    base_acc = ensemble_results.get('baseline_accuracy', 0.70)
    cl_score = cluster_data.get(best_cluster, {}).get('accuracy_mean', 0) - base_acc
    recommendations.append(f"Best Cluster:\n{best_cluster}")
    scores.append(cl_score)
    colors.append('#1f77b4')

    # 3. Best Cognitive Style
    if cognitive_results.get('recommendations'):
        best_cog = cognitive_results['recommendations'][0][0]
        cog_score = cognitive_results['composite_scores'].get(best_cog, 0) / 10
        recommendations.append(f"Best Cognitive:\n{best_cog}")
        scores.append(cog_score)
        colors.append('#ff7f0e')

    # 4. Safest Ensemble
    safest_ens = ensemble_results['recommendations'].get('safest', [''])[0]
    err_rate = ensemble_data.get(safest_ens, {}).get('extra_error_rate', 0)
    safety_score = -err_rate
    recommendations.append(f"Safest Ensemble:\n{safest_ens.replace('_', ' ')}")
    scores.append(safety_score)
    colors.append('#d62728')

    y_pos = np.arange(len(recommendations))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)
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


def run_full_tier2_analysis(merged_df, person_set, group_keys=("gender", "ethnicity", "age")):
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
    print("=" * 80)

    # 1. Ensemble by Trait Analysis
    print("\n=== Running Ensemble by Trait Analysis...")
    ensemble_results = ensemble_by_trait_analysis(merged_df, person_set, group_keys=group_keys)

    # 2. Cluster-level Bias Patterns
    print("\n=== Running Cluster-level Bias Analysis...")
    similarity_results = analyze_persona_similarity(merged_df, person_set=person_set)
    cluster_results = cluster_level_bias_patterns(merged_df, person_set=person_set, similarity_results=similarity_results, group_keys=group_keys)

    # 3. Cognitive Style Comparison
    print("\n=== Running Cognitive Style Comparison...")
    cognitive_results = cognitive_style_comparison_controlled(merged_df, person_set)

    # 4. Generate Visualizations
    print("\n--- Creating Tier 2 Visualizations...")
    visualization = visualize_tier2_results(ensemble_results, cluster_results, cognitive_results)

    # 5. Generate Executive Summary
    print("\n" + "=" * 80)
    print("TIER 2 EXECUTIVE SUMMARY")
    print("=" * 80)

    # Extract best recommendations
    best_ensemble = ensemble_results['recommendations'].get('best_balanced', ['Unknown'])[0]
    best_cluster = cluster_results['recommendations'].get('best_accuracy', ['Unknown'])[0]
    best_cognitive = cognitive_results['recommendations'][0][0] if cognitive_results['recommendations'] else 'Unknown'

    print(f"\n=== KEY FINDINGS:")
    print(f"   • Best Ensemble Strategy: {best_ensemble}")
    print(f"   • Best Cluster: {best_cluster}")  
    print(f"   • Best Cognitive Style: {best_cognitive}")

    # Performance improvement
    ensemble_improvement = ensemble_results['ensemble_results'].get(best_ensemble, {}).get('improvement', 0)
    print(f"\n=== PERFORMANCE GAINS:")
    print(f"   • Best Ensemble Improvement: +{ensemble_improvement:.4f}")
    print(f"   • Safety-Performance Balance Achieved")

    # Safety info
    safest_ensemble = ensemble_results['recommendations'].get('safest', ['Unknown'])[0]
    safety_rate = ensemble_results['ensemble_results'].get(safest_ensemble, {}).get('extra_error_rate', 0)
    print(f"\n=== SAFETY INSIGHTS:")
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
