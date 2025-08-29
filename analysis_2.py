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
from analysis_tools import get_available_traits, get_analysis_group_keys
from analysis_tools import get_demographic_info, guarded_labelspace_analysis, resolve_plot_dir
from profiles.profile_sets import PERSON_SYSTEMATIC
from profiles.schema import PersonSet
from analysis_tools import has_cognitive_style_data
from cases.cases_config import CaseConfig


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

    debug_traits = defaultdict(set)

    for profile in profile_cols:     
        traits = person_set.get_traits(profile, group_keys)
        
        normalized_traits = {k: norm_val(traits.get(k, "Unknown")) for k in group_keys}
        
        for k, v in normalized_traits.items():
            debug_traits[k].add(v)
        
        group_parts = []
        for k in group_keys:
            val = normalized_traits[k]
            if val != "unknown":
                group_parts.append(val)
            else:
                print(f"  WARNING: {profile} has unknown {k}: {traits}")
                group_parts.append("unknown")
        
        group_name = "_".join(group_parts)
        trait_groups[group_name].append(profile)

    print(f"Discovered trait values:")
    for trait_name, values in debug_traits.items():
        print(f"  {trait_name}: {sorted(values)}")

    print(f"Created {len(trait_groups)} trait groups:")
    for group_name, profiles in sorted(trait_groups.items()):
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
    case: CaseConfig,
    group_keys=("gender", "ethnicity", "age"),
    perf_df: Optional[pd.DataFrame] = None
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

    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        perf_core = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in perf_core.columns:
            tokens_map = perf_core["tokens_per_sample"].to_dict()
        if "cost_per_sample" in perf_core.columns:
            cost_map = perf_core["cost_per_sample"].to_dict()

    
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

            ens_tokens = (float(np.nansum([tokens_map.get(p, np.nan) for p in profile_list]))
                          if tokens_map else np.nan)
            ens_cost   = (float(np.nansum([cost_map.get(p, np.nan)   for p in profile_list]))
                          if cost_map else np.nan)
            
            acc_impr = improvement
            eff_per_1k_tok = (
                acc_impr / (ens_tokens / 1000.0)
                if isinstance(ens_tokens, (int, float)) and np.isfinite(ens_tokens) and ens_tokens > 0
                else np.nan
            )
            eff_per_cost = (
                acc_impr / ens_cost
                if isinstance(ens_cost, (int, float)) and np.isfinite(ens_cost) and ens_cost > 0
                else np.nan
            )


            ensemble_results[group_name] = {
                'accuracy': ensemble_accuracy,
                'improvement': improvement,
                'rescued': rescued,
                'extra_errors': extra_errors,
                'rescue_rate': rescue_rate,
                'extra_error_rate': extra_error_rate,
                'n_profiles': len(profile_list),
                'ensemble_preds': ensemble_preds,
                'profiles': profile_list, 
                "tokens_per_sample_sum": ens_tokens,
                "cost_per_sample_sum": ens_cost,
                "improvement_per_1k_tokens": eff_per_1k_tok,
                "improvement_per_dollar": eff_per_cost,
            }

            print(f"{group_name:30s}: {ensemble_accuracy:.4f} ({improvement:+.4f}) | "
                  f"Rescue: {rescue_rate:.3f} | Extra Err: {extra_error_rate:.3f} | "
                  f"n={len(profile_list)}")
        else:
            print(f"{group_name:30s}: No valid predictions")
    
    print(f"\n{'='*60}")
    print("CATEGORY-SPECIFIC ENSEMBLE PERFORMANCE")
    print(f"{'='*60}")
    
    category_results = {}
    category_cols = getattr(case, "category_cols", None) or ["stereotype_type"]
    
    for cat_col in category_cols:
        if cat_col not in merged_df.columns:
            print(f"WARNING: category_col '{cat_col}' not found in merged_df")
            continue
    
        categories = merged_df[cat_col].dropna().unique()
    
        for category in categories:
            cat_subset = merged_df[merged_df[cat_col] == category]
            cat_true = cat_subset['true_label']
            cat_baseline = cat_subset['base_pred']
            cat_baseline_acc = accuracy_score(cat_true, cat_baseline)
    
            print(f"\n{cat_col.upper()} = {str(category).upper()} (n={len(cat_subset)}):")
            print(f"Baseline accuracy: {cat_baseline_acc:.4f}")
            print("-" * 40)
    
            category_key = f"{cat_col}:{category}"
            category_results[category_key] = {
                'baseline_accuracy': cat_baseline_acc,
                'ensembles': {}
            }
    
            for ensemble_name, ensemble_info in ensemble_results.items():
                if 'ensemble_preds' in ensemble_info:
                    cat_ensemble_preds = ensemble_info['ensemble_preds'].loc[cat_subset.index]
    
                    if not cat_ensemble_preds.eq('').all():
                        cat_ensemble_acc = accuracy_score(cat_true, cat_ensemble_preds)
                        cat_improvement = cat_ensemble_acc - cat_baseline_acc
    
                        category_results[category_key]['ensembles'][ensemble_name] = {
                            'accuracy': cat_ensemble_acc,
                            'improvement': cat_improvement
                        }
                        print(f"  {ensemble_name:25s}: {cat_ensemble_acc:.4f} ({cat_improvement:+.4f})")
            

    if ensemble_results:
        sorted_ensembles = sorted(ensemble_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        sorted_by_safety = sorted(ensemble_results.items(), key=lambda x: x[1]['extra_error_rate'])
        sorted_by_rescue = sorted(ensemble_results.items(), key=lambda x: x[1]['rescue_rate'], reverse=True)
        vals = [
            v.get("improvement_per_1k_tokens")
            for v in ensemble_results.values()
            if isinstance(v.get("improvement_per_1k_tokens"), (int, float)) and np.isfinite(v.get("improvement_per_1k_tokens"))
        ]
        if vals:
            ranked_eff = sorted(
                ensemble_results.items(),
                key=lambda kv: kv[1].get("improvement_per_1k_tokens", -np.inf),
                reverse=True
            )
            print("\nMost budget-efficient ensembles (Δacc per 1k tokens):")
            for i, (name, r) in enumerate(ranked_eff[:5], 1):
                print(f"  {i}. {name}: {r['improvement_per_1k_tokens']:.4f}  | tokens={r.get('tokens_per_sample_sum', np.nan):.1f}")
    
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
    case: CaseConfig,
    similarity_results=None, 
    group_keys=("gender", "ethnicity", "age"),
    perf_df: Optional[pd.DataFrame] = None,
    archetype_parameters = {
        "risk_benefit": [0.15, 0.05],
        "extra_error": 0.03,
        "rescue": 0.2,
        "accuracy": 0.72,
        "internal_agreement": 0.95,
        "accuracy_consensus": True,
    }
):
    """
    Cluster-level Bias and Rescue Pattern Analysis - PersonSet version
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
    
    # Prepare rescue/bias tables
    try:
        category_cols = getattr(case, "category_cols", None)
        if not category_cols:
            print("Error in retrieving category columns - defaulting back to ['stereotype_type']")
            category_cols = ["stereotype_type"]

        rescue_stats_list, bias_patterns_list = [], []
        from analysis_tools import guarded_labelspace_analysis

        for cat_col in category_cols:
            if cat_col in merged_df.columns:
                rs = guarded_labelspace_analysis(
                    rescue_stats_by_category,
                    merged_df,
                    case=case,
                    category_col=cat_col,
                    person_set=person_set
                )
                rs["category_col"] = cat_col
                rescue_stats_list.append(rs)

                bp = guarded_labelspace_analysis(
                    detect_systematic_biases,
                    merged_df,
                    case=case,
                    person_set=person_set,
                    category_col=cat_col
                )
                bp["category_col"] = cat_col
                bias_patterns_list.append(bp)

        if not rescue_stats_list:
            raise ValueError("No valid category columns found in merged_df.")

        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True)

    except NameError:
        print("WARNING: rescue_stats_by_category or detect_systematic_biases not found")
        print("=== [ Creating random statistics to avoid error ] ===")
        profile_cols = [col for col in merged_df.columns if col.startswith("profile")]

        random_rescue_data = []
        for profile in profile_cols:
            if profile in merged_df.columns:
                accuracy = (merged_df[profile] == merged_df['true_label']).mean()
                random_rescue_data.append({
                    'profile': profile,
                    'rescue_rate': max(0, accuracy - 0.71 + np.random.normal(0, 0.02)),
                    'extra_err_rate': max(0.01, 0.05 - (accuracy - 0.71) * 2 + np.random.normal(0, 0.01)),
                    'rescued': int(np.random.uniform(5, 50)),
                    'extra_errors': int(np.random.uniform(2, 20))
                })
        rescue_stats = pd.DataFrame(random_rescue_data)

        random_bias_data = []
        for profile in profile_cols:
            random_bias_data.append({
                'profile': profile,
                'bias_magnitude': np.random.uniform(0.02, 0.15),
                'mislabelling_rate': np.random.uniform(0.01, 0.08),
                'bias_direction': np.random.choice(case.valid_labels)
            })
        bias_patterns = pd.DataFrame(random_bias_data)
    
    cluster_analysis = {}
    baseline_accuracy = accuracy_score(merged_df["true_label"], merged_df["base_pred"])

    # Optional perf maps
    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        core = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in core.columns:
            tokens_map = core["tokens_per_sample"].to_dict()
        if "cost_per_sample" in core.columns:
            cost_map = core["cost_per_sample"].to_dict()

    # Iterate clusters
    for cluster_id, cluster_info in similarity_results['clusters'].items():
        cluster_profiles = cluster_info['profiles']
        available_profiles = [p for p in cluster_profiles if p in merged_df.columns]

        print(f"\n{cluster_id.upper()} ({len(available_profiles)} profiles):")

        # Trait composition
        trait_composition = defaultdict(list)
        for profile in available_profiles:
            traits = person_set.get_traits(profile, group_keys)
            for trait_name, trait_value in traits.items():
                if trait_value != "Unknown":
                    trait_composition[trait_name].append(str(trait_value).lower())
        
        print("Trait composition:")
        for trait_name, values in trait_composition.items():
            value_counts = Counter(values)
            print(f"  {trait_name}: {dict(value_counts)}")
        
        print(f"Sample profiles: {', '.join(p for p in available_profiles[:5])}")
        if len(available_profiles) > 5:
            print(f"                 ... and {len(available_profiles) - 5} more")

        # Accuracy first
        accuracies = [(merged_df[p] == merged_df['true_label']).mean() for p in available_profiles]
        acc_mean = float(np.mean(accuracies)) if accuracies else 0.0
        acc_std  = float(np.std(accuracies)) if accuracies else 0.0

        # Init metrics
        cluster_metrics = {
            'profiles': available_profiles,
            'size': len(available_profiles),
            'internal_agreement': cluster_info.get('internal_agreement', 0.0),
            'trait_composition': dict(trait_composition),
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
        }

        # Rescue/extra-error
        cluster_rescue = rescue_stats[rescue_stats['profile'].isin(available_profiles)]
        if not cluster_rescue.empty:
            cluster_metrics.update({
                'rescue_rate_mean': float(cluster_rescue['rescue_rate'].mean()),
                'rescue_rate_std': float(cluster_rescue['rescue_rate'].std()),
                'extra_error_rate_mean': float(cluster_rescue['extra_err_rate'].mean()),
                'extra_error_rate_std': float(cluster_rescue['extra_err_rate'].std()),
                'total_rescued': int(cluster_rescue['rescued'].sum()),
                'total_extra_errors': int(cluster_rescue['extra_errors'].sum())
            })
        else:
            cluster_metrics.update({
                'rescue_rate_mean': 0.0, 'rescue_rate_std': 0.0,
                'extra_error_rate_mean': 0.0, 'extra_error_rate_std': 0.0,
                'total_rescued': 0, 'total_extra_errors': 0
            })

        # Bias aggregates
        cluster_bias = bias_patterns[bias_patterns['profile'].isin(available_profiles)]
        if not cluster_bias.empty:
            cluster_metrics.update({
                'bias_magnitude_mean': float(cluster_bias['bias_magnitude'].mean()),
                'bias_magnitude_std': float(cluster_bias['bias_magnitude'].std()),
                'mislabelling_rate_mean': float(cluster_bias['mislabelling_rate'].mean()),
                'dominant_bias_direction': (
                    cluster_bias['bias_direction'].mode().iloc[0]
                    if not cluster_bias['bias_direction'].empty else 'none'
                )
            })
        else:
            cluster_metrics.update({
                'bias_magnitude_mean': 0.0, 'bias_magnitude_std': 0.0,
                'mislabelling_rate_mean': 0.0, 'dominant_bias_direction': 'none'
            })

        # Global accuracy context / consensus
        if archetype_parameters.get("accuracy_consensus", True):
            profile_cols = [c for c in merged_df.columns if c.startswith("profile")]
            per_profile_acc = []
            for p in profile_cols:
                try:
                    per_profile_acc.append(float((merged_df[p] == merged_df["true_label"]).mean()))
                except Exception as e:
                    print(e)
                    continue

            if len(per_profile_acc) >= 3:
                global_acc_mean = float(np.mean(per_profile_acc))
                global_acc_std  = float(np.std(per_profile_acc))
                global_acc_percentile = float(np.percentile(per_profile_acc, 80.0))

                cluster_metrics["global_accuracy_mean"] = global_acc_mean
                cluster_metrics["global_accuracy_std"]  = global_acc_std
                cluster_metrics["global_accuracy_p80"]  = global_acc_percentile
                cluster_metrics["accuracy_gap"] = float(cluster_metrics["accuracy_mean"] - global_acc_mean)

                above_mean_flags = [
                    float((merged_df[p] == merged_df["true_label"]).mean()) > global_acc_mean
                    for p in available_profiles
                ] if available_profiles else []
                cluster_metrics["prop_above_global_mean"] = (
                    float(np.mean(above_mean_flags)) if above_mean_flags else 0.0
                )
                condition_acc = global_acc_percentile
            else:
                condition_acc = float(archetype_parameters["accuracy"])
                cluster_metrics["global_accuracy_mean"] = None
                cluster_metrics["global_accuracy_std"]  = None
                cluster_metrics["global_accuracy_p80"]  = None
                cluster_metrics["accuracy_gap"] = None
                cluster_metrics["prop_above_global_mean"] = None
        else:
            condition_acc = float(archetype_parameters["accuracy"])
            cluster_metrics["global_accuracy_mean"] = None
            cluster_metrics["global_accuracy_std"]  = None
            cluster_metrics["global_accuracy_p80"]  = None
            cluster_metrics["accuracy_gap"] = float(cluster_metrics["accuracy_mean"] - condition_acc)
            cluster_metrics["prop_above_global_mean"] = None

        # Token/cost aggregates (optional)
        if tokens_map or cost_map:
            if tokens_map:
                tok_vals = [tokens_map.get(p, np.nan) for p in available_profiles]
                if np.isfinite(tok_vals).any():
                    cluster_metrics["tokens_per_sample_sum"]  = float(np.nansum(tok_vals))
                    cluster_metrics["tokens_per_sample_mean"] = float(np.nanmean(tok_vals))
            if cost_map:
                cost_vals = [cost_map.get(p, np.nan) for p in available_profiles]
                if np.isfinite(cost_vals).any():
                    cluster_metrics["cost_per_sample_sum"]  = float(np.nansum(cost_vals))
                    cluster_metrics["cost_per_sample_mean"] = float(np.nanmean(cost_vals))
            if "tokens_per_sample_sum" in cluster_metrics and cluster_metrics["tokens_per_sample_sum"] > 0:
                cluster_metrics["improvement_per_1k_tokens"] = (
                    (cluster_metrics["accuracy_mean"] - baseline_accuracy) /
                    (cluster_metrics["tokens_per_sample_sum"] / 1000.0)
                )

        # Archetype tagging
        if (cluster_metrics['rescue_rate_mean'] > archetype_parameters["risk_benefit"][0] and
            cluster_metrics['extra_error_rate_mean'] < archetype_parameters["risk_benefit"][1]):
            archetype = "Optimal Risk-Benefit"
        elif cluster_metrics['extra_error_rate_mean'] < archetype_parameters["extra_error"]:
            archetype = "Cautious"
        elif cluster_metrics['rescue_rate_mean'] > archetype_parameters["rescue"]:
            archetype = "High Error-Correction"
        elif cluster_metrics['accuracy_mean'] > condition_acc:
            archetype = "High Performer"
        elif cluster_metrics['internal_agreement'] > archetype_parameters["internal_agreement"]:
            archetype = "High Consistency"
        else:
            archetype = "Similar to Neutral"
        cluster_metrics['archetype'] = archetype

        # Print summary
        print(f"  Accuracy: {cluster_metrics['accuracy_mean']:.4f} ± {cluster_metrics['accuracy_std']:.4f}")
        print(f"  Rescue Rate: {cluster_metrics['rescue_rate_mean']:.3f} ± {cluster_metrics['rescue_rate_std']:.3f}")
        print(f"  Extra Error Rate: {cluster_metrics['extra_error_rate_mean']:.3f} ± {cluster_metrics['extra_error_rate_std']:.3f}")
        print(f"  Bias Magnitude: {cluster_metrics['bias_magnitude_mean']:.3f} (direction: {cluster_metrics['dominant_bias_direction']})")
        print(f"  Internal Agreement: {cluster_metrics['internal_agreement']:.3f}")
        print(f"  Archetype: {archetype}")

        cluster_analysis[cluster_id] = cluster_metrics
    
    # Recommendations
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





def trait_comparison_controlled(
    merged_df: pd.DataFrame, 
    person_set: PersonSet,
    case: CaseConfig,
    comparison_trait: str = "cognitive_style",
    control_traits: List[str] = None,
    perf_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Generalized Trait Comparison with Demographic Controls
    """
    if control_traits is None:
        control_traits = []
    
    print("=" * 80)
    print(f"TRAIT COMPARISON: {comparison_trait.upper()}")
    if control_traits:
        print(f"CONTROLLING FOR: {', '.join(control_traits).upper()}")
    print("=" * 80)
    
    # Get profile columns
    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
    
    trait_groups = defaultdict(list)
    trait_metadata = {} 
    
    for profile in profile_cols:
        traits = person_set.get_traits(profile, [comparison_trait] + control_traits)
        trait_value = str(traits.get(comparison_trait, "Unknown"))
        if trait_value != "Unknown":
            trait_groups[trait_value].append(profile)
            trait_metadata[profile] = traits
    
    trait_groups = {k: v for k, v in trait_groups.items() if v}
    if len(trait_groups) < 2:
        print(f"ERROR: Insufficient groups for {comparison_trait} comparison")
        return {'error': f'Insufficient groups for {comparison_trait}'}
    
    print(f"Found {len(trait_groups)} groups for {comparison_trait}:")
    for trait_val, profiles in trait_groups.items():
        print(f"  {trait_val}: {len(profiles)} profiles")

    # Optional perf maps
    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        core = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in core.columns:
            tokens_map = core["tokens_per_sample"].to_dict()
        if "cost_per_sample" in core.columns:
            cost_map = core["cost_per_sample"].to_dict()

    # Rescue/bias tables
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
                    person_set=person_set
                )
                rs["category_col"] = cat_col
                rescue_stats_list.append(rs)
                bp = guarded_labelspace_analysis(
                    detect_systematic_biases,
                    merged_df,
                    case=case,
                    category_col=cat_col,
                    person_set=person_set
                )
                bp["category_col"] = cat_col
                bias_patterns_list.append(bp)
        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True) if rescue_stats_list else pd.DataFrame()
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True) if bias_patterns_list else pd.DataFrame()
        if rescue_stats.empty:
            raise ValueError("No valid category columns found for rescue_stats.")
        if bias_patterns.empty:
            raise ValueError("No valid category columns found for bias_patterns.")
    except NameError:
        print("WARNING: rescue_stats_by_category or detect_systematic_biases not found")
        rescue_stats = pd.DataFrame()
        bias_patterns = pd.DataFrame()
    
    # Calculate performance metrics for each trait group
    trait_performance = {}
    
    for trait_value, profiles in trait_groups.items():
        available_profiles = [p for p in profiles if p in merged_df.columns]
        if not available_profiles:
            continue
            
        accuracies, rescue_rates, extra_error_rates, bias_magnitudes = [], [], [], []
        
        for profile in available_profiles:
            acc = accuracy_score(merged_df['true_label'], merged_df[profile])
            accuracies.append(acc)

            rs = rescue_stats[rescue_stats['profile'] == profile] if not rescue_stats.empty else pd.DataFrame()
            rescue_rates.append(rs['rescue_rate'].mean() if len(rs) > 0 else 0.0)
            extra_error_rates.append(rs['extra_err_rate'].mean() if len(rs) > 0 else 0.0)
            
            bp = bias_patterns[bias_patterns['profile'] == profile] if not bias_patterns.empty else pd.DataFrame()
            bias_magnitudes.append(bp['bias_magnitude'].mean() if len(bp) > 0 else 0.0)
        
        trait_performance[trait_value] = {
            'accuracies': accuracies,
            'rescue_rates': rescue_rates,
            'extra_error_rates': extra_error_rates,
            'bias_magnitudes': bias_magnitudes,
            'n_profiles': len(available_profiles),
            'profiles': available_profiles,
            'control_trait_distribution': {}
        }

        # Token/cost summaries (optional)
        if tokens_map:
            tok_vals = [tokens_map.get(p, np.nan) for p in available_profiles]
            trait_performance[trait_value]["tokens_per_sample_mean"] = float(np.nanmean(tok_vals)) if tok_vals else np.nan
        if cost_map:
            cost_vals = [cost_map.get(p, np.nan) for p in available_profiles]
            trait_performance[trait_value]["cost_per_sample_mean"] = float(np.nanmean(cost_vals)) if cost_vals else np.nan
        
        # Control trait distribution
        if control_traits:
            for control_trait in control_traits:
                control_values = []
                for profile in available_profiles:
                    if profile in trait_metadata:
                        control_values.append(trait_metadata[profile].get(control_trait, "Unknown"))
                control_counts = Counter(control_values)
                trait_performance[trait_value]['control_trait_distribution'][control_trait] = dict(control_counts)
        
        # Print results
        print(f"\n{str(trait_value).upper()} {comparison_trait.upper()} (n={len(available_profiles)}):")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Rescue Rate: {np.mean(rescue_rates):.3f} ± {np.std(rescue_rates):.3f}")
        print(f"  Extra Error Rate: {np.mean(extra_error_rates):.3f} ± {np.std(extra_error_rates):.3f}")
        print(f"  Bias Magnitude: {np.mean(bias_magnitudes):.3f} ± {np.std(bias_magnitudes):.3f}")
        if control_traits:
            print("  Control trait distributions:")
            for control_trait, distribution in trait_performance[trait_value]['control_trait_distribution'].items():
                print(f"    {control_trait}: {distribution}")
    
    # Statistical comparisons
    statistical_results = {}
    metrics = ['accuracies', 'rescue_rates', 'extra_error_rates', 'bias_magnitudes']
    metric_names = ['Accuracy', 'Rescue Rate', 'Extra Error Rate', 'Bias Magnitude']
    
    print(f"\n{'='*60}")
    print(f"STATISTICAL COMPARISONS: {comparison_trait.upper()}")
    print(f"{'='*60}")
    
    for metric, metric_name in zip(metrics, metric_names):
        groups, group_names = [], []
        for trait_value, data in trait_performance.items():
            if metric in data and len(data[metric]) > 0:
                groups.append(data[metric])
                group_names.append(trait_value)
        
        if len(groups) >= 2:
            combined = np.concatenate(groups)
            if np.all(combined == combined[0]):
                print(f"  {metric_name}: No variation detected")
                continue
            
            if len(groups) == 2:
                u_stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                effect_size = 1 - (2 * u_stat) / (len(groups[0]) * len(groups[1]))
                statistical_results[metric] = {
                    'test_type': 'mann_whitney',
                    'statistic': u_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05,
                    'group_means': {name: np.mean(data) for name, data in zip(group_names, groups)}
                }
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {metric_name}: U={u_stat:.1f}, p={p_value:.4f} {sig_marker} (effect={effect_size:.3f})")
            else:
                h_stat, p_value = kruskal(*groups)
                statistical_results[metric] = {
                    'test_type': 'kruskal_wallis',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'group_means': {name: np.mean(data) for name, data in zip(group_names, groups)},
                    'pairwise_comparisons': {}
                }
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {metric_name}: H={h_stat:.3f}, p={p_value:.4f} {sig_marker}")
                if p_value < 0.05:
                    print("    Pairwise comparisons:")
                    for i, (name1, g1) in enumerate(zip(group_names, groups)):
                        for j, (name2, g2) in enumerate(zip(group_names, groups)):
                            if i < j:
                                u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
                                effect_size = 1 - (2 * u_stat) / (len(g1) * len(g2))
                                statistical_results[metric]['pairwise_comparisons'][f"{name1}_vs_{name2}"] = {
                                    'u_stat': u_stat,
                                    'p_value': p_val,
                                    'effect_size': effect_size,
                                    'significant': p_val < 0.05,
                                    'mean_diff': np.mean(g1) - np.mean(g2)
                                }
                                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                                print(f"      {name1} vs {name2}: p={p_val:.4f} {sig} (effect={effect_size:.3f})")
    
    # Rankings and recommendations
    print(f"\n{'='*60}")
    print(f"{comparison_trait.upper()} RANKINGS")
    print(f"{'='*60}")
    
    rankings = {}
    for metric, metric_name in zip(metrics, metric_names):
        if metric in statistical_results:
            means = statistical_results[metric]['group_means']
            reverse = metric in ['accuracies', 'rescue_rates']  # Higher is better
            sorted_traits = sorted(means.items(), key=lambda x: x[1], reverse=reverse)
            rankings[metric_name] = sorted_traits
            print(f"\n{metric_name} Rankings:")
            for i, (trait_val, value) in enumerate(sorted_traits, 1):
                print(f"  {i}. {trait_val}: {value:.4f}")
    
    weights = {'Accuracy': 0.3, 'Rescue Rate': 0.25, 'Extra Error Rate': -0.25, 'Bias Magnitude': -0.2}
    composite_scores = {}
    for trait_value in trait_groups.keys():
        score = 0.0
        count = 0.0
        for metric_name, weight in weights.items():
            if metric_name in rankings and len(rankings[metric_name]) > 0:
                rank = next((i for i, (s, _) in enumerate(rankings[metric_name], 1) if s == trait_value), len(rankings[metric_name]) + 1)
                max_rank = len(rankings[metric_name])
                rank_score = (max_rank + 1 - rank) / max_rank  # Normalize [0,1]
                score += weight * rank_score
                count += abs(weight)
        composite_scores[trait_value] = score / count if count > 0 else 0.0
    
    recommended_traits = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"{comparison_trait.upper()} RECOMMENDATIONS")
    print(f"{'='*60}")
    print("\nComposite Ranking:")
    for i, (trait_val, score) in enumerate(recommended_traits, 1):
        print(f"  {i}. {trait_val}: {score:.3f}")
    
    return {
        'comparison_trait': comparison_trait,
        'control_traits': control_traits,
        'trait_performance': trait_performance,
        'statistical_results': statistical_results,
        'rankings': rankings,
        'composite_scores': composite_scores,
        'recommendations': recommended_traits,
        'trait_groups': dict(trait_groups)
    }



def run_multiple_trait_comparisons(
    merged_df: pd.DataFrame, 
    person_set: PersonSet,
    trait_analyses: List[Tuple[str, List[str]]] = None,
    perf_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run multiple trait comparisons efficiently.
    
    Parameters:
    - trait_analyses: List of (comparison_trait, control_traits) tuples
    """
    
    if not trait_analyses:
        print("No trait analyses specified. Skipping.")
        return {'skipped': True, 'reason': 'No traits provided'}
    
    results = {}
    
    print("=" * 80)
    print("MULTIPLE TRAIT COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"Running {len(trait_analyses)} trait comparisons...")
    
    for comparison_trait, control_traits in trait_analyses:
        print(f"\n{'='*40}")
        print(f"ANALYZING: {comparison_trait}")
        print(f"{'='*40}")
        
        try:
            result = trait_comparison_controlled(
                merged_df, person_set,
                comparison_trait=comparison_trait,
                control_traits=control_traits
            )
            results[comparison_trait] = result
            print(f"SUCCESS: {comparison_trait} analysis completed")
            
        except Exception as e:
            print(f"ERROR: {comparison_trait} analysis failed: {e}")
            results[comparison_trait] = {'error': str(e)}
    
    # Summary across all traits
    print(f"\n{'='*80}")
    print("CROSS-TRAIT SUMMARY")
    print(f"{'='*80}")
    
    successful_analyses = [k for k, v in results.items() if 'error' not in v]
    print(f"Successful analyses: {len(successful_analyses)}/{len(trait_analyses)}")
    
    for trait in successful_analyses:
        if 'recommendations' in results[trait]:
            best = results[trait]['recommendations'][0]
            print(f"  Best {trait}: {best[0]} (score: {best[1]:.3f})")
    
    return results


def plot_ensemble_performance_comparison(
    ensemble_results: Dict[str, Any], 
    figsize: tuple = (12, 8),
    key_ensembles: Optional[List[str]] = None,
    show_top_n: int = None
):
    """
    Plot ensemble performance comparison with accuracy, improvement, and risk–benefit.
    Pass in the full dict returned by ensemble_by_trait_analysis.
    """
    ensemble_data = ensemble_results['ensemble_results']
    
    # Choose ensembles
    if key_ensembles is None:
        key_ensembles = list(ensemble_data.keys())
    if show_top_n:
        sorted_ensembles = sorted(
            [(name, data) for name, data in ensemble_data.items() if name in key_ensembles],
            key=lambda x: x[1].get('accuracy', 0.0),
            reverse=True
        )
        key_ensembles = [name for name, _ in sorted_ensembles[:show_top_n]]
    
    # Collect series
    ensemble_names, accuracies, improvements, rescue_rates, extra_error_rates, sizes = [], [], [], [], [], []
    for name in key_ensembles:
        data = ensemble_data.get(name, {})
        ensemble_names.append(name.replace('_', '\n').title())
        accuracies.append(float(data.get('accuracy', 0.0)))
        improvements.append(float(data.get('improvement', 0.0)))
        rescue_rates.append(float(data.get('rescue_rate', 0.0)))
        extra_error_rates.append(float(data.get('extra_error_rate', 0.0)))
        tok = data.get("tokens_per_sample_sum")
        if isinstance(tok, (int, float)) and np.isfinite(tok) and tok > 0:
            sizes.append(max(80.0, tok * 0.5))
        else:
            sizes.append(100.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Ensemble Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy Improvement
    x_pos = np.arange(len(ensemble_names))
    colors = ['#2ca02c' if imp > 0.01 else '#1f77b4' if imp >= 0 else '#d62728' for imp in improvements]
    bars1 = ax1.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, acc, imp in zip(bars1, accuracies, improvements):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., 
                 h + 0.001 if h >= 0 else h - 0.002,
                 f'{acc:.3f}',
                 ha='center', va='bottom' if h >= 0 else 'top',
                 fontsize=9, fontweight='bold')
    ax1.set_xlabel('Ensemble Strategy', fontsize=12)
    ax1.set_ylabel('Accuracy Improvement vs Baseline', fontsize=12)
    ax1.set_title('Accuracy Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ensemble_names, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Risk–Benefit
    scatter = ax2.scatter(
        extra_error_rates, rescue_rates,
        c=improvements, s=sizes, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1
    )
    ax2.set_title('Risk–Benefit (size ∝ tokens/sample)', fontsize=14, fontweight='bold')
    for i, name in enumerate(ensemble_names):
        ax2.annotate(name.replace('\n', ' '), 
                     (extra_error_rates[i], rescue_rates[i]),
                     xytext=(8, 8), textcoords='offset points', 
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax2.set_xlabel('Extra Error Rate (Risk)', fontsize=12)
    ax2.set_ylabel('Rescue Rate (Benefit)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if extra_error_rates and rescue_rates:
        x_margin = (max(extra_error_rates) - min(extra_error_rates)) * 0.15 if len(set(extra_error_rates)) > 1 else 0.05
        y_margin = (max(rescue_rates) - min(rescue_rates)) * 0.15 if len(set(rescue_rates)) > 1 else 0.05
        ax2.set_xlim(min(extra_error_rates) - x_margin, max(extra_error_rates) + x_margin)
        ax2.set_ylim(min(rescue_rates) - y_margin, max(rescue_rates) + y_margin)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Accuracy Improvement', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return fig



def plot_cluster_analysis(
    cluster_results: Dict[str, Any],
    figsize: tuple = (14, 6)
):
    """
    Plot cluster analysis with demographic composition and performance metrics.
    """
    
    cluster_data = cluster_results['cluster_analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Cluster-Level Bias and Performance Analysis', fontsize=16, fontweight='bold')
    
    # Professional archetype color mapping
    archetype_colors = {
        'Optimal Risk-Benefit': '#2ca02c',
        'Cautious': '#1f77b4', 
        'High Error-Correction': '#ff7f0e',
        'High Performer': '#d62728',
        'High Consistency': '#9467bd',
        'Similar to Neutral': '#8c564b'
    }
    
    # Plot 1: Cluster Performance Metrics
    cluster_names = []
    accuracies = []
    rescue_rates = []
    extra_error_rates = []
    colors = []
    
    for cluster_id, cluster_info in cluster_data.items():
        cluster_names.append(cluster_id.replace('cluster_', 'Cluster '))
        accuracies.append(cluster_info['accuracy_mean'])
        rescue_rates.append(cluster_info['rescue_rate_mean'])
        extra_error_rates.append(cluster_info['extra_error_rate_mean'])
        
        archetype = cluster_info.get('archetype', 'Unknown')
        colors.append(archetype_colors.get(archetype, '#bcbd22'))
    
    x_pos = np.arange(len(cluster_names))
    width = 0.25
    
    bars1 = ax1.bar(x_pos - width, accuracies, width, label='Accuracy', 
                   color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x_pos, rescue_rates, width, label='Rescue Rate', 
                   color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars3 = ax1.bar(x_pos + width, extra_error_rates, width, label='Extra Error Rate', 
                   color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Performance Metrics', fontsize=12)
    ax1.set_title('Cluster Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cluster_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk-Benefit with Archetype Colors
    scatter = ax2.scatter(extra_error_rates, rescue_rates, 
                         c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add cluster labels and archetype info
    for i, (cluster_name, cluster_info) in enumerate(zip(cluster_names, cluster_data.values())):
        archetype = cluster_info.get('archetype', 'Unknown')
        ax2.annotate(f"{cluster_name}\n({archetype})", 
                    (extra_error_rates[i], rescue_rates[i]),
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='black', alpha=0.8))
    
    ax2.set_xlabel('Extra Error Rate (Risk)', fontsize=12)
    ax2.set_ylabel('Rescue Rate (Benefit)', fontsize=12)
    ax2.set_title('Cluster Risk-Benefit Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add padding to prevent label cutoff in cluster plot
    if extra_error_rates and rescue_rates:
        x_margin = (max(extra_error_rates) - min(extra_error_rates)) * 0.2
        y_margin = (max(rescue_rates) - min(rescue_rates)) * 0.2
        ax2.set_xlim(min(extra_error_rates) - x_margin, max(extra_error_rates) + x_margin)
        ax2.set_ylim(min(rescue_rates) - y_margin, max(rescue_rates) + y_margin)
    
    # Add archetype legend
    legend_elements = []
    for archetype, color in archetype_colors.items():
        if any(cluster_info.get('archetype') == archetype for cluster_info in cluster_data.values()):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10, 
                                           label=archetype, markeredgecolor='black'))
    
    if legend_elements:
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_trait_comparison_results(
    trait_results: Dict[str, Any],
    trait_name: str = None,
    baseline_accuracy: float = None,
    figsize: tuple = (14, 8)
):
    """
    Plot results from trait_comparison_controlled function.
    
    Parameters:
    - trait_results: Results from trait_comparison_controlled
    - trait_name: Name of the trait being analyzed (auto-detected if None)
    - baseline_accuracy: Baseline to center improvements around (auto-detected if None)
    - figsize: Figure size
    """
    
    if trait_name is None:
        trait_name = trait_results.get('comparison_trait', 'Trait')
    
    # Handle both old cognitive style format and new trait comparison format
    trait_performance = trait_results.get('trait_performance', trait_results.get('style_performance', {}))
    
    if not trait_performance:
        print(f"No trait performance data found for {trait_name}")
        return None
    
    # Auto-detect baseline if not provided
    if baseline_accuracy is None:
        # Use the mean accuracy across all trait levels as baseline
        all_accuracies = []
        for trait_val, data in trait_performance.items():
            if 'accuracies' in data:
                all_accuracies.extend(data['accuracies'])
        baseline_accuracy = np.mean(all_accuracies) if all_accuracies else 0.70
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{trait_name.title()} Comparison Analysis (Relative to Baseline)', fontsize=16, fontweight='bold')
    
    trait_values = list(trait_performance.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(trait_values)))
    
    metrics = [
        ('accuracies', 'Accuracy Improvement', axes[0, 0], baseline_accuracy),
        ('rescue_rates', 'Rescue Rate', axes[0, 1], 0),  # Rescue rate is already relative
        ('extra_error_rates', 'Extra Error Rate', axes[1, 0], 0),  # Error rate is already relative
        ('bias_magnitudes', 'Bias Magnitude', axes[1, 1], 0)  # Bias magnitude is already relative
    ]
    
    for metric_key, metric_name, ax, baseline in metrics:
        means = []
        stds = []
        
        for trait_val in trait_values:
            data = trait_performance[trait_val].get(metric_key, [])
            if data:
                # Calculate improvement relative to baseline for accuracy
                if metric_key == 'accuracies' and baseline > 0:
                    improvements = [acc - baseline for acc in data]
                    means.append(np.mean(improvements))
                    stds.append(np.std(improvements))
                else:
                    means.append(np.mean(data))
                    stds.append(np.std(data))
            else:
                means.append(0)
                stds.append(0)
        
        x_pos = np.arange(len(trait_values))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            # Adjust label position based on bar height to prevent cutoff
            if height >= 0:
                label_y = height + std + 0.001
                va = 'bottom'
            else:
                label_y = height - std - 0.001
                va = 'top'
            
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.3f}±{std:.3f}', ha='center', va=va, 
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel(f'{trait_name.title()} Level', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        
        # Different titles for relative vs absolute metrics
        if metric_key == 'accuracies':
            ax.set_title(f'{metric_name} vs Baseline ({baseline:.3f})', fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{metric_name} by {trait_name.title()}', fontsize=14, fontweight='bold')
            
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(tv).title() for tv in trait_values], rotation=45, ha='right')
        
        # Add baseline reference line for accuracy
        if metric_key == 'accuracies':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')
            ax.legend()
        
        # Add y-axis padding to prevent label cutoff
        y_data = [m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)]
        if y_data:
            y_range = max(y_data) - min(y_data)
            y_margin = y_range * 0.15
            ax.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_system_level_recommendations(
    ensemble_results: Dict[str, Any],
    cluster_results: Dict[str, Any],
    trait_results: Dict[str, Any] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot system-level recommendations across analyses.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('System-Level AI Safety Recommendations', fontsize=16, fontweight='bold')
    
    recommendations, scores, colors = [], [], []
    ensemble_data = ensemble_results.get('ensemble_results', {})
    lambda_tok = 0.0005  # token penalty weight

    # Best Ensemble (with token penalty applied only to this bar)
    best_ens_name = None
    best_ens_score = 0.0
    if 'recommendations' in ensemble_results:
        best_balanced = ensemble_results['recommendations'].get('best_balanced')
        if isinstance(best_balanced, tuple):
            best_ens_name = best_balanced[0]
            best_ens_score = float(best_balanced[1])
        elif best_balanced:
            best_ens_name = best_balanced[0]
            best_ens_score = float(ensemble_data.get(best_ens_name, {}).get('improvement', 0.0))
        if best_ens_name:
            tok = ensemble_data.get(best_ens_name, {}).get("tokens_per_sample_sum")
            if isinstance(tok, (int, float)) and np.isfinite(tok) and tok > 0:
                best_ens_score = best_ens_score - lambda_tok * tok
            recommendations.append(f"Best Ensemble\n{best_ens_name.replace('_',' ').title()}")
            scores.append(best_ens_score)
            colors.append('#2ca02c')

    # Best Cluster (score relative to baseline)
    if 'recommendations' in cluster_results:
        best_cluster = cluster_results['recommendations'].get('best_accuracy')
        if isinstance(best_cluster, tuple):
            cluster_name, cluster_item = best_cluster[0], best_cluster[1]
            base = ensemble_results.get('baseline_accuracy', 0.70)
            cluster_score = float(cluster_item.get('accuracy_mean', 0.0)) - float(base)
            recommendations.append(f"Best Cluster\n{cluster_name.replace('_',' ').title()}")
            scores.append(cluster_score)
            colors.append('#1f77b4')

    # Best trait (optional)
    if trait_results and 'recommendations' in trait_results and trait_results['recommendations']:
        best_trait = trait_results['recommendations'][0]
        trait_name = trait_results.get('comparison_trait', 'Trait')
        trait_label = f"Best {trait_name.title()}\n{str(best_trait[0]).title()}"
        # normalize composite score for plotting
        trait_score = float(best_trait[1]) / 10.0
        recommendations.append(trait_label)
        scores.append(trait_score)
        colors.append('#ff7f0e')
    
    # Safest ensemble (lowest extra error)
    if 'recommendations' in ensemble_results:
        safest = ensemble_results['recommendations'].get('safest')
        safest_name = safest[0] if safest else None
        if safest_name:
            err_rate = float(ensemble_data.get(safest_name, {}).get('extra_error_rate', 0.0))
            safety_score = -err_rate
            recommendations.append(f"Safest Ensemble\n{safest_name.replace('_',' ').title()}")
            scores.append(safety_score)
            colors.append('#d62728')

    # Plot
    y_pos = np.arange(len(recommendations))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, score in zip(bars, scores):
        w = bar.get_width()
        ax.text(w + (0.001 if w >= 0 else -0.001),
                bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}',
                ha='left' if w >= 0 else 'right',
                va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(recommendations)
    ax.set_xlabel('Performance Score', fontsize=12)
    ax.set_title('Deployment Recommendations by Analysis Type', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    return fig



def create_all_tier2_visualizations(
    ensemble_results: Dict[str, Any],
    cluster_results: Dict[str, Any],
    trait_results: Dict[str, Any] = None,
    baseline_accuracy: float = None,
    show_top_ensembles: int = 8,
    save_paths: Optional[Dict[str, Optional[str]]] = None
):
    """
    Create all Tier 2 visualizations with proper spacing and formatting.
    
    Parameters:
    - ensemble_results: Results from ensemble_by_trait_analysis
    - cluster_results: Results from cluster_level_bias_patterns  
    - trait_results: Results from trait_comparison_controlled (optional)
    - show_top_ensembles: Number of top ensembles to show
    """
    
    print("Creating Tier 2 Visualizations...")
    print("="*50)
    
    figures = {}
    
    try:
        print("1. Ensemble Performance Analysis...")
        fig1 = plot_ensemble_performance_comparison(
            ensemble_results, 
            show_top_n=show_top_ensembles
        )
        figures['ensemble'] = fig1
        if save_paths and save_paths.get('ensemble'):
            os.makedirs(os.path.dirname(save_paths['ensemble']), exist_ok=True)
            fig1.savefig(save_paths['ensemble'])
        
    except Exception as e:
        print(f"   Error creating ensemble plot: {e}")
    
    try:
        print("2. Cluster Analysis...")
        fig2 = plot_cluster_analysis(cluster_results)
        figures['cluster'] = fig2
        if save_paths and save_paths.get('cluster'):
            os.makedirs(os.path.dirname(save_paths['cluster']), exist_ok=True)
            fig2.savefig(save_paths['cluster'])

        
    except Exception as e:
        print(f"   Error creating cluster plot: {e}")
    
    if trait_results:
        try:
            print("3. Trait Comparison Analysis...")
            fig3 = plot_trait_comparison_results(
                trait_results, 
                baseline_accuracy=baseline_accuracy
            )
            figures['trait'] = fig3
            if save_paths and save_paths.get('trait'):
                os.makedirs(os.path.dirname(save_paths['trait']), exist_ok=True)
                fig3.savefig(save_paths['trait'])
            
        except Exception as e:
            print(f"   Error creating trait plot: {e}")
    
    try:
        print("4. System-Level Recommendations...")
        fig4 = plot_system_level_recommendations(
            ensemble_results, cluster_results, trait_results
        )
        figures['recommendations'] = fig4
        if save_paths and save_paths.get('recommendations'):
            os.makedirs(os.path.dirname(save_paths['recommendations']), exist_ok=True)
            fig4.savefig(save_paths['recommendations'])
        
    except Exception as e:
        print(f"   Error creating recommendations plot: {e}")
    
    print(f"\nCompleted! Created {len(figures)} visualizations.")
    return figures

def run_permutation_tests(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    n_permutations: int = 1000,
    traits: List[str] = ["gender", "ethnicity"],
    random_seed: int = 42,
    baseline_col: str = "base_pred"
) -> Dict[str, Any]:
    """
    Run permutation tests comparing demographic-conditioned profiles against baseline.
    
    This tests the null hypothesis that demographic conditioning introduces no
    systematic bias beyond what would be expected by random chance. The test
    compares the variance of demographic group accuracies against baseline
    with the distribution of variances under random profile assignments.
    
    Key insight: We're not testing if demographic groups differ from each other,
    but rather if demographic conditioning creates patterns that wouldn't arise
    from random variation around the baseline performance.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        DataFrame with profile predictions, baseline predictions, and true labels
    person_set : PersonSet
        Metadata for profiles including demographic information
    n_permutations : int, default=1000
        Number of random permutations to perform
    traits : List[str], default=["gender", "ethnicity"]
        Traits to test for bias
    random_seed : int, default=42
        Random seed for reproducibility
    baseline_col : str, default="base_pred"
        Column containing baseline model predictions
        
    Returns:
    --------
    Dict containing permutation test results
    """
    
    np.random.seed(random_seed)
    
    if baseline_col not in merged_df.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in data")
    
    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
    if not profile_cols:
        raise ValueError("No profile columns found")
    
    # Calculate baseline accuracy
    baseline_accuracy = (merged_df[baseline_col] == merged_df['true_label']).mean()
    
    # Map profiles to demographics
    profile_demographics = {}
    for profile in profile_cols:
        demo_info = get_demographic_info(profile, person_set)
        profile_demographics[profile] = demo_info
    
    results = {
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "baseline_accuracy": baseline_accuracy,
        "baseline_col": baseline_col,
        "trait_tests": {},
        "overall_bias_test": {},
        "summary": {}
    }
    
    # Test each trait for systematic bias relative to baseline
    for trait in traits:
        if trait not in ["gender", "ethnicity"]:
            continue
            
        trait_results = _run_baseline_comparison_test(
            merged_df, profile_demographics, trait, baseline_accuracy, n_permutations
        )
        results["trait_tests"][trait] = trait_results
    
    # Overall test: Do demographic-conditioned profiles show more variance than expected?
    overall_results = _run_overall_bias_test(
        merged_df, profile_demographics, baseline_accuracy, n_permutations
    )
    results["overall_bias_test"] = overall_results
    
    # Summary statistics
    results["summary"] = _summarize_baseline_permutation_results(results)
    
    return results


def _run_baseline_comparison_test(
    merged_df: pd.DataFrame,
    profile_demographics: Dict[str, str],
    trait: str,
    baseline_accuracy: float,
    n_permutations: int
) -> Dict[str, Any]:
    """
    Test if demographic groups show systematic deviations from baseline beyond random chance.
    
    Instead of comparing groups to each other, this compares each demographic group's
    mean accuracy to baseline, then tests if the observed pattern of deviations
    could arise from random assignment of profiles to demographic groups.
    """
    
    profile_cols = list(profile_demographics.keys())
    
    # Group profiles by trait value
    trait_groups = {}
    for profile, demo in profile_demographics.items():
        if trait == "gender":
            trait_value = demo.split("_")[1] if "_" in demo else "unknown"
        elif trait == "ethnicity":
            trait_value = demo.split("_")[0] if "_" in demo else "unknown"
        else:
            continue
            
        if trait_value not in trait_groups:
            trait_groups[trait_value] = []
        trait_groups[trait_value].append(profile)
    
    # Only proceed if we have multiple groups
    valid_groups = {k: v for k, v in trait_groups.items() if len(v) >= 1}
    if len(valid_groups) < 2:
        return {"error": f"Insufficient groups for {trait} baseline comparison test"}
    
    # Calculate observed deviations from baseline for each group
    observed_deviations = {}
    group_accuracies = {}
    
    for group, profiles in valid_groups.items():
        # Mean accuracy for this demographic group
        group_acc = np.mean([(merged_df[p] == merged_df['true_label']).mean() for p in profiles])
        group_accuracies[group] = group_acc
        observed_deviations[group] = group_acc - baseline_accuracy
    
    # Test statistic: Sum of squared deviations from baseline
    observed_test_stat = np.sum([dev**2 for dev in observed_deviations.values()])
    
    # Run permutations: shuffle profile assignments to demographic groups
    permuted_test_stats = []
    
    for perm_i in range(n_permutations):
        # Shuffle profiles while maintaining group sizes
        shuffled_profiles = profile_cols.copy()
        np.random.shuffle(shuffled_profiles)
        
        # Assign shuffled profiles to demographic groups (maintaining group sizes)
        perm_deviations = []
        start_idx = 0
        
        for group, original_profiles in valid_groups.items():
            group_size = len(original_profiles)
            group_profiles = shuffled_profiles[start_idx:start_idx + group_size]
            
            # Calculate mean accuracy for this permuted group
            group_acc = np.mean([(merged_df[p] == merged_df['true_label']).mean() for p in group_profiles])
            deviation = group_acc - baseline_accuracy
            perm_deviations.append(deviation**2)
            
            start_idx += group_size
        
        # Test statistic for this permutation
        perm_test_stat = np.sum(perm_deviations)
        permuted_test_stats.append(perm_test_stat)
    
    # Calculate p-value
    p_value = (np.sum(np.array(permuted_test_stats) >= observed_test_stat) + 1) / (n_permutations + 1)
    
    return {
        "trait": trait,
        "baseline_accuracy": baseline_accuracy,
        "group_accuracies": group_accuracies,
        "observed_deviations": observed_deviations,
        "observed_test_statistic": observed_test_stat,
        "permuted_test_statistics": permuted_test_stats,
        "p_value": p_value,
        "trait_groups": {k: len(v) for k, v in valid_groups.items()},
        "interpretation": f"Tests if {trait} groups deviate from baseline more than expected by chance"
    }


def _run_overall_bias_test(
    merged_df: pd.DataFrame,
    profile_demographics: Dict[str, str],
    baseline_accuracy: float,
    n_permutations: int
) -> Dict[str, Any]:
    """
    Test if demographic-conditioned profiles as a whole show more variance than expected.
    
    This tests whether the overall spread of profile accuracies is larger than
    what we'd expect from random variation around the baseline.
    """
    
    profile_cols = list(profile_demographics.keys())
    
    # Calculate observed profile accuracies
    profile_accuracies = {}
    for profile in profile_cols:
        acc = (merged_df[profile] == merged_df['true_label']).mean()
        profile_accuracies[profile] = acc
    
    # Observed variance of profile accuracies around baseline
    deviations_from_baseline = [acc - baseline_accuracy for acc in profile_accuracies.values()]
    observed_variance = np.var(deviations_from_baseline)
    
    # Permutation test: shuffle profile predictions randomly
    permuted_variances = []
    
    for perm_i in range(n_permutations):
        # For each profile, randomly sample its predictions from all profiles
        # This breaks the demographic conditioning while preserving individual profile variation
        perm_profile_accuracies = []
        
        for profile in profile_cols:
            # Randomly select a different profile's predictions
            random_profile = np.random.choice(profile_cols)
            perm_acc = (merged_df[random_profile] == merged_df['true_label']).mean()
            perm_profile_accuracies.append(perm_acc)
        
        # Calculate variance of permuted accuracies around baseline
        perm_deviations = [acc - baseline_accuracy for acc in perm_profile_accuracies]
        perm_variance = np.var(perm_deviations)
        permuted_variances.append(perm_variance)
    
    # Calculate p-value
    p_value = (np.sum(np.array(permuted_variances) >= observed_variance) + 1) / (n_permutations + 1)
    
    return {
        "baseline_accuracy": baseline_accuracy,
        "profile_accuracies": profile_accuracies,
        "observed_variance": observed_variance,
        "permuted_variances": permuted_variances,
        "p_value": p_value,
        "n_profiles": len(profile_cols),
        "interpretation": "Tests if profile variance around baseline exceeds random expectation"
    }


def _summarize_baseline_permutation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics for baseline-focused permutation test results."""
    
    summary = {
        "significant_trait_tests": 0,
        "total_trait_tests": 0,
        "significant_overall_bias": False,
        "min_p_value": 1.0,
        "baseline_accuracy": results.get("baseline_accuracy", 0.0),
        "significant_findings": []
    }
    
    # Summarize trait tests (comparisons against baseline)
    for trait, trait_results in results["trait_tests"].items():
        if "error" in trait_results:
            continue
            
        summary["total_trait_tests"] += 1
        p_value = trait_results["p_value"]
        summary["min_p_value"] = min(summary["min_p_value"], p_value)
        
        if p_value < 0.05:
            summary["significant_trait_tests"] += 1
            summary["significant_findings"].append({
                "type": f"{trait}_bias_vs_baseline",
                "p_value": p_value,
                "test_statistic": trait_results["observed_test_statistic"],
                "interpretation": f"{trait} groups deviate from baseline more than expected by chance"
            })
    
    # Summarize overall bias test
    if "overall_bias_test" in results and "error" not in results["overall_bias_test"]:
        overall_p = results["overall_bias_test"]["p_value"]
        summary["min_p_value"] = min(summary["min_p_value"], overall_p)
        
        if overall_p < 0.05:
            summary["significant_overall_bias"] = True
            summary["significant_findings"].append({
                "type": "overall_profile_variance",
                "p_value": overall_p,
                "observed_variance": results["overall_bias_test"]["observed_variance"],
                "interpretation": "Profile variance around baseline exceeds random expectation"
            })
    
    return summary


def print_permutation_results(results: Dict[str, Any]) -> None:
    """Print comprehensive baseline-focused permutation test results."""
    
    print("\n" + "="*80)
    print("PERMUTATION TESTS vs. BASELINE")
    print("="*80)
    
    print(f"\nTEST CONFIGURATION:")
    print(f"  • Number of permutations: {results['n_permutations']:,}")
    print(f"  • Random seed: {results['random_seed']}")
    print(f"  • Baseline accuracy: {results['baseline_accuracy']:.4f}")
    print(f"  • Baseline column: {results['baseline_col']}")
    
    print(f"\nNULL HYPOTHESIS:")
    print(f"  Demographic conditioning introduces no systematic bias beyond")
    print(f"  what would be expected from random variation around baseline performance.")
    
    # Individual trait results (vs baseline)
    for trait, trait_results in results["trait_tests"].items():
        if "error" in trait_results:
            print(f"\n{trait.upper()} vs BASELINE: {trait_results['error']}")
            continue
            
        print(f"\n{trait.upper()} BIAS TEST (vs Baseline):")
        print("-" * 50)
        print(f"Groups: {trait_results['trait_groups']}")
        print(f"Test: Sum of squared deviations from baseline")
        
        p_val = trait_results["p_value"]
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"Observed test statistic: {trait_results['observed_test_statistic']:.6f}")
        print(f"Permutation p-value: {p_val:.4f} {significance}")
        
        print(f"\nGroup accuracies vs baseline ({results['baseline_accuracy']:.4f}):")
        for group, acc in trait_results["group_accuracies"].items():
            deviation = trait_results["observed_deviations"][group]
            sign = "+" if deviation > 0 else ""
            print(f"  {group}: {acc:.4f} ({sign}{deviation:.4f})")
    
    # Overall bias test
    if "overall_bias_test" in results:
        overall = results["overall_bias_test"]
        if "error" not in overall:
            print(f"\nOVERALL PROFILE VARIANCE TEST:")
            print("-" * 50)
            print(f"Profiles tested: {overall['n_profiles']}")
            print(f"Observed variance around baseline: {overall['observed_variance']:.6f}")
            
            p_val = overall["p_value"]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"Permutation p-value: {p_val:.4f} {significance}")
            
            #if "intersectional_tests" in results:
            #    intersectional = results["intersectional_tests"]
            #print(f"\nGroup means:")
            #for group, mean_acc in intersectional["group_means"].items():
            #    print(f"  {group}: {mean_acc:.4f}")
    
    # Summary
    summary = results["summary"]
    print(f"\nSUMMARY:")
    print("-" * 30)
    print(f"  • Trait tests performed: {summary['total_trait_tests']}")
    print(f"  • Significant trait biases: {summary['significant_trait_tests']}")
    if summary['total_trait_tests'] > 0:
        sig_rate = summary['significant_trait_tests'] / summary['total_trait_tests']
        print(f"  • Significance rate: {sig_rate:.1%}")
    print(f"  • Minimum p-value: {summary['min_p_value']:.4f}")
    print(f"  • Overall bias significant: {summary['significant_overall_bias']}")

    if summary["significant_findings"]:
        print(f"\nSIGNIFICANT FINDINGS:")
        for finding in summary["significant_findings"]:
            print(f"  • {finding['type']}: p={finding['p_value']:.4f}")
            print(f"    {finding['interpretation']}")
    else:
        print(f"\nNO SIGNIFICANT FINDINGS DETECTED:")
        print(f"  No significant findings were detected in the permutation tests.")


def plot_permutation_distributions(
    results: Dict[str, Any],
    trait: str = "ethnicity",
    figsize: Tuple[int, int] = (12, 8),
    savepath: str = None
) -> plt.Figure:
    """
    Plot baseline-focused permutation test distributions.
    
    Creates a 2x2 subplot showing:
    - Trait-specific test statistic distribution vs. observed
    - Baseline deviations for each demographic group
    - Overall variance test distribution (if available)
    - P-value summary across all tests
    """
    
    if trait not in results["trait_tests"]:
        raise ValueError(f"Trait {trait} not found in results")
    
    trait_results = results["trait_tests"][trait]
    if "error" in trait_results:
        raise ValueError(f"Error in {trait} results: {trait_results['error']}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    
    # Top-left: Trait test statistic distribution
    null_dist = trait_results["permuted_test_statistics"]
    observed = trait_results["observed_test_statistic"]
    p_val = trait_results["p_value"]
    baseline_acc = results["baseline_accuracy"]
    
    axes[0,0].hist(null_dist, bins=50, alpha=0.7, density=True, color='lightblue', 
                  edgecolor='black', linewidth=0.5)
    axes[0,0].axvline(observed, color='red', linestyle='--', linewidth=2, 
                     label=f'Observed: {observed:.6f}')
    axes[0,0].set_xlabel('Sum of Squared Deviations from Baseline')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title(f'{trait.title()} Bias Test vs. Baseline\np-value = {p_val:.4f}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Top-right: Group deviations from baseline
    groups = list(trait_results["group_accuracies"].keys())
    group_accs = [trait_results["group_accuracies"][g] for g in groups]
    deviations = [trait_results["observed_deviations"][g] for g in groups]
    
    colors = ['red' if abs(dev) > 0.01 else 'lightblue' for dev in deviations]
    bars = axes[0,1].barh(range(len(groups)), deviations, color=colors)
    axes[0,1].set_yticks(range(len(groups)))
    axes[0,1].set_yticklabels([g.replace('_', ' ').title() for g in groups], fontsize=9)
    axes[0,1].axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    axes[0,1].set_xlabel('Deviation from Baseline')
    axes[0,1].set_title(f'{trait.title()} Group Deviations\n(Baseline: {baseline_acc:.4f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add accuracy labels
    for i, (acc, dev) in enumerate(zip(group_accs, deviations)):
        axes[0,1].text(dev + (0.002 if dev >= 0 else -0.002), i, f'{acc:.3f}', 
                      ha='left' if dev >= 0 else 'right', va='center', fontsize=8)
    
    # Bottom-left: Overall variance test (if available)
    if "overall_bias_test" in results and "error" not in results["overall_bias_test"]:
        overall = results["overall_bias_test"]
        null_variances = overall["permuted_variances"]
        observed_var = overall["observed_variance"]
        p_val_overall = overall["p_value"]
        
        axes[1,0].hist(null_variances, bins=50, alpha=0.7, density=True, color='lightgreen',
                      edgecolor='black', linewidth=0.5)
        axes[1,0].axvline(observed_var, color='red', linestyle='--', linewidth=2,
                         label=f'Observed: {observed_var:.6f}')
        axes[1,0].set_xlabel('Variance of Profile Accuracies around Baseline')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title(f'Overall Profile Variance Test\np-value = {p_val_overall:.4f}')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'Overall variance\ntest not available', 
                      ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
        axes[1,0].set_title('Overall Variance Test')
    
    # Bottom-right: P-value summary
    p_values = []
    test_names = []
    
    # Add trait test p-value
    p_values.append(trait_results["p_value"])
    test_names.append(f'{trait.title()} bias')
    
    # Add overall test p-value if available
    if "overall_bias_test" in results and "error" not in results["overall_bias_test"]:
        p_values.append(results["overall_bias_test"]["p_value"])
        test_names.append('Overall variance')
    
    # Add other trait tests
    for other_trait, other_results in results["trait_tests"].items():
        if other_trait != trait and "error" not in other_results:
            p_values.append(other_results["p_value"])
            test_names.append(f'{other_trait.title()} bias')
    
    if p_values:
        colors = ['red' if p < 0.05 else 'lightblue' for p in p_values]
        bars = axes[1,1].barh(range(len(p_values)), p_values, color=colors)
        axes[1,1].set_yticks(range(len(p_values)))
        axes[1,1].set_yticklabels(test_names, fontsize=9)
        axes[1,1].axvline(0.05, color='red', linestyle=':', alpha=0.7, label='α = 0.05')
        axes[1,1].set_xlabel('P-value')
        axes[1,1].set_title('All Permutation Test P-values')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlim(0, max(0.1, max(p_values) * 1.1))
    else:
        axes[1,1].text(0.5, 0.5, 'No p-values\navailable', 
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('P-value Summary')
    
    plt.suptitle(f'Permutation Tests vs. Baseline: {trait.title()} Analysis\n'
                 f'({results["n_permutations"]:,} permutations, baseline: {baseline_acc:.4f})', 
                 fontsize=14)
    
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
    
    return fig




def run_full_tier2_analysis(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = None, 
    create_visualizations: bool = True,
    perf_df: Optional[pd.DataFrame] = None,
    n_permutations: int = 1000,
    permutation_seed: int = 42, 
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,
    stage: str = "tier2",
    per_figure_subdirs: Optional[Dict[str, str]] = None
):
    """
    Run complete Tier 2 analysis pipeline with conditional cognitive style execution.
    """

    print("EXECUTING COMPREHENSIVE TIER 2 ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Group keys: {group_keys}")
    print(f"Dataset shape: {merged_df.shape}")
    print(f"Category columns from CaseConfig: {getattr(case, 'category_cols', None)}")

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

    category_cols = getattr(case, "category_cols", None) or ["stereotype_type"]

    subdirs = per_figure_subdirs or {}
    stage_dir     = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage)
    ensembles_dir = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("ensembles") or "ensembles")
    clusters_dir  = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("clusters") or "clusters")
    traits_dir    = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("traits") or "trait_comparison")
    recs_dir      = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("recommendations") or "recommendations")
    perm_dir      = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("permutations") or "permutation_tests")

    # STEP 1: ENSEMBLE BY TRAIT ANALYSIS
    try:
        print("\n=== Running Step 1: Ensemble by Trait Analysis...")
        ensemble_results = ensemble_by_trait_analysis(
            merged_df,
            person_set,
            case=case,
            group_keys=group_keys,
            perf_df=perf_df
        )
        print("SUCCESS: Ensemble analysis completed successfully")
    except Exception as e:
        print(f"ERROR: Ensemble analysis failed: {e}")
        ensemble_results = {'error': str(e)}

    # STEP 2: CLUSTER-LEVEL BIAS PATTERNS
    try:
        print("\n=== Running Step 2: Cluster-level Bias Analysis...")
        similarity_results = analyze_persona_similarity(
            merged_df,
            person_set=person_set
        )
        cluster_results = cluster_level_bias_patterns(
            merged_df,
            person_set=person_set,
            case=case,
            similarity_results=similarity_results,
            group_keys=group_keys,
            perf_df=perf_df
        )
        print("SUCCESS: Cluster analysis completed successfully")
    except Exception as e:
        print(f"ERROR: Cluster analysis failed: {e}")
        cluster_results = {'error': str(e)}

    has_cognitive_data = has_cognitive_style_data(person_set)
    cognitive_results = {'skipped': True, 'reason': 'No cognitive style data found'}



    j = 0
    if has_cognitive_data:
        try:
            print("\n=== Running Step 3: Multiple Trait Comparisons...")
            trait_analyses = [("cognitive_style", ["gender", "ethnicity"])]
            trait_comparison_results = run_multiple_trait_comparisons(
                merged_df, person_set, trait_analyses=trait_analyses, perf_df=perf_df
            )
            cognitive_results = trait_comparison_results.get(
                "cognitive_style", {'error': 'Missing cognitive results'}
            )
            print("SUCCESS: Trait comparisons (cognitive_style) completed successfully")

            j = 1
        except Exception as e:
            print(f"ERROR: Trait comparisons failed: {e}")
            cognitive_results = {'error': str(e)}
    else:
        print("[No cognitive trait data, part skipped]")


    results = {}
    # PERMUTATION TESTS
    print(f"\n\n=== Running Step {3+j}: Permutation Tests ===")
    try:
        permutation_results = run_permutation_tests(
            merged_df,
            person_set=person_set,
            n_permutations=n_permutations,
            traits=["gender", "ethnicity"],
            random_seed=permutation_seed,
            baseline_col="base_pred"
        )
        print_permutation_results(permutation_results)
        results["permutation_tests"] = permutation_results
    except Exception as e:
        print(f"Error running permutation tests: {e}")
        results["permutation_tests"] = {"error": str(e)}

    try:
        for trait_name, tr in permutation_results.get("trait_tests", {}).items():
            if "error" in tr:
                continue
            outp = os.path.join(perm_dir, f"permutation_{trait_name}.pdf")
            plot_permutation_distributions(permutation_results, trait=trait_name, savepath=outp)
    except Exception as e:
        print(f"WARNING: Saving permutation plots failed: {e}")
    

    # STEP 4: VISUALIZATIONS
    visualization_figures = {}
    if create_visualizations:
        try:
            print("\n=== Running Step 4: Creating Visualizations...")
    
            # Use cognitive style results if available for the trait plot
            trait_results_for_plots = (
                cognitive_results if (has_cognitive_data and isinstance(cognitive_results, dict)
                                      and 'error' not in cognitive_results and 'skipped' not in cognitive_results)
                else None
            )
    
            save_paths = {
                "ensemble": os.path.join(ensembles_dir, "ensemble_performance.pdf"),
                "cluster": os.path.join(clusters_dir, "cluster_analysis.pdf"),
                "trait": (
                    os.path.join(
                        traits_dir,
                        f"{trait_results_for_plots.get('comparison_trait','trait')}_comparison.pdf"
                    ) if trait_results_for_plots else None
                ),
                "recommendations": os.path.join(recs_dir, "system_recommendations.pdf"),
            }
    
            visualization_figures = create_all_tier2_visualizations(
                ensemble_results=ensemble_results,
                cluster_results=cluster_results,
                trait_results=trait_results_for_plots,
                show_top_ensembles=8,
                save_paths=save_paths
            )
            print("SUCCESS: Visualizations created successfully")
        except Exception as e:
            print(f"✗ ERROR: Visualization creation failed: {e}")
            visualization_figures = {'error': str(e)}
    
    # STEP 5: EXECUTIVE SUMMARY
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TIER 2 EXECUTIVE SUMMARY")
    print("=" * 80)
    summary = {}

    # Ensemble findings
    if isinstance(ensemble_results, dict) and 'recommendations' in ensemble_results:
        best_ensemble = ensemble_results['recommendations'].get('best_balanced', ['Unknown'])[0]
        summary['best_ensemble'] = best_ensemble
        print(f"   • Best Overall Ensemble: {best_ensemble}")


    # Cluster findings
    if isinstance(cluster_results, dict) and 'recommendations' in cluster_results:
        best_cluster = cluster_results['recommendations'].get('best_accuracy', ['Unknown'])[0]
        summary['best_cluster'] = best_cluster
        print(f"   • Best Cluster: {best_cluster}")

    
    # Cognitive style findings if they exist
    if has_cognitive_data and isinstance(cognitive_results, dict) \
       and ('error' not in cognitive_results) and ('skipped' not in cognitive_results):
        if 'recommendations' in cognitive_results and cognitive_results['recommendations']:
            best_cog = cognitive_results['recommendations'][0]
            best_cog_name = best_cog[0] if isinstance(best_cog, (list, tuple)) else str(best_cog)
            summary['best_cognitive_style'] = best_cog_name
            print(f"   • Best Cognitive Style: {best_cog_name}")
    else:
        if isinstance(cognitive_results, dict) and 'skipped' in cognitive_results:
            print(f"   • Cognitive style analysis skipped: {cognitive_results.get('reason', 'No reason provided')}")
        else:
            err = (cognitive_results or {}).get('error', 'Unknown error')
            print(f"   • Cognitive style analysis failed: {err}")
    
    return {
        'ensemble_analysis': ensemble_results,
        'cluster_analysis': cluster_results,
        'permutation_analysis': results["permutation_tests"],
        'cognitive_analysis': cognitive_results,
        'visualizations': visualization_figures,
        'executive_summary': summary,
        'analysis_metadata': {
            'group_keys': group_keys,
            'category_cols': category_cols,
            'has_cognitive_data': has_cognitive_data,
            'cognitive_analysis_status': (
                'skipped' if ('skipped' in (cognitive_results or {})) else
                ('failed' if ('error' in (cognitive_results or {})) else 'completed')
            )
        }
    }