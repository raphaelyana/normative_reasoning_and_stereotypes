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
from profiles.profile_sets import PERSON_SYSTEMATIC
from profiles.schema import PersonSet
from analysis_tools import has_cognitive_style_data
from cases.cases_config import CaseConfig


def build_trait_groups(merged_df: pd.DataFrame, person_set: PersonSet, group_keys=("gender", "ethnicity", "age")) -> Dict[str, list]:
    """
    Dynamically build profile groups based on group_keys using PersonSet metadata.
    Works for enums and strings, lowercases everything.
    """
    def norm_val(v):
        if hasattr(v, "value"):  # Enum
            v = v.value
        if v is None or str(v).lower() in {"", "unknown", "none", "nan"}:
            return "unknown"
        return str(v).strip().lower()

    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
    trait_groups = defaultdict(list)

    print(f"Building trait groups with keys: {group_keys}")
    print(f"Found {len(profile_cols)} profile columns")

    debug_traits = defaultdict(set)

    for profile in profile_cols:
        traits = person_set.get_traits(profile, group_keys)
        normalized_traits = {k: norm_val(traits.get(k, "unknown")) for k in group_keys}

        for k, v in normalized_traits.items():
            debug_traits[k].add(v)

        group_parts = []
        for k in group_keys:
            group_parts.append(normalized_traits.get(k, "unknown"))

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
    Majority vote across profile predictions with normalization.
    Returns '' when no valid vote.
    """
    available_profiles = [p for p in profile_list if p in df.columns]
    if not available_profiles:
        print(f"WARNING: No available profiles from list: {profile_list}")
        return pd.Series([''] * len(df), index=df.index)

    preds = df[available_profiles].applymap(lambda x: str(x).strip().lower() if pd.notna(x) else "")

    def get_majority_vote(row):
        valid_votes = [v for v in row if v not in ("", "nan", "none")]
        if not valid_votes:
            return ''
        counts = Counter(valid_votes)
        top = counts.most_common()
        if len(top)>=2 and top[0][1]==top[1][1]:
            tied = [k for k, c in counts.items() if c == top[0][1]]
            return sorted(tied)[0]
        return top[0][0]

    return preds.apply(get_majority_vote, axis=1)



def ensemble_by_trait_analysis(
    merged_df: pd.DataFrame, 
    person_set: PersonSet, 
    case: CaseConfig,
    group_keys=("gender", "ethnicity", "age")
) -> Dict[str, Any]:
    print("="*80)
    print("ENSEMBLE BY TRAIT ANALYSIS - PERSONSET VERSION")
    print("="*80)
    print(f"Group keys: {group_keys}")

    df = merged_df.copy()
    profile_cols = [c for c in df.columns if c.startswith("profile")]
    for c in ['true_label', 'base_pred', *profile_cols]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    trait_groups = build_trait_groups(df, person_set, group_keys)

    # Baseline performance
    true_labels = df['true_label']
    baseline_preds = df['base_pred']
    baseline_accuracy = accuracy_score(true_labels, baseline_preds)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print("\nEnsemble Performance by Trait Group:")
    print("-" * 50)

    ensemble_results = {}
    for group_name, profile_list in trait_groups.items():
        if len(profile_list) == 0:
            continue

        ensemble_preds = majority_vote_ensemble(df, profile_list)
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
                'ensemble_preds': ensemble_preds,
                'profiles': profile_list
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
    category_cols = getattr(case, "category_cols", None) or ["subject"]  # for MMLU

    for cat_col in category_cols:
        if cat_col not in df.columns:
            print(f"WARNING: category_col '{cat_col}' not found in data")
            continue

        categories = df[cat_col].dropna().unique()
        for category in categories:
            cat_subset = df[df[cat_col] == category]
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
    archetype_parameters = {
        "risk_benefit": [0.15, 0.05],
        "extra_error": 0.03,
        "rescue": 0.2,
        "accuracy": 0.72,
        "internal_agreement": 0.95,
        "accuracy_consensus": True,
    }
):
    print("="*80)
    print("CLUSTER-LEVEL BIAS AND RESCUE PATTERNS - PERSONSET VERSION")
    print("="*80)
    print(f"Group keys: {group_keys}")

    # Normalize data
    df = merged_df.copy()
    profile_cols = [c for c in df.columns if c.startswith("profile")]
    for c in ['true_label', 'base_pred', *profile_cols]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Get clustering results if not provided (must pass person_set!)
    if similarity_results is None:
        similarity_results = analyze_persona_similarity(df, person_set=person_set)

    # Rescue & bias patterns (multi-category)
    try:
        category_cols = getattr(case, "category_cols", None)
        if not category_cols:
            print("Error retrieving category columns - defaulting to ['subject']")
            category_cols = ["subject"]

        rescue_stats_list = []
        bias_patterns_list = []
        for cat_col in category_cols:
            if cat_col in df.columns:
                rs = rescue_stats_by_category(df, category_col=cat_col)
                rs["category_col"] = cat_col
                rescue_stats_list.append(rs)

                bp = detect_systematic_biases(
                    df, person_set=person_set, category_col=cat_col,
                    baseline_col="base_pred", profile_prefix="profile",
                    label_col="true_label"
                )
                bp["category_col"] = cat_col
                bias_patterns_list.append(bp)

        if not rescue_stats_list:
            raise ValueError("No valid category columns found in df.")

        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)
        bias_patterns = pd.concat(bias_patterns_list, ignore_index=True)

    except Exception as e:
        print("WARNING: Could not compute rescue/bias stats:", e)
        # (Optional) leave your fallback random generation here if you want

        rescue_stats = pd.DataFrame()
        bias_patterns = pd.DataFrame()

    cluster_analysis = {}
    baseline_accuracy = accuracy_score(df["true_label"], df["base_pred"])

    for cluster_id, cluster_info in similarity_results['clusters'].items():
        cluster_profiles = cluster_info['profiles']
        available_profiles = [p for p in cluster_profiles if p in df.columns]
        print(f"\n{cluster_id.upper()} ({len(available_profiles)} profiles):")

        trait_composition = defaultdict(list)
        for profile in available_profiles:
            traits = person_set.get_traits(profile, group_keys)
            for trait_name, trait_value in traits.items():
                if hasattr(trait_value, "value"):
                    trait_value = trait_value.value
                val = "unknown" if trait_value is None else str(trait_value).lower()
                trait_composition[trait_name].append(val)

        print(f"Trait composition:")
        for trait_name, values in trait_composition.items():
            value_counts = Counter(values)
            print(f"  {trait_name}: {dict(value_counts)}")

        print(f"Sample profiles: {', '.join(p for p in available_profiles[:5])}")
        if len(available_profiles) > 5:
            print(f"                 ... and {len(available_profiles) - 5} more")

        cluster_metrics = {
            'profiles': available_profiles,
            'size': len(available_profiles),
            'internal_agreement': cluster_info.get('internal_agreement', 0),
            'trait_composition': dict(trait_composition)
        }

        accuracies = [(df[p] == df['true_label']).mean() for p in available_profiles]
        cluster_metrics['accuracy_mean'] = float(np.mean(accuracies)) if accuracies else 0.0
        cluster_metrics['accuracy_std'] = float(np.std(accuracies)) if accuracies else 0.0

        if not rescue_stats.empty:
            cluster_rescue = rescue_stats[rescue_stats['profile'].isin(available_profiles)]
            cluster_metrics.update({
                'rescue_rate_mean': float(cluster_rescue['rescue_rate'].mean()) if len(cluster_rescue) else 0.0,
                'rescue_rate_std': float(cluster_rescue['rescue_rate'].std()) if len(cluster_rescue) else 0.0,
                'extra_error_rate_mean': float(cluster_rescue['extra_err_rate'].mean()) if len(cluster_rescue) else 0.0,
                'extra_error_rate_std': float(cluster_rescue['extra_err_rate'].std()) if len(cluster_rescue) else 0.0,
                'total_rescued': int(cluster_rescue['rescued'].sum()) if len(cluster_rescue) else 0,
                'total_extra_errors': int(cluster_rescue['extra_errors'].sum()) if len(cluster_rescue) else 0
            })
        else:
            cluster_metrics.update({
                'rescue_rate_mean': 0.0, 'rescue_rate_std': 0.0,
                'extra_error_rate_mean': 0.0, 'extra_error_rate_std': 0.0,
                'total_rescued': 0, 'total_extra_errors': 0
            })

        if not bias_patterns.empty:
            cluster_bias = bias_patterns[bias_patterns['profile'].isin(available_profiles)]
            cluster_metrics.update({
                'bias_magnitude_mean': float(cluster_bias['bias_magnitude'].mean()) if len(cluster_bias) else 0.0,
                'bias_magnitude_std': float(cluster_bias['bias_magnitude'].std()) if len(cluster_bias) else 0.0,
                'mislabelling_rate_mean': float(cluster_bias['mislabelling_rate'].mean()) if len(cluster_bias) else 0.0,
                'dominant_bias_direction': cluster_bias['bias_direction'].mode().iloc[0] if (len(cluster_bias) and not cluster_bias['bias_direction'].empty) else 'none'
            })
        else:
            cluster_metrics.update({
                'bias_magnitude_mean': 0.0, 'bias_magnitude_std': 0.0,
                'mislabelling_rate_mean': 0.0, 'dominant_bias_direction': 'none'
            })

        # Accuracy consensus gate
        if archetype_parameters["accuracy_consensus"]:
            per_profile_acc = []
            for p in profile_cols:
                try:
                    per_profile_acc.append(float((df[p] == df["true_label"]).mean()))
                except Exception:
                    continue
            if len(per_profile_acc) >= 3:
                global_acc_mean = float(np.mean(per_profile_acc))
                percentile = 80.0
                global_acc_p80 = float(np.percentile(per_profile_acc, percentile))
                cluster_metrics["global_accuracy_mean"] = global_acc_mean
                cluster_metrics["global_accuracy_p80"] = global_acc_p80
                cluster_metrics["accuracy_gap"] = float(cluster_metrics["accuracy_mean"] - global_acc_mean)
                above_mean_flags = [
                    float((df[p] == df["true_label"]).mean()) > global_acc_mean for p in available_profiles
                ] if available_profiles else []
                cluster_metrics["prop_above_global_mean"] = float(np.mean(above_mean_flags)) if above_mean_flags else 0.0
                condition_acc = global_acc_p80
            else:
                condition_acc = float(archetype_parameters["accuracy"])
                cluster_metrics["global_accuracy_mean"] = None
                cluster_metrics["global_accuracy_p80"] = None
                cluster_metrics["accuracy_gap"] = None
                cluster_metrics["prop_above_global_mean"] = None
        else:
            condition_acc = float(archetype_parameters["accuracy"])
            cluster_metrics["global_accuracy_mean"] = None
            cluster_metrics["global_accuracy_p80"] = None
            cluster_metrics["accuracy_gap"] = float(cluster_metrics["accuracy_mean"] - condition_acc)
            cluster_metrics["prop_above_global_mean"] = None

        # Archetype
        if (cluster_metrics['rescue_rate_mean'] > archetype_parameters["risk_benefit"][0] 
            and cluster_metrics['extra_error_rate_mean'] < archetype_parameters["risk_benefit"][1]):
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

        print(f"  Accuracy: {cluster_metrics['accuracy_mean']:.4f} ± {cluster_metrics['accuracy_std']:.4f}")
        print(f"  Rescue Rate: {cluster_metrics['rescue_rate_mean']:.3f} ± {cluster_metrics['rescue_rate_std']:.3f}")
        print(f"  Extra Error Rate: {cluster_metrics['extra_error_rate_mean']:.3f} ± {cluster_metrics['extra_error_rate_std']:.3f}")
        print(f"  Bias Magnitude: {cluster_metrics['bias_magnitude_mean']:.3f} (direction: {cluster_metrics['dominant_bias_direction']})")
        print(f"  Internal Agreement: {cluster_metrics['internal_agreement']:.3f}")
        print(f"  Archetype: {archetype}")

        cluster_analysis[cluster_id] = cluster_metrics

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
            ensemble_preds = majority_vote_ensemble(df, cluster_profiles)
            if not ensemble_preds.eq('').all():
                ensemble_accuracy = accuracy_score(df['true_label'], ensemble_preds)
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
) -> Dict[str, Any]:
    if control_traits is None:
        control_traits = []

    print("=" * 80)
    print(f"TRAIT COMPARISON: {comparison_trait.upper()}")
    if control_traits:
        print(f"CONTROLLING FOR: {', '.join(control_traits).upper()}")
    print("=" * 80)

    df = merged_df.copy()
    profile_cols = [c for c in df.columns if c.startswith("profile")]
    for c in ['true_label', 'base_pred', *profile_cols]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    trait_groups = defaultdict(list)
    trait_metadata = {}

    def norm(v):
        if hasattr(v, "value"): v = v.value
        return "unknown" if v is None else str(v).lower()

    for profile in profile_cols:
        traits = person_set.get_traits(profile, [comparison_trait] + control_traits)
        trait_value = norm(traits.get(comparison_trait, "unknown"))
        if trait_value != "unknown":
            trait_groups[trait_value].append(profile)
            trait_metadata[profile] = {k: norm(v) for k, v in traits.items()}

    trait_groups = {k: v for k, v in trait_groups.items() if v}
    if len(trait_groups) < 2:
        print(f"ERROR: Insufficient groups for {comparison_trait} comparison")
        return {'error': f'Insufficient groups for {comparison_trait}'}

    print(f"Found {len(trait_groups)} groups for {comparison_trait}:")
    for trait_val, profiles in trait_groups.items():
        print(f"  {trait_val}: {len(profiles)} profiles")

    # Rescue & bias
    category_cols = getattr(case, "category_cols", None) or ["subject"]
    rescue_stats_list, bias_patterns_list = [], []
    for cat_col in category_cols:
        if cat_col in df.columns:
            rs = rescue_stats_by_category(df, category_col=cat_col)
            rs["category_col"] = cat_col
            rescue_stats_list.append(rs)

            bp = detect_systematic_biases(
                df, person_set=person_set, category_col=cat_col,
                baseline_col="base_pred", profile_prefix="profile", label_col="true_label"
            )
            bp["category_col"] = cat_col
            bias_patterns_list.append(bp)

    rescue_stats = pd.concat(rescue_stats_list, ignore_index=True) if rescue_stats_list else pd.DataFrame()
    bias_patterns = pd.concat(bias_patterns_list, ignore_index=True) if bias_patterns_list else pd.DataFrame()

    trait_performance = {}
    for trait_value, profiles in trait_groups.items():
        available_profiles = [p for p in profiles if p in df.columns]
        if not available_profiles:
            continue

        accuracies, rescue_rates, extra_error_rates, bias_magnitudes = [], [], [], []
        for profile in available_profiles:
            accuracies.append(accuracy_score(df['true_label'], df[profile]))

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

        if control_traits:
            for control_trait in control_traits:
                control_values = []
                for profile in available_profiles:
                    control_values.append(trait_metadata.get(profile, {}).get(control_trait, "unknown"))
                trait_performance[trait_value]['control_trait_distribution'][control_trait] = dict(Counter(control_values))

        print(f"\n{str(trait_value).upper()} {comparison_trait.upper()} (n={len(available_profiles)}):")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Rescue Rate: {np.mean(rescue_rates):.3f} ± {np.std(rescue_rates):.3f}")
        print(f"  Extra Error Rate: {np.mean(extra_error_rates):.3f} ± {np.std(extra_error_rates):.3f}")
        print(f"  Bias Magnitude: {np.mean(bias_magnitudes):.3f} ± {np.std(bias_magnitudes):.3f}")
        if control_traits:
            print(f"  Control trait distributions:")
            for ct, dist in trait_performance[trait_value]['control_trait_distribution'].items():
                print(f"    {ct}: {dist}")
    
    # Statistical comparisons (your existing logic)
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
            # Check for variation
            combined = np.concatenate(groups)
            if np.all(combined == combined[0]):
                print(f"  {metric_name}: No variation detected")
                continue
            
            if len(groups) == 2:
                # Two groups: Mann-Whitney U test
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
                # Multiple groups: Kruskal-Wallis
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
                
                # Post-hoc pairwise comparisons if significant
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
    
    # Composite scoring
    weights = {'Accuracy': 0.3, 'Rescue Rate': 0.25, 'Extra Error Rate': -0.25, 'Bias Magnitude': -0.2}
    composite_scores = {}
    
    for trait_value in trait_groups.keys():
        score = 0
        count = 0
        for metric_name, weight in weights.items():
            if metric_name in rankings:
                rank = next((i for i, (s, _) in enumerate(rankings[metric_name], 1) if s == trait_value), len(rankings[metric_name]) + 1)
                max_rank = len(rankings[metric_name])
                rank_score = (max_rank + 1 - rank) / max_rank  # Normalize to [0,1]
                score += weight * rank_score
                count += abs(weight)
        
        composite_scores[trait_value] = score / count if count > 0 else 0
    
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
    case: CaseConfig,
    person_set: PersonSet,
    trait_analyses: List[Tuple[str, List[str]]] = None
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
                case=case,
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
    figsize: tuple = (10, 6),
    key_ensembles: Optional[List[str]] = None,
    show_top_n: int = None
):
    ensemble_data = ensemble_results.get('ensemble_results', {})
    if not ensemble_data:
        print("No ensemble data to plot.")
        return None

    # Filter and sort ensembles
    if key_ensembles is None:
        key_ensembles = list(ensemble_data.keys())
    if show_top_n:
        sorted_ensembles = sorted(
            [(name, data) for name, data in ensemble_data.items() if name in key_ensembles],
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        key_ensembles = [name for name, _ in sorted_ensembles[:show_top_n]]

    ensemble_names, accuracies, improvements, rescue_rates, extra_error_rates = [], [], [], [], []
    for name in key_ensembles:
        if name in ensemble_data:
            d = ensemble_data[name]
            ensemble_names.append(name.replace('_', '\n').title())
            accuracies.append(d.get('accuracy', 0.0))
            improvements.append(d.get('improvement', 0.0))
            rescue_rates.append(d.get('rescue_rate', 0.0))
            extra_error_rates.append(d.get('extra_error_rate', 0.0))

    if not ensemble_names:
        print("No selected ensembles to display.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Ensemble Performance Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Accuracy Improvement
    x_pos = np.arange(len(ensemble_names))
    colors = ['#2ca02c' if imp > 0.01 else '#1f77b4' if imp >= 0 else '#d62728' for imp in improvements]
    bars1 = ax1.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    for i, (bar, acc, imp) in enumerate(zip(bars1, accuracies, improvements)):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + (0.001 if h >= 0 else -0.002),
                 f'{acc:.3f}', ha='center', va='bottom' if h >= 0 else 'top',
                 fontsize=9, fontweight='bold')

    ax1.set_xlabel('Ensemble Strategy', fontsize=12)
    ax1.set_ylabel('Accuracy Improvement vs Baseline', fontsize=12)
    ax1.set_title('Accuracy Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ensemble_names, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Risk-Benefit
    scatter = ax2.scatter(extra_error_rates, rescue_rates, c=improvements, cmap='RdYlGn',
                          s=100, alpha=0.7, edgecolors='black', linewidth=1)

    for i, name in enumerate(ensemble_names):
        ax2.annotate(name.replace('\n', ' '), (extra_error_rates[i], rescue_rates[i]),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax2.set_xlabel('Extra Error Rate (Risk)', fontsize=12)
    ax2.set_ylabel('Rescue Rate (Benefit)', fontsize=12)
    ax2.set_title('Risk-Benefit Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    if len(extra_error_rates) >= 2:
        x_min, x_max = min(extra_error_rates), max(extra_error_rates)
        y_min, y_max = min(rescue_rates), max(rescue_rates)
        x_margin = (x_max - x_min) * 0.15 if x_max > x_min else 0.05
        y_margin = (y_max - y_min) * 0.15 if y_max > y_min else 0.05
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Accuracy Improvement', fontsize=10)

    plt.tight_layout()
    plt.show()
    return fig


def plot_cluster_analysis(
    cluster_results: Dict[str, Any],
    figsize: tuple = (14, 6)
):
    cluster_data = cluster_results.get('cluster_analysis', {})
    if not cluster_data:
        print("No cluster analysis data to plot.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Cluster-Level Bias and Performance Analysis', fontsize=16, fontweight='bold')

    archetype_colors = {
        'Optimal Risk-Benefit': '#2ca02c',
        'Cautious': '#1f77b4', 
        'High Error-Correction': '#ff7f0e',
        'High Performer': '#d62728',
        'High Consistency': '#9467bd',
        'Similar to Neutral': '#8c564b'
    }

    cluster_names, accuracies, rescue_rates, extra_error_rates, colors = [], [], [], [], []
    for cluster_id, cluster_info in cluster_data.items():
        cluster_names.append(cluster_id.replace('cluster_', 'Cluster '))
        accuracies.append(cluster_info.get('accuracy_mean', 0.0))
        rescue_rates.append(cluster_info.get('rescue_rate_mean', 0.0))
        extra_error_rates.append(cluster_info.get('extra_error_rate_mean', 0.0))
        archetype = cluster_info.get('archetype', 'Unknown')
        colors.append(archetype_colors.get(archetype, '#bcbd22'))

    x_pos = np.arange(len(cluster_names))
    width = 0.25

    bars1 = ax1.bar(x_pos - width, accuracies, width, label='Accuracy', color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x_pos, rescue_rates, width, label='Rescue Rate', color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5)
    bars3 = ax1.bar(x_pos + width, extra_error_rates, width, label='Extra Error Rate', color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Performance Metrics', fontsize=12)
    ax1.set_title('Cluster Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cluster_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    scatter = ax2.scatter(extra_error_rates, rescue_rates, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)

    for i, (cluster_name, cluster_info) in enumerate(zip(cluster_names, cluster_data.values())):
        archetype = cluster_info.get('archetype', 'Unknown')
        ax2.annotate(f"{cluster_name}\n({archetype})", (extra_error_rates[i], rescue_rates[i]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))

    ax2.set_xlabel('Extra Error Rate (Risk)', fontsize=12)
    ax2.set_ylabel('Rescue Rate (Benefit)', fontsize=12)
    ax2.set_title('Cluster Risk-Benefit Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    if extra_error_rates and rescue_rates:
        x_min, x_max = min(extra_error_rates), max(extra_error_rates)
        y_min, y_max = min(rescue_rates), max(rescue_rates)
        x_margin = (x_max - x_min) * 0.2 if x_max > x_min else 0.05
        y_margin = (y_max - y_min) * 0.2 if y_max > y_min else 0.05
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)

    legend_elements = []
    for archetype, color in archetype_colors.items():
        if any(ci.get('archetype') == archetype for ci in cluster_data.values()):
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
    if trait_name is None:
        trait_name = trait_results.get('comparison_trait', 'Trait')

    trait_performance = trait_results.get('trait_performance', {})
    if not trait_performance:
        print(f"No trait performance data found for {trait_name}")
        return None

    if baseline_accuracy is None:
        all_accuracies = []
        for _, data in trait_performance.items():
            all_accuracies.extend(data.get('accuracies', []))
        baseline_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{trait_name.title()} Comparison Analysis (Relative to Baseline)', fontsize=16, fontweight='bold')

    trait_values = list(trait_performance.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(trait_values)))
    metrics = [
        ('accuracies', 'Accuracy Improvement', axes[0, 0], baseline_accuracy),
        ('rescue_rates', 'Rescue Rate', axes[0, 1], 0),
        ('extra_error_rates', 'Extra Error Rate', axes[1, 0], 0),
        ('bias_magnitudes', 'Bias Magnitude', axes[1, 1], 0)
    ]

    for metric_key, metric_name, ax, baseline in metrics:
        means, stds = [], []
        for trait_val in trait_values:
            data = trait_performance[trait_val].get(metric_key, [])
            if data:
                if metric_key == 'accuracies':
                    improvements = [acc - baseline for acc in data]
                    means.append(np.mean(improvements))
                    stds.append(np.std(improvements))
                else:
                    means.append(np.mean(data))
                    stds.append(np.std(data))
            else:
                means.append(0.0)
                stds.append(0.0)

        x_pos = np.arange(len(trait_values))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        for bar, mean, std in zip(bars, means, stds):
            h = bar.get_height()
            label_y = (h + std + 0.001) if h >= 0 else (h - std - 0.001)
            ax.text(bar.get_x() + bar.get_width()/2., label_y, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=9, fontweight='bold')

        ax.set_xlabel(f'{trait_name.title()} Level', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(
            f'{metric_name} vs Baseline ({baseline:.3f})' if metric_key == 'accuracies'
            else f'{metric_name} by {trait_name.title()}',
            fontsize=14, fontweight='bold'
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(tv).title() for tv in trait_values], rotation=45, ha='right')
        if metric_key == 'accuracies':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')
            ax.legend()

        y_data = [m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)]
        if y_data:
            y_min, y_max = min(y_data), max(y_data)
            if y_max == y_min:
                y_min, y_max = y_min - 0.05, y_max + 0.05
            y_margin = (y_max - y_min) * 0.15
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

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
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('System-Level AI Safety Recommendations', fontsize=16, fontweight='bold')

    recommendations, scores, colors = [], [], []
    ensemble_data = ensemble_results.get('ensemble_results', {})
    cluster_data = cluster_results.get('cluster_analysis', {})
    baseline_acc = ensemble_results.get('baseline_accuracy', 0.0)

    # Best overall ensemble (balanced)
    best_balanced = (ensemble_results.get('recommendations', {}) or {}).get('best_balanced')
    if best_balanced:
        if isinstance(best_balanced, tuple):
            best_ens_name, best_ens_score = best_balanced[0], best_balanced[1]
        else:
            best_ens_name = best_balanced[0] if best_balanced else 'Unknown'
            best_ens_score = ensemble_data.get(best_ens_name, {}).get('improvement', 0.0)
        recommendations.append(f"Best Ensemble\n{best_ens_name.replace('_', ' ').title()}")
        scores.append(float(best_ens_score))
        colors.append('#2ca02c')

    # Best cluster
    best_cluster = (cluster_results.get('recommendations', {}) or {}).get('best_accuracy')
    if best_cluster:
        if isinstance(best_cluster, tuple):
            cluster_name, cluster_info = best_cluster[0], best_cluster[1]
            cluster_score = cluster_info.get('accuracy_mean', 0.0) - baseline_acc
        else:
            cluster_name = best_cluster[0] if best_cluster else 'Unknown'
            cluster_score = cluster_data.get(cluster_name, {}).get('accuracy_mean', 0.0) - baseline_acc
        recommendations.append(f"Best Cluster\n{cluster_name.replace('_', ' ').title()}")
        scores.append(float(cluster_score))
        colors.append('#1f77b4')

    # Best trait (if provided)
    if trait_results and 'recommendations' in trait_results and trait_results['recommendations']:
        best_trait = trait_results['recommendations'][0]
        trait_name = trait_results.get('comparison_trait', 'Trait')
        recommendations.append(f"Best {trait_name.title()}\n{str(best_trait[0]).title()}")
        scores.append(float(best_trait[1]) / 10.0)  # normalize
        colors.append('#ff7f0e')

    # Safest ensemble (lowest extra error rate)
    safest = (ensemble_results.get('recommendations', {}) or {}).get('safest')
    if safest:
        safest_name = safest[0] if isinstance(safest, (list, tuple)) and safest else 'Unknown'
        err_rate = (ensemble_data.get(safest_name, {}) or {}).get('extra_error_rate', 0.0)
        recommendations.append(f"Safest Ensemble\n{safest_name.replace('_', ' ').title()}")
        scores.append(-float(err_rate))
        colors.append('#d62728')

    if not recommendations:
        print("No recommendations to plot.")
        return None

    y_pos = np.arange(len(recommendations))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, score in zip(bars, scores):
        w = bar.get_width()
        ax.text(w + (0.001 if w >= 0 else -0.001), bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left' if w >= 0 else 'right',
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
    show_top_ensembles: int = 8
):
    print("Creating Tier 2 Visualizations...")
    print("="*50)
    figures = {}

    try:
        print("1. Ensemble Performance Analysis...")
        fig1 = plot_ensemble_performance_comparison(ensemble_results, show_top_n=show_top_ensembles)
        if fig1: figures['ensemble'] = fig1
    except Exception as e:
        print(f"   Error creating ensemble plot: {e}")

    try:
        print("2. Cluster Analysis...")
        fig2 = plot_cluster_analysis(cluster_results)
        if fig2: figures['cluster'] = fig2
    except Exception as e:
        print(f"   Error creating cluster plot: {e}")

    if trait_results and 'error' not in trait_results and 'skipped' not in trait_results:
        try:
            print("3. Trait Comparison Analysis...")
            fig3 = plot_trait_comparison_results(trait_results, baseline_accuracy=baseline_accuracy)
            if fig3: figures['trait'] = fig3
        except Exception as e:
            print(f"   Error creating trait plot: {e}")

    try:
        print("4. System-Level Recommendations...")
        fig4 = plot_system_level_recommendations(
            ensemble_results, cluster_results, trait_results
        )
        if fig4: figures['recommendations'] = fig4
    except Exception as e:
        print(f"   Error creating recommendations plot: {e}")

    print(f"\nCompleted! Created {len(figures)} visualizations.")
    return figures



def run_full_tier2_analysis(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = None, 
    create_visualizations: bool = True,
):
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
    
    # Use case.category_cols directly (MMLU: ['subject'])
    category_cols = getattr(case, "category_cols", None) or ["subject"]

    # STEP 1: ENSEMBLE BY TRAIT ANALYSIS
    try:
        print("\n=== Running Step 1: Ensemble by Trait Analysis...")
        ensemble_results = ensemble_by_trait_analysis(
            merged_df, person_set, case=case, group_keys=group_keys
        )
        print("SUCCESS: Ensemble analysis completed successfully")
    except Exception as e:
        print(f"ERROR: Ensemble analysis failed: {e}")
        ensemble_results = {'error': str(e), 'baseline_accuracy': 0.0}

    # STEP 2: CLUSTER-LEVEL BIAS PATTERNS
    try:
        print("\n=== Running Step 2: Cluster-level Bias Analysis...")
        similarity_results = analyze_persona_similarity(merged_df, person_set=person_set)
        cluster_results = cluster_level_bias_patterns(
            merged_df, person_set=person_set, case=case,
            similarity_results=similarity_results, group_keys=group_keys
        )
        print("SUCCESS: Cluster analysis completed successfully")
    except Exception as e:
        print(f"ERROR: Cluster analysis failed: {e}")
        cluster_results = {'error': str(e)}

    has_cognitive_data = has_cognitive_style_data(person_set)
    cognitive_results = {'skipped': True, 'reason': 'No cognitive style data found'}

    # STEP 3: TRAIT COMPARISONS (optional)
    print("\n=== Running Step 3: Multiple Trait Comparisons...")
    if has_cognitive_data:
        try:
            trait_analyses = [("cognitive_style", ["gender", "ethnicity"])]
            trait_comparison_results = run_multiple_trait_comparisons(
                merged_df, person_set, case, trait_analyses=trait_analyses
            )
            cognitive_results = trait_comparison_results.get("cognitive_style", {'error': 'Missing cognitive results'})
            print("SUCCESS: Trait comparisons (cognitive_style) completed successfully")
        except Exception as e:
            print(f"ERROR: Trait comparisons failed: {e}")
            cognitive_results = {'error': str(e)}
    else:
        print("[No cognitive trait data, part skipped]")

    # STEP 4: VISUALIZATIONS
    visualization_figures = {}
    if create_visualizations:
        try:
            print("\n=== Running Step 4: Creating Visualizations...")
            baseline_acc = ensemble_results.get('baseline_accuracy', 0.0) if isinstance(ensemble_results, dict) else 0.0
            trait_for_plot = cognitive_results if ('error' not in cognitive_results and 'skipped' not in cognitive_results) else None
            visualization_figures = create_all_tier2_visualizations(
                ensemble_results=ensemble_results,
                cluster_results=cluster_results,
                trait_results=trait_for_plot,
                baseline_accuracy=baseline_acc,
                show_top_ensembles=8
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

    # Cognitive style findings (if any)
    if has_cognitive_data and isinstance(cognitive_results, dict) and ('error' not in cognitive_results) and ('skipped' not in cognitive_results):
        if cognitive_results.get('recommendations'):
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
