import pandas as pd
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

from analysis_tools import get_demographic_info, get_analysis_group_keys, guarded_labelspace_analysis
from profiles.schema import *
from profiles.profile_sets import PERSON_ETHNICS
from cases.cases_config import CaseConfig

from plot_tools import apply_neurips_figure_style


__all__ = [
    "test_comprehensive_demographic_accuracy_differences",
    "print_comprehensive_demographic_results",
    "extract_high_disagreement_cases",
    "print_disagreement_analysis",
    "rescue_stats_by_category",
    "analyze_rescue_performance",
    "detect_systematic_biases",
    "analyze_systematic_bias_patterns",
    "analyze_persona_similarity",
    "print_persona_similarity_analysis",
    "plot_accuracy_deltas_with_ci",
    "run_full_preliminary_analysis",
    #"compute_pairwise_demographic_diffs",
    #"plot_volcano_demographic_diffs",
    #"plot_effect_size_heatmap",
    #"plot_intersectional_accuracy_heatmap",
    #"plot_demographic_accuracy_composite",
]


__all__ += [
    "compute_pairwise_demographic_diffs",
    "plot_volcano_demographic_diffs",
    "plot_effect_size_heatmap",
    "plot_intersectional_accuracy_heatmap",
    "plot_demographic_accuracy_composite",
]


# ============================================================================
# Preliminary Analysis
# ============================================================================

def test_comprehensive_demographic_accuracy_differences(
    merged_df,
    person_set: PersonSet = PERSON_ETHNICS,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of demographic group accuracy differences.
    Adapted to work with datasets where profile columns are simply 'profileX'.
    """
    
    results = {}

    df_cols = {c for c in merged_df.columns if c.startswith("profile")}
    meta_cols = set(person_set.metadata.keys())
    
    covered = df_cols & meta_cols
    missing_in_meta = df_cols - meta_cols
    unused_meta = meta_cols - df_cols
    
    print(f"[INFO] Profiles covered by metadata: {len(covered)}/{len(df_cols)}")
    if missing_in_meta:
        some = ", ".join(sorted(list(missing_in_meta))[:10])
        print(f"[WARN] {len(missing_in_meta)} dataframe columns have no metadata (e.g., {some} …)")
    if unused_meta:
        print(f"[INFO] {len(unused_meta)} metadata profiles have no column in dataframe (ignored)")

    def get_profiles_by_trait(trait_name, trait_value):
        """Return profile column names in merged_df that match a given trait."""
        matching = []
        if hasattr(trait_value, "value"):
            trait_value = trait_value.value
        trait_value = str(trait_value).lower()
    
        for pid, meta in person_set.metadata.items():
            v = getattr(meta, trait_name, None)
            if hasattr(v, "value"): 
                v = v.value
            v = None if v is None else str(v).lower()
        
            if v == trait_value and pid in merged_df.columns:
                matching.append(pid)
        return matching

    def calculate_group_accuracy(profiles, df):
        return [(df[p] == df['true_label']).mean() for p in profiles if p in df.columns]

    def compare_groups(group1_profiles, group2_profiles, group1_name, group2_name, df):
        acc1 = calculate_group_accuracy(group1_profiles, df)
        acc2 = calculate_group_accuracy(group2_profiles, df)

        if not acc1 or not acc2:
            print(f"=== Skipping comparison between {group1_name} and {group2_name}: No data")
            return None
        
        t_stat, p_val = ttest_ind(acc1, acc2)
        pooled_std = np.sqrt((np.var(acc1) + np.var(acc2))/2)
        effect_size = (np.mean(acc1)-np.mean(acc2))/pooled_std if pooled_std > 0 else 0
        return {
            f'{group1_name}_accuracy': np.mean(acc1),
            f'{group2_name}_accuracy': np.mean(acc2),
            'difference': np.mean(acc1) - np.mean(acc2),
            'p_value': p_val,
            'significant': p_val < 0.05,
            'effect_size': effect_size,
            'sample_sizes': f"{group1_name}: {len(acc1)}, {group2_name}: {len(acc2)}",
            'interpretation': f'{"Significant" if p_val < 0.05 else "Non-significant"} accuracy difference between groups'
        }
    
    available_traits = list(PersonMeta.__dataclass_fields__.keys())

    # Gender comparison
    if "gender" in available_traits:
        men = get_profiles_by_trait("gender", Gender.man)
        women = get_profiles_by_trait("gender", Gender.woman)
        res = compare_groups(men, women, "man", "woman", merged_df)
        if res: results["men_vs_women"] = res

    # Ethnicity comparisons
    ethnicities = {getattr(meta, "ethnicity") for meta in person_set.metadata.values()}
    for e1, e2 in combinations(sorted(ethnicities, key=lambda x: str(x)), 2):
        p1 = get_profiles_by_trait("ethnicity", e1)
        p2 = get_profiles_by_trait("ethnicity", e2)
        if p1 and p2:
            res = compare_groups(
                p1, p2,
                str(getattr(e1, "value", e1)).lower(),
                str(getattr(e2, "value", e2)).lower(),
                merged_df
            )
            if res:
                results[f"{str(getattr(e1, 'value', e1)).lower()}_vs_{str(getattr(e2, 'value', e2)).lower()}"] = res

    # Other traits comparisons
    extra_traits = [t for t in available_traits if t not in ("gender", "ethnicity")]
    for trait in extra_traits:
        values = set(getattr(meta, trait) for meta in person_set.metadata.values() if getattr(meta, trait) is not None)
        if len(values) > 1:
            for v1, v2 in combinations(sorted(values), 2):
                p1 = get_profiles_by_trait(trait, v1)
                p2 = get_profiles_by_trait(trait, v2)
                res = compare_groups(p1, p2, str(v1).lower(), str(v2).lower(), merged_df)
                if res: results[f"{trait}_{str(v1).lower()}_vs_{str(v2).lower()}"] = res

    # Intersectional comparisons
    if "gender" in available_traits and "ethnicity" in available_traits:
        genders = set(getattr(meta, "gender") for meta in person_set.metadata.values())
        ethnicities = set(getattr(meta, "ethnicity") for meta in person_set.metadata.values())
        for g1, g2 in combinations(genders, 2):
            for eth in ethnicities:
                g1_profiles = [
                    pid for pid, meta in person_set.metadata.items()
                    if getattr(meta, "gender") == g1 and getattr(meta, "ethnicity") == eth
                    and pid in merged_df.columns
                ]
                g2_profiles = [
                    pid for pid, meta in person_set.metadata.items()
                    if getattr(meta, "gender") == g2 and getattr(meta, "ethnicity") == eth
                    and pid in merged_df.columns
                ]
                if g1_profiles and g2_profiles:
                    res = compare_groups(
                        g1_profiles, g2_profiles,
                        f"{getattr(eth, 'value', eth).lower()}_{getattr(g1, 'value', g1).lower()}",
                        f"{getattr(eth, 'value', eth).lower()}_{getattr(g2, 'value', g2).lower()}",
                        merged_df
                    )
                    if res:
                        results[f"intersectional_{eth.value}_{g1.value}_vs_{g2.value}"] = res

    significant_comparisons = sum(1 for r in results.values() if isinstance(r, dict) and r.get('significant', False))
    total_comparisons = sum(1 for r in results.values() if isinstance(r, dict))
    effect_sizes = [(k, v["effect_size"]) for k, v in results.items() if "effect_size" in v]
    effect_sizes.sort(key=lambda x: abs(x[1]), reverse=True)

    results['summary'] = {
        'total_comparisons': total_comparisons,
        'significant_comparisons': significant_comparisons,
        'significance_rate': significant_comparisons/total_comparisons if total_comparisons > 0 else 0,
        'largest_effects': effect_sizes[:5],
    }

    return results




def print_comprehensive_demographic_results(results):
    """
    Generate comprehensive report of demographic accuracy differences analysis.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from test_comprehensive_demographic_accuracy_differences()
    
    Returns:
    --------
    dict : Dictionary of significant results for further analysis
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEMOGRAPHIC ACCURACY DIFFERENCES ANALYSIS")  
    print("="*80)
    
    if not results:
        print("No results found in the analysis.")
        return {}

    # Executive Summary
    if 'summary' in results:
        summary = results['summary']
        print(f"\nEXECUTIVE SUMMARY")
        print("-" * 40)
        print(f"Total Comparisons Performed: {summary['total_comparisons']}")
        print(f"Significant Differences Found: {summary['significant_comparisons']}")
        print(f"Statistical Significance Rate: {summary['significance_rate']:.1%}")
        
        print(f"\nLARGEST EFFECT SIZES")
        print("-" * 40)
        for i, (comparison, effect_size) in enumerate(summary['largest_effects'][:5], 1):
            magnitude = "Large" if abs(effect_size) > 0.8 else "Medium" if abs(effect_size) > 0.5 else "Small"
            print(f"{i:2d}. {comparison:<35} Effect Size: {effect_size:6.3f} ({magnitude})")
    
    print(f"\n" + "="*80)
    print("DETAILED SIGNIFICANT RESULTS")
    print("="*80)

    def format_comparison_result(test_name, result):
        """Format comparison row for printing."""
        acc_keys = [k for k in result if k.endswith("_accuracy")]
        if len(acc_keys) == 2:
            acc1_key, acc2_key = acc_keys
            acc1 = result[acc1_key]
            acc2 = result[acc2_key]
            g1 = acc1_key.replace("_accuracy", "").replace("_", " ").title()
            g2 = acc2_key.replace("_accuracy", "").replace("_", " ").title()
            acc_display = f"{g1}: {acc1:.3f} | {g2}: {acc2:.3f}"
        else:
            acc_display = "Accuracy unavailable"
        return f"{'[SIGNIFICANT]' if result['significant'] else '[NON-SIGNIFICANT]    '} {test_name:<35} | {acc_display}"

    sig_results = {}
    for key, result in results.items():
        if key == 'summary':
            continue
        print(format_comparison_result(key, result))
        print(f"         Δ = {result['difference']:+.3f} | p = {result['p_value']:.3f} | d = {result.get('effect_size', 0):.3f}")
        if result.get('significant', False):
            sig_results[key] = result


    print(f"\n" + "="*80)
    print("HIGH-LEVEL INTERPRETATION")
    print("="*80)

    total_tests = results['summary']['total_comparisons']
    sig_count = len(sig_results)
    
    if sig_count > total_tests * 0.3:
        level = "HIGH"
        advice = "Results (High/Moderate/Low): High bias detected across demographics."
    elif sig_count > total_tests * 0.1:
        level = "MODERATE"
        advice = "Results (High/Moderate/Low): Moderate bias detected across demographics."
    else:
        level = "LOW"
        advice = "Results (High/Moderate/Low): Minimal bias observed across demographics."

    print(f"Bias Level: {level}")
    print(f"Recommendation: {advice}")
    
    if summary := results.get("summary"):
        if summary["largest_effects"]:
            biggest = summary["largest_effects"][0]
            magnitude = "Substantial" if abs(biggest[1]) > 0.8 else "Moderate" if abs(biggest[1]) > 0.5 else "Small"
            print(f"\nLargest Detected Bias: {biggest[0]} (d = {biggest[1]:.3f}, {magnitude})")

    return sig_results



def extract_high_disagreement_cases(
    merged: pd.DataFrame, 
    threshold: float = 0.7,
    sample_id_col: str = "sample_id",
    label_col: str = "true_label",
    baseline_col: str = "base_pred",
    person_set: Optional[PersonSet] = None
) -> pd.DataFrame:
    """
    Identify cases with high disagreement among profile predictions.
    
    This function analyzes prediction consensus across profiles to identify
    samples where profiles show substantial disagreement. High disagreement
    cases often represent challenging or ambiguous samples that warrant
    closer examination for bias analysis and model improvement.
    
    Parameters:
    -----------
    merged : pd.DataFrame
        DataFrame containing profile predictions and metadata
    threshold : float, default=0.7
        Minimum disagreement score (0-1) to classify as high disagreement.
        Score represents proportion of profiles disagreeing with modal prediction.
    sample_id_col : str, default="sample_id"
        Column name containing unique sample identifiers
    label_col : str, default="true_label"
        Column name containing ground truth labels
    baseline_col : str, default="base_pred"
        Column name containing baseline model predictions
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing high disagreement cases with columns:
        - sample_id: Unique identifier for the sample
        - disagreement_score: Proportion of profiles disagreeing with mode (0-1)
        - prediction_distribution: Counter object showing prediction frequency
        - modal_prediction: Most common prediction across profiles
        - minority_predictions: List of non-modal predictions
        - true_label: Ground truth label
        - base_pred: Baseline model prediction
        - consensus_strength: Strength of modal prediction (inverse of disagreement)
        - prediction_entropy: Information-theoretic measure of disagreement
    
    Raises:
    -------
    ValueError
        If required columns are missing or no profile columns found
    """
    
    required_cols = [sample_id_col, label_col, baseline_col]
    missing = [col for col in required_cols if col not in merged.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Use profile keys from PROFILE_META
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    if not profile_cols:
        raise ValueError("No valid profile prediction columns found in DataFrame.")

    trait_by_profile =(
        {p: get_demographic_info(p, person_set) for p in profile_cols} 
        if person_set is not None else None
    )

    print(f"Analyzing disagreement across {len(profile_cols)} profiles...")
    print(f"Using disagreement threshold: {threshold}")

    records = []

    for idx, row in merged.iterrows():
        preds = [row[p] for p in profile_cols]
        counts = Counter(preds)
        total = len(preds)
        modal, modal_count = counts.most_common(1)[0]
        disagreement = (total - modal_count) / total
        entropy = -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)

        modal_trait_dist = None
        minority_trait_dist = None
        if trait_by_profile is not None:
            modal_traits = [trait_by_profile[col] for col, y in zip(profile_cols, preds) if y == modal]
            minority_traits = [trait_by_profile[col] for col, y in zip(profile_cols, preds) if y != modal]
            modal_trait_dist = dict(Counter(modal_traits))
            minority_trait_dist = dict(Counter(minority_traits))

        records.append({
            "sample_id": row[sample_id_col],
            "disagreement_score": disagreement,
            "consensus_strength": modal_count / total,
            "prediction_distribution": dict(counts),
            "modal_prediction": modal,
            "minority_predictions": [k for k in counts if k != modal],
            "prediction_entropy": entropy,
            "true_label": row[label_col],
            "base_pred": row[baseline_col],
            "total_profiles": total,
            "modal_count": modal_count,
            "minority_count": total - modal_count,
            **({"modal_trait_distribution": modal_trait_dist,
               "minority_trait_distribution": minority_trait_dist} if person_set is not None else {})
        })

    df = pd.DataFrame(records)
    high_disagreement = df[df["disagreement_score"] > threshold].copy()
    high_disagreement.sort_values("disagreement_score", ascending=False, inplace=True)

    print("\nDISAGREEMENT ANALYSIS SUMMARY")
    print("-" * 50)
    print(f"Total samples analyzed: {len(df)}")
    print(f"High disagreement cases (>{threshold}): {len(high_disagreement)}")
    print(f"High disagreement rate: {len(high_disagreement)/len(df):.1%}")
    
    if len(high_disagreement):
        print(f"Average disagreement score: {high_disagreement['disagreement_score'].mean():.3f}")
        print(f"Max disagreement score: {high_disagreement['disagreement_score'].max():.3f}")
        print(f"Average entropy: {high_disagreement['prediction_entropy'].mean():.3f}")

    return high_disagreement.reset_index(drop=True)


def print_disagreement_analysis(high_disagreement_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print comprehensive analysis of high disagreement cases.
    
    Parameters:
    -----------
    high_disagreement_df : pd.DataFrame
        Output from extract_high_disagreement_cases function
    top_n : int, default=10
        Number of top cases to display in detailed analysis
    """
    
    if high_disagreement_df.empty:
        print("No high disagreement cases found.")
        return

    print("\n" + "="*80)
    print("HIGH DISAGREEMENT CASES ANALYSIS")
    print("="*80)

    # Summary
    print("\nSUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total high disagreement cases: {len(high_disagreement_df):,}")
    print(f"Average disagreement score: {high_disagreement_df['disagreement_score'].mean():.3f}")
    print(f"Standard deviation: {high_disagreement_df['disagreement_score'].std():.3f}")
    print(f"Range: {high_disagreement_df['disagreement_score'].min():.3f} - {high_disagreement_df['disagreement_score'].max():.3f}")
    print(f"Average prediction entropy: {high_disagreement_df['prediction_entropy'].mean():.3f}")

    # Top N disagreement samples
    print(f"\nTOP {top_n} HIGHEST DISAGREEMENT CASES")
    print("-" * 60)
    print(f"{'Rank':<6}{'Sample ID':<15}{'Disagreement':<13}{'Entropy':<10}{'Modal Pred':<12}{'True Label'}")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(high_disagreement_df.head(top_n).iterrows(), 1):
        print(f"{idx:<6}{str(row['sample_id']):<15}{row['disagreement_score']:<13.3f}"
              f"{row['prediction_entropy']:<10.3f}{str(row['modal_prediction']):<12}{str(row['true_label'])}")

    # Prediction distribution patterns
    print("\nPREDICTION DISTRIBUTION PATTERNS")
    print("-" * 50)
    all_distributions = high_disagreement_df['prediction_distribution'].tolist()
    pattern_counts = Counter()

    for dist in all_distributions:
        pattern = tuple(sorted(dist.items()))
        pattern_counts[pattern] += 1

    print("Most common disagreement patterns:")
    for pattern, count in pattern_counts.most_common(5):
        pattern_str = ", ".join([f"{pred}: {cnt}" for pred, cnt in pattern])
        print(f"  {pattern_str} (appears {count} times)")

    # Accuracy comparison
    print("\nACCURACY ANALYSIS FOR HIGH DISAGREEMENT CASES")
    print("-" * 50)
    modal_correct = high_disagreement_df['modal_prediction'] == high_disagreement_df['true_label']
    baseline_correct = high_disagreement_df['base_pred'] == high_disagreement_df['true_label']
    print(f"Modal prediction accuracy: {modal_correct.mean():.3f}")
    print(f"Baseline prediction accuracy: {baseline_correct.mean():.3f}")
    print(f"Correct modal predictions: {modal_correct.sum()} / {len(high_disagreement_df)}")
    print(f"Correct baseline predictions: {baseline_correct.sum()} / {len(high_disagreement_df)}")

    # Consensus strength bins
    print("\nCONSENSUS STRENGTH DISTRIBUTION")
    print("-" * 40)
    consensus_bins = pd.cut(
        high_disagreement_df['consensus_strength'], 
        bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0], 
        labels=['Very Low (≤30%)', 'Low (30–40%)', 'Medium (40–50%)', 'High (50–60%)', 'Very High (>60%)']
    )
    consensus_dist = consensus_bins.value_counts().sort_index()
    for category, count in consensus_dist.items():
        pct = count / len(high_disagreement_df) * 100
        print(f"  {category}: {count} cases ({pct:.1f}%)")

    if "modal_trait_distribution" in high_disagreement_df.columns:
        print("\nTRAIT DISTRIBUTION FOR TOP DISAGREEMENT CASES")
        print("-" * 50)
        for idx, row in high_disagreement_df.head(top_n).iterrows():
            print(f"Sample {row['sample_id']}:")
            print(f"  Modal group: {row['modal_trait_distribution']}")
            print(f"  Minority group: {row['minority_trait_distribution']}")



def rescue_stats_by_category(
    merged: pd.DataFrame,
    category_col: str,
    baseline_col: str = "base_pred",
    label_col: str = "true_label",
    profile_prefix: str = "profile",
    case: CaseConfig = None,
    person_set=None,
    **kwargs
) -> pd.DataFrame:
    """
    Dataset-agnostic rescue statistics analysis that handles any category values and None values
    """
    
    if category_col not in merged.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame")

    if baseline_col not in merged.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in DataFrame")

    if label_col not in merged.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Handle None/NaN values in category column
    merged_clean = merged.copy()
    merged_clean[category_col] = merged_clean[category_col].fillna("Unknown")
    
    # Get unique category values (whatever they are for this dataset)
    unique_categories = merged_clean[category_col].dropna().unique()
    print(f"Found category values in {category_col}: {sorted(unique_categories)}")

    # Standardize labels
    y_true = merged_clean[label_col].astype(str).str.strip().str.lower()
    y_base = merged_clean[baseline_col].astype(str).str.strip().str.lower()

    # Get profile columns
    profile_cols: List[str] = [c for c in merged_clean.columns if c.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No profile columns found with prefix '{profile_prefix}'")

    output_records = []

    # Iterate through ALL category values found in the data
    for category_value in unique_categories:
        category_df = merged_clean[merged_clean[category_col] == category_value]
        category_size = len(category_df)
        
        if category_size == 0:
            continue

        y_true_cat = y_true.loc[category_df.index]
        y_base_cat = y_base.loc[category_df.index]

        base_correct_mask = y_base_cat == y_true_cat
        base_err_count = (~base_correct_mask).sum()
        base_ok_count = base_correct_mask.sum()
        base_acc = base_ok_count/category_size if category_size > 0 else 0.0

        for profile in profile_cols:
            y_prof_cat = category_df[profile].astype(str).str.strip().str.lower()
            prof_correct_mask = y_prof_cat == y_true_cat

            rescued = ((~base_correct_mask) & prof_correct_mask).sum()
            extra_errors = (base_correct_mask & (~prof_correct_mask)).sum()

            rescue_rate = rescued / base_err_count if base_err_count > 0 else 0.0
            extra_err_rate = extra_errors / base_ok_count if base_ok_count > 0 else 0.0
            prof_acc = prof_correct_mask.mean()

            output_records.append({
                'category': str(category_value),  # Ensure string for consistency
                'profile': profile,
                'N_cat': category_size,
                'rescued': int(rescued),
                'rescue_rate': rescue_rate,
                'extra_errors': int(extra_errors),
                'extra_err_rate': extra_err_rate,
                'profile_acc': prof_acc,
                'baseline_acc': base_acc
            })

    result_df = pd.DataFrame(output_records)
    if not result_df.empty:
        result_df = result_df.sort_values(["category", "rescued"], ascending=[True, False])
    
    return result_df


def print_rescue_analysis_results(rescue_analysis: Dict[str, Any], category_name: str) -> None:
    """
    Updated rescue analysis results printer with better terminology
    """
    if "error" in rescue_analysis:
        print(f"Error in rescue analysis for {category_name}: {rescue_analysis['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"RESCUE ANALYSIS FOR CATEGORY COLUMN: {category_name.upper()}")
    print(f"{'='*60}")
    
    # Summary statistics
    summary = rescue_analysis['summary']
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Unique category values: {summary['total_categories']}")
    print(f"  Total profiles analyzed: {summary['total_profiles']}")
    print(f"  Total samples: {summary['total_samples']:,}")
    print(f"  Total rescues: {summary['total_rescues']}")
    print(f"  Total extra errors: {summary['total_extra_errors']}")
    print(f"  Average rescue rate: {summary['avg_rescue_rate']:.3f}")
    print(f"  Average extra error rate: {summary['avg_extra_error_rate']:.3f}")
    print(f"  Net rescue benefit: {summary['total_rescues'] - summary['total_extra_errors']}")
    
    # Interpretation
    net_benefit = summary['total_rescues'] - summary['total_extra_errors']
    if net_benefit > 0:
        print(f"  -- Profiles provide net benefit (more rescues than extra errors)")
    elif net_benefit < 0:
        print(f"  -- Profiles cause net harm (more extra errors than rescues)")
    else:
        print(f"  -- Profiles have neutral impact")
    
    # Rest of the function remains the same...
    print(f"\nTOP 10 RESCUE PERFORMERS:")
    print(f"{'Profile':<12}{'Category Value':<20}{'Rescue Rate':<12}{'Rescues':<8}{'Accuracy':<10}")
    print("-" * 70)
    for performer in rescue_analysis['top_rescue_performers']:
        category_short = str(performer['category'])[:18]
        print(f"{performer['profile']:<12}{category_short:<20}"
              f"{performer['rescue_rate']:<12.3f}{performer['rescued']:<8}"
              f"{performer['profile_acc']:<10.3f}")
    
    print(f"\nTOP 10 HIGHEST ERROR RISK PROFILES:")
    print(f"{'Profile':<12}{'Category Value':<20}{'Error Rate':<12}{'Extra Errors':<12}{'Accuracy':<10}")
    print("-" * 75)
    for risk_profile in rescue_analysis['highest_error_risk']:
        category_short = str(risk_profile['category'])[:18]
        print(f"{risk_profile['profile']:<12}{category_short:<20}"
              f"{risk_profile['extra_err_rate']:<12.3f}{risk_profile['extra_errors']:<12}"
              f"{risk_profile['profile_acc']:<10.3f}")
    
    print(f"\nPERFORMANCE BY CATEGORY VALUE:")
    print(f"{'Category Value':<20}{'Avg Rescue':<12}{'Max Rescue':<12}{'Avg Extra Err':<15}{'Avg Accuracy':<12}{'Sample Size'}")
    print("-" * 95)
    for category, stats in rescue_analysis['category_performance'].items():
        category_short = str(category)[:18]
        sample_size = stats[('N_cat', 'first')] if ('N_cat', 'first') in stats else "N/A"
        print(f"{category_short:<20}{stats[('rescue_rate', 'mean')]:<12.3f}"
              f"{stats[('rescue_rate', 'max')]:<12.3f}{stats[('extra_err_rate', 'mean')]:<15.3f}"
              f"{stats[('profile_acc', 'mean')]:<12.3f}{sample_size}")


def analyze_rescue_performance(rescue_stats_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Dataset-agnostic rescue performance analysis
    """
    if rescue_stats_df.empty:
        return {"error": "Empty rescue statistics DataFrame provided"}

    analysis = {}

    # Overall summary metrics
    analysis['summary'] = {
        'total_categories': rescue_stats_df['category'].nunique(),
        'total_profiles': rescue_stats_df['profile'].nunique(),
        'total_samples': rescue_stats_df['N_cat'].sum(),
        'total_rescues': rescue_stats_df['rescued'].sum(),
        'total_extra_errors': rescue_stats_df['extra_errors'].sum(),
        'avg_rescue_rate': rescue_stats_df['rescue_rate'].mean(),
        'avg_extra_error_rate': rescue_stats_df['extra_err_rate'].mean()
    }

    # Top profiles by rescue rate
    analysis['top_rescue_performers'] = (
        rescue_stats_df.nlargest(10, 'rescue_rate')[
            ['profile', 'category', 'rescue_rate', 'rescued', 'profile_acc']
        ].to_dict('records')
    )

    # Profiles with highest extra error rate
    analysis['highest_error_risk'] = (
        rescue_stats_df.nlargest(10, 'extra_err_rate')[
            ['profile', 'category', 'extra_err_rate', 'extra_errors', 'profile_acc']
        ].to_dict('records')
    )

    # Category-level aggregation - works with any category values
    category_stats = rescue_stats_df.groupby('category').agg({
        'rescue_rate': ['mean', 'std', 'max'],
        'extra_err_rate': ['mean', 'std', 'max'],
        'profile_acc': ['mean', 'std'],
        'N_cat': 'first'  # Sample size for each category
    }).round(3)

    analysis['category_performance'] = category_stats.to_dict('index')

    # Profile-level aggregation
    profile_stats = rescue_stats_df.groupby('profile').agg({
        'rescue_rate': ['mean', 'std'],
        'extra_err_rate': ['mean', 'std'],
        'profile_acc': ['mean', 'std'],
        'rescued': 'sum',
        'extra_errors': 'sum'
    }).round(3)

    analysis['profile_performance'] = profile_stats.to_dict('index')

    return analysis

def print_all_rescue_analyses(rescue_analysis_all: Dict[str, Dict[str, Any]]) -> None:
    """
    Print rescue analyses for all category columns with cross-category summary
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESCUE STATISTICS ANALYSIS")
    print(f"{'='*80}")
    
    if not rescue_analysis_all:
        print("No rescue analyses to display.")
        return
    
    # Print individual category column analyses
    for cat_col, rescue_analysis in rescue_analysis_all.items():
        print_rescue_analysis_results(rescue_analysis, cat_col)
    
    # Cross-category summary (if multiple category columns)
    if len(rescue_analysis_all) > 1:
        print(f"\n{'='*60}")
        print("CROSS-CATEGORY COLUMN SUMMARY")
        print(f"{'='*60}")
        
        total_rescues = sum(
            analysis['summary']['total_rescues'] 
            for analysis in rescue_analysis_all.values() 
            if 'summary' in analysis
        )
        total_extra_errors = sum(
            analysis['summary']['total_extra_errors'] 
            for analysis in rescue_analysis_all.values() 
            if 'summary' in analysis
        )
        avg_rescue_rate = np.mean([
            analysis['summary']['avg_rescue_rate'] 
            for analysis in rescue_analysis_all.values() 
            if 'summary' in analysis
        ])
        avg_extra_error_rate = np.mean([
            analysis['summary']['avg_extra_error_rate'] 
            for analysis in rescue_analysis_all.values() 
            if 'summary' in analysis
        ])
        
        print(f"OVERALL STATISTICS ACROSS ALL CATEGORY COLUMNS:")
        print(f"  • Total rescues: {total_rescues}")
        print(f"  • Total extra errors: {total_extra_errors}")
        print(f"  • Average rescue rate: {avg_rescue_rate:.3f}")
        print(f"  • Average extra error rate: {avg_extra_error_rate:.3f}")
        print(f"  • Net benefit: {total_rescues - total_extra_errors} (rescues - extra errors)")
        
        # Best performing category columns
        category_rescue_rates = {
            cat: analysis['summary']['avg_rescue_rate'] 
            for cat, analysis in rescue_analysis_all.items() 
            if 'summary' in analysis
        }
        
        if category_rescue_rates:
            best_rescue_category = max(category_rescue_rates, key=category_rescue_rates.get)
            
            category_error_rates = {
                cat: analysis['summary']['avg_extra_error_rate'] 
                for cat, analysis in rescue_analysis_all.items() 
                if 'summary' in analysis
            }
            safest_category = min(category_error_rates, key=category_error_rates.get)
            
            print(f"  • Best rescue category column: {best_rescue_category} ({category_rescue_rates[best_rescue_category]:.3f})")
            print(f"  • Safest category column: {safest_category} ({category_error_rates[safest_category]:.3f})")
    else:
        print(f"\nSingle category column analysis complete.")


def detect_systematic_biases(
    merged: pd.DataFrame,
    person_set: PersonSet,
    category_col: str = "stereotype_type",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile",
    positive_label: str = "stereotype",
    negative_label: str = "unrelated",
    case: CaseConfig = None,
) -> pd.DataFrame:
    """
    Fixed version using get_demographic_info_fixed
    
    Parameters:
    -----------
    merged : pd.DataFrame
        DataFrame containing predictions and categories
    person_set : PersonSet
        PersonSet containing profile metadata
    category_col : str, default="stereotype_type"
        Column name containing category labels
    baseline_col : str, default="base_pred"
        Column name containing baseline predictions
    profile_prefix : str, default="profile"
        Prefix for identifying profile columns
    positive_label : str, default="stereotype"
        Label representing positive class
    negative_label : str, default="unrelated"
        Label representing negative class
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing bias patterns with columns:
        - category: Category value
        - profile: Profile identifier
        - demographic: Demographic information for profile
        - bias_direction: Direction of bias (more_positive, more_negative, neutral)
        - bias_magnitude: Magnitude of bias (0-1)
        - weighted_bias_magnitude: Bias weighted by category size
        - n_mislabelling: Total number of mislabeled instances
        - mislabelling_rate: Rate of mislabeling in category
        - category_size: Number of samples in category
        - positive_mislabelling: Number of false positives
        - negative_mislabelling: Number of false negatives
    """

    required_cols = [category_col, baseline_col]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    profile_cols = [c for c in merged.columns if c.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No columns found with prefix '{profile_prefix}'")

    # Map profiles to demographics once using fixed function
    trait_by_profile = {p: get_demographic_info(p, person_set) for p in profile_cols}

    bias_patterns = []
    global_size = len(merged)

    for category_value in merged[category_col].dropna().unique():
        category_data = merged[merged[category_col] == category_value]
        baseline = category_data[baseline_col]
        size = len(category_data)

        for profile in profile_cols:
            profile_preds = category_data[profile]
            
            # Count mislabeling in both directions
            pos_mislabelling = ((baseline == negative_label) & (profile_preds == positive_label)).sum()
            neg_mislabelling = ((baseline == positive_label) & (profile_preds == negative_label)).sum()

            total_mislabelling = pos_mislabelling + neg_mislabelling
            mislabelling_rate = total_mislabelling / size if size > 0 else 0

            # Determine bias direction and magnitude
            if pos_mislabelling > neg_mislabelling:
                bias_direction = "more_positive"
                bias_magnitude = (pos_mislabelling - neg_mislabelling) / size
            elif neg_mislabelling > pos_mislabelling:
                bias_direction = "more_negative"
                bias_magnitude = (neg_mislabelling - pos_mislabelling) / size
            else:
                bias_direction = "neutral"
                bias_magnitude = 0.0

            # Weight bias by category size relative to total dataset
            weighted_bias_magnitude = bias_magnitude * (size / global_size)

            bias_patterns.append({
                "category": category_value,
                "profile": profile,
                "demographic": trait_by_profile[profile],
                "bias_direction": bias_direction,
                "bias_magnitude": bias_magnitude,
                "weighted_bias_magnitude": weighted_bias_magnitude,
                "n_mislabelling": total_mislabelling,
                "mislabelling_rate": mislabelling_rate,
                "category_size": size,
                "positive_mislabelling": pos_mislabelling,
                "negative_mislabelling": neg_mislabelling,
            })

    bias_df = pd.DataFrame(bias_patterns)
    bias_df = bias_df.sort_values("weighted_bias_magnitude", ascending=False, key=abs)
    return bias_df


def analyze_systematic_bias_patterns(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    category_col: str = "stereotype_type",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive analysis of systematic bias patterns with statistical rigor.
    
    Performs complete bias pattern analysis including sample size considerations,
    reliability assessments, and meaningful pattern identification. Provides
    comprehensive statistical analysis of prediction biases across demographic
    profiles and content categories.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        DataFrame containing predictions, categories, and profile information
    person_set : PersonSet
        PersonSet containing profile metadata
    category_col : str, default="stereotype_type"
        Column name containing category labels
    baseline_col : str, default="base_pred"
        Column name containing baseline predictions
    profile_prefix : str, default="profile"
        Prefix for identifying profile columns
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - bias_patterns: Complete bias analysis results
        - meaningful_patterns: Statistically significant patterns only
        - category_summary: Category-level aggregated statistics
    """
    
    print("SYSTEMATIC BIAS PATTERN ANALYSIS")
    print("=" * 60)

    # Run bias detection with fixed function
    bias_patterns = detect_systematic_biases(
        merged_df,
        person_set,
        category_col=category_col,
        baseline_col=baseline_col,
        profile_prefix=profile_prefix
    )
    bias_patterns = bias_patterns.sort_values("weighted_bias_magnitude", ascending=False, key=abs)

    # Dataset composition
    category_sizes = merged_df.groupby(category_col).size().reset_index()
    category_sizes.columns = ['category', 'sample_size']
    total_samples = len(merged_df)
    category_sizes['percentage'] = (category_sizes['sample_size'] / total_samples * 100).round(1)

    print(f"\nDATASET COMPOSITION BY CATEGORY")
    print("-" * 40)
    for _, row in category_sizes.sort_values('sample_size', ascending=False).iterrows():
        print(f"{str(row['category']):<20}{row['sample_size']:<10}{row['percentage']:.1f}%")

    print(f"\nBIAS DETECTION RESULTS SUMMARY")
    print("-" * 40)
    print(f"Total bias patterns analyzed: {len(bias_patterns):,}")
    print(f"Unique categories: {bias_patterns['category'].nunique()}")
    print(f"Unique profiles: {bias_patterns['profile'].nunique()}")
    print(f"Average category size: {bias_patterns['category_size'].mean():.1f} samples")

    print(f"\nTOP 20 STRONGEST BIAS PATTERNS")
    print("-" * 90)
    print(f"{'Category':<15}{'Profile':<20}{'Direction':<15}{'Mag.':<10}{'Weighted':<10}{'Mislab.':<10}{'Cat.Size'}")
    print("-" * 90)
    for _, row in bias_patterns.head(20).iterrows():
        print(f"{str(row['category']):<15}{str(row['profile']):<20}{row['bias_direction']:<15}"
              f"{row['bias_magnitude']:<10.3f}{row['weighted_bias_magnitude']:<10.3f}{row['n_mislabelling']:<8}{row['category_size']}")

    # Reliability analysis
    large = bias_patterns[bias_patterns['category_size'] >= 100]
    medium = bias_patterns[(bias_patterns['category_size'] >= 50) & (bias_patterns['category_size'] < 100)]
    small = bias_patterns[bias_patterns['category_size'] < 50]

    print(f"\nRELIABILITY ANALYSIS BY SAMPLE SIZE")
    print("-" * 50)
    print(f"Large categories (≥100): {large['category'].nunique()}")
    print(f"Medium (50–99): {medium['category'].nunique()}")
    print(f"Small (<50): {small['category'].nunique()}")

    # Statistically meaningful bias detection
    def assess_bias_significance(row):
        if row['category_size'] >= 100:
            return abs(row['bias_magnitude']) >= 0.02
        elif row['category_size'] >= 50:
            return abs(row['bias_magnitude']) >= 0.03
        else:
            return abs(row['bias_magnitude']) >= 0.05

    bias_patterns['statistically_meaningful'] = bias_patterns.apply(assess_bias_significance, axis=1)
    meaningful = bias_patterns[bias_patterns['statistically_meaningful']].copy()
    meaningful = meaningful.sort_values("weighted_bias_magnitude", key=abs, ascending=False)

    print(f"\nSTATISTICALLY MEANINGFUL BIAS PATTERNS")
    print("-" * 70)
    print(f"Identified {len(meaningful)} statistically meaningful patterns")

    print(f"\nTOP 15 STATISTICALLY MEANINGFUL PATTERNS")
    print("-" * 90)
    print(f"{'Category':<15}{'Profile':<20}{'Direction':<15}{'Mag.':<10}{'Weighted':<10}{'Size':<8}{'Threshold'}")
    print("-" * 90)
    for _, row in meaningful.head(15).iterrows():
        threshold = (
            "≥2.0%" if row['category_size'] >= 100 else
            "≥3.0%" if row['category_size'] >= 50 else
            "≥5.0%"
        )
        print(f"{str(row['category']):<15}{str(row['profile']):<20}{row['bias_direction']:<15}"
              f"{row['bias_magnitude']:<10.3f}{row['weighted_bias_magnitude']:<10.3f}{row['category_size']:<8}{threshold}")

    # Category summary
    category_summary = bias_patterns.groupby('category').agg({
        'bias_magnitude': ['mean', 'max', 'std'],
        'weighted_bias_magnitude': 'sum',
        'n_mislabelling': 'sum',
        'mislabelling_rate': 'mean',
        'category_size': 'first'
    }).round(3)
    
    category_summary.columns = [
        'avg_bias_magnitude', 'max_bias_magnitude', 'bias_std',
        'total_weighted_bias', 'total_mislabelling', 'avg_mislabelling_rate', 'sample_size'
    ]

    print(f"\nCATEGORY-LEVEL BIAS SUMMARY")
    print("-" * 70)
    print(f"{'Category':<15}{'Avg Bias':<12}{'Max Bias':<12}{'Weighted Bias':<15}{'Total Mislab.':<16}{'Sample Size'}")
    print("-" * 70)
    for category in category_summary.sort_values('max_bias_magnitude', ascending=False).index:
        row = category_summary.loc[category]
        print(f"{category:<15}{row['avg_bias_magnitude']:<12.3f}{row['max_bias_magnitude']:<12.3f}"
              f"{row['total_weighted_bias']:<15.3f}{row['total_mislabelling']:<16.0f}{row['sample_size']:<12.0f}")

    print(f"\nFINAL ASSESSMENT")
    print("-" * 40)
    if len(meaningful) > 0:
        print(f"{len(meaningful)} statistically meaningful patterns detected")
        print(f"Affected categories: {meaningful['category'].nunique()}")
        print(f"Affected profiles: {meaningful['profile'].nunique()}")
    else:
        print("No statistically meaningful bias patterns detected")

    return bias_patterns, meaningful, category_summary



def analyze_systematic_bias_patterns_multi_category(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    case: CaseConfig,
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile"
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Analyze systematic bias patterns across all category columns in the case
    
    Returns:
    --------
    Dict mapping category_col -> (bias_patterns, meaningful_patterns, category_summary)
    """
    
    print("MULTI-CATEGORY SYSTEMATIC BIAS PATTERN ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    for cat_col in case.category_cols:
        if cat_col not in merged_df.columns:
            print(f"  Category '{cat_col}' not found in data, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"ANALYZING CATEGORY: {cat_col.upper()}")
        print(f"{'='*60}")
        
        # Check if this category has sufficient data
        category_counts = merged_df[cat_col].value_counts()
        print(f"Category distribution for {cat_col}:")
        for cat_val, count in category_counts.head(10).items():
            print(f"  {cat_val}: {count} samples")
        
        if len(category_counts) < 2:
            print(f"  Category '{cat_col}' has insufficient variation, skipping...")
            continue
            
        try:
            bias_patterns, meaningful_patterns, category_summary = analyze_systematic_bias_patterns(
                merged_df,
                person_set=person_set,
                category_col=cat_col,
                baseline_col=baseline_col,
                profile_prefix=profile_prefix
            )
            
            results[cat_col] = (bias_patterns, meaningful_patterns, category_summary)
            
            # Print summary for this category
            print(f"\nSUMMARY FOR {cat_col.upper()}:")
            print(f"  • Total bias patterns: {len(bias_patterns)}")
            print(f"  • Meaningful patterns: {len(meaningful_patterns)}")
            print(f"  • Categories analyzed: {bias_patterns['category'].nunique()}")
            print(f"  • Profiles with bias: {bias_patterns['profile'].nunique()}")
            
            if len(meaningful_patterns) > 0:
                top_bias = meaningful_patterns.iloc[0]
                print(f"  • Strongest bias: {top_bias['category']} | {top_bias['profile']} | "
                      f"{top_bias['bias_direction']} | mag={top_bias['bias_magnitude']:.3f}")
            
        except Exception as e:
            print(f"=== Error analyzing category '{cat_col}': {e}")
            continue
    
    print(f"\n{'='*80}")
    print("MULTI-CATEGORY ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully analyzed {len(results)} out of {len(case.category_cols)} categories")
    
    return results



def print_multi_category_bias_summary(bias_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]):
    """
    Print a comprehensive summary across all category columns with proper terminology
    """
    print("\n" + "="*80)
    print("CROSS-CATEGORY BIAS ANALYSIS SUMMARY")
    print("="*80)
    
    total_patterns = 0
    total_meaningful = 0
    category_stats = []
    
    for cat_col, (bias_patterns, meaningful_patterns, category_summary) in bias_results.items():
        total_patterns += len(bias_patterns)
        total_meaningful += len(meaningful_patterns)
        
        category_stats.append({
            'category_column': cat_col,
            'total_patterns': len(bias_patterns),
            'meaningful_patterns': len(meaningful_patterns),
            'meaningful_rate': len(meaningful_patterns) / len(bias_patterns) if len(bias_patterns) > 0 else 0,
            'max_bias_magnitude': bias_patterns['bias_magnitude'].max() if len(bias_patterns) > 0 else 0,
            'avg_bias_magnitude': bias_patterns['bias_magnitude'].mean() if len(bias_patterns) > 0 else 0,
            'unique_category_values': bias_patterns['category'].nunique() if len(bias_patterns) > 0 else 0
        })
    
    print(f"OVERALL STATISTICS:")
    print(f"  • Total category columns analyzed: {len(bias_results)}")
    print(f"  • Total bias patterns: {total_patterns:,}")
    print(f"  • Total meaningful patterns: {total_meaningful:,}")
    print(f"  • Overall meaningful rate: {total_meaningful/total_patterns:.1%}")
    
    print(f"\nPER-CATEGORY COLUMN BREAKDOWN:")
    print(f"{'Category Column':<20}{'Patterns':<10}{'Meaningful':<12}{'Rate':<8}{'Max Bias':<10}{'Avg Bias':<10}{'Values'}")
    print("-" * 85)
    
    for stats in sorted(category_stats, key=lambda x: x['meaningful_patterns'], reverse=True):
        print(f"{stats['category_column']:<20}{stats['total_patterns']:<10}{stats['meaningful_patterns']:<12}"
              f"{stats['meaningful_rate']:<8.1%}{stats['max_bias_magnitude']:<10.3f}"
              f"{stats['avg_bias_magnitude']:<10.3f}{stats['unique_category_values']}")
    
    # Find cross-category patterns - Fixed terminology
    print(f"\nCROSS-CATEGORY INSIGHTS:")
    
    # Most biased profiles across all category columns
    all_meaningful = pd.concat([
        meaningful.assign(source_category_column=cat) 
        for cat, (_, meaningful, _) in bias_results.items()
    ], ignore_index=True)
    
    if len(all_meaningful) > 0:
        profile_bias_counts = all_meaningful['profile'].value_counts()
        print(f"  • Most frequently biased profiles across category values:")
        for profile, count in profile_bias_counts.head(5).items():
            print(f"    - {profile}: appears in {count} category values")
        
        # Most problematic category columns
        if len(bias_results) > 1:
            category_bias_counts = all_meaningful['source_category_column'].value_counts()
            print(f"  • Category columns with most bias patterns:")
            for category, count in category_bias_counts.items():
                print(f"    - {category}: {count} meaningful bias patterns")
        else:
            # Single category column - show breakdown by category values
            category_value_counts = all_meaningful['category'].value_counts()
            print(f"  • Category values with most bias patterns in {list(bias_results.keys())[0]}:")
            for category_value, count in category_value_counts.items():
                print(f"    - {category_value}: {count} meaningful bias patterns")




def analyze_persona_similarity(merged: pd.DataFrame, person_set: PersonSet) -> Dict[str, Any]:
    """
    Enhanced persona clustering analysis with demographic mapping and validation
    """
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    if not profile_cols:
        return {"error": "No profile columns found"}

    print(f"Analyzing similarity patterns across {len(profile_cols)} personas...")

    # Map once
    trait_by_profile = {p: get_demographic_info(p, person_set) for p in profile_cols}

    n_profiles = len(profile_cols)
    distance_matrix = np.zeros((n_profiles, n_profiles))
    for i, p1 in enumerate(profile_cols):
        for j, p2 in enumerate(profile_cols):
            if i != j:
                distance_matrix[i, j] = np.mean(merged[p1] != merged[p2])

    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    n_ethnicities = len({traits.split("_")[0] for traits in trait_by_profile.values()})
    max_clusters = min(n_profiles - 1, n_ethnicities + 2)

    cluster_quality = {}
    
    for n_clust in range(2, max_clusters + 1):
        clusters = fcluster(linkage_matrix, t=n_clust, criterion='maxclust')
        
        if len(np.unique(clusters)) > 1:
            sil_score = silhouette_score(distance_matrix, clusters, metric='precomputed')
            cluster_quality[n_clust] = sil_score
    
    if cluster_quality:
        optimal_n_clusters = max(cluster_quality, key=cluster_quality.get)
        optimal_silhouette = cluster_quality[optimal_n_clusters]
    else:
        optimal_n_clusters = 3
        optimal_silhouette = 0.0
    
    clusters = fcluster(linkage_matrix, t=optimal_n_clusters, criterion='maxclust')
    
    cluster_analysis = {}
    demographic_distribution = {}
    
    for cluster_id in np.unique(clusters):
        cluster_profiles = [prof for prof, c in zip(profile_cols, clusters) if c == cluster_id]
        
        demo_composition = {}
        for prof in cluster_profiles:
            demo = get_demographic_info(prof, person_set)
            demo_composition[demo] = demo_composition.get(demo, 0) + 1
        
        if 'true_label' in merged.columns:
            avg_accuracy = np.mean([
                accuracy_score(merged["true_label"], merged[prof]) 
                for prof in cluster_profiles
            ])
        else:
            avg_accuracy = np.nan
        
        if len(cluster_profiles) > 1:
            internal_agreement = np.mean([
                np.mean(merged[p1] == merged[p2]) 
                for p1, p2 in combinations(cluster_profiles, 2)
            ])
        else:
            internal_agreement = 1.0
        
        if len(cluster_profiles) > 1:
            avg_agreements = []
            for prof in cluster_profiles:
                agreements = [np.mean(merged[prof] == merged[other]) 
                            for other in cluster_profiles if other != prof]
                avg_agreements.append(np.mean(agreements))
            centroid_idx = np.argmax(avg_agreements)
            centroid_profile = cluster_profiles[centroid_idx]
        else:
            centroid_profile = cluster_profiles[0]
        
        cluster_analysis[f"cluster_{cluster_id}"] = {
            "profiles": cluster_profiles,
            "size": len(cluster_profiles),
            "avg_accuracy": avg_accuracy,
            "internal_agreement": internal_agreement,
            "demographic_composition": demo_composition,
            "centroid_profile": centroid_profile,
            "dominant_demographic": max(demo_composition, key=demo_composition.get) if demo_composition else "unknown"
        }
        
        for demo, count in demo_composition.items():
            if demo not in demographic_distribution:
                demographic_distribution[demo] = []
            demographic_distribution[demo].extend([cluster_id] * count)
    

    demo_cluster_summary = {}
    for demo, cluster_assignments in demographic_distribution.items():
        demo_cluster_summary[demo] = {
            "primary_cluster": max(set(cluster_assignments), key=cluster_assignments.count),
            "cluster_distribution": {cid: cluster_assignments.count(cid) for cid in set(cluster_assignments)},
            "clustering_consistency": max(set(cluster_assignments), key=cluster_assignments.count) / len(cluster_assignments)
        }
    

    cluster_distances = {}
    for i, cluster_i in enumerate(np.unique(clusters)):
        for j, cluster_j in enumerate(np.unique(clusters)):
            if i < j:
                profiles_i = [prof for prof, c in zip(profile_cols, clusters) if c == cluster_i]
                profiles_j = [prof for prof, c in zip(profile_cols, clusters) if c == cluster_j]
                
                distances = []
                for pi in profiles_i:
                    for pj in profiles_j:
                        pi_idx = profile_cols.index(pi)
                        pj_idx = profile_cols.index(pj)
                        distances.append(distance_matrix[pi_idx, pj_idx])
                
                cluster_distances[f"cluster_{cluster_i}_vs_cluster_{cluster_j}"] = np.mean(distances)
    
    return {
        "clusters": cluster_analysis,
        "linkage_matrix": linkage_matrix,
        "distance_matrix": distance_matrix,
        "optimal_n_clusters": optimal_n_clusters,
        "optimal_silhouette_score": optimal_silhouette,
        "cluster_quality_scores": cluster_quality,
        "demographic_clustering": demo_cluster_summary,
        "inter_cluster_distances": cluster_distances,
        "summary": {
            "total_profiles": len(profile_cols),
            "n_clusters_found": len(np.unique(clusters)),
            "avg_cluster_size": np.mean([len(cluster_analysis[f"cluster_{cid}"]["profiles"]) for cid in np.unique(clusters)]),
            "most_cohesive_cluster": max(cluster_analysis, key=lambda x: cluster_analysis[x]["internal_agreement"]),
            "most_accurate_cluster": max(cluster_analysis, key=lambda x: cluster_analysis[x]["avg_accuracy"]) if not np.isnan(avg_accuracy) else None
        }
    }



def print_persona_similarity_analysis(similarity_results: Dict[str, Any]):
    """Print comprehensive persona similarity analysis"""

    if "error" in similarity_results:
        print(f"Error: {similarity_results['error']}")
        return

    print("\n" + "="*80)
    print("PERSONA SIMILARITY & CLUSTERING ANALYSIS")
    print("="*80)

    summary = similarity_results["summary"]
    clusters = similarity_results["clusters"]
    demo_clustering = similarity_results["demographic_clustering"]

    print(f"\nSUMMARY:")
    print(f"  • Total personas analyzed: {summary['total_profiles']}")
    print(f"  • Optimal number of clusters: {similarity_results['optimal_n_clusters']}")
    print(f"  • Clustering quality (silhouette): {similarity_results['optimal_silhouette_score']:.3f}")
    print(f"  • Average cluster size: {summary['avg_cluster_size']:.1f}")

    print(f"\nCLUSTER COMPOSITION:")
    for cluster_name, cluster_info in clusters.items():
        print(f"\n{cluster_name.upper()} ({cluster_info['size']} personas):")
        print(f"  • Dominant demographic: {cluster_info['dominant_demographic']}")
        print(f"  • Internal agreement: {cluster_info['internal_agreement']:.3f}")
        print(f"  • Average accuracy: {cluster_info['avg_accuracy']:.3f}")
        print(f"  • Centroid profile: {cluster_info['centroid_profile']}")
        print(f"  • Demographic breakdown: {cluster_info['demographic_composition']}")

    print(f"\nDEMOGRAPHIC CLUSTERING PATTERNS:")
    for demo, demo_info in demo_clustering.items():
        consistency = demo_info['clustering_consistency']
        primary_cluster = demo_info['primary_cluster']
        label = "High" if consistency > 0.8 else "Medium" if consistency > 0.6 else "Low"
        print(f"{demo}: {consistency:.1%} in cluster_{primary_cluster} ({label} consistency)")

    print(f"\nINTER-CLUSTER DISTANCES:")
    for comparison, distance in similarity_results["inter_cluster_distances"].items():
        print(f"{comparison}: {distance:.3f}")

    print(f"\nANALYSIS FINDINGS:")
    high_consistency_demos = [
        demo for demo, info in demo_clustering.items()
        if info['clustering_consistency'] > 0.8
    ]
    if high_consistency_demos:
        print(f"Strong demographic clustering observed: {', '.join(high_consistency_demos)}")
    else:
        print("Weak demographic clustering - personas group by factors other than demographics")

    score = similarity_results['optimal_silhouette_score']
    if score > 0.5:
        print("High clustering quality - distinct persona groups identified")
    elif score > 0.3:
        print("Moderate clustering quality - some persona groupings exist")
    else:
        print("Low clustering quality - personas show similar behavior patterns")

    most_cohesive = summary['most_cohesive_cluster']
    print(f"Most cohesive cluster: {most_cohesive} (agreement: {clusters[most_cohesive]['internal_agreement']:.3f})")





def plot_accuracy_deltas_with_ci(
    merged_df,
    person_set: PersonSet,
    group_keys=("gender", "ethnicity"),
    colormap: str = "tab10",
    figsize=(14, 6),
    savepath: str = None,
):
    """
    Accuracy plot (NeurIPS-friendly):
      • Far-left bar = baseline accuracy (base_pred if present else zero_shot)
      • Remaining bars = absolute accuracy per Ethnicity×Gender (mean over profiles) with 95% CI
      • Consensus line = global mean across all profiles
      • Y-axis is zoomed so the bottom starts at 80% of the lowest accuracy (baseline + groups)
    """
    try:
        apply_neurips_figure_style()
    except Exception:
        pass

    profile_cols = [c for c in merged_df.columns if c.startswith("profile")]
    if not profile_cols:
        raise ValueError("No profile columns found for plotting")

    # Global consensus across profiles
    consensus_accuracy_global = float(np.mean([
        (merged_df[p] == merged_df["true_label"]).mean()
        for p in profile_cols
    ]))

    # Baseline accuracy (leftmost bar)
    baseline_label = "Baseline"
    if "base_pred" in merged_df.columns:
        baseline_acc = float((merged_df["base_pred"] == merged_df["true_label"]).mean())
    elif "zero_shot" in merged_df.columns:
        baseline_acc = float((merged_df["zero_shot"] == merged_df["true_label"]).mean())
    else:
        baseline_acc = np.nan

    # Traits
    all_genders, all_ethnicities = set(), set()
    profile_demographics = {}
    for p in profile_cols:
        traits = person_set.get_traits(p, group_keys=["gender", "ethnicity"])
        gender = str(traits.get("gender", "unknown")).lower()
        ethnicity = str(traits.get("ethnicity", "unknown")).lower()
        all_genders.add(gender); all_ethnicities.add(ethnicity)
        profile_demographics[p] = (ethnicity, gender)

    genders = sorted(all_genders)
    ethnicities = sorted(all_ethnicities)

    # Ordered combinations (ethnicity × gender)
    ordered_combinations = [(e, g) for e in ethnicities for g in genders]

    # Group profiles by combo
    demo_groups = {combo: [] for combo in ordered_combinations}
    for p, (e, g) in profile_demographics.items():
        if (e, g) in demo_groups:
            demo_groups[(e, g)].append(p)

    # Accuracy & 95% CI per group (across profiles)
    group_stats = {}
    for combo in ordered_combinations:
        profs = demo_groups[combo]
        prof_accs = [(merged_df[p] == merged_df["true_label"]).mean()
                     for p in profs if p in merged_df.columns]
        n = len(prof_accs)
        if n == 0:
            mean_acc, ci = np.nan, 0.0
        elif n == 1:
            mean_acc, ci = float(prof_accs[0]), 0.0
        else:
            mean_acc = float(np.mean(prof_accs))
            ci = float(sem(prof_accs) * t.ppf(0.975, n - 1))
        group_stats[combo] = {"mean_acc": mean_acc, "ci": ci, "n": n}

    # Colors: same color per ethnicity
    cmap = plt.get_cmap(colormap)
    ethnicity_colors = {
        e: cmap(i / max(1, len(ethnicities) - 1)) for i, e in enumerate(ethnicities)
    }

    # Build arrays (baseline first if available)
    labels, means, ci_err, colors = [], [], [], []
    include_baseline = not np.isnan(baseline_acc)
    if include_baseline:
        labels.append(baseline_label)
        means.append(baseline_acc)
        ci_err.append(0.0)
        colors.append("0.5")  # grey baseline

    abbrev_gender = {"man": "M", "woman": "W", "nonbinary": "NB"}
    for e, g in ordered_combinations:
        st = group_stats[(e, g)]
        labels.append(f"{e.replace('_','-').title()}\n{abbrev_gender.get(g, g[:1].upper())}")
        means.append(st["mean_acc"])
        ci_err.append(st["ci"])
        colors.append(ethnicity_colors[e])

    means_arr = np.array(means, dtype=float)
    ci_arr = np.array(ci_err, dtype=float)

    # ---------------------------
    # Y-AXIS ZOOM (your request):
    # bottom starts at 80% of the lowest accuracy (baseline + groups)
    # ---------------------------
    finite_means = means_arr[np.isfinite(means_arr)]
    if finite_means.size == 0:
        raise ValueError("All group accuracies are NaN; cannot plot.")

    min_acc = float(np.min(finite_means))
    ymin = max(0.0, 0.9 * min_acc)
    ymax = min(1.0, float(np.nanmax(means_arr + ci_arr)) + 0.03)
    if ymin >= ymax:
        ymax = ymin + 0.05  # fallback to a small visible range

    # Plot
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    width = 0.65  # slightly thinner bars

    bars = ax.bar(
        x, means_arr, yerr=ci_arr, width=width, capsize=3,
        color=colors, edgecolor="black", linewidth=0.8, alpha=0.9, rasterized=True
    )

    # Vertical separators between ethnicities (skip baseline slot)
    if len(ethnicities) > 1:
        offset = 1 if include_baseline else 0
        for i in range(1, len(ethnicities)):
            pos = offset + i * len(genders) - 0.5
            ax.axvline(pos, color="0.6", linestyle=":", linewidth=1, alpha=0.8)

    # Consensus line
    ax.axhline(consensus_accuracy_global, color="0.2", linestyle="--", linewidth=1.2,
               label="Consensus accuracy")

    # Δ annotations (relative to consensus), above top + error
    EPS = 0.002
    for i, m in enumerate(means_arr):
        if not np.isfinite(m):
            continue
        ax.text(x[i], m + ci_arr[i] + EPS, f"{(m - consensus_accuracy_global):+0.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")

    ax.set_ylabel("Accuracy")
    ax.set_title("accuracy by ethnicity × gender (95% CI)", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_facecolor("white")
    ax.set_ylim(ymin, ymax)

    # Legend (ethnicity colors + baseline)
    legend_handles, legend_labels = [], []
    if include_baseline:
        legend_handles.append(plt.Rectangle((0,0),1,1,color="0.5",ec="black"))
        legend_labels.append("Baseline")
    for e in ethnicities:
        legend_handles.append(plt.Rectangle((0,0),1,1,color=ethnicity_colors[e],ec="black"))
        legend_labels.append(e.replace("_","-").title())
    ax.legend(legend_handles, legend_labels, title="Groups", loc="upper right", framealpha=0.9)

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")

    # Tidy summary
    rows = []
    if include_baseline:
        rows.append({
            "combination": "baseline",
            "ethnicity": None, "gender": None,
            "mean_accuracy": baseline_acc,
            "delta_from_consensus": baseline_acc - consensus_accuracy_global,
            "ci": 0.0, "n_profiles": None
        })
    for (e,g), st in group_stats.items():
        rows.append({
            "combination": f"{e}_{g}",
            "ethnicity": e, "gender": g,
            "mean_accuracy": st["mean_acc"],
            "delta_from_consensus": (st["mean_acc"] - consensus_accuracy_global) if np.isfinite(st["mean_acc"]) else np.nan,
            "ci": st["ci"], "n_profiles": st["n"]
        })
    return pd.DataFrame(rows)


def _profile_cols(df):
    return [c for c in df.columns if str(c).startswith("profile")]

def _traits_for_profile(person_set, profile, keys=("ethnicity", "gender")):
    t = person_set.get_traits(profile, group_keys=list(keys))
    return {k: ("" if t.get(k) is None else str(t[k]).lower()) for k in keys}

def _group_profiles_by_trait(merged_df, person_set, trait="ethnicity", min_profiles=1):
    groups = {}
    for p in _profile_cols(merged_df):
        tr = _traits_for_profile(person_set, p)
        g = tr.get(trait, "unknown") or "unknown"
        groups.setdefault(g, []).append(p)
    return {g: profs for g, profs in groups.items() if len(profs) >= min_profiles}

def _profile_accuracies(df, profiles):
    return np.array([(df[p] == df["true_label"]).mean() for p in profiles])

def _cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, (n1 + n2 - 2)))
    if sp == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / sp

def compute_pairwise_demographic_diffs(
    merged_df,
    person_set,
    trait="ethnicity",
    min_profiles=2,
    p_adjust="fdr_bh"
):
    """
    Returns a DataFrame with columns:
      group1, group2, mean1, mean2, diff, p, p_adj, d, n1, n2, neglog10_p, neglog10_p_adj
    """
    from scipy.stats import ttest_ind
    import pandas as pd

    groups = _group_profiles_by_trait(merged_df, person_set, trait=trait, min_profiles=min_profiles)
    keys = sorted(groups.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g1, g2 = keys[i], keys[j]
            a1 = _profile_accuracies(merged_df, groups[g1])
            a2 = _profile_accuracies(merged_df, groups[g2])
            if len(a1) < 2 or len(a2) < 2:
                continue
            _, p = ttest_ind(a1, a2, equal_var=False)
            diff = float(np.mean(a1) - np.mean(a2))
            d = float(_cohens_d(a1, a2))
            rows.append({
                "group1": g1, "group2": g2,
                "mean1": float(np.mean(a1)), "mean2": float(np.mean(a2)),
                "diff": diff, "p": float(p), "d": d,
                "n1": int(len(a1)), "n2": int(len(a2)),
            })
    out = pd.DataFrame(rows)
    if len(out):
        _, p_adj, _, _ = multipletests(out["p"].values, method=p_adjust)
        out["p_adj"] = p_adj
        out["neglog10_p"] = -np.log10(out["p"].clip(lower=1e-300))
        out["neglog10_p_adj"] = -np.log10(out["p_adj"].clip(lower=1e-300))
    return out.sort_values("diff", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)

# ---- Plot helpers: make each figure standalone at (10, 6) ----
from matplotlib import patheffects as pe

def plot_volcano_demographic_diffs(
    pair_df,
    ax=None,
    title="(a) Volcano — Difference vs −log10(p)",
    rank_by="diff",         
    top_n=5,
    alpha_sig=0.05,
    figsize=(10, 6),
    label_style="halo",  
    label_fontsize=9,
):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if pair_df is None or len(pair_df) == 0:
        ax.text(0.5, 0.5, "No pairwise stats", ha="center", va="center")
        return ax

    x = pair_df["diff"].values
    y = pair_df["neglog10_p"].values
    sig = (pair_df["p_adj"].values < alpha_sig) if "p_adj" in pair_df else (pair_df["p"].values < alpha_sig)

    ax.scatter(x[~sig], y[~sig], s=26, alpha=0.7, edgecolors="none", rasterized=True)
    ax.scatter(x[sig],  y[sig],  s=36, alpha=0.9, edgecolors="black", linewidths=0.4, rasterized=True, zorder=3)

    # annotate top-N by |Δ| or |d|
    rank_col = "d" if rank_by == "d" else "diff"
    top_idx = np.argsort(-np.abs(pair_df[rank_col].values))[:min(top_n, len(pair_df))]
    for idx in top_idx:
        row = pair_df.iloc[idx]
        txt = ax.annotate(
            f"{row['group1']} vs {row['group2']}",
            (row["diff"], row["neglog10_p"]),
            xytext=(6, 6), textcoords="offset points", fontsize=9
        )
        if label_style == "halo":
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        elif label_style == "box":
            txt.set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.6, alpha=0.95))
        # "none" => plain text

    ax.axvline(0, ls="--", lw=1, color="0.4")
    ax.axhline(-np.log10(alpha_sig), ls=":", lw=1, color="0.4")

    ax.set_xlabel("Δ accuracy (group1 − group2)")
    ax.set_ylabel("−log10(p)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.set_facecolor("white")
    ax.margins(x=0.05)
    return ax


def plot_effect_size_heatmap(
    pair_df,
    ax=None,
    title="(b) effect size — ethnicity × ethnicity",
    cmap="coolwarm",
    figsize=(10, 6),
):
    """Diverging heatmap of Cohen's d, centered at 0 (standalone figure)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if pair_df is None or len(pair_df) == 0:
        ax.text(0.5, 0.5, "No effect sizes", ha="center", va="center"); 
        return ax, 0.0

    groups = sorted(set(pair_df["group1"]).union(set(pair_df["group2"])))
    idx = {g: i for i, g in enumerate(groups)}
    M = np.zeros((len(groups), len(groups)))
    for _, r in pair_df.iterrows():
        i, j = idx[r["group1"]], idx[r["group2"]]
        M[i, j] = r["d"]
        M[j, i] = -r["d"]
    np.fill_diagonal(M, 0.0)

    vmax = max(1e-6, np.max(np.abs(M)))
    im = ax.imshow(M, vmin=-vmax, vmax=vmax, cmap=cmap, rasterized=True)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("_", "-").title() for g in groups], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g.replace("_", "-").title() for g in groups], fontsize=9)
    ax.set_title(title, fontsize=10)

    if M.size <= 30:
        for i in range(len(groups)):
            for j in range(len(groups)):
                if i == j: 
                    continue
                ax.text(j, i, f"{M[i, j]:+0.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cohen's d (signed)")
    ax.set_facecolor("white")
    return ax, vmax


def plot_intersectional_accuracy_heatmap(
    merged_df,
    person_set,
    ax=None,
    title="(c) intersectional — ethnicity × gender",
    normalize=True,    # Δ from global mean for diverging scale compatibility
    cmap="coolwarm",
    figsize=(10, 6),
):
    """Heatmap of accuracy or Δ-accuracy per (ethnicity, gender) (standalone figure)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    profiles = _profile_cols(merged_df)
    if not profiles:
        ax.text(0.5, 0.5, "No profiles", ha="center", va="center"); 
        return ax, 0.0

    eth_set, gen_set, by_combo = set(), set(), {}
    for p in profiles:
        tr = _traits_for_profile(person_set, p, keys=("ethnicity", "gender"))
        eth, gen = tr["ethnicity"], tr["gender"]
        eth_set.add(eth); gen_set.add(gen)
        by_combo.setdefault((eth, gen), []).append(p)

    eth_list = sorted(eth_set); gen_list = sorted(gen_set)
    A = np.full((len(eth_list), len(gen_list)), np.nan)
    per_prof_acc = {p: (merged_df[p] == merged_df["true_label"]).mean() for p in profiles}
    global_mean = float(np.mean(list(per_prof_acc.values())))

    for i, e in enumerate(eth_list):
        for j, g in enumerate(gen_list):
            profs = by_combo.get((e, g), [])
            if profs:
                A[i, j] = float(np.mean([per_prof_acc[p] for p in profs]))

    if normalize:
        A = A - global_mean
        vmax = np.nanmax(np.abs(A)) if np.isfinite(A).any() else 0.0
        im = ax.imshow(A, vmin=-vmax, vmax=vmax, cmap=cmap, rasterized=True)
    else:
        im = ax.imshow(A, vmin=np.nanmin(A), vmax=np.nanmax(A), cmap=cmap, rasterized=True)
        vmax = float(np.nanmax(np.abs(A - global_mean))) if np.isfinite(A).any() else 0.0

    ax.set_xticks(range(len(gen_list))); ax.set_xticklabels([g.title() for g in gen_list], fontsize=9)
    ax.set_yticks(range(len(eth_list))); ax.set_yticklabels([e.replace("_","-").title() for e in eth_list], fontsize=9)
    ax.set_title(title, fontsize=10)

    if np.isfinite(A).sum() <= 30:
        for i in range(len(eth_list)):
            for j in range(len(gen_list)):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:+0.3f}" if normalize else f"{A[i, j]:0.3f}",
                            ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ accuracy" if normalize else "Accuracy")
    ax.set_facecolor("white")
    return ax, vmax


def plot_demographic_accuracy_composite(
    merged_df,
    person_set,
    trait="ethnicity",
    top_n=5,
    normalize_intersectional=True,
    savepath=None,
    figsize=(15, 4.5),
    use_neurips_style=True,
):
    """
    Builds the 3-panel composite figure:
      (A) Volcano Δ vs −log10 p
      (B) Effect size heatmap (Cohen's d)
      (C) Intersectional accuracy (Ethnicity × Gender)
    Notes:
      • Calls NeurIPS rcParams when use_neurips_style=True (default).
      • No suptitle (caption goes below the figure in LaTeX).
    """
    if use_neurips_style:
        try:
            apply_neurips_figure_style()
        except Exception:
            # If the helper is missing, silently continue with defaults
            pass

    pair = compute_pairwise_demographic_diffs(merged_df, person_set, trait=trait, min_profiles=2)

    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # (A) Volcano
    plot_volcano_demographic_diffs(pair, ax=axs[0], top_n=top_n)

    # (B) Effect size heatmap
    _, vmax_d = plot_effect_size_heatmap(pair, ax=axs[1])

    # (C) Intersectional heatmap
    _, vmax_acc = plot_intersectional_accuracy_heatmap(
        merged_df, person_set, ax=axs[2], normalize=normalize_intersectional
    )

    # No suptitle; figure caption will be in LaTeX
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")  # PDF will embed fonts via rcParams

    return fig, {"pairwise": pair, "vmax_effect": vmax_d, "vmax_intersectional": vmax_acc}

def save_demographic_figures_individual(
    merged_df,
    person_set,
    out_dir,
    figsize=(10, 6),
    normalize_intersectional=True,
    top_n=5,
):
    os.makedirs(out_dir, exist_ok=True)

    # Pairwise table once
    pair = compute_pairwise_demographic_diffs(
        merged_df, person_set=person_set, trait="ethnicity", min_profiles=2
    )

    # Volcano
    fig_v, ax_v = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_volcano_demographic_diffs(pair, ax=ax_v, top_n=top_n, figsize=figsize, label_style="halo")
    fig_v.savefig(os.path.join(out_dir, "volcano_demographic.pdf"), bbox_inches="tight")
    plt.close(fig_v)

    # Effect size heatmap
    fig_h, ax_h = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_effect_size_heatmap(pair, ax=ax_h, figsize=figsize)
    fig_h.savefig(os.path.join(out_dir, "effect_size_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig_h)

    # Intersectional heatmap
    fig_i, ax_i = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_intersectional_accuracy_heatmap(
        merged_df, person_set, ax=ax_i, normalize=normalize_intersectional, figsize=figsize
    )
    fig_i.savefig(os.path.join(out_dir, "intersectional_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig_i)

    return {
        "volcano": os.path.join(out_dir, "volcano_demographic.pdf"),
        "effect_size": os.path.join(out_dir, "effect_size_heatmap.pdf"),
        "intersectional": os.path.join(out_dir, "intersectional_heatmap.pdf"),
    }


def run_full_preliminary_analysis(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    df: Optional[pd.DataFrame] = None,
    person_set: PersonSet = None,
    threshold_disagreement=0.3
) -> Dict[str, Any]:

    if person_set is None:
        raise ValueError("person_set is required for analysis")

    # NeurIPS style once
    try:
        apply_neurips_figure_style()
    except Exception:
        pass

    results = {}

    # Ensure baseline column
    if "base_pred" not in merged_df.columns:
        if "zero_shot" in merged_df.columns:
            merged_df["base_pred"] = merged_df["zero_shot"]
        else:
            raise ValueError("Need either base_pred or zero_shot in merged_df.")

    # Merge missing category cols if any
    if df is not None and "sample_id" in merged_df.columns and "sample_id" in df.columns:
        missing_cols = [col for col in case.category_cols if col not in merged_df.columns]
        if missing_cols:
            print(f"Merging missing category columns: {missing_cols}")
            merged_df = merged_df.merge(
                df[["sample_id"] + missing_cols],
                on="sample_id",
                how="left"
            )
        else:
            print("All category columns already present in merged_df")

    group_keys = get_analysis_group_keys(person_set)

    print("\n\n=== DEMOGRAPHIC ACCURACY DIFFERENCES ===")
    demographic_results = test_comprehensive_demographic_accuracy_differences(
        merged_df,
        person_set=person_set
    )
    print_comprehensive_demographic_results(demographic_results)
    results["demographic"] = demographic_results

    # SYSTEMATIC BIAS PATTERNS
    print("\n\n=== SYSTEMATIC BIAS PATTERNS (AUTOMATIQUE) ===")
    bias_results = guarded_labelspace_analysis(
        analyze_systematic_bias_patterns_multi_category,
        merged_df,
        case,
        person_set=person_set
    )
    bias_patterns_all = {}
    meaningful_patterns_all = {}
    category_summary_all = {}
    for cat_col, (bias_patterns, meaningful_patterns, category_summary) in bias_results.items():
        bias_patterns_all[cat_col] = bias_patterns
        meaningful_patterns_all[cat_col] = meaningful_patterns
        category_summary_all[cat_col] = category_summary
        print(f"Found {len(bias_patterns)} bias patterns, {len(meaningful_patterns)} meaningful for {cat_col}")
    print_multi_category_bias_summary(bias_results)
    results["bias_patterns"] = bias_patterns_all
    results["meaningful_bias_patterns"] = meaningful_patterns_all
    results["category_summary"] = category_summary_all

    print("\n\n=== HIGH DISAGREEMENT CASES ===")
    disagreement_df = extract_high_disagreement_cases(
        merged_df,
        threshold=threshold_disagreement,
        person_set=person_set
    )
    print(f"Found {len(disagreement_df)} high disagreement cases")
    results["disagreement"] = disagreement_df

    print("\n\n=== RESCUE STATISTICS BY CATEGORY ===")
    rescue_stats_all = {}
    rescue_analysis_all = {}
    for cat_col in case.category_cols:
        if cat_col in merged_df.columns:
            print(f"\nAnalyzing rescue statistics for category column: {cat_col}")
            rescue_df = guarded_labelspace_analysis(
                rescue_stats_by_category,
                merged_df,
                case=case,
                category_col=cat_col
            )
            rescue_analysis = analyze_rescue_performance(rescue_df)
            rescue_stats_all[cat_col] = rescue_df
            rescue_analysis_all[cat_col] = rescue_analysis
            print(f"Computed rescue stats for {cat_col}: {len(rescue_df)} records")
        else:
            print(f"Warning: Category column '{cat_col}' not found for rescue analysis")
    print_all_rescue_analyses(rescue_analysis_all)
    results["rescue_stats"] = rescue_stats_all
    results["rescue_analysis"] = rescue_analysis_all

    print("\n\n=== PERSONA SIMILARITY CLUSTERING ===")
    persona_similarity = analyze_persona_similarity(merged_df, person_set=person_set)
    print_persona_similarity_analysis(persona_similarity)
    results["persona_similarity"] = persona_similarity

    # ---------- PLOTTING (NeurIPS-ready PDFs) ----------
    print("\n\n=== FIGURES (NEURIPS) ===")
    fig_dir = os.path.join("results", "figs", case.case_name)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Accuracy with CI — already supports figsize and y-zoom
    try:
        acc_summary = plot_accuracy_deltas_with_ci(
            merged_df,
            person_set=person_set,
            group_keys=group_keys,
            figsize=(10, 6),
            savepath=os.path.join(fig_dir, "accuracy_by_group.pdf"),
        )
        results["accuracy_summary"] = acc_summary
        print("Saved:", os.path.join(fig_dir, "accuracy_by_group.pdf"))
    except Exception as e:
        print(f"Error generating accuracy plot: {e}")
        results["accuracy_summary"] = None
    
    # Individual demographic figures (no composite)
    try:
        paths = save_demographic_figures_individual(
            merged_df,
            person_set=person_set,
            out_dir=fig_dir,
            figsize=(10, 6),
            normalize_intersectional=True,
            top_n=5,
        )
        results["figure_paths"] = {"accuracy_by_group": os.path.join(fig_dir, "accuracy_by_group.pdf"), **paths}
        print("Saved:", paths)
    except Exception as e:
        print(f"Error saving individual demographic figures: {e}")

    # ---------- SUMMARY ----------
    print(f"\n\n=== ANALYSIS SUMMARY ===")
    print(f"Dataset: {case.case_name}")
    print(f"Category columns analyzed: {[col for col in case.category_cols if col in merged_df.columns]}")
    print(f"Missing category columns: {[col for col in case.category_cols if col not in merged_df.columns]}")
    print(f"Total profiles: {len([c for c in merged_df.columns if c.startswith('profile')])}")
    print(f"Group keys used: {group_keys}")

    return results
