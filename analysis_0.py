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

from analysis_tools import get_demographic_info
from profiles.schema import *
from profiles.profile_sets import PERSON_ETHNICS
from cases import CaseConfig


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
    "run_full_preliminary_analysis"
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
            print(f"⚠️ Skipping comparison between {group1_name} and {group2_name}: No data")
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
    profile_prefix: str = "profile"
) -> pd.DataFrame:
    """
    Analyze rescue statistics for each category and profile combination.
    
    This function computes comprehensive rescue metrics for each profile within each
    category, measuring how often profiles correct baseline errors versus introducing
    new errors. Essential for understanding profile-specific performance patterns
    across different data segments.
    
    Parameters:
    -----------
    merged : pd.DataFrame
        DataFrame containing predictions, true labels, and category information
    category_col : str
        Column name containing category labels for analysis segmentation
    baseline_col : str, default="base_pred"
        Column name containing baseline model predictions
    label_col : str, default="true_label"
        Column name containing ground truth labels
    profile_prefix : str, default="profile"
        Prefix for identifying profile prediction columns
    
    Returns:
    --------
    pd.DataFrame
        Long-format DataFrame with rescue statistics for each (category, profile) pair.
        Columns include:
        - category: Category identifier
        - profile: Profile identifier
        - N_cat: Total samples in category
        - rescued: Count of baseline errors corrected by profile
        - rescue_rate: Proportion of baseline errors rescued
        - extra_errors: Count of new errors introduced by profile
        - extra_err_rate: Proportion of baseline correct predictions made incorrect
        - profile_acc: Profile accuracy within category
        - baseline_acc: Baseline accuracy within category
    
    Raises:
    -------
    ValueError
        If category_col is not found in the DataFrame
    """
    
    if category_col not in merged.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame")

    if baseline_col not in merged.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in DataFrame")

    if label_col not in merged.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # Standardize labels
    y_true = merged[label_col].astype(str).str.strip().str.lower()
    y_base = merged[baseline_col].astype(str).str.strip().str.lower()

    # Get profile columns
    profile_cols: List[str] = [c for c in merged.columns if c.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No profile columns found with prefix '{profile_prefix}'")

    output_records = []

    for category_value, category_df in merged.groupby(category_col):
        category_size = len(category_df)

        y_true_cat = y_true.loc[category_df.index]
        y_base_cat = y_base.loc[category_df.index]

        base_correct_mask = y_base_cat == y_true_cat
        base_err_count = (~base_correct_mask).sum()
        base_ok_count = base_correct_mask.sum()
        base_acc = base_ok_count/category_size

        for profile in profile_cols:
            y_prof_cat = category_df[profile].astype(str).str.strip().str.lower()
            prof_correct_mask = y_prof_cat == y_true_cat

            rescued = ((~base_correct_mask) & prof_correct_mask).sum()
            extra_errors = (base_correct_mask & (~prof_correct_mask)).sum()

            rescue_rate = rescued / base_err_count if base_err_count > 0 else 0.0
            extra_err_rate = extra_errors / base_ok_count if base_ok_count > 0 else 0.0
            prof_acc = prof_correct_mask.mean()

            output_records.append({
                'category': category_value,
                'profile': profile,
                'N_cat': category_size,
                'rescued': int(rescued),
                'rescue_rate': rescue_rate,
                'extra_errors': int(extra_errors),
                'extra_err_rate': extra_err_rate,
                'profile_acc': prof_acc,
                'baseline_acc': base_acc
            })

    return pd.DataFrame(output_records).sort_values(["category", "rescued"], ascending=[True, False])


def analyze_rescue_performance(rescue_stats_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive analysis of rescue statistics performance.
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

    # Category-level aggregation
    category_stats = rescue_stats_df.groupby('category').agg({
        'rescue_rate': ['mean', 'std', 'max'],
        'extra_err_rate': ['mean', 'std', 'max'],
        'profile_acc': ['mean', 'std'],
        'N_cat': 'first'
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



def detect_systematic_biases(
    merged: pd.DataFrame,
    person_set: PersonSet,
    category_col: str = "stereotype_type",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile",
    positive_label: str = "stereotype",
    negative_label: str = "unrelated"
) -> pd.DataFrame:

    required_cols = [category_col, baseline_col]
    missing_cols = [c for c in required_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    profile_cols = [c for c in merged.columns if c.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No columns found with prefix '{profile_prefix}'")

    # Map profiles to demographics once
    trait_by_profile = {p: get_demographic_info(p, person_set) for p in profile_cols}

    bias_patterns = []
    global_size = len(merged)

    for category_value in merged[category_col].dropna().unique():
        category_data = merged[merged[category_col] == category_value]
        baseline = category_data[baseline_col]
        size = len(category_data)

        for profile in profile_cols:
            profile_preds = category_data[profile]
            pos_mislabelling = ((baseline == negative_label) & (profile_preds == positive_label)).sum()
            neg_mislabelling = ((baseline == positive_label) & (profile_preds == negative_label)).sum()

            total_mislabelling = pos_mislabelling + neg_mislabelling
            mislabelling = total_mislabelling / size

            if pos_mislabelling > neg_mislabelling:
                bias_direction = "more_positive"
                bias_magnitude = (pos_mislabelling - neg_mislabelling) / size
            elif neg_mislabelling > pos_mislabelling:
                bias_direction = "more_negative"
                bias_magnitude = (neg_mislabelling - pos_mislabelling) / size
            else:
                bias_direction = "neutral"
                bias_magnitude = 0.0

            weighted_bias_magnitude = bias_magnitude * (size / global_size)

            bias_patterns.append({
                "category": category_value,
                "profile": profile,
                "demographic": trait_by_profile[profile],
                "bias_direction": bias_direction,
                "bias_magnitude": bias_magnitude,
                "weighted_bias_magnitude": weighted_bias_magnitude,
                "n_mislabelling": total_mislabelling,
                "mislabelling_rate": mislabelling,
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

    # Run bias detection
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
        'total_mislabelling', 'avg_mislabelling_rate', 'sample_size',
        'total_weighted_bias'
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
    color_key: str = "ethnicity",
    colormap: str = "tab10",
):
    def norm_val(v):
        if hasattr(v, "value"):  # Enum
            return str(v.value).lower()
        return "unknown" if v is None else str(v).lower()

    profile_cols = [c for c in merged_df.columns if c.startswith("profile")]
    if not profile_cols:
        raise ValueError("No profile columns found for plotting")
    
    trait_by_profile = {p: person_set.get_traits(p, group_keys) for p in profile_cols}
    
    consensus_accuracy_global = np.mean([
        (merged_df[p] == merged_df["true_label"]).mean()
        for p in profile_cols
    ])

    demo_groups = {}
    for p in profile_cols:
        traits = trait_by_profile[p]
        gname = " ".join([str(traits.get(k, "unknown")).lower() for k in group_keys])
        demo_groups.setdefault(gname, []).append(p)
    
    print(f"DEBUG: Found demographic groups: {dict(demo_groups)}")

    genders = sorted({norm_val(getattr(m, "gender", None)) for m in person_set.metadata.values()})
    ethnicities = sorted({norm_val(getattr(m, "ethnicity", None)) for m in person_set.metadata.values()})

    genders = [g for g in genders if g != "unknown"] if len(genders) > 1 else genders
    ethnicities = [e for e in ethnicities if e != "unknown"] if len(ethnicities) > 1 else ethnicities


    groups = [f"{g} {e}" for e in ethnicities for g in genders]  
    
    print(f"DEBUG: Expected groups: {groups}")


    for g in groups:
        if g not in demo_groups:
            demo_groups[g] = []

    group_data = {}
    for g in groups:  # Only iterate over valid groups
        profs = demo_groups[g] if g in demo_groups else []
        accs = [(merged_df[p] == merged_df["true_label"]).mean() for p in profs if p in merged_df.columns]
        
        # Calculate consensus from ALL OTHER profiles (not in this demographic group)
        others = [p for other_g, other_profs in demo_groups.items() 
                 if other_g != g for p in other_profs if p in merged_df.columns]
        
        if others:
            consensus = np.mean([(merged_df[p] == merged_df["true_label"]).mean() for p in others])
        else:
            consensus = consensus_accuracy_global
        
        deltas = [a - consensus for a in accs]
        n = len(deltas)
        se = sem(deltas) if n > 1 else 0.0
        
        # Split group name back into traits - maintain order from group_keys
        traits_dict = {}
        parts = g.split(" ")
        for i, key in enumerate(group_keys):
            traits_dict[key] = parts[i] if i < len(parts) else "unknown"
        
        group_data[g] = {
            "mean_delta": float(np.mean(deltas)) if n else 0.0,
            "sem": float(se),
            "n": n,
            "traits": traits_dict,
        }
        
    print(f"DEBUG: Group data keys: {list(group_data.keys())}")  # Debug line

    color_key = "ethnicity"  # Back to ethnicity for colors
    cmap = plt.get_cmap(colormap)
    trait_values = ethnicities or ["unknown"]
    trait_to_color = {val: cmap(i / max(1, len(trait_values)-1)) for i, val in enumerate(trait_values)}

    means = [group_data[g]["mean_delta"] for g in groups]
    errors = [
        group_data[g]["sem"] * t.ppf(0.975, max(1, group_data[g]["n"] - 1)) if group_data[g]["n"] > 1 else 0.0
        for g in groups
    ]
    colors = [trait_to_color.get(group_data[g]["traits"].get(color_key, "unknown"), 'gray') for g in groups]

    means = [group_data[g]["mean_delta"] for g in groups]
    errors = [
        group_data[g]["sem"] * t.ppf(0.975, max(1, group_data[g]["n"] - 1)) if group_data[g]["n"] > 1 else 0.0
        for g in groups
    ]
    colors = [trait_to_color.get(group_data[g]["traits"].get(color_key, "unknown"), 'gray') for g in groups]

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, means, yerr=errors, capsize=4, color=colors, edgecolor="black", alpha=0.9)

    abbrev_gender = {"man": "M", "woman": "W"}
    xticklabels = []
    for g in groups:
        parts = g.split(" ")
        if len(parts) >= 2:
            gdr, eth = parts[0], parts[1]  # gender, ethnicity
            gdr_short = abbrev_gender.get(gdr, gdr[:1].upper())
            xticklabels.append(f"{gdr_short}·{eth}")
        else:
            xticklabels.append(g)
    
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)

    for i in range(1, len(ethnicities)):
        cut = i * len(genders) - 0.5
        ax.axvline(cut, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Consensus baseline")
    if "base_pred" in merged_df.columns:
        base_acc = (merged_df["base_pred"] == merged_df["true_label"]).mean()
        ax.axhline(base_acc - consensus_accuracy_global, color="purple", linestyle=":", linewidth=2, label="No-roleplaying baseline")

    for i, g in enumerate(groups):
        if group_data[g]["n"] > 0:
            m, e = means[i], errors[i]
            y = m + (e if m >= 0 else -e)
            va = "bottom" if m >= 0 else "top"
            ax.text(i, y + (0.0005 if m >= 0 else -0.0005), f"{m:+.3f}", ha="center", va=va, fontsize=8, fontweight="bold")

    ax.set_ylabel("Accuracy Delta from Consensus")
    ax.set_title("Group Accuracy Deviations from Consensus (95% CI)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.subplots_adjust(bottom=0.28)
    plt.show()

    summary = []
    for g in groups:
        data = group_data[g]
        ci = data["sem"] * t.ppf(0.975, max(1, data["n"] - 1)) if data["n"] > 1 else 0.0
        summary.append({
            "group": g,
            "mean_delta": data["mean_delta"],
            "ci_lower": data["mean_delta"] - ci,
            "ci_upper": data["mean_delta"] + ci,
            "n": data["n"],
            "gender": data["traits"].get("gender", "unknown"),
            "ethnicity": data["traits"].get("ethnicity", "unknown"),
        })
    return pd.DataFrame(summary).sort_values("group")



def run_full_preliminary_analysis(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    df: Optional[pd.DataFrame] = None,
    person_set: PersonSet = PERSON_ETHNICS,
    threshold_disagreement=0.3
) -> Dict[str, Any]:
    results = {}

    if "base_pred" not in merged_df.columns:
        if "zero_shot" in merged_df.columns:
            merged_df["base_pred"] = merged_df["zero_shot"]
        else:
            raise ValueError("Need either base_pred or zero_shot in merged_df.")

    if df is not None and "sample_id" in merged_df.columns and "sample_id" in df.columns:
        missing_cols = [col for col in case.category_cols if col not in merged_df.columns]
        if missing_cols:
            merged_df = merged_df.merge(
                df[["sample_id"] + missing_cols],
                on="sample_id",
                how="left"
            )

    if "stereotype_type" not in merged_df.columns and df is not None:
        if "sample_id" in merged_df.columns and "sample_id" in df.columns:
            merged_df = merged_df.merge(
                df[["sample_id", "stereotype_type"]],
                on="sample_id",
                how="left"
            )

    base_keys = ["gender", "ethnicity"]
    optional_keys = []

    trait_seen = {"cognitive_style": False, "age": False}
    for meta in person_set.metadata.values():
        if getattr(meta, "cognitive_style", None) is not None:
            trait_seen["cognitive_style"] = True
        if getattr(meta, "age", None) is not None:
            trait_seen["age"] = True
        if all(trait_seen.values()):
            break
    if trait_seen["cognitive_style"]:
        optional_keys.append("cognitive_style")
    if trait_seen["age"]:
        optional_keys.append("age")

    group_keys = tuple(base_keys + optional_keys)


    print("\n\n=== DEMOGRAPHIC ACCURACY DIFFERENCES ===")
    demographic_results = test_comprehensive_demographic_accuracy_differences(
        merged_df,
        person_set=person_set
    )
    print_comprehensive_demographic_results(demographic_results)
    results["demographic"] = demographic_results

    print("\n\n=== SYSTEMATIC BIAS PATTERNS ===")
    bias_patterns, meaningful_patterns, category_summary = analyze_systematic_bias_patterns(
        merged_df,
        person_set=person_set
    )
    print(bias_patterns.head(20))
    results["bias_patterns"] = bias_patterns
    results["meaningful_bias_patterns"] = meaningful_patterns
    results["category_summary"] = category_summary

    print("\n\n=== HIGH DISAGREEMENT CASES ===")
    disagreement_df = extract_high_disagreement_cases(
        merged_df,
        threshold=threshold_disagreement,
        person_set=person_set     
    )
    print(disagreement_df.head(10))
    results["disagreement"] = disagreement_df

    print("\n\n=== RESCUE STATISTICS BY CATEGORY ===")
    rescue_stats_all = {}
    rescue_analysis_all = {}
    for cat_col in case.category_cols:
        if cat_col in merged_df.columns:
            rescue_df = rescue_stats_by_category(merged_df, category_col=cat_col)
            rescue_analysis = analyze_rescue_performance(rescue_df)
            rescue_stats_all[cat_col] = rescue_df
            rescue_analysis_all[cat_col] = rescue_analysis
            print(f"\n-- {cat_col} --")
            example_cat = rescue_df["category"].unique()[0]
            print(rescue_df[rescue_df["category"] == example_cat].head(10))

    results["rescue_stats"] = rescue_stats_all
    results["rescue_analysis"] = rescue_analysis_all

    print("\n\n=== PERSONA SIMILARITY CLUSTERING ===")
    persona_similarity = analyze_persona_similarity(merged_df, person_set=person_set)
    print_persona_similarity_analysis(persona_similarity)
    results["persona_similarity"] = persona_similarity

    print("\n=== PLOT OF ACCURACY WITH CI ===")
    color_key = "ethnicity" if "ethnicity" in group_keys else group_keys[-1]
    delta_summary = plot_accuracy_deltas_with_ci(
        merged_df,
        person_set=person_set,
        group_keys=group_keys,      
        color_key=color_key
    )
    results["accuracy_delta_summary"] = delta_summary

    return results