import os
import glob
import json
from collections import Counter
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
    bootstrap, f_oneway, ttest_ind, kruskal, mannwhitneyu, pearsonr
)

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# ============================================================================
# Preliminary Analysis
# ============================================================================


def test_comprehensive_demographic_accuracy_differences(merged_df):
    """
    Comprehensive analysis of demographic group accuracy differences.
    
    Provides foundational evidence of classification patterns across demographic
    groups before running advanced statistical models. Tests for systematic
    differences in accuracy across gender, ethnicity, cognitive styles, and
    intersectional categories.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        DataFrame containing profile columns and true_label column
    
    Returns:
    --------
    dict : Dictionary containing statistical test results for all group comparisons
    """
    
    results = {}

    # Profile definitions by demographic groups
    MEN_PROFILES = [f"profile{i}_passive" for i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]]
    WOMEN_PROFILES = [f"profile{i}_passive" for i in [6,7,8,9,10, 16,17,18,19,20, 26,27,28,29,30]]
    
    WHITE_PROFILES = [f"profile{i}_passive" for i in range(1, 11)]
    BLACK_PROFILES = [f"profile{i}_passive" for i in range(11, 21)]
    ASIAN_PROFILES = [f"profile{i}_passive" for i in range(21, 31)]
    
    # Cognitive style profiles
    EXPANSIVE_PROFILES = [f"profile{i}_passive" for i in [1,6,11,16,21,26]]        
    LITERAL_PROFILES = [f"profile{i}_passive" for i in [2,7,12,17,22,27]]         
    HIGH_HARM_PROFILES = [f"profile{i}_passive" for i in [3,8,13,18,23,28]]        
    LOW_HARM_PROFILES = [f"profile{i}_passive" for i in [4,9,14,19,24,29]]       
    BALANCED_PROFILES = [f"profile{i}_passive" for i in [5,10,15,20,25,30]]
    
    # Intersectional groups
    WHITE_MEN = [f"profile{i}_passive" for i in range(1, 6)]
    WHITE_WOMEN = [f"profile{i}_passive" for i in range(6, 11)]
    BLACK_MEN = [f"profile{i}_passive" for i in range(11, 16)]
    BLACK_WOMEN = [f"profile{i}_passive" for i in range(16, 21)]
    ASIAN_MEN = [f"profile{i}_passive" for i in range(21, 26)]
    ASIAN_WOMEN = [f"profile{i}_passive" for i in range(26, 31)]

    def calculate_group_accuracy(profiles, df):
        """Calculate accuracy metrics for a group of profiles."""
        accuracies = []
        for profile in profiles:
            if profile in df.columns:
                acc = (df[profile] == df['true_label']).mean()
                accuracies.append(acc)
        return accuracies

    def compare_groups(group1_profiles, group2_profiles, group1_name, group2_name, df):
        """Perform statistical comparison between two demographic groups."""
        acc1 = calculate_group_accuracy(group1_profiles, df)
        acc2 = calculate_group_accuracy(group2_profiles, df)
        
        if acc1 and acc2:
            t_stat, p_val = ttest_ind(acc1, acc2)
            pooled_std = np.sqrt((np.var(acc1) + np.var(acc2)) / 2)
            effect_size = (np.mean(acc1) - np.mean(acc2)) / pooled_std if pooled_std > 0 else 0
            
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
        return None

    # Gender comparison
    gender_result = compare_groups(MEN_PROFILES, WOMEN_PROFILES, 'men', 'women', merged_df)
    if gender_result:
        results['men_vs_women'] = gender_result

    # Ethnicity comparisons
    ethnicity_pairs = [
        (WHITE_PROFILES, BLACK_PROFILES, 'white', 'black'),
        (WHITE_PROFILES, ASIAN_PROFILES, 'white', 'asian'),
        (BLACK_PROFILES, ASIAN_PROFILES, 'black', 'asian')
    ]
    
    for profiles1, profiles2, name1, name2 in ethnicity_pairs:
        result = compare_groups(profiles1, profiles2, name1, name2, merged_df)
        if result:
            results[f'{name1}_vs_{name2}'] = result

    # Cognitive style comparisons
    cognitive_pairs = [
        (EXPANSIVE_PROFILES, LITERAL_PROFILES, 'expansive', 'literal'),
        (HIGH_HARM_PROFILES, LOW_HARM_PROFILES, 'high_harm_sensitivity', 'low_harm_sensitivity'),
        (EXPANSIVE_PROFILES, BALANCED_PROFILES, 'expansive', 'balanced'),
        (LITERAL_PROFILES, BALANCED_PROFILES, 'literal', 'balanced'),
        (HIGH_HARM_PROFILES, BALANCED_PROFILES, 'high_harm_sensitivity', 'balanced'),
        (LOW_HARM_PROFILES, BALANCED_PROFILES, 'low_harm_sensitivity', 'balanced')
    ]
    
    for profiles1, profiles2, name1, name2 in cognitive_pairs:
        result = compare_groups(profiles1, profiles2, name1, name2, merged_df)
        if result:
            results[f'cognitive_{name1}_vs_{name2}'] = result

    # Intersectional comparisons
    intersectional_groups = [
        (WHITE_MEN, 'white_men'),
        (WHITE_WOMEN, 'white_women'),
        (BLACK_MEN, 'black_men'),
        (BLACK_WOMEN, 'black_women'),
        (ASIAN_MEN, 'asian_men'),
        (ASIAN_WOMEN, 'asian_women')
    ]

    for i, (profiles1, name1) in enumerate(intersectional_groups):
        for profiles2, name2 in intersectional_groups[i+1:]:
            result = compare_groups(profiles1, profiles2, name1, name2, merged_df)
            if result:
                results[f'intersectional_{name1}_vs_{name2}'] = result

    # Majority vs minority group analysis
    NON_WHITE_PROFILES = BLACK_PROFILES + ASIAN_PROFILES
    result = compare_groups(WHITE_PROFILES, NON_WHITE_PROFILES, 'white', 'non_white', merged_df)
    if result:
        results['white_vs_non_white'] = result
    
    # White men vs everyone else
    EVERYONE_ELSE = list(set(WOMEN_PROFILES + BLACK_PROFILES + ASIAN_PROFILES))
    result = compare_groups(WHITE_MEN, EVERYONE_ELSE, 'white_men', 'everyone_else', merged_df)
    if result:
        results['white_men_vs_everyone_else'] = result

    # Category-specific analysis (if stereotype_type exists)
    if 'stereotype_type' in merged_df.columns:
        categories = merged_df['stereotype_type'].unique()
        
        for category in categories:
            if category is not None:
                category_df = merged_df[merged_df['stereotype_type'] == category]
                
                if len(category_df) > 5:
                    # Gender differences within category
                    result = compare_groups(MEN_PROFILES, WOMEN_PROFILES, 'men', 'women', category_df)
                    if result:
                        results[f'category_{category}_men_vs_women'] = result
                    
                    # Ethnicity differences within category
                    result = compare_groups(WHITE_PROFILES, NON_WHITE_PROFILES, 'white', 'non_white', category_df)
                    if result:
                        results[f'category_{category}_white_vs_non_white'] = result

    # Summary statistics
    significant_comparisons = sum(1 for result in results.values() if result.get('significant', False))
    total_comparisons = len(results)
    
    # Effect size analysis
    effect_sizes = [(key, result.get('effect_size', 0)) for key, result in results.items() if 'effect_size' in result]
    effect_sizes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Group performance summary
    group_accuracies = {}
    all_groups = [
        (MEN_PROFILES, 'men'),
        (WOMEN_PROFILES, 'women'),
        (WHITE_PROFILES, 'white'),
        (BLACK_PROFILES, 'black'),
        (ASIAN_PROFILES, 'asian'),
        (WHITE_MEN, 'white_men'),
        (BLACK_WOMEN, 'black_women')
    ]
    
    for profiles, name in all_groups:
        accuracies = calculate_group_accuracy(profiles, merged_df)
        if accuracies:
            group_accuracies[name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'n_profiles': len(accuracies)
            }
    
    results['summary'] = {
        'total_comparisons': total_comparisons,
        'significant_comparisons': significant_comparisons,
        'significance_rate': significant_comparisons / total_comparisons if total_comparisons > 0 else 0,
        'largest_effects': effect_sizes[:5],
        'group_performance_summary': group_accuracies
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
        
        # Effect size ranking
        print(f"\nLARGEST EFFECT SIZES")
        print("-" * 40)
        for i, (comparison, effect_size) in enumerate(summary['largest_effects'][:5], 1):
            magnitude = "Large" if abs(effect_size) > 0.8 else "Medium" if abs(effect_size) > 0.5 else "Small"
            print(f"{i:2d}. {comparison:<35} Effect Size: {effect_size:6.3f} ({magnitude})")
        
        # Group performance ranking
        print(f"\nGROUP PERFORMANCE RANKING")
        print("-" * 40)
        group_perf = summary['group_performance_summary']
        sorted_groups = sorted(group_perf.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
        print(f"{'Rank':<6}{'Group':<15}{'Accuracy':<12}{'Std Dev':<10}{'N Profiles'}")
        print("-" * 50)
        for i, (group, stats) in enumerate(sorted_groups, 1):
            print(f"{i:<6}{group:<15}{stats['mean_accuracy']:<12.3f}{stats['std_accuracy']:<10.3f}{stats['n_profiles']}")
    
    print(f"\n" + "="*80)
    print("DETAILED STATISTICAL RESULTS")
    print("="*80)
    
    def format_comparison_result(test_name, result):
        """Format individual comparison results for clean display."""
        sig_indicator = "[SIGNIFICANT]" if result['significant'] else "[NON-SIG]    "
        
        # Extract group names and accuracies dynamically
        groups = test_name.replace('intersectional_', '').replace('cognitive_', '').replace('category_', '').split('_vs_')
        if len(groups) == 2:
            group1, group2 = groups
            acc1_key = f'{group1}_accuracy'
            acc2_key = f'{group2}_accuracy'
            
            group1_acc = result.get(acc1_key, 'N/A')
            group2_acc = result.get(acc2_key, 'N/A')
            
            if isinstance(group1_acc, (int, float)) and isinstance(group2_acc, (int, float)):
                acc_display = f"{group1.replace('_', ' ').title()}: {group1_acc:.3f} | {group2.replace('_', ' ').title()}: {group2_acc:.3f}"
            else:
                acc_display = "Accuracy data unavailable"
        else:
            acc_display = "Complex comparison"
        
        return f"{sig_indicator} {test_name:<35} | {acc_display}"
    
    # Gender Comparisons
    print(f"\nGENDER COMPARISONS")
    print("-" * 60)
    gender_tests = [k for k in results.keys() if 'men_vs_women' in k]
    if not gender_tests:
        print("No gender comparisons available")
    else:
        for test in gender_tests:
            result = results[test]
            print(format_comparison_result(test, result))
            print(f"         Difference: {result['difference']:+.3f} | p-value: {result['p_value']:.3f} | Effect Size: {result.get('effect_size', 0):.3f}")
    
    # Ethnicity Comparisons  
    print(f"\nETHNICITY COMPARISONS")
    print("-" * 60)
    ethnicity_tests = [k for k in results.keys() if any(eth in k for eth in ['white_vs_black', 'white_vs_asian', 'black_vs_asian'])]
    if not ethnicity_tests:
        print("No ethnicity comparisons available")
    else:
        for test in ethnicity_tests:
            result = results[test]
            print(format_comparison_result(test, result))
            print(f"         Difference: {result['difference']:+.3f} | p-value: {result['p_value']:.3f} | Effect Size: {result.get('effect_size', 0):.3f}")
    
    # Cognitive Style Comparisons
    print(f"\nCOGNITIVE STYLE COMPARISONS")
    print("-" * 60)
    cognitive_tests = [k for k in results.keys() if k.startswith('cognitive_')]
    if not cognitive_tests:
        print("No cognitive style comparisons available")
    else:
        for test in cognitive_tests:
            result = results[test]
            clean_test = test.replace('cognitive_', '')
            print(format_comparison_result(clean_test, result))
            print(f"         Difference: {result['difference']:+.3f} | p-value: {result['p_value']:.3f} | Effect Size: {result.get('effect_size', 0):.3f}")
    
    # Significant Intersectional Results
    print(f"\nSIGNIFICANT INTERSECTIONAL COMPARISONS")
    print("-" * 60)
    intersectional_tests = [k for k in results.keys() if k.startswith('intersectional_') and results[k]['significant']]
    if not intersectional_tests:
        print("No significant intersectional differences detected")
    else:
        for test in intersectional_tests[:10]:  # Show top 10
            result = results[test]
            clean_test = test.replace('intersectional_', '')
            print(f"[SIGNIFICANT] {clean_test}")
            print(f"         Difference: {result['difference']:+.3f} | p-value: {result['p_value']:.3f} | Effect Size: {result.get('effect_size', 0):.3f}")
    
    # Majority/Minority Analysis
    print(f"\nMAJORITY/MINORITY GROUP ANALYSIS")
    print("-" * 60)
    majority_tests = [k for k in results.keys() if any(x in k for x in ['white_vs_non_white', 'white_men_vs_everyone'])]
    if not majority_tests:
        print("No majority/minority comparisons available")
    else:
        for test in majority_tests:
            result = results[test]
            print(format_comparison_result(test, result))
            print(f"         Difference: {result['difference']:+.3f} | p-value: {result['p_value']:.3f} | Effect Size: {result.get('effect_size', 0):.3f}")
    
    # Category-Specific Results
    category_tests = [k for k in results.keys() if k.startswith('category_') and results[k]['significant']]
    if category_tests:
        print(f"\nCATEGORY-SPECIFIC SIGNIFICANT PATTERNS")
        print("-" * 60)
        for test in category_tests:
            result = results[test]
            clean_test = test.replace('category_', '')
            print(f"[SIGNIFICANT] {clean_test}")
            print(f"         Difference: {result['difference']:+.3f} | p-value: {result['p_value']:.3f}")
    
    # Key Insights and Interpretation
    print(f"\n" + "="*80)
    print("ANALYSIS INTERPRETATION")
    print("="*80)
    
    sig_results = {k: v for k, v in results.items() if v.get('significant', False) and k != 'summary'}
    total_tests = len([k for k in results.keys() if k != 'summary'])
    
    print(f"\nBIAS ASSESSMENT")
    print("-" * 40)
    if len(sig_results) > total_tests * 0.3:
        bias_level = "HIGH"
        recommendation = "Immediate bias mitigation required across multiple demographic dimensions"
    elif len(sig_results) > total_tests * 0.1:
        bias_level = "MODERATE"
        recommendation = "Targeted bias mitigation needed for specific group comparisons"
    else:
        bias_level = "LOW"
        recommendation = "System demonstrates good demographic balance with minimal bias"
    
    print(f"Bias Level: {bias_level}")
    print(f"Recommendation: {recommendation}")
    
    # Effect size interpretation
    if 'summary' in results and results['summary']['largest_effects']:
        largest_effect = results['summary']['largest_effects'][0]
        print(f"\nLARGEST BIAS DETECTED")
        print("-" * 40)
        print(f"Comparison: {largest_effect[0]}")
        print(f"Effect Size: {largest_effect[1]:.3f}")
        
        magnitude = "substantial" if abs(largest_effect[1]) > 0.8 else "moderate" if abs(largest_effect[1]) > 0.5 else "small"
        print(f"Magnitude: {magnitude.title()} practical significance")
    
    # Pattern analysis
    cognitive_sig = len([k for k in sig_results if k.startswith('cognitive_')])
    demographic_sig = len([k for k in sig_results if not k.startswith('cognitive_') and not k.startswith('category_')])
    
    print(f"\nPATTERN ANALYSIS")
    print("-" * 40)
    if cognitive_sig > demographic_sig:
        print("Primary variation source: Cognitive processing styles")
        print("Implication: Individual differences in reasoning approach drive accuracy variations")
    elif demographic_sig > cognitive_sig:
        print("Primary variation source: Demographic characteristics")
        print("Implication: Systematic demographic bias present in classification system")
    else:
        print("Balanced variation: Both cognitive and demographic factors contribute equally")
    
    return sig_results







def extract_high_disagreement_cases(
    merged: pd.DataFrame, 
    threshold: float = 0.7,
    sample_id_col: str = "sample_id",
    label_col: str = "true_label",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile"
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
    profile_prefix : str, default="profile"
        Prefix for identifying profile prediction columns
    
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
    
    # Input validation
    required_cols = [sample_id_col, label_col, baseline_col]
    missing_cols = [col for col in required_cols if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Identify profile columns
    profile_cols = [col for col in merged.columns if col.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No columns found with prefix '{profile_prefix}'")
    
    print(f"Analyzing disagreement across {len(profile_cols)} profiles...")
    print(f"Using disagreement threshold: {threshold}")
    
    disagreement_records = []
    
    # Analyze each sample
    for idx, row in merged.iterrows():
        # Extract predictions from all profiles
        profile_predictions = [row[col] for col in profile_cols]
        
        # Calculate prediction distribution
        prediction_counts = Counter(profile_predictions)
        total_profiles = len(profile_predictions)
        
        # Identify modal prediction (most common)
        modal_prediction = prediction_counts.most_common(1)[0][0]
        modal_count = prediction_counts[modal_prediction]
        
        # Calculate disagreement metrics
        disagreement_score = (total_profiles - modal_count) / total_profiles
        consensus_strength = modal_count / total_profiles
        
        # Calculate prediction entropy for information-theoretic disagreement measure
        probabilities = [count / total_profiles for count in prediction_counts.values()]
        prediction_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        
        # Identify minority predictions
        minority_predictions = [pred for pred, count in prediction_counts.items() if pred != modal_prediction]
        
        # Compile record
        record = {
            'sample_id': row[sample_id_col],
            'disagreement_score': disagreement_score,
            'prediction_distribution': dict(prediction_counts),
            'modal_prediction': modal_prediction,
            'minority_predictions': minority_predictions,
            'consensus_strength': consensus_strength,
            'prediction_entropy': prediction_entropy,
            'true_label': row[label_col],
            'base_pred': row[baseline_col],
            'total_profiles': total_profiles,
            'modal_count': modal_count,
            'minority_count': total_profiles - modal_count
        }
        disagreement_records.append(record)
    
    # Create DataFrame and filter for high disagreement
    disagreement_df = pd.DataFrame(disagreement_records)
    high_disagreement_cases = disagreement_df[disagreement_df['disagreement_score'] > threshold].copy()
    
    # Sort by disagreement score (highest first)
    high_disagreement_cases = high_disagreement_cases.sort_values(
        'disagreement_score', 
        ascending=False
    ).reset_index(drop=True)
    
    # Add analysis metadata
    total_samples = len(disagreement_df)
    high_disagreement_count = len(high_disagreement_cases)
    
    print(f"\nDISAGREEMENT ANALYSIS SUMMARY")
    print("-" * 50)
    print(f"Total samples analyzed: {total_samples:,}")
    print(f"High disagreement cases (>{threshold}): {high_disagreement_count:,}")
    print(f"High disagreement rate: {high_disagreement_count/total_samples:.1%}")
    
    if high_disagreement_count > 0:
        print(f"Average disagreement score: {high_disagreement_cases['disagreement_score'].mean():.3f}")
        print(f"Maximum disagreement score: {high_disagreement_cases['disagreement_score'].max():.3f}")
        print(f"Average prediction entropy: {high_disagreement_cases['prediction_entropy'].mean():.3f}")
    
    return high_disagreement_cases


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
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total high disagreement cases: {len(high_disagreement_df):,}")
    print(f"Average disagreement score: {high_disagreement_df['disagreement_score'].mean():.3f}")
    print(f"Standard deviation: {high_disagreement_df['disagreement_score'].std():.3f}")
    print(f"Range: {high_disagreement_df['disagreement_score'].min():.3f} - {high_disagreement_df['disagreement_score'].max():.3f}")
    print(f"Average prediction entropy: {high_disagreement_df['prediction_entropy'].mean():.3f}")
    
    # Top disagreement cases
    print(f"\nTOP {top_n} HIGHEST DISAGREEMENT CASES")
    print("-" * 60)
    print(f"{'Rank':<6}{'Sample ID':<15}{'Disagreement':<13}{'Entropy':<10}{'Modal Pred':<12}{'True Label'}")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(high_disagreement_df.head(top_n).iterrows(), 1):
        print(f"{idx:<6}{str(row['sample_id']):<15}{row['disagreement_score']:<13.3f}"
              f"{row['prediction_entropy']:<10.3f}{str(row['modal_prediction']):<12}{str(row['true_label'])}")
    
    # Prediction distribution analysis
    print(f"\nPREDICTION DISTRIBUTION PATTERNS")
    print("-" * 50)
    
    # Analyze common prediction patterns
    all_distributions = high_disagreement_df['prediction_distribution'].tolist()
    pattern_counts = Counter()
    
    for dist in all_distributions:
        # Convert to sorted tuple for pattern matching
        pattern = tuple(sorted(dist.items()))
        pattern_counts[pattern] += 1
    
    print("Most common disagreement patterns:")
    for pattern, count in pattern_counts.most_common(5):
        pattern_str = ", ".join([f"{pred}: {cnt}" for pred, cnt in pattern])
        print(f"  {pattern_str} (appears {count} times)")
    
    # Accuracy analysis
    print(f"\nACCURACY ANALYSIS FOR HIGH DISAGREEMENT CASES")
    print("-" * 50)
    
    # Compare modal prediction accuracy vs true labels
    modal_correct = high_disagreement_df['modal_prediction'] == high_disagreement_df['true_label']
    baseline_correct = high_disagreement_df['base_pred'] == high_disagreement_df['true_label']
    
    print(f"Modal prediction accuracy: {modal_correct.mean():.3f}")
    print(f"Baseline prediction accuracy: {baseline_correct.mean():.3f}")
    print(f"Cases where modal prediction is correct: {modal_correct.sum()} / {len(high_disagreement_df)}")
    print(f"Cases where baseline is correct: {baseline_correct.sum()} / {len(high_disagreement_df)}")
    
    # Consensus strength analysis
    print(f"\nCONSENSUS STRENGTH DISTRIBUTION")
    print("-" * 40)
    consensus_bins = pd.cut(high_disagreement_df['consensus_strength'], 
                           bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0], 
                           labels=['Very Low (≤30%)', 'Low (30-40%)', 'Medium (40-50%)', 
                                  'High (50-60%)', 'Very High (>60%)'])
    
    consensus_dist = consensus_bins.value_counts().sort_index()
    for category, count in consensus_dist.items():
        percentage = count / len(high_disagreement_df) * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%)")











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
    
    # Input validation
    if category_col not in merged.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame")
    
    if baseline_col not in merged.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in DataFrame")
    
    if label_col not in merged.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")
    
    # Standardize labels for consistent comparison
    y_true = merged[label_col].astype(str).str.strip().str.lower()
    y_base = merged[baseline_col].astype(str).str.strip().str.lower()
    
    # Identify profile columns
    profile_cols: List[str] = [c for c in merged.columns if c.startswith(profile_prefix)]
    
    if not profile_cols:
        raise ValueError(f"No columns found with prefix '{profile_prefix}'")
    
    output_records = []
    
    # Process each category
    for category_value, category_df in merged.groupby(category_col):
        category_size = len(category_df)
        
        # Extract category-specific true labels and baseline predictions
        y_true_category = y_true.loc[category_df.index]
        y_base_category = y_base.loc[category_df.index]
        
        # Calculate baseline performance metrics
        baseline_correct_mask = (y_base_category == y_true_category)
        baseline_errors_count = (~baseline_correct_mask).sum()
        baseline_correct_count = baseline_correct_mask.sum()
        baseline_accuracy = baseline_correct_count / category_size
        
        # Analyze each profile within this category
        for profile_name in profile_cols:
            # Standardize profile predictions
            y_profile_category = category_df[profile_name].astype(str).str.strip().str.lower()
            profile_correct_mask = (y_profile_category == y_true_category)
            
            # Calculate rescue metrics
            rescued_errors = ((~baseline_correct_mask) & profile_correct_mask).sum()
            additional_errors = (baseline_correct_mask & (~profile_correct_mask)).sum()
            
            # Calculate rates (handle division by zero)
            rescue_rate = rescued_errors / baseline_errors_count if baseline_errors_count > 0 else 0.0
            additional_error_rate = additional_errors / baseline_correct_count if baseline_correct_count > 0 else 0.0
            
            # Profile accuracy within category
            profile_accuracy = profile_correct_mask.mean()
            
            # Compile results
            record = {
                'category': category_value,
                'profile': profile_name,
                'N_cat': category_size,
                'rescued': int(rescued_errors),
                'rescue_rate': rescue_rate,
                'extra_errors': int(additional_errors),
                'extra_err_rate': additional_error_rate,
                'profile_acc': profile_accuracy,
                'baseline_acc': baseline_accuracy
            }
            output_records.append(record)
    
    # Create and sort results DataFrame
    results_df = pd.DataFrame(output_records)
    results_df = results_df.sort_values(["category", "rescued"], ascending=[True, False])
    
    return results_df


def analyze_rescue_performance(rescue_stats_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive analysis of rescue statistics performance.
    
    Parameters:
    -----------
    rescue_stats_df : pd.DataFrame
        Output from rescue_stats_by_category function
    
    Returns:
    --------
    Dict[str, Any]
        Comprehensive analysis including top performers, category insights, and summary metrics
    """
    
    if rescue_stats_df.empty:
        return {"error": "Empty rescue statistics DataFrame provided"}
    
    analysis = {}
    
    # Overall performance metrics
    analysis['summary'] = {
        'total_categories': rescue_stats_df['category'].nunique(),
        'total_profiles': rescue_stats_df['profile'].nunique(),
        'total_samples': rescue_stats_df['N_cat'].sum(),
        'total_rescues': rescue_stats_df['rescued'].sum(),
        'total_extra_errors': rescue_stats_df['extra_errors'].sum(),
        'avg_rescue_rate': rescue_stats_df['rescue_rate'].mean(),
        'avg_extra_error_rate': rescue_stats_df['extra_err_rate'].mean()
    }
    
    # Top performing profiles by rescue rate
    analysis['top_rescue_performers'] = (
        rescue_stats_df.nlargest(10, 'rescue_rate')[
            ['profile', 'category', 'rescue_rate', 'rescued', 'profile_acc']
        ].to_dict('records')
    )
    
    # Profiles with highest extra error rates (potential problems)
    analysis['highest_error_risk'] = (
        rescue_stats_df.nlargest(10, 'extra_err_rate')[
            ['profile', 'category', 'extra_err_rate', 'extra_errors', 'profile_acc']
        ].to_dict('records')
    )
    
    # Category-level analysis
    category_stats = rescue_stats_df.groupby('category').agg({
        'rescue_rate': ['mean', 'std', 'max'],
        'extra_err_rate': ['mean', 'std', 'max'],
        'profile_acc': ['mean', 'std'],
        'N_cat': 'first'
    }).round(3)
    
    analysis['category_performance'] = category_stats.to_dict('index')
    
    # Profile-level analysis
    profile_stats = rescue_stats_df.groupby('profile').agg({
        'rescue_rate': ['mean', 'std'],
        'extra_err_rate': ['mean', 'std'],
        'profile_acc': ['mean', 'std'],
        'rescued': 'sum',
        'extra_errors': 'sum'
    }).round(3)
    
    analysis['profile_performance'] = profile_stats.to_dict('index')
    
    return analysis



### NORMALIZE PER CATEGORY

def detect_systematic_biases(
    merged: pd.DataFrame, 
    category_col: str = "stereotype_type",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile",
    positive_label: str = "stereotype",
    negative_label: str = "unrelated"
) -> pd.DataFrame:
    """
    Identify systematic biases where profiles consistently deviate from baseline predictions.
    
    Analyzes directional prediction shifts across categories to detect patterns where
    specific profiles systematically favor certain prediction outcomes relative to
    the baseline model. Essential for understanding systematic bias patterns in
    demographic profile predictions.
    
    Parameters:
    -----------
    merged : pd.DataFrame
        DataFrame containing baseline predictions, profile predictions, and categories
    category_col : str, default="stereotype_type"
        Column name containing category labels for bias analysis
    baseline_col : str, default="base_pred"
        Column name containing baseline model predictions
    profile_prefix : str, default="profile"
        Prefix for identifying profile prediction columns
    positive_label : str, default="stereotype"
        Label representing positive class predictions
    negative_label : str, default="unrelated"
        Label representing negative class predictions
    
    Returns:
    --------
    pd.DataFrame
        Bias patterns DataFrame with columns:
        - category: Category identifier
        - profile: Profile identifier  
        - bias_direction: Direction of bias ("more_positive" or "more_negative")
        - bias_magnitude: Magnitude of directional bias (proportion of category)
        - n_flips: Total number of prediction flips from baseline
        - flip_rate: Proportion of predictions that differ from baseline
        - category_size: Number of samples in category
        - positive_flips: Baseline negative -> Profile positive flips
        - negative_flips: Baseline positive -> Profile negative flips
    
    Raises:
    -------
    ValueError
        If required columns are missing or no profile columns found
    """
    
    # Input validation
    required_cols = [category_col, baseline_col]
    missing_cols = [col for col in required_cols if col not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Identify profile columns
    profile_cols = [col for col in merged.columns if col.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No columns found with prefix '{profile_prefix}'")
    
    bias_patterns = []
    
    # Analyze each category
    for category_value in merged[category_col].unique():
        if pd.isna(category_value):
            continue
            
        category_data = merged[merged[category_col] == category_value]
        baseline_predictions = category_data[baseline_col]
        category_size = len(category_data)
        
        if category_size == 0:
            continue
        
        # Analyze each profile within this category
        for profile_name in profile_cols:
            profile_predictions = category_data[profile_name]
            
            # Calculate directional bias shifts
            positive_flips = (
                (baseline_predictions == negative_label) & 
                (profile_predictions == positive_label)
            ).sum()
            
            negative_flips = (
                (baseline_predictions == positive_label) & 
                (profile_predictions == negative_label)
            ).sum()
            
            total_flips = positive_flips + negative_flips
            flip_rate = total_flips / category_size
            
            # Determine bias direction and magnitude
            if positive_flips > negative_flips:
                bias_direction = "more_positive"
                bias_magnitude = (positive_flips - negative_flips) / category_size
            elif negative_flips > positive_flips:
                bias_direction = "more_negative"
                bias_magnitude = (negative_flips - positive_flips) / category_size
            else:
                bias_direction = "neutral"
                bias_magnitude = 0.0
            
            # Compile bias pattern record
            pattern_record = {
                "category": category_value,
                "profile": profile_name,
                "bias_direction": bias_direction,
                "bias_magnitude": bias_magnitude,
                "n_flips": total_flips,
                "flip_rate": flip_rate,
                "category_size": category_size,
                "positive_flips": positive_flips,
                "negative_flips": negative_flips
            }
            bias_patterns.append(pattern_record)
    
    # Create and sort results DataFrame
    bias_df = pd.DataFrame(bias_patterns)
    bias_df = bias_df.sort_values("bias_magnitude", ascending=False, key=abs)
    
    return bias_df


def analyze_systematic_bias_patterns(
    merged_df: pd.DataFrame,
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
    
    # Execute bias detection
    bias_patterns = detect_systematic_biases(
        merged_df, 
        category_col=category_col,
        baseline_col=baseline_col,
        profile_prefix=profile_prefix
    )
    
    # Dataset composition analysis
    category_sizes = merged_df.groupby(category_col).size().reset_index()
    category_sizes.columns = ['category', 'sample_size']
    total_samples = len(merged_df)
    category_sizes['percentage'] = (category_sizes['sample_size'] / total_samples * 100).round(1)
    
    print(f"\nDATASET COMPOSITION BY CATEGORY")
    print("-" * 40)
    print(f"{'Category':<20}{'Samples':<10}{'Percentage'}")
    print("-" * 40)
    for _, row in category_sizes.sort_values('sample_size', ascending=False).iterrows():
        print(f"{str(row['category']):<20}{row['sample_size']:<10}{row['percentage']:.1f}%")
    
    # Analysis summary
    print(f"\nBIAS DETECTION RESULTS SUMMARY")
    print("-" * 40)
    print(f"Total bias patterns analyzed: {len(bias_patterns):,}")
    print(f"Unique categories: {bias_patterns['category'].nunique()}")
    print(f"Unique profiles: {bias_patterns['profile'].nunique()}")
    print(f"Average category size: {bias_patterns['category_size'].mean():.1f} samples")
    
    # Top bias patterns
    print(f"\nTOP 20 STRONGEST BIAS PATTERNS")
    print("-" * 90)
    print(f"{'Category':<15}{'Profile':<20}{'Direction':<15}{'Magnitude':<12}{'Flips':<8}{'Cat.Size'}")
    print("-" * 90)
    
    for _, row in bias_patterns.head(20).iterrows():
        print(f"{str(row['category']):<15}{str(row['profile']):<20}{row['bias_direction']:<15}"
              f"{row['bias_magnitude']:<12.3f}{row['n_flips']:<8}{row['category_size']}")
    
    # Reliability analysis by sample size
    print(f"\nRELIABILITY ANALYSIS BY SAMPLE SIZE")
    print("-" * 50)
    
    large_categories = bias_patterns[bias_patterns['category_size'] >= 100]
    medium_categories = bias_patterns[(bias_patterns['category_size'] >= 50) & (bias_patterns['category_size'] < 100)]
    small_categories = bias_patterns[bias_patterns['category_size'] < 50]
    
    print(f"Large categories (≥100 samples): {large_categories['category'].nunique()} categories")
    print(f"Medium categories (50-99 samples): {medium_categories['category'].nunique()} categories")  
    print(f"Small categories (<50 samples): {small_categories['category'].nunique()} categories")
    
    # High-reliability patterns
    if len(large_categories) > 0:
        print(f"\nHIGH-RELIABILITY BIAS PATTERNS (≥100 samples)")
        print("-" * 80)
        print(f"{'Category':<15}{'Profile':<20}{'Direction':<15}{'Magnitude':<12}{'Flips'}")
        print("-" * 80)
        
        for _, row in large_categories.head(15).iterrows():
            print(f"{str(row['category']):<15}{str(row['profile']):<20}{row['bias_direction']:<15}"
                  f"{row['bias_magnitude']:<12.3f}{row['n_flips']}")
    
    # Category-level aggregated analysis
    print(f"\nCATEGORY-LEVEL BIAS SUMMARY")
    print("-" * 70)
    
    category_summary = bias_patterns.groupby('category').agg({
        'bias_magnitude': ['mean', 'max', 'std'],
        'n_flips': 'sum',
        'flip_rate': 'mean',
        'category_size': 'first'
    }).round(3)
    
    category_summary.columns = ['avg_bias_magnitude', 'max_bias_magnitude', 'bias_std', 
                               'total_flips', 'avg_flip_rate', 'sample_size']
    
    print(f"{'Category':<15}{'Avg Bias':<12}{'Max Bias':<12}{'Total Flips':<12}{'Sample Size'}")
    print("-" * 70)
    
    for category in category_summary.sort_values('max_bias_magnitude', ascending=False).index:
        row = category_summary.loc[category]
        print(f"{str(category):<15}{row['avg_bias_magnitude']:<12.3f}{row['max_bias_magnitude']:<12.3f}"
              f"{row['total_flips']:<12.0f}{row['sample_size']:<12.0f}")
    
    # Statistical significance assessment
    print(f"\nSTATISTICALLY MEANINGFUL BIAS PATTERNS")
    print("-" * 70)
    
    def assess_bias_significance(row):
        """Apply sample-size adjusted significance thresholds."""
        if row['category_size'] >= 100:
            return abs(row['bias_magnitude']) >= 0.02  # 2% threshold for large samples
        elif row['category_size'] >= 50:
            return abs(row['bias_magnitude']) >= 0.03  # 3% threshold for medium samples
        else:
            return abs(row['bias_magnitude']) >= 0.05  # 5% threshold for small samples
    
    bias_patterns['statistically_meaningful'] = bias_patterns.apply(assess_bias_significance, axis=1)
    meaningful_patterns = bias_patterns[bias_patterns['statistically_meaningful']].copy()
    
    print(f"Identified {len(meaningful_patterns)} statistically meaningful bias patterns")
    print(f"Significance thresholds applied:")
    print(f"  Large categories (≥100 samples): ≥2.0% bias magnitude")
    print(f"  Medium categories (50-99 samples): ≥3.0% bias magnitude") 
    print(f"  Small categories (<50 samples): ≥5.0% bias magnitude")
    
    if len(meaningful_patterns) > 0:
        print(f"\nTOP 15 STATISTICALLY MEANINGFUL PATTERNS")
        print("-" * 90)
        print(f"{'Category':<15}{'Profile':<20}{'Direction':<15}{'Magnitude':<12}{'Size':<8}{'Threshold'}")
        print("-" * 90)
        
        for _, row in meaningful_patterns.head(15).iterrows():
            if row['category_size'] >= 100:
                threshold = "≥2.0%"
            elif row['category_size'] >= 50:
                threshold = "≥3.0%"
            else:
                threshold = "≥5.0%"
                
            print(f"{str(row['category']):<15}{str(row['profile']):<20}{row['bias_direction']:<15}"
                  f"{row['bias_magnitude']:<12.3f}{row['category_size']:<8}{threshold}")
    
    # Statistical insights
    print(f"\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*80)
    
    # Sample size distribution impact
    small_cat_count = len(small_categories['category'].unique())
    if small_cat_count > 0:
        print(f"Sample Size Impact: {small_cat_count} categories have <50 samples")
        print(f"                   Bias estimates for small categories may be unreliable")
    
    # Primary findings
    if len(meaningful_patterns) > 0:
        strongest_pattern = meaningful_patterns.iloc[0]
        print(f"\nStrongest Reliable Bias:")
        print(f"  Category: {strongest_pattern['category']}")
        print(f"  Profile: {strongest_pattern['profile']}")
        print(f"  Magnitude: {strongest_pattern['bias_magnitude']:.3f}")
        print(f"  Direction: {strongest_pattern['bias_direction']}")
        print(f"  Sample Size: {strongest_pattern['category_size']}")
    
    # Bias direction analysis
    if len(meaningful_patterns) > 0:
        direction_analysis = meaningful_patterns['bias_direction'].value_counts()
        print(f"\nBias Direction Distribution:")
        for direction, count in direction_analysis.items():
            percentage = count / len(meaningful_patterns) * 100
            print(f"  {direction}: {count} patterns ({percentage:.1f}%)")
    
    # Category-level insights
    reliable_categories = large_categories['category'].unique()
    if len(reliable_categories) > 0:
        reliable_summary = (large_categories.groupby('category')['bias_magnitude']
                          .apply(lambda x: x.abs().mean())
                          .sort_values(ascending=False))
        print(f"\nMost Biased Categories (High Reliability):")
        for i, (category, avg_bias) in enumerate(reliable_summary.head(5).items(), 1):
            print(f"  {i}. {category}: {avg_bias:.3f} average bias magnitude")
    
    # Final assessment
    print(f"\nFINAL ASSESSMENT")
    print("-" * 40)
    if len(meaningful_patterns) > 0:
        affected_categories = meaningful_patterns['category'].nunique()
        affected_profiles = meaningful_patterns['profile'].nunique()
        
        print(f"Systematic bias patterns confirmed:")
        print(f"  {len(meaningful_patterns)} statistically meaningful patterns detected")
        print(f"  {affected_categories} categories show systematic bias")
        print(f"  {affected_profiles} profiles exhibit biased behavior")
        print(f"  Sample size considerations applied for statistical rigor")
    else:
        print(f"No statistically meaningful bias patterns detected")
        print(f"Analysis controlled for sample size effects")
        print(f"Results suggest minimal systematic bias presence")
    
    return bias_patterns, meaningful_patterns, category_summary



def analyze_persona_similarity(merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhanced persona clustering analysis with demographic mapping and validation
    """
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    
    if len(profile_cols) == 0:
        return {"error": "No profile columns found"}
    
    print(f"Analyzing similarity patterns across {len(profile_cols)} personas...")
    
    n_profiles = len(profile_cols)
    distance_matrix = np.zeros((n_profiles, n_profiles))
    
    for i, prof1 in enumerate(profile_cols):
        for j, prof2 in enumerate(profile_cols):
            if i != j:
                distance_matrix[i, j] = np.mean(merged[prof1] != merged[prof2])
    
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    max_clusters = min(8, n_profiles - 1)
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
    
    def get_demographic_info(profile_name):
        """Extract demographic info from profile name"""
        
        profile_num = int(''.join(filter(str.isdigit, profile_name)))
        
        if 1 <= profile_num <= 5:
            return "white_men"
        elif 6 <= profile_num <= 10:
            return "white_women"
        elif 11 <= profile_num <= 15:
            return "black_men"
        elif 16 <= profile_num <= 20:
            return "black_women"
        elif 21 <= profile_num <= 25:
            return "asian_men"
        elif 26 <= profile_num <= 30:
            return "asian_women"
        else:
            return "unknown"
    
    cluster_analysis = {}
    demographic_distribution = {}
    
    for cluster_id in np.unique(clusters):
        cluster_profiles = [prof for prof, c in zip(profile_cols, clusters) if c == cluster_id]
        
        demo_composition = {}
        for prof in cluster_profiles:
            demo = get_demographic_info(prof)
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
    print("-" * 60)
    for cluster_name, cluster_info in clusters.items():
        print(f"\n{cluster_name.upper()} ({cluster_info['size']} personas):")
        print(f"  • Dominant demographic: {cluster_info['dominant_demographic']}")
        print(f"  • Internal agreement: {cluster_info['internal_agreement']:.3f}")
        print(f"  • Average accuracy: {cluster_info['avg_accuracy']:.3f}")
        print(f"  • Centroid profile: {cluster_info['centroid_profile']}")
        print(f"  • Demographic breakdown: {cluster_info['demographic_composition']}")
    
    # Demographic clustering patterns
    print(f"\nDEMOGRAPHIC CLUSTERING PATTERNS:")
    print("-" * 50)
    for demo, demo_info in demo_clustering.items():
        consistency = demo_info['clustering_consistency']
        primary_cluster = demo_info['primary_cluster']
        consistency_label = "High" if consistency > 0.8 else "Medium" if consistency > 0.6 else "Low"
        
        print(f"{demo}: {consistency:.1%} in cluster_{primary_cluster} ({consistency_label} consistency)")
    

    print(f"\nINTER-CLUSTER DISTANCES:")
    print("-" * 40)
    for comparison, distance in similarity_results["inter_cluster_distances"].items():
        print(f"{comparison}: {distance:.3f}")
    
    print(f"\nANALYSIS FINDINGS:")
    print("-" * 30)
    
    high_consistency_demos = [demo for demo, info in demo_clustering.items() 
                            if info['clustering_consistency'] > 0.8]
    
    if high_consistency_demos:
        print(f"Strong demographic clustering observed: {', '.join(high_consistency_demos)}")
    else:
        print(f"Weak demographic clustering - personas group by factors other than demographics")
    
    if similarity_results['optimal_silhouette_score'] > 0.5:
        print(f"High clustering quality - distinct persona groups identified")
    elif similarity_results['optimal_silhouette_score'] > 0.3:
        print(f"Moderate clustering quality - some persona groupings exist")
    else:
        print(f"Low clustering quality - personas show similar behavior patterns")
    
    # Most/least cohesive clusters
    most_cohesive = summary['most_cohesive_cluster']
    print(f"Most cohesive cluster: {most_cohesive} (agreement: {clusters[most_cohesive]['internal_agreement']:.3f})")



def run_full_preliminary_analysis(merged_df: pd.DataFrame, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run the full bias, disagreement, rescue, and persona similarity analysis on a merged predictions dataset.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        DataFrame containing true_label, baseline predictions, and all profile predictions.
    df : Optional[pd.DataFrame]
        Original dataset used for merging back stereotype_type if missing.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results from all major analysis modules.
    """
    results = {}
    
    # --- Ensure required columns are present
    if "base_pred" not in merged_df.columns:
        # You can change 'zero_shot' below to your actual baseline column
        merged_df["base_pred"] = merged_df["zero_shot"]
    
    if "stereotype_type" not in merged_df.columns and df is not None:
        if "sample_id" in merged_df.columns and "sample_id" in df.columns:
            merged_df = merged_df.merge(df[["sample_id", "stereotype_type"]], on="sample_id", how="left")
    


    print("\n\n=== DEMOGRAPHIC ACCURACY DIFFERENCES ===")
    demographic_results = test_comprehensive_demographic_accuracy_differences(merged_df)
    print_comprehensive_demographic_results(demographic_results)
    results['demographic'] = demographic_results
    


    print("\n\n=== SYSTEMATIC BIAS PATTERNS ===")
    bias_patterns, meaningful_patterns, category_summary = analyze_systematic_bias_patterns(merged_df)
    print(bias_patterns.head(20))
    results['bias_patterns'] = bias_patterns
    results['meaningful_bias_patterns'] = meaningful_patterns
    results['category_summary'] = category_summary
    


    print("\n\n=== HIGH DISAGREEMENT CASES ===")
    disagreement_df = extract_high_disagreement_cases(merged_df, threshold=0.3)
    print(disagreement_df.head(10))
    results['disagreement'] = disagreement_df
    


    print("\n\n=== RESCUE STATISTICS BY CATEGORY ===")
    rescue_df = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    print(rescue_df[rescue_df['category'] == 'race'].head(10))
    rescue_analysis = analyze_rescue_performance(rescue_df)
    results['rescue_stats'] = rescue_df
    results['rescue_analysis'] = rescue_analysis
    
    

    print("\n\n=== PERSONA SIMILARITY CLUSTERING ===")
    persona_similarity = analyze_persona_similarity(merged_df)
    print_persona_similarity_analysis(persona_similarity)
    results['persona_similarity'] = persona_similarity

    print("\n=== PLOT OF ACCURACY WITH CI ===")
    
    from utils_visualisation import plot_accuracy_deltas_with_ci
    delta_summary = plot_accuracy_deltas_with_ci(merged_df)
    
    return results