import os
import glob
import json
from collections import Counter
from itertools import combinations
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
from sklearn.model_selection import KFold

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy.stats import bootstrap, binom

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


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
    print(f"🔍 Found {len(profile_files)} profile result files.")

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
    profile_columns = sorted([col for col in merged.columns if col not in fixed_columns])
    merged = merged[fixed_columns + profile_columns]

    print("=== Merged DataFrame ready with columns:\n", merged.columns.tolist())
    return merged



def evaluate_role_playing_effects(merged: pd.DataFrame):
    results = {}
    base = merged["base_pred"].astype(str).str.strip().str.lower()
    true = merged["true_label"].astype(str).str.strip().str.lower()

    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    profiles = merged[profile_cols].apply(lambda col: col.astype(str).str.strip().str.lower())

    base_acc = accuracy_score(true, base)
    acc_diffs = {}
    for profile in profile_cols:
        acc = accuracy_score(true, profiles[profile])
        acc_diffs[profile] = acc - base_acc

    results["accuracy_differences"] = acc_diffs

    mcnemar_pvals = {}
    for profile in profile_cols:
        both_correct = ((base == true) & (profiles[profile] == true)).sum()
        base_only    = ((base == true) & (profiles[profile] != true)).sum()
        profile_only = ((base != true) & (profiles[profile] == true)).sum()
        both_wrong   = ((base != true) & (profiles[profile] != true)).sum()

        table = [[both_correct, base_only],
                 [profile_only, both_wrong]]

        try:
            mcnemar_result = mcnemar(table, exact=True)
            mcnemar_pvals[profile] = (mcnemar_result.pvalue, mcnemar_result.statistic)
        except Exception as e:
            mcnemar_pvals[profile] = (np.nan, f"Error: {e}")

    results["mcnemar_tests"] = mcnemar_pvals

    def paired_diff(data, idx):
        return np.mean(data[1][idx] == true) - np.mean(data[0][idx] == true)

    bootstrap_cis = {}
    for profile in profile_cols:
        try:
            data = (base.values, profiles[profile].values)
            res = bootstrap(data, paired_diff, confidence_level=0.95, n_resamples=1000, method="percentile", random_state=0)
            bootstrap_cis[profile] = (res.confidence_interval.low, res.confidence_interval.high)
        except Exception as e:
            bootstrap_cis[profile] = (np.nan, np.nan)

    results["bootstrap_confidence_intervals"] = bootstrap_cis

    active_profiles = [col for col in profile_cols if "active" in col]
    passive_profiles = [col for col in profile_cols if "passive" in col]

    def majority_vote(df):
        return df.mode(axis=1)[0].fillna("")

    active_vote = majority_vote(profiles[active_profiles])
    passive_vote = majority_vote(profiles[passive_profiles])
    active_acc = accuracy_score(true, active_vote)
    passive_acc = accuracy_score(true, passive_vote)

    results["aggregate_active_vs_passive"] = {
        "active_accuracy": active_acc,
        "passive_accuracy": passive_acc,
        "delta": active_acc - passive_acc
    }

    agreements = {}
    for a, b in combinations(profile_cols, 2):
        agreements[f"{a} vs {b}"] = np.mean(profiles[a] == profiles[b])
    results["pairwise_agreement"] = agreements

    return results



def print_report(report):
    
    print("=== Accuracy Improvements vs Baseline")
    for k, v in report["accuracy_differences"].items():
        print(f"{k}: Δ = {v:.4f}")
    
    print("\n=== McNemar p-values:")
    for k, (pval, stat) in report["mcnemar_tests"].items():
        print(f"{k}: p = {pval:.4f}, stat = {stat:.2f}")
    
    print("\n=== Aggregate Active vs Passive")
    print(report["aggregate_active_vs_passive"])
    
    print("\n=== Pairwise Agreement between Profiles")
    for k, v in report["pairwise_agreement"].items():
        print(f"{k}: agreement = {v:.3f}")


def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def plot_accuracy_deltas_with_ci(df):
    base = df["base_pred"]
    true = df["true_label"]
    profile_cols = [col for col in df.columns if col.startswith("profile")]

    deltas = []
    ci_lows = []
    ci_highs = []

    base_acc = compute_accuracy(true, base)

    for profile in profile_cols:
        pred = df[profile]
        acc = compute_accuracy(true, pred)
        delta = acc - base_acc
        deltas.append(delta)

        bootstraps = []
        for _ in range(1000):
            idx = np.random.choice(len(df), size=len(df), replace=True)
            boot_acc = compute_accuracy(true.iloc[idx], pred.iloc[idx])
            bootstraps.append(boot_acc - base_acc)
        ci_lows.append(np.percentile(bootstraps, 2.5))
        ci_highs.append(np.percentile(bootstraps, 97.5))

    x = np.arange(len(profile_cols))
    plt.figure(figsize=(12, 6))
    plt.bar(x, deltas, yerr=[np.array(deltas) - np.array(ci_lows), np.array(ci_highs) - np.array(deltas)],
            capsize=5)
    plt.xticks(x, profile_cols, rotation=45, ha='right')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Δ Accuracy vs Baseline (± 95% CI)")
    plt.ylabel("Δ Accuracy")
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    labels = sorted(list(set(y_true) | set(y_pred)))
    matrix = pd.crosstab(pd.Series(y_true, name='Actual'),
                         pd.Series(y_pred, name='Predicted'),
                         dropna=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, matrix.values[i, j],
                    ha='center', va='center',
                    color='black')

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

def plot_deltas_per_category(df, category_col):
    base = df["base_pred"]
    true = df["true_label"]
    profile_cols = [col for col in df.columns if col.startswith("profile")]
    categories = df[category_col].unique()

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.1
    x = np.arange(len(categories))

    for i, profile in enumerate(profile_cols):
        deltas = []
        for cat in categories:
            sub_df = df[df[category_col] == cat]
            base_acc = compute_accuracy(sub_df["true_label"], sub_df["base_pred"])
            prof_acc = compute_accuracy(sub_df["true_label"], sub_df[profile])
            deltas.append(prof_acc - base_acc)

        ax.bar(x + i*width, deltas, width=width, label=profile)

    ax.set_xticks(x + width * len(profile_cols) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Δ Accuracy")
    ax.set_title(f"Δ Accuracy per {category_col}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

def compute_classification_reports(merged_df):
    true = merged_df["true_label"].astype(str).str.strip().str.lower()
    base = merged_df["base_pred"].astype(str).str.strip().str.lower()
    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]

    reports = {}

    base_report = classification_report(true, base, output_dict=True, zero_division=0)
    reports["baseline"] = base_report

    for profile in profile_cols:
        pred = merged_df[profile].astype(str).str.strip().str.lower()
        try:
            report = classification_report(true, pred, output_dict=True, zero_division=0)
            reports[profile] = report
        except Exception as e:
            print(f"=== Error generating report for {profile}: {e}")

    summary_rows = []
    for profile, report in reports.items():
        f1_macro = report["macro avg"]["f1-score"]
        f1_weighted = report["weighted avg"]["f1-score"]
        accuracy = report["accuracy"]
        precision_macro = report["macro avg"]["precision"]
        recall_macro = report["macro avg"]["recall"]
        summary_rows.append({
            "profile": profile,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)
    print("=== Classification Reports Summary ===\n")
    print(summary_df.to_string(index=False, float_format="%.3f"))

    return summary_df

def rescue_stats_by_category(
        merged: pd.DataFrame,
        category_col: str,
        baseline_col: str = "base_pred",
        label_col: str = "true_label",
        profile_prefix: str = "profile"
) -> pd.DataFrame:
    """
    Return a long-format DataFrame with, for every (category, profile):
        N_cat           – total samples in category
        rescued         – baseline wrong & profile correct
        rescue_rate     – rescued / baseline_errors_cat
        extra_errors    – baseline correct & profile wrong
        extra_err_rate  – extra_errors / baseline_correct_cat
        profile_acc     – accuracy of the profile on the category
        baseline_acc    – accuracy of the baseline on the category
    """

    if category_col not in merged.columns:
        raise ValueError(f"{category_col} not in DataFrame")

    y_true = merged[label_col].astype(str).str.strip().str.lower()
    y_base = merged[baseline_col].astype(str).str.strip().str.lower()

    profile_cols: List[str] = [c for c in merged.columns if c.startswith(profile_prefix)]
    out_rows = []

    for cat, df_cat in merged.groupby(category_col):
        n_cat = len(df_cat)
        y_true_cat = y_true.loc[df_cat.index]
        y_base_cat = y_base.loc[df_cat.index]

        base_correct = (y_base_cat == y_true_cat)

        base_errors_cat  = (~base_correct).sum()
        base_correct_cat = base_correct.sum()

        base_acc_cat = base_correct_cat / n_cat

        for prof in profile_cols:
            y_prof_cat = df_cat[prof].astype(str).str.strip().str.lower()

            prof_correct = (y_prof_cat == y_true_cat)

            rescued      = ((~base_correct) & prof_correct).sum()
            extra_errors = (base_correct & (~prof_correct)).sum()

            rescue_rate    = rescued      / base_errors_cat  if base_errors_cat  else 0.0
            extra_err_rate = extra_errors / base_correct_cat if base_correct_cat else 0.0

            row = dict(
                category       = cat,
                profile        = prof,
                N_cat          = n_cat,
                rescued        = int(rescued),
                rescue_rate    = rescue_rate,
                extra_errors   = int(extra_errors),
                extra_err_rate = extra_err_rate,
                profile_acc    = prof_correct.mean(),
                baseline_acc   = base_acc_cat,
            )
            out_rows.append(row)

    stats_df = pd.DataFrame(out_rows)
    stats_df = stats_df.sort_values(["category", "rescued"], ascending=[True, False])

    return stats_df



def detect_systematic_biases(merged: pd.DataFrame, category_col: str = "stereotype_type") -> pd.DataFrame:
    """Identify categories where personas consistently disagree with baseline"""
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    bias_patterns = []
    
    for category in merged[category_col].unique():
        cat_data = merged[merged[category_col] == category]
        base_preds = cat_data["base_pred"]
        
        for profile in profile_cols:
            profile_preds = cat_data[profile]
            
            # Calculate directional bias
            to_positive = ((base_preds == "no") & (profile_preds == "yes")).sum()
            to_negative = ((base_preds == "yes") & (profile_preds == "no")).sum()
            
            if to_positive > to_negative:
                bias_direction = "more_positive"
                bias_magnitude = (to_positive - to_negative) / len(cat_data)
            else:
                bias_direction = "more_negative"
                bias_magnitude = (to_negative - to_positive) / len(cat_data)
            
            bias_patterns.append({
                "category": category,
                "profile": profile,
                "bias_direction": bias_direction,
                "bias_magnitude": bias_magnitude,
                "n_flips": to_positive + to_negative,
                "flip_rate": (to_positive + to_negative) / len(cat_data)
            })
    
    return pd.DataFrame(bias_patterns).sort_values("bias_magnitude", ascending=False)


def analyze_persona_similarity(merged: pd.DataFrame) -> Dict[str, Any]:
    """Cluster personas based on their prediction patterns"""
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    
    # Create distance matrix based on prediction disagreement
    n_profiles = len(profile_cols)
    distance_matrix = np.zeros((n_profiles, n_profiles))
    
    for i, prof1 in enumerate(profile_cols):
        for j, prof2 in enumerate(profile_cols):
            if i != j:
                # Distance = proportion of disagreements
                distance_matrix[i, j] = np.mean(merged[prof1] != merged[prof2])
    
    # Hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    # Find natural clusters
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
    
    # Analyze cluster characteristics
    cluster_analysis = {}
    for cluster_id in np.unique(clusters):
        cluster_profiles = [prof for prof, c in zip(profile_cols, clusters) if c == cluster_id]
        
        # Average accuracy for cluster
        avg_accuracy = np.mean([
            accuracy_score(merged["true_label"], merged[prof]) 
            for prof in cluster_profiles
        ])
        
        cluster_analysis[f"cluster_{cluster_id}"] = {
            "profiles": cluster_profiles,
            "size": len(cluster_profiles),
            "avg_accuracy": avg_accuracy,
            "internal_agreement": np.mean([
                np.mean(merged[p1] == merged[p2]) 
                for p1, p2 in combinations(cluster_profiles, 2)
            ]) if len(cluster_profiles) > 1 else 1.0
        }
    
    return {
        "clusters": cluster_analysis,
        "linkage_matrix": linkage_matrix,
        "distance_matrix": distance_matrix
    }

def evaluate_with_correction(merged: pd.DataFrame) -> pd.DataFrame:
    """Apply Bonferroni correction for multiple hypothesis testing"""
    
    results = evaluate_role_playing_effects(merged)
    
    # Extract p-values
    p_values = []
    profile_names = []
    
    for profile, (pval, stat) in results["mcnemar_tests"].items():
        if not np.isnan(pval):
            p_values.append(pval)
            profile_names.append(profile)
    
    # Apply correction
    reject, pvals_corrected, _, _ = multipletests(
        p_values, 
        alpha=0.05, 
        method='bonferroni'
    )
    
    # Create summary
    summary = pd.DataFrame({
        'profile': profile_names,
        'p_value_raw': p_values,
        'p_value_corrected': pvals_corrected,
        'significant': reject,
        'accuracy_delta': [results["accuracy_differences"][p] for p in profile_names]
    })
    
    return summary.sort_values('p_value_corrected')


def extract_high_disagreement_cases(merged: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Find cases where personas disagree most"""
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    
    # Calculate disagreement score for each sample
    disagreement_scores = []
    
    for idx, row in merged.iterrows():
        predictions = [row[col] for col in profile_cols]
        
        # Proportion of profiles that disagree with mode
        mode = max(set(predictions), key=predictions.count)
        disagreement = sum(1 for p in predictions if p != mode) / len(predictions)
        
        disagreement_scores.append({
            'sample_id': row['sample_id'],
            'disagreement_score': disagreement,
            'predictions': Counter(predictions),
            'true_label': row['true_label'],
            'base_pred': row['base_pred']
        })
    
    disagreement_df = pd.DataFrame(disagreement_scores)
    high_disagreement = disagreement_df[disagreement_df['disagreement_score'] > threshold]
    
    return high_disagreement.sort_values('disagreement_score', ascending=False)

def analyze_temporal_stability(merged: pd.DataFrame, n_folds: int = 5) -> Dict[str, Any]:
    """Check if persona effects are stable across data splits"""
    
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = {profile: [] for profile in profile_cols}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(merged)):
        test_data = merged.iloc[test_idx]
        
        # Calculate accuracy for each profile on this fold
        for profile in profile_cols:
            acc = accuracy_score(
                test_data["true_label"], 
                test_data[profile]
            )
            fold_results[profile].append(acc)
    
    # Analyze stability
    stability_analysis = {}
    for profile, accs in fold_results.items():
        stability_analysis[profile] = {
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'cv_coefficient': np.std(accs) / np.mean(accs) if np.mean(accs) > 0 else np.inf,
            'fold_accuracies': accs
        }
    
    return stability_analysis



def plot_bias_heatmap_matplotlib(merged: pd.DataFrame):
    """Matplotlib version of heatmap showing which personas favor which labels for each category"""
    
    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    categories = merged["stereotype_type"].unique()
    
    # Calculate bias matrix
    bias_matrix = np.zeros((len(categories), len(profile_cols)))
    
    for i, cat in enumerate(categories):
        cat_data = merged[merged["stereotype_type"] == cat]
        base_yes_rate = (cat_data["base_pred"] == "yes").mean()
        
        for j, profile in enumerate(profile_cols):
            profile_yes_rate = (cat_data[profile] == "yes").mean()
            # Positive = more likely to say yes, Negative = more likely to say no
            bias_matrix[i, j] = profile_yes_rate - base_yes_rate
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(bias_matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    fig.colorbar(cax, label='Bias (vs baseline)', ax=ax)

    ax.set_xticks(np.arange(len(profile_cols)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels([p.replace("profile", "P") for p in profile_cols], rotation=45, ha='right')
    ax.set_yticklabels(categories)
    ax.set_title("Persona Bias by Stereotype Category")

    for i in range(len(categories)):
        for j in range(len(profile_cols)):
            ax.text(j, i, f"{bias_matrix[i, j]:.2f}", ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()