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
        mode_df = df.mode(axis=1)
        if mode_df.shape[1] == 0:
            return pd.Series([""] * df.shape[0], index=df.index)
        return mode_df.iloc[:, 0].fillna("")

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
        method='fdr_bh'
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


def generate_person_report(
    merged: pd.DataFrame,
    persona_name: str,
    output_dir: str,
    tools: Dict[str, Any],
    run_full_analysis: bool = False
) -> None:
    """
    Generate a detailed report for a single persona or for all personas (if run_full_analysis=True).
    """
    os.makedirs(output_dir, exist_ok=True)

    if run_full_analysis:
        report_df = tools["compute_classification_reports"](merged)
        report_df.to_csv(os.path.join(output_dir, "classification_summary_all.csv"), index=False)

        roleplay_report = tools["evaluate_role_playing_effects"](merged)
        tools["print_report"](roleplay_report)

        with open(os.path.join(output_dir, "evaluation_report_all.json"), "w") as f:
            json.dump(roleplay_report, f, indent=2)

        tools["plot_accuracy_deltas_with_ci"](merged)
        plt.savefig(os.path.join(output_dir, "accuracy_delta_ci_all.png"))
        plt.close()

        tools["plot_deltas_per_category"](merged, category_col="stereotype_type")
        plt.savefig(os.path.join(output_dir, "category_deltas_all.png"))
        plt.close()

        bias_df = tools["detect_systematic_biases"](merged)
        bias_df.to_csv(os.path.join(output_dir, "bias_patterns_all.csv"), index=False)

        stability = tools["analyze_temporal_stability"](merged)
        with open(os.path.join(output_dir, "temporal_stability_all.json"), "w") as f:
            json.dump(stability, f, indent=2)

        disagreements = tools["extract_high_disagreement_cases"](merged)
        disagreements.to_csv(os.path.join(output_dir, "high_disagreement_samples_all.csv"), index=False)

        tools["plot_bias_heatmap_matplotlib"](merged)
        plt.savefig(os.path.join(output_dir, "bias_heatmap_all.png"))
        plt.close()

        cluster_info = tools["analyze_persona_similarity"](merged)
        cluster_info["linkage_matrix"] = cluster_info["linkage_matrix"].tolist()
        cluster_info["distance_matrix"] = cluster_info["distance_matrix"].tolist()
        with open(os.path.join(output_dir, "clustering_summary_all.json"), "w") as f:
            json.dump(cluster_info, f, indent=2)

    else:
        subset = merged[["sample_id", "true_label", "base_pred", persona_name, "stereotype_type"]].copy()
        subset = subset.rename(columns={persona_name: "pred_label"})

        report_df = tools["compute_classification_reports"](merged[["true_label", "base_pred", persona_name]])
        report_df.to_csv(os.path.join(output_dir, f"{persona_name}_classification_summary.csv"), index=False)

        roleplay_report = tools["evaluate_role_playing_effects"](merged)
        tools["print_report"](roleplay_report)

        with open(os.path.join(output_dir, f"{persona_name}_evaluation_report.json"), "w") as f:
            json.dump(roleplay_report, f, indent=2)

        tools["plot_accuracy_deltas_with_ci"](merged[["true_label", "base_pred", persona_name]])
        plt.savefig(os.path.join(output_dir, f"{persona_name}_accuracy_delta_ci.png"))
        plt.close()

        tools["plot_confusion_matrix"](subset["true_label"], subset["pred_label"],
                                       title=f"Confusion Matrix - {persona_name}")
        plt.savefig(os.path.join(output_dir, f"{persona_name}_confusion_matrix.png"))
        plt.close()

        rescue_df = tools["rescue_stats_by_category"](merged, category_col="stereotype_type")
        rescue_df = rescue_df[rescue_df["profile"] == persona_name]
        rescue_df.to_csv(os.path.join(output_dir, f"{persona_name}_rescue_stats.csv"), index=False)

        bias_df = tools["detect_systematic_biases"](merged)
        bias_df = bias_df[bias_df["profile"] == persona_name]
        bias_df.to_csv(os.path.join(output_dir, f"{persona_name}_bias_patterns.csv"), index=False)

        stability = tools["analyze_temporal_stability"](merged[["true_label", persona_name]])
        with open(os.path.join(output_dir, f"{persona_name}_temporal_stability.json"), "w") as f:
            json.dump(stability[persona_name], f, indent=2)

        disagreements = tools["extract_high_disagreement_cases"](merged)
        disagreements.to_csv(os.path.join(output_dir, f"{persona_name}_high_disagreement_samples.csv"), index=False)

        tools["plot_bias_heatmap_matplotlib"](merged[["true_label", "base_pred", "stereotype_type", persona_name]])
        plt.savefig(os.path.join(output_dir, f"{persona_name}_bias_heatmap.png"))
        plt.close()



from scipy.stats import f_oneway, ttest_ind
import numpy as np

def analyze_demographic_effects(merged_df):
    """
    Test main effects of gender and ethnicity on stereotype detection.
    This is specifically for your 30-persona factorial design.
    """
    

    MEN_PROFILES = [f"profile{i}_passive" for i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]]
    WOMEN_PROFILES = [f"profile{i}_passive" for i in [6,7,8,9,10, 16,17,18,19,20, 26,27,28,29,30]]
    WHITE_PROFILES = [f"profile{i}_passive" for i in range(1, 11)]
    BLACK_PROFILES = [f"profile{i}_passive" for i in range(11, 21)]
    ASIAN_PROFILES = [f"profile{i}_passive" for i in range(21, 31)]
    
    results = {}
    
    men_accuracy = []
    women_accuracy = []
    
    for profile in MEN_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            men_accuracy.append(acc)
            
    for profile in WOMEN_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            women_accuracy.append(acc)
    
    if men_accuracy and women_accuracy:
        t_stat, p_val = ttest_ind(men_accuracy, women_accuracy)
        results['gender_effect'] = {
            'men_mean': np.mean(men_accuracy),
            'women_mean': np.mean(women_accuracy),
            'difference': np.mean(women_accuracy) - np.mean(men_accuracy),
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    white_accuracy = []
    black_accuracy = []
    asian_accuracy = []
    
    for profile in WHITE_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            white_accuracy.append(acc)
            
    for profile in BLACK_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            black_accuracy.append(acc)
            
    for profile in ASIAN_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            asian_accuracy.append(acc)
    
    if white_accuracy and black_accuracy and asian_accuracy:
        f_stat, p_val = f_oneway(white_accuracy, black_accuracy, asian_accuracy)
        results['ethnicity_effect'] = {
            'white_mean': np.mean(white_accuracy),
            'black_mean': np.mean(black_accuracy), 
            'asian_mean': np.mean(asian_accuracy),
            'f_statistic': f_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    return results

def analyze_cognitive_effects(merged_df):
    """
    Test main effects of cognitive styles on stereotype detection.
    """
    
    # Define cognitive groups
    EXPANSIVE_PROFILES = [f"profile{i}_passive" for i in [1,6,11,16,21,26]]
    LITERAL_PROFILES = [f"profile{i}_passive" for i in [2,7,12,17,22,27]]
    HIGH_HARM_PROFILES = [f"profile{i}_passive" for i in [3,8,13,18,23,28]]
    LOW_HARM_PROFILES = [f"profile{i}_passive" for i in [4,9,14,19,24,29]]
    
    results = {}
    
    # Interpretation style effect
    expansive_acc = []
    literal_acc = []
    
    for profile in EXPANSIVE_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            expansive_acc.append(acc)
            
    for profile in LITERAL_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            literal_acc.append(acc)
    
    if expansive_acc and literal_acc:
        t_stat, p_val = ttest_ind(expansive_acc, literal_acc)
        results['interpretation_style'] = {
            'expansive_mean': np.mean(expansive_acc),
            'literal_mean': np.mean(literal_acc),
            'difference': np.mean(expansive_acc) - np.mean(literal_acc),
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    # Harm sensitivity effect
    high_harm_acc = []
    low_harm_acc = []
    
    for profile in HIGH_HARM_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            high_harm_acc.append(acc)
            
    for profile in LOW_HARM_PROFILES:
        if profile in merged_df.columns:
            acc = (merged_df[profile] == merged_df['true_label']).mean()
            low_harm_acc.append(acc)
    
    if high_harm_acc and low_harm_acc:
        t_stat, p_val = ttest_ind(high_harm_acc, low_harm_acc)
        results['harm_sensitivity'] = {
            'high_harm_mean': np.mean(high_harm_acc),
            'low_harm_mean': np.mean(low_harm_acc),
            'difference': np.mean(high_harm_acc) - np.mean(low_harm_acc),
            'p_value': p_val,
            'significant': p_val < 0.05
        }
    
    return results


def test_own_group_sensitivity(merged_df):
    """
    Test if personas detect stereotypes about their own demographic groups differently.
    """
    results = {}
    
    if 'stereotype_type' in merged_df.columns:
        # Test Black personas on race stereotypes
        race_subset = merged_df[merged_df['stereotype_type'] == 'race']
        if len(race_subset) > 0:
            BLACK_PROFILES = [f"profile{i}_passive" for i in range(11, 21)]
            NONBLACK_PROFILES = [f"profile{i}_passive" for i in list(range(1, 11)) + list(range(21, 31))]
            
            black_accs = []
            nonblack_accs = []
            
            for profile in BLACK_PROFILES:
                if profile in race_subset.columns:
                    acc = (race_subset[profile] == race_subset['true_label']).mean()
                    black_accs.append(acc)
            
            for profile in NONBLACK_PROFILES:
                if profile in race_subset.columns:
                    acc = (race_subset[profile] == race_subset['true_label']).mean()
                    nonblack_accs.append(acc)
            
            if black_accs and nonblack_accs:
                t_stat, p_val = ttest_ind(black_accs, nonblack_accs)
                
                results['black_on_race'] = {
                    'black_accuracy': np.mean(black_accs),
                    'nonblack_accuracy': np.mean(nonblack_accs),
                    'difference': np.mean(black_accs) - np.mean(nonblack_accs),
                    'p_value': p_val,
                    'interpretation': 'Higher accuracy suggests better alignment with annotator reasoning on race stereotypes'
                }
        
        # Test Women personas on gender stereotypes
        gender_subset = merged_df[merged_df['stereotype_type'] == 'gender']
        if len(gender_subset) > 0:
            WOMEN_PROFILES = [f"profile{i}_passive" for i in [6,7,8,9,10, 16,17,18,19,20, 26,27,28,29,30]]
            MEN_PROFILES = [f"profile{i}_passive" for i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]]
            
            women_accs = []
            men_accs = []
            
            for profile in WOMEN_PROFILES:
                if profile in gender_subset.columns:
                    acc = (gender_subset[profile] == gender_subset['true_label']).mean()
                    women_accs.append(acc)
            
            for profile in MEN_PROFILES:
                if profile in gender_subset.columns:
                    acc = (gender_subset[profile] == gender_subset['true_label']).mean()
                    men_accs.append(acc)
            
            if women_accs and men_accs:
                t_stat, p_val = ttest_ind(women_accs, men_accs)
                
                results['women_on_gender'] = {
                    'women_accuracy': np.mean(women_accs),
                    'men_accuracy': np.mean(men_accs),
                    'difference': np.mean(women_accs) - np.mean(men_accs),
                    'p_value': p_val,
                    'interpretation': 'Higher accuracy suggests better alignment with annotator reasoning on gender stereotypes'
                }
    
    return results

def test_comprehensive_own_group_sensitivity(merged_df):
    """
    Test ALL demographic groups for own-group sensitivity effects.
    This tests if each group handles stereotypes about their demographics differently.
    """
    from scipy.stats import ttest_ind
    import numpy as np
    
    results = {}
    
    if 'stereotype_type' in merged_df.columns:
        
        # =================================================================
        # TEST 1: ALL ETHNICITIES ON RACE STEREOTYPES
        # =================================================================
        race_subset = merged_df[merged_df['stereotype_type'] == 'race']
        if len(race_subset) > 0:
            
            # Define ethnic groups
            WHITE_PROFILES = [f"profile{i}_passive" for i in range(1, 11)]
            BLACK_PROFILES = [f"profile{i}_passive" for i in range(11, 21)]
            ASIAN_PROFILES = [f"profile{i}_passive" for i in range(21, 31)]
            
            # Calculate accuracies for each ethnic group on race stereotypes
            white_on_race = []
            black_on_race = []
            asian_on_race = []
            
            for profile in WHITE_PROFILES:
                if profile in race_subset.columns:
                    acc = (race_subset[profile] == race_subset['true_label']).mean()
                    white_on_race.append(acc)
            
            for profile in BLACK_PROFILES:
                if profile in race_subset.columns:
                    acc = (race_subset[profile] == race_subset['true_label']).mean()
                    black_on_race.append(acc)
                    
            for profile in ASIAN_PROFILES:
                if profile in race_subset.columns:
                    acc = (race_subset[profile] == race_subset['true_label']).mean()
                    asian_on_race.append(acc)
            
            # Compare each group to others
            if white_on_race and black_on_race:
                t_stat, p_val = ttest_ind(white_on_race, black_on_race)
                results['white_vs_black_on_race'] = {
                    'white_accuracy': np.mean(white_on_race),
                    'black_accuracy': np.mean(black_on_race),
                    'difference': np.mean(white_on_race) - np.mean(black_on_race),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'race_samples': len(race_subset),
                    'interpretation': 'Higher accuracy suggests better alignment with annotator reasoning on race stereotypes'
                }
            
            if white_on_race and asian_on_race:
                t_stat, p_val = ttest_ind(white_on_race, asian_on_race)
                results['white_vs_asian_on_race'] = {
                    'white_accuracy': np.mean(white_on_race),
                    'asian_accuracy': np.mean(asian_on_race),
                    'difference': np.mean(white_on_race) - np.mean(asian_on_race),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'race_samples': len(race_subset),
                    'interpretation': 'Higher accuracy suggests better alignment with annotator reasoning on race stereotypes'
                }
            
            if black_on_race and asian_on_race:
                t_stat, p_val = ttest_ind(black_on_race, asian_on_race)
                results['black_vs_asian_on_race'] = {
                    'black_accuracy': np.mean(black_on_race),
                    'asian_accuracy': np.mean(asian_on_race),
                    'difference': np.mean(black_on_race) - np.mean(asian_on_race),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'race_samples': len(race_subset),
                    'interpretation': 'Higher accuracy suggests better alignment with annotator reasoning on race stereotypes'
                }
            
            # Store individual group means for summary
            results['race_stereotype_summary'] = {
                'white_mean': np.mean(white_on_race) if white_on_race else None,
                'black_mean': np.mean(black_on_race) if black_on_race else None,
                'asian_mean': np.mean(asian_on_race) if asian_on_race else None,
                'best_performing_group': None,
                'worst_performing_group': None
            }
            
            # Identify best and worst performing groups
            group_means = {}
            if white_on_race:
                group_means['white'] = np.mean(white_on_race)
            if black_on_race:
                group_means['black'] = np.mean(black_on_race)
            if asian_on_race:
                group_means['asian'] = np.mean(asian_on_race)
            
            if group_means:
                best_group = max(group_means, key=group_means.get)
                worst_group = min(group_means, key=group_means.get)
                results['race_stereotype_summary']['best_performing_group'] = best_group
                results['race_stereotype_summary']['worst_performing_group'] = worst_group
        
        # =================================================================
        # TEST 2: MEN VS WOMEN ON GENDER STEREOTYPES
        # =================================================================
        gender_subset = merged_df[merged_df['stereotype_type'] == 'gender']
        if len(gender_subset) > 0:
            
            MEN_PROFILES = [f"profile{i}_passive" for i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]]
            WOMEN_PROFILES = [f"profile{i}_passive" for i in [6,7,8,9,10, 16,17,18,19,20, 26,27,28,29,30]]
            
            men_on_gender = []
            women_on_gender = []
            
            for profile in MEN_PROFILES:
                if profile in gender_subset.columns:
                    acc = (gender_subset[profile] == gender_subset['true_label']).mean()
                    men_on_gender.append(acc)
            
            for profile in WOMEN_PROFILES:
                if profile in gender_subset.columns:
                    acc = (gender_subset[profile] == gender_subset['true_label']).mean()
                    women_on_gender.append(acc)
            
            if men_on_gender and women_on_gender:
                t_stat, p_val = ttest_ind(men_on_gender, women_on_gender)
                results['men_vs_women_on_gender'] = {
                    'men_accuracy': np.mean(men_on_gender),
                    'women_accuracy': np.mean(women_on_gender),
                    'difference': np.mean(men_on_gender) - np.mean(women_on_gender),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'gender_samples': len(gender_subset),
                    'interpretation': 'Higher accuracy suggests better alignment with annotator reasoning on gender stereotypes'
                }
        
        # =================================================================
        # TEST 3: INTERSECTIONAL EFFECTS (e.g., Black women on gender vs race)
        # =================================================================
        
        # Black women on race vs gender stereotypes
        BLACK_WOMEN = [f"profile{i}_passive" for i in range(16, 21)]
        
        if race_subset is not None and len(race_subset) > 0 and gender_subset is not None and len(gender_subset) > 0:
            
            black_women_on_race = []
            black_women_on_gender = []
            
            for profile in BLACK_WOMEN:
                if profile in race_subset.columns:
                    acc = (race_subset[profile] == race_subset['true_label']).mean()
                    black_women_on_race.append(acc)
                    
                if profile in gender_subset.columns:
                    acc = (gender_subset[profile] == gender_subset['true_label']).mean()
                    black_women_on_gender.append(acc)
            
            if black_women_on_race and black_women_on_gender:
                t_stat, p_val = ttest_ind(black_women_on_race, black_women_on_gender)
                results['black_women_race_vs_gender'] = {
                    'race_accuracy': np.mean(black_women_on_race),
                    'gender_accuracy': np.mean(black_women_on_gender),
                    'difference': np.mean(black_women_on_race) - np.mean(black_women_on_gender),
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'interpretation': 'Difference in how Black women handle race vs gender stereotypes'
                }
        
        # =================================================================
        # TEST 4: WHITE MEN VS EVERYONE ELSE (testing for majority bias)
        # =================================================================
        
        # Combine all stereotype types
        WHITE_MEN = [f"profile{i}_passive" for i in range(1, 6)]
        EVERYONE_ELSE = [f"profile{i}_passive" for i in list(range(6, 31))]  # Everyone except white men
        
        white_men_acc = []
        everyone_else_acc = []
        
        for profile in WHITE_MEN:
            if profile in merged_df.columns:
                acc = (merged_df[profile] == merged_df['true_label']).mean()
                white_men_acc.append(acc)
        
        for profile in EVERYONE_ELSE:
            if profile in merged_df.columns:
                acc = (merged_df[profile] == merged_df['true_label']).mean()
                everyone_else_acc.append(acc)
        
        if white_men_acc and everyone_else_acc:
            t_stat, p_val = ttest_ind(white_men_acc, everyone_else_acc)
            results['white_men_vs_everyone'] = {
                'white_men_accuracy': np.mean(white_men_acc),
                'everyone_else_accuracy': np.mean(everyone_else_acc),
                'difference': np.mean(white_men_acc) - np.mean(everyone_else_acc),
                'p_value': p_val,
                'significant': p_val < 0.05,
                'interpretation': 'Tests if White men align better with annotators across all stereotype types'
            }
    
    return results

def print_comprehensive_own_group_results(results):
    """
    Print the comprehensive own-group sensitivity results in a clear format.
    """
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE OWN-GROUP SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Race stereotype comparisons
    print("\n=== RACE STEREOTYPE HANDLING ===")
    
    if 'race_stereotype_summary' in results:
        summary = results['race_stereotype_summary']
        print("Individual group performance on race stereotypes:")
        if summary['white_mean'] is not None:
            print(f"  White personas: {summary['white_mean']:.4f}")
        if summary['black_mean'] is not None:
            print(f"  Black personas: {summary['black_mean']:.4f}")
        if summary['asian_mean'] is not None:
            print(f"  Asian personas: {summary['asian_mean']:.4f}")
        
        if summary['best_performing_group'] and summary['worst_performing_group']:
            print(f"\n  Best performing: {summary['best_performing_group']}")
            print(f"  Worst performing: {summary['worst_performing_group']}")
    
    # Pairwise comparisons for race
    race_comparisons = [k for k in results.keys() if 'on_race' in k and 'vs' in k]
    for comparison in race_comparisons:
        data = results[comparison]
        significance_marker = "***" if data['significant'] else ""
        print(f"\n{comparison}: {significance_marker}")
        print(f"  Difference: {data['difference']:.4f}")
        print(f"  P-value: {data['p_value']:.4f}")
        print(f"  Samples: {data.get('race_samples', 'N/A')}")
    
    # Gender stereotype comparison
    print("\n=== GENDER STEREOTYPE HANDLING ===")
    if 'men_vs_women_on_gender' in results:
        data = results['men_vs_women_on_gender']
        significance_marker = "***" if data['significant'] else ""
        print(f"Men vs Women on gender stereotypes: {significance_marker}")
        print(f"  Men accuracy: {data['men_accuracy']:.4f}")
        print(f"  Women accuracy: {data['women_accuracy']:.4f}")
        print(f"  Difference: {data['difference']:.4f}")
        print(f"  P-value: {data['p_value']:.4f}")
        print(f"  Samples: {data['gender_samples']}")
    
    # Intersectional effects
    print("\n=== INTERSECTIONAL EFFECTS ===")
    if 'black_women_race_vs_gender' in results:
        data = results['black_women_race_vs_gender']
        significance_marker = "***" if data['significant'] else ""
        print(f"Black women: Race vs Gender stereotypes: {significance_marker}")
        print(f"  Race accuracy: {data['race_accuracy']:.4f}")
        print(f"  Gender accuracy: {data['gender_accuracy']:.4f}")
        print(f"  Difference: {data['difference']:.4f}")
        print(f"  P-value: {data['p_value']:.4f}")
    
    # Majority group analysis
    print("\n=== MAJORITY GROUP ANALYSIS ===")
    if 'white_men_vs_everyone' in results:
        data = results['white_men_vs_everyone']
        significance_marker = "***" if data['significant'] else ""
        print(f"White men vs Everyone else: {significance_marker}")
        print(f"  White men accuracy: {data['white_men_accuracy']:.4f}")
        print(f"  Everyone else accuracy: {data['everyone_else_accuracy']:.4f}")
        print(f"  Difference: {data['difference']:.4f}")
        print(f"  P-value: {data['p_value']:.4f}")
        print(f"  {data['interpretation']}")
    
    # Summary of significant findings
    significant_findings = [k for k, v in results.items() if isinstance(v, dict) and v.get('significant', False)]
    
    print("\n" + "=" * 50)
    print("SUMMARY OF SIGNIFICANT FINDINGS")
    print("=" * 50)
    
    if significant_findings:
        print(f"Found {len(significant_findings)} significant own-group effects:")
        for finding in significant_findings:
            data = results[finding]
            print(f"  • {finding}: p={data['p_value']:.4f}, diff={data['difference']:.4f}")
    else:
        print("No significant own-group sensitivity effects detected.")
    
    return significant_findings



def run_factorial_analysis(merged_df):
    """
    Main function to run complete factorial design analysis.
    """
    
    print("=" * 80)
    print("FACTORIAL DESIGN ANALYSIS - 30 PERSONA STUDY")
    print("=" * 80)
    
    # 1. Demographic effects
    print("\n=== DEMOGRAPHIC EFFECTS ===")
    demo_results = analyze_demographic_effects(merged_df)
    
    if 'gender_effect' in demo_results:
        g = demo_results['gender_effect']
        print(f"Gender Effect:")
        print(f"  Men accuracy: {g['men_mean']:.4f}")
        print(f"  Women accuracy: {g['women_mean']:.4f}")
        print(f"  Difference: {g['difference']:.4f}")
        print(f"  P-value: {g['p_value']:.4f}")
        print(f"  Significant: {g['significant']}")
    
    if 'ethnicity_effect' in demo_results:
        e = demo_results['ethnicity_effect']
        print(f"\nEthnicity Effect:")
        print(f"  White accuracy: {e['white_mean']:.4f}")
        print(f"  Black accuracy: {e['black_mean']:.4f}")
        print(f"  Asian accuracy: {e['asian_mean']:.4f}")
        print(f"  P-value: {e['p_value']:.4f}")
        print(f"  Significant: {e['significant']}")
    
    # 2. Cognitive effects
    print("\n=== COGNITIVE EFFECTS ===")
    cog_results = analyze_cognitive_effects(merged_df)
    
    if 'interpretation_style' in cog_results:
        i = cog_results['interpretation_style']
        print(f"Interpretation Style Effect:")
        print(f"  Expansive accuracy: {i['expansive_mean']:.4f}")
        print(f"  Literal accuracy: {i['literal_mean']:.4f}")
        print(f"  Difference: {i['difference']:.4f}")
        print(f"  P-value: {i['p_value']:.4f}")
        print(f"  Significant: {i['significant']}")
    
    if 'harm_sensitivity' in cog_results:
        h = cog_results['harm_sensitivity']
        print(f"\nHarm Sensitivity Effect:")
        print(f"  High harm accuracy: {h['high_harm_mean']:.4f}")
        print(f"  Low harm accuracy: {h['low_harm_mean']:.4f}")
        print(f"  Difference: {h['difference']:.4f}")
        print(f"  P-value: {h['p_value']:.4f}")
        print(f"  Significant: {h['significant']}")
    
    # 3. Own-group sensitivity
    print("\n=== OWN-GROUP SENSITIVITY ===")
    own_group_results = test_own_group_sensitivity(merged_df)
    
    for effect, data in own_group_results.items():
        print(f"{effect}:")
        print(f"  Difference: {data['difference']:.4f}")
        print(f"  P-value: {data['p_value']:.4f}")
        print(f"  {data['interpretation']}")
    
    return {
        'demographic_effects': demo_results,
        'cognitive_effects': cog_results,
        'own_group_sensitivity': own_group_results
    }



def evaluate_with_correction_per_category(merged: pd.DataFrame) -> dict:
    """
    Apply multiple testing correction (FDR) separately within each stereotype category.
    Returns a dict of DataFrames, one per category.
    """
    categories = merged['stereotype_type'].unique()
    all_results = {}

    for cat in categories:
        subset = merged[merged['stereotype_type'] == cat]
        result = evaluate_role_playing_effects(subset)

        # Extract p-values
        p_values = []
        profile_names = []

        for profile, (pval, _) in result["mcnemar_tests"].items():
            if not np.isnan(pval):
                p_values.append(pval)
                profile_names.append(profile)

        if not p_values:
            continue  # skip category if no testable profiles

        # Apply FDR correction
        reject, pvals_corrected, _, _ = multipletests(
            p_values, alpha=0.05, method='fdr_bh'
        )

        summary = pd.DataFrame({
            'profile': profile_names,
            'p_value_raw': p_values,
            'p_value_corrected': pvals_corrected,
            'significant': reject,
            'accuracy_delta': [result["accuracy_differences"][p] for p in profile_names]
        })

        all_results[cat] = summary.sort_values('p_value_corrected')

    return all_results


# ============================================================================
# TIER 1 ANALYSES: FACTORIAL ANALYSIS & RISK-BENEFIT FRONTIER
# Add these functions to your analysis_tools.py
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations
from sklearn.metrics import accuracy_score

def factorial_analysis_3way_anova(merged_df):
    """
    3-way ANOVA: Gender × Ethnicity × Cognitive_Style → Performance Metrics
    
    Tests which profile traits (gender, ethnicity, cognitive style) affect:
    - Accuracy
    - Rescue rate  
    - Bias magnitude
    - Extra error rate
    
    Returns comprehensive statistical analysis of main effects and interactions.
    """
    
    # Define persona groups with _passive suffix
    profile_cols = [col for col in merged_df.columns if col.startswith("profile") and "_passive" in col]
    
    # Create mapping from profile to traits
    profile_traits = {}
    for i in range(1, 31):
        profile_name = f"profile{i}_passive"
        
        # Gender assignment
        if i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]:
            gender = 'Male'
        else:
            gender = 'Female'
        
        # Ethnicity assignment  
        if i in range(1, 11):
            ethnicity = 'White'
        elif i in range(11, 21):
            ethnicity = 'Black'
        else:
            ethnicity = 'Asian'
        
        # Cognitive style assignment
        if i in [1,6,11,16,21,26]:
            cognitive_style = 'Expansive'
        elif i in [2,7,12,17,22,27]:
            cognitive_style = 'Literal'
        elif i in [3,8,13,18,23,28]:
            cognitive_style = 'High_Harm'
        elif i in [4,9,14,19,24,29]:
            cognitive_style = 'Low_Harm'
        else:  # [5,10,15,20,25,30]
            cognitive_style = 'Balanced'
        
        profile_traits[profile_name] = {
            'gender': gender,
            'ethnicity': ethnicity, 
            'cognitive_style': cognitive_style
        }
    
    # Calculate performance metrics for each profile
    performance_data = []
    
    # Get rescue stats for reference
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    bias_patterns = detect_systematic_biases(merged_df, category_col="stereotype_type")
    
    for profile in profile_cols:
        if profile in merged_df.columns:
            
            # Basic accuracy
            accuracy = (merged_df[profile] == merged_df['true_label']).mean()
            
            # Get rescue stats for this profile
            profile_rescue = rescue_stats[rescue_stats['profile'] == profile]
            avg_rescue_rate = profile_rescue['rescue_rate'].mean() if len(profile_rescue) > 0 else 0
            avg_extra_error_rate = profile_rescue['extra_err_rate'].mean() if len(profile_rescue) > 0 else 0
            
            # Get bias patterns for this profile  
            profile_bias = bias_patterns[bias_patterns['profile'] == profile]
            avg_bias_magnitude = profile_bias['bias_magnitude'].mean() if len(profile_bias) > 0 else 0
            avg_flip_rate = profile_bias['flip_rate'].mean() if len(profile_bias) > 0 else 0
            
            # Append to performance data
            traits = profile_traits[profile]
            performance_data.append({
                'profile': profile,
                'gender': traits['gender'],
                'ethnicity': traits['ethnicity'],
                'cognitive_style': traits['cognitive_style'],
                'accuracy': accuracy,
                'rescue_rate': avg_rescue_rate,
                'extra_error_rate': avg_extra_error_rate,
                'bias_magnitude': avg_bias_magnitude,
                'flip_rate': avg_flip_rate
            })
    
    performance_df = pd.DataFrame(performance_data)
    
    # ========================================================================
    # ANOVA TESTS FOR EACH DEPENDENT VARIABLE
    # ========================================================================
    
    results = {}
    dependent_vars = ['accuracy', 'rescue_rate', 'extra_error_rate', 'bias_magnitude', 'flip_rate']
    
    for dv in dependent_vars:
        
        dv_results = {'dependent_variable': dv}
        
        # Main effects
        print(f"\n{'='*60}")
        print(f"FACTORIAL ANOVA: {dv.upper()}")
        print(f"{'='*60}")
        
        # Gender main effect
        male_vals = performance_df[performance_df['gender'] == 'Male'][dv].values
        female_vals = performance_df[performance_df['gender'] == 'Female'][dv].values
        
        if len(male_vals) > 0 and len(female_vals) > 0:
            t_stat, p_val = ttest_ind(male_vals, female_vals)
            dv_results['gender_main_effect'] = {
                'male_mean': np.mean(male_vals),
                'female_mean': np.mean(female_vals),
                'difference': np.mean(female_vals) - np.mean(male_vals),
                't_statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'effect_size_d': (np.mean(female_vals) - np.mean(male_vals)) / np.sqrt((np.var(male_vals) + np.var(female_vals)) / 2)
            }
            
            print(f"GENDER MAIN EFFECT:")
            print(f"  Male mean: {np.mean(male_vals):.4f}")
            print(f"  Female mean: {np.mean(female_vals):.4f}")
            print(f"  Difference: {np.mean(female_vals) - np.mean(male_vals):.4f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
            print(f"  Cohen's d: {dv_results['gender_main_effect']['effect_size_d']:.3f}")
        
        # Ethnicity main effect
        white_vals = performance_df[performance_df['ethnicity'] == 'White'][dv].values
        black_vals = performance_df[performance_df['ethnicity'] == 'Black'][dv].values
        asian_vals = performance_df[performance_df['ethnicity'] == 'Asian'][dv].values
        
        if len(white_vals) > 0 and len(black_vals) > 0 and len(asian_vals) > 0:
            f_stat, p_val = f_oneway(white_vals, black_vals, asian_vals)
            dv_results['ethnicity_main_effect'] = {
                'white_mean': np.mean(white_vals),
                'black_mean': np.mean(black_vals),
                'asian_mean': np.mean(asian_vals),
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            
            print(f"\nETHNICITY MAIN EFFECT:")
            print(f"  White mean: {np.mean(white_vals):.4f}")
            print(f"  Black mean: {np.mean(black_vals):.4f}")
            print(f"  Asian mean: {np.mean(asian_vals):.4f}")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
        
        # Cognitive style main effect
        cognitive_styles = performance_df['cognitive_style'].unique()
        cognitive_vals = []
        cognitive_means = {}
        
        for style in cognitive_styles:
            style_vals = performance_df[performance_df['cognitive_style'] == style][dv].values
            if len(style_vals) > 0:
                cognitive_vals.append(style_vals)
                cognitive_means[style] = np.mean(style_vals)
        
        if len(cognitive_vals) >= 2:
            f_stat, p_val = f_oneway(*cognitive_vals)
            dv_results['cognitive_main_effect'] = {
                'style_means': cognitive_means,
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
            
            print(f"\nCOGNITIVE STYLE MAIN EFFECT:")
            for style, mean_val in cognitive_means.items():
                print(f"  {style} mean: {mean_val:.4f}")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.05 else ''}")
        
        # Two-way interactions (Gender × Ethnicity)
        print(f"\nINTERACTION EFFECTS:")
        interaction_means = {}
        for gender in ['Male', 'Female']:
            for ethnicity in ['White', 'Black', 'Asian']:
                subset = performance_df[
                    (performance_df['gender'] == gender) & 
                    (performance_df['ethnicity'] == ethnicity)
                ][dv].values
                if len(subset) > 0:
                    interaction_means[f"{gender}_{ethnicity}"] = np.mean(subset)
                    print(f"  {gender} {ethnicity}: {np.mean(subset):.4f}")
        
        dv_results['interaction_means'] = interaction_means
        
        results[dv] = dv_results
    
    # Summary of significant effects
    print(f"\n{'='*60}")
    print("FACTORIAL ANOVA SUMMARY")
    print(f"{'='*60}")
    
    significant_effects = []
    for dv, dv_results in results.items():
        print(f"\n{dv.upper()}:")
        
        for effect_type in ['gender_main_effect', 'ethnicity_main_effect', 'cognitive_main_effect']:
            if effect_type in dv_results and dv_results[effect_type]['significant']:
                print(f"  ✓ {effect_type}: p={dv_results[effect_type]['p_value']:.4f}")
                significant_effects.append(f"{dv}_{effect_type}")
            elif effect_type in dv_results:
                print(f"    {effect_type}: p={dv_results[effect_type]['p_value']:.4f}")
    
    print(f"\nTotal significant effects found: {len(significant_effects)}")
    
    return {
        'performance_data': performance_df,
        'anova_results': results,
        'significant_effects': significant_effects,
        'profile_traits': profile_traits
    }


def plot_risk_benefit_frontier(merged_df, figsize=(12, 8)):
    """
    Risk-Benefit Frontier Analysis
    
    Plots each profile on:
    - X-axis: Extra error rate (risk)
    - Y-axis: Rescue rate (benefit)
    
    Identifies Pareto frontier of "safely bold" profiles that maximize
    rescue while minimizing extra errors.
    """
    
    # Get rescue stats
    rescue_stats = rescue_stats_by_category(merged_df, category_col="stereotype_type")
    
    # Calculate overall rescue and extra error rates per profile
    profile_performance = rescue_stats.groupby('profile').agg({
        'rescue_rate': 'mean',
        'extra_err_rate': 'mean', 
        'rescued': 'sum',
        'extra_errors': 'sum'
    }).reset_index()
    
    # Add demographic info for coloring
    profile_traits = {}
    for i in range(1, 31):
        profile_name = f"profile{i}_passive"
        
        if i in [1,2,3,4,5, 11,12,13,14,15, 21,22,23,24,25]:
            gender = 'Male'
        else:
            gender = 'Female'
        
        if i in range(1, 11):
            ethnicity = 'White'
        elif i in range(11, 21):
            ethnicity = 'Black'
        else:
            ethnicity = 'Asian'
        
        if i in [1,6,11,16,21,26]:
            cognitive_style = 'Expansive'
        elif i in [2,7,12,17,22,27]:
            cognitive_style = 'Literal'
        elif i in [3,8,13,18,23,28]:
            cognitive_style = 'High_Harm'
        elif i in [4,9,14,19,24,29]:
            cognitive_style = 'Low_Harm'
        else:
            cognitive_style = 'Balanced'
        
        profile_traits[profile_name] = {
            'gender': gender,
            'ethnicity': ethnicity,
            'cognitive_style': cognitive_style,
            'intersectional': f"{gender}_{ethnicity}"
        }
    
    # Add trait information to performance data
    for trait_name in ['gender', 'ethnicity', 'cognitive_style', 'intersectional']:
        profile_performance[trait_name] = profile_performance['profile'].map(
            lambda x: profile_traits.get(x, {}).get(trait_name, 'Unknown')
        )
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Risk-Benefit Frontier Analysis: Rescue Rate vs Extra Error Rate', fontsize=16, fontweight='bold')
    
    # Color schemes for different groupings
    color_schemes = {
        'gender': {'Male': '#1f77b4', 'Female': '#ff7f0e'},
        'ethnicity': {'White': '#2ca02c', 'Black': '#d62728', 'Asian': '#9467bd'},
        'cognitive_style': {
            'Expansive': '#8c564b', 'Literal': '#e377c2', 
            'High_Harm': '#7f7f7f', 'Low_Harm': '#bcbd22', 'Balanced': '#17becf'
        }
    }
    
    # Plot 1: Gender
    ax = axes[0, 0]
    for gender in ['Male', 'Female']:
        subset = profile_performance[profile_performance['gender'] == gender]
        ax.scatter(subset['extra_err_rate'], subset['rescue_rate'], 
                  c=color_schemes['gender'][gender], label=gender, alpha=0.7, s=60)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Risk-Benefit by Gender')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ethnicity  
    ax = axes[0, 1]
    for ethnicity in ['White', 'Black', 'Asian']:
        subset = profile_performance[profile_performance['ethnicity'] == ethnicity]
        ax.scatter(subset['extra_err_rate'], subset['rescue_rate'],
                  c=color_schemes['ethnicity'][ethnicity], label=ethnicity, alpha=0.7, s=60)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Risk-Benefit by Ethnicity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cognitive Style
    ax = axes[1, 0]
    for style in color_schemes['cognitive_style'].keys():
        subset = profile_performance[profile_performance['cognitive_style'] == style]
        if len(subset) > 0:
            ax.scatter(subset['extra_err_rate'], subset['rescue_rate'],
                      c=color_schemes['cognitive_style'][style], label=style, alpha=0.7, s=60)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Risk-Benefit by Cognitive Style')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Pareto Frontier Identification
    ax = axes[1, 1]
    
    # Plot all points
    scatter = ax.scatter(profile_performance['extra_err_rate'], profile_performance['rescue_rate'],
                        c='lightblue', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Calculate Pareto frontier
    points = profile_performance[['extra_err_rate', 'rescue_rate']].values
    
    # Find Pareto optimal points (minimize risk, maximize benefit)
    pareto_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        # A point is Pareto optimal if no other point dominates it
        # (lower extra_err_rate AND higher rescue_rate)
        dominated = False
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[0] <= point[0] and other_point[1] >= point[1] and 
                    (other_point[0] < point[0] or other_point[1] > point[1])):
                    dominated = True
                    break
        
        if not dominated:
            pareto_mask[i] = True
    
    # Highlight Pareto optimal points
    pareto_points = profile_performance[pareto_mask]
    ax.scatter(pareto_points['extra_err_rate'], pareto_points['rescue_rate'],
              c='red', s=100, marker='*', label='Pareto Optimal', zorder=5)
    
    # Add profile labels for Pareto optimal points
    for idx, row in pareto_points.iterrows():
        ax.annotate(row['profile'].replace('_passive', ''), 
                   (row['extra_err_rate'], row['rescue_rate']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Extra Error Rate (Risk)')
    ax.set_ylabel('Rescue Rate (Benefit)')
    ax.set_title('Pareto Frontier: "Safely Bold" Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print Pareto optimal profiles
    print("\n" + "="*60)
    print("PARETO FRONTIER ANALYSIS")
    print("="*60)
    print("\nPareto Optimal Profiles (Safely Bold):")
    for idx, row in pareto_points.iterrows():
        traits = profile_traits.get(row['profile'], {})
        print(f"  {row['profile']}: {traits.get('gender', 'Unknown')} {traits.get('ethnicity', 'Unknown')} {traits.get('cognitive_style', 'Unknown')}")
        print(f"    Rescue Rate: {row['rescue_rate']:.3f}, Extra Error Rate: {row['extra_err_rate']:.3f}")
    
    # Risk-Benefit statistics
    print(f"\nRisk-Benefit Statistics:")
    print(f"  Average rescue rate: {profile_performance['rescue_rate'].mean():.3f}")
    print(f"  Average extra error rate: {profile_performance['extra_err_rate'].mean():.3f}")
    print(f"  Best rescue rate: {profile_performance['rescue_rate'].max():.3f}")
    print(f"  Lowest extra error rate: {profile_performance['extra_err_rate'].min():.3f}")
    
    # Identify specific profile archetypes
    print(f"\nProfile Archetypes:")
    
    # Safest (lowest extra error rate)
    safest = profile_performance.loc[profile_performance['extra_err_rate'].idxmin()]
    safest_traits = profile_traits.get(safest['profile'], {})
    print(f"  Safest Profile: {safest['profile']} ({safest_traits.get('gender', '')} {safest_traits.get('ethnicity', '')} {safest_traits.get('cognitive_style', '')})")
    print(f"    Extra Error Rate: {safest['extra_err_rate']:.3f}")
    
    # Most Beneficial (highest rescue rate)
    most_beneficial = profile_performance.loc[profile_performance['rescue_rate'].idxmax()]
    beneficial_traits = profile_traits.get(most_beneficial['profile'], {})
    print(f"  Most Beneficial Profile: {most_beneficial['profile']} ({beneficial_traits.get('gender', '')} {beneficial_traits.get('ethnicity', '')} {beneficial_traits.get('cognitive_style', '')})")
    print(f"    Rescue Rate: {most_beneficial['rescue_rate']:.3f}")
    
    return {
        'performance_data': profile_performance,
        'pareto_optimal': pareto_points,
        'profile_traits': profile_traits,
        'safest_profile': safest,
        'most_beneficial_profile': most_beneficial
    }


def calculate_effect_sizes(merged_df, significant_findings):
    """
    Calculate Cohen's d effect sizes for significant findings from comprehensive analysis.
    
    Effect size interpretation:
    - Small: d = 0.2
    - Medium: d = 0.5  
    - Large: d = 0.8
    """
    
    effect_sizes = {}
    
    # Calculate for significant findings from comprehensive own-group analysis
    if 'white_vs_black_on_race' in significant_findings:
        race_subset = merged_df[merged_df['stereotype_type'] == 'race']
        
        white_profiles = [f"profile{i}_passive" for i in range(1, 11)]
        black_profiles = [f"profile{i}_passive" for i in range(11, 21)]
        
        white_accs = []
        black_accs = []
        
        for profile in white_profiles:
            if profile in race_subset.columns:
                acc = (race_subset[profile] == race_subset['true_label']).mean()
                white_accs.append(acc)
        
        for profile in black_profiles:
            if profile in race_subset.columns:
                acc = (race_subset[profile] == race_subset['true_label']).mean()
                black_accs.append(acc)
        
        if white_accs and black_accs:
            pooled_std = np.sqrt((np.var(white_accs, ddof=1) + np.var(black_accs, ddof=1)) / 2)
            cohens_d = (np.mean(white_accs) - np.mean(black_accs)) / pooled_std
            
            effect_sizes['white_vs_black_on_race'] = {
                'cohens_d': cohens_d,
                'magnitude': 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small',
                'white_mean': np.mean(white_accs),
                'black_mean': np.mean(black_accs),
                'difference': np.mean(white_accs) - np.mean(black_accs)
            }
    
    # Black women race vs gender effect size
    if 'black_women_race_vs_gender' in significant_findings:
        race_subset = merged_df[merged_df['stereotype_type'] == 'race']
        gender_subset = merged_df[merged_df['stereotype_type'] == 'gender']
        
        black_women = [f"profile{i}_passive" for i in range(16, 21)]
        
        race_accs = []
        gender_accs = []
        
        for profile in black_women:
            if profile in race_subset.columns:
                acc = (race_subset[profile] == race_subset['true_label']).mean()
                race_accs.append(acc)
            
            if profile in gender_subset.columns:
                acc = (gender_subset[profile] == gender_subset['true_label']).mean()
                gender_accs.append(acc)
        
        if race_accs and gender_accs:
            pooled_std = np.sqrt((np.var(race_accs, ddof=1) + np.var(gender_accs, ddof=1)) / 2)
            cohens_d = (np.mean(race_accs) - np.mean(gender_accs)) / pooled_std
            
            effect_sizes['black_women_race_vs_gender'] = {
                'cohens_d': cohens_d,
                'magnitude': 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small',
                'race_mean': np.mean(race_accs),
                'gender_mean': np.mean(gender_accs), 
                'difference': np.mean(race_accs) - np.mean(gender_accs)
            }
    
    # Print effect sizes
    print("\n" + "="*60)
    print("EFFECT SIZE ANALYSIS (Cohen's d)")
    print("="*60)
    
    for finding, stats in effect_sizes.items():
        print(f"\n{finding}:")
        print(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['magnitude']} effect)")
        print(f"  Mean difference: {stats['difference']:.4f}")
        
        if 'white_mean' in stats:
            print(f"  White personas: {stats['white_mean']:.4f}")
            print(f"  Black personas: {stats['black_mean']:.4f}")
        elif 'race_mean' in stats:
            print(f"  Race stereotypes: {stats['race_mean']:.4f}")
            print(f"  Gender stereotypes: {stats['gender_mean']:.4f}")
    
    return effect_sizes


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import accuracy_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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
    
    for cluster_id, cluster_info in similarity_results['clusters'].items():
        cluster_profiles = cluster_info['profiles']
        
        print(f"\n{cluster_id.upper()} ({cluster_info['size']} profiles):")
        print(f"Profiles: {', '.join([p.replace('_passive', '') for p in cluster_profiles[:5]])}")
        if len(cluster_profiles) > 5:
            print(f"          ... and {len(cluster_profiles) - 5} more")
        
        # Calculate cluster-level metrics
        cluster_metrics = {
            'profiles': cluster_profiles,
            'size': len(cluster_profiles),
            'internal_agreement': cluster_info['internal_agreement']
        }
        
        # Accuracy metrics
        accuracies = []
        for profile in cluster_profiles:
            if profile in merged_df.columns:
                acc = accuracy_score(merged_df['true_label'], merged_df[profile])
                accuracies.append(acc)
        
        cluster_metrics['accuracy_mean'] = np.mean(accuracies) if accuracies else 0
        cluster_metrics['accuracy_std'] = np.std(accuracies) if accuracies else 0
        
        # Rescue metrics
        cluster_rescue_stats = rescue_stats[rescue_stats['profile'].isin(cluster_profiles)]
        if len(cluster_rescue_stats) > 0:
            cluster_metrics['rescue_rate_mean'] = cluster_rescue_stats['rescue_rate'].mean()
            cluster_metrics['rescue_rate_std'] = cluster_rescue_stats['rescue_rate'].std()
            cluster_metrics['extra_error_rate_mean'] = cluster_rescue_stats['extra_err_rate'].mean()
            cluster_metrics['extra_error_rate_std'] = cluster_rescue_stats['extra_err_rate'].std()
            cluster_metrics['total_rescued'] = cluster_rescue_stats['rescued'].sum()
            cluster_metrics['total_extra_errors'] = cluster_rescue_stats['extra_errors'].sum()
        else:
            cluster_metrics.update({
                'rescue_rate_mean': 0, 'rescue_rate_std': 0,
                'extra_error_rate_mean': 0, 'extra_error_rate_std': 0,
                'total_rescued': 0, 'total_extra_errors': 0
            })
        
        # Bias metrics
        cluster_bias_stats = bias_patterns[bias_patterns['profile'].isin(cluster_profiles)]
        if len(cluster_bias_stats) > 0:
            cluster_metrics['bias_magnitude_mean'] = cluster_bias_stats['bias_magnitude'].mean()
            cluster_metrics['bias_magnitude_std'] = cluster_bias_stats['bias_magnitude'].std()
            cluster_metrics['flip_rate_mean'] = cluster_bias_stats['flip_rate'].mean()
            
            # Dominant bias direction
            bias_directions = cluster_bias_stats['bias_direction'].value_counts()
            cluster_metrics['dominant_bias_direction'] = bias_directions.index[0] if len(bias_directions) > 0 else 'none'
        else:
            cluster_metrics.update({
                'bias_magnitude_mean': 0, 'bias_magnitude_std': 0,
                'flip_rate_mean': 0, 'dominant_bias_direction': 'none'
            })
        
        # Print cluster summary
        print(f"  Accuracy: {cluster_metrics['accuracy_mean']:.4f} ± {cluster_metrics['accuracy_std']:.4f}")
        print(f"  Rescue Rate: {cluster_metrics['rescue_rate_mean']:.3f} ± {cluster_metrics['rescue_rate_std']:.3f}")
        print(f"  Extra Error Rate: {cluster_metrics['extra_error_rate_mean']:.3f} ± {cluster_metrics['extra_error_rate_std']:.3f}")
        print(f"  Bias Magnitude: {cluster_metrics['bias_magnitude_mean']:.3f} (direction: {cluster_metrics['dominant_bias_direction']})")
        print(f"  Internal Agreement: {cluster_metrics['internal_agreement']:.3f}")
        
        # Identify cluster archetype
        if cluster_metrics['rescue_rate_mean'] > 0.15 and cluster_metrics['extra_error_rate_mean'] < 0.05:
            archetype = "🎯 Safely Bold"
        elif cluster_metrics['extra_error_rate_mean'] < 0.03:
            archetype = "🛡️ Ultra Safe"
        elif cluster_metrics['rescue_rate_mean'] > 0.20:
            archetype = "⚡ High Rescue"
        elif cluster_metrics['accuracy_mean'] > 0.72:
            archetype = "🔥 High Performer"
        elif cluster_metrics['internal_agreement'] > 0.95:
            archetype = "🤝 Highly Consistent"
        else:
            archetype = "⚖️ Balanced"
        
        cluster_metrics['archetype'] = archetype
        print(f"  Archetype: {archetype}")
        
        cluster_analysis[cluster_id] = cluster_metrics
    
    # Cluster comparison and recommendations
    print(f"\n{'='*60}")
    print("CLUSTER COMPARISON AND RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Find best cluster for different objectives
    best_accuracy = max(cluster_analysis.items(), key=lambda x: x[1]['accuracy_mean'])
    safest_cluster = min(cluster_analysis.items(), key=lambda x: x[1]['extra_error_rate_mean'])
    best_rescue = max(cluster_analysis.items(), key=lambda x: x[1]['rescue_rate_mean'])
    
    print(f"\nBest Overall Performance: {best_accuracy[0]} (accuracy: {best_accuracy[1]['accuracy_mean']:.4f})")
    print(f"Safest Cluster: {safest_cluster[0]} (extra error rate: {safest_cluster[1]['extra_error_rate_mean']:.3f})")
    print(f"Best Rescue Cluster: {best_rescue[0]} (rescue rate: {best_rescue[1]['rescue_rate_mean']:.3f})")
    
    # Calculate cluster ensemble performance
    print(f"\n{'='*60}")
    print("CLUSTER ENSEMBLE PERFORMANCE")
    print(f"{'='*60}")
    
    def majority_vote_cluster(df, cluster_profiles):
        """Calculate majority vote for a cluster"""
        available_profiles = [p for p in cluster_profiles if p in df.columns]
        if not available_profiles:
            return pd.Series([''] * len(df), index=df.index)
        
        ensemble_preds = df[available_profiles]
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
    
    baseline_accuracy = accuracy_score(merged_df['true_label'], merged_df['base_pred'])
    
    for cluster_id, cluster_info in cluster_analysis.items():
        cluster_preds = majority_vote_cluster(merged_df, cluster_info['profiles'])
        
        if not cluster_preds.eq('').all():
            cluster_ensemble_acc = accuracy_score(merged_df['true_label'], cluster_preds)
            improvement = cluster_ensemble_acc - baseline_accuracy
            
            print(f"{cluster_id}: {cluster_ensemble_acc:.4f} ({improvement:+.4f}) | {cluster_info['archetype']}")
    
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
        '🎯 Safely Bold': '#2ca02c',
        '🛡️ Ultra Safe': '#1f77b4', 
        '⚡ High Rescue': '#ff7f0e',
        '🔥 High Performer': '#d62728',
        '🤝 Highly Consistent': '#9467bd',
        '⚖️ Balanced': '#8c564b'
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