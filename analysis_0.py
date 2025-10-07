import os
import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

from sklearn.metrics import silhouette_score

from scipy import stats
from scipy.stats import sem, t
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage

from statsmodels.stats.multitest import multipletests

from analysis_tools import (
    get_demographic_info,
    get_analysis_group_keys,
    guarded_labelspace_analysis,
    resolve_plot_dir,
)
from plot_tools import apply_neurips_figure_style
from profiles.profile_sets import PERSON_ETHNICS
from profiles.schema import Gender, PersonMeta, PersonSet
from cases.cases_config import CaseConfig
from tokens_metrics import compute_token_analysis

__all__ = [
    "test_comprehensive_demographic_accuracy_differences",
    "print_comprehensive_demographic_results",
    "extract_high_disagreement_cases",
    "print_disagreement_analysis",
    "rescue_stats_by_category",
    "analyze_rescue_performance",
    "compute_error_direction_shifts",
    "analyze_systematic_error_patterns",
    "analyze_profile_similarity",
    "print_profile_similarity_analysis",
    "plot_accuracy_deltas_with_ci",
    "run_full_preliminary_analysis",
    "compute_pairwise_demographic_diffs",
    "plot_volcano_demographic_diffs",
    "plot_effect_size_heatmap",
    "plot_intersectional_accuracy_heatmap",
    "plot_demographic_accuracy_composite",
]

def _pretty(s: str) -> str:
    s = str(s)
    s = s.replace("_", " ").replace(".", " ")
    s = " ".join(s.split())
    s = re.sub(r"\bage\s+age\b", "age", s, flags=re.I)
    s = re.sub(r"\bage\s*([^. _-]?)\s*", lambda m: "Age " + m.group(1), s, flags=re.I)
    out = s.title().replace(" Vs ", " vs ")
    return out

def _profile_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("profile")]

def _traits_for_profile(person_set: PersonSet, profile: str, keys=("ethnicity", "gender")) -> Dict[str, str]:
    t = person_set.get_traits(profile, group_keys=list(keys))
    return {k: ("" if t.get(k) is None else str(t[k]).lower()) for k in keys}

def _group_profiles_by_trait(merged_df: pd.DataFrame, person_set: PersonSet, trait="ethnicity", min_profiles=1):
    groups = {}
    for p in _profile_cols(merged_df):
        tr = _traits_for_profile(person_set, p)
        g = tr.get(trait, "unknown") or "unknown"
        groups.setdefault(g, []).append(p)
    return {g: profs for g, profs in groups.items() if len(profs) >= min_profiles}

def _cohens_h(p1: float, p2: float) -> float:
    p1 = float(np.clip(p1, 1e-12, 1 - 1e-12))
    p2 = float(np.clip(p2, 1e-12, 1 - 1e-12))
    return 2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2))

def _bootstrap_group_accuracy_diff(
    df: pd.DataFrame,
    group1_profiles: List[str],
    group2_profiles: List[str],
    label_col: str = "true_label",
    n_boot: int = 2000,
    random_state: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    n_items = len(df)
    if n_items == 0 or not group1_profiles or not group2_profiles:
        return {"ok": False}

    def _group_mean_acc(cols):
        per_prof = [np.mean(df[c].values == df[label_col].values) for c in cols]
        return float(np.mean(per_prof)) if per_prof else np.nan

    acc1_hat = _group_mean_acc(group1_profiles)
    acc2_hat = _group_mean_acc(group2_profiles)
    diff_hat = acc1_hat - acc2_hat

    diffs = np.empty(n_boot, dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        y = df[label_col].values[idx]

        def _acc(cols):
            if not cols:
                return np.nan
            per_prof = [(df[c].values[idx] == y).mean() for c in cols]
            return float(np.mean(per_prof))

        a1 = _acc(group1_profiles)
        a2 = _acc(group2_profiles)
        diffs[_] = a1 - a2

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_boot = 2.0 * min(np.mean(diffs >= 0.0), np.mean(diffs <= 0.0))
    p_boot = float(min(1.0, max(0.0, p_boot)))

    return {
        "ok": True,
        "acc1": acc1_hat,
        "acc2": acc2_hat,
        "diff": diff_hat,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_raw": p_boot,
        "n_boot": n_boot,
        "n_items": n_items,
    }

def test_comprehensive_demographic_accuracy_differences(
    merged_df: pd.DataFrame,
    person_set: PersonSet = PERSON_ETHNICS,
    n_boot: int = 2000,
    random_state: int = 0,
) -> Dict[str, Any]:
    results = {}

    df_cols = {c for c in merged_df.columns if c.startswith("profile")}
    meta_cols = set(person_set.metadata.keys())
    covered = df_cols & meta_cols
    missing_in_meta = df_cols - meta_cols
    unused_meta = meta_cols - df_cols

    print(f"Profiles covered by metadata: {len(covered)}/{len(df_cols)}")
    if missing_in_meta:
        some = ", ".join(sorted(list(missing_in_meta))[:10])
        print(f"Warning: {len(missing_in_meta)} dataframe columns have no metadata (e.g., {some} …)")
    if unused_meta:
        print(f"Info: {len(unused_meta)} metadata profiles have no column in dataframe (ignored)")

    def get_profiles_by_trait(trait_name, trait_value):
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

    def run_test(group1_profiles, group2_profiles, g1_name, g2_name):
        if not group1_profiles or not group2_profiles:
            return None
        boot = _bootstrap_group_accuracy_diff(
            merged_df, group1_profiles, group2_profiles, label_col="true_label", n_boot=n_boot, random_state=random_state
        )
        if not boot["ok"]:
            return None
        acc1, acc2 = boot["acc1"], boot["acc2"]
        diff, lo, hi, p_raw = boot["diff"], boot["ci_low"], boot["ci_high"], boot["p_raw"]
        h = _cohens_h(acc1, acc2)
        return {
            f"{g1_name}_accuracy": acc1,
            f"{g2_name}_accuracy": acc2,
            "difference": diff,
            "ci_low": lo,
            "ci_high": hi,
            "p_raw": p_raw,
            "h": h,
            "n_items": boot["n_items"],
            "n_boot": boot["n_boot"],
            "n_profiles": f"{g1_name}:{len(group1_profiles)} | {g2_name}:{len(group2_profiles)}",
            "interpretation": "Bootstrap diff in mean profile accuracy (clustered by item)",
        }

    available_traits = list(PersonMeta.__dataclass_fields__.keys())
    tests_sequence = []

    if "gender" in available_traits:
        men = get_profiles_by_trait("gender", Gender.man)
        women = get_profiles_by_trait("gender", Gender.woman)
        res = run_test(men, women, "man", "woman")
        if res:
            key = "man_vs_woman"
            results[key] = res
            tests_sequence.append(key)

    ethnicities = sorted(
        {getattr(meta, "ethnicity") for meta in person_set.metadata.values()},
        key=lambda x: str(getattr(x, "value", x)),
    )
    for e1, e2 in combinations(ethnicities, 2):
        name1 = str(getattr(e1, "value", e1)).lower()
        name2 = str(getattr(e2, "value", e2)).lower()
        p1 = get_profiles_by_trait("ethnicity", e1)
        p2 = get_profiles_by_trait("ethnicity", e2)
        res = run_test(p1, p2, name1, name2)
        if res:
            key = f"{name1}_vs_{name2}"
            results[key] = res
            tests_sequence.append(key)

    extra_traits = [t for t in available_traits if t not in ("gender", "ethnicity")]
    if "age" in extra_traits:
        ages = sorted(
            {
                getattr(meta, "age")
                for meta in person_set.metadata.values()
                if getattr(meta, "age") is not None
            }
        )
        for v1, v2 in combinations(ages, 2):
            p1 = get_profiles_by_trait("age", v1)
            p2 = get_profiles_by_trait("age", v2)
            res = run_test(p1, p2, str(v1), str(v2))
            if res:
                key = f"age_{str(v1).lower()}_vs_{str(v2).lower()}"
                results[key] = res
                tests_sequence.append(key)

    if "gender" in available_traits and "ethnicity" in available_traits:
        genders = {getattr(meta, "gender") for meta in person_set.metadata.values()}
        if Gender.man in genders and Gender.woman in genders:
            for eth in {getattr(meta, "ethnicity") for meta in person_set.metadata.values()}:
                g1_profiles = [
                    pid
                    for pid, meta in person_set.metadata.items()
                    if getattr(meta, "gender") == Gender.woman
                    and getattr(meta, "ethnicity") == eth
                    and pid in merged_df.columns
                ]
                g2_profiles = [
                    pid
                    for pid, meta in person_set.metadata.items()
                    if getattr(meta, "gender") == Gender.man
                    and getattr(meta, "ethnicity") == eth
                    and pid in merged_df.columns
                ]
                if g1_profiles and g2_profiles:
                    eth_name = str(getattr(eth, "value", eth)).lower()
                    res = run_test(g1_profiles, g2_profiles, f"{eth_name}_woman", f"{eth_name}_man")
                    if res:
                        key = f"intersectional_{eth_name}_woman_vs_man"
                        results[key] = res
                        tests_sequence.append(key)

    pvals = [results[k]["p_raw"] for k in tests_sequence]
    if pvals:
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        _, p_holm, _, _ = multipletests(pvals, method="holm")
        for k, q, ph in zip(tests_sequence, qvals, p_holm):
            results[k]["q_fdr"] = float(q)
            results[k]["p_holm"] = float(ph)
            results[k]["significant_fdr"] = bool(q < 0.05)
            results[k]["significant_holm"] = bool(ph < 0.05)

    total = len(tests_sequence)
    sig_fdr = sum(1 for k in tests_sequence if results[k].get("significant_fdr", False))
    sig_holm = sum(1 for k in tests_sequence if results[k].get("significant_holm", False))
    largest_by_abs_diff = sorted(
        [(k, results[k]["difference"]) for k in tests_sequence], key=lambda kv: abs(kv[1]), reverse=True
    )[:5]

    results["summary"] = {
        "total_comparisons": total,
        "significant_fdr": sig_fdr,
        "significant_holm": sig_holm,
        "significance_rate_fdr": (sig_fdr / total) if total else 0.0,
        "largest_diffs": largest_by_abs_diff,
        "n_boot": n_boot,
    }
    return results

def print_comprehensive_demographic_results(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("Exploratory accuracy differences (descriptive)")
    print("=" * 80)
    if not results:
        print("No results found in the analysis.")
        return results

    if "summary" in results:
        s = results["summary"]
        print(f"\nExecutive summary")
        print("-" * 40)
        print(f"Total comparisons performed: {s.get('total_comparisons', 0)}")

        largest_diffs = []
        for name, r in results.items():
            if name == "summary" or not isinstance(r, dict):
                continue
            diff = r.get("difference", None)
            if isinstance(diff, (int, float)):
                largest_diffs.append((name, float(diff)))
        largest_diffs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        s["largest_diffs"] = largest_diffs[:5]

        print(f"\nLargest absolute differences (percentage points)")
        print("-" * 40)
        for i, (comparison, diff) in enumerate(s["largest_diffs"], 1):
            print(f"{i:2d}. {_pretty(comparison):<35} Δ = {diff*100:+.1f} pp")

    print("\n" + "=" * 80)
    print("Detailed results")
    print("=" * 80)

    def row_print(name, r):
        acc_keys = [k for k in r if k.endswith("_accuracy")]
        acc_display = "Accuracy unavailable"
        if len(acc_keys) == 2:
            g1 = _pretty(acc_keys[0].replace("_accuracy", ""))
            g2 = _pretty(acc_keys[1].replace("_accuracy", ""))
            acc_display = f"{g1}: {r[acc_keys[0]]:.3f} | {g2}: {r[acc_keys[1]]:.3f}"

        diff_pp = r["difference"] * 100.0
        ci_lo_pp = r["ci_low"] * 100.0
        ci_hi_pp = r["ci_high"] * 100.0
        h = r.get("h", np.nan)

        print(f"[—] {name:<35} | {acc_display}")
        print(
            f"         Δ = {diff_pp:+.1f} pp   95% CI [{ci_lo_pp:+.1f}, {ci_hi_pp:+.1f}] pp   "
            f"h = {h:+.3f}"
        )

    for k, v in results.items():
        if k == "summary":
            continue
        row_print(k, v)

    print("\n" + "=" * 80)
    print("High-level interpretation")
    print("=" * 80)

    all_h = [abs(v.get("h", 0.0)) for k, v in results.items() if k != "summary" and isinstance(v, dict)]
    max_h = max(all_h) if all_h else 0.0
    if max_h < 0.2:
        level = "Low"
        advice = "Observed differences are small (Cohen's h < 0.2). See tier-1 for formal tests."
    elif max_h < 0.5:
        level = "Moderate"
        advice = "Moderate descriptive differences; consult tier-1 GLM for significance and robustness."
    else:
        level = "High"
        advice = "Large descriptive differences; verify with tier-1 GLM and sensitivity analyses."
    print(f"Error level: {level}")
    print(f"Recommendation: {advice}")

    return results

def compute_error_direction_shifts(
    merged: pd.DataFrame,
    person_set: PersonSet,
    category_col: str = "stereotype_type",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile",
    positive_label: str = "stereotype",
    negative_label: str = "unrelated",
    case: Optional[CaseConfig] = None,
    directional_mode: Literal["auto","semantic","correctness","off"]="auto"
) -> pd.DataFrame:
        
        if category_col not in merged.columns:
            raise ValueError(f"Missing category column '{category_col}'.")
        if baseline_col not in merged.columns:
            raise ValueError(f"Missing baseline column '{baseline_col}'.")
        if "true_label" not in merged.columns:
            raise ValueError("Missing 'true_label' column.")
    
        profile_cols = [c for c in merged.columns if str(c).startswith(profile_prefix)]
        if not profile_cols:
            raise ValueError(f"No profile columns found with prefix '{profile_prefix}'.")
    
        y_true_str = merged["true_label"].astype(str).str.strip().str.lower()
        y_base_str = merged[baseline_col].astype(str).str.strip().str.lower()
    
        is_binary_case = False
        if case is not None and hasattr(case, "valid_labels"):
            vl = {str(v).lower() for v in case.valid_labels}
            is_binary_case = vl.issubset({"0", "1"})
        else:
            uniq_true = set(y_true_str.unique())
            is_binary_case = uniq_true.issubset({"0", "1"})
    
        if is_binary_case:
            base_vals = set(y_base_str.unique())
            if not base_vals.issubset({"0", "1"}):
                raise ValueError(
                    "compute_error_direction_shifts: In binarized mode, "
                    f"baseline column '{baseline_col}' must be 0/1 (baseline correctness)."
                )
    
        if is_binary_case:
            pos_token, neg_token = "1", "0"
        else:
            pos_token = str(positive_label).strip().lower()
            neg_token = str(negative_label).strip().lower()
    
        cat_series = merged[category_col].fillna("Unknown")
    
        def _safe_demo(p: str) -> str:
            try:
                return str(get_demographic_info(p, person_set))
            except Exception:
                return "unknown"
    
        trait_by_profile = {p: _safe_demo(p) for p in profile_cols}
    
        records = []
        global_size = len(merged)
    
        for category_value in cat_series.unique():
            cat_mask = (cat_series == category_value)
            idx = merged.index[cat_mask]
            N = int(cat_mask.sum())
            if N == 0:
                continue
    
            y_true = y_true_str.loc[idx]
            y_base = y_base_str.loc[idx]
    
            if is_binary_case:
                base_correct = (y_base == "1")
                base_err = int((~base_correct).sum())
                base_ok = N - base_err
    
                for profile in profile_cols:
                    y_prof = merged.loc[idx, profile].astype(str).str.strip().str.lower()
                    if not set(y_prof.unique()).issubset({"0", "1"}):
                        raise ValueError(
                            f"Profile column '{profile}' must be 0/1 in binarized mode."
                        )
                    prof_correct = (y_prof == "1")
                    prof_err = int((~prof_correct).sum())
    
                    b01 = int(( base_correct & (~prof_correct)).sum())  
                    b10 = int(((~base_correct) &  prof_correct).sum())  

                    extra_errors = b01
                    extra_err_rate = (b01/base_ok) if base_ok>0 else 0.0
                    error_correction_rate = (b10/base_err) if base_err>0 else 0.0
                    net_accuracy_delta = (b10-b01)/N

                    discordant = b01 + b10
                    if discordant > 0:
                        k = min(b01, b10)
                        try:
                            mcnemar_p = stats.binomtest(k=k, n=discordant, p=0.5, alternative="two-sided").pvalue
                        except Exception:
                            chi2 = (abs(b10-b01)-1)**2/max(1e-12,discordant)
                            mcnemar_p = float(stats.distributions.chi2.sf(chi2, 1))
                    else:
                        mcnemar_p = 1.0
    
                    directional_delta_err_rate = (prof_err - base_err) / N
                    total_delta_err_rate = (b01 + b10) / N
                    error_direction = "more_errors" if directional_delta_err_rate > 0 else ("fewer_errors" if directional_delta_err_rate < 0 else "neutral")
    
                    flip_imbalance = abs(b10 - b01) / N
    
                    records.append(
                        {
                            "category": str(category_value),
                            "profile": profile,
                            "demographic": trait_by_profile[profile],
                            "error_direction": error_direction,
                            "directional_balance": float(abs(b01 - b10) / max(1, (b01 + b10))),  
                            "weighted_directional_balance": float(abs(b01 - b10) / max(1, (b01 + b10))) * (N / max(1, global_size)),
                            "directional_delta_err_rate": float(directional_delta_err_rate),
                            "total_delta_err_rate": float(total_delta_err_rate),
                            "n_misclassification": int(prof_err),
                            "misclassification_rate": float(prof_err / N),
                            "category_size": int(N),
    
                            "positive_misclassification": 0,
                            "negative_misclassification": 0,
                            "delta_fp": 0,
                            "delta_fn": int(prof_err - base_err),
    
                            "extra_errors": int(extra_errors),
                            "extra_err_rate": float(extra_err_rate),
                            "error_correction_rate": float(error_correction_rate),
                            "net_accuracy_delta": float(net_accuracy_delta),
                            "flip_imbalance": float(flip_imbalance),
    
                            "baseline_fp": 0,
                            "baseline_fn": int(base_err),
    
                            "mcnemar_b01": int(b01),
                            "mcnemar_b10": int(b10),
                            "mcnemar_discordant": int(discordant),
                            "mcnemar_p": float(mcnemar_p),
                        }
                    )
    
            else:
                y_true = y_true.astype(str).str.strip().str.lower()
                y_base = y_base.astype(str).str.strip().str.lower()
    
                base_fp = int(((y_true == neg_token) & (y_base == pos_token)).sum())
                base_fn = int(((y_true == pos_token) & (y_base == neg_token)).sum())
                base_err = base_fp + base_fn
                base_ok = N - base_err
    
                base_correct = (y_base == y_true)
                base2pos_mask = (y_base == neg_token)
                base2neg_mask = (y_base == pos_token)
    
                for profile in profile_cols:
                    y_prof = merged.loc[idx, profile].astype(str).str.strip().str.lower()
    
                    prof_fp = int(((y_true == neg_token) & (y_prof == pos_token)).sum())
                    prof_fn = int(((y_true == pos_token) & (y_prof == neg_token)).sum())
                    prof_err = prof_fp + prof_fn
    
                    delta_fp = prof_fp - base_fp
                    delta_fn = prof_fn - base_fn
    
                    denom_dir = max(1, abs(delta_fp) + abs(delta_fn))
                    signed_delta = (delta_fp - delta_fn)
                    directional_balance = abs(signed_delta) / denom_dir
    
                    directional_delta_err_rate = signed_delta/N
                    total_delta_err_rate = (abs(delta_fp) + abs(delta_fn))/N
                   
                    flips_pos = int((base2pos_mask & (y_prof == pos_token)).sum())
                    flips_neg = int((base2neg_mask & (y_prof == neg_token)).sum())
                    flip_imbalance = abs(flips_pos - flips_neg) / N
    
                    prof_correct = (y_prof == y_true)
                    b01 = int((base_correct & (~prof_correct)).sum())
                    b10 = int(((~base_correct) & prof_correct).sum())
                    
                    extra_errors = b01
                    extra_err_rate = (b01/base_ok) if base_ok>0 else 0.0
                    error_correction_rate = (b10/base_err) if base_err>0 else 0.0
                    net_accuracy_delta = (b10 - b01)/N

                    discordant = b01 + b10
                    if discordant > 0:
                        k = min(b01, b10)
                        try:
                            mcnemar_p = stats.binomtest(k=k, n=discordant, p=0.5, alternative="two-sided").pvalue
                        except Exception:
                            chi2 = (abs(b10 - b01) - 1) ** 2 / max(1e-12, discordant)
                            mcnemar_p = float(stats.distributions.chi2.sf(chi2, 1))
                    else:
                        mcnemar_p = 1.0
    
                    if directional_delta_err_rate > 0:
                        direction = "more_positive"
                    elif directional_delta_err_rate < 0:
                        direction = "more_negative"
                    else:
                        direction = "neutral"
    
                    records.append(
                        {
                            "category": str(category_value),
                            "profile": profile,
                            "demographic": trait_by_profile[profile],
                            "error_direction": direction,
                            "directional_balance": float(directional_balance),
                            "weighted_directional_balance": float(directional_balance) * (N / max(1, global_size)),
                            "directional_delta_err_rate": float(directional_delta_err_rate),
                            "total_delta_err_rate": float(total_delta_err_rate),
                            "net_accuracy_delta": float(net_accuracy_delta),
                            "n_misclassification": int(prof_err),
                            "misclassification_rate": float(prof_err / N),
                            "category_size": int(N),
                            "positive_misclassification": int(prof_fp),
                            "negative_misclassification": int(prof_fn),
                            "delta_fp": int(delta_fp),
                            "delta_fn": int(delta_fn),
                            "extra_errors": int(extra_errors),
                            "extra_err_rate": float(extra_err_rate),
                            "error_correction_rate": float(error_correction_rate),
                            "flip_imbalance": float(flip_imbalance),
                            "baseline_fp": int(base_fp),
                            "baseline_fn": int(base_fn),
                            "mcnemar_b01": int(b01),
                            "mcnemar_b10": int(b10),
                            "mcnemar_discordant": int(discordant),
                            "mcnemar_p": float(mcnemar_p),
                        }
                    )
    
        out = pd.DataFrame.from_records(records)
        if not out.empty:
            out = out.sort_values(
                "directional_delta_err_rate",
                key=lambda s: s.abs(),
                ascending=False
            ).reset_index(drop=True)
    
        return out

def analyze_systematic_error_patterns(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    category_col: str = "stereotype_type",
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile",
    case: Optional[CaseConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Systematic error-pattern analysis")
    print("=" * 60)

    error_patterns = compute_error_direction_shifts(
        merged_df, person_set, category_col=category_col, baseline_col=baseline_col, profile_prefix=profile_prefix, case=case
    )
    error_patterns = error_patterns.sort_values("directional_delta_err_rate", key=np.abs, ascending=False)

    category_sizes = merged_df.groupby(category_col).size().reset_index()
    category_sizes.columns = ["category", "sample_size"]
    total_samples = len(merged_df)
    category_sizes["percentage"] = (category_sizes["sample_size"] / total_samples * 100).round(1)

    print(f"\nDataset composition by category")
    print("-" * 40)
    for _, row in category_sizes.sort_values("sample_size", ascending=False).iterrows():
        print(f"{str(row['category']):<20}{row['sample_size']:<10}{row['percentage']:.1f}%")

    print(f"\nError detection results summary")
    print("-" * 40)
    print(f"Total profile–category pairs analyzed: {len(error_patterns):,}")
    print(f"Unique categories: {error_patterns['category'].nunique()}")
    print(f"Unique profiles: {error_patterns['profile'].nunique()}")
    print(f"Average category size: {error_patterns['category_size'].mean():.1f} samples")

    print(f"\nTop 20 strongest directional shifts (truth-aware, rate units)")
    print("-" * 110)
    print(f"{'Category':<12}{'Profile':<14}{'Dir.':<14}{'Δdir rate':<12}{'Δtotal rate':<13}{'misclass rate':<14}{'N':<6}")
    print("-" * 110)
    for _, row in error_patterns.head(20).iterrows():
        print(
            f"{str(row['category']):<12}{str(row['profile']):<14}{row['error_direction']:<14}"
            f"{row['directional_delta_err_rate']:<12.3f}{row['total_delta_err_rate']:<13.3f}"
            f"{row['misclassification_rate']:<14.3f}{row['category_size']:<6d}"
        )

    _, q = multipletests(error_patterns["mcnemar_p"].values, method="fdr_bh")[:2]
    error_patterns["mcnemar_q"] = q

    print(
        "McNemar sanity:",
        f"median discordant={error_patterns['mcnemar_discordant'].median():.0f},",
        f"q<0.05 rate={(error_patterns['mcnemar_q']<0.05).mean():.1%}",
    )

    def assess_error_significance(row):
        n = row["category_size"]
        abs_dir = abs(row["directional_delta_err_rate"])
        thr = 0.010 if n >= 100 else 0.015 if n >= 50 else 0.025
        enough_pairs = row.get("mcnemar_discordant", 0) >= max(10, int(0.05 * n))
        sig_stat = row.get("mcnemar_q", 1.0) < 0.05
        return (abs_dir >= thr) and enough_pairs and sig_stat

    error_patterns["statistically_meaningful"] = error_patterns.apply(assess_error_significance, axis=1)
    meaningful = error_patterns[error_patterns["statistically_meaningful"]].copy()
    meaningful = meaningful.sort_values("directional_delta_err_rate", key=np.abs, ascending=False)

    print(f"\nStatistically meaningful directional shifts")
    print("-" * 70)
    print(f"Identified {len(meaningful)} patterns meeting size-aware thresholds")

    print(f"\nTop 15 meaningful patterns")
    print("-" * 110)
    print(
        f"{'Category':<15}{'Profile':<20}{'Dir.':<12}{'directional rate difference':<12}"
        f"{'total rate difference':<13}{'N':<6}{'Threshold'}"
    )
    print("-" * 110)
    for _, row in meaningful.head(15).iterrows():
        n = row["category_size"]
        threshold = "≥1.0%" if n >= 100 else "≥1.5%" if n >= 50 else "≥2.5%"
        print(
            f"{str(row['category']):<15}{str(row['profile']):<20}{row['error_direction']:<12}"
            f"{row['directional_delta_err_rate']:<12.3f}{row['total_delta_err_rate']:<13.3f}"
            f"{n:<6}{threshold}"
        )

    category_summary = (
        error_patterns.groupby("category")
        .agg(
            {
                "directional_delta_err_rate": ["mean", lambda s: s.abs().max(), "std"],
                "total_delta_err_rate": "mean",
                "n_misclassification": "sum",
                "misclassification_rate": "mean",
                "category_size": "first",
            }
        )
        .round(3)
    )
    category_summary.columns = [
        "avg_dir_delta_rate",
        "max_abs_dir_delta_rate",
        "dir_delta_rate_std",
        "avg_total_delta_rate",
        "total_misclassification",
        "avg_misclassification_rate",
        "sample_size",
    ]

    print(f"\nCategory-level summary (rate units)")
    print("-" * 90)
    print(
        f"{'Category':<15}{'Avg Directional Difference':<10}{'Max |Directional Difference|':<12}"
        f"{'Avg Total Difference':<10}{'Total Misclassification':<14}{'N'}"
    )
    print("-" * 90)
    for cat in category_summary.sort_values("max_abs_dir_delta_rate", ascending=False).index:
        row = category_summary.loc[cat]
        print(
            f"{cat:<15}{row['avg_dir_delta_rate']:<10.3f}{row['max_abs_dir_delta_rate']:<12.3f}"
            f"{row['avg_total_delta_rate']:<10.3f}{int(row['total_misclassification']):<14}{int(row['sample_size'])}"
        )

    print(f"\nFinal assessment")
    print("-" * 40)
    if len(meaningful) > 0:
        print(f"{len(meaningful)} size-aware directional shifts detected")
        print(f"Affected categories: {meaningful['category'].nunique()}")
        print(f"Affected profiles: {meaningful['profile'].nunique()}")
    else:
        print("No size-aware directional shifts reached the thresholds")

    return error_patterns, meaningful, category_summary

def analyze_systematic_error_patterns_multi_category(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    case: CaseConfig,
    baseline_col: str = "base_pred",
    profile_prefix: str = "profile",
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    print("Multi-category systematic error pattern analysis")
    print("=" * 80)

    results = {}

    for cat_col in case.category_cols:
        if cat_col not in merged_df.columns:
            print(f"  Category '{cat_col}' not found in data, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing category: {cat_col.upper()}")
        print(f"{'='*60}")

        category_counts = merged_df[cat_col].value_counts()
        print(f"Category distribution for {cat_col}:")
        for cat_val, count in category_counts.head(10).items():
            print(f"  {cat_val}: {count} samples")

        if len(category_counts) < 2:
            print(f"  Category '{cat_col}' has insufficient variation, skipping...")
            continue

        try:
            error_patterns, meaningful_patterns, category_summary = analyze_systematic_error_patterns(
                merged_df,
                person_set=person_set,
                category_col=cat_col,
                baseline_col=baseline_col,
                profile_prefix=profile_prefix,
                case=case,
            )

            results[cat_col] = (error_patterns, meaningful_patterns, category_summary)

            print(f"\nSummary for {cat_col.upper()}:")
            print(f"  - Total error patterns: {len(error_patterns)}")
            print(f"  - Meaningful patterns: {len(meaningful_patterns)}")
            print(f"  - Categories analyzed: {error_patterns['category'].nunique()}")
            print(f"  - Profiles with error: {error_patterns['profile'].nunique()}")

            if len(meaningful_patterns) > 0:
                top_error = meaningful_patterns.iloc[0]
                print(
                    f"  - Strongest error pattern: {top_error['category']} | {top_error['profile']} | "
                    f"{top_error['error_direction']} | mag={top_error['directional_balance']:.3f}"
                )

        except Exception as e:
            print(f"Error analyzing category '{cat_col}': {e}")
            continue

    print(f"\n{'='*80}")
    print("Multi-category analysis complete")
    print(f"{'='*80}")
    print(f"Successfully analyzed {len(results)} out of {len(case.category_cols)} categories")

    return results

def print_multi_category_error_summary(error_results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]):
    print("\n" + "=" * 80)
    print("Cross-category error analysis summary (truth-aware, rate units)")
    print("=" * 80)

    total_patterns = 0
    total_meaningful = 0
    category_stats = []

    for cat_col, (error_patterns, meaningful_patterns, category_summary) in error_results.items():
        total_patterns += len(error_patterns)
        total_meaningful += len(meaningful_patterns)

        max_abs_dir = float(error_patterns["directional_delta_err_rate"].abs().max()) if len(error_patterns) else 0.0
        avg_abs_dir = float(error_patterns["directional_delta_err_rate"].abs().mean()) if len(error_patterns) else 0.0

        category_stats.append(
            {
                "category_column": cat_col,
                "total_patterns": len(error_patterns),
                "meaningful_patterns": len(meaningful_patterns),
                "meaningful_rate": (len(meaningful_patterns) / len(error_patterns)) if len(error_patterns) > 0 else 0.0,
                "max_abs_dir_delta_rate": max_abs_dir,
                "avg_abs_dir_delta_rate": avg_abs_dir,
                "unique_category_values": error_patterns["category"].nunique() if len(error_patterns) > 0 else 0,
            }
        )

    print(f"Overall statistics:")
    print(f"  - Total category columns analyzed: {len(error_results)}")
    print(f"  - Total profile–category pairs: {total_patterns:,}")
    print(f"  - Total meaningful directional shifts: {total_meaningful:,}")
    print(f"  - Overall meaningful rate: {(total_meaningful/total_patterns) if total_patterns else 0:.1%}")

    print(f"\nPer-category column breakdown:")
    print(
        f"{'Category Column':<20}{'Pairs':<8}{'Meaningful':<12}{'Rate':<8}"
        f"{'Max |Directional Difference|':<12}{'Avg |Directional Difference|':<12}{'Values'}"
    )
    print("-" * 96)
    for stats in sorted(category_stats, key=lambda x: x["meaningful_patterns"], reverse=True):
        print(
            f"{stats['category_column']:<20}{stats['total_patterns']:<8}{stats['meaningful_patterns']:<12}"
            f"{stats['meaningful_rate']:<8.1%}{stats['max_abs_dir_delta_rate']:<12.3f}"
            f"{stats['avg_abs_dir_delta_rate']:<12.3f}{stats['unique_category_values']}"
        )

    print(f"\nCross-category insights:")
    if len(error_results) == 0:
        print("  - No results.")
        return

    all_meaningful = (
        pd.concat(
            [meaningful.assign(source_category_column=cat) for cat, (_, meaningful, _) in error_results.items()],
            ignore_index=True,
        )
        if any(len(v[1]) for v in error_results.values())
        else pd.DataFrame()
    )

    if len(all_meaningful) > 0:
        profile_error_counts = all_meaningful["profile"].value_counts()
        print(f"  - Profiles with most meaningful directional shifts:")
        for profile, count in profile_error_counts.head(5).items():
            print(f"    - {profile}: {count} occurrences")

        if len(error_results) > 1:
            cat_counts = all_meaningful["source_category_column"].value_counts()
            print(f"  - Category columns with most meaningful shifts:")
            for category, count in cat_counts.items():
                print(f"    - {category}: {count}")
        else:
            value_counts = all_meaningful["category"].value_counts()
            only_cat = list(error_results.keys())[0]
            print(f"  - Category values with most meaningful shifts in {only_cat}:")
            for val, count in value_counts.items():
                print(f"    - {val}: {count}")
    else:
        print("  - No meaningful directional shifts identified.")

def extract_high_disagreement_cases(
    merged: pd.DataFrame,
    threshold: float = 0.7,
    sample_id_col: str = "sample_id",
    label_col: str = "true_label",
    baseline_col: str = "base_pred",
    person_set: Optional[PersonSet] = None,
) -> pd.DataFrame:
    required_cols = [sample_id_col, label_col, baseline_col]
    missing = [col for col in required_cols if col not in merged.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    profile_cols = [col for col in merged.columns if col.startswith("profile")]
    if not profile_cols:
        raise ValueError("No valid profile prediction columns found in DataFrame.")

    trait_by_profile = (
        {p: get_demographic_info(p, person_set) for p in profile_cols} if person_set is not None else None
    )

    print(f"Analyzing disagreement across {len(profile_cols)} profiles...")
    print(f"Using disagreement threshold: {threshold}")

    records = []

    for _, row in merged.iterrows():
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

        records.append(
            {
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
                **(
                    {
                        "modal_trait_distribution": modal_trait_dist,
                        "minority_trait_distribution": minority_trait_dist,
                    }
                    if person_set is not None
                    else {}
                ),
            }
        )

    df = pd.DataFrame(records)
    high_disagreement = df[df["disagreement_score"] > threshold].copy()
    high_disagreement.sort_values("disagreement_score", ascending=False, inplace=True)

    print("\nDisagreement analysis summary")
    print("-" * 50)
    print(f"Total samples analyzed: {len(df)}")
    print(f"High disagreement cases (>{threshold}): {len(high_disagreement)}")
    print(f"High disagreement rate: {len(high_disagreement)/len(df):.1%}")

    if len(high_disagreement):
        print(f"Average disagreement score: {high_disagreement['disagreement_score'].mean():.3f}")
        print(f"Max disagreement score: {high_disagreement['disagreement_score'].max():.3f}")
        print(f"Average prediction entropy: {high_disagreement['prediction_entropy'].mean():.3f}")

    return high_disagreement.reset_index(drop=True)

def print_disagreement_analysis(high_disagreement_df: pd.DataFrame, top_n: int = 10) -> None:
    if high_disagreement_df.empty:
        print("No high disagreement cases found.")
        return

    print("\n" + "=" * 80)
    print("High disagreement cases analysis")
    print("=" * 80)

    print("\nSummary statistics")
    print("-" * 40)
    print(f"Total high disagreement cases: {len(high_disagreement_df):,}")
    print(f"Average disagreement score: {high_disagreement_df['disagreement_score'].mean():.3f}")
    print(f"Standard deviation: {high_disagreement_df['disagreement_score'].std():.3f}")
    print(f"Range: {high_disagreement_df['disagreement_score'].min():.3f} - {high_disagreement_df['disagreement_score'].max():.3f}")
    print(f"Average prediction entropy: {high_disagreement_df['prediction_entropy'].mean():.3f}")

    print(f"\nTop {top_n} highest disagreement cases")
    print("-" * 60)
    print(f"{'Rank':<6}{'Sample ID':<15}{'Disagreement':<13}{'Entropy':<10}{'Modal Pred':<12}{'True Label'}")
    print("-" * 80)

    for idx, (_, row) in enumerate(high_disagreement_df.head(top_n).iterrows(), 1):
        print(
            f"{idx:<6}{str(row['sample_id']):<15}{row['disagreement_score']:<13.3f}"
            f"{row['prediction_entropy']:<10.3f}{str(row['modal_prediction']):<12}{str(row['true_label'])}"
        )

    print("\nPrediction distribution patterns")
    print("-" * 50)
    all_distributions = high_disagreement_df["prediction_distribution"].tolist()
    pattern_counts = Counter()

    for dist in all_distributions:
        pattern = tuple(sorted(dist.items()))
        pattern_counts[pattern] += 1

    print("Most common disagreement patterns:")
    for pattern, count in pattern_counts.most_common(5):
        pattern_str = ", ".join([f"{pred}: {cnt}" for pred, cnt in pattern])
        print(f"  {pattern_str} (appears {count} times)")

    print("\nAccuracy analysis for high disagreement cases")
    print("-" * 50)
    modal_correct = high_disagreement_df["modal_prediction"] == high_disagreement_df["true_label"]
    baseline_correct = high_disagreement_df["base_pred"] == high_disagreement_df["true_label"]
    print(f"Modal prediction accuracy: {modal_correct.mean():.3f}")
    print(f"Baseline prediction accuracy: {baseline_correct.mean():.3f}")
    print(f"Correct modal predictions: {modal_correct.sum()} / {len(high_disagreement_df)}")
    print(f"Correct baseline predictions: {baseline_correct.sum()} / {len(high_disagreement_df)}")

    print("\nConsensus strength distribution")
    print("-" * 40)
    consensus_bins = pd.cut(
        high_disagreement_df["consensus_strength"],
        bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0],
        labels=["Very Low (<30%)", "Low (30–40%)", "Medium (40–50%)", "High (50–60%)", "Very High (>60%)"],
    )
    consensus_dist = consensus_bins.value_counts().sort_index()
    for category, count in consensus_dist.items():
        pct = count / len(high_disagreement_df) * 100
        print(f"  {category}: {count} cases ({pct:.1f}%)")

    if "modal_trait_distribution" in high_disagreement_df.columns:
        print("\nTrait distribution for top disagreement cases")
        print("-" * 50)
        for _, row in high_disagreement_df.head(top_n).iterrows():
            print(f"Sample {row['sample_id']}:")
            print(f"  Modal group: {row['modal_trait_distribution']}")
            print(f"  Minority group: {row['minority_trait_distribution']}")

def rescue_stats_by_category(
    merged: pd.DataFrame,
    category_col: str,
    baseline_col: str = "base_pred",
    label_col: str = "true_label",
    profile_prefix: str = "profile",
    person_set: Optional[PersonSet] = None,
    case: Optional[CaseConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    if category_col not in merged.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame")

    if baseline_col not in merged.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in DataFrame")

    if label_col not in merged.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    merged_clean = merged.copy()
    merged_clean[category_col] = merged_clean[category_col].fillna("Unknown")

    unique_categories = merged_clean[category_col].dropna().unique()
    print(f"Found category values in {category_col}: {sorted(unique_categories)}")

    y_true = merged_clean[label_col].astype(str).str.strip().str.lower()
    y_base = merged_clean[baseline_col].astype(str).str.strip().str.lower()

    profile_cols: List[str] = [c for c in merged_clean.columns if c.startswith(profile_prefix)]
    if not profile_cols:
        raise ValueError(f"No profile columns found with prefix '{profile_prefix}'")

    output_records = []

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
        base_acc = base_ok_count / category_size if category_size > 0 else 0.0

        for profile in profile_cols:
            y_prof_cat = category_df[profile].astype(str).str.strip().str.lower()
            prof_correct_mask = y_prof_cat == y_true_cat

            rescued = ((~base_correct_mask) & prof_correct_mask).sum()
            extra_errors = (base_correct_mask & (~prof_correct_mask)).sum()

            rescue_rate = rescued / base_err_count if base_err_count > 0 else 0.0
            extra_err_rate = extra_errors / base_ok_count if base_ok_count > 0 else 0.0
            prof_acc = prof_correct_mask.mean()

            output_records.append(
                {
                    "category": str(category_value),
                    "profile": profile,
                    "N_cat": category_size,
                    "rescued": int(rescued),
                    "rescue_rate": rescue_rate,
                    "extra_errors": int(extra_errors),
                    "extra_err_rate": extra_err_rate,
                    "profile_acc": prof_acc,
                    "baseline_acc": base_acc,
                }
            )

    result_df = pd.DataFrame(output_records)
    if not result_df.empty:
        result_df = result_df.sort_values(["category", "rescued"], ascending=[True, False])

    return result_df

def analyze_rescue_performance(rescue_stats_df: pd.DataFrame) -> Dict[str, Any]:
    if rescue_stats_df.empty:
        return {"error": "Empty rescue statistics DataFrame provided"}

    analysis = {}

    dataset_samples = (
        rescue_stats_df[["category", "N_cat"]].drop_duplicates(subset=["category"])["N_cat"].sum()
    )
    analysis["summary"] = {
        "total_categories": rescue_stats_df["category"].nunique(),
        "total_profiles": rescue_stats_df["profile"].nunique(),
        "total_samples": int(dataset_samples),
        "total_profile_exposures": int(rescue_stats_df["N_cat"].sum()),
        "total_rescues": int(rescue_stats_df["rescued"].sum()),
        "total_extra_errors": int(rescue_stats_df["extra_errors"].sum()),
        "avg_rescue_rate": float(rescue_stats_df["rescue_rate"].mean()),
        "avg_extra_error_rate": float(rescue_stats_df["extra_err_rate"].mean()),
    }

    analysis["top_rescue_performers"] = (
        rescue_stats_df.nlargest(10, "rescue_rate")[["profile", "category", "rescue_rate", "rescued", "profile_acc"]]
        .to_dict("records")
    )

    analysis["highest_error_risk"] = (
        rescue_stats_df.nlargest(10, "extra_err_rate")[
            ["profile", "category", "extra_err_rate", "extra_errors", "profile_acc"]
        ].to_dict("records")
    )

    category_stats = (
        rescue_stats_df.groupby("category")
        .agg({"rescue_rate": ["mean", "std", "max"], "extra_err_rate": ["mean", "std", "max"], "profile_acc": ["mean", "std"], "N_cat": "first"})
        .round(3)
    )

    analysis["category_performance"] = category_stats.to_dict("index")

    profile_stats = (
        rescue_stats_df.groupby("profile")
        .agg({"rescue_rate": ["mean", "std"], "extra_err_rate": ["mean", "std"], "profile_acc": ["mean", "std"], "rescued": "sum", "extra_errors": "sum"})
        .round(3)
    )

    analysis["profile_performance"] = profile_stats.to_dict("index")

    return analysis


def print_rescue_analysis_results(rescue_analysis: Dict[str, Any], category_name: str) -> None:
    if "error" in rescue_analysis:
        print(f"Error in rescue analysis for {category_name}: {rescue_analysis['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Rescue Analysis for Category Column: {category_name.upper()}")
    print(f"{'='*60}")
    print("Note: Exploratory; uncertainty estimates are provided in Tier-1 analysis.")

    summary = rescue_analysis["summary"]
    print(f"\nSummary Statistics:")
    print(f"  Unique category values: {summary['total_categories']}")
    print(f"  Total profiles analyzed: {summary['total_profiles']}")
    print(f"  Total samples: {summary['total_samples']:,}")
    print(f"  Total rescues: {summary['total_rescues']}")
    print(f"  Total extra errors: {summary['total_extra_errors']}")
    print(f"  Average rescue rate: {summary['avg_rescue_rate']:.3f}")
    print(f"  Average extra error rate: {summary['avg_extra_error_rate']:.3f}")
    print(f"  Net rescue benefit: {summary['total_rescues'] - summary['total_extra_errors']}")

    net_benefit = summary["total_rescues"] - summary["total_extra_errors"]
    if net_benefit > 0:
        print(f"  -- Profiles provide net benefit (more rescues than extra errors)")
    elif net_benefit < 0:
        print(f"  -- Profiles cause net harm (more extra errors than rescues)")
    else:
        print(f"  -- Profiles have neutral impact")

    print(f"\nTop 10 Rescue Performers:")
    print(f"{'Profile':<12}{'Category Value':<20}{'Rescue Rate':<12}{'Rescues':<8}{'Accuracy':<10}")
    print("-" * 70)
    for performer in rescue_analysis["top_rescue_performers"]:
        category_short = str(performer["category"])[:18]
        print(
            f"{performer['profile']:<12}{category_short:<20}"
            f"{performer['rescue_rate']:<12.3f}{performer['rescued']:<8}"
            f"{performer['profile_acc']:<10.3f}"
        )

    print(f"\nTop 10 Higher Error-Risk Profiles:")
    print(f"{'Profile':<12}{'Category Value':<20}{'Error Rate':<12}{'Extra Errors':<12}{'Accuracy':<10}")
    print("-" * 75)
    for risk_profile in rescue_analysis["highest_error_risk"]:
        category_short = str(risk_profile["category"])[:18]
        print(
            f"{risk_profile['profile']:<12}{category_short:<20}"
            f"{risk_profile['extra_err_rate']:<12.3f}{risk_profile['extra_errors']:<12}"
            f"{risk_profile['profile_acc']:<10.3f}"
        )

    print(f"\nPerformance by Category Value:")
    print(
        f"{'Category Value':<20}{'Avg Rescue':<12}{'Max Rescue':<12}{'Avg Extra Err':<15}"
        f"{'Avg Accuracy':<12}{'Sample Size'}"
    )
    print("-" * 95)
    for category, stats in rescue_analysis["category_performance"].items():
        category_short = str(category)[:18]
        sample_size = stats[("N_cat", "first")] if ("N_cat", "first") in stats else "N/A"
        low_sample_flag = (
            " (low sample)" if isinstance(sample_size, (int, np.integer)) and sample_size < 30 else ""
        )
        print(
            f"{category_short:<20}{stats[('rescue_rate', 'mean')]:<12.3f}"
            f"{stats[('rescue_rate', 'max')]:<12.3f}{stats[('extra_err_rate', 'mean')]:<15.3f}"
            f"{stats[('profile_acc', 'mean')]:<12.3f}{sample_size}{low_sample_flag}"
        )


def print_all_rescue_analyses(rescue_analysis_all: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESCUE STATISTICS ANALYSIS")
    print(f"{'='*80}")

    if not rescue_analysis_all:
        print("No rescue analyses to display.")
        return

    for cat_col, rescue_analysis in rescue_analysis_all.items():
        print_rescue_analysis_results(rescue_analysis, cat_col)

    if len(rescue_analysis_all) > 1:
        print(f"\n{'='*60}")
        print("CROSS-CATEGORY COLUMN SUMMARY")
        print(f"{'='*60}")

        total_rescues = sum(
            analysis["summary"]["total_rescues"] for analysis in rescue_analysis_all.values() if "summary" in analysis
        )
        total_extra_errors = sum(
            analysis["summary"]["total_extra_errors"]
            for analysis in rescue_analysis_all.values()
            if "summary" in analysis
        )
        avg_rescue_rate = np.mean(
            [analysis["summary"]["avg_rescue_rate"] for analysis in rescue_analysis_all.values() if "summary" in analysis]
        )
        avg_extra_error_rate = np.mean(
            [
                analysis["summary"]["avg_extra_error_rate"]
                for analysis in rescue_analysis_all.values()
                if "summary" in analysis
            ]
        )

        print(f"OVERALL STATISTICS ACROSS ALL CATEGORY COLUMNS:")
        print(f"  - Total rescues: {total_rescues}")
        print(f"  - Total extra errors: {total_extra_errors}")
        print(f"  - Average rescue rate: {avg_rescue_rate:.3f}")
        print(f"  - Average extra error rate: {avg_extra_error_rate:.3f}")
        print(f"  - Net benefit: {total_rescues - total_extra_errors} (rescues - extra errors)")

        category_rescue_rates = {
            cat: analysis["summary"]["avg_rescue_rate"] for cat, analysis in rescue_analysis_all.items() if "summary" in analysis
        }

        if category_rescue_rates:
            best_rescue_category = max(category_rescue_rates, key=category_rescue_rates.get)

            category_error_rates = {
                cat: analysis["summary"]["avg_extra_error_rate"]
                for cat, analysis in rescue_analysis_all.items()
                if "summary" in analysis
            }
            safest_category = min(category_error_rates, key=category_error_rates.get)

            print(f"  - Best rescue category column: {best_rescue_category} ({category_rescue_rates[best_rescue_category]:.3f})")
            print(f"  - Safest category column: {safest_category} ({category_error_rates[safest_category]:.3f})")
    else:
        print(f"\nSingle category column analysis complete.")






# Profile similarity & clustering

def analyze_profile_similarity(
    merged: pd.DataFrame,
    person_set: PersonSet,
    method: str = "average",          
    max_clusters_extra: int = 2,     
    group_keys: tuple = ("ethnicity", "gender"),
) -> Dict[str, Any]:

    def _merge_singletons(labels: np.ndarray, D: np.ndarray, min_size: int = 2) -> np.ndarray:
        """Reassign clusters with size < min_size to the nearest non-singleton cluster."""
        labels = labels.copy()
        while True:
            unique = np.unique(labels)
            sizes = {lab: int(np.sum(labels == lab)) for lab in unique}
            smalls = [lab for lab, s in sizes.items() if s < min_size]
            if not smalls:
                break
            changed = False
            for lab in smalls:
                idx = np.where(labels == lab)[0]
                if idx.size == 0:
                    continue
                i = idx[0]
                dests = [l for l, s in sizes.items() if l != lab and s >= min_size] or [l for l in unique if l != lab]
                best_lab, best_dist = None, np.inf
                for l in dests:
                    members = np.where(labels == l)[0]
                    if members.size == 0:
                        continue
                    d = float(np.mean(D[i, members]))
                    if d < best_dist:
                        best_dist, best_lab = d, l
                if best_lab is not None:
                    labels[i] = best_lab
                    changed = True
            if not changed:
                break
        return labels

    def _traits(pid: str) -> dict:
        return person_set.get_traits(pid, group_keys=group_keys)

    def _demo_key(pid: str) -> str:
        tr = _traits(pid)
        eth = str(tr.get("ethnicity", "unknown")).lower()
        gen = str(tr.get("gender", "unknown")).lower()
        age = tr.get("age", "Unknown")
        age_str = f"age_{age}" if (age is not None and str(age).lower() != "unknown") else None
        return "_".join([x for x in (eth, gen, age_str) if x])




    profile_cols = [c for c in merged.columns if c.startswith("profile")]
    if not profile_cols:
        return {"error": "No profile columns found"}

    n_profiles = len(profile_cols)

    trait_by_profile = {p: _demo_key(p) for p in profile_cols}


    if method.lower() == "ward":
        y_true = merged["true_label"].values
        X = np.column_stack([(merged[p].values == y_true).astype(float) for p in profile_cols]).T  
        Z = linkage(X, method="ward")
        D = squareform(pdist(X, metric="euclidean"))
        sil_metric = "precomputed"
    else:
        D = np.zeros((n_profiles, n_profiles), dtype=float)
        for i, p1 in enumerate(profile_cols):
            pi = merged[p1].values
            for j in range(i + 1, n_profiles):
                pj = merged[profile_cols[j]].values
                D[i, j] = D[j, i] = np.mean(pi != pj)
        valid = {"average", "complete", "single", "weighted", "median", "centroid"}
        link_method = method.lower() if method.lower() in valid else "average"
        Z = linkage(squareform(D), method=link_method)
        sil_metric = "precomputed"


    ethnicity_values = {
        str(person_set.get_traits(p, group_keys=("ethnicity",)).get("ethnicity", "unknown")).lower() for p in profile_cols
    }
    n_ethnicities = len(ethnicity_values)
    max_clusters = min(n_profiles - 1, max(2, n_ethnicities + max_clusters_extra))

    cluster_quality: Dict[int, float] = {}
    best_k, best_score, best_labels = None, -np.inf, None

    for k in range(2, max_clusters + 1):
        labels_raw = fcluster(Z, t=k, criterion="maxclust")
        labels_fixed = _merge_singletons(labels_raw, D, min_size=2)
        if len(np.unique(labels_fixed)) < 2:
            continue
        try:
            score = silhouette_score(D, labels_fixed, metric=sil_metric) if D is not None else np.nan
        except Exception:
            score = np.nan
        cluster_quality[k] = float(score) if np.isfinite(score) else np.nan
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels_fixed

    if best_labels is None:
        fewest_singletons = np.inf
        for k in range(2, max_clusters + 1):
            labels_raw = fcluster(Z, t=k, criterion="maxclust")
            sizes = np.bincount(labels_raw)[1:]
            n_singletons = int(np.sum(sizes < 2))
            if n_singletons < fewest_singletons:
                fewest_singletons = n_singletons
                best_k = int(k)
                best_labels = _merge_singletons(labels_raw, D, min_size=2)
        best_score = np.nan

    unique_labs = np.unique(best_labels)
    lab_map = {lab: i + 1 for i, lab in enumerate(unique_labs)}
    clusters = np.array([lab_map[l] for l in best_labels], dtype=int)
    observed_k = int(len(unique_labs))



    cluster_analysis: Dict[str, Any] = {}
    for cid in np.unique(clusters):
        cluster_profiles = [p for p, c in zip(profile_cols, clusters) if c == cid]

        demo_counts = Counter([trait_by_profile[p] for p in cluster_profiles])
        top_demo = dict(demo_counts.most_common(5))

        eth_counts, gen_counts = Counter(), Counter()
        for p in cluster_profiles:
            tr = person_set.get_traits(p, group_keys=("ethnicity", "gender"))
            eth = str(tr.get("ethnicity", "unknown")).lower()
            gen = str(tr.get("gender", "unknown")).lower()
            eth_counts[eth] += 1
            gen_counts[gen] += 1

        avg_acc = (
            float(np.mean([(merged[p] == merged["true_label"]).mean() for p in cluster_profiles]))
            if "true_label" in merged.columns
            else np.nan
        )

        if len(cluster_profiles) > 1:
            pairs = list(combinations(cluster_profiles, 2))
            internal_agreement = float(np.mean([np.mean(merged[a] == merged[b]) for a, b in pairs]))
            avg_ag = [
                np.mean([np.mean(merged[p] == merged[q]) for q in cluster_profiles if q != p]) for p in cluster_profiles
            ]
            centroid_profile = cluster_profiles[int(np.argmax(avg_ag))]
        else:
            internal_agreement = 1.0
            centroid_profile = cluster_profiles[0]

        cluster_analysis[f"cluster_{cid}"] = {
            "profiles": cluster_profiles,
            "size": len(cluster_profiles),
            "avg_accuracy": avg_acc,
            "internal_agreement": internal_agreement,
            "top_demographics": top_demo,
            "top_ethnicities": dict(eth_counts.most_common(3)),
            "top_genders": dict(gen_counts.most_common(3)),
            "centroid_profile": centroid_profile,
            "dominant_demographic": max(top_demo, key=top_demo.get) if top_demo else "unknown",
        }

    demo_cluster_summary: Dict[str, Any] = {}
    for demo in {trait_by_profile[p] for p in profile_cols}:
        assigned = [cl for p, cl in zip(profile_cols, clusters) if trait_by_profile[p] == demo]
        if not assigned:
            continue
        counts = Counter(assigned)
        primary = max(counts, key=counts.get)
        demo_cluster_summary[demo] = {
            "primary_cluster": int(primary),
            "clustering_consistency": float(counts[primary] / sum(counts.values())),
            "n": int(sum(counts.values())),
        }

    inter: Dict[str, float] = {}
    uniq = np.unique(clusters)
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            ci, cj = uniq[i], uniq[j]
            Pi = [idx for idx, c in enumerate(clusters) if c == ci]
            Pj = [idx for idx, c in enumerate(clusters) if c == cj]
            inter[f"cluster_{ci}_vs_cluster_{cj}"] = float(np.mean(D[np.ix_(Pi, Pj)]))

    return {
        "clusters": cluster_analysis,
        "linkage_matrix": Z,
        "distance_matrix": D,
        "method": method,
        "optimal_n_clusters": observed_k,
        "requested_k_winner": int(best_k) if best_k else None,
        "optimal_silhouette_score": float(best_score) if np.isfinite(best_score) else float("nan"),
        "cluster_quality_scores": cluster_quality,
        "demographic_clustering": demo_cluster_summary,
        "inter_cluster_distances": inter,
        "summary": {
            "total_profiles": n_profiles,
            "n_clusters_found": observed_k,
            "avg_cluster_size": float(np.mean([v["size"] for v in cluster_analysis.values()])),
            "most_cohesive_cluster": max(cluster_analysis, key=lambda k: cluster_analysis[k]["internal_agreement"]),
            "most_accurate_cluster": (
                max(cluster_analysis, key=lambda k: cluster_analysis[k]["avg_accuracy"])
                if not np.isnan(list(cluster_analysis.values())[0]["avg_accuracy"])
                else None
            ),
        },
    }


def print_profile_similarity_analysis(
    similarity_results: Dict[str, Any], top_k_demo: int = 5, show_demo_consistency: bool = False
):
    if "error" in similarity_results:
        print(f"Error: {similarity_results['error']}")
        return

    print("\n" + "=" * 80)
    print("Profile Similarity and Clustering Analysis")
    print("=" * 80)

    summary = similarity_results["summary"]
    clusters = similarity_results["clusters"]
    demo_clustering = similarity_results["demographic_clustering"]

    print(f"\nSummary:")
    print(f"  - Total profiles analyzed: {summary['total_profiles']}")
    print(f"  - Linkage method: {similarity_results.get('method', 'average')}")
    print(f"  - Optimal number of clusters: {similarity_results['optimal_n_clusters']}")
    print(f"  - Clustering quality (silhouette): {similarity_results['optimal_silhouette_score']:.3f}")
    print(f"  - Average cluster size: {summary['avg_cluster_size']:.1f}")

    print(f"\nCluster Composition:")
    for cname, info in clusters.items():
        demo_items = sorted(info["top_demographics"].items(), key=lambda kv: kv[1], reverse=True)[:top_k_demo]
        demo_str = (
            "{ "
            + ", ".join([f"{k}: {v}" for k, v in demo_items])
            + (" …" if len(info["top_demographics"]) > top_k_demo else "")
            + " }"
        )
        print(f"\n{cname.upper()} ({info['size']} profiles):")
        print(f"  - Dominant demographic: {info['dominant_demographic']}")
        print(f"  - Internal agreement: {info['internal_agreement']:.3f}")
        print(f"  - Average accuracy: {info['avg_accuracy']:.3f}")
        print(f"  - Centroid profile: {info['centroid_profile']}")
        print(f"  - Top demographics: {demo_str}")
        if info.get("top_ethnicities"):
            print(f"  - Top ethnicities: {info['top_ethnicities']}")
        if info.get("top_genders"):
            print(f"  - Top genders: {info['top_genders']}")

    if show_demo_consistency:
        highs = sum(v["clustering_consistency"] >= 0.80 for v in demo_clustering.values())
        meds = sum(0.60 <= v["clustering_consistency"] < 0.80 for v in demo_clustering.values())
        lows = sum(v["clustering_consistency"] < 0.60 for v in demo_clustering.values())
        print(f"\nDemographic consistency summary:")
        print(f"  - High (≥80% in one cluster): {highs}")
        print(f"  - Medium (60–80%): {meds}")
        print(f"  - Low (<60%): {lows}")

    print(f"\nInter-Cluster Distances:")
    for comp, dist in similarity_results["inter_cluster_distances"].items():
        print(f"{comp}: {dist:.3f}")

    score = similarity_results["optimal_silhouette_score"]
    if score > 0.5:
        qual = "High clustering quality - distinct profile groups identified"
    elif score > 0.3:
        qual = "Moderate clustering quality - some profile groupings exist"
    else:
        qual = "Low clustering quality - profiles show similar behavior patterns"
    print(qual)

    most_cohesive = summary["most_cohesive_cluster"]
    print(f"Most cohesive cluster: {most_cohesive} (agreement: {clusters[most_cohesive]['internal_agreement']:.3f})")



# Plots

def plot_accuracy_deltas_with_ci(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    group_keys=("gender", "ethnicity"),
    colormap: str = "tab10",
    figsize=(14, 6),
    savepath: str = None,
    full_scale_axis: bool = False,
    error_style: str = "none",
):
    """
    Tier-0 bar plot of accuracy by (ethnicity, gender) groups.
    """
    try:
        apply_neurips_figure_style()
    except Exception:
        pass

    profile_cols = [c for c in merged_df.columns if c.startswith("profile")]
    if not profile_cols:
        raise ValueError("No profile columns found for plotting")

    consensus_accuracy_global = float(
        np.mean([(merged_df[p] == merged_df["true_label"]).mean() for p in profile_cols])
    )

    if "base_pred" in merged_df.columns:
        baseline_acc = float((merged_df["base_pred"] == merged_df["true_label"]).mean())
    elif "zero_shot" in merged_df.columns:
        baseline_acc = float((merged_df["zero_shot"] == merged_df["true_label"]).mean())
    else:
        baseline_acc = np.nan
    include_baseline = not np.isnan(baseline_acc)

    all_genders, all_ethnicities = set(), set()
    profile_demographics = {}
    for p in profile_cols:
        traits = person_set.get_traits(p, group_keys=group_keys)
        gender = str(traits.get("gender", "unknown")).lower()
        ethnicity = str(traits.get("ethnicity", "unknown")).lower()
        all_genders.add(gender)
        all_ethnicities.add(ethnicity)
        profile_demographics[p] = (ethnicity, gender)

    genders = sorted(all_genders)
    ethnicities = sorted(all_ethnicities)
    ordered_combinations = [(e, g) for e in ethnicities for g in genders]

    demo_groups = {combo: [] for combo in ordered_combinations}
    for p, (e, g) in profile_demographics.items():
        if (e, g) in demo_groups:
            demo_groups[(e, g)].append(p)

    group_stats = {}
    for combo in ordered_combinations:
        profs = demo_groups[combo]
        prof_accs = [(merged_df[p] == merged_df["true_label"]).mean() for p in profs if p in merged_df.columns]
        n = len(prof_accs)
        if n == 0:
            mean_acc, disp = np.nan, 0.0
        elif n == 1:
            mean_acc, disp = float(prof_accs[0]), 0.0
        else:
            mean_acc = float(np.mean(prof_accs))
            if error_style == "sd":
                disp = float(np.std(prof_accs, ddof=1))
            elif error_style == "t":
                disp = float(sem(prof_accs) * t.ppf(0.975, n - 1))
            else:
                disp = 0.0
        group_stats[combo] = {"mean_acc": mean_acc, "disp": disp, "n": n}

    cmap = plt.get_cmap(colormap)
    ethnicity_colors = {e: cmap(i / max(1, len(ethnicities) - 1)) for i, e in enumerate(ethnicities)}

    labels, means, disp_err, colors = [], [], [], []
    if include_baseline:
        labels.append("Baseline")
        means.append(baseline_acc)
        disp_err.append(0.0)
        colors.append("0.5")

    abbrev_gender = {"man": "M", "woman": "W", "nonbinary": "NB"}
    for e, g in ordered_combinations:
        st = group_stats[(e, g)]
        labels.append(f"{e.replace('_','-').title()}\n{abbrev_gender.get(g, g[:1].upper())}")
        means.append(st["mean_acc"])
        disp_err.append(st["disp"])
        colors.append(ethnicity_colors.get(e, "0.7"))

    means_arr = np.array(means, dtype=float)
    disp_arr = np.array(disp_err, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.65
    yerr = None if error_style == "none" else disp_arr
    ax.bar(
        x,
        means_arr,
        yerr=yerr,
        width=width,
        capsize=3 if yerr is not None else 0,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        rasterized=True,
    )

    if len(ethnicities) > 1:
        offset = 1 if include_baseline else 0
        for i in range(1, len(ethnicities)):
            pos = offset + i * len(genders) - 0.5
            ax.axvline(pos, color="0.6", linestyle=":", linewidth=1, alpha=0.8)

    ax.axhline(consensus_accuracy_global, color="0.2", linestyle="--", linewidth=1.2, label="Consensus accuracy")

    EPS = 0.002
    for i, m in enumerate(means_arr):
        if not np.isfinite(m):
            continue
        d = m - consensus_accuracy_global
        up = (0 if yerr is None else yerr[i]) + EPS
        ax.text(x[i], m + up, f"{d:+0.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("Accuracy")

    bar_title = "Accuracy by ethnicity–gender pair"
    if error_style == "sd":
        bar_title += " (±1 SD across profiles)"
    elif error_style == "t":
        bar_title += " (±95% t-interval across profiles)"
    ax.set_title(bar_title, fontsize=9)

    ax.grid(axis="y", linestyle=":", alpha=0.35)

    if full_scale_axis:
        ax.set_ylim(0.0, 1.0)
    else:
        finite_means = means_arr[np.isfinite(means_arr)]
        if finite_means.size:
            min_acc = float(np.min(finite_means))
            ymin = max(0.0, 0.9 * min_acc)
            ymax = min(1.0, float(np.nanmax(means_arr + (disp_arr if yerr is not None else 0))) + 0.03)
            if ymin >= ymax:
                ymax = ymin + 0.05
            ax.set_ylim(ymin, ymax)

    legend_handles, legend_labels = [], []
    if include_baseline:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color="0.5", ec="black"))
        legend_labels.append("Baseline")
    for e in ethnicities:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=ethnicity_colors.get(e, "0.7"), ec="black"))
        legend_labels.append(e.replace("_", "-").title())
    ax.legend(legend_handles, legend_labels, title="Groups", loc="upper right", framealpha=0.9)

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)

    rows = []
    if include_baseline:
        rows.append(
            {
                "combination": "baseline",
                "ethnicity": None,
                "gender": None,
                "mean_accuracy": baseline_acc,
                "delta_from_consensus": baseline_acc - consensus_accuracy_global,
                "ci": 0.0,
                "n_profiles": None,
                "error_style": error_style,
            }
        )
    for (e, g), st in group_stats.items():
        rows.append(
            {
                "combination": f"{e}_{g}",
                "ethnicity": e,
                "gender": g,
                "mean_accuracy": st["mean_acc"],
                "delta_from_consensus": (st["mean_acc"] - consensus_accuracy_global)
                if np.isfinite(st["mean_acc"])
                else np.nan,
                "ci": st["disp"],
                "n_profiles": st["n"],
                "error_style": error_style,
            }
        )
    return pd.DataFrame(rows)





def compute_pairwise_demographic_diffs(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    trait="ethnicity",
    min_profiles=2,
    p_adjust="fdr_bh",
    n_boot: int = 2000,
    random_state: int = 0,
):
    groups = _group_profiles_by_trait(merged_df, person_set, trait=trait, min_profiles=min_profiles)
    keys = sorted(groups.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g1, g2 = keys[i], keys[j]
            p1, p2 = groups[g1], groups[g2]
            if len(p1) < 1 or len(p2) < 1:
                continue
            boot = _bootstrap_group_accuracy_diff(
                merged_df, p1, p2, label_col="true_label", n_boot=n_boot, random_state=random_state
            )
            if not boot["ok"]:
                continue
            mean1, mean2 = boot["acc1"], boot["acc2"]
            diff, lo, hi, p = boot["diff"], boot["ci_low"], boot["ci_high"], boot["p_raw"]
            rows.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "mean1": float(mean1),
                    "mean2": float(mean2),
                    "diff": float(diff),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p": float(p),
                    "h": float(_cohens_h(mean1, mean2)),
                    "n1": int(len(p1)),
                    "n2": int(len(p2)),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        _, p_adj, _, _ = multipletests(out["p"].values, method=p_adjust)
        out["p_adj"] = p_adj
        out["neglog10_p"] = -np.log10(out["p"].clip(lower=1e-300))
        out["neglog10_p_adj"] = -np.log10(out["p_adj"].clip(lower=1e-300))
    return out.sort_values("diff", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)


def plot_volcano_demographic_diffs(
    pair_df: pd.DataFrame,
    ax=None,
    title="Volcano — Difference vs −log10(p) (exploratory)",
    rank_by="diff",
    top_n=5,
    alpha_sig=0.05,
    figsize=(10, 6),
    label_style="halo",
    label_fontsize=9,
    show_alpha_line=False,
    guide_label=True,
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
    ax.scatter(x[sig], y[sig], s=36, alpha=0.9, edgecolors="black", linewidths=0.4, rasterized=True, zorder=3)

    rank_col = "d" if rank_by == "d" else "diff"
    top_idx = np.argsort(-np.abs(pair_df[rank_col].values))[: min(top_n, len(pair_df))]
    for idx in top_idx:
        row = pair_df.iloc[idx]
        txt = ax.annotate(
            f"{row['group1']} vs {row['group2']}",
            (row["diff"], row["neglog10_p"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=label_fontsize,
        )
        if label_style == "halo":
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        elif label_style == "box":
            txt.set_bbox(dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.6, alpha=0.95))

    ax.axvline(0, ls="--", lw=1, color="0.4")
    if show_alpha_line:
        ax.axhline(-np.log10(alpha_sig), ls=":", lw=1, color="0.4")
        if guide_label:
            ax.text(0.99, -np.log10(alpha_sig) + 0.03, "exploratory guide", ha="right", va="bottom", fontsize=8, color="0.4")

    ax.set_xlabel("Accuracy difference (group1 - group2)")
    ax.set_ylabel("Statistical signal (−log10 p)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.set_facecolor("white")
    ax.margins(x=0.05)

    ax.text(
        1.02,
        1.02,
        "Exploratory view; Tier-1 reports formal tests",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.3",
    )
    return ax


def plot_effect_size_heatmap(
    pair_df: pd.DataFrame,
    ax=None,
    title="Effect size — trait pairs",
    cmap="coolwarm",
    figsize=(10, 6),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if pair_df is None or len(pair_df) == 0:
        ax.text(0.5, 0.5, "No effect sizes", ha="center", va="center")
        return ax, 0.0

    use_h = "h" in pair_df.columns and pair_df["h"].notna().any()
    val_col = "h" if use_h else "d"

    groups = sorted(set(pair_df["group1"]).union(set(pair_df["group2"])))
    idx = {g: i for i, g in enumerate(groups)}
    M = np.zeros((len(groups), len(groups)))
    for _, r in pair_df.iterrows():
        i, j = idx[r["group1"]], idx[r["group2"]]
        M[i, j] = r[val_col]
        M[j, i] = -r[val_col]
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
    cbar.set_label("Cohen's h (signed)" if use_h else "Cohen's d (signed)")
    ax.set_facecolor("white")
    return ax, vmax


def plot_intersectional_accuracy_heatmap(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    ax=None,
    title="Intersectional — (ethnicity, gender)",
    normalize=True,
    cmap="coolwarm",
    figsize=(10, 6),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    profiles = _profile_cols(merged_df)
    if not profiles:
        ax.text(0.5, 0.5, "No profiles", ha="center", va="center")
        return ax, 0.0

    eth_set, gen_set, by_combo = set(), set(), {}
    for p in profiles:
        tr = _traits_for_profile(person_set, p, keys=("ethnicity", "gender"))
        eth, gen = tr["ethnicity"], tr["gender"]
        eth_set.add(eth)
        gen_set.add(gen)
        by_combo.setdefault((eth, gen), []).append(p)

    eth_list = sorted(eth_set)
    gen_list = sorted(gen_set)
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

    ax.set_xticks(range(len(gen_list)))
    ax.set_xticklabels([g.title() for g in gen_list], fontsize=9)
    ax.set_yticks(range(len(eth_list)))
    ax.set_yticklabels([e.replace("_", "-").title() for e in eth_list], fontsize=9)
    ax.set_title(title, fontsize=10)

    if np.isfinite(A).sum() <= 30:
        for i in range(len(eth_list)):
            for j in range(len(gen_list)):
                if np.isfinite(A[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{A[i, j]:+0.3f}" if normalize else f"{A[i, j]:0.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Difference in accuracy" if normalize else "Accuracy")
    ax.set_facecolor("white")
    return ax, vmax


def plot_demographic_accuracy_composite(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    trait="ethnicity",
    top_n=5,
    normalize_intersectional=True,
    savepath=None,
    figsize=(15, 4.5),
    use_neurips_style=True,
):
    if use_neurips_style:
        try:
            apply_neurips_figure_style()
        except Exception:
            pass

    pair = compute_pairwise_demographic_diffs(merged_df, person_set, trait=trait, min_profiles=2)

    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    plot_volcano_demographic_diffs(pair, ax=axs[0], top_n=top_n)
    _, vmax_d = plot_effect_size_heatmap(pair, ax=axs[1])
    _, vmax_acc = plot_intersectional_accuracy_heatmap(
        merged_df, person_set, ax=axs[2], normalize=normalize_intersectional
    )
    for ax, tag in zip(axs, ["(a)", "(b)", "(c)"]):
        ax.text(0.01, 0.98, tag, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    for a in axs:
        a.set_title(a.get_title().split(") ", 1)[-1] if ") " in a.get_title() else a.get_title())

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)

    return fig, {"pairwise": pair, "vmax_effect": vmax_d, "vmax_intersectional": vmax_acc}


def save_demographic_figures_individual(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    out_dir: str,
    figsize=(10, 6),
    normalize_intersectional=True,
    top_n=5,
):
    os.makedirs(out_dir, exist_ok=True)

    pair = compute_pairwise_demographic_diffs(merged_df, person_set=person_set, trait="ethnicity", min_profiles=2)

    fig_v, ax_v = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_volcano_demographic_diffs(pair, ax=ax_v, top_n=top_n, figsize=figsize, label_style="halo")
    fig_v.savefig(os.path.join(out_dir, "volcano_demographic.pdf"), bbox_inches="tight")
    plt.close(fig_v)

    fig_h, ax_h = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_effect_size_heatmap(pair, ax=ax_h, figsize=figsize)
    fig_h.savefig(os.path.join(out_dir, "effect_size_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig_h)

    fig_i, ax_i = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_intersectional_accuracy_heatmap(merged_df, person_set, ax=ax_i, normalize=normalize_intersectional, figsize=figsize)
    fig_i.savefig(os.path.join(out_dir, "intersectional_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig_i)

    return {
        "volcano": os.path.join(out_dir, "volcano_demographic.pdf"),
        "effect_size": os.path.join(out_dir, "effect_size_heatmap.pdf"),
        "intersectional": os.path.join(out_dir, "intersectional_heatmap.pdf"),
    }



# Token analysis

def print_token_analysis_summary(token_stats: dict, top_n: int = 10):
    perf = token_stats.get("per_profile", pd.DataFrame())
    overall = token_stats.get("overall", {}) or {}

    print("\n\n=== Token Economics ===")
    if overall:
        def fmt(x):
            return "—" if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.3f}" if isinstance(x, float) else str(x)
        print("Overall:")
        print(f"  - Total calls: {overall.get('total_calls', '—')}")
        print(f"  - Total tokens: {fmt(overall.get('total_tokens', np.nan))}")
        print(f"  - Mean tokens/sample: {fmt(overall.get('tokens_per_sample_mean', np.nan))}")
        if "total_cost" in overall and not pd.isna(overall["total_cost"]):
            print(f"  - Total cost: ${fmt(overall.get('total_cost', np.nan))}")
            print(f"  - Mean cost/sample: ${fmt(overall.get('cost_per_sample_mean', np.nan))}")
            if "cost_per_correct" in overall and not pd.isna(overall["cost_per_correct"]):
                print(f"  - Cost per correct: ${fmt(overall.get('cost_per_correct', np.nan))}")
        if "baseline_accuracy" in overall and not pd.isna(overall["baseline_accuracy"]):
            print(f"  - Baseline accuracy: {overall['baseline_accuracy']:.3f}")

    if perf is None or perf.empty:
        print("No per-profile token metrics available.")
        return

    if "efficiency_acc_per_1k_tokens" in perf.columns:
        tmp = perf[["profile", "accuracy", "tokens_per_sample", "efficiency_acc_per_1k_tokens"]].dropna()
        tmp = tmp.sort_values("efficiency_acc_per_1k_tokens", ascending=False).head(top_n)
        print("\nTop profiles by efficiency (accuracy per 1k tokens):")
        for _, r in tmp.iterrows():
            print(
                f"  {r['profile']:<12} eff={r['efficiency_acc_per_1k_tokens']:.2f} | "
                f"acc={r['accuracy']:.3f} | tok/sample={r['tokens_per_sample']:.1f}"
            )

    if "cost_per_correct" in perf.columns and perf["cost_per_correct"].notna().any():
        tmp = perf[["profile", "cost_per_correct", "accuracy", "tokens_per_sample"]].dropna()
        tmp = tmp.sort_values("cost_per_correct", ascending=True).head(top_n)
        print("\nBest (lowest) cost per correct:")
        for _, r in tmp.iterrows():
            print(
                f"  {r['profile']:<12} ${r['cost_per_correct']:.4f} | acc={r['accuracy']:.3f} | tok/sample={r['tokens_per_sample']:.1f}"
            )


def plot_token_analysis_figures(perf: pd.DataFrame, out_dir: str, have_pricing: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    try:
        apply_neurips_figure_style()
    except Exception:
        pass

    if {"tokens_per_sample", "accuracy"}.issubset(perf.columns):
        x = perf["tokens_per_sample"].astype(float)
        y = perf["accuracy"].astype(float) * 100.0
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        fig.set_layout_engine("constrained")
        ax.scatter(x, y, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("Tokens per sample")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Tokens vs Accuracy (per profile)")
        pth = os.path.join(out_dir, "tokens_vs_accuracy.pdf")
        #plt.savefig(pth)
        plt.close()

    if have_pricing and {"cost_per_sample", "accuracy"}.issubset(perf.columns) and perf["cost_per_sample"].notna().any():
        x = perf["cost_per_sample"].astype(float)
        y = perf["accuracy"].astype(float) * 100.0
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        fig.set_layout_engine("constrained")
        ax.scatter(x, y, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.set_xlabel("Cost per sample ($)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Cost vs Accuracy (per profile)")
        pth = os.path.join(out_dir, "cost_vs_accuracy.pdf")
        #plt.savefig(pth)
        plt.close()


# Main function

def run_full_preliminary_analysis(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    df: Optional[pd.DataFrame] = None,
    person_set: PersonSet = None,
    threshold_disagreement=0.3,
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,
    stage: str = "preliminary",
    per_figure_subdirs: Optional[Dict[str, str]] = None,
    sub_case: Optional[str] = None,
) -> Dict[str, Any]:

    if person_set is None:
        raise ValueError("person_set is required for analysis")

    try:
        apply_neurips_figure_style()
    except Exception:
        pass

    results: Dict[str, Any] = {}

    if "base_pred" not in merged_df.columns:
        if "zero_shot" in merged_df.columns:
            merged_df["base_pred"] = merged_df["zero_shot"]
        else:
            raise ValueError("Need either base_pred or zero_shot in merged_df.")

    if df is not None and "sample_id" in merged_df.columns and "sample_id" in df.columns:
        missing_cols = [col for col in case.category_cols if col not in merged_df.columns]
        if missing_cols:
            print(f"Merging missing category columns: {missing_cols}")
            merged_df = merged_df.merge(df[["sample_id"] + missing_cols], on="sample_id", how="left")
        else:
            print("All category columns already present in merged_df")

    group_keys = get_analysis_group_keys(person_set)

    print("\n\n=== DEMOGRAPHIC ACCURACY DIFFERENCES ===")
    demographic_results = test_comprehensive_demographic_accuracy_differences(merged_df, person_set=person_set)
    print_comprehensive_demographic_results(demographic_results)
    results["demographic"] = demographic_results

    print("\n\n=== SYSTEMATIC ERROR PATTERNS ===")
    error_results = guarded_labelspace_analysis(
        analyze_systematic_error_patterns_multi_category, merged_df, case, person_set=person_set
    )
    error_patterns_all: Dict[str, pd.DataFrame] = {}
    meaningful_patterns_all: Dict[str, pd.DataFrame] = {}
    category_summary_all: Dict[str, pd.DataFrame] = {}
    for cat_col, (error_patterns, meaningful_patterns, category_summary) in error_results.items():
        error_patterns_all[cat_col] = error_patterns
        meaningful_patterns_all[cat_col] = meaningful_patterns
        category_summary_all[cat_col] = category_summary
        print(f"Found {len(error_patterns)} error patterns, {len(meaningful_patterns)} meaningful for {cat_col}")
    print_multi_category_error_summary(error_results)
    results["error_patterns"] = error_patterns_all
    results["meaningful_error_patterns"] = meaningful_patterns_all
    results["category_summary"] = category_summary_all

    print("\n\n=== HIGH DISAGREEMENT CASES ===")
    disagreement_df = extract_high_disagreement_cases(merged_df, threshold=threshold_disagreement, person_set=person_set)
    print(f"Found {len(disagreement_df)} high disagreement cases")
    results["disagreement"] = disagreement_df

    print("\n\n=== RESCUE STATISTICS BY CATEGORY ===")
    rescue_stats_all: Dict[str, pd.DataFrame] = {}
    rescue_analysis_all: Dict[str, Dict[str, Any]] = {}
    for cat_col in case.category_cols:
        if cat_col in merged_df.columns:
            print(f"\nAnalyzing rescue statistics for category column: {cat_col}")
            rescue_df = guarded_labelspace_analysis(rescue_stats_by_category, merged_df, case=case, category_col=cat_col)
            rescue_analysis = analyze_rescue_performance(rescue_df)
            rescue_stats_all[cat_col] = rescue_df
            rescue_analysis_all[cat_col] = rescue_analysis
            print(f"Computed rescue stats for {cat_col}: {len(rescue_df)} records")
        else:
            print(f"Warning: Category column '{cat_col}' not found for rescue analysis")
    print_all_rescue_analyses(rescue_analysis_all)
    results["rescue_stats"] = rescue_stats_all
    results["rescue_analysis"] = rescue_analysis_all

    print("\n\n=== Profile Similarity Clustering ===")
    profile_similarity = analyze_profile_similarity(merged_df, person_set=person_set)
    print_profile_similarity_analysis(profile_similarity)
    results["profile_similarity"] = profile_similarity

    print("\n\n=== FIGURES ===")
    subdirs = per_figure_subdirs or {}
    _stage_dir = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage, sub_case=sub_case)  # just to ensure root exists

    acc_dir = resolve_plot_dir(
        case, plots_root=plots_root, strategy=strategy, stage=stage, extra_subdir=subdirs.get("accuracy_by_group") or "accuracy_by_group", sub_case=sub_case
    )
    demo_dir = resolve_plot_dir(
        case, plots_root=plots_root, strategy=strategy, stage=stage, extra_subdir=subdirs.get("demographic") or "demographic", sub_case=sub_case
    )
    tokens_dir = resolve_plot_dir(
        case, plots_root=plots_root, strategy=strategy, stage=stage, extra_subdir=subdirs.get("token_stats") or "token_stats", sub_case=sub_case
    )

    try:
        acc_summary = plot_accuracy_deltas_with_ci(
            merged_df,
            person_set=person_set,
            group_keys=group_keys,
            error_style="t",
            figsize=(10, 6),
            savepath=os.path.join(acc_dir, "accuracy_by_group.pdf"),
        )

        paths = save_demographic_figures_individual(
            merged_df,
            person_set=person_set,
            out_dir=demo_dir,
            figsize=(10, 6),
            normalize_intersectional=True,
            top_n=5,
        )

        results["figure_paths"] = {"accuracy_by_group": os.path.join(acc_dir, "accuracy_by_group.pdf"), **paths}
        results["accuracy_summary"] = acc_summary
        print("Saved:", os.path.join(acc_dir, "accuracy_by_group.pdf"))
    except Exception as e:
        print(f"Error saving individual demographic figures: {e}")

    pricing = None
    token_stats = None
    try:
        token_stats = compute_token_analysis(
            merged_df,
            pricing=pricing,
            rescue_stats_all=rescue_stats_all,
            rae_lambda=2.0,
            baseline_col="base_pred",
        )
    except Exception as e:
        print(f"Error computing token economics: {e}")

    results["token_analysis"] = token_stats

    if isinstance(token_stats, dict):
        perf = token_stats.get("per_profile", pd.DataFrame())
        if perf is not None and not perf.empty:
            print_token_analysis_summary(token_stats)
            tok_csv = os.path.join(tokens_dir, "token_analysis_per_profile.csv")
            perf.to_csv(tok_csv, index=False)
            plot_token_analysis_figures(perf, tokens_dir, have_pricing=(pricing is not None))

    print(f"\n\n=== ANALYSIS SUMMARY ===")
    print(f"Dataset: {case.case_name}")
    print(f"Category columns analyzed: {[col for col in case.category_cols if col in merged_df.columns]}")
    print(f"Missing category columns: {[col for col in case.category_cols if col not in merged_df.columns]}")
    print(f"Total profiles: {len([c for c in merged_df.columns if c.startswith('profile')])}")
    print(f"Group keys used: {group_keys}")

    return results