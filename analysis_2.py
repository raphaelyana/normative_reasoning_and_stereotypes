import os
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Sequence
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scipy.stats import (
    binomtest
)
from statsmodels.stats.multitest import multipletests

from analysis_0 import *
from analysis_tools import get_available_traits, get_analysis_group_keys
from analysis_tools import resolve_plot_dir
from plot_tools import apply_neurips_figure_style
from profiles.schema import PersonSet
from analysis_tools import has_cognitive_style_data
from cases.cases_config import CaseConfig


MIN_CATEGORY_N = 50        
N_BOOT = 2000             
RANDOM_STATE = 42
DEV_TEST_SPLIT = 0.3  

FALLBACK_CFG = {
    "trigger_ratio": 0.50,
    "topk": 3,
    "min_n_secondary": 25,
    "n_bins_max": 5
}

PRE_SPEC_CRITERIA = [
    ("balanced_v1", lambda r: 0.6*float(r["improvement"]) + 0.4*(1.0 - float(r["extra_error_rate"]))),
    ("max_accuracy", lambda r: float(r["accuracy"])),
    ("risk_averse", lambda r: float(r["improvement"]) - 2.0*float(r["extra_error_rate"])),
]

PRE_SPEC_PARETO = [
    ("primary",       5e-4, 2.0),
    ("token_sensitive",1e-3, 2.0),
    ("risk_averse",   5e-4, 3.0),
]

def _rng_from(random_seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return a numpy Generator, preferring an injected rng; else create from seed."""
    return rng if rng is not None else np.random.default_rng(random_seed)

def _suppression_ratio(df: pd.DataFrame, category_col: str, min_n: int) -> Tuple[float, pd.Series]:
    counts = df[category_col].fillna("Unknown").value_counts(dropna=False)
    total = int(len(counts))
    suppressed = int((counts < min_n).sum())
    ratio = suppressed / max(total, 1)
    return ratio, counts

def _coalesce_low_n(s: pd.Series, min_n: int, other_label: Optional[str] = None) -> pd.Series:
    other = other_label or f"Other (<{min_n})"
    counts = s.fillna("Unknown").value_counts(dropna=False)
    keep = set(counts[counts >= min_n].index.astype(str))
    return s.fillna("Unknown").astype(str).apply(lambda v: v if v in keep else other)

def _difficulty_score(merged_df: pd.DataFrame) -> np.ndarray:
    """1 − mean(correct) across all profile columns → higher = harder."""
    prof = [c for c in merged_df.columns if c.startswith("profile")]
    if not prof:
        return np.zeros(len(merged_df), dtype=float)
    y = merged_df["true_label"].astype(str).to_numpy()
    M = np.stack([(merged_df[p].astype(str).to_numpy() == y).astype(float) for p in prof], axis=1)
    return 1.0 - M.mean(axis=1)

def _apply_category_fallbacks(dev_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              merged_df: pd.DataFrame,
                              category_col: str,
                              min_n: int,
                              cfg: Dict[str, Any] = FALLBACK_CFG) -> Tuple[str, int, str]:
    """
    Returns (effective_category_col, effective_min_n, mode)
    mode ∈ {"primary","coalesce","topk_other","difficulty_bins"}
    """
    r_dev, cnt_dev = _suppression_ratio(dev_df, category_col, min_n)
    r_test, cnt_test = _suppression_ratio(test_df, category_col, min_n)
    worst_ratio = max(r_dev, r_test)

    if worst_ratio < cfg["trigger_ratio"]:
        print(f"[FALLBACK] Not triggered for {category_col} (suppression {worst_ratio:.0%} < {int(cfg['trigger_ratio']*100)}%).")
        return category_col, min_n, "primary"

    other_lbl = f"Other (<{min_n})"
    dev_coal  = _coalesce_low_n(dev_df[category_col],  min_n, other_lbl)
    test_coal = _coalesce_low_n(test_df[category_col], min_n, other_lbl)
    if dev_coal.nunique() >= 2 and test_coal.nunique() >= 2:
        new_col = f"{category_col}_coalesced"
        dev_df[new_col]  = dev_coal
        test_df[new_col] = test_coal
        print(f"[FALLBACK] Using {new_col} (rare values→'{other_lbl}').")
        return new_col, min_n, "coalesce"

    k = int(cfg["topk"])
    top_vals = cnt_dev.sort_values(ascending=False).head(k).index.tolist()
    if len(top_vals) >= 1:
        dev_top  = dev_df[category_col].where(dev_df[category_col].isin(top_vals),  other_lbl)
        test_top = test_df[category_col].where(test_df[category_col].isin(top_vals), other_lbl)
        if dev_top.nunique() >= 2 and test_top.nunique() >= 2:
            new_col = f"{category_col}_top{k}_other"
            dev_df[new_col]  = dev_top
            test_df[new_col] = test_top
            print(f"[FALLBACK] Using {new_col} (top {k} + Other).")
            return new_col, int(cfg["min_n_secondary"]), "topk_other"

    dif = _difficulty_score(merged_df) 
    q = max(2, min(int(cfg["n_bins_max"]), max(2, len(test_df) // max(min_n,1))))
    labels = [f"bin{i+1}" for i in range(q)]
    merged_bins = pd.qcut(pd.Series(dif, index=merged_df.index), q=q, labels=labels, duplicates="drop")
    dev_df["difficulty_bin"]  = merged_bins.loc[dev_df.index]
    test_df["difficulty_bin"] = merged_bins.loc[test_df.index]
    print(f"[FALLBACK] Using difficulty_bin with {merged_bins.nunique()} bins (equal-count; ≥~{len(test_df)//max(merged_bins.nunique(),1)} per bin).")
    return "difficulty_bin", int(cfg["min_n_secondary"]), "difficulty_bins"

def evaluate_prespecified_criteria_devselect(
    ens_dev_eval: Dict[str, Any],       
    ens_test_boot: pd.DataFrame,
    criteria = PRE_SPEC_CRITERIA
) -> Dict[str, Any]:
    """
    For each pre-specified criterion, pick on DEV; attach TEST p-values and BH across criteria.
    """
    dev_tbl = _build_ensemble_table_for_pareto(ens_dev_eval)
    picks = []
    for name, scorer in criteria:
        s = dev_tbl.apply(scorer, axis=1)
        idx = _safe_idxmax(s)
        if idx is None:
            continue
        row = dev_tbl.loc[idx].to_dict()
        row["criterion"] = name
        row["ensemble"]  = dev_tbl.loc[idx, "ensemble"]
        picks.append(row)

    picks_df = pd.DataFrame(picks)
    if picks_df.empty:
        return {"picks": pd.DataFrame(), "q_values": pd.DataFrame()}

    boot = ens_test_boot.rename(columns={
        "delta_acc_p_boot":"p_delta",
        "rescue_gt_extra_p":"p_rescue",
        "extra_gt_rescue_p":"p_extra",
    })
    merged = picks_df.merge(
        boot[["ensemble","p_delta","p_rescue","p_extra"]],
        on="ensemble", how="left"
    )

    out = {"picks": merged}
    for col, key in [("p_delta","q_delta"), ("p_rescue","q_rescue"), ("p_extra","q_extra")]:
        if col in merged.columns and merged[col].notna().any():
            q = _apply_fdr(merged[col].fillna(1.0))
            out.setdefault("q_values", {})
            out["q_values"][key] = q.values
            merged[key] = q.values
        else:
            merged[key] = np.nan
    return out


def pareto_prespecified_devselect(
    ens_dev_eval: Dict[str, Any],
    ens_test_eval: Dict[str, Any],
    test_boot_df: pd.DataFrame,
    case: Any,
    lambdas = PRE_SPEC_PARETO
) -> pd.DataFrame:
    """
    For each pre-registered λ, choose the ensemble on DEV (frontier + scalarization),
    then attach TEST metrics and p-values (from test_boot_df).
    """
    dev_tbl = _build_ensemble_table_for_pareto(ens_dev_eval)
    required = ["accuracy", "extra_error_rate", "tokens_per_sample_sum"]
    if dev_tbl.empty or dev_tbl[required].isna().all().any():
        return pd.DataFrame()

    if ("tokens_per_sample_sum" not in dev_tbl.columns) or dev_tbl["tokens_per_sample_sum"].isna().all():
        dev_tbl["tokens_per_sample_sum"] = dev_tbl["n_profiles"].astype(float)

    rows = []
    for lname, lt, le in lambdas:
        dev_valid = dev_tbl.dropna(subset=required).copy()
        if dev_valid.empty:
            continue

        pmask = _pareto_mask(dev_valid, x_col="extra_error_rate", y_col="accuracy",
                     minimize_x=True, maximize_y=True)
        
        front = dev_valid.loc[pmask].copy()
        if front.empty:
            continue
        front["score"] = _score_on_frontier(front, lambda_tok=lt, lambda_extra=le)
        idx = _safe_idxmax(front["score"])
        if idx is None:
            continue
        chosen = front.loc[idx, "ensemble"]

        rec = {
            "lambda_name": lname,
            "ensemble": chosen,
            "lambda_tok": lt,
            "lambda_extra": le,
        }

        test_ens = (ens_test_eval.get("ensemble_results") or {}).get(chosen, {})
        for k in ("accuracy", "improvement", "rescue_rate", "extra_error_rate",
                  "tokens_per_sample_sum", "n_profiles"):
            if k in test_ens:
                rec[k] = test_ens[k]

        if isinstance(test_boot_df, pd.DataFrame) and not test_boot_df.empty:
            boot_row = test_boot_df[test_boot_df["ensemble"] == chosen]
            if not boot_row.empty:
                b = boot_row.iloc[0]
                rec["delta_acc_p_boot"]   = float(b.get("delta_acc_p_boot", np.nan))
                rec["rescue_gt_extra_p"]  = float(b.get("rescue_gt_extra_p", np.nan))
                rec["extra_gt_rescue_p"]  = float(b.get("extra_gt_rescue_p", np.nan))
                rec["mcnemar_p"]          = float(b.get("mcnemar_p", np.nan))

        rows.append(rec)

    out = pd.DataFrame(rows)
    return out



def _slice_counts(merged_df: pd.DataFrame, category_col: str) -> Dict[str, int]:
    vc = merged_df[category_col].fillna("Unknown").value_counts(dropna=False)
    return vc.to_dict()


def _primary_category_values(merged_df: pd.DataFrame, category_col: str, min_n: int = MIN_CATEGORY_N) -> List[str]:
    counts = _slice_counts(merged_df, category_col)
    return [v for v, n in counts.items() if n >= min_n]

def _suppressed_category_values(merged_df: pd.DataFrame, category_col: str, min_n: int = MIN_CATEGORY_N) -> List[str]:
    counts = _slice_counts(merged_df, category_col)
    return [v for v, n in counts.items() if n < min_n]


def paired_bootstrap_report_global(
    base: pd.Series, ens: pd.Series, true: pd.Series,
    n_boot: int = N_BOOT, random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    base = base.astype(str).values
    ens  = ens.astype(str).values
    true = true.astype(str).values
    n = len(true)

    base_correct = (base == true)
    ens_correct  = (ens  == true)

    acc_base = float(base_correct.mean())
    acc_ens  = float(ens_correct.mean())
    delta_acc = acc_ens - acc_base

    rescued = int((~base_correct &  ens_correct).sum())
    extra   = int(( base_correct & ~ens_correct).sum())
    base_errs = int((~base_correct).sum())
    base_ok   = int(( base_correct).sum())

    rescue_rate = (rescued / base_errs) if base_errs > 0 else 0.0
    extra_rate  = (extra   / base_ok)   if base_ok   > 0 else 0.0
    net_benefit = float(rescued - extra)


    b01 = int(( base_correct & ~ens_correct).sum())
    b10 = int((~base_correct &  ens_correct).sum())
    discordant = b01 + b10
    if discordant > 0:
        mcnemar_p = float(binomtest(k=min(b01, b10), n=discordant, p=0.5).pvalue)
        rescue_gt_extra_p = float(binomtest(k=b10, n=discordant, p=0.5, alternative="greater").pvalue)
        extra_gt_rescue_p = float(binomtest(k=b01, n=discordant, p=0.5, alternative="greater").pvalue)
    else:
        mcnemar_p = 1.0
        rescue_gt_extra_p = 1.0
        extra_gt_rescue_p = 1.0

    acc_ens_bs, delta_bs, rescue_bs, extra_bs, net_bs = [], [], [], [], []
    rng = np.random.default_rng(random_state)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bc = (base[idx] == true[idx])
        ec = (ens[idx]  == true[idx])
        acc_ens_bs.append(float(ec.mean()))
        delta_bs.append(float(ec.mean() - bc.mean()))
    
        resc = int((~bc & ec).sum()); extrae = int((bc & ~ec).sum())
        base_errs_bs = int((~bc).sum()); base_ok_bs = int(bc.sum())
        rescue_bs.append(float(resc / base_errs_bs) if base_errs_bs > 0 else 0.0)
        extra_bs.append(float(extrae / base_ok_bs) if base_ok_bs > 0 else 0.0)
        net_bs.append(float(resc - extrae))

    def _ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(lo), float(hi)

    out = dict(
        acc_base=acc_base, acc_ens=acc_ens, delta_acc=delta_acc,
        rescue_rate=float(rescue_rate), extra_error_rate=float(extra_rate),
        net_benefit=net_benefit,
        b01=b01, b10=b10, mcnemar_p=mcnemar_p,
        rescue_gt_extra_p=rescue_gt_extra_p,
        extra_gt_rescue_p=extra_gt_rescue_p,
        acc_ens_ci=_ci(acc_ens_bs),
        delta_acc_ci=_ci(delta_bs),
        rescue_rate_ci=_ci(rescue_bs),
        extra_error_rate_ci=_ci(extra_bs),
        net_benefit_ci=_ci(net_bs),
        n_boot=int(n_boot)
    )

    out["delta_acc_p_boot"] = float(2 * min(np.mean(np.array(delta_bs) >= 0),
                                            np.mean(np.array(delta_bs) <= 0)))
    return out


def paired_bootstrap_report_by_category(
    merged_df: pd.DataFrame,
    ens_preds: pd.Series,
    category_col: str = "stereotype_type",
    min_n: int = MIN_CATEGORY_N,
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    if category_col not in merged_df.columns:
        return pd.DataFrame()

    df = merged_df.copy()
    df[category_col] = df[category_col].fillna("Unknown")
    valid_vals = _primary_category_values(df, category_col, min_n=min_n)
    rows = []
    for v in valid_vals:
        sub = df[df[category_col] == v]
        if len(sub) < min_n:
            continue
        base = sub["base_pred"]
        ens  = ens_preds.loc[sub.index]
        true = sub["true_label"]
        g = paired_bootstrap_report_global(base, ens, true, n_boot=n_boot, random_state=random_state)
        rows.append({
            "value": str(v), "n": int(len(sub)),
            **{k: g[k] for k in g}
        })
    return pd.DataFrame(rows)

def _apply_fdr(series_of_pvals: pd.Series) -> pd.Series:
    if series_of_pvals.empty:
        return series_of_pvals
    _, q, _, _ = multipletests(series_of_pvals.values, method="fdr_bh")
    return pd.Series(q, index=series_of_pvals.index)


def fdr_families(
    global_table: pd.DataFrame,           
    percat_table: pd.DataFrame            
) -> Dict[str, Any]:
    """
    Build FDR (BH) families:
      - Δacc over ensembles (global + per-category)
      - rescue direction (rescues > extra) over ensembles (global + per-category)
      - extra direction (extra > rescues) over ensembles (global + per-category)
    Falls back to legacy columns if directional p's are absent.
    """
    out = {}

    def _apply(df, idx_cols, p_col, out_key):
        if df.empty or p_col not in df.columns:
            out[out_key] = pd.DataFrame()
            return
        p = df.set_index(idx_cols)[p_col]
        q = _apply_fdr(p).reset_index().rename(columns={0: out_key})
        out[out_key] = q


    _apply(global_table, ["ensemble"], "delta_acc_p_boot", "q_global_delta_acc")
    if not percat_table.empty and "delta_acc_p_boot" in percat_table.columns:
        _apply(percat_table, ["ensemble","value"], "delta_acc_p_boot", "q_per_category_delta_acc")
    else:
        out["q_per_category_delta_acc"] = pd.DataFrame()


    rescue_p_col = "rescue_gt_extra_p" if ("rescue_gt_extra_p" in global_table.columns) else \
                   ("rescue_rate_p_boot" if "rescue_rate_p_boot" in global_table.columns else None)
    if rescue_p_col:
        _apply(global_table, ["ensemble"], rescue_p_col, "q_global_rescue")
    else:
        out["q_global_rescue"] = pd.DataFrame()
    rescue_p_col_cat = "rescue_gt_extra_p" if ("rescue_gt_extra_p" in percat_table.columns) else \
                       ("rescue_rate_p_boot" if "rescue_rate_p_boot" in percat_table.columns else None)
    if rescue_p_col_cat:
        _apply(percat_table, ["ensemble","value"], rescue_p_col_cat, "q_per_category_rescue")
    else:
        out["q_per_category_rescue"] = pd.DataFrame()


    extra_p_col = "extra_gt_rescue_p" if ("extra_gt_rescue_p" in global_table.columns) else \
                  ("extra_error_rate_p_boot" if "extra_error_rate_p_boot" in global_table.columns else None)
    if extra_p_col:
        _apply(global_table, ["ensemble"], extra_p_col, "q_global_extra_error")
    else:
        out["q_global_extra_error"] = pd.DataFrame()
    extra_p_col_cat = "extra_gt_rescue_p" if ("extra_gt_rescue_p" in percat_table.columns) else \
                      ("extra_error_rate_p_boot" if "extra_error_rate_p_boot" in percat_table.columns else None)
    if extra_p_col_cat:
        _apply(percat_table, ["ensemble","value"], extra_p_col_cat, "q_per_category_extra_error")
    else:
        out["q_per_category_extra_error"] = pd.DataFrame()

    return out




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



def majority_vote_ensemble(
    df: pd.DataFrame,
    profile_list: list,
    baseline_col: str = "base_pred"
) -> pd.Series:
    """
    Majority vote across profile predictions with deterministic, risk-averse tie-breaking:
      1) If tie, prefer baseline prediction if it is among tied labels.
      2) Else break tie lexicographically to be fully deterministic.
    """
    available_profiles = [p for p in profile_list if p in df.columns]
    if not available_profiles:
        print(f"WARNING: No available profiles from list: {profile_list}")
        return pd.Series(['']*len(df), index=df.index)

    base = df[baseline_col] if baseline_col in df.columns else None
    preds = df[available_profiles]

    def vote(row):
        votes = [v for v in row if pd.notna(v) and v != '']
        if not votes:
            return '' if base is None else base.loc[row.name]
        counts = Counter(votes).most_common()

        if len(counts) == 1 or counts[0][1] > counts[1][1]:
            return counts[0][0]

        max_count = counts[0][1]
        tied = sorted([lbl for lbl, c in counts if c == max_count])
        if base is not None and base.loc[row.name] in tied:
            return base.loc[row.name]
        return tied[0]

    return preds.apply(vote, axis=1)


def _enrich_ensembles_with_metrics_for_split(
    df: pd.DataFrame,
    ens_dict: Dict[str, Any],
    perf_df: Optional[pd.DataFrame] = None,
    baseline_col: str = "base_pred"
) -> Dict[str, Any]:
    """
    For each ensemble in ens_dict["ensemble_results"], compute on this df:
      accuracy, improvement vs baseline, rescue_rate, extra_error_rate, tokens_per_sample_sum.
    Requires keys: "ensemble_preds" (Series aligned to df.index), optional "profiles".
    """
    base = df[baseline_col]
    true = df["true_label"]

    tokens_map = {}
    if perf_df is not None and not perf_df.empty and "profile" in perf_df.columns:
        core = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in core.columns:
            tokens_map = core["tokens_per_sample"].to_dict()

    out = {"ensemble_results": {}}
    for name, d in (ens_dict.get("ensemble_results") or {}).items():
        preds = d.get("ensemble_preds")
        if preds is None or len(preds) != len(df):
            continue
        preds = preds.reindex(df.index)

        base_correct = (base == true)
        ens_correct  = (preds == true)

        acc_base = float(base_correct.mean())
        acc_ens  = float(ens_correct.mean())
        improvement = acc_ens - acc_base

        rescued = int((~base_correct &  ens_correct).sum())
        extra   = int(( base_correct & ~ens_correct).sum())
        base_errs = int((~base_correct).sum())
        base_ok   = int(( base_correct).sum())

        rescue_rate = (rescued / base_errs) if base_errs > 0 else 0.0
        extra_rate  = (extra   / base_ok)   if base_ok   > 0 else 0.0

        profiles = d.get("profiles", [])
        tok_sum = float(np.nansum([tokens_map.get(p, np.nan) for p in profiles])) if profiles and tokens_map else np.nan

        out["ensemble_results"][name] = {
            **d,
            "accuracy": acc_ens,
            "improvement": improvement,
            "rescue_rate": float(rescue_rate),
            "extra_error_rate": float(extra_rate),
            "n_profiles": int(len(profiles) if profiles else 0),
            "tokens_per_sample_sum": tok_sum
        }
    out["baseline_accuracy"] = float((base == true).mean())
    return out



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

    trait_groups = build_trait_groups(merged_df, person_set, group_keys)
    
    true_labels = merged_df['true_label']
    baseline_preds = merged_df['base_pred']
    baseline_accuracy = accuracy_score(true_labels, baseline_preds)
    
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print("\nEnsemble Performance by Trait Group:")
    print("-" * 50)

    ensemble_results = {}
    
    for group_name, profile_list in trait_groups.items():
        if len(profile_list) == 0:
            continue
            
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

        valid_values = _primary_category_values(merged_df, cat_col, min_n=MIN_CATEGORY_N)
        suppressed   = _suppressed_category_values(merged_df, cat_col, min_n=MIN_CATEGORY_N)
        if suppressed:
            print(f"[INFO] Suppressing low-n values in {cat_col}: {sorted(suppressed)} (n<{MIN_CATEGORY_N})")

        for category in valid_values:
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
                'ensembles': {},
                'n': int(len(cat_subset)),
                'suppressed_values': suppressed
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
            'best_balanced': best_balanced,
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
        'baseline_accuracy': baseline_accuracy,
        'inference': 'descriptive'
    }

def cluster_level_error_patterns(
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
    Cluster-level Error and Rescue Pattern Analysis (STRICT: no mock data).
    Fails with a clear error if prerequisites aren't available/computable.
    """
    print("="*80)
    print("Cluster level error and rescue patterns")
    print("="*80)
    print(f"Group keys: {group_keys}")

    if similarity_results is None:
        similarity_results = analyze_profile_similarity(
            merged_df,
            person_set=person_set
        )
    if not isinstance(similarity_results, dict) or "clusters" not in similarity_results:
        raise ValueError("analyze_profile_similarity did not return a dict with 'clusters'.")
    if not similarity_results["clusters"]:
        raise ValueError("No clusters found in similarity_results.")

    category_cols = getattr(case, "category_cols", None) or ["stereotype_type"]
    if not any(col in merged_df.columns for col in category_cols):
        raise ValueError("None of the required category columns are present in merged_df.")

    from analysis_tools import guarded_labelspace_analysis
    rescue_stats_list, error_patterns_list = [], []
    for cat_col in category_cols:
        if cat_col not in merged_df.columns:
            continue
        rs = guarded_labelspace_analysis(
            rescue_stats_by_category,
            merged_df,
            case=case,
            person_set=person_set,
            category_col=cat_col,
        )
        rs["category_col"] = cat_col
        rescue_stats_list.append(rs)

        ep = guarded_labelspace_analysis(
            compute_error_direction_shifts,
            merged_df,
            case=case,
            person_set=person_set,
            category_col=cat_col
        )
        ep["category_col"] = cat_col
        error_patterns_list.append(ep)

    if not rescue_stats_list:
        raise ValueError("Could not compute rescue_stats_by_category for any category column.")
    rescue_stats = pd.concat(rescue_stats_list, ignore_index=True)

    error_patterns = (
        pd.concat(error_patterns_list, ignore_index=True)
        if error_patterns_list else pd.DataFrame()
    )

    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        core = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in core.columns:
            tokens_map = core["tokens_per_sample"].to_dict()
        if "cost_per_sample" in core.columns:
            cost_map = core["cost_per_sample"].to_dict()

    baseline_accuracy = accuracy_score(merged_df["true_label"], merged_df["base_pred"])
    cluster_analysis = {}


    for cluster_id, cluster_info in similarity_results["clusters"].items():
        cluster_profiles = list(cluster_info.get("profiles", []))
        available_profiles = [p for p in cluster_profiles if p in merged_df.columns]

        if not available_profiles:
            raise ValueError(f"{cluster_id} has no available profiles present in merged_df.")

        print(f"\n{cluster_id.upper()} ({len(available_profiles)} profiles):")

        trait_composition = defaultdict(list)
        for profile in available_profiles:
            traits = person_set.get_traits(profile, group_keys)
            for trait_name, trait_value in traits.items():
                if trait_value != "Unknown":
                    trait_composition[trait_name].append(str(trait_value).lower())

        for trait_name, values in trait_composition.items():
            print(f"  {trait_name}: {dict(Counter(values))}")

        accuracies = [(merged_df[p] == merged_df['true_label']).mean() for p in available_profiles]
        acc_mean = float(np.mean(accuracies))
        acc_std  = float(np.std(accuracies))


        internal_agreement = cluster_info.get("internal_agreement", None)
        if internal_agreement is None and len(available_profiles) > 1:
            pairs = list(combinations(available_profiles, 2))
            internal_agreement = float(np.mean([np.mean(merged_df[a] == merged_df[b]) for a, b in pairs]))
        elif internal_agreement is None:
            internal_agreement = None 


        cluster_rescue = rescue_stats[rescue_stats['profile'].isin(available_profiles)]
        if not cluster_rescue.empty:
            rescue_rate_mean = float(cluster_rescue['rescue_rate'].mean())
            rescue_rate_std  = float(cluster_rescue['rescue_rate'].std())
            extra_err_mean   = float(cluster_rescue['extra_err_rate'].mean())
            extra_err_std    = float(cluster_rescue['extra_err_rate'].std())
            total_rescued    = int(cluster_rescue['rescued'].sum())
            total_extra      = int(cluster_rescue['extra_errors'].sum())
        else:
            rescue_rate_mean = rescue_rate_std = None
            extra_err_mean = extra_err_std = None
            total_rescued = total_extra = None

        cluster_error = error_patterns[error_patterns['profile'].isin(available_profiles)] if not error_patterns.empty else pd.DataFrame()
        err_mag_col = None
        for cand in ("error_magnitude", "total_delta_err_rate", "misclassification_rate"):
            if not cluster_error.empty and cand in cluster_error.columns:
                err_mag_col = cand
                break
        if not cluster_error.empty and err_mag_col is not None:
            err_mean = float(cluster_error[err_mag_col].mean())
            err_std  = float(cluster_error[err_mag_col].std())
            ml_mean  = float(cluster_error["misclassification_rate"].mean()) if "misclassification_rate" in cluster_error.columns else None
            dom_dir  = cluster_error["error_direction"].mode().iloc[0] if "error_direction" in cluster_error.columns and not cluster_error["error_direction"].empty else None
        else:
            err_mean = err_std = ml_mean = dom_dir = None

   
        condition_acc = None
        if archetype_parameters.get("accuracy_consensus", True):
            profile_cols = [c for c in merged_df.columns if c.startswith("profile")]
            per_profile_acc = [float((merged_df[p] == merged_df["true_label"]).mean()) for p in profile_cols]
            if len(per_profile_acc) >= 3:
                global_acc_mean = float(np.mean(per_profile_acc))
                global_acc_std  = float(np.std(per_profile_acc))
                global_acc_p80  = float(np.percentile(per_profile_acc, 80.0))
                acc_gap = float(acc_mean - global_acc_mean)
                above_mean_flags = [float((merged_df[p] == merged_df["true_label"]).mean()) > global_acc_mean for p in available_profiles]
                prop_above = float(np.mean(above_mean_flags)) if above_mean_flags else None
                condition_acc = global_acc_p80
            else:
                global_acc_mean = global_acc_std = global_acc_p80 = acc_gap = prop_above = None
        else:
            global_acc_mean = global_acc_std = global_acc_p80 = prop_above = None
            acc_gap = float(acc_mean - float(archetype_parameters["accuracy"]))
            condition_acc = float(archetype_parameters["accuracy"])

        tok_sum = tok_mean = cost_sum = cost_mean = None
        if tokens_map:
            tok_vals = [tokens_map.get(p, np.nan) for p in available_profiles]
            if np.isfinite(tok_vals).any():
                tok_sum = float(np.nansum(tok_vals))
                tok_mean = float(np.nanmean(tok_vals))
        if cost_map:
            cost_vals = [cost_map.get(p, np.nan) for p in available_profiles]
            if np.isfinite(cost_vals).any():
                cost_sum = float(np.nansum(cost_vals))
                cost_mean = float(np.nanmean(cost_vals))

        improvement_per_1k = None
        if tok_sum and tok_sum > 0:
            improvement_per_1k = (acc_mean - baseline_accuracy) / (tok_sum / 1000.0)

        archetype = "Similar to Neutral"
        if rescue_rate_mean is not None and extra_err_mean is not None:
            if (rescue_rate_mean > archetype_parameters["risk_benefit"][0]
                and extra_err_mean < archetype_parameters["risk_benefit"][1]):
                archetype = "Optimal Risk-Benefit"
            elif extra_err_mean < archetype_parameters["extra_error"]:
                archetype = "Cautious"
            elif rescue_rate_mean > archetype_parameters["rescue"]:
                archetype = "High Error-Correction"
        if condition_acc is not None and acc_mean > condition_acc:
            archetype = "High Performer"
        if internal_agreement is not None and internal_agreement > archetype_parameters["internal_agreement"]:
            archetype = "High Consistency" if archetype == "Similar to Neutral" else archetype

        cluster_metrics = {
            'profiles': available_profiles,
            'size': len(available_profiles),
            'internal_agreement': internal_agreement,
            'trait_composition': dict(trait_composition),
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'rescue_rate_mean': rescue_rate_mean,
            'rescue_rate_std': rescue_rate_std,
            'extra_error_rate_mean': extra_err_mean,
            'extra_error_rate_std': extra_err_std,
            'total_rescued': total_rescued,
            'total_extra_errors': total_extra,
            'error_magnitude_mean': err_mean,
            'error_magnitude_std': err_std,
            'misclassification_rate_mean': ml_mean,
            'dominant_error_direction': dom_dir,
            'global_accuracy_mean': global_acc_mean,
            'global_accuracy_std':  global_acc_std,
            'global_accuracy_p80':  global_acc_p80,
            'accuracy_gap': acc_gap if 'acc_gap' in locals() else None,
            'tokens_per_sample_sum': tok_sum,
            'tokens_per_sample_mean': tok_mean,
            'cost_per_sample_sum': cost_sum,
            'cost_per_sample_mean': cost_mean,
            'improvement_per_1k_tokens': improvement_per_1k,
            'archetype': archetype
        }
        print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  Rescue Rate (mean): {rescue_rate_mean if rescue_rate_mean is not None else 'N/A'}")
        print(f"  Extra Error Rate (mean): {extra_err_mean if extra_err_mean is not None else 'N/A'}")
        print(f"  Internal Agreement: {internal_agreement if internal_agreement is not None else 'N/A'}")
        print(f"  Archetype: {archetype}")

        cluster_analysis[cluster_id] = cluster_metrics

    print(f"\n{'='*60}")
    print("Cluster Ensemble Performance")
    print(f"{'='*60}")
    for cluster_id, info in cluster_analysis.items():
        preds = majority_vote_ensemble(merged_df, info['profiles'])
        if preds.empty or preds.eq('').all():
            cluster_analysis[cluster_id]['ensemble_accuracy'] = None
            cluster_analysis[cluster_id]['ensemble_improvement'] = None
            print(f"{cluster_id}: Ensemble not computable (no valid predictions).")
            continue
        ens_acc = accuracy_score(merged_df['true_label'], preds)
        cluster_analysis[cluster_id]['ensemble_accuracy'] = float(ens_acc)
        cluster_analysis[cluster_id]['ensemble_improvement'] = float(ens_acc - baseline_accuracy)
        print(f"{cluster_id}: {ens_acc:.4f} ({ens_acc - baseline_accuracy:+.4f}) | {info['archetype']}")


    if not cluster_analysis:
        raise ValueError("Cluster analysis produced no results.")

    best_accuracy = max(cluster_analysis.items(), key=lambda x: x[1]['accuracy_mean'])
    safest_cluster = min(
        cluster_analysis.items(),
        key=lambda x: float('inf') if x[1]['extra_error_rate_mean'] is None else x[1]['extra_error_rate_mean']
    )
    best_rescue = max(
        cluster_analysis.items(),
        key=lambda x: -float('inf') if x[1]['rescue_rate_mean'] is None else x[1]['rescue_rate_mean']
    )

    recommendations = {
        'best_accuracy': best_accuracy,
        'safest': safest_cluster,
        'best_rescue': best_rescue
    }

    return {
        'cluster_analysis': cluster_analysis,
        'recommendations': recommendations,
        'similarity_results': similarity_results,
        'group_keys': group_keys,
        'baseline_accuracy': baseline_accuracy,
        'inference': 'descriptive'
    }




def _build_ensemble_table_for_pareto(ensemble_results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    base = ensemble_results.get('baseline_accuracy', None)
    ens = ensemble_results.get('ensemble_results', {})
    for name, d in ens.items():
        rows.append({
            "ensemble": name,
            "accuracy": float(d.get("accuracy", np.nan)),
            "improvement": float(d.get("improvement", np.nan)),
            "rescue_rate": float(d.get("rescue_rate", np.nan)),
            "extra_error_rate": float(d.get("extra_error_rate", np.nan)),
            "n_profiles": int(d.get("n_profiles", 0) or 0),
            "tokens_per_sample_sum": (
                float(d.get("tokens_per_sample_sum"))
                if isinstance(d.get("tokens_per_sample_sum"), (int,float)) and np.isfinite(d.get("tokens_per_sample_sum"))
                else np.nan
            ),
            "cost_per_sample_sum": (
                float(d.get("cost_per_sample_sum"))
                if isinstance(d.get("cost_per_sample_sum"), (int,float)) and np.isfinite(d.get("cost_per_sample_sum"))
                else np.nan
            ),
        })
    df = pd.DataFrame(rows)
    df["baseline_accuracy"] = base
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)




def _pareto_mask(df: pd.DataFrame, x_col: str, y_col: str, minimize_x=True, maximize_y=True) -> np.ndarray:
    X = df[[x_col, y_col]].astype(float).to_numpy()

    if not minimize_x: X[:, 0] = -X[:, 0]
    if not maximize_y: X[:, 1] = -X[:, 1]

    order = np.lexsort((-X[:, 1], X[:, 0]))
    best_y = -np.inf
    keep = np.zeros(len(X), dtype=bool)

    for idx in order:
        y = X[idx, 1]
        if y >= best_y - 1e-12:
            keep[idx] = True
            if y > best_y:
                best_y = y
    return keep


def _score_on_frontier(df_frontier: pd.DataFrame, lambda_tok: float = 5e-4, lambda_extra: float = 2.0) -> pd.Series:
    """
    Scalarization for a recommendation on-frontier:
        score = accuracy − λ_extra·extra_err − λ_tok·tokens
    All terms already in [0,1] except tokens; λ_tok expects 'per sample' tokens scale (e.g., 100–10k).
    """
    t = df_frontier["tokens_per_sample_sum"].astype(float).values
    e = df_frontier["extra_error_rate"].astype(float).values
    a = df_frontier["accuracy"].astype(float).values
    return pd.Series(a - lambda_extra*e - lambda_tok*t, index=df_frontier.index)

def _category_safety_table_from_preds(
    ensemble_results: Dict[str, Any],
    merged_df: pd.DataFrame,
    category_col: str = "stereotype_type",
    min_n: int = MIN_CATEGORY_N
) -> pd.DataFrame:
    """
    For each ensemble and each category value, compute:
      - accuracy, improvement vs baseline
      - rescue_rate, extra_error_rate
    Applies low-n suppression consistent with the rest of Tier 2.
    """
    if category_col not in merged_df.columns:
        return pd.DataFrame()

    base = merged_df["base_pred"]
    true = merged_df["true_label"]

    counts = merged_df[category_col].fillna("Unknown").value_counts(dropna=False)
    valid_vals = [v for v, n in counts.items() if int(n) >= int(min_n)]

    rows = []
    for ens_name, d in (ensemble_results.get("ensemble_results") or {}).items():
        preds = d.get("ensemble_preds", None)
        if preds is None or len(preds) != len(merged_df):
            continue
        preds = preds.reindex(merged_df.index)

        base_correct = (base == true)
        ens_correct  = (preds == true)

        for cv in valid_vals:
            idx = (merged_df[category_col].fillna("Unknown") == cv)
            if idx.sum() < min_n:
                continue

            acc_base = float((base_correct[idx]).mean())
            acc_ens  = float((ens_correct[idx]).mean())
            improvement = acc_ens - acc_base

            rescued = ((~base_correct[idx]) & (ens_correct[idx])).sum()
            extra   = ((base_correct[idx]) & (~ens_correct[idx])).sum()

            base_errs      = (~base_correct[idx]).sum()
            base_correct_n = (base_correct[idx]).sum()

            rescue_rate = (rescued / base_errs) if base_errs > 0 else 0.0
            extra_rate  = (extra   / base_correct_n) if base_correct_n > 0 else 0.0

            rows.append({
                "ensemble": ens_name,
                "category": category_col,
                "value": str(cv),
                "n": int(idx.sum()),
                "acc_baseline": acc_base,
                "acc_ensemble": acc_ens,
                "improvement": improvement,
                "rescue_rate": rescue_rate,
                "extra_error_rate": extra_rate
            })
    return pd.DataFrame(rows).sort_values(
        ["category","value","improvement"], ascending=[True, True, False]
    )


def plot_pareto_ensembles(
    df: pd.DataFrame,
    out_dir: Optional[str],
    color_by: str = "improvement",
    title_suffix: str = ""
) -> Dict[str,str]:
    """
    2D: Acc vs Risk (extra_error_rate). 3D inset keeps tokens axis.
    """
    paths = {}

    x, y, c = "extra_error_rate", "accuracy", color_by
    needed = [x, y] + ([c] if c is not None else [])
    dfp = df.dropna(subset=[col for col in needed if col in df.columns]).copy()
    if dfp.empty:
        print("[Pareto] Nothing to plot (missing accuracy/extra_error_rate).")
        return paths

    try:
        apply_neurips_figure_style()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.set_layout_engine("constrained")

    kw_scatter = dict(s=80, alpha=0.9, edgecolor="black", linewidth=0.5)
    if c is not None and c in dfp.columns:
        sc = ax.scatter(dfp[x], dfp[y], c=dfp[c], **kw_scatter)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(c.replace("_", " "))
    else:
        ax.scatter(dfp[x], dfp[y], **kw_scatter)

    ax.set_xlabel("Extra error rate (risk ↓)")
    ax.set_ylabel("Accuracy (↑)")
    ax.set_title(f"Ensemble Pareto (Accuracy ↑, Risk ↓){' — ' + title_suffix if title_suffix else ''}")

    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    pmask = _pareto_mask(dfp, x_col=x, y_col=y, minimize_x=True, maximize_y=True)
    fr = dfp.loc[pmask].sort_values([x, y], ascending=[True, False])
    ax.scatter(fr[x], fr[y], c="none", s=140, marker="*", zorder=6,
               edgecolors="darkred", linewidth=1.2, label="Pareto")
    if len(fr) > 1:
        ax.plot(fr[x], fr[y], "r--", lw=1.6, alpha=0.8)
    for _, r in fr.iterrows():
        ax.annotate(
            r["ensemble"], (r[x], r[y]),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.6, alpha=0.9)
        )
    ax.legend(framealpha=0.92)

    if out_dir:
        p2d = os.path.join(out_dir, "pareto_2d_acc_risk.pdf")
        fig.savefig(p2d, dpi=200)
        plt.close(fig)
        paths["pareto_2d"] = p2d
    else:
        plt.show()

    try:
        fig3 = plt.figure(figsize=(8, 6))
        fig3.set_layout_engine("constrained")
        ax3 = fig3.add_subplot(111, projection="3d")
        X = dfp.get("tokens_per_sample_sum", pd.Series(np.nan, index=dfp.index)).astype(float)
        Y = dfp[y].astype(float)
        Z = dfp[x].astype(float)
        ax3.scatter(X, Y, Z, s=60, alpha=0.85, edgecolor="black", linewidth=0.3)
        ax3.set_xlabel("Tokens ↓")
        ax3.set_ylabel("Accuracy ↑")
        ax3.set_zlabel("Extra-error ↓")
        ax3.set_title(f"3D view (Tokens, Accuracy, Extra-error){' — ' + title_suffix if title_suffix else ''}")
        if out_dir:
            p3d = os.path.join(out_dir, "pareto_3d_inset.pdf")
            fig3.savefig(p3d, dpi=200)
            plt.close(fig3)
            paths["pareto_3d"] = p3d
        else:
            plt.show()
    except Exception:
        pass

    return paths



def pareto_sensitivity_sweep(
    ensemble_results: Dict[str, Any],
    merged_df: pd.DataFrame,
    case: Any,
    lambda_tok_grid = (1e-4, 5e-4, 1e-3, 2e-3),
    lambda_extra_grid = (0.5, 1.0, 2.0, 3.0),
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    recs = []
    for lt in lambda_tok_grid:
        for le in lambda_extra_grid:
            pe = pareto_frontier_for_ensembles(
                ensemble_results=ensemble_results,
                merged_df=merged_df,
                case=case,
                lambda_tok=lt,
                lambda_extra=le,
                out_dir=out_dir
            )
            rec = pe.get("recommended") or {}
            row = {"lambda_tok": lt, "lambda_extra": le, "ensemble": rec.get("ensemble")}

            if rec:
                row.update(rec)
            recs.append(row)
    df = pd.DataFrame(recs)
    stability = (
        df.groupby("ensemble", dropna=False)
          .size()
          .sort_values(ascending=False)
          .rename("count")
          .reset_index()
    )
    return {"grid": df, "stability": stability}

def _paired_bootstrap_indices(n: int, n_boot: int = N_BOOT, random_state: int = RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    for _ in range(n_boot):
        yield rng.integers(0, n, size=n)

def _coerce_scalars(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all simple numeric fields are plain Python floats/ints (not 0-d/1-d arrays).
    Leaves tuples (CIs) as-is.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.item() if v.shape == () else v
        else:
            out[k] = v
    return out

def hierarchical_bootstrap_trait_delta(
    merged_df: pd.DataFrame,
    groupA_profiles: Sequence[str],
    groupB_profiles: Sequence[str],
    true_col: str = "true_label",
    n_boot: int = 5000,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Item-level paired bootstrap. Within each bootstrap draw:
      1) resample items (rows) with replacement,
      2) for each trait group, compute per-item mean correctness across its profiles,
      3) average over items, then take Δ = acc(A) - acc(B).
    Returns point estimate, CI, and two-sided sign bootstrap p-value.
    """
    idx = merged_df.index.to_numpy()
    y_true = merged_df[true_col].astype(str).to_numpy()

    def _group_item_means(profiles):

        if not profiles:
            return np.zeros(len(idx), dtype=float)
        mat = np.stack([(merged_df[p].astype(str).to_numpy() == y_true).astype(float) for p in profiles], axis=1)
        return mat.mean(axis=1)

    a_item_means = _group_item_means(groupA_profiles)
    b_item_means = _group_item_means(groupB_profiles)


    point = float(a_item_means.mean() - b_item_means.mean())

    boot = []
    n = len(idx)
    for bi in _paired_bootstrap_indices(n, n_boot=n_boot, random_state=random_state):
        boot.append(float(a_item_means[bi].mean() - b_item_means[bi].mean()))
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = float(2 * min(np.mean(boot >= 0), np.mean(boot <= 0))) 

    return {
        "delta_acc": point,
        "delta_ci": (float(lo), float(hi)),
        "p_boot": p_two,
        "n_boot": int(n_boot),
        "n_items": int(n),
        "nA": int(len(groupA_profiles)),
        "nB": int(len(groupB_profiles)),
    }



def pareto_frontier_for_ensembles(
    ensemble_results: Dict[str, Any],
    merged_df: pd.DataFrame,
    case: Any,
    lambda_tok: float = 5e-4,
    lambda_extra: float = 2.0,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True) if out_dir else None

    df = _build_ensemble_table_for_pareto(ensemble_results)
    if df.empty:
        return {"error": "No ensembles to evaluate."}


    if ("tokens_per_sample_sum" not in df.columns) or df["tokens_per_sample_sum"].isna().all():
        df["tokens_per_sample_sum"] = df["n_profiles"].astype(float)
        print("[Pareto] tokens_per_sample_sum unavailable; using n_profiles as a proxy for token cost.")

    required = ["accuracy", "extra_error_rate", "tokens_per_sample_sum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing columns in frontier table: {missing}"
        print(f"[Pareto] {msg}")
        return {
            "table": df, "frontier": pd.DataFrame(), "recommended": None,
            "reason": msg, "fig_paths": {},
            "lambdas": {"lambda_tok": lambda_tok, "lambda_extra": lambda_extra}
        }

    valid = df.dropna(subset=required).copy()
    if valid.empty:
        msg = "All candidate rows have NaNs in required fields (accuracy/extra_error_rate/tokens)."
        print(f"[Pareto] {msg}")
        return {
            "table": df, "frontier": pd.DataFrame(), "recommended": None,
            "reason": msg, "fig_paths": {},
            "lambdas": {"lambda_tok": lambda_tok, "lambda_extra": lambda_extra}
        }

    pmask = _pareto_mask(valid, x_col="extra_error_rate", y_col="accuracy",
                     minimize_x=True, maximize_y=True)

    fig_paths = plot_pareto_ensembles(
        valid,
        out_dir=out_dir,
        color_by="improvement",                             
        title_suffix=getattr(case, "case_name", "")
    )

    valid["pareto"] = pmask
    frontier = valid.loc[pmask].copy()

    if frontier.empty:
        msg = "No Pareto-optimal points found."
        print(f"[Pareto] {msg}")
        fig_paths = plot_pareto_ensembles(valid, out_dir=out_dir, color_by="extra_error_rate",
                                          title_suffix=getattr(case, "case_name", ""))
        return {
            "table": valid, "frontier": pd.DataFrame(), "recommended": None,
            "reason": msg, "fig_paths": fig_paths,
            "lambdas": {"lambda_tok": lambda_tok, "lambda_extra": lambda_extra}
        }

    

    frontier["score"] = _score_on_frontier(frontier, lambda_tok=lambda_tok, lambda_extra=lambda_extra)
    best_idx = _safe_idxmax(frontier["score"])
    if best_idx is None:
        msg = "Score vector is all-NaN or empty; cannot select recommendation."
        print(f"[Pareto] {msg}")
        fig_paths = plot_pareto_ensembles(valid, out_dir=out_dir, color_by="extra_error_rate",
                                          title_suffix=getattr(case, "case_name", ""))
        return {
            "table": valid, "frontier": frontier, "recommended": None,
            "reason": msg, "fig_paths": fig_paths,
            "lambdas": {"lambda_tok": lambda_tok, "lambda_extra": lambda_extra}
        }

    rec = frontier.loc[best_idx].to_dict()
    rec["utility"] = float(frontier.loc[best_idx, "score"])

    fig_paths = plot_pareto_ensembles(valid, out_dir=out_dir, color_by="extra_error_rate",
                                      title_suffix=getattr(case, "case_name", ""))

    frontier_small = frontier.sort_values(
        ["accuracy", "extra_error_rate"], ascending=[False, True]
        )[["ensemble", "accuracy", "extra_error_rate", "rescue_rate", "tokens_per_sample_sum", "score"]].reset_index(drop=True)

    return {
        "table": valid,
        "frontier": frontier_small,
        "recommended": rec,
        "category_safety": _category_safety_table_from_preds(
            ensemble_results,
            merged_df,
            category_col=(getattr(case, "category_cols", ["stereotype_type"]) or ["stereotype_type"])[0],
            min_n=globals().get("MIN_CATEGORY_N", 50)
        ),
        "fig_paths": fig_paths,
        "lambdas": {"lambda_tok": lambda_tok, "lambda_extra": lambda_extra}
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


    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        core = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in core.columns:
            tokens_map = core["tokens_per_sample"].to_dict()
        if "cost_per_sample" in core.columns:
            cost_map = core["cost_per_sample"].to_dict()

    try:
        category_cols = getattr(case, "category_cols", None) or ["stereotype_type"]
        rescue_stats_list, error_patterns_list = [], []
        from analysis_tools import guarded_labelspace_analysis
        for cat_col in category_cols:
            if cat_col in merged_df.columns:
                rs = guarded_labelspace_analysis(
                    rescue_stats_by_category,
                    merged_df,
                    case=case,
                    person_set=person_set,
                    category_col=cat_col,
                )
                rs["category_col"] = cat_col
                rescue_stats_list.append(rs)
                bp = guarded_labelspace_analysis(
                    compute_error_direction_shifts,
                    merged_df,
                    case=case,
                    person_set=person_set,
                    category_col=cat_col,
                )
                bp["category_col"] = cat_col
                error_patterns_list.append(bp)
        rescue_stats = pd.concat(rescue_stats_list, ignore_index=True) if rescue_stats_list else pd.DataFrame()
        error_patterns = pd.concat(error_patterns_list, ignore_index=True) if error_patterns_list else pd.DataFrame()
        if rescue_stats.empty:
            raise ValueError("No valid category columns found for rescue_stats.")
        if error_patterns.empty:
            raise ValueError("No valid category columns found for error_patterns.")
    except NameError:
        print("WARNING: rescue_stats_by_category or compute_error_direction_shifts not found")
        rescue_stats = pd.DataFrame()
        error_patterns = pd.DataFrame()
    

    trait_performance = {}
    
    for trait_value, profiles in trait_groups.items():
        available_profiles = [p for p in profiles if p in merged_df.columns]
        if not available_profiles:
            continue
            
        accuracies, rescue_rates, extra_error_rates, error_magnitudes = [], [], [], []
        
        for profile in available_profiles:
            acc = accuracy_score(merged_df['true_label'], merged_df[profile])
            accuracies.append(acc)

            rs = rescue_stats[rescue_stats['profile'] == profile] if not rescue_stats.empty else pd.DataFrame()
            rescue_rates.append(rs['rescue_rate'].mean() if len(rs) > 0 else 0.0)
            extra_error_rates.append(rs['extra_err_rate'].mean() if len(rs) > 0 else 0.0)
            
            bp = error_patterns[error_patterns['profile'] == profile] if not error_patterns.empty else pd.DataFrame()
            error_magnitudes.append(bp['error_magnitude'].mean() if len(bp) > 0 else 0.0)
        
        trait_performance[trait_value] = {
            'accuracies': accuracies,
            'rescue_rates': rescue_rates,
            'extra_error_rates': extra_error_rates,
            'error_magnitudes': error_magnitudes,
            'n_profiles': len(available_profiles),
            'profiles': available_profiles,
            'control_trait_distribution': {}
        }

        if tokens_map:
            tok_vals = [tokens_map.get(p, np.nan) for p in available_profiles]
            trait_performance[trait_value]["tokens_per_sample_mean"] = float(np.nanmean(tok_vals)) if tok_vals else np.nan
        if cost_map:
            cost_vals = [cost_map.get(p, np.nan) for p in available_profiles]
            trait_performance[trait_value]["cost_per_sample_mean"] = float(np.nanmean(cost_vals)) if cost_vals else np.nan
        
        if control_traits:
            for control_trait in control_traits:
                control_values = []
                for profile in available_profiles:
                    if profile in trait_metadata:
                        control_values.append(trait_metadata[profile].get(control_trait, "Unknown"))
                control_counts = Counter(control_values)
                trait_performance[trait_value]['control_trait_distribution'][control_trait] = dict(control_counts)
        
        print(f"\n{str(trait_value).upper()} {comparison_trait.upper()} (n={len(available_profiles)}):")
        print(f"  Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Rescue Rate: {np.mean(rescue_rates):.3f} ± {np.std(rescue_rates):.3f}")
        print(f"  Extra Error Rate: {np.mean(extra_error_rates):.3f} ± {np.std(extra_error_rates):.3f}")
        print(f"  Error Magnitude: {np.mean(error_magnitudes):.3f} ± {np.std(error_magnitudes):.3f}")
        if control_traits:
            print("  Control trait distributions:")
            for control_trait, distribution in trait_performance[trait_value]['control_trait_distribution'].items():
                print(f"    {control_trait}: {distribution}")
    

    statistical_results = {}
    metrics = ['accuracies', 'rescue_rates', 'extra_error_rates', 'error_magnitudes']
    metric_names = ['Accuracy', 'Rescue Rate', 'Extra Error Rate', 'Error Magnitude']
    
    print(f"\n{'='*60}")
    print(f"Statistical Comparisons (Item-level Bootstrap): {comparison_trait.upper()}")
    print(f"{'='*60}")


    groups_by_trait = {tv: data["profiles"] for tv, data in trait_performance.items() if data.get("profiles")}
    trait_values_sorted = sorted(groups_by_trait.keys())

    statistical_results = {"pairwise": {}}
    for i in range(len(trait_values_sorted)):
        for j in range(i+1, len(trait_values_sorted)):
            a, b = trait_values_sorted[i], trait_values_sorted[j]
            res = hierarchical_bootstrap_trait_delta(
                merged_df,
                groups_by_trait[a],
                groups_by_trait[b],
                true_col="true_label",
                n_boot=2000,
                random_state=RANDOM_STATE
            )
            statistical_results["pairwise"][f"{a}_vs_{b}"] = res
            sig = "***" if res["p_boot"] < 0.001 else "**" if res["p_boot"] < 0.01 else "*" if res["p_boot"] < 0.05 else ""
            print(f"  {a} vs {b}: Δacc={res['delta_acc']:+.4f} "
                  f"CI[{res['delta_ci'][0]:+.4f},{res['delta_ci'][1]:+.4f}] "
                  f"p={res['p_boot']:.4f} {sig}")


    print(f"\n{'='*60}")
    print(f"{comparison_trait.upper()} Rankings")
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
    
    weights = {'Accuracy': 0.3, 'Rescue Rate': 0.25, 'Extra Error Rate': -0.25, 'Error Magnitude': -0.2}
    composite_scores = {}
    for trait_value in trait_groups.keys():
        score = 0.0
        count = 0.0
        for metric_name, weight in weights.items():
            if metric_name in rankings and len(rankings[metric_name]) > 0:
                rank = next((i for i, (s, _) in enumerate(rankings[metric_name], 1) if s == trait_value), len(rankings[metric_name]) + 1)
                max_rank = len(rankings[metric_name])
                rank_score = (max_rank+1-rank)/max_rank
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
    
    print("=" * 60)
    print("Multiple Trait Comparison Analysis")
    print("=" * 60)
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
    

    print(f"\n{'='*80}")
    print("Cross-Trait Summary")
    print(f"{'='*80}")
    
    successful_analyses = [k for k, v in results.items() if 'error' not in v]
    print(f"Successful analyses: {len(successful_analyses)}/{len(trait_analyses)}")
    
    for trait in successful_analyses:
        if 'recommendations' in results[trait]:
            best = results[trait]['recommendations'][0]
            print(f"  Best {trait}: {best[0]} (score: {best[1]:.3f})")
    
    return results

def _save_and_close(fig: Optional[plt.Figure], path: Optional[str]) -> None:
    if fig is None:
        return
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_ensemble_performance_comparison(
    ensemble_results: Dict[str, Any], 
    figsize: tuple = (12, 8),
    key_ensembles: Optional[List[str]] = None,
    show_top_n: int = None,
    show: bool = False,
):
    """
    Plot ensemble performance comparison with accuracy, improvement, and risk–benefit.
    Pass in the full dict returned by ensemble_by_trait_analysis.
    """
    ensemble_data = ensemble_results['ensemble_results']
    if key_ensembles is None:
        key_ensembles = list(ensemble_data.keys())
    if show_top_n:
        sorted_ensembles = sorted(
            [(name, data) for name, data in ensemble_data.items() if name in key_ensembles],
            key=lambda x: x[1].get('accuracy', 0.0),
            reverse=True
        )
        key_ensembles = [name for name, _ in sorted_ensembles[:show_top_n]]
    

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
    fig.set_layout_engine("constrained")  
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

    if show:
        plt.show()
    
    return fig



def plot_cluster_analysis(
    cluster_results: Dict[str, Any],
    figsize: tuple = (14, 6),
    show: bool = False,
):
    """
    Plot cluster analysis with demographic composition and performance metrics.
    """
    
    cluster_data = cluster_results['cluster_analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.set_layout_engine("constrained")
    fig.suptitle('Cluster-Level Error and Performance Analysis', fontsize=16, fontweight='bold')
    

    archetype_colors = {
        'Optimal Risk-Benefit': '#2ca02c',
        'Cautious': '#1f77b4', 
        'High Error-Correction': '#ff7f0e',
        'High Performer': '#d62728',
        'High Consistency': '#9467bd',
        'Similar to Neutral': '#8c564b'
    }
    

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
    

    scatter = ax2.scatter(extra_error_rates, rescue_rates, 
                         c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)
    

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
    

    if extra_error_rates and rescue_rates:
        x_margin = (max(extra_error_rates) - min(extra_error_rates)) * 0.2
        y_margin = (max(rescue_rates) - min(rescue_rates)) * 0.2
        ax2.set_xlim(min(extra_error_rates) - x_margin, max(extra_error_rates) + x_margin)
        ax2.set_ylim(min(rescue_rates) - y_margin, max(rescue_rates) + y_margin)
    

    legend_elements = []
    for archetype, color in archetype_colors.items():
        if any(cluster_info.get('archetype') == archetype for cluster_info in cluster_data.values()):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10, 
                                           label=archetype, markeredgecolor='black'))
    
    if legend_elements:
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    if show:
        plt.show()

    return fig


def plot_trait_comparison_results(
    trait_results: Dict[str, Any],
    trait_name: str = None,
    baseline_accuracy: float = None,
    figsize: tuple = (14, 8),
    show: bool = False,
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
    

    trait_performance = trait_results.get('trait_performance', trait_results.get('style_performance', {}))
    
    if not trait_performance:
        print(f"No trait performance data found for {trait_name}")
        return None
    

    if baseline_accuracy is None:
        all_accuracies = []
        for trait_val, data in trait_performance.items():
            if 'accuracies' in data:
                all_accuracies.extend(data['accuracies'])
        baseline_accuracy = np.mean(all_accuracies) if all_accuracies else 0.70
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.set_layout_engine("constrained")
    fig.suptitle(f'{trait_name.title()} Comparison Analysis (Relative to Baseline)', fontsize=16, fontweight='bold')
    
    trait_values = list(trait_performance.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(trait_values)))
    
    metrics = [
        ('accuracies', 'Accuracy Improvement', axes[0, 0], baseline_accuracy),
        ('rescue_rates', 'Rescue Rate', axes[0, 1], 0), 
        ('extra_error_rates', 'Extra Error Rate', axes[1, 0], 0), 
        ('error_magnitudes', 'Error Magnitude', axes[1, 1], 0)
    ]
    
    for metric_key, metric_name, ax, baseline in metrics:
        means = []
        stds = []
        
        for trait_val in trait_values:
            data = trait_performance[trait_val].get(metric_key, [])
            if data:
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


        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            if height>=0:
                label_y=height+std+0.001
                va = 'bottom'
            else:
                label_y=height-std-0.001
                va = 'top'
            
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.3f}±{std:.3f}', ha='center', va=va, 
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel(f'{trait_name.title()} Level', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        

        if metric_key == 'accuracies':
            ax.set_title(f'{metric_name} vs Baseline ({baseline:.3f})', fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{metric_name} by {trait_name.title()}', fontsize=14, fontweight='bold')
            
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(tv).title() for tv in trait_values], rotation=45, ha='right')
        

        if metric_key == 'accuracies':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline:.3f})')
            ax.legend()
        

        y_data = [m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)]
        if y_data:
            y_range = max(y_data) - min(y_data)
            y_margin = y_range * 0.15
            ax.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
        
        ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig


def plot_system_level_recommendations(
    ensemble_results: Dict[str, Any],
    cluster_results: Dict[str, Any],
    trait_results: Dict[str, Any] = None,
    figsize: tuple = (3.5, 6),
    baseline_line_value: float = 0.0,  
    show: bool = False, 
):
    """
    Vertical bar chart. Bars rise/fall around zero; horizontal baseline line at y=0.
    Scores are the same quantities you were computing (improvement-like).
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.set_layout_engine("constrained")
    ax.set_title('System-Level Recommendations', fontsize=16, fontweight='bold')

    recommendations, scores, colors = [], [], []
    ensemble_data = ensemble_results.get('ensemble_results', {})
    base_acc = float(ensemble_results.get('baseline_accuracy', 0.0))
    lambda_tok = 0.0005 

    best_ens_name = None
    best_bal = ensemble_results.get('recommendations', {}).get('best_balanced')
    if isinstance(best_bal, tuple):
        best_ens_name = best_bal[0]
        best_score = float(best_bal[1])
    elif best_bal:
        best_ens_name = best_bal[0]
        best_score = float(ensemble_data.get(best_ens_name, {}).get('improvement', 0.0))
    if best_ens_name:
        tok = ensemble_data.get(best_ens_name, {}).get("tokens_per_sample_sum")
        if isinstance(tok, (int, float)) and np.isfinite(tok) and tok > 0:
            best_score -= lambda_tok * tok
        recommendations.append(f"Best Ensemble\n{best_ens_name.replace('_',' ').title()}")
        scores.append(best_score)
        colors.append('#2ca02c')

    if 'recommendations' in cluster_results:
        best_cluster = cluster_results['recommendations'].get('best_accuracy')
        if isinstance(best_cluster, tuple):
            cluster_name, cluster_item = best_cluster[0], best_cluster[1]
            cluster_score = float(cluster_item.get('accuracy_mean', 0.0)) - base_acc
            recommendations.append(f"Best Cluster\n{cluster_name.replace('_',' ').title()}")
            scores.append(cluster_score)
            colors.append('#1f77b4')

    if trait_results and 'recommendations' in trait_results and trait_results['recommendations']:
        best_trait = trait_results['recommendations'][0]
        trait_name = trait_results.get('comparison_trait', 'Trait')
        trait_label = f"Best {trait_name.title()}\n{str(best_trait[0]).title()}"
        trait_score = float(best_trait[1]) / 10.0
        recommendations.append(trait_label)
        scores.append(trait_score)
        colors.append('#ff7f0e')

    safest = ensemble_results.get('recommendations', {}).get('safest')
    if safest:
        safest_name = safest[0]
        err_rate = float(ensemble_data.get(safest_name, {}).get('extra_error_rate', 0.0))
        safety_score = -err_rate
        recommendations.append(f"Safest Ensemble\n{safest_name.replace('_',' ').title()}")
        scores.append(safety_score)
        colors.append('#d62728')

    x = np.arange(len(recommendations))
    bars = ax.bar(x, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.7)

    for xi, (bar, s) in enumerate(zip(bars, scores)):
        y = bar.get_height()
        va = 'bottom' if y >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y + (0.002 if y >= 0 else -0.002),
                f'{s:.3f}', ha='center', va=va, fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(recommendations, rotation=20, ha='right')
    ax.set_ylabel('Performance Score (relative to baseline)')
    ax.grid(True, axis='y', alpha=0.3)

    ax.axhline(y=baseline_line_value, color='black', linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'Baseline ({baseline_line_value:+.3f})')
    ax.legend(loc='upper left')

    if scores:
        m = max(abs(min(scores)), abs(max(scores)))
        ax.set_ylim(-m*1.25, m*1.25)

    if show:
        plt.show()
    return fig




def create_all_tier2_visualizations(
    ensemble_results: Dict[str, Any],
    cluster_results: Dict[str, Any],
    trait_results: Dict[str, Any] = None,
    baseline_accuracy: float = None,
    show_top_ensembles: int = 8,
    save_paths: Optional[Dict[str, Optional[str]]] = None,
    *,
    case=None,
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,
    stage: str = "tier2",
    sub_case: Optional[str] = None,
):
    """
    Create Tier 2 visualizations.
    - If `save_paths` is None, default paths are built with `resolve_plot_dir`.
    - All figures are saved (if a path is defined) and then closed.
    """
    print("Creating Tier 2 Visualizations...")
    print("="*50)


    if save_paths is None:
        ensembles_dir = resolve_plot_dir(case, plots_root, strategy, stage, "ensembles", sub_case=sub_case)
        clusters_dir  = resolve_plot_dir(case, plots_root, strategy, stage, "clusters", sub_case=sub_case)
        traits_dir    = resolve_plot_dir(case, plots_root, strategy, stage, "trait_comparison", sub_case=sub_case)
        recs_dir      = resolve_plot_dir(case, plots_root, strategy, stage, "recommendations", sub_case=sub_case)

        trait_fname = None
        if trait_results:
            trait_fname = f"{trait_results.get('comparison_trait','trait')}_comparison.pdf"

        save_paths = {
            "ensemble": os.path.join(ensembles_dir, "ensemble_performance.pdf"),
            "cluster": os.path.join(clusters_dir, "cluster_analysis.pdf"),
            "trait": os.path.join(traits_dir, trait_fname) if trait_fname else None,
            "recommendations": os.path.join(recs_dir, "system_recommendations.pdf"),
        }
        print(f"[viz] Auto-saving via resolve_plot_dir under: {os.path.commonpath([ensembles_dir, clusters_dir, recs_dir])}")

    figures = {}

    try:
        print("1. Ensemble Performance Analysis...")
        fig1 = plot_ensemble_performance_comparison(
            ensemble_results,
            show_top_n=show_top_ensembles,
            show=False,
        )
        figures["ensemble"] = fig1
        _save_and_close(fig1, save_paths.get("ensemble"))
    except Exception as e:
        print(f"   Error creating ensemble plot: {e}")

    try:
        print("2. Cluster Analysis...")
        fig2 = plot_cluster_analysis(cluster_results, show=False)
        figures["cluster"] = fig2
        _save_and_close(fig2, save_paths.get("cluster"))
    except Exception as e:
        print(f"   Error creating cluster plot: {e}")

    if trait_results:
        try:
            print("3. Trait Comparison Analysis...")
            fig3 = plot_trait_comparison_results(
                trait_results,
                baseline_accuracy=baseline_accuracy,
                show=False,
            )
            figures["trait"] = fig3
            _save_and_close(fig3, save_paths.get("trait"))
        except Exception as e:
            print(f"   Error creating trait plot: {e}")

    try:
        print("4. System-Level Recommendations...")
        fig4 = plot_system_level_recommendations(
            ensemble_results, cluster_results, trait_results, show=False
        )
        figures["recommendations"] = fig4
        _save_and_close(fig4, save_paths.get("recommendations"))
    except Exception as e:
        print(f"   Error creating recommendations plot: {e}")

    print(f"\nCompleted! Created {len(figures)} visualizations.")
    return figures




def _run_overall_error_test(
    merged_df: pd.DataFrame,
    profile_demographics: Dict[str, str],
    baseline_accuracy: float,
    n_permutations: int,
    random_seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """
    Test whether the variance of per-profile accuracies around baseline is larger than expected.
    Efficient version: accumulates permuted row means without allocating profile×item arrays.
    """
    R = _rng_from(random_seed, rng)
    profile_cols = list(profile_demographics.keys())
    y_true = merged_df["true_label"].astype(str).to_numpy()

    C = np.stack([(merged_df[p].astype(str).to_numpy() == y_true).astype(float) for p in profile_cols], axis=0)
    P, N = C.shape

    prof_acc = C.mean(axis=1)
    observed_variance = float(np.var(prof_acc - baseline_accuracy))

    permuted_variances: List[float] = []
    row_sums = np.empty(P, dtype=float)

    for _ in range(n_permutations):
        row_sums.fill(0.0)
        for j in range(N):
            perm = R.permutation(P)               
            row_sums += C[perm, j]               
        acc_perm = row_sums / float(N)
        permuted_variances.append(float(np.var(acc_perm - baseline_accuracy)))

    p_value = (np.sum(np.array(permuted_variances) >= observed_variance) + 1) / (n_permutations + 1)
    return {
        "baseline_accuracy": float(baseline_accuracy),
        "profile_accuracies": {p: float(a) for p, a in zip(profile_cols, prof_acc)},
        "observed_variance": observed_variance,
        "permuted_variances": permuted_variances,
        "p_value": float(p_value),
        "n_profiles": len(profile_cols),
        "interpretation": "Item-wise permutation preserves item difficulty while destroying trait association."
    }



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
    Seeds are threaded via a numpy Generator; also returns BH/FDR across trait tests.
    """
    if baseline_col not in merged_df.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in data")

    profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
    if not profile_cols:
        raise ValueError("No profile columns found")

    baseline_accuracy = float((merged_df[baseline_col].astype(str) == merged_df['true_label'].astype(str)).mean())


    profile_demographics: Dict[str, Dict[str, str]] = {}
    for profile in profile_cols:
        tr = person_set.get_traits(profile, ["ethnicity", "gender"])
        profile_demographics[profile] = {
            "ethnicity": str(tr.get("ethnicity", "unknown")).lower(),
            "gender": str(tr.get("gender", "unknown")).lower(),
        }

    R = np.random.default_rng(random_seed)

    results = {
        "n_permutations": int(n_permutations),
        "random_seed": int(random_seed),
        "baseline_accuracy": baseline_accuracy,
        "baseline_col": baseline_col,
        "trait_tests": {},
        "overall_error_test": {},
        "summary": {}
    }


    for trait in traits:
        tr_res = _run_baseline_comparison_test(
            merged_df,
            profile_demographics,
            trait,
            baseline_accuracy,
            n_permutations,
            random_seed=random_seed,
            rng=R
        )
        results["trait_tests"][trait] = tr_res



    overall_results = _run_overall_error_test(
        merged_df,
        profile_demographics,
        baseline_accuracy,
        n_permutations,
        random_seed=random_seed,
        rng=R
    )
    results["overall_error_test"] = overall_results



    trait_ps = []
    trait_names = []
    for t, tr in results["trait_tests"].items():
        if "error" in tr:
            continue
        trait_names.append(t)
        trait_ps.append(tr["p_value"])
    if "p_value" in overall_results:
        trait_names.append("overall")
        trait_ps.append(overall_results["p_value"])

    if trait_ps:
        _, q_all, _, _ = multipletests(trait_ps, method="fdr_bh")
        q_map = {name: float(q) for name, q in zip(trait_names, q_all)}
        results["fdr_permutation"] = q_map
        for t in results["trait_tests"]:
            if "error" not in results["trait_tests"][t] and t in q_map:
                results["trait_tests"][t]["q_value"] = q_map[t]
        if "overall" in q_map:
            results["overall_error_test"]["q_value"] = q_map["overall"]


    results["summary"] = _summarize_baseline_permutation_results(results)
    return results




def _run_baseline_comparison_test(
    merged_df: pd.DataFrame,
    profile_demographics: Dict[str, Dict[str, str]],
    trait: str,
    baseline_accuracy: float,
    n_permutations: int,
    random_seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Any]:
    """
    Variance-aware trait test vs. baseline using item-wise permutations.
    Efficient version: no full matrix copies; computes stats by accumulating per-item means.
    """
    R = _rng_from(random_seed, rng)
    profile_cols = list(profile_demographics.keys())
    if len(profile_cols) == 0:
        return {"error": "No profiles provided"}

    trait_of = {p: (profile_demographics[p].get(trait, "unknown") or "unknown") for p in profile_cols}
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(profile_cols):
        groups.setdefault(trait_of[p], []).append(i)
    groups = {g: idxs for g, idxs in groups.items() if idxs}
    if len(groups) < 2:
        return {"error": f"Insufficient groups for {trait} baseline comparison test"}

    y_true = merged_df["true_label"].astype(str).to_numpy()
    C = np.stack([(merged_df[p].astype(str).to_numpy() == y_true).astype(float) for p in profile_cols], axis=0)
    P, N = C.shape

    G = list(groups.keys())
    idxs = {g: np.asarray(groups[g], dtype=int) for g in G}

    def per_item_means_direct(C_mat, idx_array):
        return C_mat[idx_array, :].mean(axis=0)

    if len(G) == 2:
        gA, gB = G[0], G[1]
        mA = per_item_means_direct(C, idxs[gA])
        mB = per_item_means_direct(C, idxs[gB])
        d = mA - mB
        obs_t = float(d.mean() / (d.std(ddof=1) / (np.sqrt(len(d)) + 1e-12) + 1e-12))

        perm_stats: List[float] = []
        for _ in range(n_permutations):
            count = 0
            mean_d = 0.0
            M2 = 0.0
            for j in range(N):
                perm = R.permutation(P)
                mA_j = float(C[perm[idxs[gA]], j].mean())
                mB_j = float(C[perm[idxs[gB]], j].mean())
                d_j = mA_j - mB_j
                count += 1
                delta = d_j - mean_d
                mean_d += delta / count
                M2 += delta * (d_j - mean_d)
            var_d = M2 / max(count - 1, 1)
            t_p = float(mean_d / (np.sqrt(var_d / max(count, 1)) + 1e-12))
            perm_stats.append(t_p)

        p_value = (np.sum(np.abs(perm_stats) >= abs(obs_t)) + 1) / (n_permutations + 1)
        stat_name, observed_stat = "studentized_mean_diff", obs_t

    else:
        def item_F_for_perm():
            total_F = 0.0
            for j in range(N):
                perm = R.permutation(P)
                m_g = np.array([float(C[perm[idxs[g]], j].mean()) for g in G], dtype=float)
                n_g = np.array([len(idxs[g]) for g in G], dtype=float)
                grand = np.average(m_g, weights=n_g)
                ssb = float(np.sum(n_g * (m_g - grand) ** 2))
                dfb = len(G) - 1
                msb = ssb / max(dfb, 1)
                ssw = 0.0
                dfw = 0
                for g in G:
                    x = C[perm[idxs[g]], j]
                    ssw += float(((x - x.mean()) ** 2).sum())
                    dfw += (len(x) - 1)
                msw = ssw / max(dfw, 1)
                total_F += msb / (msw + 1e-12)
            return total_F / float(N)


        total_F_obs = 0.0
        for j in range(N):
            m_g = np.array([float(C[idxs[g], j].mean()) for g in G], dtype=float)
            n_g = np.array([len(idxs[g]) for g in G], dtype=float)
            grand = np.average(m_g, weights=n_g)
            ssb = float(np.sum(n_g * (m_g - grand) ** 2))
            dfb = len(G) - 1
            msb = ssb / max(dfb, 1)
            ssw = 0.0
            dfw = 0
            for g in G:
                x = C[idxs[g], j]
                ssw += float(((x - x.mean()) ** 2).sum())
                dfw += (len(x) - 1)
            msw = ssw / max(dfw, 1)
            total_F_obs += msb / (msw + 1e-12)
        obs_F = total_F_obs / float(N)

        perm_stats = [item_F_for_perm() for _ in range(n_permutations)]
        p_value = (np.sum(np.array(perm_stats) >= obs_F) + 1) / (n_permutations + 1)
        stat_name, observed_stat = "mean_F_itemwise", obs_F

    group_accuracies = {g: float(C[idxs[g], :].mean()) for g in G}
    return {
        "trait": trait,
        "baseline_accuracy": float(baseline_accuracy),
        "group_sizes": {g: int(len(idxs[g])) for g in G},
        "group_accuracies": group_accuracies,
        "observed_test_statistic": float(observed_stat),
        "statistic_name": stat_name,
        "permuted_test_statistics": [float(x) for x in perm_stats],
        "p_value": float(p_value),
        "interpretation": "Item-wise permutation with variance-normalized statistic (efficient).",
    }





def _summarize_baseline_permutation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics for baseline-focused permutation test results."""
    
    summary = {
        "significant_trait_tests": 0,
        "total_trait_tests": 0,
        "significant_overall_error": False,
        "min_p_value": 1.0,
        "baseline_accuracy": results.get("baseline_accuracy", 0.0),
        "significant_findings": []
    }
    

    for trait, trait_results in results["trait_tests"].items():
        if "error" in trait_results:
            continue
            
        summary["total_trait_tests"] += 1
        p_value = trait_results["p_value"]
        summary["min_p_value"] = min(summary["min_p_value"], p_value)
        
        if p_value < 0.05:
            summary["significant_trait_tests"] += 1
            summary["significant_findings"].append({
                "type": f"{trait}_error_vs_baseline",
                "p_value": p_value,
                "test_statistic": trait_results["observed_test_statistic"],
                "interpretation": f"{trait} groups deviate from baseline more than expected by chance"
            })
    

    if "overall_error_test" in results and "error" not in results["overall_error_test"]:
        overall_p = results["overall_error_test"]["p_value"]
        summary["min_p_value"] = min(summary["min_p_value"], overall_p)
        
        if overall_p < 0.05:
            summary["significant_overall_error"] = True
            summary["significant_findings"].append({
                "type": "overall_profile_variance",
                "p_value": overall_p,
                "observed_variance": results["overall_error_test"]["observed_variance"],
                "interpretation": "Profile variance around baseline exceeds random expectation"
            })
    
    return summary


def print_permutation_results(results: Dict[str, Any]) -> None:
    """Print comprehensive baseline-focused permutation test results."""

    print("\n" + "="*80)
    print("Permutation test vs. Baseline")
    print("="*80)


    n_perm = results.get("n_permutations", None)
    seed   = results.get("random_seed", None)
    base   = results.get("baseline_accuracy", float("nan"))
    base_col = results.get("baseline_col", "base_pred")

    print(f"\nTest Configuration:")
    if n_perm is not None: print(f"  - Number of permutations: {n_perm:,}")
    if seed   is not None: print(f"  - Random seed: {seed}")
    print(f"  - Baseline accuracy: {base:.4f}")
    print(f"  - Baseline column: {base_col}")

    print(f"\nNull Hypothesis:")
    print(f"  Demographic conditioning introduces no systematic error beyond")
    print(f"  what would be expected from random variation around baseline performance.")


    for trait, trait_results in results.get("trait_tests", {}).items():
        if "error" in trait_results:
            print(f"\n{trait.upper()} vs BASELINE: {trait_results['error']}")
            continue

        print(f"\n{trait.upper()} ERROR-PATTERN ANALYSIS TEST (vs Baseline):")
        print("-" * 50)


        group_sizes = trait_results.get("group_sizes", {})
        if group_sizes:
            print(f"Groups: { {g: group_sizes[g] for g in group_sizes} }")


        stat_name = trait_results.get("statistic_name", "test_statistic")
        print(f"Test: {stat_name} (item-wise permutation)")


        obs = trait_results.get("observed_test_statistic", float("nan"))
        p_val = trait_results.get("p_value", float("nan"))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"Observed test statistic: {obs:.6f}")
        print(f"Permutation p-value: {p_val:.4f} {sig}")

        group_accs = trait_results.get("group_accuracies", {})
        if group_accs:
            print(f"\nGroup accuracies vs baseline ({base:.4f}):")
            for group, acc in group_accs.items():
                deviation = float(acc) - float(base)
                sign = "+" if deviation >= 0 else ""
                print(f"  {group}: {acc:.4f} ({sign}{deviation:.4f})")

    overall = results.get("overall_error_test", {})
    if overall and "error" not in overall:
        print(f"\nOVERALL PROFILE VARIANCE TEST:")
        print("-" * 50)
        n_prof = overall.get("n_profiles", None)
        if n_prof is not None:
            print(f"Profiles tested: {n_prof}")
        obs_var = overall.get("observed_variance", float("nan"))
        print(f"Observed variance around baseline: {obs_var:.6f}")
        p_val = overall.get("p_value", float("nan"))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"Permutation p-value: {p_val:.4f} {sig}")


    summary = results.get("summary", {})
    print(f"\nSummary:")
    print("-" * 30)
    total = summary.get("total_trait_tests", 0)
    sig_t = summary.get("significant_trait_tests", 0)
    print(f"  - Trait tests performed: {total}")
    print(f"  - Significant trait errors: {sig_t}")
    if total > 0:
        sig_rate = sig_t / total
        print(f"  - Significance rate: {sig_rate:.1%}")
    print(f"  - Minimum p-value: {summary.get('min_p_value', 1.0):.4f}")
    print(f"  - Overall error significant: {summary.get('significant_overall_error', False)}")

    if summary.get("significant_findings"):
        print(f"\nSignificant Findings:")
        for finding in summary["significant_findings"]:
            print(f"  - {finding['type']}: p={finding['p_value']:.4f}")
            print(f"    {finding['interpretation']}")
    else:
        print(f"\nNo Significant Findings Detected:")
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

    if "trait_tests" not in results or trait not in results["trait_tests"]:
        raise ValueError(f"Trait {trait} not found in results['trait_tests'].")

    trait_results = results["trait_tests"][trait]
    if "error" in trait_results:
        raise ValueError(f"Error in {trait} results: {trait_results['error']}")

    baseline_acc = float(results.get("baseline_accuracy", float("nan")))
    stat_name = trait_results.get("statistic_name", "test_statistic")
    null_dist = trait_results.get("permuted_test_statistics", [])
    observed  = float(trait_results.get("observed_test_statistic", float("nan")))
    p_val     = float(trait_results.get("p_value", float("nan")))

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)


    axes[0, 0].hist(null_dist, bins=50, alpha=0.7, density=True,
                    edgecolor='black', linewidth=0.5)
    axes[0, 0].axvline(observed, linestyle='--', linewidth=2,
                       label=f'Observed: {observed:.6f}')
    axes[0, 0].set_xlabel(stat_name.replace("_", " ").title())
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'{trait.title()} Test ({stat_name})\np = {p_val:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    group_accs = trait_results.get("group_accuracies", {})
    groups = list(group_accs.keys())
    accs   = [float(group_accs[g]) for g in groups]
    deviations = [a - baseline_acc for a in accs]

    axes[0, 1].barh(range(len(groups)), deviations)
    axes[0, 1].set_yticks(range(len(groups)))
    axes[0, 1].set_yticklabels([g.replace('_', ' ').title() for g in groups], fontsize=9)
    axes[0, 1].axvline(0, linewidth=1)
    axes[0, 1].set_xlabel('Deviation from Baseline')
    axes[0, 1].set_title(f'{trait.title()} Group Deviations\n(Baseline: {baseline_acc:.4f})')
    axes[0, 1].grid(True, alpha=0.3)

    for i, (acc, dev) in enumerate(zip(accs, deviations)):
        axes[0, 1].text(dev + (0.002 if dev >= 0 else -0.002), i, f'{acc:.3f}',
                        ha='left' if dev >= 0 else 'right', va='center', fontsize=8)


    overall = results.get("overall_error_test", {})
    if overall and "error" not in overall:
        null_variances = overall.get("permuted_variances", [])
        observed_var   = float(overall.get("observed_variance", float("nan")))
        p_val_overall  = float(overall.get("p_value", float("nan")))

        axes[1, 0].hist(null_variances, bins=50, alpha=0.7, density=True,
                        edgecolor='black', linewidth=0.5)
        axes[1, 0].axvline(observed_var, linestyle='--', linewidth=2,
                           label=f'Observed: {observed_var:.6f}')
        axes[1, 0].set_xlabel('Variance of Profile Accuracies around Baseline')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'Overall Profile Variance Test\np = {p_val_overall:.4f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Overall variance\ntest not available',
                        ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Overall Variance Test')


    p_values = []
    labels   = []


    p_values.append(p_val)
    labels.append(f'{trait.title()} error')


    if overall and "error" not in overall:
        p_values.append(float(overall.get("p_value", float("nan"))))
        labels.append('Overall variance')


    for other_trait, other in results.get("trait_tests", {}).items():
        if other_trait == trait or "error" in other:
            continue
        p_values.append(float(other.get("p_value", float("nan"))))
        labels.append(f'{other_trait.title()} error')

    if p_values:
        axes[1, 1].barh(range(len(p_values)), p_values)
        axes[1, 1].set_yticks(range(len(p_values)))
        axes[1, 1].set_yticklabels(labels, fontsize=9)
        axes[1, 1].axvline(0.05, linestyle=':', label='α = 0.05')
        axes[1, 1].set_xlabel('P-value')
        axes[1, 1].set_title('Permutation Test P-values')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, max(0.1, max([v for v in p_values if np.isfinite(v)]) * 1.1))
    else:
        axes[1, 1].text(0.5, 0.5, 'No p-values\navailable',
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('P-value Summary')

    plt.suptitle(
        f'Permutation Tests vs. Baseline: {trait.title()} ({stat_name})\n'
        f'({results.get("n_permutations", "N")} permutations, baseline: {baseline_acc:.4f})',
        fontsize=14
    )

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return fig



def stratified_dev_test_split(
    merged_df: pd.DataFrame,
    category_col: str = "stereotype_type",
    test_size: float = DEV_TEST_SPLIT,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if category_col not in merged_df.columns:
        idx = np.arange(len(merged_df))
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
        return merged_df.iloc[train_idx].copy(), merged_df.iloc[test_idx].copy()

    train_parts, test_parts = [], []
    for v, sub in merged_df.groupby(category_col, dropna=False):
        idx = np.arange(len(sub))
        if len(sub) < 2:
            train_parts.append(sub)
            continue
        tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
        train_parts.append(sub.iloc[tr_idx])
        test_parts.append(sub.iloc[te_idx])
    dev = pd.concat(train_parts).sample(frac=1.0, random_state=random_state) 
    test = pd.concat(test_parts) if len(test_parts) else merged_df.iloc[0:0].copy()

    return dev.reset_index(drop=True), test.reset_index(drop=True)

def dev_test_selection_pipeline(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    case: CaseConfig,
    group_keys=("gender","ethnicity"),
    perf_df: Optional[pd.DataFrame] = None,
    min_category_n: int = MIN_CATEGORY_N,
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:

    dev_df, test_df = stratified_dev_test_split(merged_df, category_col=(getattr(case,"category_cols",["stereotype_type"])[0]))
    print(f"[dev/test] dev n={len(dev_df)}, test n={len(test_df)}")

    ens_dev = ensemble_by_trait_analysis(dev_df, person_set, case=case, group_keys=group_keys, perf_df=perf_df)
    
    best_name = ens_dev.get("recommendations",{}).get("best_balanced",[None])[0]
    if best_name is None:
        return {"error": "No best ensemble found on dev."}

    profiles = ens_dev["ensemble_results"][best_name]["profiles"]

    dev_preds  = majority_vote_ensemble(dev_df, profiles)
    test_preds = majority_vote_ensemble(test_df, profiles)

    dev_boot = paired_bootstrap_report_global(dev_df["base_pred"], dev_preds, dev_df["true_label"], n_boot=n_boot, random_state=random_state)
    test_boot= paired_bootstrap_report_global(test_df["base_pred"], test_preds, test_df["true_label"], n_boot=n_boot, random_state=random_state)
    dev_boot = _coerce_scalars(dev_boot)
    test_boot = _coerce_scalars(test_boot)
    selection_gap = float(dev_boot["delta_acc"] - test_boot["delta_acc"])


    category_col = (getattr(case,"category_cols",["stereotype_type"]) or ["stereotype_type"])[0]
    dev_cat  = paired_bootstrap_report_by_category(dev_df, dev_preds, category_col=category_col, min_n=min_category_n, n_boot=n_boot, random_state=random_state)
    test_cat = paired_bootstrap_report_by_category(test_df, test_preds, category_col=category_col, min_n=min_category_n, n_boot=n_boot, random_state=random_state)


    dev_global = pd.DataFrame([{"ensemble": best_name, **dev_boot}])
    test_global= pd.DataFrame([{"ensemble": best_name, **test_boot}])
    test_cat["ensemble"] = best_name

    qtables = fdr_families(
        global_table=pd.DataFrame([{"ensemble": best_name, **test_boot}]),
        percat_table=test_cat
    )
    devtest_eval = {
        "best_ensemble": best_name,
        "profiles": profiles,
        "dev_global": dev_global,
        "test_global": test_global,
        "selection_gap": selection_gap,
        "dev_per_category": dev_cat,
        "test_per_category": test_cat,
        "fdr": qtables,
        "min_category_n": min_category_n,
        "suppressed_in_test": _suppressed_category_values(test_df, category_col, min_n=min_category_n),
        "inference": "confirmatory"
    }
    print(f"[Selection gap] Δacc(dev) − Δacc(test): {selection_gap:+.4f}")

    return devtest_eval


def bootstrap_all_ensembles_global(
    merged_df: pd.DataFrame,
    ens_results: Dict[str, Any],
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    rows = []
    true = merged_df["true_label"]
    base = merged_df["base_pred"]
    for name, d in (ens_results.get("ensemble_results") or {}).items():
        preds = d.get("ensemble_preds")
        if preds is None or preds.eq("").all():
            continue
        g = paired_bootstrap_report_global(base, preds, true, n_boot=n_boot, random_state=random_state)
        rows.append({"ensemble": name, **g})
    return pd.DataFrame(rows).sort_values("acc_ens", ascending=False).reset_index(drop=True)

def _safe_idxmax(s: pd.Series):
    s = s.copy()
    if s.isna().all() or len(s) == 0:
        return None
    return s.idxmax()

def _require_cols(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c not in df.columns]

def _as_scalar(name, v):
    arr = np.asarray(v).squeeze()
    if arr.size != 1:
        raise ValueError(f"{name} expected scalar but got shape {arr.shape}. "
                         f"Check paired_bootstrap_report_global returns scalars.")
    return float(arr.item())

def _force_scalar(x, name: str = "value") -> float:
    """Return a float even if x is a numpy array/list; use mean if not size-1."""
    arr = np.asarray(x).squeeze()
    if arr.size == 1:
        return float(arr)
    print(f"[WARN] {name} had shape {arr.shape}; using mean()")
    return float(arr.mean())



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
    per_figure_subdirs: Optional[Dict[str, str]] = None,
    show: bool = False,
    sub_case: Optional[str] = None,
):
    """
    Run complete Tier 2 analysis pipeline with:
      - Pre-specified inclusion thresholds (drop category cells with n < MIN_CATEGORY_N)
      - Hold-out selection (dev) and strict reporting (test) with selection-bias gap
      - Paired bootstrap CIs + McNemar p for ensembles and slices
      - Multiple comparisons control (BH/FDR) for global and per-category Δacc
      - Cost/efficiency Pareto frontier as a main result + sensitivity sweep
    """

    MIN_N = globals().get("MIN_CATEGORY_N", 50)
    N_BOOT = globals().get("N_BOOT", 2000)
    DEV_TEST = globals().get("DEV_TEST_SPLIT", 0.30)
    RNG = globals().get("RANDOM_STATE", 42)
    FALLBACK_CFG = {}

    def _worst_case_halfwidth(n: int) -> float:
        return 1.96*float(np.sqrt(0.25/max(n, 1)))

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
    primary_category = category_cols[0] if category_cols else "stereotype_type"

    subdirs = per_figure_subdirs or {}
    stage_dir     = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage, sub_case=sub_case)
    ensembles_dir = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("ensembles") or "ensembles", sub_case=sub_case)
    clusters_dir  = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("clusters") or "clusters", sub_case=sub_case)
    traits_dir    = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("traits") or "trait_comparison", sub_case=sub_case)
    recs_dir      = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("recommendations") or "recommendations", sub_case=sub_case)
    perm_dir      = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("permutations") or "permutation_tests", sub_case=sub_case)
    pareto_dir    = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage,
                                     extra_subdir=subdirs.get("pareto") or "pareto", sub_case=sub_case)


    suppressed_summary = {}
    for cat_col in category_cols:
        if cat_col in merged_df.columns:
            counts = merged_df[cat_col].fillna("Unknown").value_counts()
            suppressed = counts[counts < MIN_N]
            if len(suppressed):
                print(f"[INFO] Suppressing low-n values in {cat_col} (n<{MIN_N}): "
                      f"{', '.join([f'{k} (n={v})' for k, v in suppressed.items()])}")
                for k, v in suppressed.items():
                    print(f"   - Worst-case 95% half-width at n={v}: ±{_worst_case_halfwidth(int(v))*100:.1f}pp")
            suppressed_summary[cat_col] = suppressed.to_dict()

    try:
        dev_df, test_df = stratified_dev_test_split(
            merged_df, category_col=primary_category, test_size=DEV_TEST, random_state=RNG
        )
        print(f"[dev/test] dev n={len(dev_df)}, test n={len(test_df)} (stratified by {primary_category})")
    except Exception as e:
        print(f"ERROR: Dev/Test split failed ({e}), falling back to random split.")
        dev_df, test_df = train_test_split(merged_df, test_size=DEV_TEST, random_state=RNG, shuffle=True)
        dev_df, test_df = dev_df.reset_index(drop=True), test_df.reset_index(drop=True)

    try:
        primary_category, EFFECTIVE_MIN_N, fallback_mode = _apply_category_fallbacks(
            dev_df, test_df, merged_df, primary_category, MIN_N, cfg=FALLBACK_CFG
        )
        print(f"[Tier-2] Primary category for per-category tests: {primary_category} (mode={fallback_mode}, min_n={EFFECTIVE_MIN_N})")
    except Exception as e:
        print(f"WARNING: Category fallback failed ({e}), using original settings")
        EFFECTIVE_MIN_N = MIN_N
        fallback_mode = "none"


    try:
        print("\n=== Running Step 1: Ensemble by Trait Analysis (dev selection) ...")
        ens_dev = ensemble_by_trait_analysis(
            dev_df,
            person_set,
            case=case,
            group_keys=group_keys,
            perf_df=perf_df
        )
        print("SUCCESS: Ensemble analysis (dev) completed successfully")
    except Exception as e:
        print(f"ERROR: Ensemble analysis on dev failed: {e}")
        ens_dev = {'error': str(e), 'ensemble_results': {}}


    best_ensemble = None
    try:
        best_ensemble = ens_dev.get('recommendations', {}).get('best_balanced', [None])[0]
        print(f"[dev] Selected recommended ensemble: {best_ensemble}")
    except Exception:
        pass

    ens_test_eval = {"ensemble_results": {}}
    try:
        candidates = list((ens_dev.get("ensemble_results") or {}).keys())
        for name in candidates:
            profiles = ens_dev["ensemble_results"][name].get("profiles", [])
            if not profiles:
                continue
            preds_test = majority_vote_ensemble(test_df, profiles)
            ens_test_eval["ensemble_results"][name] = {
                "ensemble_preds": preds_test,
                "profiles": profiles
            }
    except Exception as e:
        print(f"WARNING: Could not prepare test-time ensemble preds for all ensembles: {e}")
    
    ens_test_eval = _enrich_ensembles_with_metrics_for_split(
        test_df, ens_test_eval, perf_df=perf_df, baseline_col="base_pred"
    )


    devtest_eval = {}
    try:
        if best_ensemble is None:
            raise RuntimeError("No best ensemble available from dev.")
        best_profiles = ens_dev["ensemble_results"][best_ensemble]["profiles"]
        dev_preds  = majority_vote_ensemble(dev_df,  best_profiles)
        test_preds = majority_vote_ensemble(test_df, best_profiles)

        dev_boot = paired_bootstrap_report_global(
            dev_df["base_pred"], dev_preds, dev_df["true_label"], n_boot=N_BOOT, random_state=RNG
        )
        test_boot = paired_bootstrap_report_global(
            test_df["base_pred"], test_preds, test_df["true_label"], n_boot=N_BOOT, random_state=RNG
        )

        selection_gap = _force_scalar(dev_boot.get("delta_acc"), "dev.delta_acc") - \
                        _force_scalar(test_boot.get("delta_acc"), "test.delta_acc")


        dev_cat  = paired_bootstrap_report_by_category(
            dev_df, dev_preds,
            category_col=primary_category,
            min_n=EFFECTIVE_MIN_N, n_boot=N_BOOT, random_state=RNG
        )
        test_cat = paired_bootstrap_report_by_category(
            test_df, test_preds,
            category_col=primary_category,
            min_n=EFFECTIVE_MIN_N, n_boot=N_BOOT, random_state=RNG
        )
        test_cat["ensemble"] = best_ensemble

        qtables = fdr_families(
            global_table=pd.DataFrame([{"ensemble": best_ensemble, **test_boot}]),
            percat_table=test_cat
        )

        devtest_eval = {
            "best_ensemble": best_ensemble,
            "dev_global": pd.DataFrame([{"ensemble": best_ensemble, **dev_boot}]),
            "test_global": pd.DataFrame([{"ensemble": best_ensemble, **test_boot}]),
            "selection_gap": float(selection_gap),
            "dev_per_category": dev_cat,
            "test_per_category": test_cat,
            "fdr": qtables,
            "min_category_n": MIN_N,
            "suppressed_in_test": _suppressed_category_values(test_df, primary_category, min_n=MIN_N)
        }
        print(f"[Selection gap] Δacc(dev) − Δacc(test): {selection_gap:+.4f}")
    except Exception as e:
        print(f"ERROR: Dev/Test evaluation failed: {e}")
        devtest_eval = {"error": str(e), "best_ensemble": best_ensemble}


    ens_dev_eval = {"ensemble_results": {}}
    for name, d in (ens_dev.get("ensemble_results") or {}).items():
        profiles = d.get("profiles", [])
        if not profiles:
            continue
        preds_dev = majority_vote_ensemble(dev_df, profiles)
        ens_dev_eval["ensemble_results"][name] = {"ensemble_preds": preds_dev, "profiles": profiles}
    ens_dev_eval = _enrich_ensembles_with_metrics_for_split(dev_df, ens_dev_eval, perf_df=perf_df, baseline_col="base_pred")
    

    global_test_ci = bootstrap_all_ensembles_global(test_df, ens_test_eval, n_boot=N_BOOT, random_state=RNG)
    

    qglob = fdr_families(global_table=global_test_ci, percat_table=pd.DataFrame())
    q_global_delta  = qglob.get("q_global_delta_acc",   pd.DataFrame())
    q_global_rescue = qglob.get("q_global_rescue",      pd.DataFrame())
    q_global_extra  = qglob.get("q_global_extra_error", pd.DataFrame())
    

    criteria_eval = evaluate_prespecified_criteria_devselect(
        ens_dev_eval=ens_dev_eval,
        ens_test_boot=global_test_ci,
        criteria=PRE_SPEC_CRITERIA
    )


    try:
        pareto_named = pareto_prespecified_devselect(
            ens_dev_eval=ens_dev_eval,
            ens_test_eval=ens_test_eval,
            test_boot_df=global_test_ci,
            case=case,
            lambdas=PRE_SPEC_PARETO
        )
    except Exception as e:
        print(f"WARNING: Pre-specified Pareto evaluation failed: {e}")
        pareto_named = pd.DataFrame()


    try:
        print("\n=== Cost/Efficiency Pareto (TEST) ===")
        pareto_test = pareto_frontier_for_ensembles(
            ensemble_results=ens_test_eval,
            merged_df=test_df,
            case=case,
            lambda_tok=5e-4,     
            lambda_extra=2.0,  
            out_dir=pareto_dir
        )
    except Exception as e:
        print(f"ERROR: Pareto frontier analysis (TEST) failed: {e}")
        pareto_test = {'error': str(e)}

    try:
        sweep = pareto_sensitivity_sweep(
            ensemble_results=ens_test_eval,
            merged_df=test_df,
            case=case,
            lambda_tok_grid=(1e-4, 5e-4, 1e-3, 2e-3),
            lambda_extra_grid=(0.5, 1.0, 2.0, 3.0),
            out_dir=pareto_dir
        )
        print("[Pareto stability] Top ensembles by count across λ-grid:")
        print(sweep["stability"].head())
    except Exception as e:
        print(f"WARNING: Pareto sensitivity sweep failed: {e}")
        sweep = {"grid": pd.DataFrame(), "stability": pd.DataFrame()}

    try:
        print("\n=== Running Step 2: Cluster-level Error Analysis...")
        similarity_results = analyze_profile_similarity(
            merged_df,
            person_set=person_set
        )
        cluster_results = cluster_level_error_patterns(
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


    print(f"\n\n=== Running Step {3+j}: Permutation Tests (TEST split) ===")
    try:
        permutation_results = run_permutation_tests(
            test_df,
            person_set=person_set,
            n_permutations=n_permutations,
            traits=["gender", "ethnicity"],
            random_seed=permutation_seed,
            baseline_col="base_pred"
        )
        print_permutation_results(permutation_results)
    except Exception as e:
        print(f"Error running permutation tests: {e}")
        permutation_results = {"error": str(e)}

    try:
        if isinstance(permutation_results, dict) and "trait_tests" in permutation_results:
            for trait in ("ethnicity", "gender"):
                if trait in permutation_results["trait_tests"] and "error" not in permutation_results["trait_tests"][trait]:
                    savepath = os.path.join(perm_dir, f"permutation_{trait}.pdf")
                    plot_permutation_distributions(permutation_results, trait=trait, savepath=savepath)
                    print(f"[Permutation plots] Saved: {savepath}")
    except Exception as e:
        print(f"WARNING: Failed to save permutation plots: {e}")


    visualization_figures = {}
    if create_visualizations:
        try:
            print("\n=== Running Step 4: Creating Visualizations (standard suite) ===")
            try:
                ensemble_results_full = ensemble_by_trait_analysis(
                    merged_df, person_set, case=case, group_keys=group_keys, perf_df=perf_df
                )
            except Exception:
                ensemble_results_full = ens_dev 

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
                ensemble_results=ensemble_results_full,
                cluster_results=cluster_results,
                trait_results=trait_results_for_plots,
                show_top_ensembles=8,
                save_paths=save_paths,
                case=case,                    
                plots_root=plots_root,     
                strategy=strategy,    
                stage=stage,       
            )
            print("SUCCESS: Visualizations created successfully")
        except Exception as e:
            print(f"✗ ERROR: Visualization creation failed: {e}")
            visualization_figures = {'error': str(e)}


    try:
        os.makedirs(stage_dir, exist_ok=True)
        if isinstance(devtest_eval.get("dev_global"), pd.DataFrame):
            devtest_eval["dev_global"].to_csv(os.path.join(stage_dir, "dev_global.csv"), index=False)
        if isinstance(devtest_eval.get("test_global"), pd.DataFrame):
            devtest_eval["test_global"].to_csv(os.path.join(stage_dir, "test_global.csv"), index=False)
        if isinstance(devtest_eval.get("test_per_category"), pd.DataFrame):
            devtest_eval["test_per_category"].to_csv(os.path.join(stage_dir, "test_per_category.csv"), index=False)
        if isinstance(devtest_eval.get("fdr", {}).get("q_per_category_delta_acc"), pd.DataFrame):
            devtest_eval["fdr"]["q_per_category_delta_acc"].to_csv(os.path.join(stage_dir, "test_per_category_FDR.csv"), index=False)
        if 'global_test_ci' in locals() and isinstance(global_test_ci, pd.DataFrame) and len(global_test_ci):
            global_test_ci.to_csv(os.path.join(stage_dir, "test_global_all_ensembles.csv"), index=False)
        if isinstance(global_test_ci, pd.DataFrame) and len(global_test_ci):
            global_test_ci.to_csv(os.path.join(stage_dir, "test_global_all_ensembles.csv"), index=False)
        
        if isinstance(q_global_delta, pd.DataFrame) and len(q_global_delta):
            q_global_delta.to_csv(os.path.join(stage_dir, "test_global_FDR_delta.csv"), index=False)
        if isinstance(q_global_rescue, pd.DataFrame) and len(q_global_rescue):
            q_global_rescue.to_csv(os.path.join(stage_dir, "test_global_FDR_rescue.csv"), index=False)
        if isinstance(q_global_extra, pd.DataFrame) and len(q_global_extra):
            q_global_extra.to_csv(os.path.join(stage_dir, "test_global_FDR_extra.csv"), index=False)

        if isinstance(sweep.get("grid"), pd.DataFrame) and len(sweep["grid"]):
            sweep["grid"].to_csv(os.path.join(pareto_dir, "pareto_sensitivity_grid.csv"), index=False)
        if isinstance(sweep.get("stability"), pd.DataFrame) and len(sweep["stability"]):
            sweep["stability"].to_csv(os.path.join(pareto_dir, "pareto_stability_counts.csv"), index=False)
        pd.DataFrame([
            {"category": c, "value": v, "n": n, "hw_95pp": _worst_case_halfwidth(int(n))*100.0}
            for c, d in suppressed_summary.items() for v, n in d.items()
        ]).to_csv(os.path.join(stage_dir, "suppressed_cells_appendix.csv"), index=False)
    except Exception as e:
        print(f"WARNING: Failed to save CSV artifacts: {e}")


    print("\n" + "=" * 80)
    print("Comprehensive Tier 2 Executive Summary")
    print("=" * 80)
    summary = {}

    if best_ensemble:
        summary['best_ensemble_dev_selected'] = best_ensemble
        print(f"   - Best Ensemble (selected on DEV): {best_ensemble}")

    if isinstance(devtest_eval.get("test_global"), pd.DataFrame) and not devtest_eval["test_global"].empty:
        tg = devtest_eval["test_global"].iloc[0]
        print(f"   - TEST Δacc vs baseline: {tg.get('delta_acc', float('nan')):+.4f} "
              f"[95% CI {tg.get('delta_acc_ci', (np.nan, np.nan))[0]:+.4f}, "
              f"{tg.get('delta_acc_ci', (np.nan, np.nan))[1]:+.4f}] "
              f"| McNemar p={tg.get('mcnemar_p', float('nan')):.4g}")

    if "selection_gap" in devtest_eval:
        print(f"   - Selection gap (dev-test accuracy difference): {devtest_eval['selection_gap']:+.4f}")

    if isinstance(pareto_test, dict) and pareto_test.get('recommended'):
        r = pareto_test['recommended']
        try:
            print(f"   - Pareto (test) recommendation: {r.get('ensemble')} | "
                  f"acc={float(r.get('accuracy', float('nan'))):.3f}, "
                  f"extra={float(r.get('extra_error_rate', float('nan'))):.3f}, "
                  f"tokens={float(r.get('tokens_per_sample_sum', float('nan'))):.0f}")
        except Exception:
            print(f"   - Pareto (test) recommendation: {r}")

    if isinstance(cluster_results, dict) and 'recommendations' in cluster_results:
        best_cluster = cluster_results['recommendations'].get('best_accuracy', ['Unknown'])[0]
        summary['best_cluster'] = best_cluster
        print(f"   - Best Cluster: {best_cluster}")

    if has_cognitive_data and isinstance(cognitive_results, dict) \
       and ('error' not in cognitive_results) and ('skipped' not in cognitive_results):
        if 'recommendations' in cognitive_results and cognitive_results['recommendations']:
            best_cog = cognitive_results['recommendations'][0]
            best_cog_name = best_cog[0] if isinstance(best_cog, (list, tuple)) else str(best_cog)
            summary['best_cognitive_style'] = best_cog_name
            print(f"   - Best Cognitive Style: {best_cog_name}")
    else:
        if isinstance(cognitive_results, dict) and 'skipped' in cognitive_results:
            print(f"   - Cognitive style analysis skipped: {cognitive_results.get('reason', 'No reason provided')}")
        else:
            err = (cognitive_results or {}).get('error', 'Unknown error')
            print(f"   - Cognitive style analysis failed: {err}")

    return {
        'dev_test': {
            **devtest_eval,
            'global_test_all_ensembles': global_test_ci if 'global_test_ci' in locals() else pd.DataFrame(),
            'q_global_test': {
                'delta':  q_global_delta,
                'rescue': q_global_rescue,
                'extra':  q_global_extra,
            },
            'criteria_eval': criteria_eval
        },
        'pareto_test': pareto_test,
        'pareto_prespecified': pareto_named,
        'pareto_sensitivity': sweep,
        'cluster_analysis': cluster_results,
        'cognitive_analysis': cognitive_results,
        'permutation_analysis': permutation_results,
        'visualizations': visualization_figures,
        'suppressed_cells': suppressed_summary,
        'analysis_metadata': {
            'group_keys': group_keys,
            'category_cols': category_cols,
            'primary_category': primary_category,
            'min_category_n': MIN_N,
            'dev_n': len(dev_df),
            'test_n': len(test_df),
            'has_cognitive_data': has_cognitive_data,
        },
        'executive_summary': summary
    }