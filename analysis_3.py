import os
from typing import  Dict, Any, Optional, Tuple
from collections import Counter


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression


from scipy.stats import (
    ttest_ind, pearsonr
)


from analysis_0 import *
from analysis_tools import (
    get_analysis_group_keys, get_available_traits, resolve_plot_dir,
)
from profiles.profile_sets import PERSON_ETHNICS
from profiles.schema import PersonSet
from cases.cases_config import CaseConfig

from plot_tools import apply_neurips_figure_style, new_pub_fig, bootstrap_line_ci



def _item_consensus_excluding(df, exclude_col=None, min_strength: Optional[float] = None):
    """
    Returns:
      cons_label: pd.Series of consensus labels (mode; deterministic tie-break)
      cons_strength: pd.Series in [0.5, 1] when there is ≥1 valid vote; NaN otherwise
      ambiguity: 1 - cons_strength
    Consensus computed across all profile* columns except 'exclude_col' if provided.
    """
    profile_cols = [c for c in df.columns if c.startswith("profile")]
    profs = [c for c in profile_cols if c != exclude_col]
    if len(profs) == 0:
        raise ValueError("No profile* columns available for consensus.")

    preds = df[profs].values
    cons_label, cons_strength = [], []

    for row in preds:
        vals = [v for v in row if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) == 0:
            cons_label.append(None)
            cons_strength.append(np.nan)
            continue

        counts = Counter(vals)
        ranked = counts.most_common()
        maxc = ranked[0][1]
        candidates = sorted([lab for lab, c in ranked if c == maxc])
        lab = candidates[0]

        strength = maxc / len(vals)
        if (min_strength is not None) and np.isfinite(min_strength):
            strength = max(min_strength, strength)

        cons_label.append(lab)
        cons_strength.append(strength)

    cons = pd.Series(cons_label, index=df.index, name="cons_label")
    S = pd.Series(cons_strength, index=df.index, name="cons_strength")
    A = (1.0 - S).rename("ambiguity")
    return cons, S, A



def cohens_h(p1: float, p2: float) -> float:
    p1 = np.clip(p1, 0.0, 1.0)
    p2 = np.clip(p2, 0.0, 1.0)
    return 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def bootstrap_cohens_h(p1_samples: np.ndarray,
                       p2_samples: np.ndarray,
                       B: int = 2000,
                       seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI for h from two vectors of per-unit proportions."""
    rng = np.random.default_rng(seed)
    p1_samples = np.asarray(p1_samples, float)
    p2_samples = np.asarray(p2_samples, float)
    n1, n2 = len(p1_samples), len(p2_samples)
    hs = []
    for _ in range(B):
        b1 = p1_samples[rng.integers(0, n1, n1)].mean()
        b2 = p2_samples[rng.integers(0, n2, n2)].mean()
        hs.append(cohens_h(b1, b2))
    hs = np.asarray(hs)
    return float(np.mean(hs)), float(np.percentile(hs, 2.5)), float(np.percentile(hs, 97.5))

def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    iters: int = 10000,
    seed: int = 42,
    early_stop_alpha: Optional[float] = None,
    check_every: int = 1000,
    conf_delta: float = 1e-3,
    early_stop_for_significance: bool = False,
) -> float:
    """
    Two-sided permutation test on difference in means (Monte Carlo).
    Uses Phipson–Smyth small-sample correction: p = (r+1)/(N+1).

    Early stopping (optional):
      - If `early_stop_alpha` is set, every `check_every` iterations we compute
        Hoeffding bounds for the Bernoulli success rate r/N where a 'success'
        is abs(perm_stat) >= abs(obs_stat).
        * Non-significance stop: if lower confidence bound > alpha, return p_hat.
        * Significance stop (optional): if upper bound < alpha, return p_hat.

    Returns
    -------
    float
        Estimated two-sided p-value.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan

    obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    nx = x.size

    cnt = 0
    check_every = max(1, int(check_every))

    for i in range(1, iters + 1):
        perm = rng.permutation(pooled)
        xs = perm[:nx]
        ys = perm[nx:]
        stat = float(np.mean(xs) - np.mean(ys))
        if abs(stat) >= abs(obs):
            cnt += 1

        if (early_stop_alpha is not None) and (i % check_every == 0):
            p_hat = (cnt+1.0)/(i+1.0)

            rad = math.sqrt(math.log(1.0/conf_delta)/(2.0 * i))
            lower = max(0.0, p_hat-rad)
            upper = min(1.0, p_hat+rad)

            if lower > early_stop_alpha:
                return p_hat

            if early_stop_for_significance and (upper < early_stop_alpha):
                return p_hat

    return (cnt + 1.0) / (iters + 1.0)

def benjamini_hochberg(pvals: list, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR control.
    Returns:
      reject : boolean mask (same length as pvals); False for invalid/NaN inputs
      qvals  : BH-adjusted p-values (NaN where input was invalid)
    """
    p = np.asarray(pvals, dtype=float).ravel()
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)


    valid = np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
    q_full = np.full(n, np.nan, dtype=float)
    reject = np.zeros(n, dtype=bool)

    m = int(valid.sum())
    if m == 0:
        return reject, q_full

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]


    denom = np.arange(1, m + 1, dtype=float)
    q = ranked*m/denom
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    valid_idx = np.nonzero(valid)[0]
    q_full[valid_idx[order]] = q


    reject[valid] = q_full[valid] <= alpha
    return reject, q_full


def _cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx-1)*vx + (ny-1)*vy) / max(nx+ny-2, 1))
    if sp == 0:
        return 0.0
    return (x.mean() - y.mean()) / sp

def _bootstrap_ci_stat(x, y, stat_fn, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return (np.nan, np.nan)
    stats = []
    for _ in range(B):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        stats.append(float(stat_fn(xb, yb)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _bh_adjust(pvals):
    _, qvals = benjamini_hochberg(pvals, alpha=0.05)
    return np.where(np.isfinite(qvals), qvals, np.nan)



def boldness_metrics(df, profile_col, baseline_col="base_pred", min_consensus_strength: Optional[float]=None):
    """
    Computes COI (primary), ATI, CAI for one profile.
    COI uses leave-one-out consensus (exclude this profile from the vote).
    """
    cons, S, A = _item_consensus_excluding(df, exclude_col=profile_col, min_strength=min_consensus_strength)
    b = df[baseline_col]
    yhat = df[profile_col]
    F = (yhat != b)

    mask_coi = (b == cons)
    num = (F & mask_coi) * S
    den = S[mask_coi]
    coi = float(num.sum() / (den.sum() + 1e-12))

    q_hi = A.quantile(0.75)
    q_lo = A.quantile(0.25)
    hi = A >= q_hi
    lo = A <= q_lo
    ati = float(F[hi].mean() - F[lo].mean())

    cai = float((F * A).sum() / (A.sum() + 1e-12))

    return {"COI": coi, "ATI": ati, "CAI": cai}


def _safe_pearsonr(a, b):
    """Drop NaNs; return (nan, nan) if <3 points or zero variance."""
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if len(a) < 3 or np.isclose(a.var(ddof=1), 0.0) or np.isclose(b.var(ddof=1), 0.0):
        return np.nan, np.nan
    try:
        return pearsonr(a, b)
    except Exception:
        return np.nan, np.nan
    


def consistency_vs_boldness_analysis(merged_df, 
                                     case: CaseConfig,
                                     n_folds=5,
                                     person_set: PersonSet = PERSON_ETHNICS,
                                     group_keys=("gender", "ethnicity", "age"),
                                     perf_df: Optional[pd.DataFrame] = None,
                                     perm_iters: int = 10000,
                                     perm_early_stop_alpha: Optional[float] = None,
                                     seed: int = 42):
    """
    Consistency vs Boldness Tradeoff Analysis — with token/cost metrics
    
    New in this version:
      - Accepts perf_df with per-profile 'tokens_per_sample' and 'cost_per_sample'
      - Carries tokens/cost into the analysis_df
      - Adds efficiency features (rescue/accuracy per 1k tokens and per $)
    """
    print("=" * 80)
    print("Consistency vs Boldness Tradeoff Analysis")
    print("=" * 80)
    print(f"Group keys for analysis: {group_keys}")
  
  
    def get_valid_profiles(merged_df, person_set):
        all_profile_cols = [col for col in merged_df.columns if col.startswith("profile")]
        valid_profiles = [col for col in all_profile_cols if col in person_set.metadata]
        print(f"Found {len(valid_profiles)} valid profiles out of {len(all_profile_cols)} total profile columns")
        return valid_profiles
    
    profile_cols = get_valid_profiles(merged_df, person_set)
    if len(profile_cols) == 0:
        print("ERROR: No valid profiles found in PersonSet metadata")
        return {'error': 'No valid profiles found'}



    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        perf = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in perf.columns:
            tokens_map = perf["tokens_per_sample"].to_dict()
        if "cost_per_sample" in perf.columns:
            cost_map = perf["cost_per_sample"].to_dict()
        have_tokens = len(tokens_map) > 0
        have_cost   = len(cost_map) > 0
        print(f"Token/cost availability -> tokens:{have_tokens} cost:{have_cost}")
    else:
        have_tokens = have_cost = False


    if 'true_label' not in merged_df.columns:
        raise ValueError("true_label column is required for stratified CV")
    if 'case_family' in merged_df.columns:
        strat_labels = pd.factorize(list(zip(merged_df['true_label'], merged_df['case_family'])))[0]
    else:
        strat_labels = merged_df['true_label'].astype(str)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    consistency_data = {}
    print(f"=== Calculating consistency across {n_folds} stratified folds...")
    
    for profile in profile_cols:
        if profile not in merged_df.columns:
            continue
    

        fold_accuracies = []
        fold_rescue_rates = [] # P(fix given baseline wrong) on the fold
        fold_extra_error_rates = [] # P(harm given baseline right) on the fold
        fold_disagreements = [] # P(profile != baseline) on the fold
        fold_error_magnitudes = [] # your existing signed-imbalance magnitude

        fold_rescued_num = []
        fold_base_errors_den = []
        fold_harmed_num = []
        fold_base_correct_den = []
    
        fold_coi, fold_ati, fold_cai = [], [], [] # boldness metrics per fold
    
        for _, (_, test_idx) in enumerate(skf.split(merged_df, strat_labels)):
            test_data = merged_df.iloc[test_idx]
    
            acc = accuracy_score(test_data['true_label'], test_data[profile])
            fold_accuracies.append(acc)
    
            base_correct    = (test_data['base_pred'] == test_data['true_label'])
            profile_correct = (test_data[profile]   == test_data['true_label'])
    
            base_errors_cnt   = (~base_correct).sum()
            base_correct_cnt  = ( base_correct).sum()
    
            rescued_cnt       = ((~base_correct) & profile_correct).sum()
            harmed_cnt        = ( base_correct  & (~profile_correct)).sum()
    
            rescue_rate_fold      = (rescued_cnt / base_errors_cnt)  if base_errors_cnt  > 0 else np.nan
            extra_error_rate_fold = (harmed_cnt  / base_correct_cnt) if base_correct_cnt > 0 else np.nan

            fold_rescue_rates.append(float(rescue_rate_fold))
            fold_extra_error_rates.append(float(extra_error_rate_fold))
            fold_rescued_num.append(int(rescued_cnt))
            fold_base_errors_den.append(int(base_errors_cnt))
            fold_harmed_num.append(int(harmed_cnt))
            fold_base_correct_den.append(int(base_correct_cnt))
    
            fold_disagreements.append(float((test_data[profile] != test_data['base_pred']).mean()))
    
            to_positive = ((test_data['base_pred'] == "no")  & (test_data[profile] == "yes")).sum()
            to_negative = ((test_data['base_pred'] == "yes") & (test_data[profile] == "no")).sum()
            error_magnitude = abs(to_positive - to_negative) / len(test_data)
            fold_error_magnitudes.append(float(error_magnitude))
    
            bm_fold = boldness_metrics(test_data, profile, baseline_col="base_pred")
            fold_coi.append(float(bm_fold["COI"]))
            fold_ati.append(float(bm_fold["ATI"]))
            fold_cai.append(float(bm_fold["CAI"]))
    
        acc_mean = float(np.mean(fold_accuracies))
        acc_std  = float(np.std(fold_accuracies))
        rr_unweighted_mean  = float(np.nanmean(fold_rescue_rates))
        rr_std  = float(np.std(fold_rescue_rates))
        eer_unweighted_mean = float(np.nanmean(fold_extra_error_rates))
        eer_std  = float(np.std(fold_extra_error_rates))

        rescued_total = int(np.sum(fold_rescued_num))
        base_err_total = int(np.sum(fold_base_errors_den))
        harmed_total = int(np.sum(fold_harmed_num))
        base_ok_total = int(np.sum(fold_base_correct_den))

        rr_mean  = float(rescued_total/base_err_total) if base_err_total > 0 else np.nan
        eer_mean = float(harmed_total/base_ok_total)   if base_ok_total  > 0 else np.nan

        accuracy_cv = float(acc_std/acc_mean) if acc_mean > 0 else np.nan
        rescue_rate_cv = float(rr_std/rr_mean) if np.isfinite(rr_mean) and rr_mean > 0 else np.nan

        dis_mean = float(np.mean(fold_disagreements))
        dis_std  = float(np.std(fold_disagreements))
        em_mean = float(np.mean(fold_error_magnitudes))
        em_std  = float(np.std(fold_error_magnitudes))
        coi_mean = float(np.mean(fold_coi));  coi_std = float(np.std(fold_coi))
        ati_mean = float(np.mean(fold_ati));  ati_std = float(np.std(fold_ati))
        cai_mean = float(np.mean(fold_cai));  cai_std = float(np.std(fold_cai))

        consistency_data[profile] = {
            'accuracy_mean': acc_mean,
            'accuracy_std':  acc_std,
            'accuracy_cv':   accuracy_cv,
            'rescue_rate_mean': rr_mean,          
            'rescue_rate_std':  rr_std,
            'rescue_rate_mean_unweighted': rr_unweighted_mean,
            'rescue_rate_cv':   rescue_rate_cv,
            'fold_rescue_rates': fold_rescue_rates,
            'extra_error_rate_mean': eer_mean,
            'extra_error_rate_std':  eer_std,
            'extra_error_rate_mean_unweighted': eer_unweighted_mean,
            'fold_extra_error_rates': fold_extra_error_rates,
            'rescued_total': rescued_total,
            'base_errors_total': base_err_total,
            'harmed_total': harmed_total,
            'base_correct_total': base_ok_total,
            'disagreement_rate_mean': dis_mean,
            'disagreement_rate_std':  dis_std,
            'fold_disagreements': fold_disagreements,
            'error_magnitude_mean': em_mean,
            'error_magnitude_std':  em_std,
            'fold_error_magnitudes': fold_error_magnitudes,
            'coi_mean': coi_mean, 
            'coi_std': coi_std, 
            'fold_coi': fold_coi,
            'ati_mean': ati_mean, 
            'ati_std': ati_std, 
            'fold_ati': fold_ati,
            'cai_mean': cai_mean, 
            'cai_std': cai_std, 
            'fold_cai': fold_cai,
            'fold_accuracies': fold_accuracies,
            'tokens_per_sample': float(tokens_map.get(profile, np.nan)) if have_tokens else np.nan,
            'cost_per_sample':   float(cost_map.get(profile, np.nan))   if have_cost   else np.nan,
        }
    

    print("=== Calculating boldness metrics (COI/ATI/CAI) and global rescue/extra-error...")

    boldness_data = {}
    for profile in profile_cols:
        cd = consistency_data.get(profile, {})
        if cd:
            coi = cd.get('coi_mean', np.nan)
            ati = cd.get('ati_mean', np.nan)
            cai = cd.get('cai_mean', np.nan)
            rescue_rate      = cd.get('rescue_rate_mean', np.nan)
            extra_error_rate = cd.get('extra_error_rate_mean', np.nan)
            disagreement_rate= cd.get('disagreement_rate_mean', np.nan)
        else:
            base_ok_full = (merged_df['base_pred'] == merged_df['true_label'])
            p_right = float(base_ok_full.mean()); p_wrong = 1.0 - p_right
            prof_ok = (merged_df[profile] == merged_df['true_label'])
            rescue_rate      = ((~base_ok_full) & prof_ok).mean() / (p_wrong + 1e-12)
            extra_error_rate = ( base_ok_full & (~prof_ok)).mean() / (p_right + 1e-12)
            disagreement_rate= (merged_df[profile] != merged_df['base_pred']).mean()
            bm = boldness_metrics(merged_df, profile, baseline_col="base_pred")
            coi, ati, cai = float(bm["COI"]), float(bm["ATI"]), float(bm["CAI"])
    
        toks = float(cd.get('tokens_per_sample', np.nan))
        cost = float(cd.get('cost_per_sample',   np.nan))
        tok_k = toks/1000.0 if np.isfinite(toks) and toks > 0 else np.nan
        cost_d= cost        if np.isfinite(cost) and cost > 0 else np.nan
    
        boldness_data[profile] = {
            'boldness_score': float(coi),
            'COI': float(coi), 'ATI': float(ati), 'CAI': float(cai),
            'rescue_rate': float(rescue_rate),
            'extra_error_rate': float(extra_error_rate),
            'disagreement_rate': float(disagreement_rate),
            'rescue_per_1k_tokens':    (rescue_rate / tok_k) if np.isfinite(tok_k) else np.nan,
            'extra_error_per_1k_tokens': (extra_error_rate / tok_k) if np.isfinite(tok_k) else np.nan,
            'rescue_per_dollar':       (rescue_rate / cost_d) if np.isfinite(cost_d) else np.nan,
        }


    print("=== Analyzing consistency-boldness correlations by demographics...")
    analysis_data = []
    for profile in profile_cols:
        if profile in consistency_data and profile in boldness_data:
            traits = person_set.get_traits(profile, group_keys)
            toks = consistency_data[profile].get('tokens_per_sample', np.nan)
            cost = consistency_data[profile].get('cost_per_sample', np.nan)
            row = {
                'profile': profile,
                'volatility': consistency_data[profile]['accuracy_std'],
                'consistency': 1 / (1 + consistency_data[profile]['accuracy_std']),
                'boldness_score': boldness_data[profile]['boldness_score'],
                'COI': boldness_data[profile]['COI'],
                'ATI': boldness_data[profile]['ATI'],
                'CAI': boldness_data[profile]['CAI'],
                'rescue_rate': boldness_data[profile]['rescue_rate'],
                'extra_error_rate': boldness_data[profile]['extra_error_rate'],
                'accuracy_mean': consistency_data[profile]['accuracy_mean'],
                'disagreement_rate': boldness_data[profile]['disagreement_rate'],
                'tokens_per_sample': consistency_data[profile].get('tokens_per_sample', np.nan),
                'cost_per_sample':   consistency_data[profile].get('cost_per_sample',   np.nan),
                'rescue_per_1k_tokens': boldness_data[profile]['rescue_per_1k_tokens'],
                'extra_error_per_1k_tokens': boldness_data[profile]['extra_error_per_1k_tokens'],
                'rescue_per_dollar': boldness_data[profile]['rescue_per_dollar'],
            }
            

            for trait_name in group_keys:
                row[trait_name] = traits.get(trait_name, "Unknown")
            analysis_data.append(row)
    analysis_df = pd.DataFrame(analysis_data)
    if len(analysis_df) == 0:
        print("ERROR: No data available for correlation analysis")
        return {'error': 'No analysis data available'}
    print(f"Analysis dataset shape: {analysis_df.shape}")

    correlations = {}
    demographic_correlations = {}

    if len(analysis_df) >= 3:
        corr_vol_bold, p_val_vol_bold = _safe_pearsonr(analysis_df['volatility'], analysis_df['boldness_score'])
        correlations['volatility_vs_boldness'] = {'correlation': corr_vol_bold, 'p_value': p_val_vol_bold, 'significant': p_val_vol_bold < 0.05,
                                                  'interpretation': 'Positive correlation means less consistent profiles are bolder'}
        corr_cons_bold, p_val_cons_bold = _safe_pearsonr(analysis_df['consistency'], analysis_df['boldness_score'])
        correlations['consistency_vs_boldness'] = {'correlation': corr_cons_bold, 'p_value': p_val_cons_bold, 'significant': p_val_cons_bold < 0.05,
                                                   'interpretation': 'Negative correlation means more consistent profiles are less bold'}
        corr_vol_rescue, p_val_vol_rescue = _safe_pearsonr(analysis_df['volatility'], analysis_df['rescue_rate'])
        correlations['volatility_vs_rescue'] = {'correlation': corr_vol_rescue, 'p_value': p_val_vol_rescue, 'significant': p_val_vol_rescue < 0.05}

        corr_bold_acc, p_val_bold_acc = _safe_pearsonr(analysis_df['boldness_score'], analysis_df['accuracy_mean'])
        correlations['boldness_vs_accuracy'] = {'correlation': corr_bold_acc, 'p_value': p_val_bold_acc, 'significant': p_val_bold_acc < 0.05}

        corr_bold_rescue, p_bold_rescue = _safe_pearsonr(analysis_df['boldness_score'], analysis_df['rescue_rate'])
        correlations['boldness_vs_rescue'] = {
            'correlation': corr_bold_rescue, 'p_value': p_bold_rescue,
            'significant': p_bold_rescue < 0.05
        }

        if 'extra_error_rate' in analysis_df.columns:
            corr_bold_extra, p_bold_extra = _safe_pearsonr(analysis_df['boldness_score'], analysis_df['extra_error_rate'])
            correlations['boldness_vs_extra_error'] = {
                'correlation': corr_bold_extra, 'p_value': p_bold_extra,
                'significant': p_bold_extra < 0.05
            }


        if analysis_df['tokens_per_sample'].notna().sum() >= 3:
            m = analysis_df[['tokens_per_sample', 'rescue_rate']].dropna()
            if len(m) >= 3:
                r_tok_rescue, p_tok_rescue = _safe_pearsonr(m['tokens_per_sample'], m['rescue_rate'])
                correlations['tokens_vs_rescue'] = {'correlation': r_tok_rescue, 'p_value': p_tok_rescue, 'significant': p_tok_rescue < 0.05}
            m = analysis_df[['tokens_per_sample', 'accuracy_mean']].dropna()
            if len(m) >= 3:
                r_tok_acc, p_tok_acc = _safe_pearsonr(m['tokens_per_sample'], m['accuracy_mean'])
                correlations['tokens_vs_accuracy'] = {'correlation': r_tok_acc, 'p_value': p_tok_acc, 'significant': p_tok_acc < 0.05}

        if analysis_df['cost_per_sample'].notna().sum() >= 3:
            m = analysis_df[['cost_per_sample', 'rescue_rate']].dropna()
            if len(m) >= 3:
                r_cost_rescue, p_cost_rescue = _safe_pearsonr(m['cost_per_sample'], m['rescue_rate'])
                correlations['cost_vs_rescue'] = {'correlation': r_cost_rescue, 'p_value': p_cost_rescue, 'significant': p_cost_rescue < 0.05}
            m = analysis_df[['cost_per_sample', 'accuracy_mean']].dropna()
            if len(m) >= 3:
                r_cost_acc, p_cost_acc = _safe_pearsonr(m['cost_per_sample'], m['accuracy_mean'])
                correlations['cost_vs_accuracy'] = {'correlation': r_cost_acc, 'p_value': p_cost_acc, 'significant': p_cost_acc < 0.05}

        p_items = [(k, v['p_value']) for k, v in correlations.items() if np.isfinite(v['p_value'])]
        if p_items:
            keys, pvals = zip(*p_items)
            qvals = _bh_adjust(pvals)
            for k, q in zip(keys, qvals):
                correlations[k]['q_value'] = float(q)
                correlations[k]['significant_fdr_5pct'] = (q < 0.05)

        print(f"\n--- Overall Correlation Results:")

        qvb = correlations['volatility_vs_boldness'].get('q_value', np.nan)
        print(f"  Volatility vs Boldness: r={corr_vol_bold:.3f}, p={p_val_vol_bold:.4f}, q={qvb:.4f}"
              f" {'***' if np.isfinite(qvb) and qvb < 0.05 else ''}")
        
        print(f"  Consistency vs Boldness: r={corr_cons_bold:.3f}, p={p_val_cons_bold:.4f} {'***' if p_val_cons_bold < 0.05 else ''}")
        print(f"  Volatility vs Rescue Rate: r={corr_vol_rescue:.3f}, p={p_val_vol_rescue:.4f} {'***' if p_val_vol_rescue < 0.05 else ''}")
        print(f"  Boldness vs Accuracy: r={corr_bold_acc:.3f}, p={p_val_bold_acc:.4f} {'***' if p_val_bold_acc < 0.05 else ''}")
        if 'tokens_vs_rescue' in correlations:
            print(f"  Tokens vs Rescue: r={correlations['tokens_vs_rescue']['correlation']:.3f}, p={correlations['tokens_vs_rescue']['p_value']:.4f} {'***' if correlations['tokens_vs_rescue']['p_value'] < 0.05 else ''}")
        if 'cost_vs_rescue' in correlations:
            print(f"  Cost vs Rescue: r={correlations['cost_vs_rescue']['correlation']:.3f}, p={correlations['cost_vs_rescue']['p_value']:.4f} {'***' if correlations['cost_vs_rescue']['p_value'] < 0.05 else ''}")


        for trait_name in group_keys:
            if trait_name not in analysis_df.columns: 
                continue
            valid_values = [v for v in analysis_df[trait_name].dropna().unique() if v != "Unknown"]
            if not valid_values: 
                continue
            print(f"\n--- Correlations by {trait_name.upper()}:")
            demographic_correlations[trait_name] = {}
    
            p_accum_keys, p_accum = [], []
    
            for value in valid_values:
                subset = analysis_df[analysis_df[trait_name] == value]
                if len(subset) >= 3:
                    try:
                        corr_vol_bold, p_val = _safe_pearsonr(subset['volatility'], subset['boldness_score'])
                        demographic_correlations[trait_name][str(value)] = {
                            'volatility_vs_boldness': {
                                'correlation': corr_vol_bold, 'p_value': p_val, 'n': len(subset),
                                'significant': p_val < 0.05
                            }
                        }
                        if np.isfinite(p_val):
                            p_accum_keys.append(str(value)); p_accum.append(p_val)
                        print(f"  {value} (n={len(subset)}): Volatility vs Boldness r={corr_vol_bold:.3f}, p={p_val:.4f} {'***' if p_val < 0.05 else ''}")
                    except Exception:
                        print(f"  {value} (n={len(subset)}): Could not calculate correlation")
                else:
                    print(f"  {value} (n={len(subset)}): Insufficient data for correlation")
    
            if p_accum:
                qvals = _bh_adjust(p_accum)
                for val, q in zip(p_accum_keys, qvals):
                    demographic_correlations[trait_name][val]['volatility_vs_boldness']['q_value'] = float(q)
                    demographic_correlations[trait_name][val]['volatility_vs_boldness']['significant_fdr_5pct'] = (q < 0.05)
    


    print(f"\n=== Profile Classification by Demographics:")
    profile_archetypes = {}
    volatility_median = analysis_df['volatility'].median()
    boldness_median = analysis_df['boldness_score'].median()
    archetype_by_demographics = {t:{} for t in group_keys if t in analysis_df.columns}

    for _, row in analysis_df.iterrows():
        profile = row['profile']
        vol = row['volatility']
        bold = row['boldness_score']
        rescue = row['rescue_rate']
        acc = row['accuracy_mean']

        if vol > volatility_median and bold > boldness_median:
            archetype, description = "Inconsistent Bold", "High volatility, high boldness"
        elif vol < volatility_median and bold > boldness_median:
            archetype, description = "Consistent Bold", "Low volatility, high boldness"
        elif vol > volatility_median and bold < boldness_median:
            archetype, description = "Inconsistent Cautious", "High volatility, low boldness"
        else:
            archetype, description = "Consistent Cautious", "Low volatility, low boldness"


        profile_archetypes[profile] = {'archetype': archetype, 'description': description, 'volatility': vol, 'boldness': bold, 'rescue_rate': rescue, 'accuracy': acc}
        for trait_name in group_keys:
            if trait_name in row and row[trait_name] != "Unknown":
                trait_val = str(row[trait_name])
                archetype_by_demographics.setdefault(trait_name, {}).setdefault(trait_val, {}).setdefault(archetype, []).append(profile)
        traits = {k: row[k] for k in group_keys if k in row and row[k] != "Unknown"}
        trait_str = ", ".join(f"{k}={v}" for k, v in traits.items())
        print(f"  {profile} ({trait_str}): {archetype} – {description}")

    print(f"\n=== Demographic Error Detection in Archetypes:")
    archetype_error_analysis = {}
    for trait_name, groups in archetype_by_demographics.items():
        print(f"\n--- {trait_name.upper()} Error Analysis:")
        archetype_error_analysis[trait_name] = {}
        for trait_val, archetypes in groups.items():
            total_profiles = sum(len(profiles) for profiles in archetypes.values())
            print(f"  {trait_val} (n={total_profiles}):")
            archetype_error_analysis[trait_name][trait_val] = {}
            for archetype, profiles_list in archetypes.items():
                percentage = (len(profiles_list) / total_profiles) * 100 if total_profiles > 0 else 0
                archetype_error_analysis[trait_name][trait_val][archetype] = {'count': len(profiles_list), 'percentage': percentage, 'profiles': profiles_list}
                print(f"    {archetype}: {len(profiles_list)} ({percentage:.1f}%)")


    print(f"\n=== Normative Value Assessment by Demographics:")
    normative_assessment = {}
    

    high_volatility_profiles = analysis_df[analysis_df['volatility'] > volatility_median]
    low_volatility_profiles  = analysis_df[analysis_df['volatility'] <= volatility_median]
    
    def _rate_compare(x_vals, y_vals, B=2000, seed=42,
                      perm_iters: int = 10000,
                      perm_early_stop_alpha: Optional[float] = None):
        """Compare rescue rates between two groups with robust stats."""
        x = pd.Series(x_vals, dtype=float).dropna().to_numpy()
        y = pd.Series(y_vals, dtype=float).dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            return None
    
        p1 = float(np.mean(x)); p2 = float(np.mean(y))
        h_point = cohens_h(p1, p2)
        h_boot_mean, h_lo, h_hi = bootstrap_cohens_h(x, y, B=B, seed=seed)
        p_perm = permutation_pvalue(x, y, iters=perm_iters, seed=seed,
                                    early_stop_alpha=perm_early_stop_alpha)
    
        if len(x) >= 2 and len(y) >= 2:
            t_stat, p_val_welch = ttest_ind(x, y, equal_var=False)
            d_point = _cohens_d(x, y)
            d_lo, d_hi = _bootstrap_ci_stat(x, y, _cohens_d, B=B, seed=seed)
        else:
            t_stat = p_val_welch = d_point = d_lo = d_hi = np.nan
    
        return {
            'p1': p1, 'p2': p2,
            'h_point': h_point, 'h_boot_mean': h_boot_mean, 'h_lo': h_lo, 'h_hi': h_hi,
            'p_perm': p_perm, 't_stat': t_stat, 'p_val_welch': p_val_welch,
            'd_point': d_point, 'd_lo': d_lo, 'd_hi': d_hi,
            'n1': len(x), 'n2': len(y),
        }
    

    if len(high_volatility_profiles) > 0 and len(low_volatility_profiles) > 0:
        res = _rate_compare(
            high_volatility_profiles['rescue_rate'],
            low_volatility_profiles['rescue_rate'],
            B=2000, seed=seed,
            perm_iters=perm_iters,
            perm_early_stop_alpha=perm_early_stop_alpha,
        )

        if res is not None:
            normative_assessment['overall'] = {
                'high_volatility_rescue': res['p1'],
                'low_volatility_rescue':  res['p2'],
                'difference': res['p1'] - res['p2'],
                'effect_size_h': {'h': float(res['h_point']),
                                  'ci95': [float(res['h_lo']), float(res['h_hi'])],
                                  'boot_mean': float(res['h_boot_mean'])},
                'test_permutation': {'p_value': float(res['p_perm'])},
                'test_welch': {'t_stat': float(res['t_stat']), 'p_value': float(res['p_val_welch'])},
                'effect_size_d': {'d': float(res['d_point']),
                                  'ci95': [float(res['d_lo']), float(res['d_hi'])]}
            }
            print("  OVERALL:")
            print(f"    Inconsistent Profiles Rescue Rate: {res['p1']:.3f} (n={res['n1']})")
            print(f"    Consistent Profiles Rescue Rate:   {res['p2']:.3f} (n={res['n2']})")
            print(f"    Δ = {res['p1'] - res['p2']:.3f},  h = {res['h_point']:.3f} "
                  f"[CI {res['h_lo']:.3f},{res['h_hi']:.3f}],  p_perm={res['p_perm']:.4f},  "
                  f"Welch p={res['p_val_welch']:.4f}, d={res['d_point']:.3f} "
                  f"[{res['d_lo']:.3f},{res['d_hi']:.3f}]")
    
    
    for trait_name in group_keys:
        if trait_name not in analysis_df.columns:
            continue
        valid_values = [v for v in analysis_df[trait_name].dropna().unique() if v != "Unknown"]
        if not valid_values:
            continue
    
        print(f"\n  BY {trait_name.upper()}:")
        normative_assessment[trait_name] = {}
    

        p_accum_labels, p_accum = [], []
    
        for value in valid_values:
            subset = analysis_df[analysis_df[trait_name] == value]
            if len(subset) < 4:
                print(f"    {value}: Insufficient data (n={len(subset)})")
                continue
    
            med = subset['volatility'].median()
            sub_hi = subset[subset['volatility'] >  med]['rescue_rate']
            sub_lo = subset[subset['volatility'] <= med]['rescue_rate']

            if len(sub_hi) == 0 or len(sub_lo) == 0:
                print(f"    {value}: Not enough profiles on one side of the median")
                continue
    
            res = _rate_compare(sub_hi, sub_lo,
                                B=2000, seed=seed,
                                perm_iters=perm_iters,
                                perm_early_stop_alpha=perm_early_stop_alpha)
            if res is None:
                print(f"    {value}: Not enough valid rescue rate values")
                continue
    
            normative_assessment[trait_name][str(value)] = {
                'high_volatility_rescue': res['p1'],
                'low_volatility_rescue':  res['p2'],
                'difference': res['p1'] - res['p2'],
                'effect_size_h': {'h': float(res['h_point']),
                                  'ci95': [float(res['h_lo']), float(res['h_hi'])],
                                  'boot_mean': float(res['h_boot_mean'])},
                'test_permutation': {'p_value': float(res['p_perm'])},
                'test_welch': {'t_stat': float(res['t_stat']), 'p_value': float(res['p_val_welch'])},
                'effect_size_d': {'d': float(res['d_point']),
                                  'ci95': [float(res['d_lo']), float(res['d_hi'])]}
            }
    
            print(f"    {value}: Inconsistent={res['p1']:.3f} vs Consistent={res['p2']:.3f}, "
                  f"Δ={res['p1'] - res['p2']:.3f}, h={res['h_point']:.3f} "
                  f"[CI {res['h_lo']:.3f},{res['h_hi']:.3f}], "
                  f"p_perm={res['p_perm']:.4f},  Welch p={res['p_val_welch']:.4f}, "
                  f"d={res['d_point']:.3f} [{res['d_lo']:.3f},{res['d_hi']:.3f}]")
    
            p_accum_labels.append(f"{trait_name}:{value}")
            p_accum.append(res['p_perm'])
    
        if p_accum:
            reject, qvals = benjamini_hochberg(p_accum, alpha=0.05)
            print("\n  FDR (Benjamini–Hochberg) over subgroup tests:")
            for lab, pv, q, r in zip(p_accum_labels, p_accum, qvals, reject):
                print(f"    {lab:25s} p={pv:.4f}  q={q:.4f}  {'REJECT' if r else '—'}")
            for (lab, q) in zip(p_accum_labels, qvals):
                _, val = lab.split(":", 1)
                if val in normative_assessment[trait_name]:
                    normative_assessment[trait_name][val].setdefault('test_permutation', {})
                    normative_assessment[trait_name][val]['test_permutation']['q_value'] = float(q)



    return {
        'consistency_data': consistency_data,
        'boldness_data': boldness_data,
        'analysis_df': analysis_df,
        'correlations': correlations,
        'demographic_correlations': demographic_correlations,
        'profile_archetypes': profile_archetypes,
        'archetype_error_analysis': archetype_error_analysis,
        'normative_assessment': normative_assessment,
        'group_keys': group_keys,
        'valid_profiles': profile_cols
    }



def plot_consistency_boldness_analysis(consistency_results, 
                                       person_set: PersonSet = None,
                                       group_keys=("gender", "ethnicity", "age"),
                                       figsize=(6.5, 4.2)):
    apply_neurips_figure_style()

    consistency_data = consistency_results['consistency_data']
    boldness_data    = consistency_results['boldness_data']
    archetypes       = consistency_results['profile_archetypes']
    analysis_df      = consistency_results.get('analysis_df', pd.DataFrame())

    profiles = list(archetypes.keys())
    volatilities = np.array([archetypes[p]['volatility'] for p in profiles], dtype=float)
    boldness_scores = np.array([archetypes[p]['boldness'] for p in profiles], dtype=float) 
    rescue_rates = np.array([archetypes[p]['rescue_rate'] for p in profiles], dtype=float)
    accuracies = np.array([archetypes[p]['accuracy'] for p in profiles], dtype=float)

    def _sizes(ps):
        out = []

        for p in ps:
            toks = consistency_data.get(p, {}).get('tokens_per_sample', np.nan)
            if np.isfinite(toks) and toks > 0:
                s = 60 + 6.0*np.sqrt(toks)  
            else:
                s = 100
            out.append(s)
        arr = np.clip(np.array(out, float), 40, 600)

        return arr
    
    sizes = _sizes(profiles)

    archetype_colors = {
        'Inconsistent Bold':    '#d62728',
        'Consistent Bold':      '#2ca02c',
        'Inconsistent Cautious':'#ff7f0e',
        'Consistent Cautious':  '#1f77b4'
    }

    colors_by_arch = [archetype_colors.get(archetypes[p]['archetype'], '#8c564b') for p in profiles]

    figs = {}
    figA, ax = new_pub_fig('Consistency vs Boldness (COI) — size ∝ tokens/sample', figsize)

    if person_set is not None and isinstance(analysis_df, pd.DataFrame) and len(analysis_df) > 0:
        x = analysis_df['volatility'].to_numpy(float)
        y = analysis_df['boldness_score'].to_numpy(float)
        ati = analysis_df['ATI'].to_numpy(float) if 'ATI' in analysis_df.columns else np.zeros_like(x)
        ps = analysis_df['profile'].tolist()
        sz = _sizes(ps)

        sc = ax.scatter(x, y, c=ati, s=sz, alpha=0.8, edgecolors='black', linewidth=0.6)
        cbar = figA.colorbar(sc, ax=ax, shrink=0.88)
        cbar.set_label('ATI (ambiguity targeting, high -> more on hard items)')
    else:
        sc = ax.scatter(volatilities, boldness_scores, c=colors_by_arch, s=sizes, alpha=0.8,
                        edgecolors='black', linewidth=0.6)

    try:
        xfit = analysis_df['volatility'].to_numpy(float) if len(analysis_df) > 0 else volatilities
        yfit = analysis_df['boldness_score'].to_numpy(float) if len(analysis_df) > 0 else boldness_scores
        if np.unique(xfit[~np.isnan(xfit)]).size >= 6:
            gx, mid, lo, hi = bootstrap_line_ci(xfit, yfit)
            ax.plot(gx, mid, linestyle='--')
            ax.fill_between(gx, lo, hi, alpha=0.15)
    except Exception:
        pass


    if len(boldness_scores) > 0:
        k = min(6, len(profiles))
        top_bold_idx = np.argsort(-(boldness_scores))[:k]
        for i in top_bold_idx:
            ax.annotate(profiles[i].replace('profile','P'),
                        (volatilities[i], boldness_scores[i]), xytext=(5,5),
                        textcoords='offset points', fontsize=8)
    ax.set_xlabel('Volatility (accuracy std; higher = less consistent)')
    ax.set_ylabel('Boldness (COI)')
    ax.grid(True, alpha=0.3)
    figs['consistency_vs_boldness'] = figA


    figB, ax = new_pub_fig('Consistency vs Rescue Rate (size ∝ tokens/sample)', figsize)
    ax.scatter(volatilities, rescue_rates, c=colors_by_arch, s=sizes, alpha=0.8,
               edgecolors='black', linewidth=0.6)
    try:
        if np.unique(volatilities[~np.isnan(volatilities)]).size >= 6:
            gx, mid, lo, hi = bootstrap_line_ci(volatilities, rescue_rates)
            ax.plot(gx, mid, linestyle='--'); ax.fill_between(gx, lo, hi, alpha=0.15)
    except Exception:
        pass
    ax.set_xlabel('Volatility (accuracy std; higher = less consistent)')
    ax.set_ylabel('Rescue Rate  P(fix given baseline wrong)')
    ax.grid(True, alpha=0.3)
    figs['volatility_vs_rescue'] = figB


    if isinstance(analysis_df, pd.DataFrame) and 'extra_error_rate' in analysis_df.columns:
        figFR, ax = new_pub_fig('Benefit–Risk Frontier: Rescue vs Extra Error', figsize)
        x = analysis_df['extra_error_rate'].to_numpy(float)
        y = analysis_df['rescue_rate'].to_numpy(float)
        ps = analysis_df['profile'].tolist()
        sz = _sizes(ps)
        ax.scatter(x, y, s=sz, alpha=0.8, edgecolors='black', linewidth=0.6)

        lim = max(float(np.nanmax(x)), float(np.nanmax(y))) if len(x) else 0.2
        ax.plot([0, lim], [0, lim], linestyle='--', alpha=0.6)
        ax.set_xlabel('Extra Error  P(harm given baseline right)')
        ax.set_ylabel('Rescue       P(fix given baseline wrong)')
        ax.grid(True, alpha=0.3)
        figs['benefit_risk_frontier'] = figFR




    title_c = f"Archetypes by {group_keys[0].capitalize()}" if group_keys else "Archetypes by Group"
    figC, ax = new_pub_fig(title_c, figsize)
    if person_set is not None and len(analysis_df) > 0 and group_keys and group_keys[0] in analysis_df.columns:
        primary_trait = group_keys[0]
        trait_values = [v for v in analysis_df[primary_trait].unique() if v != "Unknown"]
        archetype_by_trait = {}
        for trait_val in trait_values:
            archetype_by_trait[trait_val] = {}
            subset = analysis_df[analysis_df[primary_trait] == trait_val]
            for _, row in subset.iterrows():
                profile = row['profile']
                if profile in archetypes:
                    arch = archetypes[profile]['archetype']
                    archetype_by_trait[trait_val][arch] = archetype_by_trait[trait_val].get(arch, 0) + 1
        archetypes_list = list(archetype_colors.keys())
        x = np.arange(len(trait_values)); bottom = np.zeros(len(trait_values))
        for arch in archetypes_list:
            counts = [archetype_by_trait[trait_val].get(arch, 0) for trait_val in trait_values]
            ax.bar(x, counts, bottom=bottom, color=archetype_colors[arch], label=arch, alpha=0.85)
            bottom += counts
        ax.set_xticks(x); ax.set_xticklabels(trait_values, rotation=45, ha='right')
        ax.set_xlabel(primary_trait.capitalize()); ax.set_ylabel('Number of Profiles')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=False)
    else:
        archetype_counts = {}
        for profile_data in archetypes.values():
            arch = profile_data['archetype']
            archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
        names = list(archetype_counts.keys())
        counts = [archetype_counts[k] for k in names]
        x = np.arange(len(names))
        ax.bar(x, counts, color=[archetype_colors.get(n, '#8c564b') for n in names], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('Number of Profiles')
    ax.grid(True, axis='y', alpha=0.3)
    figs['archetype_by_demo'] = figC




    figD, ax = new_pub_fig('Consistency Across CV Folds', figsize)
    representative_profiles = []
    for arch in archetype_colors.keys():
        for profile, data in archetypes.items():
            if data['archetype'] == arch and profile not in representative_profiles:
                representative_profiles.append(profile); break
    representative_profiles = representative_profiles[:4]

    if consistency_data:
        def _infer_nfolds(p):
            cd = consistency_data.get(p, {})
            for key in ('fold_accuracies', 'fold_coi', 'fold_rescue_rates', 'fold_extra_error_rates'):
                seq = cd.get(key, [])
                if isinstance(seq, (list, tuple)) and len(seq) > 0:
                    return len(seq)
            return 0

        nfolds = 0
        for p in profiles:
            nfolds = _infer_nfolds(p)
            if nfolds > 0:
                break

        if nfolds > 0:
            fold_numbers = list(range(1, nfolds + 1))
            for profile in representative_profiles:
                cd = consistency_data.get(profile, {})
                seq = cd.get('fold_accuracies') or cd.get('fold_coi') or cd.get('fold_rescue_rates') or []
                if seq:
                    arch = archetypes[profile]['archetype']
                    color = archetype_colors.get(arch, '#8c564b')
                    if person_set:
                        traits = person_set.get_traits(profile, group_keys)
                        demo_info = "_".join(str(traits.get(k, '?'))[:3] for k in group_keys[:2] if traits.get(k) != "Unknown")
                        legend_label = f"{demo_info} ({arch})"
                    else:
                        legend_label = f"{profile.replace('profile', 'P')} ({arch})"
                    ax.plot(fold_numbers, seq, 'o-', alpha=0.85, label=legend_label, color=color)
            ax.set_xlabel('Cross-Validation Fold'); ax.set_ylabel('Accuracy (or proxy)')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=False)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No per-fold data available', ha='center', va='center', transform=ax.transAxes)
    figs['cv_trends'] = figD



    figE, ax = new_pub_fig('Boldness (COI) vs Accuracy', figsize)
    ax.scatter(boldness_scores, accuracies, c=colors_by_arch, s=sizes, alpha=0.8,
               edgecolors='black', linewidth=0.6)
    try:
        if np.unique(boldness_scores[~np.isnan(boldness_scores)]).size >= 6:
            gx, mid, lo, hi = bootstrap_line_ci(boldness_scores, accuracies)
            ax.plot(gx, mid, linestyle='--'); ax.fill_between(gx, lo, hi, alpha=0.15)
    except Exception:
        pass
    ax.set_xlabel('Boldness (COI)'); ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    figs['boldness_vs_accuracy'] = figE


    return figs


def _shapley_r2_two_groups(y, X_demo, X_other):
    """
    Shapley (LMG) R^2 for two predictor groups.
    Returns: r2_demo, r2_other, r2_full, shap_demo, shap_other, shared
    """
    n = len(y)
    def _r2(X):
        if X.shape[1] == 0: 
            return 0.0
        return float(LinearRegression().fit(X, y).score(X, y))

    r2_d = _r2(X_demo)
    r2_o = _r2(X_other)
    X_full = X_demo if X_other.shape[1] == 0 else (X_other if X_demo.shape[1] == 0 else np.column_stack([X_demo, X_other]))
    r2_full = _r2(X_full)

    # Shapley test for 2 players: 0.5 * [R2(A) + (R2(A+B)-R2(B))]
    shap_demo  = (r2_d+(r2_full-r2_o))/2
    shap_other = (r2_o+(r2_full-r2_d))/2

    shap_demo  = float(np.clip(shap_demo,  0.0, r2_full))
    shap_other = float(np.clip(shap_other, 0.0, r2_full - shap_demo))
    shared = max(0.0, r2_full - shap_demo - shap_other)
    return r2_d, r2_o, r2_full, shap_demo, shap_other, shared


def _bootstrap_shapley_two_groups(y, X_demo, X_other, B=1000, seed=42):
    """
    Bootstrap Shapley/LMG for two predictor groups by resampling profiles.
    Returns dict with arrays and 95% CIs for shap_demo, shap_other, r2_full, and dominance ratio.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    vals_demo, vals_other, vals_full, dom = [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        Xdb = X_demo[idx] if X_demo.shape[1] else X_demo
        Xob = X_other[idx] if X_other.shape[1] else X_other
        r2_d, r2_o, r2_f, s_d, s_o, _ = _shapley_r2_two_groups(yb, Xdb, Xob)
        vals_demo.append(s_d); vals_other.append(s_o); vals_full.append(r2_f)
        denom = (s_o + 1e-8)
        ratio = s_d/denom
        ratio = float(np.clip(ratio, -1e6, 1e6))
        dom.append(ratio)

    def ci(a):
        a = np.asarray(a, float)
        return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
    return {
        'shap_demo': np.array(vals_demo, float),
        'shap_other': np.array(vals_other, float),
        'r2_full': np.array(vals_full, float),
        'dominance': np.array(dom, float),
        'ci': {
            'shap_demo': ci(vals_demo),
            'shap_other': ci(vals_other),
            'r2_full': ci(vals_full),
            'dominance': ci(dom),
        }
    }



def predictive_attribution_modeling(merged_df, 
                               person_set: PersonSet, 
                               case: CaseConfig,
                               demographic_traits: Tuple[str, ...] = ('gender','ethnicity'),
                               group_keys=("gender", "ethnicity", "age"),
                               consistency_results=None,
                               perf_df: Optional[pd.DataFrame] = None,
                               enable_compute_analysis: bool = True,
                               cv_lmg: bool = True,
                               cv_splits: int = 5,
                               seed: int = 42):
    """
    NOTE on rates:
       rescue_rate = P(profile correct given baseline wrong) = ((~base_ok) & prof_ok).mean() / P(baseline wrong)
       extra_error_rate = P(profile wrong given baseline right) = (( base_ok) & ~prof_ok).mean() / P(baseline right)
    Both denominators use the GLOBAL baseline event rates (p_wrong, p_right) for comparability across profiles.
    """

    print("=" * 60)
    print("Simplified Predictive Attribution Modeling: traits -> boldness (COI/ATI/CAI), benefit–risk, efficiency")
    print("=" * 60)
    print(f"Group keys: {group_keys}")


    def get_valid_profiles(merged_df, person_set):
        all_profile_cols = [c for c in merged_df.columns if c.startswith("profile")]
        return [c for c in all_profile_cols if c in person_set.metadata]

    profile_cols = get_valid_profiles(merged_df, person_set)
    if len(profile_cols) == 0:
        print("ERROR: No valid profiles found in PersonSet metadata")
        return {'error': 'No valid profiles found'}
    print(f"Found {len(profile_cols)} valid profiles for predictive attribution modeling")

    tokens_map, cost_map = {}, {}
    if perf_df is not None and not perf_df.empty:
        perf = perf_df.drop_duplicates("profile").set_index("profile")
        if "tokens_per_sample" in perf.columns:
            tokens_map = perf["tokens_per_sample"].to_dict()
        if "cost_per_sample" in perf.columns:
            cost_map = perf["cost_per_sample"].to_dict()
    have_tokens = len(tokens_map) > 0
    have_cost   = len(cost_map) > 0

    all_trait_keys, trait_reference_categories = set(), {}
    for profile in profile_cols:
        traits = person_set.get_traits(profile, group_keys)
        for trait_name, value in traits.items():
            if value != "Unknown":
                if trait_name == "age" and hasattr(value, 'value'):
                    all_trait_keys.add(f"{trait_name}_{value.value}")
                else:
                    all_trait_keys.add(f"{trait_name}_{value}")


    for trait_name in group_keys:
        counts = Counter()
        for p in profile_cols:
            v = person_set.get_traits(p, group_keys).get(trait_name, "Unknown")
            if v != "Unknown":
                v = v.value if (trait_name == "age" and hasattr(v, 'value')) else v
                counts[str(v)] += 1
        if counts:
            majority_val = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            raise ValueError(f"No valid values found for trait '{trait_name}' among the profiles for predictive attribution modelling.")
    
        if majority_val is not None:
            ref = f"{trait_name}_{majority_val}"
            trait_reference_categories[trait_name] = ref
            if ref in all_trait_keys:
                all_trait_keys.remove(ref)
    
    print(f"Reference categories (majority baseline): {list(trait_reference_categories.values())}")

    base_ok_full = (merged_df['base_pred'] == merged_df['true_label'])
    p_right = float(base_ok_full.mean())
    p_wrong = 1.0 - p_right

    predictive_rows = []
    for profile_name in profile_cols:
        if profile_name not in merged_df.columns:
            continue

        traits = person_set.get_traits(profile_name, group_keys)
        row = {"profile": profile_name}

        for trait_name in group_keys:
            raw = traits.get(trait_name, "Unknown")
            if trait_name == "age" and hasattr(raw, 'value'):
                raw = raw.value
            row[f"raw_{trait_name}"] = raw


        for trait_name in group_keys:
            value = traits.get(trait_name, "Unknown")
            if value != "Unknown":
                key = f"{trait_name}_{value.value}" if (trait_name == "age" and hasattr(value, 'value')) else f"{trait_name}_{value}"
                if key in all_trait_keys:
                    row[key] = 1
        for key in all_trait_keys:
            row.setdefault(key, 0)


        prof_ok = (merged_df[profile_name] == merged_df['true_label'])
        rescue_rate = ((~base_ok_full) & prof_ok).mean() / (p_wrong+1e-12) # P(fix given base wrong)
        extra_error_rate = (base_ok_full & (~prof_ok)).mean() / (p_right+1e-12) # P(harm given base right)
        disagreement_rate = (merged_df[profile_name] != merged_df['base_pred']).mean()


        accuracy = float(prof_ok.mean())
        volatility = 0.0
        if consistency_results and profile_name in consistency_results.get('consistency_data', {}):
            volatility = float(consistency_results['consistency_data'][profile_name]['accuracy_std'])


        bm = boldness_metrics(merged_df, profile_name, baseline_col="base_pred")
        coi, ati, cai = float(bm["COI"]), float(bm["ATI"]), float(bm["CAI"])


        toks = float(tokens_map.get(profile_name, np.nan)) if have_tokens else np.nan
        cost = float(cost_map.get(profile_name, np.nan)) if have_cost else np.nan
        tok_k  = toks/1000.0 if np.isfinite(toks) and toks > 0 else np.nan
        cost_d = cost        if np.isfinite(cost) and cost > 0 else np.nan

        row.update({
            'accuracy': accuracy,
            'volatility': volatility,
            'rescue_rate': float(rescue_rate),
            'extra_error_rate': float(extra_error_rate),
            'disagreement_rate': float(disagreement_rate),
            'COI': coi,
            'ATI': ati,
            'CAI': cai,
            'tokens_per_sample': toks if have_tokens else np.nan,
            'cost_per_sample':   cost if have_cost else np.nan,
            'accuracy_per_1k_tokens': (accuracy / tok_k) if np.isfinite(tok_k) else np.nan,
            'rescue_per_1k_tokens':   (rescue_rate / tok_k) if np.isfinite(tok_k) else np.nan,
            'extra_error_per_1k_tokens': (extra_error_rate/tok_k) if np.isfinite(tok_k) else np.nan,
            'accuracy_per_dollar': (accuracy / cost_d) if np.isfinite(cost_d) else np.nan,
            'rescue_per_dollar':   (rescue_rate / cost_d) if np.isfinite(cost_d) else np.nan,
            'extra_error_per_dollar': (extra_error_rate / cost_d) if np.isfinite(cost_d) else np.nan,
        })
        predictive_rows.append(row)

    predictive_df = pd.DataFrame(predictive_rows)
    print(f"Predictive dataset prepared with {len(predictive_df)} profiles")
    print(f"    Trait dummies: {len([c for c in predictive_df.columns if any(c.startswith(f'{t}_') for t in group_keys)])}")
    print(f"    Reference categories: {list(trait_reference_categories.values())}")


    trait_predictors = sorted([c for c in predictive_df.columns
                               if any(c.startswith(f'{t}_') for t in group_keys) and not c.endswith('_Unknown')])
    
    demographic_predictors = [c for c in trait_predictors if any(c.startswith(f'{t}_') for t in demographic_traits)]
    other_predictors = [c for c in trait_predictors if c not in demographic_predictors]
    all_predictors = demographic_predictors + other_predictors
    print(f"Demographic predictors ({len(demographic_predictors)}), Other traits ({len(other_predictors)})")


    outcomes = ['COI', 'ATI', 'CAI',  # boldness outcomes
                'rescue_rate', 'extra_error_rate',  # benefit–risk
                'accuracy', 'disagreement_rate']  # performance/behavior
    if consistency_results:
        outcomes.append('volatility')
    if have_tokens:
        outcomes += ['tokens_per_sample', 'accuracy_per_1k_tokens', 'rescue_per_1k_tokens', 'extra_error_per_1k_tokens']
    if have_cost:
        outcomes += ['cost_per_sample', 'accuracy_per_dollar', 'rescue_per_dollar', 'extra_error_per_dollar']


    predictive_association_results = {}
    for outcome in outcomes:
        print(f"\n--- Modeling: {outcome.upper()} ---")
        y = predictive_df[outcome].values
        mask = np.isfinite(y)
        if mask.sum() < 3:
            print("  WARNING: Insufficient non-NaN points for regression")
            continue

        y_masked = y[mask]
        X_demo = predictive_df.loc[mask, demographic_predictors].values if demographic_predictors else np.zeros((mask.sum(), 0))
        X_other = predictive_df.loc[mask, other_predictors].values if other_predictors else np.zeros((mask.sum(), 0))
        X_full  = predictive_df.loc[mask, all_predictors].values if all_predictors else np.zeros((mask.sum(), 0))

        r2_demo, r2_other, r2_full, shap_demo, shap_other, shared = _shapley_r2_two_groups(y_masked, X_demo, X_other)

        oof_scores = []
        if X_full.shape[1] > 0 and len(y_masked) >= 5:
            kf = KFold(n_splits=min(5, max(2,len(y_masked)//3)), shuffle=True, random_state=seed)
            for train_idx, test_idx in kf.split(X_full):
                try:
                    m = LinearRegression().fit(X_full[train_idx], y_masked[train_idx])
                    oof_scores.append(float(m.score(X_full[test_idx], y_masked[test_idx])))
                except Exception:
                    pass
        oof_mean = float(np.nanmean(oof_scores)) if oof_scores else np.nan
        oof_std  = float(np.nanstd(oof_scores))  if oof_scores else np.nan
        

        boot = _bootstrap_shapley_two_groups(y_masked, X_demo, X_other, B=1000, seed=seed)
        dom_ratio = shap_demo / (shap_other + 1e-8)
        
        cv_stats = {}
        if cv_lmg and X_full.shape[1] > 0 and len(y_masked) >= max(4, cv_splits):
            rng = np.random.default_rng(seed)
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            s_demo, s_other, r2_full_list = [], [], []
            for tr, te in kf.split(y_masked):
                Xd_tr, Xd_te = X_demo[tr], X_demo[te]
                Xo_tr, Xo_te = X_other[tr], X_other[te]
                Xf_tr, Xf_te = X_full[tr],  X_full[te]
                y_tr, y_te   = y_masked[tr], y_masked[te]
                m_d = LinearRegression().fit(Xd_tr, y_tr) if Xd_tr.shape[1] else None
                m_o = LinearRegression().fit(Xo_tr, y_tr) if Xo_tr.shape[1] else None
                m_f = LinearRegression().fit(Xf_tr, y_tr) if Xf_tr.shape[1] else None
                def r2(m, X, y):
                    if (m is None) or (X.shape[1]==0): return 0.0
                    yhat = m.predict(X); ym = y.mean()
                    ss_res = float(np.sum((y - yhat)**2))
                    ss_tot = float(np.sum((y - ym)**2)) + 1e-12
                    return 1.0 - ss_res/ss_tot
                r2_d = r2(m_d, Xd_te, y_te)
                r2_o = r2(m_o, Xo_te, y_te)
                r2_f = r2(m_f, Xf_te, y_te)
                s_d  = 0.5 * (r2_d + (r2_f - r2_o))
                s_o  = 0.5 * (r2_o + (r2_f - r2_d))
                s_demo.append(max(0.0, s_d)); s_other.append(max(0.0, s_o)); r2_full_list.append(max(0.0, r2_f))
            s_demo = float(np.mean(s_demo)); s_other = float(np.mean(s_other)); r2_f = float(np.mean(r2_full_list))
            cv_stats = {
                'cv_demo_unique': s_demo,
                'cv_other_unique': s_other,
                'cv_r2_full': r2_f,
                'cv_dominance_ratio': (s_demo / (s_other + 1e-8))
            }

        if cv_lmg and cv_stats:
            print(f"  CV-LMG: demo={cv_stats['cv_demo_unique']:.3f} "
                  f"other={cv_stats['cv_other_unique']:.3f},  CV R²(full)={cv_stats['cv_r2_full']:.3f}")
        additivity_gap = r2_full - (r2_demo + r2_other)
        print(f"  In-sample R²: demo={r2_demo:.3f},  other={r2_other:.3f},  full={r2_full:.3f}, "
              f" gap={additivity_gap:+.3f}")
        print(f"  In-sample Shapley: demo={shap_demo:.3f},  other={shap_other:.3f},  Shared≈{shared:.3f}")

        coeffs = {}
        coeffs_std = {}
        if X_full.shape[1] > 0:
            model_full = LinearRegression().fit(X_full, y_masked)
            y_sd = float(np.std(y_masked, ddof=1)) if np.std(y_masked, ddof=1) > 0 else np.nan
            X_sd = np.std(X_full, axis=0, ddof=1)
            for i, predictor in enumerate(all_predictors):
                beta = float(model_full.coef_[i])
                coeffs[predictor] = beta

                if np.isfinite(y_sd) and X_sd[i] > 0:
                    coeffs_std[predictor] = float(beta*(X_sd[i]/y_sd))
        else:
            model_full = None
        
        primary_demo  = cv_stats.get('cv_demo_unique', shap_demo) if cv_lmg else shap_demo
        primary_other = cv_stats.get('cv_other_unique', shap_other) if cv_lmg else shap_other
        primary_full  = cv_stats.get('cv_r2_full', r2_full) if cv_lmg else r2_full

        predictive_association_results[outcome] = {     
            'r2_full': primary_full,       
            'r2_full_oof_mean': oof_mean,
            'r2_full_oof_std':  oof_std,
            'r2_demographics': r2_demo,      
            'r2_other': r2_other,      
            'demo_unique':  primary_demo,    
            'other_unique': primary_other,  
            'shared_variance': shared,
            'dominance_ratio': float(dom_ratio),
            'shapley_bootstrap_ci': {
                'demo_unique': list(boot['ci']['shap_demo']),
                'other_unique': list(boot['ci']['shap_other']),
                'r2_full': list(boot['ci']['r2_full']),
                'dominance_ratio': list(boot['ci']['dominance'])
            },
            'coefficients': coeffs,
            'coefficients_std': coeffs_std,
            'models': {'demographics': None, 'other': None, 'full': model_full}
        }
        predictive_association_results[outcome].update(cv_stats)
        
        print(f"  Demographics R²: {r2_demo:.3f},  Other traits R²: {r2_other:.3f},  Full R²: {r2_full:.3f},   OOF R²≈ {oof_mean:.3f}±{oof_std:.3f}")
        print(f"  Shapley: demo={shap_demo:.3f} [{boot['ci']['shap_demo'][0]:.3f},{boot['ci']['shap_demo'][1]:.3f}]"
              f" other={shap_other:.3f} [{boot['ci']['shap_other'][0]:.3f},{boot['ci']['shap_other'][1]:.3f}]"
              f",  dominance={dom_ratio:.2f} [{boot['ci']['dominance'][0]:.2f},{boot['ci']['dominance'][1]:.2f}],  Shared≈{shared:.3f}")


    print(f"\n{'='*60}")
    print("Demographic Predictiveness check (CI-based, associational)")
    print(f"{'='*60}")
    
    error_detection = {}
    for outcome in outcomes:
        if outcome not in predictive_association_results:
            continue
        results = predictive_association_results[outcome]
    
        total_demo   = float(results.get('r2_demographics', 0.0))
        demo_unique  = float(results.get('demo_unique', 0.0))
        other_unique = float(results.get('other_unique', 0.0))
        dom_ratio    = float(results.get('dominance_ratio', np.nan))
        total_demo      = float(results.get('r2_demographics', 0.0))
        demo_unique_in  = float(results.get('demo_unique', 0.0))
        other_unique_in = float(results.get('other_unique', 0.0))
        dom_ratio_in    = float(results.get('dominance_ratio', np.nan))
        demo_unique_cv  = float(results.get('cv_demo_unique',  np.nan))
        other_unique_cv = float(results.get('cv_other_unique', np.nan))
        dom_ratio_cv    = float(results.get('cv_dominance_ratio', np.nan))
    

        boot_ci      = results.get('shapley_bootstrap_ci', {})
        dom_ci       = boot_ci.get('dominance_ratio', [np.nan, np.nan])
        demo_ci      = boot_ci.get('demo_unique', [np.nan, np.nan])
        other_ci     = boot_ci.get('other_unique', [np.nan, np.nan])
    

        flag_dominant = (np.isfinite(dom_ci[0]) and dom_ci[0] > 1.0)
    
        coeffs = results.get('coefficients', {})
        demographic_coeffs = {k: v for k, v in coeffs.items()
                              if any(k.startswith(f'{t}_') for t in demographic_traits)}
        strongest = max(demographic_coeffs.items(), key=lambda kv: abs(kv[1])) if demographic_coeffs else None
    
        error_detection[outcome] = {
                'in_sample': {
                    'r2_demographics': total_demo,
                    'demo_unique': demo_unique_in,
                    'other_unique': other_unique_in,
                    'dominance_ratio': dom_ratio_in,
                    'ci': {
                        'demo_unique': [float(demo_ci[0]), float(demo_ci[1])],
                        'other_unique': [float(other_ci[0]), float(other_ci[1])],
                        'dominance':   [float(dom_ci[0]),  float(dom_ci[1]) ],
                    },
                    'flag_dominant': bool(flag_dominant),
                },
                'cv': {
                    'demo_unique': demo_unique_cv,
                    'other_unique': other_unique_cv,
                    'dominance_ratio': dom_ratio_cv,
                    'r2_full': float(results.get('cv_r2_full', np.nan)),
                    'oof_r2_mean': float(results.get('r2_full_oof_mean', np.nan)),
                    'oof_r2_std':  float(results.get('r2_full_oof_std',  np.nan)),
                },
                'coefficients_demo': demographic_coeffs,
                'strongest_demographic': strongest,
            }

        error_detection[outcome].update({
            'r2_demographics': float(total_demo),
            'demo_unique': float(demo_unique_in),
            'demo_unique_ci': [float(demo_ci[0]), float(demo_ci[1])],
            'other_unique': float(other_unique_in),
            'other_unique_ci': [float(other_ci[0]), float(other_ci[1])],
            'dominance_ratio': float(dom_ratio_in),
            'dominance_ci': [float(dom_ci[0]), float(dom_ci[1])],
            'flag_dominant': bool(flag_dominant),
            'demographic_coefficients': demographic_coeffs,
            'level': '',
        })

        msg_flag = " (dominance CI>1)" if flag_dominant else ""
        print(f"\n{outcome.upper()}: demo R²={total_demo:.3f},  "
              f"demo_unique(in)={demo_unique_in:.3f} [{demo_ci[0]:.3f},{demo_ci[1]:.3f}],  "
              f"other_unique(in)={other_unique_in:.3f} [{other_ci[0]:.3f},{other_ci[1]:.3f}],  "
              f"dominance(in)={dom_ratio_in:.2f} [{dom_ci[0]:.2f},{dom_ci[1]:.2f}]{msg_flag}")
        if np.isfinite(demo_unique_cv):
            print(f"  CV-LMG: demo_unique={demo_unique_cv:.3f} "
                  f"other_unique={other_unique_cv:.3f},  dominance={dom_ratio_cv:.2f}")
        if strongest:
            print(f"  strongest demographic beta: {strongest[0]} = {strongest[1]:.3f}")
    


    print(f"\n{'='*60}")
    print("Predictor–Outcome Association Interpretation")
    print(f"{'='*60}")

    strongest_predictors = {}
    for outcome, results in predictive_association_results.items():
        if not results['coefficients']:
            continue
        best = max(results['coefficients'].items(), key=lambda kv: abs(kv[1]))
        strongest_predictors[outcome] = {
            'predictor': best[0],
            'coefficient': float(best[1]),
            'direction': 'increases' if best[1] > 0 else 'decreases',
            'is_demographic': any(best[0].startswith(f'{t}_') for t in demographic_traits)
        }
        print(f"{outcome.upper()}: strongest {best[0]} (beta={best[1]:.3f})")


    print(f"\n{'='*60}")
    print("Recommendations (boldness/benefit–risk)")
    print(f"{'='*60}")

    recommendations = {}


    coi_coeffs = predictive_association_results.get('COI', {}).get('coefficients', {})
    risk_coeffs = predictive_association_results.get('extra_error_rate', {}).get('coefficients', {})
    safe_bold = []

    for k, v in coi_coeffs.items():

        if any(k.startswith(f'{t}_') for t in demographic_traits):
            continue 

        risk_v = risk_coeffs.get(k, 0.0)

        if v > 0 and risk_v <= 0.0:
            safe_bold.append((k, float(v), float(risk_v)))
            
    safe_bold.sort(key=lambda t: t[1], reverse=True)
    recommendations['safe_bold_traits'] = safe_bold[:5]
    if safe_bold:
        print("  Traits that raise COI without increasing extra_error_rate:")
        for k, v, rv in safe_bold[:5]:
            print(f"    + {k}: beta_COI={v:.3f}, beta_risk={rv:.3f}")
    else:
        print("  No non-demographic traits jointly improving COI with non-positive risk beta found.")


    risk_sorted = [(k, v) for k, v in risk_coeffs.items()
                   if not any(k.startswith(f'{t}_') for t in demographic_traits)]
    reducers = [(k, v) for (k, v) in risk_sorted if v < 0]
    reducers.sort(key=lambda t: t[1])  

    recommendations['risk_minimizing_traits'] = reducers[:5]
    if reducers:
        print("  Traits that reduce extra_error_rate:")
        for k, v in reducers[:5]:
            print(f"    + {k}: beta_risk={v:.3f}")
    else:
        print("  No non-demographic traits with negative beta on extra_error_rate.")


    acc_coeffs = predictive_association_results.get('accuracy', {}).get('coefficients', {})
    acc_sorted = [(k, v) for k, v in acc_coeffs.items()
                  if not any(k.startswith(f'{t}_') for t in demographic_traits)]
    acc_sorted.sort(key=lambda t: t[1], reverse=True)
    recommendations['accuracy_traits'] = acc_sorted[:5]


    mediation_evidence = {}
    for outcome in outcomes:
        if outcome not in predictive_association_results:
            continue
        demo_direct = predictive_association_results[outcome].get('demo_unique', 0.0)
        other_dir   = predictive_association_results[outcome].get('other_unique', 0.0)
        shared_var  = predictive_association_results[outcome].get('shared_variance', 0.0)
        if demo_direct > 1e-3:
            mediation_strength = other_dir / demo_direct
            suppression_strength = shared_var / demo_direct
        else:
            mediation_strength = float('inf') if other_dir > 0.01 else 0.0
            suppression_strength = 0.0
        mediation_evidence[outcome] = {
            'mediation_strength': float(mediation_strength),
            'suppression_strength': float(suppression_strength)
        }


    return {
        'predictive_data': predictive_df,
        'predictive_association_results': predictive_association_results,
        'error_detection': error_detection,
        'strongest_predictors': strongest_predictors,
        'mediation_evidence': mediation_evidence,
        'recommendations': recommendations,
            'parameters': {
            'group_keys': tuple(group_keys),
            'demographic_traits': tuple(demographic_traits),
            'cv_lmg': bool(cv_lmg),
            'cv_splits': int(cv_splits),
            'seed': int(seed),
            },
            'overall_assessment': {
            'outcomes_modeled': outcomes,
            'demographic_predictiveness_summary': {
                k: {
                    'r2_demographics': v.get('r2_demographics'),
                    'demo_unique': v.get('demo_unique'),
                    'other_unique': v.get('other_unique'),
                    'dominance_ratio': v.get('dominance_ratio'),
                    'dominance_ci': v.get('dominance_ci'),
                    'flag_dominant': v.get('flag_dominant', False),
                }
                for k, v in error_detection.items()
            }
        },
        'theoretical_framework': {
            'boldness_primary': "COI (consensus-oriented intervention); ATI/CAI as secondary",
            'benefit_risk': "rescue_rate vs extra_error_rate (truth-aware)",
            'fairness_view': "report demographic predictiveness (R², dominance), avoid normative language"
        }
    }

def save_predictive_demo_r2_summary(
    predictive_association_results: Dict[str, Any],
    output_dir: str,
    filename: str = "demo_r2_summary.csv"
):
    """Save demo R^2 summary in the format expected by paper_figures.py"""
    

    predictive_dir = os.path.join(output_dir, "predictive")
    os.makedirs(predictive_dir, exist_ok=True)
    

    target_metrics = {
        'accuracy': 'accuracy',
        'extra_error': 'extra_error_rate', 
        'disagreement': 'disagreement_rate'
    }
    
    rows = []
    for display_name, outcome_key in target_metrics.items():
        if outcome_key in predictive_association_results:
            results = predictive_association_results[outcome_key]
            
            demo_r2 = float(results.get('r2_demographics', 0.0))
            
            boot_ci = results.get('shapley_bootstrap_ci', {})
            demo_ci = boot_ci.get('demo_unique', [np.nan, np.nan])
            
            rows.append({
                'metric': display_name,
                'demo_r2': demo_r2,
                'ci_low': float(demo_ci[0]) if len(demo_ci) > 0 else np.nan,
                'ci_high': float(demo_ci[1]) if len(demo_ci) > 1 else np.nan
            })
    
    if rows:
        df = pd.DataFrame(rows)
        output_path = os.path.join(predictive_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved demo R² summary to: {output_path}")
        return output_path
    
    return None


def visualize_predictive_attribution(results, figsize=(6.5, 4.2), coef_bootstrap_iters=300, topk_forest=12, seed=42):
    apply_neurips_figure_style()


    if isinstance(results, dict) and 'predictive_association_results' in results:
        predictive_association_results = results['predictive_association_results']
        mediation = results.get('mediation_evidence', {})
        strongest_predictors = results.get('strongest_predictors', {})
        predictive_df = results.get('predictive_data', None)
        error_detection = results.get('error_detection', {})
    else:
        predictive_association_results = results or {}
        mediation = {}
        strongest_predictors = {}
        predictive_df = None
        error_detection = {}

    dem_traits = (results.get('parameters', {}).get('demographic_traits') if isinstance(results, dict) else None)
    if dem_traits is None:
        dem_traits = ['gender', 'ethnicity']
    else:
        dem_traits = list(dem_traits)


    all_outcomes = list(predictive_association_results.keys())
    bold_first = [o for o in ['COI','ATI','CAI'] if o in all_outcomes]
    others = [o for o in all_outcomes if o not in bold_first]
    outcomes = bold_first + others


    core_cols = [c for c in ['COI','ATI','CAI','rescue_rate','extra_error_rate','disagreement_rate'] if c in outcomes]
    perf_cols = [c for c in [
        'accuracy','volatility',
        'tokens_per_sample','cost_per_sample',
        'accuracy_per_1k_tokens','rescue_per_1k_tokens','extra_error_per_1k_tokens',
        'accuracy_per_dollar','rescue_per_dollar','extra_error_per_dollar'
    ] if c in outcomes]


    all_coeffs = []
    for o in outcomes:
        coeffs = predictive_association_results.get(o, {}).get('coefficients', {})
        if coeffs:
            all_coeffs = list(coeffs.keys())
            break
    
    demographic_predictors = [p for p in all_coeffs if any(p.startswith(f'{t}_') for t in dem_traits)]
    other_predictors = [p for p in all_coeffs if p not in demographic_predictors]
    all_predictors = demographic_predictors + other_predictors

    figs = {}


    def _r2_plot(title, subset):
        fig, ax = new_pub_fig(title, figsize)
        if subset:
            demo_contrib   = [float(predictive_association_results[o].get('demo_unique', 0.0)) for o in subset]
            other_contrib  = [float(predictive_association_results[o].get('other_unique', 0.0)) for o in subset]
            shared_contrib = [float(predictive_association_results[o].get('shared_variance', 0.0)) for o in subset]
            r2s            = [float(predictive_association_results[o].get('r2_full', 0.0)) for o in subset]
            x = np.arange(len(subset)); width = 0.6
            ax.bar(x, demo_contrib,  width, label='Demographics',  alpha=0.85)
            ax.bar(x, other_contrib, width, bottom=demo_contrib, label='Other Traits', alpha=0.85)
            ax.bar(x, shared_contrib,width, bottom=np.array(demo_contrib)+np.array(other_contrib), label='Shared', alpha=0.85)
            for i, r2 in enumerate(r2s):
                ax.text(i, r2 + 0.01, f'R²={r2:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            ax.set_xlabel('Outcome Variables')
            ax.set_ylabel('Variance Explained (R²)')
            ax.set_xticks(x)
            ax.set_xticklabels([o.replace('_', ' ').title() for o in subset], rotation=45, ha='right')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No outcomes available', ha='center', va='center', transform=ax.transAxes)
        return fig


    figs['variance_decomposition'] = _r2_plot('Variance Decomposition (Associational R²): Demographics vs Other Traits', outcomes)

    if core_cols:
        figs['variance_decomposition_core'] = _r2_plot('Variance Decomposition — Core (COI/ATI/CAI & Benefit–Risk)', core_cols)
    if perf_cols:
        figs['variance_decomposition_perf'] = _r2_plot('Variance Decomposition — Performance/Stability/Efficiency', perf_cols)


    def _coef_heatmap(title, subset):
        fig, ax = new_pub_fig(title, figsize)
        if not (all_predictors and subset):
            ax.text(0.5, 0.5, 'No coefficients available', ha='center', va='center', transform=ax.transAxes)
            return fig


        coef_matrix = np.zeros((len(all_predictors), len(subset)))
        for j, outcome in enumerate(subset):
            coefs_std = predictive_association_results.get(outcome, {}).get('coefficients_std', {})
            coefs_raw = predictive_association_results.get(outcome, {}).get('coefficients', {})
            for i, predictor in enumerate(all_predictors):
                val = coefs_std.get(predictor, None)
                if val is None:
                    val = coefs_raw.get(predictor, 0.0)
                coef_matrix[i, j] = float(val)


        row_strength = np.nanmax(np.abs(coef_matrix), axis=1)
        row_order = np.argsort(-row_strength)
        coef_matrix = coef_matrix[row_order, :]
        ordered_predictors = [all_predictors[i] for i in row_order]


        vmax = float(np.nanpercentile(np.abs(coef_matrix), 98))
        vmax = max(vmax, 1e-3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto', norm=norm)


        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        ticks = np.linspace(-vmax, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:+.2f}' for t in ticks])
        cbar.set_label('Coefficient Value', fontsize=9)

        annot_thresh = max(0.35 * vmax, 0.01)
        for i in range(coef_matrix.shape[0]):
            for j in range(coef_matrix.shape[1]):
                v = coef_matrix[i, j]
                if abs(v) >= annot_thresh:
                    ax.text(j, i, f'{v:+.2f}',
                            ha='center', va='center',
                            color=('white' if abs(v) > 0.6*vmax else 'black'),
                            fontsize=8, fontweight='bold')

        ax.set_xticks(np.arange(len(subset)))
        ax.set_yticks(np.arange(len(ordered_predictors)))
        ax.set_xticklabels([s.replace('_',' ').title() for s in subset], rotation=35, ha='right', fontsize=9)
        ax.set_yticklabels([p.replace('_',' ').title() for p in ordered_predictors], fontsize=9)


        demo_idx = [i for i, p in enumerate(ordered_predictors) if p.startswith('gender_') or p.startswith('ethnicity_')]
        if demo_idx:
            contiguous = 0
            for i in range(len(ordered_predictors)):
                if i in demo_idx: contiguous = i
                else: break
            ax.hlines(contiguous + 0.5, -0.5, len(subset)-0.5, colors='k', linestyles=':', lw=0.6, alpha=0.5)


        for s in ax.spines.values(): s.set_visible(False)
        ax.set_facecolor('#f7f7f7')
        ax.grid(False)
        return fig


    figs['coef_heatmap'] = _coef_heatmap('Standardized Coefficients (beta; associational)', outcomes)
 
    figs['coef_heatmap_core'] = _coef_heatmap('Standardized Coefficients — Core (COI/ATI/CAI & Benefit–Risk)', core_cols)
    figs['coef_heatmap_perf'] = _coef_heatmap('Standardized Coefficients — Performance/Stability/Efficiency', perf_cols)


    figC, ax = new_pub_fig('Attribution Ratio (Other vs Demographics)', figsize)
    if mediation and outcomes:
        valid_outcomes = [o for o in outcomes if o in mediation]
        ratios = [float(mediation[o].get('mediation_strength', 0.0)) for o in valid_outcomes]
        if len(valid_outcomes) > 0:
            bars = ax.bar(range(len(valid_outcomes)), ratios, alpha=0.85)
            ax.axhline(1.5, linestyle='--', alpha=0.7, label='High ratio')
            ax.axhline(0.8,  linestyle='--', alpha=0.7, label='Moderate ratio')
            for bar, r in zip(bars, ratios):
                ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.1, f'{r:.2f}',
                        ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel('Attribution Ratio')
            ax.set_xticks(range(len(valid_outcomes)))
            ax.set_xticklabels([o.replace('_', ' ').title() for o in valid_outcomes], rotation=45, ha='right')
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No mediation data available', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No mediation analysis available', ha='center', va='center', transform=ax.transAxes)
    figs['mediation'] = figC


    figD, ax = new_pub_fig('Associational Network Structure', figsize)
    if outcomes:
        outcomes_sorted = sorted(outcomes, key=lambda o: float(predictive_association_results[o].get('r2_full', 0.0)), reverse=True)
        draw_outcomes = outcomes_sorted[:5]
        outcome_positions = [(0.8, y) for y in np.linspace(0.85, 0.15, num=len(draw_outcomes))]
        node_pos = {'Demographics': (0.2, 0.75), 'Other Traits': (0.2, 0.35)}
        for i, o in enumerate(draw_outcomes):
            node_pos[o] = outcome_positions[i]
        for node, (x0, y0) in node_pos.items():
            if node == 'Demographics':
                color = '#1f77b4'; size = 1200
            elif node == 'Other Traits':
                color = '#ff7f0e'; size = 1200
            else:
                color = '#2ca02c'; size = 800
            ax.scatter(x0, y0, s=size, c=color, edgecolors='black', alpha=0.7, linewidth=2)
            ax.text(x0, y0-0.06, node.replace('_',' ').title(), ha='center', va='top',
                    fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        for o in draw_outcomes:
            sp = strongest_predictors.get(o, {})
            pred = sp.get('predictor', None)
            coef = float(sp.get('coefficient', 0.0)) if sp else 0.0
            if not pred or o not in node_pos:
                continue
            start = 'Demographics' if (pred.startswith('gender_') or pred.startswith('ethnicity_')) else 'Other Traits'
            if start not in node_pos:
                continue
            lw = min(abs(coef) * 20, 4) + 0.5
            color = '#2ca02c' if coef > 0 else '#d62728'
            ax.annotate('', xy=node_pos[o], xytext=node_pos[start],
                        arrowprops=dict(arrowstyle='-', lw=lw, color=color, alpha=0.7))
            if 'Demographics' in node_pos and 'Other Traits' in node_pos:
                # undirected
                ax.annotate('', xy=node_pos['Other Traits'], xytext=node_pos['Demographics'],
                            arrowprops=dict(arrowstyle='-', lw=2, alpha=0.5, linestyle='--'))
        ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.text(0.02, 0.02, 'Associations: Green = positive beta, Red = negative beta\nLine thickness ∝ abs(beta)',
                 transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.9),
                 fontsize=8, va='bottom')
    else:
        ax.text(0.5, 0.5, 'No outcomes available', ha='center', va='center', transform=ax.transAxes)
    figs['associational_network'] = figD



    figE, ax = new_pub_fig('Demographic Predictiveness vs Dominance', figsize)
    xs, ys, labels = [], [], []
    yerr_lo, yerr_hi = [], []

    for o, det in error_detection.items():
        x = float(det.get('r2_demographics',
                          det.get('in_sample', {}).get('r2_demographics', np.nan)))
        dom = float(det.get('dominance_ratio',
                            det.get('in_sample', {}).get('dominance_ratio', np.nan)))
        ci = det.get('in_sample', {}).get('ci', {}).get('dominance', None)
        if ci is None:
            ci = det.get('dominance_ci', [np.nan, np.nan])

        if not (np.isfinite(x) and np.isfinite(dom) and
                isinstance(ci, (list, tuple)) and len(ci) == 2 and
                np.isfinite(ci[0]) and np.isfinite(ci[1])):
            continue

        lo, hi = float(ci[0]), float(ci[1])
        if hi < lo:                      
            lo, hi = hi, lo

        lower = max(0.0, dom - lo)
        upper = max(0.0, hi - dom)

        xs.append(x); ys.append(dom)
        yerr_lo.append(lower); yerr_hi.append(upper)
        labels.append(o.replace('_', ' ').title())

    if xs:
        yerr_arr = np.vstack([yerr_lo, yerr_hi])
        ax.errorbar(xs, ys, yerr=yerr_arr, fmt='o', ecolor='black', capsize=3, linestyle='none')
        ax.scatter(xs, ys, s=60, alpha=0.85)
        ax.axhline(1.0, linestyle='--', alpha=0.5, label='Dominance = 1')
        ax.axvline(0.10, linestyle='--', alpha=0.5, label='R² dem = 0.10')
        ax.set_xlabel('R² (Demographics)')
        ax.set_ylabel('Dominance Ratio (demo_unique / other_unique)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No demographic audit data', ha='center', va='center', transform=ax.transAxes)

    figs['error_landscape'] = figE




    figF, ax = new_pub_fig('Trait Associations: Boldness (COI) vs Risk (Extra Error Rate)', figsize)

    def _coef_map(outcome: str):

        res = predictive_association_results.get(outcome, {})
        std = res.get('coefficients_std', {}) or {}
        raw = res.get('coefficients', {}) or {}
        out = {}

        for p in all_predictors:
            v = std.get(p, None)
            if v is None or not np.isfinite(v):
                v = raw.get(p, None)
            if v is not None and np.isfinite(v):
                out[p] = float(v)
        return out

    coi_map  = _coef_map('COI')
    risk_map = _coef_map('extra_error_rate')

    if coi_map and risk_map:
        common = [p for p in all_predictors if (p in coi_map and p in risk_map)]
        xs = np.array([coi_map[p]  for p in common], dtype=float)
        ys = np.array([risk_map[p] for p in common], dtype=float)
        if xs.size == 0:
            ax.text(0.5, 0.5, 'No overlapping predictors between COI and Extra Error models',
                    ha='center', va='center', transform=ax.transAxes)
        else:

            vmax = float(np.nanpercentile(np.abs(np.concatenate([xs, ys])), 98)) if np.isfinite(xs).any() else 0.0

            if not np.isfinite(vmax) or vmax < 1e-3:
                vmax = 0.1
                
            ax.axhline(0, linestyle='--', alpha=0.6)
            ax.axvline(0, linestyle='--', alpha=0.6)
            ax.set_xlim(-vmax, vmax)
            ax.set_ylim(-vmax, vmax)


            ax.axvspan(0, vmax, ymin=0.0, ymax=0.5, alpha=0.06)

    
            is_demo = [p.startswith('gender_') or p.startswith('ethnicity_') for p in common]
            for xi, yi, p, dem in zip(xs, ys, common, is_demo):
                ax.scatter(xi, yi, s=50, edgecolors='black', linewidth=0.6,
                           c=('#1f77b4' if dem else '#2ca02c'), alpha=0.85)


            safe = [(p, float(coi_map[p]), float(risk_map[p]))
                    for p in common
                    if not (p.startswith('gender_') or p.startswith('ethnicity_'))]
            safe = [t for t in safe if (t[1] > 0 and t[2] <= 0)]
            safe.sort(key=lambda t: (t[1] - max(0.0, t[2])), reverse=True)
            for p, bx, by in safe[:8]:
                ax.annotate(p.replace('_',' '), (bx, by), xytext=(5, 4),
                            textcoords='offset points', fontsize=7)

            ax.set_xlabel('Standardized beta (COI; Boldness ↑)')
            ax.set_ylabel('Standardized beta (Extra Error; Risk ↑)')
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.02, 'Shaded: increase boldness without increasing risk',
                    transform=ax.transAxes, fontsize=8, alpha=0.8)
    else:
        ax.text(0.5, 0.5, 'COI or Extra Error coefficients unavailable',
                ha='center', va='center', transform=ax.transAxes)

    figs['safe_boldness_traits'] = figF


    if predictive_df is not None and all_predictors:
        target_outcome = 'COI' if 'COI' in predictive_df.columns else ('rescue_rate' if 'rescue_rate' in predictive_df.columns else None)
        if target_outcome is not None:
            X = predictive_df[all_predictors].values if all(p in predictive_df.columns for p in all_predictors) else None
            y = predictive_df[target_outcome].values
            if X is not None and y is not None:
                mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
                X = X[mask]; y = y[mask]
                if X.shape[0] >= 3 and X.shape[1] >= 1:
                    rng = np.random.default_rng(seed)
                    B = int(coef_bootstrap_iters)
                    coefs = np.zeros((B, X.shape[1])) * np.nan
                    for b in range(B):
                        idx = rng.integers(0, X.shape[0], size=X.shape[0])
                        xb = X[idx]; yb = y[idx]
                        try:
                            model = LinearRegression().fit(xb, yb)
                            coefs[b] = model.coef_
                        except Exception:
                            pass
                    mean = np.nanmean(coefs, axis=0)
                    lo   = np.nanpercentile(coefs, 2.5, axis=0)
                    hi   = np.nanpercentile(coefs, 97.5, axis=0)
                    rows = [(p, float(mean[i]), float(lo[i]), float(hi[i])) for i, p in enumerate(all_predictors)]
                    rows.sort(key=lambda t: abs(t[1]), reverse=True)
                    rows = rows[:topk_forest]
                    figG, ax = new_pub_fig(f'Coefficient Forest ({target_outcome.replace("_"," ").title()})', figsize)
                    y_pos = np.arange(len(rows))
                    m = np.array([r[1] for r in rows]); lo_arr = np.array([r[2] for r in rows]); hi_arr = np.array([r[3] for r in rows])
                    xerr = np.vstack([m - lo_arr, hi_arr - m])
                    ax.errorbar(m, y_pos, xerr=xerr, fmt='o', ecolor='black', elinewidth=1, capsize=3, color='black')
                    for yi, (name, mi, _, _) in enumerate(rows):
                        color = '#1f77b4' if (name.startswith('gender_') or name.startswith('ethnicity_')) else '#2ca02c'
                        ax.scatter(mi, yi, s=36, color=color, zorder=3)
                    ax.axvline(0, linestyle='--', alpha=0.6)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([r[0].replace('_',' ') for r in rows], fontsize=8)
                    ax.set_xlabel('Coefficient (beta)')
                    ax.set_ylabel('Predictors')
                    ax.grid(True, axis='x', alpha=0.3)
                    figs['coef_forest'] = figG

    return figs



def _dir(case, plots_root=None, strategy=None, stage="tier3", extra=None, sub_case=None):
    return resolve_plot_dir(
        case, plots_root=plots_root, strategy=strategy, stage=stage, extra_subdir=extra, sub_case=sub_case
    )

def _ensure(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _save(
    fig,
    case,
    plots_root=None,
    strategy=None,
    stage: str = "tier3",
    subdir: str = None,
    filename: str = None,
    fmt: str = "pdf",
    dpi: int = None,
    sub_case: Optional[str] = None
):
    """
    Save a matplotlib Figure to the resolved tier3 directory.
    - If filename is None, uses 'figure.<fmt>'.
    - If filename has no extension, appends .<fmt>.
    """
    out_dir = _ensure(_dir(case, plots_root=plots_root, strategy=strategy, stage=stage, extra=subdir, sub_case=sub_case))

    if filename is None:
        filename = f"figure.{fmt}"
    else:
        root, ext = os.path.splitext(filename)
        if not ext: 
            filename = f"{filename}.{fmt}"

    path = os.path.join(out_dir, filename)

    try:
        fig.canvas.draw()
    except Exception:
        pass

    fig.savefig(path, bbox_inches="tight", dpi=dpi, format=fmt)
    try:
        plt.close(fig)
    except Exception:
        pass
    return path


def save_tier3_outputs(
    consistency_results: Dict[str, Any],
    predictive_association_results: Dict[str, Any],
    case: CaseConfig,
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,
    stage: str = "tier3",
    subdirs: Optional[Dict[str, str]] = None,
    sub_case: Optional[str] = None,
    *,
    fmt: str = "pdf",
    dpi: int = None,
    person_set: Optional[PersonSet] = None,
    group_keys: Optional[Tuple[str, ...]] = None,
    figs_consistency: Optional[Dict[str, Any]] = None,
    figs_predictive: Optional[Dict[str, Any]] = None,
):
    subdirs = subdirs or {}
    dir_consistency = _dir(case, plots_root, strategy, stage, extra=subdirs.get("consistency_boldness", "consistency_boldness"), sub_case=sub_case)
    dir_predictive = _dir(case, plots_root, strategy, stage, extra=subdirs.get("predictive", "predictive"), sub_case=sub_case)
    _ensure(dir_consistency); _ensure(dir_predictive)

    if figs_consistency is None:
        try:
            figs_consistency = plot_consistency_boldness_analysis(
                consistency_results,
                person_set=person_set,
                group_keys=(group_keys or ("gender", "ethnicity", "age")),
            ) or {}
        except Exception as e:
            print(f"[WARN] Could not build consistency/boldness figs: {e}")
            figs_consistency = {}

    if figs_predictive is None:
        try:
            figs_predictive = visualize_predictive_attribution(predictive_association_results) or {}
        except Exception as e:
            print(f"[WARN] Could not build predictive association figs: {e}")
            figs_predictive = {}

    saved = {"consistency_boldness": {}, "predictive": {}}


    if "consistency_vs_boldness" in figs_consistency:
        saved["consistency_boldness"]["consistency_vs_boldness"] = _save(
            figs_consistency["consistency_vs_boldness"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="consistency_vs_boldness", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "volatility_vs_rescue" in figs_consistency:
        saved["consistency_boldness"]["volatility_vs_rescue"] = _save(
            figs_consistency["volatility_vs_rescue"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="volatility_vs_rescue", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "boldness_vs_accuracy" in figs_consistency:
        saved["consistency_boldness"]["boldness_vs_accuracy"] = _save(
            figs_consistency["boldness_vs_accuracy"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="boldness_vs_accuracy", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "archetype_by_demo" in figs_consistency:
        saved["consistency_boldness"]["archetype_by_demo"] = _save(
            figs_consistency["archetype_by_demo"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="archetype_by_demo", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "cv_trends" in figs_consistency:
        saved["consistency_boldness"]["cv_trends"] = _save(
            figs_consistency["cv_trends"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="cv_trends", fmt=fmt, dpi=dpi, sub_case=sub_case
        )

    if "benefit_risk_frontier" in figs_consistency:
        saved["consistency_boldness"]["benefit_risk_frontier"] = _save(
            figs_consistency["benefit_risk_frontier"], case, plots_root, strategy, stage,
            subdir=subdirs.get("consistency_boldness","consistency_boldness"),
            filename="benefit_risk_frontier", fmt=fmt, dpi=dpi, sub_case=sub_case
        )


    if "variance_decomposition" in figs_predictive:
        saved["predictive"]["variance_decomposition"] = _save(
            figs_predictive["variance_decomposition"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="variance_decomposition", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "coef_heatmap" in figs_predictive:
        saved["predictive"]["coef_heatmap"] = _save(
            figs_predictive["coef_heatmap"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="coef_heatmap", fmt=fmt, dpi=dpi, sub_case=sub_case
        )

    if "coef_heatmap_core" in figs_predictive:
        saved["predictive"]["coef_heatmap_core"] = _save(
            figs_predictive["coef_heatmap_core"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="coef_heatmap_core", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "coef_heatmap_perf" in figs_predictive:
        saved["predictive"]["coef_heatmap_perf"] = _save(
            figs_predictive["coef_heatmap_perf"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="coef_heatmap_perf", fmt=fmt, dpi=dpi, sub_case=sub_case
        )

    if "mediation" in figs_predictive:
        saved["predictive"]["attribution_ratio"] = _save(
            figs_predictive["mediation"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="attribution_ratio", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "error_landscape" in figs_predictive:
        saved["predictive"]["error_landscape"] = _save(
            figs_predictive["error_landscape"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="error_landscape", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "coef_forest" in figs_predictive:
        saved["predictive"]["coef_forest"] = _save(
            figs_predictive["coef_forest"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="coef_forest", fmt=fmt, dpi=dpi, sub_case=sub_case
        )
    if "associational_network" in figs_predictive or "network" in figs_predictive:
        key = "associational_network" if "associational_network" in figs_predictive else "network"
        saved["predictive"]["associational_network"] = _save(
            figs_predictive[key], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="associational_network", fmt=fmt, dpi=dpi, sub_case=sub_case
        )

    if "safe_boldness_traits" in figs_predictive:
        saved["predictive"]["safe_boldness_traits"] = _save(
            figs_predictive["safe_boldness_traits"], case, plots_root, strategy, stage,
            subdir=subdirs.get("predictive","predictive"),
            filename="safe_boldness_traits", fmt=fmt, dpi=dpi, sub_case=sub_case
        )

    return saved



def run_full_tier3_analysis(
    merged_df,
    person_set: PersonSet,
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = None, 
    n_folds=5,
    perf_df: Optional[pd.DataFrame] = None,
    plots_root: str = "results/figs",
    strategy: str = "not_defined",
    stage: str = "tier3",
    per_figure_subdirs: Optional[Dict[str, str]] = None,
    cv_lmg: bool = True,
    save: bool = True,
    fmt: str = "pdf",
    dpi: int = 200,
    perm_iters: int = 10000,
    perm_early_stop_alpha: Optional[float] = None,
    sub_case: Optional[str] = None
):
    print("Executive Tier 3 Analysis Pipeline")
    print("=" * 80)

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

    print(f"Cross-validation folds: {n_folds}")


    if per_figure_subdirs is None:
        per_figure_subdirs = {
            "consistency_boldness": "consistency_boldness",
            "predictive": "predictive",
        }


    base_dir = resolve_plot_dir(case, plots_root=plots_root, strategy=strategy, stage=stage, sub_case=sub_case)
    if save:
        os.makedirs(base_dir, exist_ok=True)


    print("\n=== Running Consistency vs Boldness Analysis ===")
    try:
        consistency_results = consistency_vs_boldness_analysis(
            merged_df,
            case=case,
            n_folds=n_folds,
            person_set=person_set,
            group_keys=group_keys,
            perf_df=perf_df,
            perm_iters=perm_iters,
            perm_early_stop_alpha=perm_early_stop_alpha,
        )
        print("Consistency analysis completed successfully")
    except Exception as e:
        print(f"ERROR in consistency analysis: {e}")
        consistency_results = {'error': str(e)}


    print("\n=== Running Predictive Attribution (associational) ===")
    try:
        predictive_results = predictive_attribution_modeling(
            merged_df,
            person_set=person_set,
            case=case,
            group_keys=group_keys,
            consistency_results=(consistency_results if 'error' not in consistency_results else None),
            perf_df=perf_df,
            cv_lmg=cv_lmg,
        )
        print("Predictive modeling completed successfully")
    except Exception as e:
        print(f"ERROR in predictive modeling: {e}")
        predictive_results = {'error': str(e)}


    print("\n=== Creating Consistency Analysis Visualizations ===")
    consistency_viz = None
    if 'error' not in consistency_results:
        try:
            consistency_viz = plot_consistency_boldness_analysis(
                consistency_results, person_set=person_set, group_keys=group_keys
            )
            print("Consistency visualizations created successfully")
        except Exception as e:
            print(f"WARNING: Could not create consistency visualizations: {e}")

    print("\n=== Creating Predictive Model Visualizations ===")
    predictive_viz = None
    if 'error' not in predictive_results:
        try:
            predictive_viz = visualize_predictive_attribution(predictive_results)
            print("Predictive visualizations created successfully")
        except Exception as e:
            print(f"WARNING: Could not create predictive visualizations: {e}")


    saved_paths = {}
    if save:
        try:
            saved_paths = save_tier3_outputs(
                consistency_results=consistency_results if 'error' not in consistency_results else {},
                predictive_association_results=predictive_results if 'error' not in predictive_results else {},
                case=case,
                plots_root=plots_root,
                strategy=strategy,
                stage=stage,
                subdirs=per_figure_subdirs,
                fmt=fmt,
                dpi=dpi,
                person_set=person_set,
                group_keys=group_keys,
                figs_consistency=(consistency_viz or {}) if 'error' not in consistency_results else {},
                figs_predictive=(predictive_viz or {})   if 'error' not in predictive_results else {},
                sub_case=sub_case
            )
            print("\nSaved tier3 figures:")
            for group, items in saved_paths.items():
                for name, p in items.items():
                    print(f"  [{group}] {name}: {p}")
        except Exception as e:
            print(f"WARNING: Saving failed: {e}")

        try:
            for _f in (consistency_viz or {}).values():
                plt.close(_f)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("Tier 3 Theoretical Integration")
    print("=" * 80)

    consistency_insight = (consistency_results.get('normative_assessment', {}) if 'error' not in consistency_results else {})
    high_vol_rescue = consistency_insight.get('overall', {}).get('high_volatility_rescue', 0)
    low_vol_rescue  = consistency_insight.get('overall', {}).get('low_volatility_rescue', 0)

    strongest_predictors = (predictive_results.get('strongest_predictors', {}) if 'error' not in predictive_results else {})
    error_detection = (predictive_results.get('error_detection', {}) if 'error' not in predictive_results else {})

    if save and 'error' not in predictive_results:
        print("\n=== Saving PSI Data for Paper Figures ===")
        try:
            save_predictive_demo_r2_summary(
                predictive_results.get('predictive_association_results', {}),
                base_dir
            )
        except Exception as e:
            print(f"WARNING: Failed to save PSI summary: {e}")

    print("\n=== Theoretical Insights:")
    if high_vol_rescue and low_vol_rescue:
        if high_vol_rescue > low_vol_rescue + 0.01:
            vol_conclusion = "Inconsistent profiles provide higher moral value through increased rescue rates"
            vol_implication = "Controlled inconsistency in AI systems may be normatively justified"
        else:
            vol_conclusion = "Consistent profiles match or outperform inconsistent ones in moral value"
            vol_implication = "Consistency should be prioritized over boldness in AI systems"
        print(f"   - {vol_conclusion}\n   - Implication: {vol_implication}\n   - Evidence: {high_vol_rescue:.3f} vs {low_vol_rescue:.3f}")
    else:
        vol_conclusion = "Insufficient data for consistency-boldness analysis"
        print(f"   - {vol_conclusion}")

    if strongest_predictors:
        rescue_pred = strongest_predictors.get('rescue_rate', {}).get('predictor')
        accuracy_pred = strongest_predictors.get('accuracy', {}).get('predictor')
        print(f"   - Strongest rescue predictor: {rescue_pred or 'None'}")
        print(f"   - Strongest accuracy predictor: {accuracy_pred or 'None'}")

    if error_detection:
        dom_flags = sum(1 for d in error_detection.values() if d.get('flag_dominant'))
        print(f"   - Demographic dominance flags (CI>1): {dom_flags}/{len(error_detection)} outcomes")

    print(f"\n{'=' * 60}\nConclusions\n{'=' * 60}")
    conclusions = []
    if high_vol_rescue and low_vol_rescue:
        if high_vol_rescue > low_vol_rescue + 0.02:
            conclusions.append({'finding': 'Inconsistent profiles provide superior moral value',
                                'evidence': f'{high_vol_rescue:.3f} vs {low_vol_rescue:.3f} rescue rates',
                                'implication': 'Controlled inconsistency can improve moral outcomes',
                                'strength': 'Strong' if high_vol_rescue > low_vol_rescue + 0.05 else 'Moderate'})
        else:
            conclusions.append({'finding': 'Consistency and inconsistency yield similar moral value',
                                'evidence': f'Diff: {abs(high_vol_rescue - low_vol_rescue):.3f}',
                                'implication': 'Prefer consistency for predictability',
                                'strength': 'Moderate'})

    print(f"\n{'=' * 60}\nMethodological Assessment\n{'=' * 60}")
    method_assessment = {
        'data_quality': 'Good' if len(merged_df) > 100 else 'Limited',
        'statistical_power': 'Adequate' if 'error' not in consistency_results and 'error' not in predictive_results else 'Insufficient',
        'error_detection_capability': 'Strong' if error_detection else 'Limited',
        'generalizability': 'Domain-specific' if person_set else 'Unknown'
    }
    for k, v in method_assessment.items():
        print(f"   - {k.replace('_',' ').title()}: {v}")

    valid_profiles = consistency_results.get('valid_profiles', []) if 'error' not in consistency_results else []
    if group_keys:
        print(f"   - Profile coverage: {len(valid_profiles)},  Trait coverage: {len(group_keys)} ({', '.join(group_keys)})")

    return {
        'consistency_analysis': consistency_results,
        'predictive_analysis': predictive_results,
        'visualizations': {'consistency_plot': consistency_viz, 'predictive_plot': predictive_viz},
        'figure_paths': saved_paths,         
        'saved_to': base_dir if save else None,
        'theoretical_integration': {'volatility_conclusion': vol_conclusion if 'vol_conclusion' in locals() else None,
                                    'predictive_mechanisms': strongest_predictors,
                                    'error_assessment': error_detection},
        'thesis_conclusions': conclusions,
        'methodological_assessment': method_assessment,
        'analysis_parameters': {'group_keys': group_keys, 'n_folds': n_folds, 'dataset_size': len(merged_df)}
    }
