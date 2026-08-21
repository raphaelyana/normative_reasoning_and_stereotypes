from __future__ import annotations
import os, glob
from typing import Dict, Iterable, Optional, Sequence, Tuple, NamedTuple, Any
import numpy as np
import pandas as pd
from scipy.stats import t

class TokenPricing(NamedTuple):
    prompt_per_1k: float
    completion_per_1k: float

def _safe_mean(x):
    x = pd.Series(x).dropna()
    return float(x.mean()) if len(x) else float("nan")

def _per_profile_accuracy(merged_df: pd.DataFrame) -> pd.Series:
    profs = [c for c in merged_df.columns if str(c).startswith("profile")]
    return pd.Series(
        {p: (merged_df[p].astype(str) == merged_df["true_label"].astype(str)).mean() for p in profs},
        name="accuracy"
    )

def _gather_tokens_from_merged(merged_df: pd.DataFrame, profile: str) -> Optional[pd.DataFrame]:
    """
    Expects wide-style per-profile token cols in merged_df:
      prompt_tokens__profileXX, completion_tokens__profileXX
    Returns DF with columns [prompt_tokens, completion_tokens], or None.
    """
    pt = f"prompt_tokens__{profile}"
    ct = f"completion_tokens__{profile}"
    if pt in merged_df.columns and ct in merged_df.columns:
        return merged_df[[pt, ct]].rename(columns={pt: "prompt_tokens", ct: "completion_tokens"})
    return None

def _find_baseline_tokens(merged_df: pd.DataFrame, candidate_keys=None):
    """
    Try to locate baseline token columns in wide style:
      prompt_tokens__<key>, completion_tokens__<key>
    Returns (pt_series, ct_series, key) or None.
    """
    if candidate_keys is None:
        candidate_keys = ("base_pred", "zero_shot", "baseline", "base")
    for key in candidate_keys:
        pt = f"prompt_tokens__{key}"
        ct = f"completion_tokens__{key}"
        if pt in merged_df.columns and ct in merged_df.columns:
            return merged_df[pt].astype(float), merged_df[ct].astype(float), key
    return None

def attach_accuracy(
    token_df: pd.DataFrame,
    merged_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds per-profile accuracy (and #samples) computed from merged_df (wide, with true_label).
    """
    df = token_df.copy()
    prof_cols = [c for c in merged_df.columns if str(c).startswith("profile")]
    y_true = merged_df["true_label"].astype(str)

    acc = {p: float((merged_df[p].astype(str) == y_true).mean()) for p in prof_cols}
    n_items = len(merged_df)

    # Align to the index's 'profile' level
    df["accuracy"] = pd.Series(acc).reindex(df.index.get_level_values("profile")).values
    df["n_items"] = n_items  # total items for each profile
    return df

def add_cost_per_correct(token_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      cost_per_correct = cost_total / (#correct)
    Works with either 'n_rows' (file aggregation) or 'n_items' (merged path).
    """
    df = token_df.copy()
    if "cost_total" not in df or "accuracy" not in df:
        return df

    if "n_rows" in df:
        denom = (df["accuracy"]*df["n_rows"]).replace(0, np.nan)
    elif "n_items" in df:
        denom = (df["accuracy"]*df["n_items"]).replace(0, np.nan)
    else:
        return df

    df["cost_per_correct"] = df["cost_total"] / denom
    return df


def add_risk_adjusted_efficiency(
    df: pd.DataFrame,
    rescue_col: str = "rescue_rate",
    extra_err_col: str = "extra_err_rate",
    tokens_mean_col: str = "tokens_per_sample_mean",
    lam: float = 1.0,
    out_col: str = "rae_lambda"
) -> pd.DataFrame:
    """
    (7) RAE = (Rescue - ExtraError) / \bar{T}
    """
    out = df.copy()
    if all(c in out for c in [rescue_col, extra_err_col, tokens_mean_col]):
        num = out[rescue_col] - lam * out[extra_err_col]
        out[out_col] = num / out[tokens_mean_col].replace(0, np.nan)
    return out



def compute_token_analysis(
    merged_df: pd.DataFrame,
    pricing: Optional[TokenPricing] = None,
    rescue_stats_all: Optional[Dict[str, pd.DataFrame]] = None,
    rae_lambda: float = 2.0,
    baseline_col: str = "base_pred",
) -> Dict[str, Any]:
    """
    Builds per-profile token & cost table + overall summary.

    Returns:
      {
        "per_profile": DataFrame[
           profile, n_calls, accuracy, n_correct,
           prompt_tokens_mean, completion_tokens_mean, tokens_per_sample,
           total_prompt_tokens, total_completion_tokens, total_tokens,
           cost_per_sample, total_cost,
           efficiency_acc_per_1k_tokens,
           (optional) rescue_rate, extra_err_rate, RAE_lambda,
           (optional) delta_cost_per_accuracy_point_vs_baseline
        ],
        "overall": {...}
      }
    """
    profiles = [c for c in merged_df.columns if str(c).startswith("profile")]
    y_true = merged_df["true_label"].astype(str) if "true_label" in merged_df.columns else None

    # Baseline accuracy (if available)
    if baseline_col in merged_df.columns and y_true is not None:
        baseline_acc = (merged_df[baseline_col].astype(str) == y_true).mean()
    else:
        baseline_acc = np.nan

    # Baseline tokens (if available)
    baseline_tokens = _find_baseline_tokens(
        merged_df,
        candidate_keys=(baseline_col, "base_pred", "zero_shot", "baseline", "base")
    )
    if baseline_tokens is not None:
        base_pt, base_ct, base_key = baseline_tokens
        base_tokens_ps = (base_pt.add(base_ct)).mean()
        base_cost_ps = ((base_pt.mean()/1000.0)*pricing.prompt_per_1k +
                        (base_ct.mean()/1000.0)*pricing.completion_per_1k) if pricing else np.nan
    else:
        base_tokens_ps = np.nan
        base_cost_ps = np.nan

    rows = []
    total_tokens_all = 0.0
    total_cost_all = 0.0
    total_correct_all = 0
    total_calls_all = 0

    for p in profiles:
        preds = merged_df[p].astype(str)
        n_calls = int(preds.notna().sum())

        if y_true is not None:
            acc = (preds == y_true).mean()
            n_correct = int((preds == y_true).sum())
        else:
            acc = np.nan
            n_correct = 0

        tok_df = _gather_tokens_from_merged(merged_df, p)
        if tok_df is not None and not tok_df.empty:
            pt_mean = float(tok_df["prompt_tokens"].mean())
            ct_mean = float(tok_df["completion_tokens"].mean())
            pt_sum  = float(tok_df["prompt_tokens"].sum())
            ct_sum  = float(tok_df["completion_tokens"].sum())
        else:
            pt_mean = ct_mean = pt_sum = ct_sum = np.nan

        tokens_ps = (pt_mean + ct_mean) if np.isfinite(pt_mean) and np.isfinite(ct_mean) else np.nan
        total_tokens = (pt_sum + ct_sum) if np.isfinite(pt_sum) and np.isfinite(ct_sum) else np.nan

        if pricing and np.isfinite(tokens_ps):
            cost_ps = (pt_mean/1000.0)*pricing.prompt_per_1k + (ct_mean/1000.0)*pricing.completion_per_1k
            total_cost = (pt_sum/1000.0)*pricing.prompt_per_1k + (ct_sum/1000.0)*pricing.completion_per_1k
        else:
            cost_ps = np.nan
            total_cost = np.nan

        eff_1k = (100.0*acc/tokens_ps*1000.0) if (np.isfinite(acc) and np.isfinite(tokens_ps) and tokens_ps > 0) else np.nan

        if np.isfinite(baseline_acc) and np.isfinite(acc) and (acc != baseline_acc):
            if pricing and np.isfinite(cost_ps) and np.isfinite(base_cost_ps):
                delta_cost_per_acc_pt = (cost_ps - base_cost_ps) / (100.0*(acc - baseline_acc))
            elif np.isfinite(tokens_ps) and np.isfinite(base_tokens_ps):
                delta_cost_per_acc_pt = (tokens_ps - base_tokens_ps) / (100.0*(acc - baseline_acc))
            else:
                delta_cost_per_acc_pt = np.nan
        else:
            delta_cost_per_acc_pt = np.nan

        rows.append({
            "profile": p,
            "n_calls": n_calls,
            "accuracy": float(acc) if np.isfinite(acc) else np.nan,
            "n_correct": n_correct,
            "prompt_tokens_mean": pt_mean,
            "completion_tokens_mean": ct_mean,
            "tokens_per_sample": tokens_ps,
            "total_prompt_tokens": pt_sum,
            "total_completion_tokens": ct_sum,
            "total_tokens": total_tokens,
            "cost_per_sample": cost_ps,
            "total_cost": total_cost,
            "efficiency_acc_per_1k_tokens": eff_1k,
            "delta_cost_per_accuracy_point_vs_baseline": delta_cost_per_acc_pt,
        })

        if np.isfinite(total_tokens): total_tokens_all += total_tokens
        if np.isfinite(total_cost): total_cost_all += total_cost
        total_correct_all += n_correct
        total_calls_all += n_calls

    per_profile = pd.DataFrame(rows)

    if rescue_stats_all:
        rs_list = []
        for df in rescue_stats_all.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sub = df[["profile","rescue_rate","extra_err_rate","N_cat"]].copy()
                rs_list.append(sub)
        if rs_list:
            rs = pd.concat(rs_list, ignore_index=True)
            grp = rs.groupby("profile").apply(
                lambda d: pd.Series({
                    "rescue_rate": np.average(d["rescue_rate"], weights=d["N_cat"]),
                    "extra_err_rate": np.average(d["extra_err_rate"], weights=d["N_cat"]),
                    "N_weight": d["N_cat"].sum()
                })
            ).reset_index()
            per_profile = per_profile.merge(grp, on="profile", how="left")
            per_profile["RAE_lambda"] = (
                (per_profile["rescue_rate"] - rae_lambda*per_profile["extra_err_rate"])
                / per_profile["tokens_per_sample"].replace(0, np.nan)
            )

    overall = {
        "total_calls": int(total_calls_all),
        "total_tokens": float(total_tokens_all) if total_calls_all > 0 else np.nan,
        "tokens_per_sample_mean": float(per_profile["tokens_per_sample"].mean(skipna=True)) if len(per_profile) else np.nan,
        "total_cost": float(total_cost_all) if pricing else np.nan,
        "cost_per_sample_mean": float(per_profile["cost_per_sample"].mean(skipna=True)) if pricing else np.nan,
        "total_correct": int(total_correct_all),
        "cost_per_correct": (total_cost_all / total_correct_all) if (pricing and total_correct_all > 0) else np.nan,
        "baseline_accuracy": float(baseline_acc) if np.isfinite(baseline_acc) else np.nan,
    }

    return {"per_profile": per_profile, "overall": overall}



def find_profile_csv(
    base_dir: str,
    model: str,
    mode: str,
    role: str,
    person_key: str,
    pattern: str = "results_mmlu_*.csv",
) -> Optional[str]:
    """
    Returns the most recent results CSV path for a given profile key, or None.
    """
    root = os.path.join(base_dir, model, mode, "role_playing_ethnics", f"{person_key}_{role}")
    candidates = glob.glob(os.path.join(root, pattern))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _read_tokens_csv(path: str) -> pd.DataFrame:
    """
    Reads just the token columns if present; returns an empty DF if missing.
    Accepts extra columns (ignored).
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    needed = {"prompt_tokens", "completion_tokens", "tokens_used", "max_tokens"}
    present = [c for c in df.columns if c in needed]
    return df[present] if present else pd.DataFrame()


def collect_token_usage_from_files(
    profiles: Sequence[str],
    base_dir: str,
    model: str,
    mode: str,
    role: str,
    groupby_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Aggregates token metrics per profile (optionally per [fold, seed] if columns exist).
    Returns a DF indexed by 'profile' (and groupby_cols if provided) with:

      # Primary token metrics
      - tokens_per_sample_mean   = E[prompt+completion]
      - tokens_total_sum         = sum(prompt+completion)
      - prompt_tokens_mean/sum
      - completion_tokens_mean/sum
      - tokens_used_mean/sum
      - max_tokens_mean
      - n_rows                   = #samples read for this (profile,group)

      # Diagnostics
      - tokens_gap_sum           = tokens_used_sum - (prompt_sum + completion_sum)

    Notes:
      • If CSVs contain group columns (e.g., 'fold','seed'), pass groupby_cols to get per-group aggregates.
      • If not present, we still aggregate over all rows.
    """
    rows = []
    for pk in profiles:
        path = find_profile_csv(base_dir, model, mode, role, pk)
        if path is None:
            continue
        df = _read_tokens_csv(path)
        if df.empty:
            continue

        df["_tokens_sum_pc"] = df.get("prompt_tokens", 0) + df.get("completion_tokens", 0)

        gcols = [c for c in (groupby_cols or []) if c in df.columns]
        if gcols:
            g = df.groupby(gcols, dropna=False)
        else:
            df["_const"] = 1
            g = df.groupby("_const")

        agg = g.agg(
            prompt_tokens_mean=("prompt_tokens", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
            tokens_used_mean=("tokens_used", "mean"),
            max_tokens_mean=("max_tokens", "mean"),
            prompt_tokens_sum=("prompt_tokens", "sum"),
            completion_tokens_sum=("completion_tokens", "sum"),
            tokens_used_sum=("tokens_used", "sum"),
            tokens_pc_sum=("_tokens_sum_pc", "sum"),
            tokens_pc_mean=("_tokens_sum_pc", "mean"),
            n_rows=("tokens_used", "size"),
        ).reset_index(drop=False)

        agg.insert(0, "profile", pk)
        agg["tokens_per_sample_mean"] = agg["tokens_pc_mean"]
        agg["tokens_total_sum"] = agg["tokens_pc_sum"]
        agg["tokens_gap_sum"] = agg["tokens_used_sum"]-agg["tokens_pc_sum"]

        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    index_cols = ["profile"] + [c for c in (groupby_cols or []) if c in out.columns]
    out = out.set_index(index_cols).sort_index()
    preferred = [
        "tokens_per_sample_mean", "tokens_total_sum",
        "prompt_tokens_mean", "completion_tokens_mean",
        "tokens_used_mean", "max_tokens_mean",
        "prompt_tokens_sum", "completion_tokens_sum",
        "tokens_used_sum", "n_rows", "tokens_gap_sum"
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def attach_accuracy(
    token_df: pd.DataFrame,
    merged_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds per-profile accuracy (and #samples) computed from merged_df (wide, with true_label).
    """
    token_df = token_df.copy()
    prof_cols = [c for c in merged_df.columns if str(c).startswith("profile")]
    acc = {
        p: float((merged_df[p].astype(int) == merged_df["true_label"].astype(int)).mean())
        for p in prof_cols
    }
    n = len(merged_df)
    token_df["accuracy"] = pd.Series(acc).reindex(token_df.index.get_level_values("profile")).values
    token_df["n_items"]=n
    return token_df


def compute_cost_columns(
    token_df: pd.DataFrame,
    price_per_1k: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Adds $ costs (optional):
      cost_sample_mean = (\bar{T_in}/1000)*p_in + (\bar{T_out}/1000)*p_out     (3)
      cost_total       = (sum_in/1000)*p_in + (sum_out/1000)*p_out
    If price_per_1k is None, returns df unchanged (token-only workflow).
    """
    if not price_per_1k:
        return token_df

    pr = float(price_per_1k.get("prompt", 0.0))
    cr = float(price_per_1k.get("completion", 0.0))
    df = token_df.copy()

    if "prompt_tokens_mean" in df and "completion_tokens_mean" in df:
        df["cost_sample_mean"] = (df["prompt_tokens_mean"]/1000.0)*pr + (df["completion_tokens_mean"]/1000.0)*cr

    if "prompt_tokens_sum" in df and "completion_tokens_sum" in df:
        df["cost_total"] = (df["prompt_tokens_sum"]/1000.0)*pr + (df["completion_tokens_sum"]/1000.0)*cr

    return df


def add_efficiency_columns(
    token_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds:
      accuracy_per_1k_tokens = 100 * accuracy / tokens_per_sample_mean          (6)  (%-points per 1k tok if tokens_per_sample_mean is tokens)
      tokens_per_correct     = tokens_per_sample_mean / accuracy
      (robust to accuracy==0 -> NaN)
    """
    df = token_df.copy()
    if "accuracy" in df and "tokens_per_sample_mean" in df:
        df["accuracy_per_1k_tokens"] = 100.0 * df["accuracy"] / df["tokens_per_sample_mean"].replace(0, np.nan)
        df["tokens_per_correct"] = df["tokens_per_sample_mean"] / df["accuracy"].replace(0, np.nan)
    return df


def add_cost_per_correct(
    token_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds:
      cost_per_correct = cost_total / (#correct)
    Requires cost_total + n_rows + accuracy (to get #correct).
    """
    df = token_df.copy()
    if all(c in df for c in ["cost_total", "n_rows", "accuracy"]):
        correct = (df["accuracy"] * df["n_rows"]).replace(0, np.nan)
        df["cost_per_correct"] = df["cost_total"] / correct
    return df


def marginal_cost_per_accuracy_point(
    row_A: pd.Series,
    row_B: pd.Series,
    use_total_cost: bool = True
) -> float:
    """
    (5) ΔCost/ΔAcc between A→B, where Acc is in [0,1].
       If use_total_cost=True: (C_B - C_A) / (100 * (Acc_B - Acc_A))
       Else: uses cost_sample_mean instead of cost_total.
       Units: $ per percentage point (pp) of accuracy.

       Returns np.inf if denominator is ~0 or if costs missing.
    """
    cost_field = "cost_total" if use_total_cost else "cost_sample_mean"
    if cost_field not in row_A or cost_field not in row_B or "accuracy" not in row_A or "accuracy" not in row_B:
        return np.nan
    dacc = (row_B["accuracy"] - row_A["accuracy"]) * 100.0
    dcost = (row_B[cost_field] - row_A[cost_field])
    if abs(dacc) < 1e-12:
        return np.inf
    return float(dcost / dacc)


def add_risk_adjusted_efficiency(
    df: pd.DataFrame,
    rescue_col: str = "rescue_rate",
    extra_err_col: str = "extra_error_rate",
    tokens_mean_col: str = "tokens_per_sample_mean",
    lam: float = 1.0,
    out_col: str = "rae_lambda"
) -> pd.DataFrame:
    """
    (7) RAE_λ = (Rescue - λ·ExtraError) / \bar{T}
       Expects rates in [0,1] and tokens_mean_col as tokens per sample.
       Adds column <out_col>.
    """
    out = df.copy()
    if all(c in out for c in [rescue_col, extra_err_col, tokens_mean_col]):
        num = out[rescue_col] - lam * out[extra_err_col]
        out[out_col] = num / out[tokens_mean_col].replace(0, np.nan)
    return out


def build_profile_cost_table(
    merged_df: pd.DataFrame,
    base_dir: str,
    model: str,
    mode: str,
    role: str,
    profiles: Optional[Sequence[str]] = None,
    price_per_1k: Optional[Dict[str, float]] = None,
    groupby_cols: Optional[Sequence[str]] = None,
    rescue_extra_df: Optional[pd.DataFrame] = None,
    rescue_key_map: Tuple[str, str, str] = ("profile", "rescue_rate", "extra_err_rate"),
    rae_lambda: Optional[float] = None,
) -> pd.DataFrame:
    """
    One-shot builder that returns a tidy per-profile table with:
      accuracy, tokens_per_sample_mean, tokens_total_sum,
      (optional $) cost_sample_mean, cost_total, cost_per_correct,
      efficiency: accuracy_per_1k_tokens, tokens_per_correct,
      (optional) rescue_rate, extra_error_rate, and RAE_λ if rae_lambda provided.

    Arguments:
      - rescue_extra_df: optional DF with per-profile rescue/extra-error rates.
         Should contain columns [key, rescue_rate, extra_error_rate], where key defaults to "profile".
    """
    if profiles is None:
        profiles = [c for c in merged_df.columns if str(c).startswith("profile")]

    tok = collect_token_usage_from_files(
        profiles=profiles, base_dir=base_dir, model=model, mode=mode, role=role, groupby_cols=groupby_cols
    )
    if tok.empty:
        return tok

    tok = attach_accuracy(tok, merged_df)
    tok = compute_cost_columns(tok, price_per_1k=price_per_1k)
    tok = add_efficiency_columns(tok)
    tok = add_cost_per_correct(tok)

    if rescue_extra_df is not None:
        k, rk, ek = rescue_key_map
        if k in rescue_extra_df.columns:
            tok = tok.reset_index()
            tok = tok.merge(
                rescue_extra_df[[k, rk, ek]],
                left_on="profile", right_on=k, how="left"
            ).drop(columns=[k])
            tok = tok.set_index([c for c in tok.columns if c in tok.columns and c in tok.select_dtypes(include=["object"]).columns and c in ["profile"]]+
                                [c for c in tok.columns if c in (groupby_cols or [])])
            tok = tok.sort_index()
            if rae_lambda is not None:
                tok = add_risk_adjusted_efficiency(tok.reset_index(), rescue_col=rk, extra_err_col=ek,
                                                   tokens_mean_col="tokens_per_sample_mean",
                                                   lam=float(rae_lambda), out_col=f"rae_lambda_{rae_lambda:g}").set_index(tok.index.names)

    preferred = [
        "accuracy",
        "tokens_per_sample_mean", "tokens_total_sum",
        "prompt_tokens_mean", "completion_tokens_mean", "tokens_used_mean", "max_tokens_mean",
        "prompt_tokens_sum", "completion_tokens_sum", "tokens_used_sum", "tokens_gap_sum", "n_rows",
        "accuracy_per_1k_tokens", "tokens_per_correct",
        "cost_sample_mean", "cost_total", "cost_per_correct",
    ]
    cols = [c for c in preferred if c in tok.columns] + [c for c in tok.columns if c not in preferred]
    return tok[cols]


def summarize_by_trait(
    profile_table: pd.DataFrame,
    person_set,
    trait: str = "ethnicity",
    ci: float = 0.95,
    value_cols: Sequence[str] = ("accuracy", "tokens_per_sample_mean", "accuracy_per_1k_tokens")
) -> pd.DataFrame:
    """
    Aggregates per-profile metrics by a trait (e.g., ethnicity, gender).
    Returns mean and normal-approx CI across profiles for each value_col.

    Output columns: <metric>_mean, <metric>_ci
    """
    prof_to_trait = {}
    for pid, meta in person_set.metadata.items():
        if pid in profile_table.index.get_level_values("profile"):
            v = getattr(meta, trait, None)
            if hasattr(v, "value"):
                v = v.value
            prof_to_trait[pid] = None if v is None else str(v).lower()

    df = profile_table.reset_index().copy()
    df[trait] = df["profile"].map(prof_to_trait).fillna("unknown")
    rows = []
    for grp, gdf in df.groupby(trait, dropna=False):
        row = {trait: grp, "n_profiles": len(gdf)}
        for col in value_cols:
            if col not in gdf.columns:
                continue
            vals = gdf[col].dropna().astype(float).values
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan; row[f"{col}_ci"] = 0.0
            elif len(vals) == 1:
                row[f"{col}_mean"] = float(vals[0]); row[f"{col}_ci"] = 0.0
            else:
                m = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                q = float(t.ppf(0.5 + ci/2, len(vals)-1))
                row[f"{col}_mean"] = m
                row[f"{col}_ci"] = q * se
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(trait).reset_index(drop=True)
