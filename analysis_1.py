import os, re, sys, json
from itertools import combinations
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.stats import norm
from scipy.special import expit, logit

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from patsy import dmatrix

from analysis_tools import get_analysis_group_keys, resolve_plot_dir
from profiles.schema import PersonSet
from cases.cases_config import CaseConfig

from plot_tools import apply_neurips_figure_style, new_pub_fig


def _resolve_group_keys(person_set: PersonSet, group_keys):
    if group_keys is None or (isinstance(group_keys, str) and group_keys.upper() == "AUTO"):
        return tuple(get_analysis_group_keys(person_set))
    return tuple(group_keys)


def _norm_trait(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    return str(v).strip().lower()

def _profile_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("profile") and "__" not in c]

def to_long_roleplay_only(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    group_keys: Tuple[str, ...] = ("gender", "ethnicity")
) -> pd.DataFrame:
    """
    Reshape wide merged_df (profile columns) to long:
      columns required: sample_id, true_label, profileX...
      adds: correct (0/1) and attached traits from person_set
    """
    prof_cols = _profile_cols(merged_df)
    if not {"sample_id", "true_label"}.issubset(merged_df.columns):
        raise ValueError("merged_df must contain 'sample_id' and 'true_label'")

    long = merged_df[["sample_id", "true_label"] + prof_cols].melt(
        id_vars=["sample_id", "true_label"],
        value_vars=prof_cols,
        var_name="profile", value_name="pred"
    )
    long["correct"] = (long["pred"] == long["true_label"]).astype(int)

    trait_map = {}
    for p in prof_cols:
        t = person_set.get_traits(p, group_keys) if person_set is not None else {k: "unknown" for k in group_keys}
        trait_map[p] = {k: _norm_trait(t.get(k, "unknown")) for k in group_keys}
    traits_df = pd.DataFrame.from_dict(trait_map, orient="index").reset_index().rename(columns={"index": "profile"})
    long = long.merge(traits_df, on="profile", how="left")

    long["sample_id"] = long["sample_id"].astype("category")
    for k in group_keys:
        long[k] = long[k].astype("category")
    return long

def drop_saturated(long_df: pd.DataFrame, out_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Drop items/profiles that are perfectly separated (all 0s or all 1s).
    """
    var_by_item = long_df.groupby("sample_id", observed=False)["correct"].var(ddof=0)
    sat_items = set(var_by_item.index[var_by_item.fillna(0.0) == 0.0].tolist())

    var_by_prof = long_df.groupby("profile", observed=False)["correct"].var(ddof=0)
    sat_profiles = set(var_by_prof.index[var_by_prof.fillna(0.0) == 0.0].tolist())

    kept = long_df.loc[~long_df["sample_id"].isin(sat_items) & ~long_df["profile"].isin(sat_profiles)].copy()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"sample_id": sorted(sat_items)}).to_csv(
            os.path.join(out_dir, "dropped_items_saturated.csv"), index=False
        )
        pd.DataFrame({"profile": sorted(sat_profiles)}).to_csv(
            os.path.join(out_dir, "dropped_profiles_saturated.csv"), index=False
        )
    print(f"[Separation guard] items dropped: {len(sat_items)}; profiles dropped: {len(sat_profiles)}; "
          f"rows {len(long_df)} → {len(kept)}")
    return kept

def make_formula(traits: Tuple[str, ...], references: Optional[Dict[str, str]] = None) -> str:
    """
    Build a logistic-GLM formula with optional explicit Treatment references.
    If 'references' is provided, we inject the baseline in each C(...).
    """
    references = references or {}

    def Cref(k: str) -> str:
        ref = references.get(k, None)
        return (f"C({k}, Treatment(reference='{ref}'))" if ref is not None else f"C({k})")

    mains = [Cref(k) for k in traits]
    inters = []
    for a, b in combinations(traits, 2):
        Ca = Cref(a)
        Cb = Cref(b)
        inters.append(f"{Ca}:{Cb}")
    return "correct ~ " + " + ".join(mains + inters)


def _baseline_row_from_model(res):
    df = res.model.data.frame
    endog = getattr(res.model, "endog_names", None)
    row = {}
    for c in df.columns:
        if c == endog:
            continue
        s = df[c]
        if pd.api.types.is_categorical_dtype(s):
            row[c] = s.cat.categories[0]
        elif pd.api.types.is_numeric_dtype(s):
            row[c] = float(pd.to_numeric(s, errors="coerce").mean())
        else:
            vals = pd.unique(s.dropna())
            row[c] = vals[0] if len(vals) else None
    return row

def _exog_from_rows(res, rows: list[dict]) -> np.ndarray:
    """Build design rows using the model's DesignInfo (no string parsing)."""
    DI = res.model.data.design_info
    new = pd.DataFrame(rows)
    fit = res.model.data.frame
    for c in new.columns:
        if c in fit.columns and pd.api.types.is_categorical_dtype(fit[c]):
            new[c] = pd.Categorical(new[c], categories=list(fit[c].cat.categories))
    X = dmatrix(DI, new, return_type="dataframe")
    return np.asarray(X)



def wald_block_tests(res, traits: Tuple[str, ...], n_clusters: Optional[int] = None) -> pd.DataFrame:
    """
    Joint block tests with small-sample F-approximation (when available).
    Reports test type (F), stat, df1, df2).
    """
    if n_clusters is not None and n_clusters < 30:
        print(f"[warn] Only {n_clusters} clusters; using F-approximation for joint tests.")

    df_fit = res.model.data.frame
    base = _baseline_row_from_model(res)
    blocks = []

    def _add_block(label: str, R: np.ndarray):
        wt = res.wald_test(R, use_f=True, scalar=True) 
        stat = float(wt.statistic)
        pval = float(wt.pvalue)
        df_num = int(getattr(wt, "df_num", R.shape[0]) or R.shape[0])
        df_den = getattr(wt, "df_denom", None)
        test = "F" if df_den is not None else "chi2"
        row = {"block": label, "test": test, "stat": stat, "df1": df_num, "p": pval}
        if df_den is not None:
            try:
                row["df2"] = float(df_den)
            except Exception:
                row["df2"] = None
        else:
            row["df2"] = None
        blocks.append(row)

    for k in traits:
        if k not in df_fit.columns or not pd.api.types.is_categorical_dtype(df_fit[k]):
            continue
        levels = list(df_fit[k].cat.categories)
        if len(levels) < 2:
            continue
        R = []
        for lv in levels[1:]:
            r0 = base.copy()
            r1 = base.copy(); r1[k] = lv
            X = _exog_from_rows(res, [r1, r0])
            R.append((X[0] - X[1]))
        R = np.vstack(R)
        _add_block(f"main:C({k})", R)

    for i in range(len(traits)):
        for j in range(i+1, len(traits)):
            a, b = traits[i], traits[j]
            if any(t not in df_fit.columns or not pd.api.types.is_categorical_dtype(df_fit[t]) for t in (a, b)):
                continue
            A = list(df_fit[a].cat.categories)
            B = list(df_fit[b].cat.categories)
            if len(A) < 2 or len(B) < 2:
                continue
            R = []
            for ai in A[1:]:
                for bj in B[1:]:
                    r00 = base.copy()
                    r10 = base.copy(); r10[a] = ai
                    r01 = base.copy(); r01[b] = bj
                    r11 = base.copy(); r11[a] = ai; r11[b] = bj
                    X = _exog_from_rows(res, [r11, r10, r01, r00])
                    L = (X[0] - X[1]) - (X[2] - X[3])
                    R.append(L)
            R = np.vstack(R)
            _add_block(f"inter:C({a}):C({b})", R)

    cols = ["block", "test", "stat", "df1", "df2", "p"]
    return (pd.DataFrame(blocks, columns=cols).sort_values("p").reset_index(drop=True)
            if blocks else pd.DataFrame(columns=cols))


def pairwise_contrasts_fdr(res, factor: str, df_design: pd.DataFrame) -> pd.DataFrame:
    if factor not in df_design.columns or not pd.api.types.is_categorical_dtype(df_design[factor]):
        return pd.DataFrame()
    levels = list(df_design[factor].cat.categories)
    if len(levels) < 2: return pd.DataFrame()

    beta, V = res.params.values, res.cov_params().values
    base_row = _baseline_row_from_model(res)

    out = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            li, lj = levels[i], levels[j]
            r1 = base_row.copy(); r1[factor] = li
            r2 = base_row.copy(); r2[factor] = lj
            X = _exog_from_rows(res, [r1, r2])
            L = (X[0] - X[1]).reshape(1, -1)

            est = float(L @ beta)
            se  = float(np.sqrt(L @ V @ L.T))
            wt  = res.wald_test(L, use_f=True, scalar=True)
            p   = float(wt.pvalue)
            df2 = getattr(wt, "df_denom", np.nan)

            out.append({
                "factor": factor, "level_i": li, "level_j": lj,
                "logit_diff": est, "OR_ratio": np.exp(est),
                "se": se, "stat": float(wt.statistic), "df2": float(df2), "p": p
            })

    tab = pd.DataFrame(out)
    if len(tab):
        tab["q"] = multipletests(tab["p"].values, method="fdr_bh")[1]
        tab = tab.sort_values(["q", "p", "factor", "level_i", "level_j"]).reset_index(drop=True)
    return tab


def _set_ordered_categories(
    long: pd.DataFrame,
    group_keys: Tuple[str, ...],
    reference_levels: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, list]]:
    """
    Ensure each group factor is categorical with desired ordering and a clear baseline.
    Moves 'unknown' (case-insensitive) to the end if present.
    Returns (long_df, chosen_refs, ordered_levels).
    """
    reference_levels = reference_levels or {}
    chosen_refs: Dict[str, str] = {}
    ordered_levels: Dict[str, list] = {}

    for k in group_keys:
        if k not in long.columns:
            continue

        if not pd.api.types.is_categorical_dtype(long[k]):
            long[k] = long[k].astype("category")

        levels = list(long[k].cat.categories.astype(str))

        if k.lower() == "age":
            try:
                levels_sorted = sorted(levels, key=lambda s: int(str(s)))
            except Exception:
                levels_sorted = levels[:]
        else:
            levels_sorted = levels[:]

        unk = [lv for lv in levels_sorted if lv.lower() == "unknown"]
        levels_sorted = [lv for lv in levels_sorted if lv.lower() != "unknown"] + unk

        ref = reference_levels.get(k)
        if ref is None:
            defaults = {"gender": "man", "ethnicity": "white", "age": "20"}
            ref = defaults.get(k, levels_sorted[0] if levels_sorted else None)
            if ref not in levels_sorted and levels_sorted:
                ref = levels_sorted[0]
        else:
            if ref not in levels_sorted and levels_sorted:
                ref = levels_sorted[0]

        final_levels = [ref] + [lv for lv in levels_sorted if lv != ref] if levels_sorted else []

        long[k] = pd.Categorical(long[k].astype(str), categories=final_levels, ordered=False)
        chosen_refs[k] = ref
        ordered_levels[k] = final_levels

    return long, chosen_refs, ordered_levels



def _persist_glm_metadata(
    out_dir: Optional[str],
    *,
    formula: str,
    factors_levels: Dict[str, list],
    references: Dict[str, str],
    cluster_on: str,
    n_rows: int,
    n_items: int,
    n_profiles: int,
    n_clusters: int,
    seed: Optional[int] = None,
    filename: str = "glm_metadata.json",
) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "formula": formula,
        "factors": {
            k: {"levels": list(v), "reference": references.get(k)}
            for k, v in factors_levels.items()
        },
        "cluster_on": cluster_on,
        "n_rows": int(n_rows),
        "n_items": int(n_items),
        "n_profiles": int(n_profiles),
        "n_clusters": int(n_clusters),
        "seed": seed,
        "notes": "Cluster-robust SEs; binomial GLM.",
        "versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
        },
    }
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(meta, f, indent=2)




def fit_tier1_profiles_only(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    group_keys: Optional[Tuple[str, ...]] = ("gender", "ethnicity"),
    cluster_on: str = "sample_id",
    out_dir: Optional[str] = None,
    apply_separation_guard: bool = False,
    reference_levels: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    Binomial GLM on profile rows only, with specified demographic traits.
    Clustered SEs on 'cluster_on'. No item fixed effects.
    """
    group_keys = _resolve_group_keys(person_set, group_keys)
    long = to_long_roleplay_only(merged_df, person_set, group_keys=group_keys)

    if apply_separation_guard:
        long = drop_saturated(long, out_dir=out_dir)
    if len(long) == 0:
        raise ValueError("No rows left to fit after preprocessing.")

    long, chosen_refs, ordered_levels = _set_ordered_categories(
        long, group_keys=group_keys, reference_levels=reference_levels
    )

    n_clusters = long[cluster_on].nunique()

    kept = tuple(k for k in group_keys if k in long and long[k].nunique(dropna=True) >= 2)
    if not kept:
        raise ValueError("No varying traits among profiles.")

    formula = make_formula(kept, references=chosen_refs)

    if cluster_on not in ("sample_id", "profile"):
        cluster_on = "sample_id"
    if long[cluster_on].nunique() < 2:
        cov_kw = None
        cov_type = "nonrobust"
    else:
        cov_kw = {"groups": long[cluster_on], "use_correction": True, "df_correction": True}
        cov_type = "cluster"

    model = smf.glm(formula, data=long, family=sm.families.Binomial())
    try:
        res = model.fit(cov_type=cov_type, cov_kwds=cov_kw)
    except np.linalg.LinAlgError:
        res = model.fit_regularized(alpha=1e-6, L1_wt=0.0)
        res.cov_params = lambda: pd.DataFrame(np.nan, index=res.params.index, columns=res.params.index)

    summ = res.summary2().tables[1].copy()
    summ["term"] = summ.index.astype(str)
    view = summ.reset_index(drop=True)
    if {"Coef.","[0.025","0.975]"}.issubset(view.columns):
        view["OR"] = np.exp(view["Coef."])
        view["OR_ci_low"] = np.exp(view["[0.025"])
        view["OR_ci_high"] = np.exp(view["0.975]"])
    if "p" not in view.columns:
        view["p"] = (view["P>|z|"] if "P>|z|" in view.columns
                     else view["P>|t|"] if "P>|t|" in view.columns
                     else np.nan)

    blocks = wald_block_tests(res, kept, n_clusters=n_clusters)

    _persist_glm_metadata(
        out_dir,
        formula=formula,
        factors_levels=ordered_levels,
        references=chosen_refs,
        cluster_on=cluster_on,
        n_rows=len(long),
        n_items=long["sample_id"].nunique() if "sample_id" in long.columns else np.nan,
        n_profiles=long["profile"].nunique() if "profile" in long.columns else np.nan,
        n_clusters=long[cluster_on].nunique(),
        seed=None,
    )

    print("\n=== Tier-1 (profiles-only) GLM ===")
    print("Formula:", formula)
    print("Baselines:", ", ".join([f"{k}={v}" for k, v in chosen_refs.items() if k in kept]))
    print(f"n_rows={len(long):,} | n_items={long['sample_id'].nunique()} | n_profiles={long['profile'].nunique()}")
    print(f"Clustered on: {cluster_on} (n_clusters={long[cluster_on].nunique()})")
    print("\n-- Block Wald tests --")
    wb = blocks.copy()
    if len(wb):
        def _fmt_row(r):
            if r.get("test") == "F" and pd.notnull(r.get("df2", None)):
                return f"{r['block']}: F({int(r['df1'])}, {int(r['df2'])})={r['stat']:.3g}, p={r['p']:.4g}"
            else:
                return f"{r['block']}: χ²({int(r['df1'])})={r['stat']:.3g}, p={r['p']:.4g}"
        for _, r in wb.iterrows():
            print("   - " + _fmt_row(r))
    else:
        print("(none)")

    return {
        "data": long,
        "formula": formula,
        "result": res,
        "coef_table": view,
        "wald_blocks": blocks,
        "kept_traits": kept,
        "references": chosen_refs,
        "levels": ordered_levels,
    }




def tidy_glm_tables(
    fit: Dict[str, object],
    focus_factors: Iterable[str] = ("gender", "ethnicity"),
) -> Dict[str, pd.DataFrame]:
    res = fit["result"]

    params = res.params.copy()
    conf = res.conf_int()
    if isinstance(conf, np.ndarray):
        conf = pd.DataFrame(conf, index=params.index, columns=["ci_low", "ci_high"])
    else:
        conf = conf.copy()
        conf.columns = ["ci_low", "ci_high"]
    se = res.bse
    if isinstance(se, np.ndarray):
        se = pd.Series(se, index=params.index)
    z = params / se
    p = res.pvalues
    if isinstance(p, np.ndarray):
        p = pd.Series(p, index=params.index)

    coef_tab = pd.concat(
        [params.rename("coef"),
         se.rename("se"),
         z.rename("z"),
         p.rename("p"),
         conf[["ci_low", "ci_high"]]],
        axis=1
    )
    coef_tab["term"] = coef_tab.index.astype(str)
    coef_tab["OR"] = np.exp(coef_tab["coef"])
    coef_tab["OR_ci_low"] = np.exp(coef_tab["ci_low"])
    coef_tab["OR_ci_high"] = np.exp(coef_tab["ci_high"])
    coef_tab = coef_tab.reset_index(drop=True)

    coef_tab_view = coef_tab[~coef_tab["term"].str.startswith("C(sample_id)[")].reset_index(drop=True)
    wald_blocks = fit.get("wald_blocks", pd.DataFrame(columns=["block","test","stat","df1","df2","p"]))
    return {"coef_table": coef_tab_view, "wald_blocks": wald_blocks}



def run_glm_cluster_pipeline(
    merged_df: pd.DataFrame,
    person_set: PersonSet,
    out_root: str,
    group_keys: Optional[Tuple[str, ...]] = ("gender", "ethnicity"),
    cluster_on: str = "sample_id",
    apply_separation_guard: bool = False,
) -> Dict[str, object]:
    """
    Fits the Tier-1 GLM, tidies outputs, runs symmetric pairwise (FDR)
    for each kept factor, and writes CSVs (single canonical set).
    """
    os.makedirs(out_root, exist_ok=True)

    group_keys = _resolve_group_keys(person_set, group_keys)

    fit = fit_tier1_profiles_only(
        merged_df=merged_df,
        person_set=person_set,
        group_keys=group_keys,
        cluster_on=cluster_on,
        out_dir=out_root,  
        apply_separation_guard=apply_separation_guard,
    )

    kept = fit.get("kept_traits", tuple(k for k in group_keys if k in fit["data"].columns))
    tables = tidy_glm_tables(fit, focus_factors=kept)

    tables["coef_table"].to_csv(os.path.join(out_root, "glm_coef_OR.csv"), index=False)
    tables["wald_blocks"].to_csv(os.path.join(out_root, "glm_block_tests.csv"), index=False)

    for fac in kept:
        pw = pairwise_contrasts_fdr(fit["result"], fac, fit["data"])
        if len(pw):
            pw.to_csv(os.path.join(out_root, f"glm_pairwise_{fac}.csv"), index=False)

    print("\n================ GLM (profiles-only, Binomial) ================")
    print("Formula:", fit["formula"])
    print(f"n_rows={len(fit['data']):,}, n_profiles={fit['data']['profile'].nunique()}, "
          f"n_items={fit['data']['sample_id'].nunique()}")
    print(f"Clustered on: {cluster_on}")
    print("\n-- Block Wald tests --")
    print(tables["wald_blocks"].to_string(index=False) if len(tables["wald_blocks"]) else "(none)")

    return {"fit": fit, **tables}


def _predict_prob_ci_marginalized(res, factor: str, levels: list[str], alpha: float = 0.05):
    df_fit = res.model.data.frame.copy()
    DI = res.model.data.design_info
    beta = res.params.values
    V = res.cov_params().values
    z = norm.ppf(1.0 - alpha/2.0)

    cats = (list(df_fit[factor].cat.categories)
            if factor in df_fit.columns and pd.api.types.is_categorical_dtype(df_fit[factor])
            else None)

    means, los, his = [], [], []
    for lv in levels:
        df_new = df_fit.copy()
        if factor in df_new.columns:
            df_new[factor] = (pd.Categorical([lv]*len(df_new), categories=cats)
                              if cats is not None else lv)

        X = dmatrix(DI, df_new, return_type="dataframe").to_numpy()
        eta = X @ beta
        mu  = 1.0/(1.0 + np.exp(-eta))

        m = float(np.mean(mu))                 
        w = (mu * (1 - mu))[:, None]
        g = np.mean(w * X, axis=0)
        var_m = float(max(g @ V @ g, 0.0))
        se_m = np.sqrt(var_m)

        eps = 1e-8
        m_clip = float(np.clip(m, eps, 1 - eps))
        logit_m = float(logit(m_clip))
        se_logit_m = float(se_m / (m_clip * (1 - m_clip)))
        lo = float(expit(logit_m - z * se_logit_m))
        hi = float(expit(logit_m + z * se_logit_m))

        means.append(m); los.append(lo); his.append(hi)
    return np.array(means), np.array(los), np.array(his)


def _predict_prob_ci_marginalized_2d(res, factor_a: str, A: list[str], factor_b: str, B: list[str],
                                     alpha: float = 0.05) -> pd.DataFrame:
    df_fit = res.model.data.frame.copy()
    DI = res.model.data.design_info
    beta = res.params.values
    V = res.cov_params().values
    z = norm.ppf(1.0 - alpha/2.0)

    cats_a = list(df_fit[factor_a].cat.categories) if (factor_a in df_fit and
                                                       pd.api.types.is_categorical_dtype(df_fit[factor_a])) else None
    cats_b = list(df_fit[factor_b].cat.categories) if (factor_b in df_fit and
                                                       pd.api.types.is_categorical_dtype(df_fit[factor_b])) else None

    rows = []
    for a in A:
        for b in B:
            df_new = df_fit.copy()
            if factor_a in df_new.columns:
                df_new[factor_a] = (pd.Categorical([a]*len(df_new), categories=cats_a)
                                    if cats_a is not None else a)
            if factor_b in df_new.columns:
                df_new[factor_b] = (pd.Categorical([b]*len(df_new), categories=cats_b)
                                    if cats_b is not None else b)

            X = dmatrix(DI, df_new, return_type="dataframe").to_numpy()
            eta = X @ beta
            mu  = 1.0/(1.0 + np.exp(-eta))

            m = float(np.mean(mu))
            w = (mu * (1 - mu))[:, None]
            g = np.mean(w * X, axis=0)
            var_m = float(max(g @ V @ g, 0.0))
            se_m  = np.sqrt(var_m)

            eps = 1e-8
            m_clip = float(np.clip(m, eps, 1 - eps))
            logit_m = float(logit(m_clip))
            se_logit_m = float(se_m / (m_clip * (1 - m_clip)))
            lo = float(expit(logit_m - z * se_logit_m))
            hi = float(expit(logit_m + z * se_logit_m))
            rows.append((a, b, m, lo, hi))


    if not rows:
        return pd.DataFrame(columns=[factor_a, factor_b, "prob", "lo", "hi"])
    out = pd.DataFrame(rows, columns=[factor_a, factor_b, "prob", "lo", "hi"])
    if cats_a is not None: out[factor_a] = pd.Categorical(out[factor_a], categories=cats_a)
    if cats_b is not None: out[factor_b] = pd.Categorical(out[factor_b], categories=cats_b)
    return out



def plot_predicted_main_effect(res, factor, out_path, title=None, y_zoom="auto"):
    apply_neurips_figure_style()
    df_fit = res.model.data.frame
    if factor not in df_fit.columns or not pd.api.types.is_categorical_dtype(df_fit[factor]):
        return

    levels = list(df_fit[factor].cat.categories)
    p, lo, hi = _predict_prob_ci_marginalized(res, factor, levels)

    fig, ax = new_pub_fig(title or f"Predicted accuracy by {factor}", figsize=(4.8, 3.0))
    x = np.arange(len(levels))
    ax.errorbar(x, p, yerr=[p-lo, hi-p], fmt="o", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(levels, rotation=20, ha="right")
    ax.set_ylabel("Predicted accuracy (marginalized)")
    ax.grid(True, alpha=0.25)

    if y_zoom == "auto":
        pad = max(0.005, 0.15*(float(np.nanmax(p))-float(np.nanmin(p))))
        ax.set_ylim(max(0.0, float(np.nanmin(p))-pad), min(1.0, float(np.nanmax(p))+pad))
    elif isinstance(y_zoom, (tuple, list)) and len(y_zoom) == 2:
        ax.set_ylim(*y_zoom)
    else:
        ax.set_ylim(0, 1)

    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)



def plot_block_wald(wald_blocks: pd.DataFrame, out_path: str, title: str | None = None):
    if wald_blocks is None or len(wald_blocks) == 0:
        return
    apply_neurips_figure_style()
    df = wald_blocks.copy()
    df["neglog10p"] = -np.log10(df["p"].clip(lower=np.finfo(float).tiny))
    df = df.sort_values("neglog10p", ascending=True)

    fig, ax = new_pub_fig(title or "Joint block tests (−log10 p)", figsize=(4.8, 2.8))
    y = np.arange(len(df))
    ax.barh(y, df["neglog10p"])
    ax.set_yticks(y)
    ax.set_yticklabels(df["block"])
    ax.axvline(-np.log10(0.05), color="0.5", ls="--", lw=1)
    ax.set_xlabel("−log10 p (higher = stronger)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_pred(res, factor_a: str, factor_b: str, out_path: str,
                          title: str | None = None, y_zoom: tuple[float, float] | str | None = "auto",
                          inset: bool = True):
    apply_neurips_figure_style()
    df_fit = res.model.data.frame
    if factor_a not in df_fit.columns or factor_b not in df_fit.columns:
        return
    if not (pd.api.types.is_categorical_dtype(df_fit[factor_a]) and
            pd.api.types.is_categorical_dtype(df_fit[factor_b])):
        return

    A = list(df_fit[factor_a].cat.categories)
    B = list(df_fit[factor_b].cat.categories)

    new = _predict_prob_ci_marginalized_2d(res, factor_a, A, factor_b, B)

    fig, ax = new_pub_fig(title or f"Predicted accuracy: {factor_a} × {factor_b}", figsize=(5.4, 3.4))
    x = np.arange(len(B))
    dodge = np.linspace(-0.15, 0.15, num=len(A)) if len(A) > 1 else [0.0]

    for i, a in enumerate(A):
        sub = new[new[factor_a] == a].copy()
        y, ylo, yhi = sub["prob"].to_numpy(), sub["lo"].to_numpy(), sub["hi"].to_numpy()
        ax.errorbar(x + dodge[i], y, yerr=[y - ylo, yhi - y], fmt="-o", capsize=3, label=str(a), lw=1.2, ms=3)

    ax.set_xticks(x); ax.set_xticklabels(B, rotation=20, ha="right")
    ax.set_ylabel("Predicted accuracy (marginalized)")

    if y_zoom == "auto":
        ymin, ymax = float(np.nanmin(new["prob"])), float(np.nanmax(new["prob"]))
        span = max(1e-6, ymax - ymin); pad = max(0.005, 0.15 * span)
        ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))
    elif isinstance(y_zoom, (tuple, list)) and len(y_zoom) == 2:
        ax.set_ylim(*y_zoom)
    else:
        ax.set_ylim(0, 1)

    ax.legend(title=factor_a, ncol=min(3, len(A))); ax.grid(True, alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

    if inset:
        fig2, ax2 = new_pub_fig(None, figsize=(2.0, 1.8))
        for i, a in enumerate(A):
            sub = new[new[factor_a] == a].copy()
            y, ylo, yhi = sub["prob"].to_numpy(), sub["lo"].to_numpy(), sub["hi"].to_numpy()
            ax2.errorbar(x + dodge[i], y, yerr=[y - ylo, yhi - y], fmt="-o", capsize=2, lw=1.0, ms=2)
        ax2.set_xticks(x); ax2.set_xticklabels(B, rotation=45, ha="right")
        ax2.set_ylim(0, 1); ax2.set_ylabel("p"); ax2.grid(True, alpha=0.25)
        inset_path = out_path.replace(".pdf", "_inset.pdf")
        fig2.savefig(inset_path, bbox_inches="tight"); plt.close(fig2)


def _wilson_ci(k, n, alpha=0.05):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k/n
    z = norm.ppf(1.0-alpha/2.0)  
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*np.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return p, max(0.0, center-half), min(1.0, center+half)

def plot_marginal_accuracy(long_df: pd.DataFrame, factor: str, out_path: str, title: str | None = None):
    apply_neurips_figure_style()
    if factor not in long_df.columns or not pd.api.types.is_categorical_dtype(long_df[factor]):
        return

    g = long_df.groupby(factor, observed=True)["correct"].agg(["sum","count"]).reset_index()
    rows = []
    for _, r in g.iterrows():
        p, lo, hi = _wilson_ci(int(r["sum"]), int(r["count"]))
        rows.append((str(r[factor]), p, lo, hi, int(r["count"])))
    df = pd.DataFrame(rows, columns=[factor,"acc","lo","hi","n"])

    order = list(long_df[factor].cat.categories)
    df[factor] = pd.Categorical(df[factor], categories=order, ordered=True)
    df = df.sort_values(factor)

    fig, ax = new_pub_fig(title or f"Accuracy by {factor}", figsize=(4.6, 2.8))
    x = np.arange(len(df))
    ax.errorbar(x, df["acc"], yerr=[df["acc"]-df["lo"], df["hi"]-df["acc"]],
                fmt="o", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(df[factor].astype(str), rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    for xi, ni in zip(x, df["n"]):
        ax.text(xi, 0.02, f"n={ni}", ha="center", va="bottom", fontsize=7)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def make_tier1_minimal_plots(fit_dict: dict, out_dir: str, factors: Optional[Tuple[str, ...]] = None) -> dict:
    """
    Produces three figures:
      (1) Block Wald bars
      (2) Predicted accuracy for the strongest interaction (zoom+inset)
      (3) Empirical accuracy by the most relevant single factor (default to age if present)
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    wb = fit_dict.get("wald_blocks", pd.DataFrame())
    p = os.path.join(out_dir, "glm_block_wald.pdf")
    plot_block_wald(wb, p)
    paths["block_wald"] = p

    best_pair = None
    if len(wb):
        inter = wb[wb["block"].str.startswith("inter:")].copy()
        if factors:
            inter = inter[inter["block"].apply(
                lambda s: all(f in s for f in [f"C({f})" for f in factors])
            )]
        if len(inter):
            inter = inter.sort_values("p", ascending=True).iloc[0]
            m = re.match(r"inter:C\(([^)]+)\):C\(([^)]+)\)", str(inter["block"]))
            if m:
                best_pair = (m.group(1), m.group(2))

    if best_pair is None:
        df = fit_dict["data"]
        cand = [f for f in ("gender","age","ethnicity") if f in df.columns]
        if len(cand) >= 2:
            best_pair = (cand[0], cand[1])

    if best_pair is not None:
        a, b = best_pair
        p = os.path.join(out_dir, f"pred_acc_{a}_x_{b}_zoom.pdf")
        plot_interaction_pred(fit_dict["result"], a, b, p, y_zoom="auto", inset=True)
        paths["pred_interaction_zoom"] = p

    long_df = fit_dict["data"]
    if best_pair is not None:
        one_factor = best_pair[1]
    elif factors:
        one_factor = next((f for f in factors if f in long_df.columns), None)
    else:
        one_factor = "age" if "age" in long_df.columns else None

    if one_factor is not None and one_factor in long_df.columns:
        p = os.path.join(out_dir, f"acc_by_{one_factor}.pdf")
        plot_marginal_accuracy(long_df, one_factor, p)
        paths["acc_by_factor"] = p

    return paths


def run_full_tier1_analysis(
    merged_df: pd.DataFrame,
    case: CaseConfig,
    group_keys: Optional[Tuple[str, ...]] = ("gender", "ethnicity"),
    person_set: Optional[object] = None,
    plots_root: Optional[str] = None,
    strategy: Optional[str] = None,
    stage: str = "preliminary",
    per_figure_subdirs: Optional[Dict[str, str]] = None,
    figures: bool = False,
    sub_case: Optional[str] = None,
    **kwargs,
) -> Dict[str, object]:

    tables_dir = resolve_plot_dir(
        case,
        plots_root=os.path.join("results", "tables"),
        strategy=strategy,
        stage="tier1",
        extra_subdir="glm",
        sub_case=sub_case,
    )
    os.makedirs(tables_dir, exist_ok=True)

    if group_keys in (None, "auto"):
        if person_set is None:
            raise ValueError("group_keys='auto' requires person_set.")
    group_keys = _resolve_group_keys(person_set, group_keys)

    glm_out = run_glm_cluster_pipeline(
        merged_df=merged_df,
        person_set=person_set,
        out_root=tables_dir,
        group_keys=group_keys,
        cluster_on="sample_id",
        apply_separation_guard=False,
    )

    wb = glm_out["wald_blocks"]
    if len(wb) and (wb["p"] < 0.05).any():
        print("=== GLM (clustered): significant blocks")
        sig = wb[wb["p"] < 0.05].copy()
        for _, r in sig.iterrows():
            test = str(r.get("test", "")).lower()

            if test == "f" and pd.notnull(r.get("df2", None)):
                print(f"   - {r['block']}: F({int(r['df1'])}, {int(r['df2'])})={r['stat']:.3g}, p={r['p']:.4g}")
            else:
                print(f"   - {r['block']}: χ²({int(r['df1'])})={r['stat']:.3g}, p={r['p']:.4g}")
    else:
        print("=== GLM (clustered): no significant main/interaction blocks at α=0.05")


    print(f"Tables saved to: {tables_dir}")
    print("   - glm_coef_OR.csv")
    print("   - glm_block_tests.csv")
    for fac in (group_keys or ("gender", "ethnicity")):
        print(f"   - glm_pairwise_{fac}.csv")

    if figures:
        if figures:
            plots_dir = resolve_plot_dir(
                case,
                plots_root=plots_root,
                strategy=strategy,
                stage=stage,
                extra_subdir="glm",
                sub_case=sub_case,
            )
            _ = make_tier1_minimal_plots(glm_out["fit"], out_dir=plots_dir, factors=tuple(group_keys))
            print(f"Figures saved to: {plots_dir}")

    return {"glm": glm_out}
