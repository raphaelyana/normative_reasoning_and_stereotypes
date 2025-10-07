"""
Generate the eleven publication figures directly from merged outputs created by
`analysis_tools.load_and_merge_profiles`.

Example:
    dataset_frames = {
        "MGSD": {"df": merged_df_zs, "person_set": PERSON_ETHNICS},
        "MentalManip": {"df": merged_df_zs_mentalmanip, "person_set": PERSON_ETHNICS},
        "MMLU-Large": {"df": merged_df_zs_mmlu_full, "person_set": PERSON_ETHNICS},
        "MMLU": {"df": merged_df_zs_mmlu, "person_set": PERSON_ETHNICS},
    }
    results = generate_publication_figures(dataset_frames, output_dir=Path("paper_figs"))
    print(results["figure_paths"]["fig1"])
"""

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from profiles.profile_sets import PERSON_ETHNICS
from profiles.schema import PersonSet


plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "light_gray": "#E8E8E8",
    "dark_gray": "#404040",
    "medium_gray": "#808080",
}


class DatasetMetrics:
    """Container for per-dataset statistics used by the figures."""

    __slots__ = (
        "analysis_df",
        "correlations",
        "rescue_summary",
        "variance_decomposition",
        "cv_r2",
    )

    def __init__(
        self,
        analysis_df: pd.DataFrame,
        correlations: Dict[str, Tuple[float, float]],
        rescue_summary: Dict[str, float],
        variance_decomposition: Dict[str, Dict[str, float]],
        cv_r2: Dict[str, float],
    ) -> None:
        self.analysis_df = analysis_df
        self.correlations = correlations
        self.rescue_summary = rescue_summary
        self.variance_decomposition = variance_decomposition
        self.cv_r2 = cv_r2


def _profile_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col.startswith("profile") and "__" not in col and df[col].notna().any()
    ]


def _item_consensus_excluding(
    df: pd.DataFrame, profile_col: str, min_strength: Optional[float] = None
) -> Tuple[pd.Series, pd.Series]:
    """Leave-one-out consensus for COI computation."""
    profile_cols = [c for c in df.columns if c.startswith("profile") and c != profile_col]
    if not profile_cols:
        raise ValueError("Consensus requires at least one other profile column.")

    votes = df[profile_cols].to_numpy()
    labels, strengths = [], []

    for row in votes:
        valid = [v for v in row if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not valid:
            labels.append(None)
            strengths.append(np.nan)
            continue

        counts = pd.Series(valid).value_counts()
        max_count = counts.max()
        candidates = sorted(counts[counts == max_count].index.tolist())
        label = candidates[0]
        strength = max_count / len(valid)
        if min_strength is not None and np.isfinite(strength):
            strength = max(min_strength, strength)
        labels.append(label)
        strengths.append(strength)

    consensus = pd.Series(labels, index=df.index, name="consensus")
    strength = pd.Series(strengths, index=df.index, name="strength")
    return consensus, strength


def _boldness_metrics(
    df: pd.DataFrame, profile_col: str, baseline_col: str = "base_pred"
) -> Dict[str, float]:
    consensus, strength = _item_consensus_excluding(df, profile_col)
    ambiguity = 1.0 - strength

    baseline = df[baseline_col]
    profile_pred = df[profile_col]
    disagreements = profile_pred != baseline

    mask_consensus = baseline == consensus
    numerator = (disagreements & mask_consensus) * strength
    denominator = strength[mask_consensus]
    coi = float(numerator.sum() / (denominator.sum() + 1e-12))

    q_hi = ambiguity.quantile(0.75)
    q_lo = ambiguity.quantile(0.25)
    hi_mask = ambiguity >= q_hi
    lo_mask = ambiguity <= q_lo
    ati = float(disagreements[hi_mask].mean() - disagreements[lo_mask].mean())

    cai = float((disagreements * ambiguity).sum() / (ambiguity.sum() + 1e-12))

    return {"COI": coi, "ATI": ati, "CAI": cai}


def _safe_pearsonr(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float]:
    a = pd.Series(list(a), dtype=float)
    b = pd.Series(list(b), dtype=float)
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.isclose(a.var(ddof=1), 0.0) or np.isclose(b.var(ddof=1), 0.0):
        return np.nan, np.nan
    return pearsonr(a, b)


def _build_stratification_labels(df: pd.DataFrame) -> np.ndarray:
    if "case_family" in df.columns:
        encoded = pd.factorize(list(zip(df["true_label"], df["case_family"])))[0]
    else:
        encoded = pd.factorize(df["true_label"])[0]
    return encoded


def _cv_test_indices(
    strat_labels: np.ndarray, desired_splits: int, random_state: int
) -> List[np.ndarray]:
    unique, counts = np.unique(strat_labels, return_counts=True)
    min_count = counts.min() if len(counts) else 0

    if min_count < 2 or len(strat_labels) < 4:
        return [np.arange(len(strat_labels))]

    n_splits = min(desired_splits, int(min_count))
    if n_splits < 2:
        return [np.arange(len(strat_labels))]

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    dummy = np.zeros_like(strat_labels)
    return [test_idx for _, test_idx in splitter.split(dummy, strat_labels)]


def _compute_profile_statistics(
    df: pd.DataFrame,
    profile_cols: Sequence[str],
    baseline_col: str,
    n_folds: int,
    random_state: int,
) -> pd.DataFrame:
    strat_labels = _build_stratification_labels(df)
    fold_indices = _cv_test_indices(strat_labels, n_folds, random_state)

    rows = []
    for profile in profile_cols:
        fold_acc = []
        fold_rescue = []
        fold_extra = []
        fold_disagreement = []
        fold_coi = []
        fold_ati = []
        fold_cai = []

        rescued_total = 0
        base_errors_total = 0
        harmed_total = 0
        base_correct_total = 0

        for test_idx in fold_indices:
            test = df.iloc[test_idx]
            fold_acc.append(
                accuracy_score(test["true_label"], test[profile])
            )

            base_correct = test[baseline_col] == test["true_label"]
            profile_correct = test[profile] == test["true_label"]

            base_errors = (~base_correct).sum()
            base_correct_cnt = base_correct.sum()
            rescued = ((~base_correct) & profile_correct).sum()
            harmed = (base_correct & (~profile_correct)).sum()

            rescued_total += int(rescued)
            base_errors_total += int(base_errors)
            harmed_total += int(harmed)
            base_correct_total += int(base_correct_cnt)

            rescue_rate = (
                float(rescued / base_errors) if base_errors > 0 else np.nan
            )
            extra_rate = (
                float(harmed / base_correct_cnt)
                if base_correct_cnt > 0
                else np.nan
            )

            fold_rescue.append(rescue_rate)
            fold_extra.append(extra_rate)
            fold_disagreement.append(float((test[profile] != test[baseline_col]).mean()))

            metrics = _boldness_metrics(test, profile_col=profile, baseline_col=baseline_col)
            fold_coi.append(metrics["COI"])
            fold_ati.append(metrics["ATI"])
            fold_cai.append(metrics["CAI"])

        accuracy_mean = float(np.mean(fold_acc))
        accuracy_std = float(np.std(fold_acc))
        rescue_rate = (
            float(rescued_total / base_errors_total)
            if base_errors_total > 0
            else np.nan
        )
        extra_error_rate = (
            float(harmed_total / base_correct_total)
            if base_correct_total > 0
            else np.nan
        )

        rows.append(
            {
                "profile": profile,
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "rescue_rate": rescue_rate,
                "extra_error_rate": extra_error_rate,
                "disagreement_rate": float(np.mean(fold_disagreement)),
                "COI": float(np.mean(fold_coi)),
                "ATI": float(np.mean(fold_ati)),
                "CAI": float(np.mean(fold_cai)),
            }
        )

    return pd.DataFrame(rows)


def _attach_traits(
    profile_metrics: pd.DataFrame,
    person_set: PersonSet,
    group_keys: Sequence[str],
) -> pd.DataFrame:
    trait_rows = []
    for profile in profile_metrics["profile"]:
        try:
            traits = person_set.get_traits(profile, group_keys)
        except Exception:
            traits = {key: "Unknown" for key in group_keys}
        row = {"profile": profile}
        for key in group_keys:
            value = traits.get(key, "Unknown")
            if hasattr(value, "value"):
                value = value.value
            row[key] = "Unknown" if value is None else str(value)
        trait_rows.append(row)
    traits_df = pd.DataFrame(trait_rows)
    return profile_metrics.merge(traits_df, on="profile", how="left")


def _infer_group_keys(person_set: Optional[PersonSet]) -> Tuple[str, ...]:
    fallback = ("gender", "ethnicity", "age")
    if person_set is None or not getattr(person_set, "metadata", None):
        return fallback

    ordered = []
    seen = set()
    preferred = ("gender", "ethnicity", "age")

    for traits in person_set.metadata.values():
        if isinstance(traits, dict):
            keys = traits.keys()
        else:
            keys = getattr(traits, "__dict__", {}).keys()
        for key in keys:
            if key not in seen:
                seen.add(key)
                ordered.append(str(key))

    result = []
    for key in preferred:
        if key in seen:
            result.append(key)
            seen.discard(key)
    result.extend(sorted(seen))

    return tuple(result) if result else fallback


def _variance_decomposition(
    analysis_df: pd.DataFrame,
    group_keys: Sequence[str],
    demographic_traits: Sequence[str],
    outcomes: Sequence[str],
    cv_splits: int,
    random_state: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    trait_cols = [key for key in group_keys if key in analysis_df.columns]
    if not trait_cols:
        return {out: {"demo": 0.0, "other": 0.0, "total": 0.0} for out in outcomes}, {
            out: np.nan for out in outcomes
        }

    trait_df = analysis_df[["profile"] + trait_cols].set_index("profile")
    encoded = pd.get_dummies(
        trait_df,
        columns=trait_cols,
        prefix_sep="=",
        drop_first=True,
        dtype=float,
    )
    encoded = encoded.replace({np.nan: 0.0})

    demo_cols = [
        col
        for col in encoded.columns
        if any(col.startswith(f"{trait}=") for trait in demographic_traits)
    ]
    other_cols = [
        col
        for col in encoded.columns
        if col not in demo_cols
    ]

    def _fit_r2(X: np.ndarray, y: np.ndarray) -> float:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.size == 0 or np.isclose(y.var(), 0.0):
            return 0.0
        model = LinearRegression().fit(X, y)
        return float(max(model.score(X, y), 0.0))

    def _cv_r2(X: np.ndarray, y: np.ndarray) -> float:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.size == 0 or len(y) < 4:
            return np.nan
        n_splits = min(cv_splits, len(y))
        if n_splits < 2:
            return np.nan
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, test_idx in splitter.split(X):
            model = LinearRegression().fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            scores.append(r2_score(y[test_idx], y_pred))
        return float(np.nanmean(scores))

    results = {}
    cv_results = {}

    for outcome in outcomes:
        if outcome not in analysis_df.columns:
            results[outcome] = {"demo": 0.0, "other": 0.0, "total": 0.0}
            cv_results[outcome] = np.nan
            continue

        y = analysis_df.set_index("profile")[outcome].astype(float)
        valid = y.notna()
        y = y[valid].to_numpy()
        if y.size == 0 or np.isclose(y.var(), 0.0):
            results[outcome] = {"demo": 0.0, "other": 0.0, "total": 0.0}
            cv_results[outcome] = np.nan
            continue

        X_demo = encoded.loc[valid, demo_cols].to_numpy() if demo_cols else np.zeros((len(y), 0))
        X_other = encoded.loc[valid, other_cols].to_numpy() if other_cols else np.zeros((len(y), 0))

        if X_demo.shape[1] == 0 and X_other.shape[1] == 0:
            results[outcome] = {"demo": 0.0, "other": 0.0, "total": 0.0}
            cv_results[outcome] = np.nan
            continue

        r2_demo = _fit_r2(X_demo, y) if X_demo.shape[1] else 0.0
        r2_other = _fit_r2(X_other, y) if X_other.shape[1] else 0.0

        if X_demo.shape[1] == 0:
            X_full = X_other
        elif X_other.shape[1] == 0:
            X_full = X_demo
        else:
            X_full = np.column_stack([X_demo, X_other])
        r2_full = _fit_r2(X_full, y)

        if X_demo.shape[1] == 0:
            demo_unique = 0.0
            other_unique = r2_full
        elif X_other.shape[1] == 0:
            demo_unique = r2_full
            other_unique = 0.0
        else:
            demo_unique = 0.5 * (r2_demo + (r2_full - r2_other))
            other_unique = 0.5 * (r2_other + (r2_full - r2_demo))
            demo_unique = float(np.clip(demo_unique, 0.0, r2_full))
            other_unique = float(np.clip(other_unique, 0.0, r2_full - demo_unique))

        results[outcome] = {
            "demo": float(demo_unique),
            "other": float(other_unique),
            "total": float(r2_full),
        }

        cv_results[outcome] = _cv_r2(X_full, y)

    return results, cv_results


def compute_dataset_metrics(
    merged_df: pd.DataFrame,
    *,
    person_set: Optional[PersonSet] = None,
    group_keys: Optional[Sequence[str]] = None,
    demographic_traits: Sequence[str] = ("gender", "ethnicity"),
    baseline_col: str = "base_pred",
    n_folds: int = 5,
    random_state: int = 42,
) -> DatasetMetrics:
    if "true_label" not in merged_df.columns:
        raise ValueError("merged_df must contain a 'true_label' column.")
    if baseline_col not in merged_df.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found in dataframe.")

    person_set = person_set or PERSON_ETHNICS
    if not group_keys:
        group_keys = _infer_group_keys(person_set)
    else:
        group_keys = tuple(group_keys)
    profile_cols = [
        col for col in _profile_columns(merged_df) if col in person_set.metadata
    ]
    if not profile_cols:
        raise ValueError("No valid profile columns were found in the dataframe.")

    profile_metrics = _compute_profile_statistics(
        merged_df, profile_cols, baseline_col, n_folds, random_state
    )
    analysis_df = _attach_traits(profile_metrics, person_set, group_keys)

    corr_vol_bold = _safe_pearsonr(analysis_df["accuracy_std"], analysis_df["COI"])
    corr_bold_acc = _safe_pearsonr(analysis_df["COI"], analysis_df["accuracy_mean"])

    volatility_median = analysis_df["accuracy_std"].median()
    inconsistent = analysis_df[analysis_df["accuracy_std"] > volatility_median]["rescue_rate"]
    consistent = analysis_df[analysis_df["accuracy_std"] <= volatility_median]["rescue_rate"]

    inc_rate = float(inconsistent.mean())
    con_rate = float(consistent.mean())
    delta = inc_rate - con_rate

    if len(inconsistent.dropna()) >= 2 and len(consistent.dropna()) >= 2:
        _, p_value = ttest_ind(inconsistent.dropna(), consistent.dropna(), equal_var=False)
        p_value = float(p_value)
    else:
        p_value = float("nan")

    variance_decomposition, cv_r2 = _variance_decomposition(
        analysis_df,
        group_keys=group_keys,
        demographic_traits=demographic_traits,
        outcomes=("COI", "ATI", "CAI", "accuracy_mean"),
        cv_splits=n_folds,
        random_state=random_state,
    )
    if "accuracy_mean" in variance_decomposition:
        variance_decomposition["ACCURACY"] = variance_decomposition.pop("accuracy_mean")
    if "accuracy_mean" in cv_r2:
        cv_r2["ACCURACY"] = cv_r2.pop("accuracy_mean")

    correlations = {
        "volatility_vs_boldness": corr_vol_bold,
        "boldness_vs_accuracy": corr_bold_acc,
    }
    rescue_summary = {
        "inc": inc_rate,
        "con": con_rate,
        "delta": delta,
        "p": p_value,
    }

    return DatasetMetrics(
        analysis_df=analysis_df,
        correlations=correlations,
        rescue_summary=rescue_summary,
        variance_decomposition=variance_decomposition,
        cv_r2=cv_r2,
    )


def _collect_metric_maps(
    dataset_results: Mapping[str, DatasetMetrics],
    dataset_order: Sequence[str],
) -> Tuple[
    Dict[str, Tuple[float, float]],
    Dict[str, Tuple[float, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Tuple[float, float]]],
    Dict[str, Dict[str, float]],
]:
    corr_vol_bold = {
        name: dataset_results[name].correlations["volatility_vs_boldness"]
        for name in dataset_order
    }
    corr_bold_acc = {
        name: dataset_results[name].correlations["boldness_vs_accuracy"]
        for name in dataset_order
    }
    rescue_summary = {
        name: dataset_results[name].rescue_summary for name in dataset_order
    }
    variance_decomp = {
        name: {
            key: (
                dataset_results[name].variance_decomposition[key]["demo"],
                dataset_results[name].variance_decomposition[key]["other"],
            )
            for key in ("COI", "ATI", "CAI", "ACCURACY")
        }
        for name in dataset_order
    }
    cv_performance = {
        name: {
            key: dataset_results[name].cv_r2.get(key, np.nan)
            for key in ("ATI", "CAI")
        }
        for name in dataset_order
    }
    return (
        corr_vol_bold,
        corr_bold_acc,
        rescue_summary,
        variance_decomp,
        cv_performance,
    )


def _save_fig(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_figures(
    dataset_order: Sequence[str],
    corr_vol_bold: Mapping[str, Tuple[float, float]],
    corr_bold_acc: Mapping[str, Tuple[float, float]],
    rescue_summary: Mapping[str, Mapping[str, float]],
    variance_decomp: Mapping[str, Mapping[str, Tuple[float, float]]],
    cv_performance: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> Dict[str, Path]:
    dataset_colors = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["tertiary"],
        COLORS["quaternary"],
    ]
    color_cycle = [dataset_colors[i % len(dataset_colors)] for i in range(len(dataset_order))]
    figure_paths: Dict[str, Path] = {}

    x_pos = np.arange(len(dataset_order))

    fig, ax = plt.subplots(figsize=(8, 5))
    corr_vals = [corr_vol_bold[name][0] for name in dataset_order]
    p_vals = [corr_vol_bold[name][1] for name in dataset_order]
    bar_colors = [
        color_cycle[i] if np.isfinite(p_vals[i]) and p_vals[i] < 0.05 else COLORS["light_gray"]
        for i in range(len(dataset_order))
    ]
    bars = ax.bar(x_pos, corr_vals, color=bar_colors, edgecolor="black", linewidth=1.2, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    ax.set_ylabel("Correlation coefficient (r)")
    ax.set_title("Consistency-Boldness Relationship", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_order)
    ax.set_ylim(min(-0.4, min(corr_vals) - 0.05), max(0.1, max(corr_vals) + 0.05))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    for bar, val, p in zip(bars, corr_vals, p_vals):
        y_pos = bar.get_height()
        label_y = y_pos - 0.02 if y_pos < 0 else y_pos + 0.01
        label_va = "top" if y_pos < 0 else "bottom"
        color = "black" if np.isfinite(p) and p < 0.05 else COLORS["medium_gray"]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{val:.3f}",
            ha="center",
            va=label_va,
            fontsize=10,
            fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal",
            color=color,
        )
    figure_paths["fig1"] = _save_fig(fig, output_dir, "fig1_volatility_boldness")

    fig, ax = plt.subplots(figsize=(8, 5))
    corr_vals = [corr_bold_acc[name][0] for name in dataset_order]
    p_vals = [corr_bold_acc[name][1] for name in dataset_order]
    bar_colors = [
        color_cycle[i] if np.isfinite(p_vals[i]) and p_vals[i] < 0.05 else COLORS["light_gray"]
        for i in range(len(dataset_order))
    ]
    bars = ax.bar(x_pos, corr_vals, color=bar_colors, edgecolor="black", linewidth=1.2, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    ax.set_ylabel("Correlation coefficient (r)")
    ax.set_title("Boldness-Accuracy Relationship", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_order)
    ax.set_ylim(min(-0.4, min(corr_vals) - 0.05), max(0.6, max(corr_vals) + 0.05))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    for bar, val, p in zip(bars, corr_vals, p_vals):
        y_pos = bar.get_height()
        label_y = y_pos + 0.02 if y_pos > 0 else y_pos - 0.02
        label_va = "bottom" if y_pos > 0 else "top"
        color = "black" if np.isfinite(p) and p < 0.05 else COLORS["medium_gray"]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{val:.3f}",
            ha="center",
            va=label_va,
            fontsize=10,
            fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal",
            color=color,
        )
    figure_paths["fig2"] = _save_fig(fig, output_dir, "fig2_boldness_accuracy")

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    inc_rates = [rescue_summary[name]["inc"] for name in dataset_order]
    con_rates = [rescue_summary[name]["con"] for name in dataset_order]
    bars1 = ax.bar(
        x_pos - width / 2,
        inc_rates,
        width,
        label="Inconsistent",
        color=COLORS["secondary"],
        edgecolor="black",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        con_rates,
        width,
        label="Consistent",
        color=COLORS["primary"],
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_ylabel("Error Correction Rate")
    ax.set_title("Error Correction Rates by Consistency Type", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_order)
    ax.legend(frameon=False, loc="upper right")
    y_max = max(inc_rates + con_rates)
    ax.set_ylim(0, max(0.28, y_max + 0.05))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    for i, name in enumerate(dataset_order):
        p_val = rescue_summary[name]["p"]
        if np.isfinite(p_val) and p_val < 0.05:
            max_height = max(inc_rates[i], con_rates[i])
            ax.plot(
                [i - width / 2, i + width / 2],
                [max_height + 0.01, max_height + 0.01],
                "k-",
                linewidth=1.5,
            )
            ax.text(
                i,
                max_height + 0.02,
                "p < 0.05",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    figure_paths["fig3"] = _save_fig(fig, output_dir, "fig3_error_correction_rates")

    fig, ax = plt.subplots(figsize=(8, 5))
    deltas = [rescue_summary[name]["delta"] for name in dataset_order]
    p_vals = [rescue_summary[name]["p"] for name in dataset_order]
    colors = [
        COLORS["quaternary"] if np.isfinite(p) and p < 0.05 else COLORS["light_gray"]
        for p in p_vals
    ]
    bars = ax.bar(
        x_pos,
        deltas,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Error Correction (Inconsistent − Consistent)")
    ax.set_title(
        "Effect Size of Consistency on Error Correction", fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_order)
    ax.set_ylim(min(-0.02, min(deltas) - 0.005), max(0.02, max(deltas) + 0.005))
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    for i, (bar, delta, p) in enumerate(zip(bars, deltas, p_vals)):
        y_pos = delta + 0.001 if delta > 0 else delta - 0.001
        va = "bottom" if delta > 0 else "top"
        color = "black" if np.isfinite(p) and p < 0.05 else COLORS["medium_gray"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{delta:.3f}",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal",
            color=color,
        )
    figure_paths["fig4"] = _save_fig(fig, output_dir, "fig4_error_correction_deltas")

    for idx, outcome in enumerate(["COI", "ATI", "CAI", "ACCURACY"], start=5):
        fig, ax = plt.subplots(figsize=(8, 5))
        demo_vals = [variance_decomp[name][outcome][0] for name in dataset_order]
        other_vals = [variance_decomp[name][outcome][1] for name in dataset_order]
        bars1 = ax.bar(
            x_pos - width / 2,
            demo_vals,
            width,
            label="Demographics",
            color=COLORS["primary"],
            edgecolor="black",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x_pos + width / 2,
            other_vals,
            width,
            label="Other traits",
            color=COLORS["tertiary"],
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_ylabel("In-sample R²")
        title_map = {
            "COI": "COI Variance Decomposition",
            "ATI": "ATI Variance Decomposition",
            "CAI": "CAI Variance Decomposition",
            "ACCURACY": "Accuracy Prediction Variance Decomposition",
        }
        ax.set_title(title_map[outcome], fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dataset_order)
        ylim = 0.6 if outcome in ("ATI", "CAI") else 0.5 if outcome == "ACCURACY" else 0.3
        ax.set_ylim(0, ylim)
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        filename_map = {
            "COI": "coi",
            "ATI": "ati",
            "CAI": "cai",
            "ACCURACY": "accuracy",
        }
        figure_paths[f"fig{idx}"] = _save_fig(fig, output_dir, f"fig{idx}_{filename_map[outcome]}_variance")

    for idx, outcome in enumerate(["ATI", "CAI"], start=9):
        fig, ax = plt.subplots(figsize=(8, 5))
        in_sample = [
            sum(variance_decomp[name][outcome])
            for name in dataset_order
        ]
        cv_vals = [cv_performance[name][outcome] for name in dataset_order]
        bars1 = ax.bar(
            x_pos - width / 2,
            in_sample,
            width,
            label="In-sample",
            color=COLORS["dark_gray"],
            edgecolor="black",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x_pos + width / 2,
            cv_vals,
            width,
            label="Cross-validated",
            color=COLORS["quaternary"],
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_ylabel("R²")
        title = f"{outcome} Cross-Validation Performance"
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(dataset_order)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
        ax.set_ylim(-0.1, 0.7 if outcome == "CAI" else 0.65)
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        figure_paths[f"fig{idx}"] = _save_fig(fig, output_dir, f"fig{idx}_{outcome.lower()}_cv_performance")

    fig, ax = plt.subplots(figsize=(10, 7))
    summary_metrics = {
        "Volatility-Boldness r": [corr_vol_bold[name][0] for name in dataset_order],
        "Boldness-Accuracy r": [corr_bold_acc[name][0] for name in dataset_order],
        "Error Correction Difference": [rescue_summary[name]["delta"] for name in dataset_order],
        "Demo R^2 (COI)": [variance_decomp[name]["COI"][0] for name in dataset_order],
        "Demo R^2 (ATI)": [variance_decomp[name]["ATI"][0] for name in dataset_order],
        "Demo R^2 (CAI)": [variance_decomp[name]["CAI"][0] for name in dataset_order],
        "Demo R^2 (Accuracy)": [variance_decomp[name]["ACCURACY"][0] for name in dataset_order],
        "CV R^2 (ATI)": [cv_performance[name]["ATI"] for name in dataset_order],
        "CV R^2 (CAI)": [cv_performance[name]["CAI"] for name in dataset_order],
    }
    matrix = np.array(list(summary_metrics.values()))
    colors_list = [(0.2, 0.4, 0.8), (1, 1, 1), (0.8, 0.2, 0.2)]
    cmap = LinearSegmentedColormap.from_list("custom", colors_list, N=100)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels(dataset_order)
    ax.set_yticks(np.arange(len(summary_metrics)))
    ax.set_yticklabels(list(summary_metrics.keys()))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value", rotation=270, labelpad=15)

    for i, metric_name in enumerate(summary_metrics.keys()):
        for j, dataset_name in enumerate(dataset_order):
            value = matrix[i, j]
            is_significant = False
            if metric_name == "Volatility-Boldness r":
                is_significant = np.isfinite(corr_vol_bold[dataset_name][1]) and corr_vol_bold[dataset_name][1] < 0.05
            elif metric_name == "Boldness-Accuracy r":
                is_significant = np.isfinite(corr_bold_acc[dataset_name][1]) and corr_bold_acc[dataset_name][1] < 0.05
            elif metric_name == "Error Correction Difference":
                p_val = rescue_summary[dataset_name]["p"]
                is_significant = np.isfinite(p_val) and p_val < 0.05

            text_color = "white" if abs(value) > 0.3 else "black"
            fw = "bold" if is_significant else "normal"
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight=fw,
            )

            if is_significant:
                rect = Rectangle(
                    (j - 0.45, i - 0.45),
                    0.9,
                    0.9,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.0,
                )
                ax.add_patch(rect)

    ax.set_title("Summary of Key Metrics Across Datasets", fontweight="bold", pad=20)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Metric")
    legend_elements = [
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            label="p < 0.05",
        )
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        frameon=False,
    )
    figure_paths["fig11"] = _save_fig(fig, output_dir, "fig11_summary_heatmap")

    return figure_paths


def generate_publication_figures(
    dataset_frames: Mapping[str, object],
    *,
    output_dir: Union[Path, str] = Path("figs"),
    dataset_order: Optional[Sequence[str]] = None,
    person_set: Optional[PersonSet] = None,
    group_keys: Sequence[str] = ("gender", "ethnicity", "age"),
    demographic_traits: Sequence[str] = ("gender", "ethnicity"),
    baseline_col: str = "base_pred",
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    if not dataset_frames:
        raise ValueError("dataset_frames cannot be empty.")

    output_dir = Path(output_dir)
    dataset_order = dataset_order or list(dataset_frames.keys())
    missing = set(dataset_order) - set(dataset_frames.keys())
    if missing:
        raise ValueError(f"dataset_order contains missing keys: {sorted(missing)}")

    dataset_results = {}
    for name in dataset_order:
        entry = dataset_frames[name]
        if isinstance(entry, pd.DataFrame):
            df = entry
            ps = person_set
            local_group_keys = group_keys
            local_demo_traits = demographic_traits
        elif isinstance(entry, Mapping):
            if "df" not in entry:
                raise ValueError(f"Dataset '{name}' mapping must include a 'df' key.")
            df = entry["df"]
            ps = entry.get("person_set", person_set)
            local_group_keys = entry.get("group_keys", group_keys)
            local_demo_traits = entry.get("demographic_traits", demographic_traits)
        else:
            raise TypeError(
                f"Dataset entry for '{name}' must be a DataFrame or mapping, found {type(entry)!r}."
            )
        dataset_results[name] = compute_dataset_metrics(
            df,
            person_set=ps,
            group_keys=local_group_keys,
            demographic_traits=local_demo_traits,
            baseline_col=baseline_col,
            n_folds=n_folds,
            random_state=random_state,
        )

    (
        corr_vol_bold,
        corr_bold_acc,
        rescue_summary,
        variance_decomp,
        cv_performance,
    ) = _collect_metric_maps(dataset_results, dataset_order)

    figure_paths = _plot_figures(
        dataset_order=dataset_order,
        corr_vol_bold=corr_vol_bold,
        corr_bold_acc=corr_bold_acc,
        rescue_summary=rescue_summary,
        variance_decomp=variance_decomp,
        cv_performance=cv_performance,
        output_dir=output_dir,
    )

    return {
        "dataset_results": dataset_results,
        "corr_vol_bold": corr_vol_bold,
        "corr_bold_acc": corr_bold_acc,
        "rescue_summary": rescue_summary,
        "variance_decomposition": variance_decomp,
        "cv_performance": cv_performance,
        "figure_paths": figure_paths,
    }


__all__ = ["generate_publication_figures", "compute_dataset_metrics", "DatasetMetrics"]
