"""
Utility functions to recreate the benchmark bar plots directly from saved result
files. The helpers expect you to provide a mapping from (strategy, model) to the
corresponding result artifact (CSV with predictions, JSON metrics, etc.) and
will take care of computing the requested metric (accuracy by default) and
rendering a publication-grade grouped bar chart.

Example
-------
>>> from pathlib import Path
>>> from paper_figs.benchmark_results import MetricSpec, plot_benchmark_figure
>>>
>>> strategies = ["ZS", "FS", "CoT"]
>>> models = ["GPT-4.1-mini", "GPT-4o-mini"]
>>> specs = {
...     "ZS": {
...         "GPT-4.1-mini": MetricSpec(
...             path=Path("results/openai_4.1_mini/zero_shot/classic/results_stereotype_zero_shot_prompt_short_binary.csv")
...         ),
...         "GPT-4o-mini": MetricSpec(
...             path=Path("results/openai_4o_mini/zero_shot/classic/results_stereotype_zero_shot_prompt_short_binary.csv")
...         ),
...     },
...     "FS": {
...         "GPT-4.1-mini": MetricSpec(
...             path=Path("results/openai_4.1_mini/few_shot/classic/results_stereotype_few_shot_prompt_short_3examples.csv")
...         ),
...         "GPT-4o-mini": MetricSpec(
...             path=Path("results/openai_4o_mini/few_shot/classic/results_stereotype_few_shot_prompt_short_3examples.csv")
...         ),
...     },
...     "CoT": {
...         "GPT-4.1-mini": MetricSpec(
...             path=Path("results/openai_4.1_mini/cot/classic/results_stereotype_optimized_cot.csv")
...         ),
...         "GPT-4o-mini": MetricSpec(
...             path=Path("results/openai_4o_mini/cot/classic/results_stereotype_optimized_cot.csv")
...         ),
...     },
... }
>>> output = plot_benchmark_figure(
...     dataset_name="Manipulation",
...     strategies=strategies,
...     models=models,
...     metric_specs=specs,
...     annotate=True,
... )
>>> output["data"].head()

`MetricSpec` defaults to computing accuracy (%) from CSV files containing
`true_label` and `pred_label` columns, so you can usually instantiate it with
only the `path`. For other metrics (e.g., macro-F1 stored in a JSON report or a
CSV column), set `kind` and `metric_key` accordingly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_tools import apply_neurips_figure_style, new_pub_fig


__all__ = [
    "MetricSpec",
    "TokenSpec",
    "collect_benchmark_table",
    "collect_multi_dataset_table",
    "collect_token_usage_table",
    "plot_benchmark_figure",
    "plot_grouped_bars",
    "plot_performance_heatmap",
    "plot_token_consumption",
]


MetricMapping = Mapping[str, Mapping[str, "MetricSpec"]]


@dataclass(frozen=True)
class MetricSpec:
    """
    Description of how to extract a scalar metric from a saved result artifact.

    Parameters
    ----------
    path:
        Location of the artifact (CSV, JSON, etc.).
    kind:
        Determines how to extract the metric. Supported values:

        - ``"accuracy"`` (default): read a CSV with ``true_label`` and
          ``pred_label`` columns and compute accuracy.
        - ``"json"``: read a JSON file and grab the value located at
          ``metric_key`` (dot-separated path).
        - ``"csv_column"``: read a CSV file and aggregate the column named
          ``metric_key``.
    metric_key:
        Optional key used when ``kind`` is ``"json"`` or ``"csv_column"``.
        Accepts dot-separated paths and integer indices for nested structures
        (e.g., ``"metrics.macro_f1"``).
    true_label_col / pred_label_col:
        Column names used when computing accuracy.
    aggregate:
        Aggregation applied when ``kind="csv_column"``. One of "mean", "last",
        "first", or "max". Defaults to "mean".
    scale:
        Multiplier applied to the metric after extraction. Set to ``100.0`` to
        obtain percentages, or ``None`` to disable scaling.
    """

    path: Union[str, Path]
    kind: str = "accuracy"
    metric_key: Optional[str] = None
    true_label_col: str = "true_label"
    pred_label_col: str = "pred_label"
    aggregate: str = "mean"
    scale: Optional[float] = 100.0

    def resolved_path(self) -> Path:
        return Path(self.path)


@dataclass(frozen=True)
class TokenSpec:
    """
    Describe how to extract token counts for a given strategy/model run.

    Parameters
    ----------
    path:
        CSV artifact containing token information. The loader will look for a
        ``token_column`` (default ``"tokens_used"``). If that column is not
        present it will sum the ``prompt_column`` and ``completion_column``.
    token_column:
        Name of the column storing total tokens. Leave as ``None`` to attempt
        automatic detection (``tokens_used`` first).
    prompt_column / completion_column:
        Column names used when the token total needs to be reconstructed from
        prompt + completion counts.
    scale:
        Optional multiplier applied to the token counts (e.g., to convert to
        thousands). Defaults to ``1.0`` (no scaling).
    label:
        Optional free-form identifier propagated to the output table (useful
        when aggregating by model or dataset).
    """

    path: Union[str, Path]
    token_column: Optional[str] = None
    prompt_column: str = "prompt_tokens"
    completion_column: str = "completion_tokens"
    scale: float = 1.0
    label: Optional[str] = None

    def resolved_path(self) -> Path:
        return Path(self.path)


def _navigate_nested(mapping: Union[Mapping[str, object], Sequence[object]], key_path: str) -> object:
    """Traverse a mapping/list using a dot-separated path."""
    current: object = mapping
    for raw_key in key_path.split("."):
        if isinstance(current, Mapping):
            if raw_key not in current:
                raise KeyError(f"Key '{raw_key}' not present while traversing '{key_path}'.")
            current = current[raw_key]  # type: ignore[index]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                idx = int(raw_key)
            except ValueError as exc:
                raise KeyError(
                    f"Expected integer index while traversing list for key '{raw_key}' in '{key_path}'."
                ) from exc
            if idx >= len(current):
                raise KeyError(f"Index {idx} out of range for sequence while traversing '{key_path}'.")
            current = current[idx]
        else:
            raise KeyError(f"Cannot descend into object of type {type(current)!r} using key '{raw_key}'.")
    return current


def _load_metric_value(spec: MetricSpec) -> float:
    """
    Load a metric value from the artifact described by ``spec``.

    Returns the metric in raw units before scaling.
    """
    path = spec.resolved_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing result artifact: {path}")

    kind = spec.kind.lower()

    if kind == "accuracy":
        df = pd.read_csv(path)
        for column in (spec.true_label_col, spec.pred_label_col):
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in {path}")

        y_true = df[spec.true_label_col].astype(str).str.strip()
        y_pred = df[spec.pred_label_col].astype(str).str.strip()
        value = float((y_true == y_pred).mean())

    elif kind == "json":
        if not spec.metric_key:
            raise ValueError(f"'metric_key' must be provided for JSON metrics (file: {path})")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        value = _navigate_nested(payload, spec.metric_key)
        value = float(value)

    elif kind == "csv_column":
        if not spec.metric_key:
            raise ValueError(f"'metric_key' must be provided for CSV column metrics (file: {path})")
        df = pd.read_csv(path)
        if spec.metric_key not in df.columns:
            raise KeyError(f"Column '{spec.metric_key}' not found in {path}")
        series = pd.to_numeric(df[spec.metric_key], errors="coerce")
        agg = spec.aggregate.lower()
        if agg == "mean":
            value = float(series.mean())
        elif agg == "last":
            value = float(series.iloc[-1])
        elif agg == "first":
            value = float(series.iloc[0])
        elif agg == "max":
            value = float(series.max())
        else:
            raise ValueError(f"Unsupported aggregate '{spec.aggregate}'.")

    else:
        raise ValueError(f"Unsupported metric kind '{spec.kind}'.")

    if spec.scale is not None:
        value *= spec.scale
    return value


def _load_token_series(spec: TokenSpec) -> pd.Series:
    """Extract per-sample token counts according to ``spec``."""
    path = spec.resolved_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing token artifact: {path}")

    df = pd.read_csv(path)
    token_column = spec.token_column or ("tokens_used" if "tokens_used" in df.columns else None)

    if token_column and token_column in df.columns:
        series = pd.to_numeric(df[token_column], errors="coerce")
    else:
        missing_cols = [
            col for col in (spec.prompt_column, spec.completion_column) if col not in df.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Could not determine token counts in {path}. "
                f"Missing columns: {', '.join(missing_cols)}"
            )
        series = pd.to_numeric(df[spec.prompt_column], errors="coerce") + pd.to_numeric(
            df[spec.completion_column], errors="coerce"
        )

    series = series.dropna()
    return series.astype(float) * spec.scale


def collect_benchmark_table(
    dataset_name: str,
    strategies: Sequence[str],
    models: Sequence[str],
    metric_specs: MetricMapping,
    *,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """
    Aggregate metrics for ``dataset_name`` into a tidy DataFrame.

    Parameters
    ----------
    dataset_name:
        Name of the dataset (used for labelling; stored in the output).
    strategies:
        Ordered list of strategy identifiers (e.g., ``["ZS", "FS", "CoT"]``).
    models:
        Ordered list of model identifiers (column order in the bar chart).
    metric_specs:
        Nested mapping ``strategy -> model -> MetricSpec`` describing where to
        pull the metrics from.
    on_missing:
        Behaviour when a (strategy, model) pair is missing. Options:
        - ``"raise"`` (default): raise ``KeyError``.
        - ``"warn"``: issue a warning and fill with ``NaN``.
        - ``"ignore"``: silently fill with ``NaN``.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``["dataset", "strategy", "model", "value"]``.
    """
    rows: List[Dict[str, object]] = []
    missing_policy = on_missing.lower()

    for strategy in strategies:
        strategy_specs = metric_specs.get(strategy, {})
        for model in models:
            spec = strategy_specs.get(model)
            if spec is None:
                message = f"No MetricSpec for strategy='{strategy}' and model='{model}'."
                if missing_policy == "raise":
                    raise KeyError(message)
                if missing_policy == "warn":
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                value = np.nan
            else:
                value = _load_metric_value(spec)

            rows.append(
                {
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "model": model,
                    "value": value,
                }
            )

    df = pd.DataFrame(rows)
    df["strategy"] = pd.Categorical(df["strategy"], categories=strategies, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    return df


def collect_multi_dataset_table(
    datasets: Mapping[str, MetricMapping],
    strategies: Sequence[str],
    models: Sequence[str],
    *,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """
    Convenience wrapper to aggregate metrics for multiple datasets at once.

    Parameters
    ----------
    datasets:
        Mapping ``dataset_name -> metric_specs`` where ``metric_specs`` follows
        the same structure used by :func:`collect_benchmark_table`.
    strategies / models:
        Ordered lists shared across datasets.
    on_missing:
        Behaviour when a metric is unavailable (see :func:`collect_benchmark_table`).
    """
    frames: List[pd.DataFrame] = []
    for dataset_name, metric_specs in datasets.items():
        frames.append(
            collect_benchmark_table(
                dataset_name=dataset_name,
                strategies=strategies,
                models=models,
                metric_specs=metric_specs,
                on_missing=on_missing,
            )
        )
    if not frames:
        raise ValueError("No datasets provided to collect_multi_dataset_table.")
    return pd.concat(frames, ignore_index=True)


def collect_token_usage_table(
    strategies: Sequence[str],
    token_specs: Mapping[str, Sequence[TokenSpec]],
    *,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """
    Load per-example token counts for each strategy.

    Parameters
    ----------
    strategies:
        Ordered list of strategy identifiers.
    token_specs:
        Mapping ``strategy -> sequence of TokenSpec`` describing which result
        files to inspect. Multiple specs per strategy are concatenated.
    on_missing:
        Behaviour when a strategy has no specs. One of ``"raise"``, ``"warn"``,
        or ``"ignore"``.
    """
    rows: List[Dict[str, object]] = []
    policy = on_missing.lower()

    for strategy in strategies:
        specs = token_specs.get(strategy, [])
        if not specs:
            message = f"No token specs provided for strategy '{strategy}'."
            if policy == "raise":
                raise KeyError(message)
            if policy == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue

        for spec in specs:
            series = _load_token_series(spec)
            label = spec.label or spec.resolved_path().stem
            rows.extend(
                {
                    "strategy": strategy,
                    "tokens": float(value),
                    "source": label,
                    "path": str(spec.resolved_path()),
                }
                for value in series.values
            )

    if not rows:
        raise ValueError("Token table is empty; verify token specs and files.")

    df = pd.DataFrame(rows)
    df["strategy"] = pd.Categorical(df["strategy"], categories=strategies, ordered=True)
    return df


def plot_grouped_bars(
    dataset_name: str,
    pivot_table: pd.DataFrame,
    strategies: Sequence[str],
    models: Sequence[str],
    *,
    ax: Optional[plt.Axes] = None,
    colors: Optional[Mapping[str, str]] = None,
    ylabel: str = "Accuracy (%)",
    ylim: Tuple[float, float] = (0.0, 100.0),
    annotate: bool = False,
    annotation_fmt: str = "{:.1f}",
) -> plt.Axes:
    """
    Render a grouped bar chart from a pivoted table of metrics.

    Parameters
    ----------
    dataset_name:
        Title displayed above the axes.
    pivot_table:
        DataFrame indexed by ``strategy`` with columns for ``models``.
    strategies / models:
        Ordering of strategies (x-axis) and models (bar groups).
    ax:
        Optional axes to draw on; if omitted a new figure is created.
    colors:
        Optional mapping ``model -> color``.
    ylabel:
        Y-axis label (default "Accuracy (%)").
    ylim:
        Y-axis limits as ``(lower, upper)``.
    annotate:
        Whether to display numeric labels above each bar.
    annotation_fmt:
        Format string used when annotating bar heights.
    """
    if ax is None:
        _, ax = new_pub_fig(title=dataset_name, figsize=(9, 4.6))
    else:
        ax.set_title(dataset_name)

    n_strategies = len(strategies)
    n_models = len(models)
    if n_models == 0:
        raise ValueError("At least one model is required to plot grouped bars.")

    x_positions = np.arange(n_strategies, dtype=float)
    width = 0.8 / max(n_models, 1)

    color_cycle = plt.get_cmap("tab10")
    value_bounds: List[float] = []

    for idx, model in enumerate(models):
        if model not in pivot_table.columns:
            y_vals = np.full(n_strategies, np.nan)
        else:
            y_vals = pivot_table.loc[strategies, model].to_numpy(dtype=float)

        if np.isnan(y_vals).all():
            continue

        offset = (idx - (n_models - 1) / 2.0) * width
        bar_positions = x_positions + offset
        bar_color = colors.get(model) if colors else color_cycle(idx % color_cycle.N)
        bars = ax.bar(bar_positions, y_vals, width, label=model, color=bar_color, linewidth=0)

        finite_mask = np.isfinite(y_vals)
        value_bounds.extend(y_vals[finite_mask])

        if annotate:
            for rect, value, is_finite in zip(bars, y_vals, finite_mask):
                if not is_finite:
                    continue
                ax.annotate(
                    annotation_fmt.format(value),
                    xy=(rect.get_x() + rect.get_width() / 2.0, value),
                    xytext=(0.0, 3.0),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategies)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="upper right", frameon=True)

    if not value_bounds:
        warnings.warn("No finite values to plot.", RuntimeWarning, stacklevel=2)

    return ax


def plot_benchmark_figure(
    dataset_name: str,
    strategies: Sequence[str],
    models: Sequence[str],
    metric_specs: MetricMapping,
    *,
    colors: Optional[Mapping[str, str]] = None,
    ylabel: str = "Accuracy (%)",
    ylim: Tuple[float, float] = (0.0, 100.0),
    annotate: bool = False,
    annotation_fmt: str = "{:.1f}",
    apply_style: bool = True,
    on_missing: str = "raise",
) -> Dict[str, object]:
    """
    Convenience wrapper that loads metrics, pivots them, and generates the plot.

    Returns a dictionary with:
        - ``"figure"``: the matplotlib Figure object.
        - ``"axes"``: the Axes containing the plot.
        - ``"data"``: tidy DataFrame (output of :func:`collect_benchmark_table`).
        - ``"pivot"``: pivoted DataFrame used for plotting.
    """
    if apply_style:
        apply_neurips_figure_style()

    data = collect_benchmark_table(
        dataset_name=dataset_name,
        strategies=strategies,
        models=models,
        metric_specs=metric_specs,
        on_missing=on_missing,
    )

    pivot = (
        data.pivot(index="strategy", columns="model", values="value")
        .reindex(index=strategies, columns=models)
    )

    fig, ax = new_pub_fig(title=dataset_name, figsize=(9, 4.6))
    plot_grouped_bars(
        dataset_name=dataset_name,
        pivot_table=pivot,
        strategies=strategies,
        models=models,
        ax=ax,
        colors=colors,
        ylabel=ylabel,
        ylim=ylim,
        annotate=annotate,
        annotation_fmt=annotation_fmt,
    )

    return {"figure": fig, "axes": ax, "data": data, "pivot": pivot}


def plot_performance_heatmap(
    combined_data: pd.DataFrame,
    *,
    models: Sequence[str],
    datasets: Sequence[str],
    strategies: Sequence[str],
    figsize: Tuple[float, float] = (14.0, 6.0),
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    apply_style: bool = True,
) -> Dict[str, object]:
    """
    Produce a heatmap summarising performance across datasets, strategies, and models.

    Parameters
    ----------
    combined_data:
        Output of :func:`collect_multi_dataset_table`.
    models / datasets / strategies:
        Ordering for axes (rows = models, columns = dataset×strategy).
    figsize:
        Size of the produced figure.
    cmap:
        Matplotlib colormap name.
    vmin / vmax:
        Optional color scaling limits.
    """
    if apply_style:
        apply_neurips_figure_style()

    expected_multi_index = pd.MultiIndex.from_product([datasets, strategies], names=["dataset", "strategy"])
    pivot = (
        combined_data.pivot_table(index="model", columns=["dataset", "strategy"], values="value")
        .reindex(index=models, columns=expected_multi_index)
    )

    data_matrix = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)

    column_labels = [f"{ds}\n{st}" for ds in datasets for st in strategies]
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=90)

    ax.set_title("Model Performance Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric Value")

    fig.tight_layout()
    return {"figure": fig, "axes": ax, "pivot": pivot}


def plot_token_consumption(
    token_table: pd.DataFrame,
    *,
    strategies: Sequence[str],
    ylabel: str = "Tokens per example",
    title: str = "Token Consumption by Strategy",
    figsize: Tuple[float, float] = (9.0, 5.0),
    show_fliers: bool = False,
    apply_style: bool = True,
) -> Dict[str, object]:
    """
    Render a boxplot summarising token consumption for each strategy.

    Parameters
    ----------
    token_table:
        Output of :func:`collect_token_usage_table`.
    strategies:
        Ordering of strategies along the x-axis.
    ylabel / title:
        Axis label and figure title.
    figsize:
        Figure size.
    show_fliers:
        Whether to display outliers in the boxplot.
    """
    if apply_style:
        apply_neurips_figure_style()

    fig, ax = plt.subplots(figsize=figsize)
    data_for_plot = [token_table.loc[token_table["strategy"] == s, "tokens"].to_numpy(dtype=float) for s in strategies]

    ax.boxplot(data_for_plot, labels=strategies, showfliers=show_fliers)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    return {"figure": fig, "axes": ax}
