# plot_tools.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def apply_neurips_figure_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.9, "lines.linewidth": 1.2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": "0.7", "grid.linestyle": ":",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })

def new_pub_fig(title=None, figsize=(6.5, 4.2)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass
    if title:
        fig.suptitle(title, fontweight="bold")
    return fig, ax

def bootstrap_line_ci(x, y, grid=None, B=400, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if grid is None:
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    preds = np.empty((B, len(grid)))
    n = len(x)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]; yb = y[idx]
        try:
            z = np.polyfit(xb, yb, 1)
            p = np.poly1d(z)
            preds[b] = p(grid)
        except Exception:
            preds[b] = np.nan
    lo = np.nanpercentile(preds, 100*(alpha/2), axis=0)
    hi = np.nanpercentile(preds, 100*(1-alpha/2), axis=0)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    mid = p(grid)
    return grid, mid, lo, hi
