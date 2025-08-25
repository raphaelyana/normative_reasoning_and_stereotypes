import matplotlib as mpl

def apply_neurips_figure_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.9, "lines.linewidth": 1.2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": "0.7", "grid.linestyle": ":",
        "figure.constrained_layout.use": True,
    })
