"""Static figures for the evaluation report (the dashboard has the interactive versions).

Palette: validated categorical slots (blue, orange, aqua, yellow), fixed order per model.
Two slots sit below 3:1 on the surface, so every curve is direct-labelled and every
figure is paired with a table in the report.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def _axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    return fig, ax


def reliability_plot(oof: pd.DataFrame, path: str | Path, n_bins: int = 10) -> Path:
    y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    pts = [
        (p[bins == b].mean(), y[bins == b].mean(), (bins == b).sum())
        for b in range(n_bins)
        if (bins == b).any()
    ]
    fig, ax = _axes(
        "Calibration: predicted vs observed failure rate",
        "Predicted P(fail)",
        "Observed failure rate",
    )
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.98, 0.93, "perfect calibration", color=INK_2, fontsize=8, ha="right")
    xs, ys, ns = zip(*pts, strict=True)
    ax.plot(xs, ys, color=SERIES[0], linewidth=2)
    ax.scatter(
        xs,
        ys,
        s=[max(40, 160 * n / max(ns)) for n in ns],
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return Path(path)


def pr_curves(oofs: dict[str, pd.DataFrame], path: str | Path) -> Path:
    fig, ax = _axes("Precision-recall (out-of-fold, rolling-origin CV)", "Recall", "Precision")
    base = None
    labels: list[list] = []
    for i, (name, oof) in enumerate(list(oofs.items())[: len(SERIES)]):
        prec, rec, _ = precision_recall_curve(oof["y"], oof["p"])
        ax.plot(rec, prec, color=SERIES[i], linewidth=2, label=name)
        j = int(np.argmin(np.abs(rec - 0.45)))  # curves all converge at recall 1
        labels.append([name, float(rec[j]), float(prec[j])])
        base = oof["y"].mean()
    # direct labels, pushed apart vertically so close curves don't collide
    labels.sort(key=lambda lab: lab[2])
    for k in range(1, len(labels)):
        labels[k][2] = max(labels[k][2], labels[k - 1][2] + 0.06)
    for name, x, y in labels:
        ax.annotate(
            name, (x, y), xytext=(4, 6), textcoords="offset points", color=INK, fontsize=8
        )  # text in ink, not series color
    if base is not None:
        ax.axhline(base, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.01, base + 0.015, f"base rate {base:.1%}", color=INK_2, fontsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return Path(path)
