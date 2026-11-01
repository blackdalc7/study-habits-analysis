"""Matplotlib charts. One function per chart, each saves a PNG and returns the path."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from study_habits import analysis

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "figure.dpi": 110,
    }
)


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def study_scatter(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(df["study_hours_week"], df["exam_score"], s=18, color=BLUE, alpha=0.55,
               edgecolor=SURFACE, linewidth=0.5)
    slope, intercept = np.polyfit(df["study_hours_week"], df["exam_score"], 1)
    xs = np.linspace(0, df["study_hours_week"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2)
    ax.text(0.98, 0.06, f"trend: +{slope:.2f} points per extra hour", transform=ax.transAxes,
            ha="right", color=INK)
    ax.set_xlabel("study hours per week")
    ax.set_ylabel("exam score")
    ax.set_title("More study time, higher scores")
    ax.grid(axis="y", color=GRID)
    return _finish(fig, path)


def sleep_bars(df: pd.DataFrame, path: Path) -> Path:
    table = analysis.score_by_sleep(df)
    best = table[table["count"] >= 10]["mean"].idxmax()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [BLUE if label == best else "#a9c6ec" for label in table.index]
    bars = ax.bar(table.index.astype(str), table["mean"], color=colors, width=0.7)
    for bar, (mean, count) in zip(bars, table.itertuples(index=False), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.8, f"{mean:.0f}", ha="center",
                color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, 2, f"n={count}", ha="center", color=SURFACE,
                fontsize=8)
    ax.set_ylim(0, table["mean"].max() + 8)
    ax.set_xlabel("hours of sleep per night")
    ax.set_ylabel("average exam score")
    ax.set_title(f"The sweet spot is {best} hours of sleep")
    return _finish(fig, path)


def coffee_curve(df: pd.DataFrame, path: Path) -> Path:
    table = analysis.score_by_coffee(df)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(table))
    ax.errorbar(x, table["mean"], yerr=1.96 * table["sem"], color=AQUA, linewidth=2,
                marker="o", markersize=8, markeredgecolor=SURFACE, capsize=4)
    ax.set_xticks(x, table.index)
    ax.set_xlabel("cups of coffee per day")
    ax.set_ylabel("average exam score (95% CI)")
    ax.set_title("A little coffee helps, a lot does not")
    ax.grid(axis="y", color=GRID)
    return _finish(fig, path)


def correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    corr = analysis.correlations(df)
    labels = ["study", "sleep", "social media", "coffee", "attendance", "exam score"]
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ed", BLUE])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if abs(value) > 0.7 else INK)
    ax.spines[:].set_visible(False)
    ax.set_title("How everything correlates")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _finish(fig, path)


def major_boxplot(df: pd.DataFrame, path: Path) -> Path:
    order = analysis.score_by_major(df).index
    data = [df.loc[df["major"] == m, "exam_score"] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True,
               boxprops={"facecolor": "#cfe0f5", "edgecolor": BLUE},
               medianprops={"color": ORANGE, "linewidth": 2},
               whiskerprops={"color": MUTED}, capprops={"color": MUTED},
               flierprops={"markeredgecolor": MUTED, "markersize": 4})
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.invert_yaxis()
    ax.set_xlabel("exam score")
    ax.set_title("Score spread by major")
    ax.grid(axis="x", color=GRID)
    return _finish(fig, path)


ALL_CHARTS = {
    "study_vs_score.png": study_scatter,
    "sleep_vs_score.png": sleep_bars,
    "coffee_vs_score.png": coffee_curve,
    "correlation_heatmap.png": correlation_heatmap,
    "score_by_major.png": major_boxplot,
}


def save_all(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fn(df, out_dir / name) for name, fn in ALL_CHARTS.items()]
