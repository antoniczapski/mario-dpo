#!/usr/bin/env python3
"""Re-layout of the Chapter 4 (User Study) figures after reviewer feedback.

Uses the data pipeline of replanned_eda.py unchanged, so every plotted value
is identical to the original figures; only sizes and layouts differ. Figures
are drawn at the thesis text width, so the font sizes set here are the sizes
printed in the PDF when included with width=\\textwidth.

Output: latex/masters_thesis/img/updated/ (the original figures are untouched).
"""

from __future__ import annotations

import itertools
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.cluster.hierarchy import leaves_list, linkage
from seaborn.utils import relative_luminance

import replanned_eda as base

OUT_DIR = base.IMG_DIR / "updated"
# \textwidth of latex/masters_thesis/main.tex (398.34pt, reported in main.log).
TEXT_WIDTH = 398.34 / 72.27
DPI = 600
GRID_COLS = 3

FAMILY_COLORS = {
    "human": "#222222",
    "preference-trained": "#CC6677",
    "neural": "#4477AA",
    "constructive/search": "#66AA55",
    "constructive": "#66AA55",
    "search": "#66AA55",
    "pattern": "#DDCC77",
}


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8.5,
            "figure.titlesize": 10.5,
            "figure.labelsize": 9,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {path}")


def small_multiples(n_panels: int, height: float) -> tuple[plt.Figure, np.ndarray]:
    nrows = math.ceil(n_panels / GRID_COLS)
    fig, axes = plt.subplots(
        nrows, GRID_COLS, figsize=(TEXT_WIDTH, height), sharex=True, sharey=True, layout="constrained"
    )
    axes = axes.reshape(-1)
    for ax in axes[n_panels:]:
        ax.axis("off")
    # Panels above an empty slot are the bottom of their column, so they keep x tick labels.
    for ax in axes[max(0, n_panels - GRID_COLS) : n_panels]:
        ax.xaxis.set_tick_params(labelbottom=True)
    return fig, axes[:n_panels]


def plot_generator_ranking(ranking: pd.DataFrame) -> None:
    df = ranking.sort_values("score_rate", ascending=True).copy()
    colors = df["family"].map(FAMILY_COLORS).fillna("#999999")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 4.5), layout="constrained")
    xerr_low = (df["score_rate"] - df["score_ci_low"]).clip(lower=0)
    xerr_high = (df["score_ci_high"] - df["score_rate"]).clip(lower=0)
    ax.barh(df["generator_id"], df["score_rate"], color=colors, edgecolor="#333", linewidth=0.6, alpha=0.9)
    ax.errorbar(
        df["score_rate"], df["generator_id"], xerr=[xerr_low, xerr_high], fmt="none", ecolor="#222", capsize=2.5, lw=0.8
    )
    ax.axvline(0.5, color="#666", lw=0.9, ls="--")
    ax.set_xlabel("Pairwise preference score (win=1, tie=0.5, loss=0)")
    ax.set_ylabel("Generator")
    ax.set_title("Generator ranking from blind pairwise votes")
    ax.set_xlim(0, 1.0)
    legend_handles = [
        Patch(facecolor="#222222", edgecolor="#333", label="Human-authored"),
        Patch(facecolor="#CC6677", edgecolor="#333", label="Preference-trained (MarioDPO)"),
        Patch(facecolor="#4477AA", edgecolor="#333", label="Neural (GAN / LLM / diffusion)"),
        Patch(facecolor="#66AA55", edgecolor="#333", label="Constructive / search"),
        Patch(facecolor="#DDCC77", edgecolor="#333", label="Pattern-based"),
        Line2D([0], [0], color="#666", lw=0.9, ls="--", label="Neutral score (0.5)"),
    ]
    ax.legend(
        handles=legend_handles,
        title="Generator family",
        loc="lower right",
        framealpha=0.95,
        labelspacing=0.35,
        borderaxespad=0.3,
    )
    savefig(fig, "eda_generator_ranking.png")


def plot_pairwise_matrix(matrix: pd.DataFrame, counts: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("row_win", ["#8c2d04", "#f7f7f7", "#2ca25f"])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 5.7), layout="constrained")
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=0, vmax=1, center=0.5, linewidths=0.5, linecolor="white", cbar=False, square=True
    )
    # Hides the whitegrid lines that otherwise show through the empty diagonal.
    ax.grid(False)
    for i, row_gen in enumerate(matrix.index):
        for j, col_gen in enumerate(matrix.columns):
            value = matrix.loc[row_gen, col_gen]
            if i == j or pd.isna(value):
                continue
            color = ".15" if relative_luminance(cmap(value)) > 0.408 else "w"
            ax.text(j + 0.5, i + 0.40, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
            ax.text(j + 0.5, i + 0.75, f"n={counts.loc[row_gen, col_gen]}", ha="center", va="center", fontsize=6, color=color)
    ax.set_title("Pairwise generator preference matrix")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", rotation=0)
    savefig(fig, "eda_pairwise_confusion.png")


def plot_engagement(vote_df: pd.DataFrame) -> None:
    fig, axd = plt.subplot_mosaic(
        [["time", "time"], ["hist", "cum"]], figsize=(TEXT_WIDTH, 4.8), layout="constrained"
    )
    ax = axd["time"]
    daily = vote_df.assign(date=vote_df["created_at"].dt.date).groupby("date").size()
    dates = pd.to_datetime(daily.index)
    ax.bar(dates, daily.values, color="#4477AA")
    ax.set_title("Votes over time")
    ax.set_ylabel("Votes")
    ticks = [t for t in pd.date_range("2026-01-01", "2026-06-01", freq="SMS") if dates.min() <= t <= dates.max()]
    ax.set_xticks(ticks, [t.strftime("%Y-%m-%d") for t in ticks], rotation=30, ha="right", rotation_mode="anchor")

    votes_per_player = vote_df.groupby("player_id").size().sort_values(ascending=False)
    ax = axd["hist"]
    ax.hist(
        votes_per_player.values,
        bins=min(25, max(5, len(votes_per_player) // 2)),
        color="#66AA55",
        edgecolor="white",
    )
    ax.set_title("Votes per anonymous player ID")
    ax.set_xlabel("Votes")
    ax.set_ylabel("Player IDs")
    ax.set_yscale("log")

    sorted_votes = votes_per_player.values
    cumulative = np.cumsum(sorted_votes) / sorted_votes.sum()
    ax = axd["cum"]
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=2.2, lw=1.1, color="#AA3377")
    ax.set_title("Cumulative vote contribution")
    ax.set_xlabel("Top N player IDs")
    ax.set_ylabel("Fraction of votes")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    savefig(fig, "eda_engagement.png")


def plot_user_heatmap(side_df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    mat = base.player_preference_matrix(side_df, min_votes=5)
    order = [g for g in ranking["generator_id"].tolist() if g in mat.columns]
    filled = mat[order].fillna(0.5)
    filled = filled.iloc[leaves_list(linkage(filled.values, method="ward"))]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 5.8), layout="constrained")
    sns.heatmap(filled, cmap="vlag", vmin=0, vmax=1, center=0.5, ax=ax, cbar_kws={"label": "Mean preference score"})
    ax.set_title("Anonymous player ID × generator preference heatmap")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Anonymous player IDs (clustered)")
    ax.set_yticks([])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    savefig(fig, "eda_user_generator_heatmap.png")


def plot_expressive_range(level_metrics: pd.DataFrame, ranking: pd.DataFrame) -> None:
    gens = [g for g in ranking["generator_id"].tolist() if g in set(level_metrics["generator_id"])]
    rate = ranking.set_index("generator_id")["score_rate"].to_dict()
    fig, axes = small_multiples(len(gens), height=7.6)
    for ax, gen in zip(axes, gens):
        g = level_metrics[level_metrics["generator_id"] == gen]
        linearity = g["linearity_r2"].clip(0, 1)
        leniency = g["leniency"].clip(0, 1.5)
        ax.hist2d(linearity, leniency, bins=20, range=[[0, 1], [0, 1.5]], cmap="Blues", cmin=1)
        ax.scatter(linearity, leniency, s=3, color="#111", alpha=0.2, linewidths=0)
        ax.set_title(f"{gen}\nscore={rate.get(gen, np.nan):.2f}, n={len(g)}", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(False)
    # Same bins as before; the view is cropped to where levels exist (R² <= 0.65, 0.17 <= leniency <= 0.99).
    axes[0].set_xlim(0, 0.7)
    axes[0].set_ylim(0.15, 1.05)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
    fig.supxlabel("Linearity (R²)")
    fig.supylabel("Operational leniency")
    fig.suptitle("Expressive range: static level structure by generator")
    savefig(fig, "eda_expressive_range.png")


def plot_trajectory_stacks(side_df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    order = ranking["generator_id"].tolist()
    rate = ranking.set_index("generator_id")["score_rate"].to_dict()
    fig, axes = small_multiples(len(order), height=7.2)
    for ax, gen in zip(axes, order):
        g = side_df[(side_df["generator_id"] == gen) & (side_df["trajectory_length"] > 0)]
        alpha = min(0.22, max(0.035, 6 / max(len(g), 1)))
        for traj in g["trajectory"].tolist():
            xs = [p.get("x", np.nan) for p in traj]
            ys = [p.get("y", np.nan) for p in traj]
            ax.plot(xs, ys, color="#c9252d", lw=0.3, alpha=alpha)
        ax.set_title(f"{gen}\nscore={rate.get(gen, np.nan):.2f}, traces={len(g)}", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.set_facecolor("white")
        ax.grid(False)
    axes[0].set_xlim(0, 3300)
    axes[0].set_ylim(260, 0)
    axes[0].xaxis.set_major_locator(MultipleLocator(1000))
    axes[0].yaxis.set_major_locator(MultipleLocator(100))
    fig.supxlabel("x position (px)")
    fig.supylabel("y position (px)")
    fig.suptitle("Gameplay trajectory stacks by generator")
    savefig(fig, "eda_trajectory_stacks.png")


def plot_tag_semantics(side_df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    tag_cols = [f"tag_{t}" for t in base.TAG_NAMES]
    tag_counts = side_df[tag_cols].sum().sort_values(ascending=False)
    tag_labels = [c.replace("tag_", "") for c in tag_counts.index]

    # Two independent rows: a shared 2x2 grid would force the co-occurrence colorbar
    # margin onto the frequency panel and leave both heatmaps too narrow for their labels.
    fig = plt.figure(figsize=(TEXT_WIDTH, 7.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1])
    ax_freq, ax_gen = top.subplots(1, 2, gridspec_kw={"width_ratios": [0.85, 1]})
    ax_co, ax_out = bottom.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})

    ax_freq.barh(tag_labels[::-1], tag_counts.values[::-1], color="#4477AA")
    ax_freq.set_title("Tag frequency")
    ax_freq.set_xlabel("Assignments")
    ax_freq.xaxis.set_major_locator(MultipleLocator(20))

    tag_by_gen = side_df.groupby("generator_id")[tag_cols].mean().reindex(ranking["generator_id"])
    sns.heatmap(
        tag_by_gen.rename(columns=lambda c: c.replace("tag_", "")),
        ax=ax_gen,
        cmap="YlOrRd",
        vmin=0,
        annot=False,
        cbar_kws={"label": "Tag rate"},
    )
    ax_gen.set_title("Tag rate by generator")
    ax_gen.set_xlabel("Tag")
    ax_gen.set_ylabel("Generator")

    tagged = side_df[side_df["tag_count"] > 0]
    co = pd.DataFrame(0, index=base.TAG_NAMES, columns=base.TAG_NAMES, dtype=float)
    for tags in tagged["tags"]:
        unique_tags = [t for t in set(tags) if t in base.TAG_NAMES]
        for a, b in itertools.combinations(unique_tags, 2):
            co.loc[a, b] += 1
            co.loc[b, a] += 1
    sns.heatmap(co, ax=ax_co, cmap="Blues", cbar_kws={"label": "Co-occurrence count"})
    ax_co.set_title("Tag co-occurrence")
    ax_co.set_xlabel("Tag")
    ax_co.set_ylabel("Tag")

    for ax in (ax_gen, ax_co):
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)

    outcome_rates = []
    for tag in base.TAG_NAMES:
        tagged_side = side_df.loc[side_df[f"tag_{tag}"] == 1, "result_for_side"]
        outcome_rates.append(
            {
                "tag": tag,
                "winner_rate": tagged_side.eq("win").mean() if len(tagged_side) else np.nan,
                "loser_rate": tagged_side.eq("loss").mean() if len(tagged_side) else np.nan,
                "tie_rate": tagged_side.eq("tie").mean() if len(tagged_side) else np.nan,
            }
        )
    out_df = pd.DataFrame(outcome_rates).set_index("tag").loc[tag_labels]
    out_df.plot(kind="bar", stacked=True, ax=ax_out, color=["#66AA55", "#CC6677", "#DDCC77"], legend=False)
    ax_out.set_title("Outcome distribution\nwhen tag is present", pad=17)
    ax_out.set_ylabel("Fraction")
    ax_out.set_xlabel("Tag")
    ax_out.yaxis.set_major_locator(MultipleLocator(0.2))
    plt.setp(ax_out.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Stacked bars reach 1.0, so the legend sits between title and axes instead of over the bars.
    ax_out.legend(
        ["winner", "loser", "tie"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    savefig(fig, "eda_tag_semantics.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = base.load_data()
    side_df = base.normalize_side_rows(data.votes)
    vote_df = base.build_vote_level_rows(data.votes)

    # Same ranking construction (and bootstrap seed) as replanned_eda.main().
    ci = base.bootstrap_score_ci(vote_df, n_boot=800)
    ranking = base.generator_records(side_df).merge(ci, on="generator_id", how="left")
    ranking = ranking.sort_values("score_rate", ascending=False).reset_index(drop=True)
    matrix, counts = base.pairwise_matrix(vote_df, ranking["generator_id"].tolist())
    level_metrics, _ = base.compute_level_metrics()

    apply_style()
    plot_generator_ranking(ranking)
    plot_pairwise_matrix(matrix, counts)
    plot_engagement(vote_df)
    plot_user_heatmap(side_df, ranking)
    plot_expressive_range(level_metrics, ranking)
    plot_trajectory_stacks(side_df, ranking)
    plot_tag_semantics(side_df, ranking)


if __name__ == "__main__":
    main()
