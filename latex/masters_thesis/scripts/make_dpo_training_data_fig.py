"""Regenerate ``dpo_training_data.png`` for Chapter 5.

Replaces the original three-panel figure whose middle and right panels
(synthetic-pair score margins and level-score histogram) were not informative
in a preference-dataset context. The new figure has two panels that both show
the Human-vs-Synthetic split, first as raw pair counts and then after applying
the ``10x`` upweighting of human pairs used during DPO training.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

HUMAN_PAIRS = 473
SYNTHETIC_PAIRS = 2606
HUMAN_WEIGHT = 10

HUMAN_EFFECTIVE = HUMAN_PAIRS * HUMAN_WEIGHT
SYNTHETIC_EFFECTIVE = SYNTHETIC_PAIRS
TOTAL_EFFECTIVE = HUMAN_EFFECTIVE + SYNTHETIC_EFFECTIVE

OUT_PATH = Path(__file__).resolve().parent.parent / "img" / "dpo_training_data.png"

HUMAN_COLOR = "#2ca02c"
SYNTH_COLOR = "#4477b0"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: raw pair counts ------------------------------------------------
    ax = axes[0]
    labels = ["Human\n(PCG Arena votes)", "Synthetic\n(judge-labelled)"]
    counts = [HUMAN_PAIRS, SYNTHETIC_PAIRS]
    bars = ax.bar(labels, counts, color=[HUMAN_COLOR, SYNTH_COLOR], alpha=0.85,
                  edgecolor="white")
    ax.set_ylabel("Number of preference pairs")
    ax.set_title("Raw pair counts")
    ax.set_ylim(0, max(counts) * 1.18)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:,}", ha="center", va="bottom", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # --- Right: weight-adjusted contribution ---------------------------------
    ax = axes[1]
    weighted = [HUMAN_EFFECTIVE, SYNTHETIC_EFFECTIVE]
    bars = ax.bar(labels, weighted, color=[HUMAN_COLOR, SYNTH_COLOR], alpha=0.85,
                  edgecolor="white")
    ax.set_ylabel("Effective training signal (weighted pairs)")
    ax.set_title(f"After 10x upweighting of human pairs\n"
                 f"(effective total: {TOTAL_EFFECTIVE:,})")
    ax.set_ylim(0, max(weighted) * 1.18)
    for bar, raw, eff in zip(bars, [HUMAN_PAIRS, SYNTHETIC_PAIRS], weighted):
        share = eff / TOTAL_EFFECTIVE
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{eff:,}  ({share:.0%})",
                ha="center", va="bottom", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.suptitle("MarioDPO preference dataset: composition and effective weighting",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
