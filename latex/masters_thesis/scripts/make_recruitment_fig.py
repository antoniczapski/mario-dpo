"""Render Reddit recruitment reach and clicks from the country-level exports."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

THESIS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = THESIS_DIR.parent.parent / "ad_stats"
OUTPUT_PATH = THESIS_DIR / "img" / "reddit_recruitment.png"
COUNTRY_NAMES = {
    "PH": "Philippines",
    "IN": "India",
    "BR": "Brazil",
    "AR": "Argentina",
    "TR": "Turkey",
    "JP": "Japan",
    "MY": "Malaysia",
    "HR": "Croatia",
    "CL": "Chile",
    "RO": "Romania",
    "MX": "Mexico",
    "GR": "Greece",
}


def load_counts(path: Path, metric: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            value = int(row[metric])
            if value < 0:
                raise ValueError(f"Negative {metric} count in {path}")
            counts[row["Country"] or "Unknown"] += value
    return counts


def plot_counts(axis, counts: Counter[str], metric: str, color: str) -> None:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_countries = ordered[:10]
    labels = [
        f"{COUNTRY_NAMES.get(country, country)} ({country})"
        for country, _ in top_countries
    ]
    values = [value for _, value in top_countries]
    remainder = sum(value for _, value in ordered[10:])
    labels.append("Other / unknown")
    values.append(remainder)
    if sum(values) != sum(counts.values()):
        raise ValueError(f"The plotted {metric} counts do not match the export")

    bars = axis.barh(labels, values, color=[color] * 10 + ["#888888"])
    axis.invert_yaxis()
    axis.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
    axis.set_xlim(0, max(values) * 1.2)
    axis.set_xlabel("Reported reach" if metric == "Reach" else "Reported clicks")
    axis.set_title(f"{metric} by country (total: {sum(counts.values()):,})", loc="left")
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def main() -> None:
    reach = load_counts(
        DATA_DIR / "Research_audienceinsights_2026-01-01_2026-10-10.csv", "Reach"
    )
    clicks = load_counts(
        DATA_DIR / "Research_audienceinsights_2026-01-01_2026-10-10 (1).csv", "Clicks"
    )
    with plt.rc_context({"font.size": 11}):
        figure, axes = plt.subplots(2, 1, figsize=(6.5, 8), layout="constrained")
        plot_counts(axes[0], reach, "Reach", "#4477b0")
        plot_counts(axes[1], clicks, "Clicks", "#2b8c6b")
        figure.savefig(OUTPUT_PATH, dpi=200)
        plt.close(figure)
    print(f"Reported reach: {sum(reach.values()):,}")
    print(f"Reported clicks: {sum(clicks.values()):,}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()