"""
Generate Expressive Range Analysis (ERA) plot.

For each generator, computes (linearity, leniency) for every level and produces
a 2D histogram. All generators are arranged as subplots in a single figure.

Definitions:
- Linearity: R^2 of linear regression on platform-midpoint y-coordinates per column.
  Higher R^2 = more linear (closer to a straight line).
- Leniency: weighted sum of game objects per level, normalised by level length.
  Following Smith & Whitehead (2010): enemies and gaps decrease leniency,
  power-ups and coins increase it.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEVELS_DIR = ROOT / "db" / "seed" / "levels"
OUT_DIR = ROOT / "latex" / "masters_thesis" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mario AI Framework tile semantics
SOLID_TILES = set("XSDQ?@!#")  # ground / brick / question / pyramid
ENEMY_TILES = set("EgGkKrRyY*Bb")
GAP_TILES = set("-")  # only meaningful at floor row
COIN_TILES = set("o")
POWERUP_TILES = set("UL12")

# Pretty display names for generators
GEN_DISPLAY = {
    "original": "Original SMB",
    "notch": "Notch",
    "notchParam": "NotchParam",
    "notchParamRand": "NotchParamRand",
    "hopper": "Hopper",
    "ore": "ORE",
    "genetic": "GE",
    "patternCount": "Pattern Count",
    "patternOccur": "Pattern Occurrence",
    "patternWeightCount": "Pattern Wt. Count",
    "mariogan": "MarioGAN",
    "mariogpt": "MarioGPT",
    "marioDiffusion": "MarioDiffusion",
}


def load_level(path: Path) -> list[str]:
    rows = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = [r for r in rows if r.strip("\r\n") != ""]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [r.ljust(width, "-") for r in rows]


def linearity(level: list[str]) -> float | None:
    """R^2 of linear regression on highest solid tile per column."""
    if not level:
        return None
    h = len(level)
    w = len(level[0])
    ys: list[float] = []
    xs: list[float] = []
    for x in range(w):
        col = [level[y][x] for y in range(h)]
        # find topmost solid (smallest y)
        top_y = None
        for y in range(h):
            if col[y] in SOLID_TILES:
                top_y = y
                break
        if top_y is not None:
            xs.append(float(x))
            ys.append(float(top_y))
    if len(xs) < 3:
        return None
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    if ys_arr.std() < 1e-9:
        return 1.0  # perfectly flat line
    slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
    pred = slope * xs_arr + intercept
    ss_res = float(np.sum((ys_arr - pred) ** 2))
    ss_tot = float(np.sum((ys_arr - ys_arr.mean()) ** 2))
    if ss_tot < 1e-9:
        return 1.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def leniency(level: list[str]) -> float | None:
    """Smith-style leniency: enemies/gaps decrease, power-ups/coins increase.

    Weights (per Smith & Whitehead 2010, simplified):
        enemy = -1.0
        gap   = -0.5  (each empty floor column counted)
        powerup = +1.0
        coin = +0.1
    Normalised by level width to give a per-column score.
    """
    if not level:
        return None
    h = len(level)
    w = len(level[0])
    floor_y = h - 1
    enemy = 0
    powerup = 0
    coin = 0
    gap = 0
    for x in range(w):
        col = [level[y][x] for y in range(h)]
        if col[floor_y] == "-":
            gap += 1
        for ch in col:
            if ch in ENEMY_TILES:
                enemy += 1
            elif ch in POWERUP_TILES:
                powerup += 1
            elif ch in COIN_TILES:
                coin += 1
    score = (-1.0 * enemy + -0.5 * gap + 1.0 * powerup + 0.1 * coin) / max(w, 1)
    return float(score)


def collect(generator_dir: Path) -> tuple[list[float], list[float]]:
    lins: list[float] = []
    lens: list[float] = []
    for path in sorted(generator_dir.glob("*.txt")):
        lvl = load_level(path)
        lin = linearity(lvl)
        lenc = leniency(lvl)
        if lin is None or lenc is None:
            continue
        lins.append(lin)
        lens.append(lenc)
    return lins, lens


def main() -> None:
    generators = [g for g in GEN_DISPLAY if (LEVELS_DIR / g).exists()]
    print(f"Found {len(generators)} generators")

    data: dict[str, tuple[list[float], list[float]]] = {}
    for g in generators:
        lins, lens = collect(LEVELS_DIR / g)
        data[g] = (lins, lens)
        print(f"  {g:24s}  n={len(lins):4d}")

    # Determine a common range so plots are comparable
    all_lin = [v for lins, _ in data.values() for v in lins]
    all_len = [v for _, lens in data.values() for v in lens]
    lin_lo, lin_hi = 0.0, 1.0
    len_lo = float(np.percentile(all_len, 1))
    len_hi = float(np.percentile(all_len, 99))

    n = len(generators)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows),
                             sharex=True, sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, g in enumerate(generators):
        ax = axes[i]
        lins, lens = data[g]
        if not lins:
            ax.set_visible(False)
            continue
        h = ax.hist2d(lins, lens, bins=[20, 20],
                      range=[[lin_lo, lin_hi], [len_lo, len_hi]],
                      cmap="viridis")
        ax.set_title(GEN_DISPLAY[g], fontsize=10)
        ax.set_xlim(lin_lo, lin_hi)
        ax.set_ylim(len_lo, len_hi)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.supxlabel("Linearity ($R^2$)", fontsize=12)
    fig.supylabel("Leniency", fontsize=12)
    fig.suptitle("Expressive Range Analysis: linearity vs.\\ leniency per generator",
                 fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.97))

    out_path = OUT_DIR / "era-linearity-leniency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
