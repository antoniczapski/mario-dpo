"""Static structural feature extraction from ASCII Mario levels.

Every feature here is computed purely from the tile grid — no gameplay
telemetry and (critically) no generator identity. This is what lets the Judge
Function score freshly generated levels and avoids the circular
"``'pattern' in generator_id``" leak of the original implementation.

The public entry point is :func:`extract_features`, returning an ordered dict
of ``float`` features. :data:`FEATURE_NAMES` lists them in a stable order.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter

from .constants import (
    AIR,
    COIN_TILES,
    ENEMY_GROUPS,
    ENEMY_TILES,
    GROUND_LIKE_TILES,
    PIPE_TILES,
    POWERUP_TILES,
    QUESTION_TILES,
    REWARD_TILES,
    SOLID_TILES,
)


# --- Surface profile helpers ----------------------------------------------
def surface_profile(rows: list[str]) -> list[int]:
    """Height of the topmost ground-like tile per column (0 = floor).

    Returns, for each column, ``n_rows - row_index`` of the highest solid
    surface tile, or 0 if the column has no ground. Larger = taller structure.
    """
    h = len(rows)
    w = max((len(r) for r in rows), default=0)
    profile = []
    for c in range(w):
        top = 0
        for r in range(h):
            if c < len(rows[r]) and rows[r][c] in GROUND_LIKE_TILES:
                top = h - r
                break
        profile.append(top)
    return profile


def _linregress_r2(ys: list[float]) -> tuple[float, float]:
    """Return ``(r2, mean_abs_residual)`` of a least-squares line over ys.

    ``r2`` is the coefficient of determination of ``y ~ x`` (x = column index);
    higher means a flatter/more predictable surface. ``mean_abs_residual`` is
    the average absolute deviation from the fitted line (Smith & Whitehead style).
    """
    n = len(ys)
    if n < 3:
        return 0.0, 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    mean_abs_resid = sum(abs(r) for r in resid) / n
    if syy == 0:
        # Perfectly flat surface: fully explained by the (flat) line.
        return 1.0, mean_abs_resid
    ss_res = sum(r * r for r in resid)
    r2 = 1.0 - ss_res / syy
    return max(0.0, min(1.0, r2)), mean_abs_resid


def _gaps(rows: list[str]) -> tuple[int, int, int]:
    """Return ``(gap_count, max_gap_width, total_gap_cols)``.

    A "gap" is a maximal run of columns with no standable tile anywhere in the
    *bottom half* of the level (a pit Mario can fall into). Scanning the bottom
    half (rather than just the floor row) is robust to platform-based generators
    such as ORE, which build their walkable surface from ``%`` jump-through
    platforms above a ``|`` background instead of bottom-row ``X`` ground; it
    also ignores high floating platforms that sit over a real pit.
    """
    h = len(rows)
    w = max((len(r) for r in rows), default=0)
    if h < 2 or w == 0:
        return 0, 0, 0
    lo = h // 2
    is_gap = []
    for c in range(w):
        solid = any(
            c < len(rows[r]) and rows[r][c] in SOLID_TILES for r in range(lo, h)
        )
        is_gap.append(not solid)
    count = 0
    total = 0
    max_w = 0
    run = 0
    for g in is_gap:
        if g:
            run += 1
            total += 1
        else:
            if run > 0:
                count += 1
                max_w = max(max_w, run)
            run = 0
    if run > 0:
        count += 1
        max_w = max(max_w, run)
    return count, max_w, total


def _shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for n in counter.values():
        p = n / total
        ent -= p * math.log2(p)
    return ent


def _columns(rows: list[str]) -> list[str]:
    h = len(rows)
    w = max((len(r) for r in rows), default=0)
    cols = []
    for c in range(w):
        cols.append("".join(rows[r][c] if c < len(rows[r]) else AIR for r in range(h)))
    return cols


def _compression_ratio(rows: list[str]) -> float:
    raw = ("\n".join(rows)).encode("utf-8")
    if not raw:
        return 0.0
    comp = gzip.compress(raw, compresslevel=6)
    return len(comp) / len(raw)


# --- Feature names (stable order) ------------------------------------------
FEATURE_NAMES: list[str] = [
    "width",
    "height",
    "solid_density",
    "breakable_density",
    "question_density",
    "coin_density",
    "powerup_density",
    "pipe_density",
    "reward_density",
    "reward_count",
    "enemy_density",
    "enemy_goomba_density",
    "enemy_koopa_density",
    "enemy_spiky_density",
    "enemy_bullet_density",
    "enemy_types_present",
    "gap_count",
    "max_gap_width",
    "gap_fraction",
    "gap_frequency",
    "surface_linearity_r2",
    "surface_mean_abs_resid",
    "surface_roughness",
    "surface_mean_height",
    "surface_height_range",
    "vertical_solid_max",
    "vertical_solid_mean",
    "air_fraction",
    "unique_column_ratio",
    "column_entropy",
    "tile_entropy",
    "compression_ratio",
    "leniency",
]


def extract_features(rows: list[str]) -> dict[str, float]:
    """Compute the full static feature vector for one level."""
    rows = [r for r in rows if r != ""]
    h = len(rows)
    w = max((len(r) for r in rows), default=0)
    n_tiles = sum(len(r) for r in rows) or 1

    tile_counts: Counter = Counter()
    for r in rows:
        tile_counts.update(r)

    def density(tileset) -> float:
        return sum(tile_counts[t] for t in tileset) / n_tiles

    solid_density = density(SOLID_TILES)
    breakable_density = tile_counts["S"] / n_tiles
    question_density = density(QUESTION_TILES)
    coin_density = density(COIN_TILES)
    powerup_density = density(POWERUP_TILES)
    pipe_density = density(PIPE_TILES)
    reward_density = density(REWARD_TILES)
    reward_count = float(sum(tile_counts[t] for t in REWARD_TILES))

    enemy_density = density(ENEMY_TILES)
    enemy_group_density = {
        name: density(chars) for name, chars in ENEMY_GROUPS.items()
    }
    enemy_types_present = float(
        sum(1 for chars in ENEMY_GROUPS.values() if any(tile_counts[c] for c in chars))
    )

    gap_count, max_gap_width, total_gap = _gaps(rows)
    gap_fraction = total_gap / w if w else 0.0
    gap_frequency = gap_count / w if w else 0.0

    profile = surface_profile(rows)
    r2, mean_abs_resid = _linregress_r2([float(p) for p in profile])
    if len(profile) >= 2:
        roughness = sum(abs(profile[i] - profile[i - 1]) for i in range(1, len(profile))) / (len(profile) - 1)
        mean_height = sum(profile) / len(profile)
        height_range = float(max(profile) - min(profile))
    else:
        roughness = mean_height = height_range = 0.0

    # Vertical solid stacks per column (max/mean contiguous solid run).
    cols = _columns(rows)
    max_stack = 0
    stack_sum = 0
    for col in cols:
        run = best = 0
        for ch in col:
            if ch in SOLID_TILES:
                run += 1
                best = max(best, run)
            else:
                run = 0
        max_stack = max(max_stack, best)
        stack_sum += best
    vertical_solid_mean = stack_sum / len(cols) if cols else 0.0

    air_fraction = tile_counts[AIR] / n_tiles
    unique_column_ratio = len(set(cols)) / len(cols) if cols else 0.0
    column_entropy = _shannon_entropy(Counter(cols))
    tile_entropy = _shannon_entropy(tile_counts)
    compression_ratio = _compression_ratio(rows)

    # Operational leniency (within-study formulation, NOT cross-paper): reward
    # forgiving content, penalise hazards. Bounded, descriptive.
    leniency = (
        1.0 * reward_density
        - 1.0 * gap_fraction
        - 1.0 * enemy_density
        - 0.5 * (max_gap_width / w if w else 0.0)
    )

    return {
        "width": float(w),
        "height": float(h),
        "solid_density": solid_density,
        "breakable_density": breakable_density,
        "question_density": question_density,
        "coin_density": coin_density,
        "powerup_density": powerup_density,
        "pipe_density": pipe_density,
        "reward_density": reward_density,
        "reward_count": reward_count,
        "enemy_density": enemy_density,
        "enemy_goomba_density": enemy_group_density["goomba"],
        "enemy_koopa_density": enemy_group_density["koopa"],
        "enemy_spiky_density": enemy_group_density["spiky"],
        "enemy_bullet_density": enemy_group_density["bullet"],
        "enemy_types_present": enemy_types_present,
        "gap_count": float(gap_count),
        "max_gap_width": float(max_gap_width),
        "gap_fraction": gap_fraction,
        "gap_frequency": gap_frequency,
        "surface_linearity_r2": r2,
        "surface_mean_abs_resid": mean_abs_resid,
        "surface_roughness": roughness,
        "surface_mean_height": mean_height,
        "surface_height_range": height_range,
        "vertical_solid_max": float(max_stack),
        "vertical_solid_mean": vertical_solid_mean,
        "air_fraction": air_fraction,
        "unique_column_ratio": unique_column_ratio,
        "column_entropy": column_entropy,
        "tile_entropy": tile_entropy,
        "compression_ratio": compression_ratio,
        "leniency": leniency,
    }


def feature_vector(rows: list[str]) -> list[float]:
    """Return features as a list in :data:`FEATURE_NAMES` order."""
    feats = extract_features(rows)
    return [feats[name] for name in FEATURE_NAMES]
