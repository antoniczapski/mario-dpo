#!/usr/bin/env python3
"""
Replanned EDA pipeline for PCG Arena.

Generates derived analysis tables and thesis-ready figures for the revised
EDA/User Study chapter. Figures are written directly to latex/img.
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
import random
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "eda" / "data_10_05_2026"
OUT_DIR = ROOT / "eda" / "07_replanned_analysis" / "outputs"
# The thesis (latex/masters_thesis/main.tex) uses \graphicspath{{img/}}, so it
# loads figures from latex/masters_thesis/img. Write there directly. latex/img
# is a legacy, git-tracked copy that no document reads from; we keep it in sync
# (see sync_legacy_images) so the two tracked locations never diverge.
IMG_DIR = ROOT / "latex" / "masters_thesis" / "img"
LEGACY_IMG_DIR = ROOT / "latex" / "img"
VERIFIER_FILE = ROOT / "verifier_agent.md"
SEED_LEVEL_DIR = ROOT / "db" / "seed" / "levels"
MARIODPO_LEVEL_DIR = ROOT / "MarioDPO" / "generated_levels"

RANDOM_SEED = 20260510
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EXCLUDE_GENERATORS = {"test-gen"}

# Active tag vocabulary (7 tags). The deprecated tags good_flow, unfair,
# confusing, and not_mario_like were removed from the platform; they are
# excluded here so leftover traces in the database are never plotted.
TAG_NAMES = [
    "fun",
    "boring",
    "too_hard",
    "too_easy",
    "creative",
    "impossible",
    "broken_graphics",
]

GENERATOR_FAMILIES = {
    "original": "human",
    "mariodpo": "preference-trained",
    "mariogpt": "neural",
    "mariogan": "neural",
    "marioDiffusion": "neural",
    "genetic": "search",
    "ore": "constructive/search",
    "hopper": "constructive",
    "notch": "constructive",
    "notchParam": "constructive",
    "notchParamRand": "constructive",
    "patternCount": "pattern",
    "patternOccur": "pattern",
    "patternWeightCount": "pattern",
}

# Conservative tile classes. The Mario AI format has several variants; the goal
# here is consistent generator-level characterization, not engine-level physics.
SOLID_CHARS = set("X#@BCDULRS?Q!12<>[]tT|%")
BLOCK_CHARS = set("S?Q!12")
PIPE_CHARS = set("<>[]tT|")
ENEMY_CHARS = set("gGkKrRyYE")
REWARD_CHARS = set("o*Q?!12")
EMPTY_CHARS = set("-")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_datetime(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_IMG_DIR.mkdir(parents=True, exist_ok=True)


def sync_legacy_images(plot_paths: list[Path]) -> None:
    """Mirror generated figures into the legacy latex/img directory so the two
    git-tracked image locations stay identical."""
    import shutil

    for src in plot_paths:
        dst = LEGACY_IMG_DIR / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


@dataclass
class LoadedData:
    votes_envelope: dict[str, Any]
    players_envelope: dict[str, Any]
    levels_envelope: dict[str, Any]
    trajectories_envelope: dict[str, Any]
    votes: list[dict[str, Any]]
    players: list[dict[str, Any]]
    levels: list[dict[str, Any]]
    trajectories: list[dict[str, Any]]


def load_data() -> LoadedData:
    votes_env = load_json(DATA_DIR / "pcg-arena-votes-2026-05-10.json")
    players_env = load_json(DATA_DIR / "pcg-arena-player-profiles-2026-05-10.json")
    levels_env = load_json(DATA_DIR / "pcg-arena-level-stats-2026-05-10.json")
    traj_env = load_json(DATA_DIR / "pcg-arena-trajectories-2026-05-10.json")
    return LoadedData(
        votes_envelope=votes_env,
        players_envelope=players_env,
        levels_envelope=levels_env,
        trajectories_envelope=traj_env,
        votes=votes_env.get("data", []),
        players=players_env.get("data", []),
        levels=levels_env.get("data", []),
        trajectories=traj_env.get("data", []),
    )


def normalize_side_rows(votes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for vote in votes:
        result = vote.get("result")
        for side in ("left", "right"):
            other = "right" if side == "left" else "left"
            side_upper = side.upper()
            other_upper = other.upper()
            telemetry = (vote.get("telemetry") or {}).get(side) or {}
            tags = vote.get(f"{side}_tags") or []
            if result == side_upper:
                score = 1.0
                result_for_side = "win"
            elif result == other_upper:
                score = 0.0
                result_for_side = "loss"
            elif result == "TIE":
                score = 0.5
                result_for_side = "tie"
            elif result == "SKIP":
                score = np.nan
                result_for_side = "skip"
            else:
                score = np.nan
                result_for_side = "unknown"

            trajectory = telemetry.get("trajectory") or []
            events = telemetry.get("events") or []
            death_locations = telemetry.get("death_locations") or []
            max_x = max((p.get("x", 0.0) for p in trajectory), default=np.nan)
            min_x = min((p.get("x", 0.0) for p in trajectory), default=np.nan)
            ys = [p.get("y", np.nan) for p in trajectory]
            xs = [p.get("x", np.nan) for p in trajectory]
            y_std = float(np.nanstd(ys)) if ys else np.nan
            x_std = float(np.nanstd(xs)) if xs else np.nan
            traj_len = len(trajectory)
            duration = telemetry.get("duration_seconds")
            deaths = telemetry.get("deaths")
            completed = bool(telemetry.get("completed")) if telemetry else False
            played = bool(telemetry.get("played", bool(trajectory) or duration is not None))

            row = {
                "vote_id": vote.get("vote_id"),
                "battle_id": vote.get("battle_id"),
                "session_id": vote.get("session_id"),
                "player_id": vote.get("player_id"),
                "created_at_utc": vote.get("created_at_utc"),
                "created_at": safe_datetime(vote.get("created_at_utc")),
                "side": side,
                "generator_id": vote.get(f"{side}_generator_id"),
                "level_id": vote.get(f"{side}_level_id"),
                "opponent_generator_id": vote.get(f"{other}_generator_id"),
                "opponent_level_id": vote.get(f"{other}_level_id"),
                "vote_result": result,
                "result_for_side": result_for_side,
                "score_for_side": score,
                "played": played,
                "skipped_play": bool(telemetry.get("skipped", False)) if telemetry else False,
                "completed": completed,
                "deaths": int(deaths) if deaths is not None else 0,
                "died": int((deaths or 0) > 0 or len(death_locations) > 0),
                "duration_seconds": float(duration) if duration is not None else np.nan,
                "jumps": telemetry.get("jumps", np.nan),
                "coins_collected": telemetry.get("coins_collected", np.nan),
                "enemies_killed": telemetry.get("enemies_killed", np.nan),
                "trajectory": trajectory,
                "trajectory_length": traj_len,
                "trajectory_max_x": max_x,
                "trajectory_min_x": min_x,
                "trajectory_y_std": y_std,
                "trajectory_x_std": x_std,
                "events": events,
                "event_count": len(events),
                "death_locations": death_locations,
                "death_count_locations": len(death_locations),
                "tags": tags,
                "tag_count": len(tags),
            }
            for tag in TAG_NAMES:
                row[f"tag_{tag}"] = int(tag in tags)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df[~df["generator_id"].isin(EXCLUDE_GENERATORS)].copy()
    return df


def build_vote_level_rows(votes: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for vote in votes:
        if vote.get("left_generator_id") in EXCLUDE_GENERATORS or vote.get("right_generator_id") in EXCLUDE_GENERATORS:
            continue
        rows.append(
            {
                "vote_id": vote.get("vote_id"),
                "player_id": vote.get("player_id"),
                "session_id": vote.get("session_id"),
                "created_at": safe_datetime(vote.get("created_at_utc")),
                "result": vote.get("result"),
                "left_generator_id": vote.get("left_generator_id"),
                "right_generator_id": vote.get("right_generator_id"),
                "left_level_id": vote.get("left_level_id"),
                "right_level_id": vote.get("right_level_id"),
                "left_tags": vote.get("left_tags") or [],
                "right_tags": vote.get("right_tags") or [],
            }
        )
    return pd.DataFrame(rows)


def generator_records(side_df: pd.DataFrame) -> pd.DataFrame:
    valid = side_df[~side_df["score_for_side"].isna()].copy()
    grouped = valid.groupby("generator_id")
    rows = []
    for gen, g in grouped:
        outcomes = g["result_for_side"].value_counts().to_dict()
        shown = len(side_df[side_df["generator_id"] == gen])
        skips = int((side_df[side_df["generator_id"] == gen]["result_for_side"] == "skip").sum())
        rows.append(
            {
                "generator_id": gen,
                "family": GENERATOR_FAMILIES.get(gen, "other"),
                "score_sum": float(g["score_for_side"].sum()),
                "decisive_or_tie": len(g),
                "shown": shown,
                "wins": int(outcomes.get("win", 0)),
                "losses": int(outcomes.get("loss", 0)),
                "ties": int(outcomes.get("tie", 0)),
                "skips": skips,
                "score_rate": float(g["score_for_side"].mean()),
                "completion_rate": float(g["completed"].mean()),
                "death_rate": float(g["died"].mean()),
                "median_duration": float(g["duration_seconds"].clip(upper=g["duration_seconds"].quantile(0.99)).median()),
                "mean_progress_px": float(g["trajectory_max_x"].mean()),
            }
        )
    rec = pd.DataFrame(rows).sort_values("score_rate", ascending=False).reset_index(drop=True)
    rec["rank"] = np.arange(1, len(rec) + 1)
    return rec


def bootstrap_score_ci(vote_df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    gens = sorted(set(vote_df["left_generator_id"]).union(set(vote_df["right_generator_id"])) - EXCLUDE_GENERATORS)
    scores = {g: [] for g in gens}
    records = vote_df.to_dict("records")
    n = len(records)
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sums = defaultdict(float)
        counts = defaultdict(int)
        for idx in sample_idx:
            r = records[int(idx)]
            left = r["left_generator_id"]
            right = r["right_generator_id"]
            result = r["result"]
            if result == "SKIP":
                continue
            if result == "LEFT":
                sums[left] += 1.0
                sums[right] += 0.0
                counts[left] += 1
                counts[right] += 1
            elif result == "RIGHT":
                sums[left] += 0.0
                sums[right] += 1.0
                counts[left] += 1
                counts[right] += 1
            elif result == "TIE":
                sums[left] += 0.5
                sums[right] += 0.5
                counts[left] += 1
                counts[right] += 1
        for g in gens:
            if counts[g] > 0:
                scores[g].append(sums[g] / counts[g])
    rows = []
    for gen, vals in scores.items():
        if vals:
            rows.append(
                {
                    "generator_id": gen,
                    "score_ci_low": float(np.percentile(vals, 2.5)),
                    "score_ci_high": float(np.percentile(vals, 97.5)),
                    "score_ci_sd": float(np.std(vals)),
                }
            )
    return pd.DataFrame(rows)


def bradley_terry(vote_df: pd.DataFrame) -> pd.DataFrame:
    gens = sorted(set(vote_df["left_generator_id"]).union(set(vote_df["right_generator_id"])) - EXCLUDE_GENERATORS)
    idx = {g: i for i, g in enumerate(gens)}
    comps = []
    for _, row in vote_df.iterrows():
        if row["result"] == "SKIP":
            continue
        l = row["left_generator_id"]
        r = row["right_generator_id"]
        if l not in idx or r not in idx:
            continue
        if row["result"] == "LEFT":
            y = 1.0
        elif row["result"] == "RIGHT":
            y = 0.0
        elif row["result"] == "TIE":
            y = 0.5
        else:
            continue
        comps.append((idx[l], idx[r], y))

    def nll(theta_raw: np.ndarray) -> float:
        theta = theta_raw - theta_raw.mean()
        total = 0.0
        for i, j, y in comps:
            diff = theta[i] - theta[j]
            # stable log sigmoid
            logp = -np.logaddexp(0, -diff)
            log1p = -np.logaddexp(0, diff)
            total -= y * logp + (1.0 - y) * log1p
        total += 0.01 * float(np.sum(theta**2))
        return total

    res = minimize(nll, np.zeros(len(gens)), method="BFGS")
    theta = res.x - res.x.mean()
    # Scale for display only; ranking is invariant.
    rating = 1000 + 173.7178 * theta
    out = pd.DataFrame({"generator_id": gens, "bt_theta": theta, "bt_rating": rating})
    return out.sort_values("bt_rating", ascending=False).reset_index(drop=True)


def pairwise_matrix(vote_df: pd.DataFrame, order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sums = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    counts = pd.DataFrame(0, index=order, columns=order, dtype=int)
    score_sum = defaultdict(float)
    count = defaultdict(int)
    for _, row in vote_df.iterrows():
        l = row["left_generator_id"]
        r = row["right_generator_id"]
        if l not in order or r not in order or row["result"] == "SKIP":
            continue
        if row["result"] == "LEFT":
            ls, rs = 1.0, 0.0
        elif row["result"] == "RIGHT":
            ls, rs = 0.0, 1.0
        elif row["result"] == "TIE":
            ls, rs = 0.5, 0.5
        else:
            continue
        score_sum[(l, r)] += ls
        count[(l, r)] += 1
        score_sum[(r, l)] += rs
        count[(r, l)] += 1
    for i in order:
        for j in order:
            if i == j:
                sums.loc[i, j] = np.nan
            elif count[(i, j)] > 0:
                sums.loc[i, j] = score_sum[(i, j)] / count[(i, j)]
                counts.loc[i, j] = count[(i, j)]
    return sums, counts


def read_level(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip("\n")


def iter_level_files() -> Iterable[tuple[str, str, Path, str]]:
    if SEED_LEVEL_DIR.exists():
        for gen_dir in sorted(SEED_LEVEL_DIR.iterdir()):
            if not gen_dir.is_dir():
                continue
            gen = gen_dir.name
            for path in sorted(gen_dir.glob("*.txt")):
                level_id = f"{gen}::{path.name}"
                yield gen, level_id, path, read_level(path)
    if MARIODPO_LEVEL_DIR.exists():
        for path in sorted(MARIODPO_LEVEL_DIR.glob("*.txt"))[:200]:
            level_id = f"mariodpo::{path.name}"
            yield "mariodpo", level_id, path, read_level(path)


def level_lines(tilemap: str) -> list[str]:
    lines = [line.rstrip("\n") for line in tilemap.splitlines() if line.rstrip("\n")]
    if not lines:
        return []
    width = max(len(line) for line in lines)
    return [line.ljust(width, "-") for line in lines]


def surface_profile(lines: list[str]) -> np.ndarray:
    if not lines:
        return np.array([])
    height = len(lines)
    width = max(len(line) for line in lines)
    ys = []
    for x in range(width):
        y_candidates = []
        for y in range(height):
            char = lines[y][x] if x < len(lines[y]) else "-"
            above = lines[y - 1][x] if y > 0 and x < len(lines[y - 1]) else "-"
            if char in SOLID_CHARS and above not in SOLID_CHARS:
                y_candidates.append(y)
        if y_candidates:
            # Use the lowest visible surface rather than tiny floating decor, but
            # still capture platforms by taking the median top surface if multiple.
            ys.append(float(np.median(y_candidates)))
        else:
            ys.append(np.nan)
    return np.array(ys, dtype=float)


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return np.nan
    return len(gzip.compress(raw)) / len(raw)


def ncd(a: str, b: str) -> float:
    ba = a.encode("utf-8")
    bb = b.encode("utf-8")
    ca = len(gzip.compress(ba))
    cb = len(gzip.compress(bb))
    cab = len(gzip.compress(ba + b"\n" + bb))
    denom = max(ca, cb)
    if denom == 0:
        return np.nan
    return (cab - min(ca, cb)) / denom


def shannon_entropy(items: list[Any]) -> float:
    if not items:
        return 0.0
    counts = np.array(list(Counter(items).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def compute_level_metrics() -> pd.DataFrame:
    rows = []
    all_levels_by_gen: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for gen, level_id, path, text in iter_level_files():
        if gen in EXCLUDE_GENERATORS:
            continue
        lines = level_lines(text)
        height = len(lines)
        width = max((len(line) for line in lines), default=0)
        chars = [ch for line in lines for ch in line]
        total_tiles = len(chars) or 1
        solid_count = sum(ch in SOLID_CHARS for ch in chars)
        enemy_count = sum(ch in ENEMY_CHARS for ch in chars)
        reward_count = sum(ch in REWARD_CHARS for ch in chars)
        block_count = sum(ch in BLOCK_CHARS for ch in chars)
        pipe_count = sum(ch in PIPE_CHARS for ch in chars)
        empty_count = sum(ch in EMPTY_CHARS for ch in chars)

        # Gaps are columns without solid ground support in the bottom two rows.
        gap_cols = []
        for x in range(width):
            bottom_solid = False
            for y in range(max(0, height - 2), height):
                if x < len(lines[y]) and lines[y][x] in SOLID_CHARS:
                    bottom_solid = True
            gap_cols.append(not bottom_solid)
        gap_widths = []
        current = 0
        for is_gap in gap_cols:
            if is_gap:
                current += 1
            elif current:
                gap_widths.append(current)
                current = 0
        if current:
            gap_widths.append(current)

        prof = surface_profile(lines)
        valid = ~np.isnan(prof)
        if valid.sum() >= 3:
            x = np.arange(len(prof))[valid]
            y = prof[valid]
            coeff = np.polyfit(x, y, 1)
            pred = coeff[0] * x + coeff[1]
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            linearity_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            surface_resid_std = float(np.sqrt(ss_res / max(1, len(y) - 2)))
            surface_std = float(np.std(y))
        else:
            linearity_r2 = np.nan
            surface_resid_std = np.nan
            surface_std = np.nan

        columns = ["".join(line[x] for line in lines if x < len(line)) for x in range(width)]
        unique_column_ratio = len(set(columns)) / width if width else np.nan
        gap_density = sum(gap_widths) / width if width else np.nan
        enemy_density = enemy_count / width if width else np.nan
        reward_density = reward_count / width if width else np.nan
        solid_density = solid_count / total_tiles
        hazard = 2.5 * (gap_density or 0) + 0.75 * (enemy_density or 0) + 0.05 * (max(gap_widths) if gap_widths else 0)
        reward = 0.35 * (reward_density or 0) + 0.02 * block_count / max(width, 1)
        leniency = 1.0 / (1.0 + hazard) + reward
        leniency = float(max(0.0, min(1.5, leniency)))

        row = {
            "generator_id": gen,
            "level_id": level_id,
            "path": str(path),
            "width": width,
            "height": height,
            "solid_density": solid_density,
            "empty_density": empty_count / total_tiles,
            "enemy_count": enemy_count,
            "enemy_density": enemy_density,
            "reward_count": reward_count,
            "reward_density": reward_density,
            "block_density": block_count / width if width else np.nan,
            "pipe_density": pipe_count / width if width else np.nan,
            "gap_count": len(gap_widths),
            "gap_density": gap_density,
            "max_gap_width": max(gap_widths) if gap_widths else 0,
            "linearity_r2": linearity_r2,
            "nonlinearity": 1.0 - linearity_r2 if not np.isnan(linearity_r2) else np.nan,
            "surface_resid_std": surface_resid_std,
            "surface_std": surface_std,
            "leniency": leniency,
            "tile_entropy": shannon_entropy(chars),
            "column_entropy": shannon_entropy(columns),
            "unique_column_ratio": unique_column_ratio,
            "compression_ratio": compression_ratio(text),
            "text": text,
        }
        rows.append(row)
        all_levels_by_gen[gen].append((level_id, text))

    metrics = pd.DataFrame(rows)

    # NCD summaries. Limit pair counts deterministically for speed and stability.
    rng = random.Random(RANDOM_SEED)
    ncd_rows = []
    original_texts = [text for _, text in all_levels_by_gen.get("original", [])]
    for gen, items in all_levels_by_gen.items():
        sample = items[:]
        if len(sample) > 60:
            sample = rng.sample(sample, 60)
        pair_vals = []
        for (_, a), (_, b) in itertools.combinations(sample, 2):
            pair_vals.append(ncd(a, b))
        dist_orig = []
        if original_texts:
            sample_for_orig = sample[:40] if len(sample) > 40 else sample
            for _, text in sample_for_orig:
                for orig in original_texts:
                    dist_orig.append(ncd(text, orig))
        ncd_rows.append(
            {
                "generator_id": gen,
                "within_ncd": float(np.nanmean(pair_vals)) if pair_vals else np.nan,
                "distance_to_original_ncd": float(np.nanmean(dist_orig)) if dist_orig else np.nan,
                "ncd_pairs": len(pair_vals),
            }
        )
    ncd_df = pd.DataFrame(ncd_rows)
    metrics = metrics.merge(ncd_df, on="generator_id", how="left")

    feature_cols = [
        "solid_density",
        "enemy_density",
        "reward_density",
        "gap_density",
        "linearity_r2",
        "surface_std",
        "leniency",
        "tile_entropy",
        "unique_column_ratio",
        "compression_ratio",
    ]
    orig = metrics[metrics["generator_id"] == "original"][feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(orig) > 0:
        centroid = orig.mean().values
        scale = orig.std().replace(0, 1).fillna(1).values
        vals = metrics[feature_cols].replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(vals.mean())
        metrics["distance_to_original_centroid"] = np.linalg.norm((vals.values - centroid) / scale, axis=1)
    else:
        metrics["distance_to_original_centroid"] = np.nan
    return metrics.drop(columns=["text"]), metrics[["generator_id", "level_id", "text"]]


def summarize_static_metrics(level_metrics: pd.DataFrame) -> pd.DataFrame:
    agg = level_metrics.groupby("generator_id").agg(
        levels=("level_id", "count"),
        linearity_r2_mean=("linearity_r2", "mean"),
        linearity_r2_std=("linearity_r2", "std"),
        leniency_mean=("leniency", "mean"),
        leniency_std=("leniency", "std"),
        solid_density_mean=("solid_density", "mean"),
        enemy_density_mean=("enemy_density", "mean"),
        reward_density_mean=("reward_density", "mean"),
        gap_density_mean=("gap_density", "mean"),
        max_gap_width_mean=("max_gap_width", "mean"),
        tile_entropy_mean=("tile_entropy", "mean"),
        unique_column_ratio_mean=("unique_column_ratio", "mean"),
        compression_ratio_mean=("compression_ratio", "mean"),
        within_ncd=("within_ncd", "first"),
        distance_to_original_ncd=("distance_to_original_ncd", "first"),
        distance_to_original_centroid_mean=("distance_to_original_centroid", "mean"),
    )
    return agg.reset_index()


def trajectory_metrics(side_df: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gen, g in side_df.groupby("generator_id"):
        trajectories = [t for t in g["trajectory"].tolist() if isinstance(t, list) and len(t) > 0]
        if not trajectories:
            continue
        all_points = []
        visited_sets = []
        max_xs = []
        y_stds = []
        death_xs = []
        for traj, death_locs in zip(g["trajectory"], g["death_locations"]):
            if not isinstance(traj, list) or len(traj) == 0:
                continue
            xs = np.array([p.get("x", np.nan) for p in traj], dtype=float)
            ys = np.array([p.get("y", np.nan) for p in traj], dtype=float)
            valid = ~(np.isnan(xs) | np.isnan(ys))
            xs, ys = xs[valid], ys[valid]
            if len(xs) == 0:
                continue
            max_x = float(np.nanmax(xs))
            max_xs.append(max_x)
            y_stds.append(float(np.nanstd(ys)))
            # Normalize to fixed grid using observed practical range.
            xnorm = np.clip(xs / 3200.0, 0, 1)
            ynorm = np.clip(ys / 256.0, 0, 1)
            xi = np.floor(xnorm * 40).astype(int).clip(0, 39)
            yi = np.floor(ynorm * 16).astype(int).clip(0, 15)
            cells = set(zip(xi.tolist(), yi.tolist()))
            visited_sets.append(cells)
            all_points.extend(cells)
            if isinstance(death_locs, list):
                for d in death_locs:
                    if isinstance(d, dict) and d.get("x") is not None:
                        death_xs.append(float(d.get("x")))
        occ_counts = Counter(all_points)
        probs = np.array(list(occ_counts.values()), dtype=float)
        probs = probs / probs.sum() if probs.sum() else probs
        occ_entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else np.nan
        occ_area = len(occ_counts) / (40 * 16)
        # Sample Jaccard distances to avoid quadratic blowup.
        jaccards = []
        if len(visited_sets) >= 2:
            pairs = list(itertools.combinations(range(len(visited_sets)), 2))
            if len(pairs) > 2000:
                pairs = random.sample(pairs, 2000)
            for i, j in pairs:
                a, b = visited_sets[i], visited_sets[j]
                union = len(a | b)
                if union:
                    jaccards.append(1 - len(a & b) / union)
        if death_xs:
            bins = np.histogram(np.clip(np.array(death_xs) / 3200.0, 0, 1), bins=10, range=(0, 1))[0]
            death_probs = bins[bins > 0] / bins.sum()
            death_entropy = float(-(death_probs * np.log2(death_probs)).sum()) if len(death_probs) else np.nan
            max_death_bin_share = float(bins.max() / bins.sum()) if bins.sum() else np.nan
            early_death_rate = float((np.array(death_xs) < 800).mean())
        else:
            death_entropy = np.nan
            max_death_bin_share = np.nan
            early_death_rate = np.nan
        rows.append(
            {
                "generator_id": gen,
                "trajectory_count": len(trajectories),
                "occupancy_entropy": occ_entropy,
                "occupancy_area": occ_area,
                "median_max_x": float(np.nanmedian(max_xs)) if max_xs else np.nan,
                "mean_verticality": float(np.nanmean(y_stds)) if y_stds else np.nan,
                "path_jaccard_diversity": float(np.nanmean(jaccards)) if jaccards else np.nan,
                "death_entropy": death_entropy,
                "max_death_bin_share": max_death_bin_share,
                "early_death_rate": early_death_rate,
            }
        )
    tm = pd.DataFrame(rows)
    return tm.merge(ranking[["generator_id", "score_rate", "bt_rating", "rank"]], on="generator_id", how="left")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_study_overview(audit: dict[str, Any], plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_study_overview.png"
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    boxes = [
        (0.06, 0.68, "Anonymous\nplayer IDs", f"{audit['player_profiles']} profiles"),
        (0.31, 0.68, "Votes", f"{audit['vote_rows']} vote rows"),
        (0.56, 0.68, "Played sides", f"{audit['side_rows']} side records"),
        (0.81, 0.68, "Levels", f"{audit['level_stats_rows']} level stats"),
        (0.31, 0.23, "Tags", f"{audit['tag_assignments']} tag assignments"),
        (0.56, 0.23, "Telemetry", f"{audit['embedded_trajectories']} trajectories"),
        (0.81, 0.23, "Static level text", "expressive metrics"),
    ]
    for x, y, title, subtitle in boxes:
        rect = plt.Rectangle((x, y), 0.16, 0.16, facecolor="#f6f8fa", edgecolor="#444", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.08, y + 0.102, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.08, y + 0.047, subtitle, ha="center", va="center", fontsize=8, color="#555")
    arrows = [
        ((0.22, 0.76), (0.31, 0.76)),
        ((0.47, 0.76), (0.56, 0.76)),
        ((0.72, 0.76), (0.81, 0.76)),
        ((0.39, 0.68), (0.39, 0.39)),
        ((0.64, 0.68), (0.64, 0.39)),
        ((0.89, 0.68), (0.89, 0.39)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))
    ax.text(
        0.5,
        0.04,
        "Analysis joins pairwise preferences, optional qualitative tags, gameplay telemetry, trajectories, and static level structure.",
        ha="center",
        fontsize=10,
    )
    savefig(path)
    plot_paths.append(path)


def plot_engagement(vote_df: pd.DataFrame, side_df: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_engagement.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    daily = vote_df.assign(date=vote_df["created_at"].dt.date).groupby("date").size()
    axes[0].bar(pd.to_datetime(daily.index), daily.values, color="#4477AA")
    axes[0].set_title("Votes over time")
    axes[0].set_ylabel("Votes")
    axes[0].tick_params(axis="x", rotation=35)

    votes_per_player = vote_df.groupby("player_id").size().sort_values(ascending=False)
    axes[1].hist(votes_per_player.values, bins=min(25, max(5, len(votes_per_player) // 2)), color="#66AA55", edgecolor="white")
    axes[1].set_title("Votes per anonymous player ID")
    axes[1].set_xlabel("Votes")
    axes[1].set_ylabel("Player IDs")
    axes[1].set_yscale("log")

    sorted_votes = votes_per_player.values
    cumulative = np.cumsum(sorted_votes) / sorted_votes.sum()
    axes[2].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3, color="#AA3377")
    axes[2].set_title("Cumulative vote contribution")
    axes[2].set_xlabel("Top N player IDs")
    axes[2].set_ylabel("Fraction of votes")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(True, alpha=0.3)
    savefig(path)
    plot_paths.append(path)


def plot_generator_ranking(ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_generator_ranking.png"
    df = ranking.sort_values("score_rate", ascending=True).copy()
    colors = df["family"].map(
        {
            "human": "#222222",
            "preference-trained": "#CC6677",
            "neural": "#4477AA",
            "constructive/search": "#66AA55",
            "constructive": "#66AA55",
            "search": "#66AA55",
            "pattern": "#DDCC77",
        }
    ).fillna("#999999")
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    xerr_low = (df["score_rate"] - df["score_ci_low"]).clip(lower=0)
    xerr_high = (df["score_ci_high"] - df["score_rate"]).clip(lower=0)
    ax.barh(df["generator_id"], df["score_rate"], color=colors, edgecolor="#333", alpha=0.9)
    ax.errorbar(df["score_rate"], df["generator_id"], xerr=[xerr_low, xerr_high], fmt="none", ecolor="#222", capsize=3, lw=1)
    ax.axvline(0.5, color="#666", lw=1, ls="--")
    ax.set_xlabel("Pairwise preference score (win=1, tie=0.5, loss=0)")
    ax.set_ylabel("Generator")
    ax.set_title("Generator ranking from blind pairwise votes")
    ax.set_xlim(0, 1.0)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#222222", edgecolor="#333", label="Human-authored"),
        Patch(facecolor="#CC6677", edgecolor="#333", label="Preference-trained (MarioDPO)"),
        Patch(facecolor="#4477AA", edgecolor="#333", label="Neural (GAN / LLM / diffusion)"),
        Patch(facecolor="#66AA55", edgecolor="#333", label="Constructive / search"),
        Patch(facecolor="#DDCC77", edgecolor="#333", label="Pattern-based"),
        Line2D([0], [0], color="#666", lw=1, ls="--", label="Neutral score (0.5)"),
    ]
    ax.legend(
        handles=legend_handles,
        title="Generator family",
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.95,
    )
    savefig(path)
    plot_paths.append(path)


def plot_pairwise_confusion(matrix: pd.DataFrame, counts: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_pairwise_confusion.png"
    cmap = LinearSegmentedColormap.from_list("row_win", ["#8c2d04", "#f7f7f7", "#2ca25f"])
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    annot = matrix.copy().astype(object)
    for i in matrix.index:
        for j in matrix.columns:
            if i == j or pd.isna(matrix.loc[i, j]):
                annot.loc[i, j] = ""
            else:
                annot.loc[i, j] = f"{matrix.loc[i, j]:.2f}\n(n={counts.loc[i,j]})"
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row-generator score rate"},
    )
    ax.set_title("Pairwise generator preference matrix")
    ax.set_xlabel("Column generator")
    ax.set_ylabel("Row generator")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    savefig(path)
    plot_paths.append(path)


def plot_expressive_range(level_metrics: pd.DataFrame, order: list[str], ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_expressive_range.png"
    gens = [g for g in order if g in set(level_metrics["generator_id"])]
    ncols = 4
    nrows = math.ceil(len(gens) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.1 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    rate = ranking.set_index("generator_id")["score_rate"].to_dict()
    for ax, gen in zip(axes, gens):
        g = level_metrics[level_metrics["generator_id"] == gen]
        ax.hist2d(
            g["linearity_r2"].clip(0, 1),
            g["leniency"].clip(0, 1.5),
            bins=20,
            range=[[0, 1], [0, 1.5]],
            cmap="Blues",
            cmin=1,
        )
        ax.scatter(g["linearity_r2"].clip(0, 1), g["leniency"].clip(0, 1.5), s=6, color="#111", alpha=0.18)
        ax.set_title(f"{gen}\nscore={rate.get(gen, np.nan):.2f}, n={len(g)}")
        ax.grid(False)
    for ax in axes[len(gens) :]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Linearity (R²)")
    for ax in axes[::ncols]:
        ax.set_ylabel("Operational leniency")
    fig.suptitle("Expressive range: static level structure by generator", y=1.005, fontsize=13)
    savefig(path)
    plot_paths.append(path)


def plot_static_metrics_vs_rating(static_gen: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_static_metrics_vs_rating.png"
    df = static_gen.merge(ranking[["generator_id", "score_rate", "bt_rating", "family"]], on="generator_id", how="inner")
    metrics = [
        ("gap_density_mean", "Gap density"),
        ("enemy_density_mean", "Enemy density"),
        ("leniency_mean", "Leniency"),
        ("within_ncd", "Within-generator NCD"),
        ("distance_to_original_ncd", "Distance to original (NCD)"),
        ("unique_column_ratio_mean", "Unique column ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    axes = axes.reshape(-1)
    for ax, (col, title) in zip(axes, metrics):
        sns.regplot(data=df, x=col, y="score_rate", ax=ax, scatter=False, color="#666", ci=None, line_kws={"ls": "--", "lw": 1})
        sns.scatterplot(data=df, x=col, y="score_rate", hue="family", ax=ax, s=60, edgecolor="#222", legend=False)
        for _, row in df.iterrows():
            ax.text(row[col], row["score_rate"] + 0.012, row["generator_id"], fontsize=7, ha="center")
        ax.set_title(title)
        ax.set_ylabel("Preference score")
        ax.set_xlabel(title)
        ax.set_ylim(0.15, 0.92)
    savefig(path)
    plot_paths.append(path)


def plot_static_metrics_table(static_gen: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_static_metrics_table.png"
    df = static_gen.merge(ranking[["generator_id", "score_rate", "rank"]], on="generator_id", how="left")
    df = df.sort_values("rank")
    display = pd.DataFrame(
        {
            "Generator": df["generator_id"],
            "Score": df["score_rate"].map(lambda x: f"{x:.2f}"),
            "Levels": df["levels"].astype(int).astype(str),
            "Linearity": df["linearity_r2_mean"].map(lambda x: f"{x:.2f}"),
            "Leniency": df["leniency_mean"].map(lambda x: f"{x:.2f}"),
            "Gap dens.": df["gap_density_mean"].map(lambda x: f"{x:.2f}"),
            "Enemy dens.": df["enemy_density_mean"].map(lambda x: f"{x:.2f}"),
            "NCD": df["within_ncd"].map(lambda x: f"{x:.2f}"),
        }
    )
    # Raw values used to shade the five metric columns from red (low) to green
    # (high) within each column, as a per-column reading aid.
    metric_values = {
        "Linearity": df["linearity_r2_mean"].to_numpy(dtype=float),
        "Leniency": df["leniency_mean"].to_numpy(dtype=float),
        "Gap dens.": df["gap_density_mean"].to_numpy(dtype=float),
        "Enemy dens.": df["enemy_density_mean"].to_numpy(dtype=float),
        "NCD": df["within_ncd"].to_numpy(dtype=float),
    }
    metric_ranges = {name: (np.nanmin(v), np.nanmax(v)) for name, v in metric_values.items()}
    shade_cmap = plt.get_cmap("RdYlGn")
    col_widths = [0.20, 0.10, 0.10, 0.12, 0.12, 0.12, 0.13, 0.11]

    fig, ax = plt.subplots(figsize=(12.5, 0.5 * len(display) + 1.0))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    columns = list(display.columns)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")
            continue
        col_name = columns[col]
        if col_name in metric_values:
            vmin, vmax = metric_ranges[col_name]
            value = metric_values[col_name][row - 1]
            if np.isnan(value) or vmax <= vmin:
                t = 0.5
            else:
                t = (value - vmin) / (vmax - vmin)
            cell.set_facecolor(shade_cmap(0.15 + 0.70 * t))
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    savefig(path)
    plot_paths.append(path)


def plot_trajectory_stacks(side_df: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_trajectory_stacks.png"
    order = ranking["generator_id"].tolist()
    ncols = 4
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 2.9 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    rate = ranking.set_index("generator_id")["score_rate"].to_dict()
    for ax, gen in zip(axes, order):
        g = side_df[(side_df["generator_id"] == gen) & (side_df["trajectory_length"] > 0)]
        alpha = min(0.22, max(0.035, 6 / max(len(g), 1)))
        for traj in g["trajectory"].tolist():
            xs = [p.get("x", np.nan) for p in traj]
            ys = [p.get("y", np.nan) for p in traj]
            ax.plot(xs, ys, color="#c9252d", lw=0.45, alpha=alpha)
        ax.set_title(f"{gen}\nscore={rate.get(gen, np.nan):.2f}, traces={len(g)}")
        ax.set_xlim(0, 3300)
        ax.set_ylim(260, 0)
        ax.set_facecolor("white")
        ax.grid(False)
    for ax in axes[len(order) :]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("x position (px)")
    for ax in axes[::ncols]:
        ax.set_ylabel("y position (px)")
    fig.suptitle("Gameplay trajectory stacks by generator", y=1.005, fontsize=13)
    savefig(path)
    plot_paths.append(path)


def plot_trajectory_occupancy(side_df: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_trajectory_occupancy.png"
    order = ranking["generator_id"].tolist()
    ncols = 4
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 2.9 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    rate = ranking.set_index("generator_id")["score_rate"].to_dict()
    for ax, gen in zip(axes, order):
        g = side_df[(side_df["generator_id"] == gen) & (side_df["trajectory_length"] > 0)]
        xs_all = []
        ys_all = []
        for traj in g["trajectory"].tolist():
            xs_all.extend([p.get("x", np.nan) for p in traj])
            ys_all.extend([p.get("y", np.nan) for p in traj])
        ax.hist2d(xs_all, ys_all, bins=[64, 32], range=[[0, 3300], [0, 260]], cmap="magma", cmin=1)
        ax.set_title(f"{gen}\nscore={rate.get(gen, np.nan):.2f}")
        ax.set_ylim(260, 0)
        ax.grid(False)
    for ax in axes[len(order) :]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("x position (px)")
    for ax in axes[::ncols]:
        ax.set_ylabel("y position (px)")
    fig.suptitle("Trajectory occupancy heatmaps by generator", y=1.005, fontsize=13)
    savefig(path)
    plot_paths.append(path)


def plot_trajectory_metrics(tm: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_trajectory_metrics_vs_rating.png"
    metrics = [
        ("occupancy_entropy", "Occupancy entropy"),
        ("median_max_x", "Median max x reached"),
        ("mean_verticality", "Mean verticality (y std.)"),
        ("max_death_bin_share", "Death concentration"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    axes = axes.reshape(-1)
    for ax, (col, title) in zip(axes, metrics):
        data = tm.dropna(subset=[col, "score_rate"])
        sns.regplot(data=data, x=col, y="score_rate", ax=ax, scatter=False, color="#666", ci=None, line_kws={"ls": "--", "lw": 1})
        ax.scatter(data[col], data["score_rate"], s=70, color="#4477AA", edgecolor="#222")
        for _, row in data.iterrows():
            ax.text(row[col], row["score_rate"] + 0.012, row["generator_id"], fontsize=7, ha="center")
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Preference score")
        ax.set_ylim(0.15, 0.92)
    savefig(path)
    plot_paths.append(path)


def player_preference_matrix(side_df: pd.DataFrame, min_votes: int = 5) -> pd.DataFrame:
    valid = side_df[~side_df["score_for_side"].isna()].copy()
    player_vote_counts = valid.drop_duplicates("vote_id").groupby("player_id").size()
    keep_players = player_vote_counts[player_vote_counts >= min_votes].index
    valid = valid[valid["player_id"].isin(keep_players)]
    mat = valid.pivot_table(index="player_id", columns="generator_id", values="score_for_side", aggfunc="mean")
    # Sort columns by global ranking elsewhere later; rows by clustering.
    return mat


def plot_user_heatmap(side_df: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> pd.DataFrame:
    path = IMG_DIR / "eda_user_generator_heatmap.png"
    mat = player_preference_matrix(side_df, min_votes=5)
    order = [g for g in ranking["generator_id"].tolist() if g in mat.columns]
    mat = mat[order]
    filled = mat.fillna(0.5)
    if len(filled) >= 2:
        row_linkage = linkage(filled.values, method="ward")
        row_order = leaves_list(row_linkage)
        filled = filled.iloc[row_order]
    fig, ax = plt.subplots(figsize=(11.0, max(5.0, 0.22 * len(filled) + 2)))
    sns.heatmap(filled, cmap="vlag", vmin=0, vmax=1, center=0.5, ax=ax, cbar_kws={"label": "Mean preference score"})
    ax.set_title("Anonymous player ID × generator preference heatmap")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Anonymous player IDs (clustered)")
    ax.set_yticks([])
    ax.tick_params(axis="x", rotation=45)
    savefig(path)
    plot_paths.append(path)
    return mat


def plot_user_embedding(side_df: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_user_embedding.png"
    mat = player_preference_matrix(side_df, min_votes=5)
    order = [g for g in ranking["generator_id"].tolist() if g in mat.columns]
    mat = mat[order]
    if len(mat) < 3:
        # Fallback simple text figure.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Too few users with ≥5 votes for embedding", ha="center")
        savefig(path)
        plot_paths.append(path)
        return
    X = mat.fillna(0.5).values
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    vote_counts = side_df.drop_duplicates("vote_id").groupby("player_id").size().reindex(mat.index).fillna(0)
    play = side_df.groupby("player_id").agg(completion=("completed", "mean"), death=("died", "mean")).reindex(mat.index)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=30 + vote_counts.values * 1.2, c=play["completion"], cmap="viridis", edgecolor="#222", alpha=0.85)
    for i, pid in enumerate(mat.index):
        label = str(pid).replace("anon_", "")[:5]
        ax.text(coords[i, 0], coords[i, 1], label, fontsize=6, ha="center", va="center", color="white")
    ax.set_title("Exploratory user preference embedding (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Completion rate")
    savefig(path)
    plot_paths.append(path)


def plot_tag_semantics(side_df: pd.DataFrame, ranking: pd.DataFrame, plot_paths: list[Path]) -> None:
    path = IMG_DIR / "eda_tag_semantics.png"
    tag_cols = [f"tag_{t}" for t in TAG_NAMES]
    tag_counts = side_df[tag_cols].sum().sort_values(ascending=False)
    tag_labels = [c.replace("tag_", "") for c in tag_counts.index]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = axes.reshape(-1)
    axes[0].barh(tag_labels[::-1], tag_counts.values[::-1], color="#4477AA")
    axes[0].set_title("Tag frequency")
    axes[0].set_xlabel("Assignments")

    tag_by_gen = side_df.groupby("generator_id")[tag_cols].mean().reindex(ranking["generator_id"])
    sns.heatmap(tag_by_gen.rename(columns=lambda c: c.replace("tag_", "")), ax=axes[1], cmap="YlOrRd", vmin=0, annot=False, cbar_kws={"label": "Tag rate"})
    axes[1].set_title("Tag rate by generator")
    axes[1].set_xlabel("Tag")
    axes[1].set_ylabel("Generator")

    tagged = side_df[side_df["tag_count"] > 0]
    co = pd.DataFrame(0, index=TAG_NAMES, columns=TAG_NAMES, dtype=float)
    for tags in tagged["tags"]:
        unique_tags = [t for t in set(tags) if t in TAG_NAMES]
        for a, b in itertools.combinations(unique_tags, 2):
            co.loc[a, b] += 1
            co.loc[b, a] += 1
    sns.heatmap(co, ax=axes[2], cmap="Blues", cbar_kws={"label": "Co-occurrence count"})
    axes[2].set_title("Tag co-occurrence")
    axes[2].set_xlabel("Tag")
    axes[2].set_ylabel("Tag")

    outcome_rates = []
    for tag in TAG_NAMES:
        col = f"tag_{tag}"
        outcome_rates.append(
            {
                "tag": tag,
                "winner_rate": side_df.loc[side_df[col] == 1, "result_for_side"].eq("win").mean() if side_df[col].sum() else np.nan,
                "loser_rate": side_df.loc[side_df[col] == 1, "result_for_side"].eq("loss").mean() if side_df[col].sum() else np.nan,
                "tie_rate": side_df.loc[side_df[col] == 1, "result_for_side"].eq("tie").mean() if side_df[col].sum() else np.nan,
            }
        )
    out_df = pd.DataFrame(outcome_rates).set_index("tag")
    out_df = out_df.loc[tag_counts.index.str.replace("tag_", "")]
    out_df.plot(kind="bar", stacked=True, ax=axes[3], color=["#66AA55", "#CC6677", "#DDCC77"])
    axes[3].set_title("Outcome distribution when tag is present")
    axes[3].set_ylabel("Fraction")
    axes[3].set_xlabel("Tag")
    axes[3].tick_params(axis="x", rotation=45)
    axes[3].legend(["winner", "loser", "tie"], loc="upper right")
    savefig(path)
    plot_paths.append(path)


def write_outputs(
    data: LoadedData,
    side_df: pd.DataFrame,
    vote_df: pd.DataFrame,
    ranking: pd.DataFrame,
    level_metrics: pd.DataFrame,
    static_gen: pd.DataFrame,
    traj_metrics: pd.DataFrame,
    plot_paths: list[Path],
) -> dict[str, Any]:
    duration = side_df["duration_seconds"].dropna()
    tags_total = int(side_df[[f"tag_{t}" for t in TAG_NAMES]].sum().sum())
    audit = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "vote_total_reported": data.votes_envelope.get("total"),
        "vote_rows": len(data.votes),
        "player_profiles": len(data.players),
        "unique_players_in_votes": int(vote_df["player_id"].nunique()),
        "unique_sessions_in_votes": int(vote_df["session_id"].nunique()),
        "level_stats_total_reported": data.levels_envelope.get("total"),
        "level_stats_rows": len(data.levels),
        "trajectory_total_reported": data.trajectories_envelope.get("total"),
        "standalone_trajectory_rows": len(data.trajectories),
        "side_rows": len(side_df),
        "played_side_rows": int(side_df["played"].sum()),
        "embedded_trajectories": int((side_df["trajectory_length"] > 0).sum()),
        "tag_assignments": tags_total,
        "vote_result_distribution": vote_df["result"].value_counts().to_dict(),
        "death_count_distribution": side_df["deaths"].value_counts().sort_index().to_dict(),
        "duration_seconds_median": float(duration.median()) if len(duration) else None,
        "duration_seconds_p95": float(duration.quantile(0.95)) if len(duration) else None,
        "duration_seconds_p99": float(duration.quantile(0.99)) if len(duration) else None,
        "duration_seconds_max": float(duration.max()) if len(duration) else None,
        "generated_plots": [str(p) for p in plot_paths],
    }
    (OUT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    side_out = side_df.drop(columns=["trajectory", "events", "death_locations", "tags"])
    side_out.to_csv(OUT_DIR / "side_level_table.csv", index=False)
    vote_df.to_csv(OUT_DIR / "vote_table.csv", index=False)
    ranking.to_csv(OUT_DIR / "generator_ranking.csv", index=False)
    level_metrics.drop(columns=[c for c in ["path"] if c in level_metrics.columns]).to_csv(OUT_DIR / "level_static_metrics.csv", index=False)
    static_gen.to_csv(OUT_DIR / "generator_static_metrics.csv", index=False)
    traj_metrics.to_csv(OUT_DIR / "generator_trajectory_metrics.csv", index=False)
    return audit


def write_verifier_paths(plot_paths: list[Path]) -> None:
    lines = [str(p) for p in plot_paths]
    VERIFIER_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = load_data()
    side_df = normalize_side_rows(data.votes)
    vote_df = build_vote_level_rows(data.votes)

    ci = bootstrap_score_ci(vote_df, n_boot=800)
    bt = bradley_terry(vote_df)
    ranking = generator_records(side_df).merge(ci, on="generator_id", how="left").merge(bt, on="generator_id", how="left")
    ranking = ranking.sort_values("score_rate", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    order = ranking["generator_id"].tolist()
    matrix, counts = pairwise_matrix(vote_df, order)

    level_metrics, level_texts = compute_level_metrics()
    static_gen = summarize_static_metrics(level_metrics)
    traj_metrics = trajectory_metrics(side_df, ranking)

    plot_paths: list[Path] = []
    preliminary_audit = {
        "player_profiles": len(data.players),
        "vote_rows": len(data.votes),
        "side_rows": len(side_df),
        "level_stats_rows": len(data.levels),
        "embedded_trajectories": int((side_df["trajectory_length"] > 0).sum()),
        "tag_assignments": int(side_df[[f"tag_{t}" for t in TAG_NAMES]].sum().sum()),
    }

    plot_study_overview(preliminary_audit, plot_paths)
    plot_engagement(vote_df, side_df, plot_paths)
    plot_generator_ranking(ranking, plot_paths)
    plot_pairwise_confusion(matrix, counts, plot_paths)
    plot_static_metrics_table(static_gen, ranking, plot_paths)
    plot_expressive_range(level_metrics, order, ranking, plot_paths)
    plot_static_metrics_vs_rating(static_gen, ranking, plot_paths)
    plot_trajectory_stacks(side_df, ranking, plot_paths)
    plot_trajectory_occupancy(side_df, ranking, plot_paths)
    plot_trajectory_metrics(traj_metrics, plot_paths)
    plot_user_heatmap(side_df, ranking, plot_paths)
    plot_user_embedding(side_df, ranking, plot_paths)
    plot_tag_semantics(side_df, ranking, plot_paths)

    audit = write_outputs(data, side_df, vote_df, ranking, level_metrics, static_gen, traj_metrics, plot_paths)
    write_verifier_paths(plot_paths)
    sync_legacy_images(plot_paths)
    print(json.dumps({"plots": [str(p) for p in plot_paths], "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
