"""Parse the PCG Arena votes export into preference pairs and telemetry.

The export is an envelope ``{protocol_version, total, limit, offset, data: [...]}``.
Each vote has ``result`` in {LEFT, RIGHT, TIE, SKIP}, ``left/right_level_id``
(``"generator::file"``), per-side ``telemetry`` with a sampled ``trajectory``
(list of ``{tick, x, y, state}``), ``death_locations``, ``events`` and counters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PreferencePair:
    """A single human pairwise preference (winner beat loser)."""

    winner_level_id: str
    loser_level_id: str
    winner_generator: str
    loser_generator: str
    vote_id: str
    player_id: str
    session_id: str


@dataclass
class VotesBundle:
    """Loaded votes plus convenience accessors."""

    envelope: dict
    on_disk: int
    total: int
    data: list[dict] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.on_disk < self.total


def load_votes(path: str | Path) -> VotesBundle:
    """Load the votes envelope from ``path``."""
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    data = envelope.get("data", [])
    return VotesBundle(
        envelope=envelope,
        on_disk=len(data),
        total=int(envelope.get("total", len(data))),
        data=data,
    )


def iter_preference_pairs(bundle: VotesBundle) -> list[PreferencePair]:
    """Extract non-tie, non-skip preference pairs (winner, loser)."""
    pairs: list[PreferencePair] = []
    for v in bundle.data:
        result = v.get("result")
        if result == "LEFT":
            w_lvl, l_lvl = v["left_level_id"], v["right_level_id"]
            w_gen, l_gen = v["left_generator_id"], v["right_generator_id"]
        elif result == "RIGHT":
            w_lvl, l_lvl = v["right_level_id"], v["left_level_id"]
            w_gen, l_gen = v["right_generator_id"], v["left_generator_id"]
        else:  # TIE / SKIP — no strict preference
            continue
        pairs.append(
            PreferencePair(
                winner_level_id=w_lvl,
                loser_level_id=l_lvl,
                winner_generator=w_gen,
                loser_generator=l_gen,
                vote_id=v.get("vote_id", ""),
                player_id=v.get("player_id", ""),
                session_id=v.get("session_id", ""),
            )
        )
    return pairs


# --- Per-level empirical outcomes (for judge validation) -------------------
def level_outcomes(bundle: VotesBundle) -> dict[str, dict[str, float]]:
    """Aggregate per-level wins/losses/ties from votes.

    Returns ``{level_id: {wins, losses, ties, shown, win_rate}}`` where
    ``win_rate = (wins + 0.5*ties) / shown`` over decided+tied appearances.
    """
    agg: dict[str, dict[str, float]] = {}

    def bump(level_id: str, key: str) -> None:
        rec = agg.setdefault(
            level_id, {"wins": 0.0, "losses": 0.0, "ties": 0.0, "shown": 0.0}
        )
        rec[key] += 1.0

    for v in bundle.data:
        result = v.get("result")
        left, right = v.get("left_level_id"), v.get("right_level_id")
        if result == "LEFT":
            bump(left, "wins"); bump(right, "losses")
        elif result == "RIGHT":
            bump(right, "wins"); bump(left, "losses")
        elif result == "TIE":
            bump(left, "ties"); bump(right, "ties")
        else:
            continue
    for rec in agg.values():
        decided = rec["wins"] + rec["losses"] + rec["ties"]
        rec["shown"] = decided
        rec["win_rate"] = (
            (rec["wins"] + 0.5 * rec["ties"]) / decided if decided > 0 else math.nan
        )
    return agg


def generator_outcomes(bundle: VotesBundle) -> dict[str, dict[str, float]]:
    """Aggregate per-generator win rate from votes (ties = 0.5)."""
    agg: dict[str, dict[str, float]] = {}

    def bump(gen: str, key: str) -> None:
        rec = agg.setdefault(gen, {"wins": 0.0, "losses": 0.0, "ties": 0.0})
        rec[key] += 1.0

    for v in bundle.data:
        result = v.get("result")
        lg, rg = v.get("left_generator_id"), v.get("right_generator_id")
        if result == "LEFT":
            bump(lg, "wins"); bump(rg, "losses")
        elif result == "RIGHT":
            bump(rg, "wins"); bump(lg, "losses")
        elif result == "TIE":
            bump(lg, "ties"); bump(rg, "ties")
        else:
            continue
    for rec in agg.values():
        decided = rec["wins"] + rec["losses"] + rec["ties"]
        rec["n"] = decided
        rec["win_rate"] = (
            (rec["wins"] + 0.5 * rec["ties"]) / decided if decided > 0 else math.nan
        )
    return agg


# --- Trajectory features (diagnostic; NOT used by the static judge) --------
def trajectory_features(side: dict) -> dict[str, float]:
    """Compute a few trajectory descriptors for one played side.

    These are used only for descriptive plots / RQ3-style diagnostics. The
    judge itself is static-feature-only so it can score unplayed levels.
    """
    traj = side.get("trajectory") or []
    if len(traj) < 2:
        return {
            "y_sigma": 0.0,
            "path_len": 0.0,
            "max_x": 0.0,
            "hesitation": 1.0,
            "n_samples": float(len(traj)),
        }
    xs = np.array([p["x"] for p in traj], dtype=float)
    ys = np.array([p["y"] for p in traj], dtype=float)
    ticks = np.array([p["tick"] for p in traj], dtype=float)
    dx = np.diff(xs)
    dt = np.diff(ticks)
    speed = np.divide(np.abs(dx), dt, out=np.zeros_like(dx), where=dt > 0)
    return {
        "y_sigma": float(np.std(ys)),
        "path_len": float(np.sum(np.abs(np.diff(xs)) + np.abs(np.diff(ys)))),
        "max_x": float(np.max(xs)),
        "hesitation": float(np.mean(speed < 0.5)) if speed.size else 1.0,
        "n_samples": float(len(traj)),
    }
