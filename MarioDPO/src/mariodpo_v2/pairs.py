"""Build the DPO preference dataset: human pairs + judge-labelled synthetic pairs.

Each record is ``{prompt, chosen, rejected, source, weight}`` where ``chosen``
and ``rejected`` are column-major serialisations of a fixed-width opening window
of the two levels, and ``prompt`` is a constant style prefix (DPO requires the
*same* prompt for both completions). Human votes are physically oversampled.

The judge here is the static-feature scorer from :mod:`mariodpo.judge`; it
labels synthetic pairs so a clear preference margin defines winner/loser.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .features import extract_features
from .level_io import level_to_text, normalise_arena_level
from .tokenizer_io import level_to_columns, style_prompt
from .utils import normalised_compression_distance

COLUMN_SEP = "\n"
DEFAULT_STYLE = "NINTENDO"


@dataclass
class PrefRecord:
    prompt: str
    chosen: str
    rejected: str
    source: str
    weight: float


def _window_text(rows: list[str], win_cols: int) -> str:
    cols = level_to_columns(rows)
    return COLUMN_SEP.join(cols[:win_cols])


def make_human_pairs(
    resolved_pairs: list[tuple[list[str], list[str], str]],
    win_cols: int = 48,
    style: str = DEFAULT_STYLE,
    weight: float = 10.0,
) -> list[PrefRecord]:
    """``resolved_pairs`` = ``(winner_rows, loser_rows, vote_id)``."""
    prompt = style_prompt(style)
    out = []
    for w_rows, l_rows, _vid in resolved_pairs:
        out.append(PrefRecord(
            prompt=prompt,
            chosen=_window_text(w_rows, win_cols),
            rejected=_window_text(l_rows, win_cols),
            source="human",
            weight=weight,
        ))
    return out


def _ncd(rows: list[str], refs: list[str]) -> float:
    if not refs:
        return 0.0
    t = level_to_text(rows)
    return sum(normalised_compression_distance(t, r) for r in refs) / len(refs)


def score_levels(
    judge, levels: list[list[str]], ncd_refs: list[str]
) -> list[tuple[list[str], float]]:
    """Score each level with the static judge; return ``(rows, score)``."""
    scored = []
    for rows in levels:
        feats = extract_features(rows)
        feats["ncd_to_original"] = _ncd(rows, ncd_refs)
        feats["generator"] = "synthetic"
        scored.append((rows, judge.score(feats)))
    return scored


def make_synthetic_pairs(
    scored_levels: list[tuple[list[str], float]],
    n_pairs: int,
    margin: float = 0.5,
    win_cols: int = 48,
    style: str = DEFAULT_STYLE,
    weight: float = 1.0,
    seed: int = 42,
) -> list[PrefRecord]:
    """Sample level pairs and keep those whose judge-score gap exceeds ``margin``."""
    rng = random.Random(seed)
    prompt = style_prompt(style)
    out: list[PrefRecord] = []
    if len(scored_levels) < 2:
        return out
    attempts = 0
    max_attempts = n_pairs * 50
    while len(out) < n_pairs and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(scored_levels, 2)
        if abs(a[1] - b[1]) < margin:
            continue
        win, lose = (a, b) if a[1] > b[1] else (b, a)
        out.append(PrefRecord(
            prompt=prompt,
            chosen=_window_text(win[0], win_cols),
            rejected=_window_text(lose[0], win_cols),
            source="synthetic",
            weight=weight,
        ))
    return out


def expand_by_weight(records: list[PrefRecord]) -> list[dict]:
    """Physically repeat each record ``round(weight)`` times (oversampling).

    trl's DPOTrainer has no per-example weighting, so we materialise weights by
    duplication. The ``weight``/``source`` fields are retained for analysis.
    """
    rows: list[dict] = []
    for rec in records:
        reps = max(1, round(rec.weight))
        for _ in range(reps):
            rows.append(asdict(rec))
    return rows
