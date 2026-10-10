"""Level generation from a (SFT or DPO) model, with judge-based rejection sampling.

Levels are far wider than any single context window, so generation proceeds by
**sliding-window continuation** in the arena column-major space: we repeatedly
feed the last few columns as context and decode the newly produced columns until
the target width is reached (the same idea MarioGPT uses for long levels). Each
finished level is decoded, normalised, repaired, validated, and judge-scored;
rejection sampling keeps the best of N.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import ARENA_HEIGHT
from .features import extract_features
from .level_io import level_to_text, normalise_arena_level
from .postprocess import postprocess
from .tokenizer_io import COLUMN_SEP, columns_to_level, style_prompt
from .utils import normalised_compression_distance
from .validate import validate_level


@dataclass
class Candidate:
    rows: list[str]
    judge_score: float
    valid: bool
    errors: list[str]


def _model_ctx(model, default: int = 1024) -> int:
    cfg = getattr(model, "config", None)
    return int(getattr(cfg, "n_positions", default) or default)


def _clean_columns(text: str, height: int) -> list[str]:
    """Parse decoded text into fixed-height columns (drop ragged tails)."""
    cols = []
    for line in text.split(COLUMN_SEP):
        if not line:
            continue
        if len(line) >= height:
            cols.append(line[:height])
    return cols


def generate_level_columns(
    model,
    tokenizer,
    target_cols: int = 180,
    context_cols: int = 24,
    chunk_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.95,
    style: str | None = "NINTENDO",
    height: int = ARENA_HEIGHT,
    max_iters: int = 64,
    seed: int | None = None,
) -> list[str]:
    """Autoregressively grow a level to ``target_cols`` via sliding windows."""
    import torch

    device = next(model.parameters()).device
    ctx_cap = max(16, _model_ctx(model) - chunk_tokens - 8)
    if seed is not None:
        torch.manual_seed(seed)

    columns: list[str] = []
    prefix = style_prompt(style)
    iters = 0
    while len(columns) < target_cols and iters < max_iters:
        iters += 1
        ctx_cols = columns[-context_cols:]
        prompt = prefix + (COLUMN_SEP.join(ctx_cols) + COLUMN_SEP if ctx_cols else "")
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=ctx_cap).to(device)
        out = model.generate(
            **enc, do_sample=True, max_new_tokens=chunk_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_ids = out[0, enc["input_ids"].shape[1]:]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        new_cols = _clean_columns(new_text, height)
        if not new_cols:
            continue  # try again; guarded by max_iters
        columns.extend(new_cols)
    if not columns:
        columns = ["-" * height]
    return columns_to_level(columns[:target_cols], height)


def _ncd_to_original(rows: list[str], ncd_refs: list[str]) -> float:
    if not ncd_refs:
        return 0.0
    text = level_to_text(rows)
    return sum(normalised_compression_distance(text, r) for r in ncd_refs) / len(ncd_refs)


def score_candidate(judge, rows: list[str], ncd_refs: list[str]) -> Candidate:
    feats = extract_features(rows)
    feats["ncd_to_original"] = _ncd_to_original(rows, ncd_refs)
    feats["generator"] = "mariodpo"
    score = judge.score(feats)
    errors = validate_level(rows)
    return Candidate(rows=rows, judge_score=score, valid=not errors, errors=errors)


def generate_best(
    model,
    tokenizer,
    judge,
    ncd_refs: list[str],
    n_candidates: int = 10,
    target_cols: int = 180,
    min_width: int = 150,
    require_valid: bool = True,
    seed: int | None = None,
    **gen_kw,
) -> Candidate:
    """Rejection sampling: best valid candidate by judge score over N samples."""
    cands = []
    for j in range(n_candidates):
        rows = generate_level_columns(
            model, tokenizer, target_cols=target_cols,
            seed=(seed + j) if seed is not None else None, **gen_kw,
        )
        rows = normalise_arena_level(rows)
        rows, _ = postprocess(rows)
        cands.append(score_candidate(judge, rows, ncd_refs))
    valid = [c for c in cands if c.valid and c.rows and len(c.rows[0]) >= min_width]
    pool = valid if (valid and require_valid) else cands
    return max(pool, key=lambda c: c.judge_score)
