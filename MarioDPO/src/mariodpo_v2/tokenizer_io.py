"""Column-major (de)serialisation between arena levels and model token strings.

We continue-pretrain MarioGPT in the **arena** representation (height 16, the
full 37-tile alphabet). Levels are serialised **column-major** so that vertically
adjacent tiles stay adjacent in the sequence — the property that makes column
encodings work for Mario (enemies stand on platforms, pipes span rows, etc.).

Serialisation format: one *column* per line, top-to-bottom, columns left-to-right::

    col0_row0 col0_row1 ... col0_row15   <- first text line (16 chars)
    col1_row0 col1_row1 ... col1_row15   <- second text line
    ...

This is fully reversible and gives the BPE tokenizer clear column boundaries.
MarioGPT-generated candidates (native height 14, containing the ``x`` trace
token) are first passed through :func:`mariodpo.level_io.mariogpt_to_arena`
so *every* level flows through this single token space.
"""

from __future__ import annotations

from .constants import AIR, ALLOWED_TILES, ARENA_HEIGHT
from .level_io import pad_to_rectangle

COLUMN_SEP = "\n"


def level_to_columns(rows: list[str], height: int = ARENA_HEIGHT) -> list[str]:
    """Return the level as a list of column strings, each ``height`` chars."""
    rows = pad_to_rectangle(rows)
    # Force exactly ``height`` rows (pad air on top / crop sky) without importing
    # the heavier converter, to keep this module self-contained for round-trips.
    if len(rows) < height:
        width = len(rows[0]) if rows else 0
        rows = [AIR * width] * (height - len(rows)) + rows
    elif len(rows) > height:
        rows = rows[len(rows) - height:]
    width = len(rows[0]) if rows else 0
    return ["".join(rows[r][c] for r in range(height)) for c in range(width)]


def columns_to_level(columns: list[str], height: int = ARENA_HEIGHT) -> list[str]:
    """Inverse of :func:`level_to_columns`: rebuild rows from column strings."""
    clean = []
    for col in columns:
        col = "".join(ch if ch in ALLOWED_TILES else AIR for ch in col)
        if len(col) < height:
            col = col.ljust(height, AIR)
        elif len(col) > height:
            col = col[:height]
        clean.append(col)
    if not clean:
        return [AIR for _ in range(height)]
    return ["".join(col[r] for col in clean) for r in range(height)]


def serialize(rows: list[str], height: int = ARENA_HEIGHT) -> str:
    """Level -> column-major text (one column per line)."""
    return COLUMN_SEP.join(level_to_columns(rows, height))


def deserialize(text: str, height: int = ARENA_HEIGHT) -> list[str]:
    """Column-major text -> rows. Robust to ragged / noisy model output."""
    lines = [ln for ln in text.split(COLUMN_SEP) if ln != ""]
    if not lines:
        return [AIR for _ in range(height)]
    return columns_to_level(lines, height)


# --- Windowing for wide levels --------------------------------------------
def iter_column_windows(
    rows: list[str],
    win_cols: int,
    stride: int | None = None,
    height: int = ARENA_HEIGHT,
):
    """Yield serialised column windows of width ``win_cols`` over a level.

    Used so wide levels (150-250 cols) fit the model context. ``stride``
    defaults to ``win_cols`` (non-overlapping). The final partial window is
    kept if it has at least ``win_cols // 2`` columns.
    """
    cols = level_to_columns(rows, height)
    n = len(cols)
    if n == 0:
        return
    stride = stride or win_cols
    if n <= win_cols:
        yield COLUMN_SEP.join(cols)
        return
    start = 0
    while start < n:
        window = cols[start:start + win_cols]
        if len(window) >= max(1, win_cols // 2):
            yield COLUMN_SEP.join(window)
        if start + win_cols >= n:
            break
        start += stride


def style_prompt(style: str | None) -> str:
    """Optional style-conditioning prefix, e.g. ``"[STYLE:NINTENDO]\\n"``."""
    if not style:
        return ""
    return f"[STYLE:{style.upper()}]{COLUMN_SEP}"
