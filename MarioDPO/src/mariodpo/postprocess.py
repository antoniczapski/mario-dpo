"""Deterministic playability repair for generated levels.

A port of ``MarioDPO/postprocess_levels.py`` operating on ``list[str]`` rows.
This is the "safety net" applied after sampling: it guarantees a Mario spawn
near the start, a flag near the end, and solid ground at both boundaries, so a
generated level is never an automatic loss for trivial reasons. It is kept
deliberately separate from the learned model (the model learns *preference*;
this enforces hard *validity*).
"""

from __future__ import annotations

from .constants import AIR, SOLID_TILES

_STANDABLE = SOLID_TILES | set("[]")


def _grid(rows: list[str]) -> list[list[str]]:
    width = max((len(r) for r in rows), default=0)
    return [list(r.ljust(width, AIR)) for r in rows]


def _rows(grid: list[list[str]]) -> list[str]:
    return ["".join(r) for r in grid]


def _remove(grid: list[list[str]], tile: str) -> int:
    n = 0
    for row in grid:
        for c in range(len(row)):
            if row[c] == tile:
                row[c] = AIR
                n += 1
    return n


def _ground_row(grid: list[list[str]], col: int) -> int:
    for r in range(len(grid) - 1, -1, -1):
        if col < len(grid[r]) and grid[r][col] in _STANDABLE:
            return r
    return len(grid) - 2 if len(grid) >= 2 else len(grid) - 1


def ensure_mario_spawn(grid: list[list[str]]) -> bool:
    positions = [(r, c) for r, row in enumerate(grid)
                 for c, t in enumerate(row) if t == "M"]
    if any(c <= 5 for _, c in positions):
        for r, c in positions:
            if c > 5:
                grid[r][c] = AIR
        return False
    _remove(grid, "M")
    for col in range(2, 5):
        gr = _ground_row(grid, col)
        sr = gr - 1
        if 0 <= sr < len(grid) and col < len(grid[sr]) and grid[sr][col] in {AIR, "o"}:
            grid[sr][col] = "M"
            return True
    if len(grid) > 13 and len(grid[13]) > 2:
        grid[13][2] = "M"
        return True
    return False


def ensure_flag(grid: list[list[str]]) -> bool:
    width = len(grid[0]) if grid else 0
    positions = [(r, c) for r, row in enumerate(grid)
                 for c, t in enumerate(row) if t == "F"]
    if any(c >= width - 5 for _, c in positions):
        for r, c in positions:
            if c < width - 5:
                grid[r][c] = AIR
        return False
    _remove(grid, "F")
    for col in range(width - 3, max(-1, width - 6), -1):
        if col < 0:
            continue
        gr = _ground_row(grid, col)
        fr = gr - 1
        if fr >= 0 and col < len(grid[fr]):
            grid[fr][col] = "F"
            if gr < len(grid) and grid[gr][col] == AIR:
                grid[gr][col] = "X"
            return True
    return False


def ensure_ground_at_ends(grid: list[list[str]]) -> bool:
    modified = False
    width = len(grid[0]) if grid else 0
    gr = len(grid) - 2 if len(grid) >= 2 else len(grid) - 1
    for col in list(range(min(5, width))) + list(range(max(0, width - 5), width)):
        if 0 <= gr < len(grid) and col < len(grid[gr]) and grid[gr][col] == AIR:
            grid[gr][col] = "X"
            modified = True
    return modified


def postprocess(rows: list[str]) -> tuple[list[str], dict]:
    """Repair a level; return ``(rows, modifications)``."""
    grid = _grid(rows)
    mods = {
        "mario_moved": ensure_mario_spawn(grid),
        "flag_moved": ensure_flag(grid),
        "ground_fixed": ensure_ground_at_ends(grid),
    }
    return _rows(grid), mods
