"""Level I/O: parsing, normalisation, vote-ID resolution, and representation
conversion between the native MarioGPT format and the arena format.

A "level" is represented in memory as a ``list[str]`` of equal-length rows
(top row first). Helpers here keep that invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .constants import (
    AIR,
    ALLOWED_TILES,
    ARENA_HEIGHT,
    HEIGHT_PAD_TOP,
    MARIODPO_LEVELS_DIR,
    MARIOGPT_TO_ARENA,
    SEED_LEVELS_DIR,
)


# --- Basic parsing ---------------------------------------------------------
def parse_level(text: str) -> list[str]:
    """Split raw text into rows, dropping trailing blank lines."""
    rows = text.replace("\r\n", "\n").split("\n")
    while rows and rows[-1] == "":
        rows.pop()
    return rows


def level_to_text(rows: list[str]) -> str:
    """Serialise rows back to text with a trailing newline."""
    return "\n".join(rows) + "\n"


def pad_to_rectangle(rows: list[str], fill: str = AIR) -> list[str]:
    """Right-pad every row to the maximum row width so the grid is rectangular."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    return [r.ljust(width, fill) for r in rows]


def load_level(path: str | Path) -> list[str]:
    """Read a level file from disk into a rectangular list of rows."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return pad_to_rectangle(parse_level(text))


# --- Representation conversion ---------------------------------------------
def map_alphabet(rows: list[str]) -> list[str]:
    """Map characters into the arena alphabet.

    Applies the MarioGPT->arena table (notably the ``x`` trace overlay -> air);
    characters already in the arena whitelist pass through; anything else falls
    back to air. This is the same policy as the MarioGPT postprocessing notebook.
    """
    out = []
    for row in rows:
        chars = []
        for ch in row:
            if ch in MARIOGPT_TO_ARENA:
                chars.append(MARIOGPT_TO_ARENA[ch])
            elif ch in ALLOWED_TILES:
                chars.append(ch)
            else:
                chars.append(AIR)
        out.append("".join(chars))
    return out


def to_arena_height(rows: list[str], target_height: int = ARENA_HEIGHT) -> list[str]:
    """Pad/crop a level to ``target_height`` rows.

    The native MarioGPT checkpoint produces 14 rows; arena levels are 16. We add
    air rows on *top* (the sky) to reach 16, matching how the seeded MarioGPT
    levels in db/seed/levels were produced. If a level is already tall enough we
    crop excess sky rows from the top; ground rows at the bottom are preserved.
    """
    rows = pad_to_rectangle(rows)
    h = len(rows)
    if h == target_height:
        return rows
    width = len(rows[0]) if rows else 0
    if h < target_height:
        pad = [AIR * width for _ in range(target_height - h)]
        return pad + rows
    # h > target_height: drop the topmost (sky) rows.
    return rows[h - target_height:]


def mariogpt_to_arena(rows: list[str]) -> list[str]:
    """Full native-MarioGPT -> arena conversion: alphabet map then height pad."""
    return to_arena_height(map_alphabet(rows), ARENA_HEIGHT)


def normalise_arena_level(rows: list[str]) -> list[str]:
    """Coerce an arbitrary level into a clean arena level (alphabet + height)."""
    return to_arena_height(map_alphabet(rows), ARENA_HEIGHT)


# --- Vote level-ID resolution ----------------------------------------------
def split_level_id(level_id: str) -> tuple[str, str]:
    """Split a ``"generator::filename"`` id into ``(generator, filename)``."""
    if "::" in level_id:
        gen, _, fname = level_id.partition("::")
        return gen, fname
    return "", level_id


def resolve_level_path(level_id: str) -> Path | None:
    """Resolve a vote ``level_id`` to an ASCII file on disk, or ``None``.

    Handles the special case where ``mariodpo`` levels are not in
    ``db/seed/levels`` but live under ``MarioDPO/generated_levels``.
    ``test-gen`` (a single throwaway level) intentionally resolves to ``None``.
    """
    generator, fname = split_level_id(level_id)
    if not generator or not fname:
        return None
    if generator == "mariodpo":
        candidate = MARIODPO_LEVELS_DIR / fname
        return candidate if candidate.exists() else None
    if generator == "test-gen":
        return None
    candidate = SEED_LEVELS_DIR / generator / fname
    return candidate if candidate.exists() else None


def iter_seed_levels(
    generators: Iterable[str] | None = None,
) -> Iterable[tuple[str, str, Path]]:
    """Yield ``(generator, filename, path)`` for every seed level on disk.

    If ``generators`` is given, only those sub-directories are scanned.
    """
    if not SEED_LEVELS_DIR.exists():
        return
    wanted = set(generators) if generators is not None else None
    for gen_dir in sorted(SEED_LEVELS_DIR.iterdir()):
        if not gen_dir.is_dir():
            continue
        if wanted is not None and gen_dir.name not in wanted:
            continue
        for path in sorted(gen_dir.glob("*.txt")):
            yield gen_dir.name, path.name, path
