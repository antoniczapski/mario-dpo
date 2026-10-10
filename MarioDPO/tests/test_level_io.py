"""Tests for level I/O, representation conversion, and validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mariodpo.constants import ALLOWED_TILES, ARENA_HEIGHT
from mariodpo.level_io import (
    iter_seed_levels,
    load_level,
    mariogpt_to_arena,
    resolve_level_path,
)
from mariodpo.tokenizer_io import deserialize, serialize
from mariodpo.validate import is_valid, validate_level


def test_columnmajor_roundtrip_on_seed_levels():
    """arena -> column-major text -> arena must be lossless (height 16)."""
    checked = 0
    for gen, fname, path in iter_seed_levels():
        rows = load_level(path)
        if len(rows) != ARENA_HEIGHT:
            continue
        text = serialize(rows)
        back = deserialize(text)
        assert back == rows, f"round-trip mismatch for {gen}/{fname}"
        checked += 1
        if checked >= 200:
            break
    assert checked > 0


def test_serialize_is_column_major():
    rows = ["AB", "CD"] + ["--"] * 14  # 16 rows, width 2
    # Replace with valid tiles.
    rows = ["XS", "X-"] + ["--"] * 14
    cols = serialize(rows).split("\n")
    assert cols[0] == "XX" + "-" * 14  # first column top-to-bottom
    assert cols[1] == "S-" + "-" * 14


def test_mariogpt_to_arena_height_and_alphabet():
    # 14-row native MarioGPT level with an 'x' trace overlay.
    native = ["x" * 10 for _ in range(14)]
    arena = mariogpt_to_arena(native)
    assert len(arena) == ARENA_HEIGHT          # padded 14 -> 16
    assert all(len(r) == 10 for r in arena)
    # 'x' trace must be mapped away to air; no illegal tiles remain.
    assert all(ch in ALLOWED_TILES for r in arena for ch in r)
    assert arena[0] == "-" * 10                # top padding row is air


def test_deserialize_is_robust_to_noise():
    rows = deserialize("XXz!!\n\n--QQ", height=ARENA_HEIGHT)
    assert len(rows) == ARENA_HEIGHT
    assert all(len(r) == len(rows[0]) for r in rows)
    assert all(ch in ALLOWED_TILES for r in rows for ch in r)


def test_validate_catches_bad_levels():
    assert validate_level([]) == ["level is empty"]
    short = ["X" * 160] * 5
    assert any("height" in e for e in validate_level(short))
    ragged = ["X" * 160, "X" * 159] + ["X" * 160] * 14
    assert any("rectangular" in e for e in validate_level(ragged))
    good = ["-" * 160 for _ in range(15)]
    good[-1] = "X" * 160
    assert is_valid(good)


def test_mariodpo_levels_resolve():
    assert resolve_level_path("mariodpo::mariodpo_nintendo_0000.txt") is not None
    assert resolve_level_path("test-gen::lvl-024.txt") is None
    assert resolve_level_path("original::lvl-1.txt") is not None
