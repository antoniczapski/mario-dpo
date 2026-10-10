"""Shared constants: tile alphabet, level geometry, and repo paths.

The single source of truth for the arena tile vocabulary is the backend seed
importer (``backend/src/db/seed.py``). We replicate its 37-character whitelist
here so the package has no backend dependency, and assert compatibility in the
tests. Keep this in sync if the backend spec changes.
"""

from __future__ import annotations

from pathlib import Path

# --- Repo layout -----------------------------------------------------------
# This file lives at: <repo>/MarioDPO/src/mariodpo/constants.py
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]          # <repo>/MarioDPO
REPO_DIR = PACKAGE_DIR.parents[2]             # <repo>

# Seed level corpus on disk (13 generator sub-directories).
SEED_LEVELS_DIR = REPO_DIR / "db" / "seed" / "levels"
# MarioDPO levels referenced by the May votes but absent from db/seed/levels.
# Used only so those votes resolve to an ASCII file.
MARIODPO_LEVELS_DIR = REPO_DIR / "MarioDPO" / "generated_levels"

# Project-local data tree.
DATA_DIR = PROJECT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SYNTHETIC_DIR = DATA_DIR / "synthetic"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"

# Default data export used throughout the pipeline (May 2026 snapshot).
VOTES_FILENAME = "pcg-arena-votes-2026-05-10.json"

# --- Level geometry --------------------------------------------------------
ARENA_HEIGHT = 16                 # arena levels are 16 rows tall
MARIOGPT_NATIVE_HEIGHT = 14       # the pretrained MarioGPT checkpoint is 14 rows
HEIGHT_PAD_TOP = ARENA_HEIGHT - MARIOGPT_NATIVE_HEIGHT  # 2 air rows added on top

# Backend validation bounds (mirror backend/src/db/seed.py).
MIN_LEVEL_WIDTH = 1
MAX_LEVEL_WIDTH = 250
MIN_LEVEL_HEIGHT = 10
MAX_LEVEL_HEIGHT = 20
# Practical export bound used by the deployment exporter (matches thesis text).
EXPORT_MIN_WIDTH = 150
EXPORT_MAX_WIDTH = 250

AIR = "-"

# --- Arena tile alphabet (37 chars, mirrors backend seed.py) ---------------
ALLOWED_TILES = set(
    "-"   # Air
    "M"   # Mario start
    "F"   # Level exit / flag
    "y"   # Spiky
    "Y"   # Winged Spiky
    "E"   # Goomba
    "g"   # Goomba (alt)
    "G"   # Winged Goomba
    "k"   # Green Koopa
    "K"   # Winged Green Koopa
    "r"   # Red Koopa
    "R"   # Winged Red Koopa
    "X"   # Solid floor block
    "#"   # Pyramid block
    "S"   # Normal solid block
    "D"   # Used block
    "%"   # Jump-through platform
    "|"   # Background for platform
    "?"   # Question block (mushroom)
    "@"   # Question block (mushroom alt)
    "Q"   # Question block (coin)
    "!"   # Question block (coin alt)
    "C"   # Coin block
    "U"   # Mushroom block
    "L"   # 1-Up block
    "1"   # Invisible 1-Up block
    "2"   # Invisible coin block
    "o"   # Free-standing coin
    "t"   # Empty pipe
    "T"   # Flower pipe
    "<"   # Pipe top left
    ">"   # Pipe top right
    "["   # Pipe body left
    "]"   # Pipe body right
    "*"   # Bullet Bill launcher body
    "B"   # Bullet Bill head
    "b"   # Bullet Bill neck/body
)

# --- MarioGPT (native) -> arena alphabet map -------------------------------
# Mirrors generators/MarioGPT/postprocessing.ipynb. The crucial entry is the
# A* trace overlay 'x' -> air. Characters already in ALLOWED_TILES pass through.
MARIOGPT_TO_ARENA = {
    "-": "-",
    "X": "X",
    "S": "S",
    "?": "?",
    "Q": "Q",
    "o": "o",
    "E": "E",
    "<": "<",
    ">": ">",
    "[": "[",
    "]": "]",
    "T": "T",
    "B": "B",
    "b": "b",
    "M": "M",
    "F": "F",
    "x": "-",   # MarioGPT trace overlay -> air
}

# --- Semantic tile groups (for feature extraction) -------------------------
SOLID_TILES = set("X#SD%")          # tiles that obstruct / can be stood on
BREAKABLE_TILES = set("S")
GROUND_LIKE_TILES = set("X#S%[]<>")  # used to detect the surface profile
QUESTION_TILES = set("?@Q!C")        # reward-bearing blocks
POWERUP_TILES = set("U?@L")          # mushrooms / fire / 1-up sources
COIN_TILES = set("oC2Q!")            # coins (free + in blocks)
REWARD_TILES = QUESTION_TILES | set("oUL12C")
PIPE_TILES = set("t<>[]T")
PLATFORM_TILES = set("%|")
ENEMY_TILES = set("yYEgGkKrRBb*")    # all hazardous entities
ENEMY_GROUPS = {
    "goomba": set("EgG"),
    "koopa": set("kKrR"),
    "spiky": set("yY"),
    "bullet": set("Bb*"),
}
PASSABLE_TILES = set("-oM12C|")      # tiles Mario can occupy / pass through
