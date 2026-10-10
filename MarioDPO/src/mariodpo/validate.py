"""Arena format validation — mirrors ``backend/src/db/seed.py`` checks.

Used by the deployment exporter so generated levels are guaranteed to pass the
backend importer (height/width bounds, rectangular shape, tile whitelist).
"""

from __future__ import annotations

from .constants import (
    ALLOWED_TILES,
    EXPORT_MAX_WIDTH,
    EXPORT_MIN_WIDTH,
    MAX_LEVEL_HEIGHT,
    MIN_LEVEL_HEIGHT,
)


def validate_level(
    rows: list[str],
    *,
    min_width: int = EXPORT_MIN_WIDTH,
    max_width: int = EXPORT_MAX_WIDTH,
) -> list[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: list[str] = []
    if not rows:
        return ["level is empty"]
    h = len(rows)
    if not (MIN_LEVEL_HEIGHT <= h <= MAX_LEVEL_HEIGHT):
        errors.append(f"height {h} outside [{MIN_LEVEL_HEIGHT}, {MAX_LEVEL_HEIGHT}]")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        errors.append(f"non-rectangular: row widths {sorted(widths)}")
    w = next(iter(widths))
    if not (min_width <= w <= max_width):
        errors.append(f"width {w} outside [{min_width}, {max_width}]")
    bad = sorted({ch for r in rows for ch in r if ch not in ALLOWED_TILES})
    if bad:
        errors.append(f"illegal tiles: {''.join(bad)}")
    return errors


def is_valid(rows: list[str], **kw) -> bool:
    return not validate_level(rows, **kw)
