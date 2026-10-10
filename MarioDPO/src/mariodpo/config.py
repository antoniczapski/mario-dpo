"""Tiny YAML config loader with ``--set key=value`` overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    """Best-effort string -> python scalar."""
    low = value.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in value or "e" in low:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply ``key=value`` overrides."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"bad --set override (need key=value): {item!r}")
        key, _, raw = item.partition("=")
        cfg[key.strip()] = _coerce(raw.strip())
    return cfg
