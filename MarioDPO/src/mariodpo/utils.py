"""Small shared utilities: ``.env`` loading, logging, seeding, gzip-NCD."""

from __future__ import annotations

import gzip
import logging
import os
import random
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Minimal ``.env`` loader (no external dependency).

    Reads ``KEY=VALUE`` lines into ``os.environ`` without overwriting existing
    variables. Returns the parsed mapping. Missing file is a no-op.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / ".env"
    path = Path(path)
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        parsed[key] = value
        os.environ.setdefault(key, value)
    return parsed


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the package logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mariodpo")


def set_seed(seed: int = 42) -> None:
    """Seed Python/NumPy/Torch RNGs (torch optional)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional in judge phase
        pass


def normalised_compression_distance(a: str, b: str) -> float:
    """gzip-based Normalised Compression Distance between two strings.

    ``NCD(a,b) = (C(ab) - min(C(a), C(b))) / max(C(a), C(b))`` where ``C`` is the
    compressed length. 0 = identical, ~1 = unrelated.
    """
    ba, bb = a.encode("utf-8"), b.encode("utf-8")

    def c(x: bytes) -> int:
        return len(gzip.compress(x, compresslevel=6))

    ca, cb = c(ba), c(bb)
    cab = c(ba + bb)
    denom = max(ca, cb)
    if denom == 0:
        return 0.0
    return (cab - min(ca, cb)) / denom
