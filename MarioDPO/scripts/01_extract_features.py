"""Phase 1a — extract static features for every level.

Builds a feature table for (a) all seed levels on disk and (b) the deployed
``mariodpo`` levels, then augments it with an ``ncd_to_original`` column (mean
gzip-NCD to a sample of original levels). Output: ``data/processed/level_features.csv``.

Usage:
    python scripts/01_extract_features.py [--ncd-sample 8]
"""

from __future__ import annotations

import argparse
import csv
import random

import _bootstrap  # noqa: F401

from mariodpo.constants import (
    DATA_PROCESSED_DIR,
    MARIODPO_LEVELS_DIR,
)
from mariodpo.features import FEATURE_NAMES, extract_features
from mariodpo.level_io import iter_seed_levels, level_to_text, load_level
from mariodpo.utils import normalised_compression_distance, setup_logging

log = setup_logging()


def _collect_levels() -> list[tuple[str, str, list[str]]]:
    """Return ``(level_id, generator, rows)`` for all seed + mariodpo levels."""
    items: list[tuple[str, str, list[str]]] = []
    for generator, fname, path in iter_seed_levels():
        items.append((f"{generator}::{fname}", generator, load_level(path)))
    if MARIODPO_LEVELS_DIR.exists():
        for path in sorted(MARIODPO_LEVELS_DIR.glob("*.txt")):
            items.append((f"mariodpo::{path.name}", "mariodpo", load_level(path)))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncd-sample", type=int, default=8,
                    help="number of original levels to average NCD against")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "level_features.csv"

    levels = _collect_levels()
    log.info("Collected %d levels across generators", len(levels))

    # Reference set for NCD: a fixed random sample of original levels.
    original_texts = [
        level_to_text(rows) for lid, gen, rows in levels if gen == "original"
    ]
    if original_texts:
        k = min(args.ncd_sample, len(original_texts))
        ncd_refs = random.sample(original_texts, k)
    else:
        ncd_refs = []
        log.warning("No 'original' levels found; ncd_to_original will be 0.0")

    fieldnames = ["level_id", "generator", *FEATURE_NAMES, "ncd_to_original"]
    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for level_id, generator, rows in levels:
            feats = extract_features(rows)
            if ncd_refs:
                text = level_to_text(rows)
                ncd = sum(
                    normalised_compression_distance(text, ref) for ref in ncd_refs
                ) / len(ncd_refs)
            else:
                ncd = 0.0
            record = {"level_id": level_id, "generator": generator,
                      "ncd_to_original": ncd, **feats}
            writer.writerow(record)
            rows_written += 1

    log.info("Wrote %d feature rows -> %s", rows_written, out_path)
    log.info("Feature columns (%d): %s", len(FEATURE_NAMES) + 1,
             ", ".join(FEATURE_NAMES + ["ncd_to_original"]))


if __name__ == "__main__":
    main()
