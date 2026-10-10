"""Phase 0 — prepare raw data.

Copies the May 2026 votes export into ``data/raw/`` and prints a dataset audit
(result distribution, usable pairs, generators, level resolvability). Idempotent.

Usage:
    python scripts/00_prepare_data.py
"""

from __future__ import annotations

import shutil
from collections import Counter

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

from mariodpo.constants import (
    DATA_RAW_DIR,
    REPO_DIR,
    VOTES_FILENAME,
)
from mariodpo.level_io import resolve_level_path
from mariodpo.utils import setup_logging
from mariodpo.votes import iter_preference_pairs, load_votes

log = setup_logging()

SOURCE_VOTES = REPO_DIR / "eda" / "data_10_05_2026" / VOTES_FILENAME


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / VOTES_FILENAME

    if not SOURCE_VOTES.exists():
        raise SystemExit(f"Source votes export not found: {SOURCE_VOTES}")

    if dest.resolve() != SOURCE_VOTES.resolve():
        shutil.copy2(SOURCE_VOTES, dest)
        log.info("Copied votes export -> %s", dest)
    else:
        log.info("Votes export already in place: %s", dest)

    bundle = load_votes(dest)
    log.info("Votes on disk: %d (export total: %d)", bundle.on_disk, bundle.total)
    if bundle.truncated:
        log.warning(
            "Export is PAGINATED: %d of %d records present. The pipeline uses the "
            "%d available; re-export with offset=%d to obtain the remainder.",
            bundle.on_disk, bundle.total, bundle.on_disk, bundle.on_disk,
        )

    results = Counter(v.get("result") for v in bundle.data)
    log.info("Result distribution: %s", dict(results))

    pairs = iter_preference_pairs(bundle)
    log.info("Usable preference pairs (LEFT/RIGHT): %d", len(pairs))

    gens = Counter()
    for v in bundle.data:
        gens[v["left_generator_id"]] += 1
        gens[v["right_generator_id"]] += 1
    log.info("Generators appearing in votes (%d): %s", len(gens), dict(gens))

    # Resolvability check across all level ids referenced by usable pairs.
    referenced = set()
    for p in pairs:
        referenced.add(p.winner_level_id)
        referenced.add(p.loser_level_id)
    unresolved = sorted(lid for lid in referenced if resolve_level_path(lid) is None)
    log.info(
        "Referenced levels: %d distinct; unresolved on disk: %d",
        len(referenced), len(unresolved),
    )
    if unresolved:
        sample = ", ".join(unresolved[:10])
        log.warning("Unresolved level ids (sample): %s", sample)


if __name__ == "__main__":
    main()
