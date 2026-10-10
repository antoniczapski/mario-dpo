"""Phase 8 — export generated levels in arena seed format for redeployment.

Generates ``--n`` levels with the DPO model (rejection sampling + judge
re-ranking + playability repair), validates each against the backend importer
rules, and writes them to ``outputs/seed/mariodpo/`` together with a
``generator.json`` stub and per-level content hashes. Only valid levels are
kept, so the output can be dropped straight into the arena seed pipeline to
collect fresh human votes.

Usage:
    python scripts/08_export_seed.py --model models/dpo --n 100
    python scripts/08_export_seed.py --model models/dpo --dummy --n 4
"""

from __future__ import annotations

import argparse
import hashlib
import json

import _bootstrap  # noqa: F401

from mariodpo.constants import OUTPUTS_DIR, PROJECT_DIR
from mariodpo.level_io import iter_seed_levels, level_to_text, load_level
from mariodpo.utils import load_dotenv, set_seed, setup_logging
from mariodpo.validate import validate_level

log = setup_logging()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_judge():
    import joblib

    from mariodpo.constants import MODELS_DIR

    return joblib.load(MODELS_DIR / "judge.pkl")["judge"]


def _ncd_refs(n, seed):
    import random

    rng = random.Random(seed)
    texts = [level_to_text(load_level(p)) for g, f, p in iter_seed_levels(["original"])]
    return rng.sample(texts, min(n, len(texts))) if texts else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dpo")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-candidates", type=int, default=10)
    ap.add_argument("--target-cols", type=int, default=180)
    ap.add_argument("--min-width", type=int, default=150)
    ap.add_argument("--max-width", type=int, default=250)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_dotenv()
    set_seed(args.seed)

    from mariodpo.generate import generate_best
    from mariodpo.modeling import load_model, load_tokenizer

    model_dir = PROJECT_DIR / args.model
    tokenizer = load_tokenizer(str(model_dir))
    model = load_model(checkpoint=str(model_dir))
    from mariodpo.modeling import has_cuda
    if has_cuda() and not args.dummy:
        model = model.cuda()
    model.eval()

    judge = _load_judge()
    ncd_refs = _ncd_refs(8, args.seed)

    out_dir = OUTPUTS_DIR / "seed" / "mariodpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 4 if args.dummy else args.n
    n_cand = 2 if args.dummy else args.n_candidates
    target_cols = 12 if args.dummy else args.target_cols
    min_width = 1 if args.dummy else args.min_width

    manifest, kept = [], 0
    for i in range(n):
        best = generate_best(
            model, tokenizer, judge, ncd_refs,
            n_candidates=n_cand, target_cols=target_cols,
            min_width=min_width, require_valid=not args.dummy,
            seed=args.seed + i * n_cand,
        )
        text = level_to_text(best.rows)
        errors = validate_level(best.rows, min_width=min_width, max_width=args.max_width)
        if errors and not args.dummy:
            log.warning("Skipping invalid generated level %d: %s", i, errors)
            continue
        fname = f"level_{kept:03d}.txt"
        (out_dir / fname).write_text(text, encoding="utf-8")
        manifest.append({
            "file": fname,
            "content_hash": _content_hash(text),
            "width": len(best.rows[0]) if best.rows else 0,
            "height": len(best.rows),
            "judge_score": best.judge_score,
            "valid": not errors,
        })
        kept += 1

    generator_stub = {
        "id": "mariodpo",
        "name": "MarioDPO",
        "description": (
            "GPT-2 (MarioGPT) continue-pretrained into the arena representation "
            "and DPO-aligned on oversampled human arena votes plus judge-labelled "
            "synthetic preference pairs."
        ),
        "paradigm": "ML + DPO",
        "level_count": kept,
    }
    (out_dir / "generator.json").write_text(json.dumps(generator_stub, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_valid = sum(1 for m in manifest if m["valid"])
    log.info("Exported %d levels (%d valid) -> %s", kept, n_valid, out_dir)
    log.info("Wrote generator.json + manifest.json for arena registration")
    if not args.dummy and kept < n:
        log.warning("%d/%d generations were rejected as invalid", n - kept, n)


if __name__ == "__main__":
    main()
