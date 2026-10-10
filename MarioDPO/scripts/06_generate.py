"""Phase 6a — generate levels from a trained model with judge rejection sampling.

For each requested level, samples ``--n-candidates`` raws, decodes + repairs +
validates each, scores with the static judge, and keeps the best. Writes levels
to ``outputs/generated/<tag>/`` plus a ``manifest.json`` and renders a few
ASCII previews.

Usage:
    python scripts/06_generate.py --model models/dpo --n 50
    python scripts/06_generate.py --model models/dpo --dummy --n 4
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from mariodpo.constants import OUTPUTS_DIR, PROJECT_DIR
from mariodpo.level_io import iter_seed_levels, level_to_text, load_level
from mariodpo.utils import load_dotenv, set_seed, setup_logging

log = setup_logging()


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
    ap.add_argument("--tag", default=None, help="output sub-folder name")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-candidates", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-width", type=int, default=150)
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

    tag = args.tag or ("dummy" if args.dummy else model_dir.name)
    out_dir = OUTPUTS_DIR / "generated" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 4 if args.dummy else args.n
    n_cand = 3 if args.dummy else args.n_candidates
    # Tiny dummy model can't produce 150-col levels; relax width + target.
    min_width = 1 if args.dummy else args.min_width
    target_cols = 12 if args.dummy else max(args.min_width, 180)

    manifest = []
    for i in range(n):
        best = generate_best(
            model, tokenizer, judge, ncd_refs,
            n_candidates=n_cand, target_cols=target_cols, min_width=min_width,
            require_valid=not args.dummy, temperature=args.temperature,
            seed=args.seed + i * n_cand,
        )
        fname = f"mariodpo_{i:04d}.txt"
        (out_dir / fname).write_text(level_to_text(best.rows), encoding="utf-8")
        manifest.append({
            "file": fname, "judge_score": best.judge_score,
            "valid": best.valid, "width": len(best.rows[0]) if best.rows else 0,
            "errors": best.errors,
        })
        log.info("[%d/%d] %s judge=%.3f valid=%s width=%d",
                 i + 1, n, fname, best.judge_score, best.valid,
                 len(best.rows[0]) if best.rows else 0)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_valid = sum(1 for m in manifest if m["valid"])
    mean_score = sum(m["judge_score"] for m in manifest) / max(1, len(manifest))
    log.info("Generated %d levels -> %s (valid %d/%d, mean judge %.3f)",
             n, out_dir, n_valid, n, mean_score)


if __name__ == "__main__":
    main()
