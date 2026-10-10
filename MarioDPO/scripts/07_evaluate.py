"""Phase 7 — evaluate a DPO model against its SFT reference and baselines.

Three complementary views (no new human votes needed):
  1. **Held-out preference accuracy** — on test-split human pairs, does the model
     assign higher conditional log-prob to the human-preferred level? Reported
     for pi_DPO and pi_ref; an increase is the core DPO signal.
  2. **Judge-score distribution** — static judge scores of generated DPO levels
     vs the seed baselines (original, mariogpt, ore, patterns).
  3. **Validity / structure** — fraction of generated levels passing arena
     validation, plus a couple of rendered samples.

Outputs: ``outputs/evaluation/eval_report.json`` and ``eval_*.png``.

Usage:
    python scripts/07_evaluate.py --dpo models/dpo --ref models/sft
    python scripts/07_evaluate.py --dpo models/dpo --ref models/sft --dummy
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from mariodpo.constants import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_DIR,
    VOTES_FILENAME,
)
from mariodpo.level_io import level_to_text, load_level, resolve_level_path
from mariodpo.pairs import _window_text
from mariodpo.tokenizer_io import style_prompt
from mariodpo.utils import load_dotenv, set_seed, setup_logging
from mariodpo.votes import iter_preference_pairs, load_votes

log = setup_logging()
EVAL_DIR = OUTPUTS_DIR / "evaluation"


def _held_out_pairs(seed: int, test_frac: float, limit: int | None):
    """Reconstruct the same by-vote test split used to train the judge."""
    bundle = load_votes(DATA_RAW_DIR / VOTES_FILENAME)
    pref = iter_preference_pairs(bundle)
    rng = np.random.default_rng(seed)
    vote_ids = sorted({p.vote_id for p in pref})
    rng.shuffle(vote_ids)
    n_test = max(1, int(round(len(vote_ids) * test_frac)))
    test_set = set(vote_ids[:n_test])
    pairs = []
    for p in pref:
        if p.vote_id not in test_set:
            continue
        wp, lp = resolve_level_path(p.winner_level_id), resolve_level_path(p.loser_level_id)
        if wp and lp:
            pairs.append((load_level(wp), load_level(lp)))
        if limit and len(pairs) >= limit:
            break
    return pairs


def _preference_accuracy(model, tokenizer, pairs, win_cols=48) -> float:
    from mariodpo.modeling import sequence_logprob

    prompt = style_prompt("NINTENDO")
    correct = 0
    for w_rows, l_rows in pairs:
        lp_w = sequence_logprob(model, tokenizer, prompt, _window_text(w_rows, win_cols))
        lp_l = sequence_logprob(model, tokenizer, prompt, _window_text(l_rows, win_cols))
        correct += int(lp_w > lp_l)
    return correct / max(1, len(pairs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", default="models/dpo")
    ap.add_argument("--ref", default="models/sft")
    ap.add_argument("--generated", default=None,
                    help="dir of generated DPO levels (default outputs/generated/<dpo>)")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_dotenv()
    set_seed(args.seed)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    from mariodpo.modeling import load_model, load_tokenizer

    judge = joblib.load(MODELS_DIR / "judge.pkl")["judge"]

    report: dict = {}

    # --- 1. Held-out preference accuracy ----------------------------------
    limit = 20 if args.dummy else None
    pairs = _held_out_pairs(args.seed, args.test_frac, limit)
    log.info("Held-out human pairs for preference accuracy: %d", len(pairs))

    from mariodpo.modeling import has_cuda
    for tag, path in [("ref", args.ref), ("dpo", args.dpo)]:
        mdir = PROJECT_DIR / path
        tok = load_tokenizer(str(mdir))
        model = load_model(checkpoint=str(mdir))
        if has_cuda() and not args.dummy:
            model = model.cuda()
        model.eval()
        acc = _preference_accuracy(model, tok, pairs)
        report[f"pref_acc_{tag}"] = acc
        log.info("Preference accuracy (%s): %.3f", tag, acc)

    report["pref_acc_delta"] = report["pref_acc_dpo"] - report["pref_acc_ref"]

    # --- 2. Judge-score distribution of generated vs baselines ------------
    default_tag = "dummy" if args.dummy else (PROJECT_DIR / args.dpo).name
    gen_dir = OUTPUTS_DIR / "generated" / (args.generated or default_tag)
    gen_scores = _score_generated(judge, gen_dir)
    baseline_scores = _baseline_scores()
    report["generated_judge_mean"] = float(np.mean(gen_scores)) if gen_scores else None
    report["generated_n"] = len(gen_scores)
    report["baseline_judge_means"] = {g: float(np.mean(s)) for g, s in baseline_scores.items()}

    # --- 3. Validity of generated levels ----------------------------------
    report["generated_valid_fraction"] = _valid_fraction(gen_dir)

    (EVAL_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    log.info("Eval report: %s", json.dumps(report, indent=2))

    _plots(report, gen_scores, baseline_scores)
    log.info("Wrote evaluation artifacts -> %s", EVAL_DIR)


def _score_generated(judge, gen_dir) -> list[float]:
    import random

    from mariodpo.features import extract_features
    from mariodpo.level_io import iter_seed_levels
    from mariodpo.utils import normalised_compression_distance

    if not gen_dir.exists():
        log.warning("No generated dir at %s; skipping generated judge scores", gen_dir)
        return []
    rng = random.Random(42)
    refs = [level_to_text(load_level(p)) for g, f, p in iter_seed_levels(["original"])]
    refs = rng.sample(refs, min(8, len(refs))) if refs else []
    scores = []
    for p in sorted(gen_dir.glob("*.txt")):
        rows = load_level(p)
        feats = extract_features(rows)
        feats["ncd_to_original"] = (
            sum(normalised_compression_distance(level_to_text(rows), r) for r in refs)
            / len(refs) if refs else 0.0
        )
        feats["generator"] = "mariodpo"
        scores.append(judge.score(feats))
    return scores


def _baseline_scores() -> dict[str, list[float]]:
    import pandas as pd

    df = pd.read_csv(DATA_PROCESSED_DIR / "level_scores.csv")
    wanted = ["original", "mariogpt", "ore", "mariodpo",
              "patternCount", "patternOccur", "patternWeightCount"]
    out = {}
    for g in wanted:
        vals = df[df["generator"] == g]["judge_score"].tolist()
        if vals:
            out[g] = vals
    return out


def _valid_fraction(gen_dir) -> float | None:
    from mariodpo.validate import is_valid

    if not gen_dir.exists():
        return None
    files = sorted(gen_dir.glob("*.txt"))
    if not files:
        return None
    valid = sum(1 for p in files if is_valid(load_level(p), min_width=1))
    return valid / len(files)


def _plots(report, gen_scores, baseline_scores) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Preference accuracy bar.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["pi_ref (SFT)", "pi_DPO"],
           [report["pref_acc_ref"], report["pref_acc_dpo"]],
           color=["grey", "darkgreen"])
    ax.axhline(0.5, ls="--", c="red", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("held-out preference accuracy")
    ax.set_title("Does the model prefer human-preferred levels?")
    fig.tight_layout(); fig.savefig(EVAL_DIR / "eval_preference_accuracy.png", dpi=130)
    plt.close(fig)

    # Judge-score distributions.
    fig, ax = plt.subplots(figsize=(9, 5))
    data, labels = [], []
    for g, s in baseline_scores.items():
        data.append(s); labels.append(g)
    if gen_scores:
        data.append(gen_scores); labels.append("mariodpo\n(generated)")
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:  # matplotlib < 3.9
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("judge score"); ax.set_title("Judge-score distribution: generated vs baselines")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(EVAL_DIR / "eval_judge_distributions.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
