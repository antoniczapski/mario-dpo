"""Phase 1b — train and select the Judge Function.

Loads level features + votes, builds an antisymmetric pairwise dataset, splits
90/10 *by vote*, trains four candidate judges (interpretable heuristic, linear
Bradley-Terry, gradient boosting, MLP), selects the best by held-out AUC, and
writes:
  * ``models/judge.pkl``                — selected judge (+ scaler + metadata)
  * ``data/processed/level_scores.csv`` — judge score for every level
  * ``outputs/judge_model_comparison.png``
  * ``outputs/judge_score_vs_winrate.png``
  * ``outputs/judge_weights.png``
  * ``outputs/judge_report.json``

Usage:
    python scripts/02_train_judge.py [--test-frac 0.1] [--seed 42]
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from mariodpo.constants import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    VOTES_FILENAME,
)
from mariodpo.judge import (
    ALL_FEATURES,
    HEURISTIC_FEATURES,
    HeuristicJudge,
    LinearBTJudge,
    SklearnPairJudge,
    build_pair_dataset,
    evaluate_pairwise,
    grid_search_heuristic,
    split_by_group,
)
from mariodpo.utils import setup_logging
from mariodpo.votes import generator_outcomes, iter_preference_pairs, load_votes

log = setup_logging()


def _load_feature_lookup() -> tuple[dict[str, dict], pd.DataFrame]:
    df = pd.read_csv(DATA_PROCESSED_DIR / "level_features.csv")
    lookup = {row["level_id"]: row.to_dict() for _, row in df.iterrows()}
    return lookup, df


def _build_pairs(feat_lookup: dict[str, dict]):
    bundle = load_votes(DATA_RAW_DIR / VOTES_FILENAME)
    pref = iter_preference_pairs(bundle)
    pairs = []
    skipped = 0
    for p in pref:
        wf = feat_lookup.get(p.winner_level_id)
        lf = feat_lookup.get(p.loser_level_id)
        if wf is None or lf is None:
            skipped += 1
            continue
        pairs.append((wf, lf, p.vote_id))
    log.info("Built %d feature pairs from votes (%d skipped, unresolved)",
             len(pairs), skipped)
    return pairs, bundle


def _spearman_generator(judge, feat_df: pd.DataFrame, bundle) -> float:
    """Spearman between mean judge score and empirical win rate per generator."""
    from scipy.stats import spearmanr

    gen_out = generator_outcomes(bundle)
    scores = {}
    for gen, sub in feat_df.groupby("generator"):
        vals = [judge.score(row.to_dict()) for _, row in sub.iterrows()]
        scores[gen] = float(np.mean(vals))
    rows = [(scores[g], gen_out[g]["win_rate"])
            for g in scores if g in gen_out and not np.isnan(gen_out[g]["win_rate"])]
    if len(rows) < 3:
        return float("nan")
    s = spearmanr([r[0] for r in rows], [r[1] for r in rows])
    return float(s.statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feat_lookup, feat_df = _load_feature_lookup()
    pairs, bundle = _build_pairs(feat_lookup)

    # Corpus standardisation stats (fit on every level once).
    corpus = np.array([[float(r[n]) for n in ALL_FEATURES]
                       for r in feat_lookup.values()])
    means = corpus.mean(axis=0)
    stds = corpus.std(axis=0)
    stds[stds == 0] = 1.0

    # Antisymmetric pair dataset + 90/10 split by vote.
    ds = build_pair_dataset(pairs, ALL_FEATURES)
    train_mask, test_mask = split_by_group(ds, args.test_frac, args.seed)
    # Map back to (winner,loser) pair-level train/test using vote groups.
    train_votes = set(ds.groups[train_mask].tolist())
    train_pairs = [p for p in pairs if p[2] in train_votes]
    test_pairs = [p for p in pairs if p[2] not in train_votes]
    log.info("Split: %d train pairs / %d test pairs (by vote)",
             len(train_pairs), len(test_pairs))

    # Standardised diff matrices for the ML judges.
    Xtr = ds.X[train_mask] / stds
    ytr = ds.y[train_mask]

    candidates: dict[str, object] = {}

    # 1. Interpretable heuristic (grid search) ------------------------------
    best_w, tr_acc = grid_search_heuristic(
        train_pairs, list(feat_lookup.values())
    )
    hcols = list(HEURISTIC_FEATURES.values())
    harr = np.array([[float(f[c]) for c in hcols] for f in feat_lookup.values()])
    heuristic = HeuristicJudge(
        weights=best_w,
        means={c: float(harr[:, i].mean()) for i, c in enumerate(hcols)},
        stds={c: float(harr[:, i].std() or 1.0) for i, c in enumerate(hcols)},
    )
    candidates["heuristic"] = heuristic
    log.info("Heuristic best weights %s (train acc %.3f)", best_w, tr_acc)

    # 2. Linear Bradley-Terry ----------------------------------------------
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
    lr.fit(Xtr, ytr)
    linear = LinearBTJudge(coef=lr.coef_[0].copy(), means=means.copy(),
                           stds=stds.copy(), feature_names=list(ALL_FEATURES))
    candidates["linear_bt"] = linear

    # Reference set (standardised) for nonlinear per-level scoring.
    rng = np.random.default_rng(args.seed)
    ref_idx = rng.choice(corpus.shape[0], size=min(64, corpus.shape[0]),
                         replace=False)
    reference = (corpus[ref_idx] - means) / stds

    # 3. Gradient boosting --------------------------------------------------
    from sklearn.ensemble import GradientBoostingClassifier

    gb = GradientBoostingClassifier(random_state=args.seed)
    gb.fit(Xtr, ytr)
    candidates["gradient_boosting"] = SklearnPairJudge(
        model=gb, means=means.copy(), stds=stds.copy(), reference=reference,
        name="gradient_boosting", feature_names=list(ALL_FEATURES))

    # 4. MLP ----------------------------------------------------------------
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                        random_state=args.seed, early_stopping=True)
    mlp.fit(Xtr, ytr)
    candidates["mlp"] = SklearnPairJudge(
        model=mlp, means=means.copy(), stds=stds.copy(), reference=reference,
        name="mlp", feature_names=list(ALL_FEATURES))

    # --- Evaluate all on held-out test ------------------------------------
    report = {}
    for name, judge in candidates.items():
        metrics = evaluate_pairwise(judge, test_pairs)
        metrics["generator_spearman"] = _spearman_generator(judge, feat_df, bundle)
        metrics["train_accuracy"] = (
            evaluate_pairwise(judge, train_pairs)["accuracy"]
        )
        report[name] = metrics
        log.info("%-18s | test acc %.3f | auc %.3f | gen-spearman %.3f",
                 name, metrics["accuracy"], metrics["auc"],
                 metrics["generator_spearman"])

    # Select by held-out AUC (tie-break accuracy).
    best_name = max(report, key=lambda n: (report[n]["auc"], report[n]["accuracy"]))
    best_judge = candidates[best_name]
    log.info("SELECTED judge: %s", best_name)

    # --- Persist -----------------------------------------------------------
    joblib.dump(
        {"judge": best_judge, "name": best_name, "feature_names": ALL_FEATURES,
         "report": report},
        MODELS_DIR / "judge.pkl",
    )
    # Score every level and save.
    scores = [best_judge.score(r) for r in feat_lookup.values()]
    score_df = pd.DataFrame({
        "level_id": list(feat_lookup.keys()),
        "generator": [r["generator"] for r in feat_lookup.values()],
        "judge_score": scores,
    })
    score_df.to_csv(DATA_PROCESSED_DIR / "level_scores.csv", index=False)
    (OUTPUTS_DIR / "judge_report.json").write_text(
        json.dumps({"selected": best_name, "report": report}, indent=2)
    )

    _make_plots(report, best_name, best_judge, score_df, feat_df, bundle)
    log.info("Judge artifacts written to models/ and outputs/")


def _make_plots(report, best_name, best_judge, score_df, feat_df, bundle) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Model comparison.
    names = list(report)
    accs = [report[n]["accuracy"] for n in names]
    aucs = [report[n]["auc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, accs, 0.4, label="accuracy")
    ax.bar(x + 0.2, aucs, 0.4, label="AUC")
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("held-out score")
    ax.set_title("Judge candidates — held-out pairwise performance")
    for i, n in enumerate(names):
        if n == best_name:
            ax.text(i, 0.02, "selected", ha="center", fontsize=8, color="darkgreen")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "judge_model_comparison.png", dpi=130)
    plt.close(fig)

    # 2. Judge score vs generator win rate.
    gen_out = generator_outcomes(bundle)
    gmean = score_df.groupby("generator")["judge_score"].mean()
    rows = [(g, gmean[g], gen_out[g]["win_rate"]) for g in gmean.index
            if g in gen_out and not np.isnan(gen_out[g]["win_rate"])]
    fig, ax = plt.subplots(figsize=(7, 6))
    for g, s, wr in rows:
        ax.scatter(s, wr, s=40)
        ax.annotate(g, (s, wr), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("mean judge score"); ax.set_ylabel("empirical win rate")
    ax.set_title(f"Judge ({best_name}) vs human win rate by generator")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "judge_score_vs_winrate.png", dpi=130)
    plt.close(fig)

    # 3. Weights / importance (only for interpretable judges).
    fig, ax = plt.subplots(figsize=(8, 5))
    if isinstance(best_judge, HeuristicJudge):
        terms = list(best_judge.weights)
        ax.barh(terms, [best_judge.weights[t] for t in terms])
        ax.set_title("Heuristic judge — term weights")
    elif isinstance(best_judge, LinearBTJudge):
        order = np.argsort(np.abs(best_judge.coef))[-15:]
        ax.barh([best_judge.feature_names[i] for i in order],
                best_judge.coef[order])
        ax.set_title("Linear BT judge — top feature weights")
    else:
        ax.text(0.5, 0.5, f"{best_name}: no linear weights",
                ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "judge_weights.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
