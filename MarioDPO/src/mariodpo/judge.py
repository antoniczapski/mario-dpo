"""The Judge Function: a static-feature scorer of human-preferred-ness.

Design principles (fixing the original implementation's flaws):
  * **Static only** — every input comes from :mod:`mariodpo.features`, so the
    judge can score freshly generated, never-played levels.
  * **No identity leak** — the generator name is never an input.
  * **Per-level scorer** — exposes ``score(features) -> float`` so it can rank
    generated candidates and build synthetic preference pairs; pairwise
    accuracy is then ``score(a) > score(b)``.
  * **Validated honestly** — trained/selected on a 90/10 split *by vote*, with a
    secondary per-generator Spearman check against empirical win rate.

Candidate judges:
  1. :class:`HeuristicJudge`   — interpretable linear scorer over a few terms,
     grid-searched (the "thesis heuristic", static-only).
  2. :class:`LinearBTJudge`    — Bradley-Terry logistic model on feature
     differences; ``score = w · f`` (a principled linear scorer).
  3. :class:`SklearnPairJudge` — nonlinear classifier (gradient boosting / MLP)
     on feature differences; per-level score via a reference set.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES

# Features (subset of FEATURE_NAMES + ncd) the interpretable heuristic uses.
HEURISTIC_FEATURES = {
    "style": "ncd_to_original",        # closer to original -> better (negative weight)
    "gap": "gap_fraction",             # pits -> worse (negative weight)
    "reward": "reward_density",        # collectibles -> better
    "vertical": "surface_roughness",   # spatial variety -> better
    "enemy": "enemy_density",          # hazards -> mild penalty
}

# Full ordered feature list used by the ML judges (static features + ncd).
ALL_FEATURES = [*FEATURE_NAMES, "ncd_to_original"]


def features_to_array(rows: list[dict], names: list[str]) -> np.ndarray:
    """Stack a list of feature dicts into an array in ``names`` order."""
    return np.asarray([[float(r[n]) for n in names] for r in rows], dtype=float)


# --- Pairwise dataset ------------------------------------------------------
@dataclass
class PairDataset:
    """Antisymmetric pairwise dataset of feature differences.

    For each preference (winner ``w`` beats loser ``l``) we add both
    ``+(f_w - f_l)`` with label 1 and ``-(f_w - f_l)`` with label 0, so models
    learn an antisymmetric boundary. ``groups`` holds the vote id per row to
    keep both halves of a vote in the same split.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


def build_pair_dataset(
    pairs: list[tuple[dict, dict, str]],
    feature_names: list[str],
) -> PairDataset:
    """Build a :class:`PairDataset` from ``(winner_feats, loser_feats, vote_id)``."""
    xs, ys, gs = [], [], []
    for w_feat, l_feat, vote_id in pairs:
        fw = np.array([float(w_feat[n]) for n in feature_names])
        fl = np.array([float(l_feat[n]) for n in feature_names])
        diff = fw - fl
        xs.append(diff); ys.append(1); gs.append(vote_id)
        xs.append(-diff); ys.append(0); gs.append(vote_id)
    return PairDataset(
        X=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=int),
        groups=np.asarray(gs, dtype=object),
        feature_names=list(feature_names),
    )


def split_by_group(
    ds: PairDataset, test_frac: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean train/test masks splitting *by vote group* (no leakage)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(ds.groups)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_frac)))
    test_groups = set(unique[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in ds.groups])
    return ~test_mask, test_mask


# --- Judge interface -------------------------------------------------------
class BaseJudge:
    """Common interface: ``score`` a single level, ``predict_pair`` a comparison."""

    name = "base"
    feature_names: list[str] = ALL_FEATURES

    def score(self, feat: dict) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_pair(self, fa: dict, fb: dict) -> float:
        """Probability that ``a`` beats ``b`` (default: logistic of score gap)."""
        return float(1.0 / (1.0 + np.exp(-(self.score(fa) - self.score(fb)))))

    def score_pair_correct(self, fa: dict, fb: dict) -> bool:
        return self.score(fa) > self.score(fb)


@dataclass
class HeuristicJudge(BaseJudge):
    """Interpretable linear scorer over a handful of standardised terms.

    ``score = Σ_k w_k · z(feature_k)`` where ``z`` standardises each term using
    means/stds fit on the corpus. Weights are grid-searched (see
    :func:`grid_search_heuristic`).
    """

    weights: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    name: str = "heuristic"
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FEATURES))

    def _z(self, feat: dict, col: str) -> float:
        std = self.stds.get(col, 1.0) or 1.0
        return (float(feat[col]) - self.means.get(col, 0.0)) / std

    def score(self, feat: dict) -> float:
        s = 0.0
        for term, col in HEURISTIC_FEATURES.items():
            s += self.weights.get(term, 0.0) * self._z(feat, col)
        return s


@dataclass
class LinearBTJudge(BaseJudge):
    """Bradley-Terry logistic scorer: ``score = w · z(f)`` (linear, interpretable)."""

    coef: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    name: str = "linear_bt"
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FEATURES))

    def _z(self, feat: dict) -> np.ndarray:
        x = np.array([float(feat[n]) for n in self.feature_names])
        return (x - self.means) / self.stds

    def score(self, feat: dict) -> float:
        return float(self.coef @ self._z(feat))


@dataclass
class SklearnPairJudge(BaseJudge):
    """Nonlinear classifier on feature differences; per-level score via refs.

    ``score(level) = mean_r [ P(level beats r) - 0.5 ]`` over a fixed reference
    set of standardised feature vectors. Antisymmetric training keeps this well
    defined.
    """

    model: object
    means: np.ndarray
    stds: np.ndarray
    reference: np.ndarray            # standardised feature vectors (n_ref, d)
    name: str = "sklearn_pair"
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FEATURES))

    def _z(self, feat: dict) -> np.ndarray:
        x = np.array([float(feat[n]) for n in self.feature_names])
        return (x - self.means) / self.stds

    def _proba(self, diff: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(diff)[:, 1]

    def score(self, feat: dict) -> float:
        z = self._z(feat)
        diffs = z[None, :] - self.reference
        return float(np.mean(self._proba(diffs) - 0.5))

    def predict_pair(self, fa: dict, fb: dict) -> float:
        diff = (self._z(fa) - self._z(fb))[None, :]
        return float(self._proba(diff)[0])

    def score_pair_correct(self, fa: dict, fb: dict) -> bool:
        return self.predict_pair(fa, fb) > 0.5


# --- Training helpers ------------------------------------------------------
def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def grid_search_heuristic(
    train_pairs: list[tuple[dict, dict, str]],
    corpus_feats: list[dict],
    grid: dict[str, list[float]] | None = None,
) -> tuple[dict[str, float], float]:
    """Grid-search interpretable weights to maximise train pairwise accuracy.

    Returns ``(best_weights, train_accuracy)``. Signs are fixed by domain
    knowledge (style/gap/enemy negative-leaning via the grid ranges below);
    magnitudes are searched.
    """
    if grid is None:
        grid = {
            "style": [-1.5, -1.0, -0.5, 0.0],
            "gap": [-1.5, -1.0, -0.5, 0.0],
            "reward": [0.0, 0.5, 1.0, 1.5],
            "vertical": [0.0, 0.25, 0.5],
            "enemy": [-1.0, -0.5, 0.0],
        }
    cols = list(HEURISTIC_FEATURES.values())
    arr = np.array([[float(f[c]) for c in cols] for f in corpus_feats])
    means = {c: float(arr[:, i].mean()) for i, c in enumerate(cols)}
    stds = {c: float(arr[:, i].std() or 1.0) for i, c in enumerate(cols)}

    # Pre-standardise the winner/loser term vectors once.
    def zvec(feat: dict) -> dict[str, float]:
        return {
            term: (float(feat[col]) - means[col]) / stds[col]
            for term, col in HEURISTIC_FEATURES.items()
        }

    zpairs = [(zvec(w), zvec(l)) for w, l, _ in train_pairs]
    terms = list(HEURISTIC_FEATURES.keys())

    best_w, best_acc = None, -1.0
    for combo in itertools.product(*(grid[t] for t in terms)):
        w = dict(zip(terms, combo))
        if all(v == 0.0 for v in w.values()):
            continue
        correct = 0
        for zw, zl in zpairs:
            sw = sum(w[t] * zw[t] for t in terms)
            sl = sum(w[t] * zl[t] for t in terms)
            if sw > sl:
                correct += 1
        acc = correct / len(zpairs)
        if acc > best_acc:
            best_acc, best_w = acc, w
    return best_w, best_acc


def evaluate_pairwise(
    judge: BaseJudge, test_pairs: list[tuple[dict, dict, str]]
) -> dict[str, float]:
    """Return held-out pairwise accuracy and AUC of ``judge`` on test pairs."""
    from sklearn.metrics import roc_auc_score

    probs, labels = [], []
    correct = 0
    for w_feat, l_feat, _ in test_pairs:
        # Randomly orient so accuracy isn't trivially 100% from (winner,loser) order.
        probs.append(judge.predict_pair(w_feat, l_feat)); labels.append(1)
        probs.append(judge.predict_pair(l_feat, w_feat)); labels.append(0)
        if judge.score_pair_correct(w_feat, l_feat):
            correct += 1
    acc = correct / max(1, len(test_pairs))
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc, "n_pairs": len(test_pairs)}
