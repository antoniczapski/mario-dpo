"""Generate clean thesis figures for Chapter 5 (MarioDPO).

Produces a judge-score distribution box plot comparing the DPO-generated levels
against the established baselines, with clean labels (no internal versioning).
Copies the other ready-made plots into the thesis img/ folder.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mariodpo.constants import DATA_PROCESSED_DIR, OUTPUTS_DIR, REPO_DIR
from mariodpo.level_io import load_level, level_to_text, iter_seed_levels
from mariodpo.features import extract_features
from mariodpo.utils import normalised_compression_distance, load_dotenv

load_dotenv()
IMG = REPO_DIR / "latex" / "masters_thesis" / "img"

# --- 1. Clean judge-score distribution box plot -----------------------------
scores = pd.read_csv(DATA_PROCESSED_DIR / "level_scores.csv")

# Recompute judge scores for the generated levels (same judge).
import joblib, random
from mariodpo.constants import MODELS_DIR
judge = joblib.load(MODELS_DIR / "judge.pkl")["judge"]
rng = random.Random(42)
refs = [level_to_text(load_level(p)) for g, f, p in iter_seed_levels(["original"])]
refs = rng.sample(refs, min(8, len(refs)))
gen_dir = OUTPUTS_DIR / "generated" / "dpo"
gen_scores = []
for p in sorted(gen_dir.glob("*.txt")):
    rows = load_level(p)
    feats = extract_features(rows)
    feats["ncd_to_original"] = sum(normalised_compression_distance(level_to_text(rows), r) for r in refs) / len(refs)
    feats["generator"] = "mariodpo"
    gen_scores.append(judge.score(feats))

# Baselines (the arena 'mariodpo' entry is omitted here; the newly generated
# MarioDPO levels are plotted separately as 'MarioDPO (ours)').
label_map = {
    "original": "Original",
    "mariogpt": "MarioGPT",
    "ore": "ORE",
    "patternWeightCount": "Pattern\nWt.Count",
    "patternOccur": "Pattern\nOccur",
    "patternCount": "Pattern\nCount",
}
groups = {disp: scores[scores["generator"] == gen]["judge_score"].tolist()
          for gen, disp in label_map.items()}
groups["MarioDPO\n(ours)"] = gen_scores

# Order by mean score descending.
ordered = sorted(groups.items(), key=lambda kv: -(sum(kv[1]) / len(kv[1])))
labels = [k for k, _ in ordered]
data = [v for _, v in ordered]
colors = ["#2c7fb8" if lbl == "MarioDPO\n(ours)" else "#bdbdbd" for lbl in labels]

fig, ax = plt.subplots(figsize=(9, 4.5))
bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                medianprops=dict(color="black"),
                meanprops=dict(marker="^", markerfacecolor="green", markeredgecolor="green"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("static judge score")
ax.axhline(0, ls=":", c="grey", lw=0.8)
ax.set_title("Static judge score: MarioDPO-generated levels vs baselines")
fig.tight_layout()
fig.savefig(IMG / "mariodpo_judge_distributions.png", dpi=140)
plt.close(fig)
print("Wrote mariodpo_judge_distributions.png")

# --- 2. Copy ready-made plots ----------------------------------------------
copies = {
    "judge_score_vs_winrate.png": "mariodpo_judge_winrate.png",
    "judge_weights.png": "mariodpo_judge_weights.png",
    "evaluation/eval_preference_accuracy.png": "mariodpo_preference_accuracy.png",
}
for src, dst in copies.items():
    shutil.copy2(OUTPUTS_DIR / src, IMG / dst)
    print(f"Copied {src} -> {dst}")

print("Done.")
