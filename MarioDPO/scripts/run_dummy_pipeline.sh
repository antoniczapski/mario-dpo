#!/usr/bin/env bash
# End-to-end CPU smoke test for the MarioDPO pipeline.
# Runs every stage with tiny models/data and no GPU/network, proving the wiring.
# Real training uses the same scripts WITHOUT --dummy (see README).
set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=disabled   # no W&B network calls in the smoke test

echo "== [0/8] prepare data =="
python3 scripts/00_prepare_data.py

echo "== [1/8] extract features =="
python3 scripts/01_extract_features.py

echo "== [2/8] train judge =="
python3 scripts/02_train_judge.py

echo "== [3/8] SFT (dummy, 2 steps) =="
python3 scripts/03_sft.py --dummy --max-steps 2

echo "== [4/8] build preference pairs (dummy) =="
python3 scripts/04_build_pairs.py --dummy

echo "== [5/8] DPO (dummy, 2 steps) =="
python3 scripts/05_dpo.py --dummy --max-steps 2

echo "== [6/8] generate (dummy) =="
python3 scripts/06_generate.py --model models/dpo --dummy --n 4

echo "== [7/8] evaluate (dummy) =="
python3 scripts/07_evaluate.py --dpo models/dpo --ref models/sft --dummy

echo "== [8/8] export seed (dummy) =="
python3 scripts/08_export_seed.py --model models/dpo --dummy --n 4

echo
echo "ALL STAGES OK. Artifacts under models/ and outputs/."
