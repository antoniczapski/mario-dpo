# MarioDPO — Preference-Aligned Mario Level Generation

A reproducible implementation of the MarioDPO experiment for Chapter 5
of the thesis. It turns blind human preference votes collected by **PCG Arena**
into a **preference-aligned level generator**:

1. a **static-feature Judge Function** that predicts which of two levels a human
   would prefer — using *only* the tile grid, so it can score freshly generated
   levels (no gameplay telemetry, no generator identity);
2. **continue-pretraining** of the pretrained MarioGPT checkpoint into the arena
   16-row / 37-tile representation (`π_ref`);
3. **DPO alignment** on oversampled human arena votes plus judge-labelled
   synthetic preference pairs;
4. a **deployment-ready export** so the resulting generator can be put back on
   the live arena to collect fresh votes.

---

## 1. Repository layout

```
MarioDPO/
├── README.md                  ← this file
├── requirements.txt
├── pyproject.toml             ← installable package (src/mariodpo)
├── .env.example               ← copy to .env, fill tokens (gitignored)
├── configs/
│   ├── sft.yaml               ← continue-pretraining config
│   └── dpo.yaml               ← DPO config
├── scripts/                   ← numbered pipeline stages (run in order)
│   ├── 00_prepare_data.py
│   ├── 01_extract_features.py
│   ├── 02_train_judge.py
│   ├── 03_sft.py
│   ├── 04_build_pairs.py
│   ├── 05_dpo.py
│   ├── 06_generate.py
│   ├── 07_evaluate.py
│   ├── 08_export_seed.py
│   └── run_dummy_pipeline.sh  ← full CPU smoke test
├── src/mariodpo/           ← library code
├── tests/                     ← pytest round-trip / validation tests
├── data/   (raw, processed, synthetic)   ← gitignored except small tables
├── models/ (judge.pkl, sft/, dpo/)       ← large weights gitignored; judge + DPO adapter tracked
├── outputs/ (plots, runs, generated, seed, evaluation)  ← logs gitignored
└── generated_levels/      ← levels of the `mariodpo` generator deployed on the arena
```

---

## 2. Environment setup

### 2.1 Python deps (judge/data only — CPU)

For Phases 0–2 (features + judge) you only need the scientific stack:

```bash
cd MarioDPO
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib PyYAML tqdm pytest
pip install -e .          # makes `import mariodpo` work everywhere
```

### 2.2 Training deps (SFT + DPO — GPU recommended)

Install PyTorch matching your CUDA build first, then the ML stack:

```bash
# GPU box (CUDA 12.1 example) — for the NVIDIA GTX 2060:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# or CPU-only (smoke tests):
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

> **Version pins matter.** Use `transformers>=4.57` **with** `trl>=1.6`. Older
> `trl` (0.9–0.11) is incompatible with transformers 4.57 (a `get_batch_samples`
> signature change) and will crash during DPO. `rich` is a required transitive
> dependency of trl's callbacks.

### 2.3 Secrets (HuggingFace + Weights & Biases)

Tokens are **only** read from the environment — never hard-code them.

```bash
cp .env.example .env
# then edit .env and set:
#   HF_TOKEN=hf_...           (read scope; needed to pull the base checkpoint)
#   WANDB_API_KEY=...         (for experiment tracking)
#   WANDB_PROJECT=pcg-mariodpo
#   WANDB_MODE=online         (or "offline" / "disabled")
```

The scripts auto-load `.env`. To disable W&B entirely (e.g. offline), set
`WANDB_MODE=disabled`; training still logs metrics locally under
`outputs/runs/<run>/metrics.jsonl`.

> ⚠️ **Rotate any token that was ever shared in plaintext** (chat, commits, logs).
> Treat the values you pasted during development as compromised.

---

## 3. Quick start

### 3.1 Full CPU smoke test (no GPU, no network, ~2 min)

```bash
bash scripts/run_dummy_pipeline.sh
```

This runs every stage with a tiny randomly-initialised GPT-2 and tiny data
(`--dummy`, `--max-steps 2`) to prove the pipeline is wired end-to-end. It
writes `models/sft`, `models/dpo`, and artifacts under `outputs/`.

### 3.2 Real run (GPU)

```bash
# 0–2: data, features, judge  (CPU is fine, a few minutes)
python scripts/00_prepare_data.py
python scripts/01_extract_features.py
python scripts/02_train_judge.py

# 3: continue-pretrain MarioGPT -> arena representation (pi_ref)
python scripts/03_sft.py

# 4: build the DPO preference dataset (human oversampled + synthetic)
python scripts/04_build_pairs.py --n-synthetic 2000 --n-onpolicy 200

# 5: DPO alignment
python scripts/05_dpo.py

# 6–8: generate, evaluate, export for redeployment
python scripts/06_generate.py  --model models/dpo --n 100
python scripts/07_evaluate.py  --dpo models/dpo --ref models/sft
python scripts/08_export_seed.py --model models/dpo --n 100
```

---

## 4. Pipeline stages in detail

| # | Script | Input | Output | Notes |
|---|--------|-------|--------|-------|
| 0 | `00_prepare_data.py` | `eda/data_10_05_2026/…votes….json` | `data/raw/…votes….json` | Copies the May 2026 export; audits results, usable pairs, level resolvability. |
| 1 | `01_extract_features.py` | seed levels + recovered `mariodpo` | `data/processed/level_features.csv` | 34 static structural features per level + `ncd_to_original`. |
| 2 | `02_train_judge.py` | features + votes | `models/judge.pkl`, `data/processed/level_scores.csv`, `outputs/judge_*.png` | Trains 4 candidate judges, **90/10 split by vote**, selects best by held-out AUC. |
| 3 | `03_sft.py` | seed levels | `models/sft/` | Continue-pretrains MarioGPT in arena column-major space (`π_ref`). |
| 4 | `04_build_pairs.py` | votes + judge (+ SFT model) | `data/synthetic/preference_pairs.jsonl` | Human pairs (oversampled 10×) + judge-labelled synthetic pairs. |
| 5 | `05_dpo.py` | SFT model + pairs | `models/dpo/` | trl `DPOTrainer`; LoRA recommended on 6 GB. |
| 6 | `06_generate.py` | DPO model + judge | `outputs/generated/<tag>/` | Sliding-window generation + judge rejection sampling + repair. |
| 7 | `07_evaluate.py` | DPO + SFT models | `outputs/evaluation/` | Held-out preference accuracy `π_DPO` vs `π_ref`; judge-score distributions. |
| 8 | `08_export_seed.py` | DPO model | `outputs/seed/mariodpo/` | Arena-format levels + `generator.json` + content hashes for redeployment. |

### Config overrides

Both training scripts read YAML configs and accept inline overrides:

```bash
python scripts/03_sft.py --set num_train_epochs=2 learning_rate=3e-5
python scripts/05_dpo.py --set beta=0.2 use_lora=true per_device_train_batch_size=1
```

---

## 5. Method summary

**Representation.** Arena levels are 16 rows × 150–250 columns over a 37-tile
alphabet. We serialise **column-major** (one 16-char column per line) so
vertically adjacent tiles stay adjacent in the token stream. The GPT-2 BPE vocab
already covers every tile character, so no vocabulary surgery is needed. The
pretrained MarioGPT checkpoint is natively **14 rows** and contains an `x` A*
trace token; the converter maps `x → -` (air) and pads **+2 air rows on top** to
reach 16 (see `src/mariodpo/level_io.py`).

**Judge Function.** A Bradley-Terry style linear scorer over standardised static
features, `score(level) = w · z(f)`, trained on antisymmetric feature
differences from human votes. It is selected from four candidates (interpretable
heuristic, linear BT, gradient boosting, MLP) by **held-out pairwise AUC** on a
90/10 split **by vote**. The per-level score ranks generators in close agreement
with their empirical arena win rate — with **no identity leakage**.

**DPO data.** Each non-tie vote becomes a `(chosen, rejected)` pair sharing a
constant style prompt; human pairs are physically duplicated to oversample them
relative to judge-labelled synthetic pairs (from on-policy MarioGPT candidates
and reused seed levels paired by judge-score margin).

**Training.** Continue-pretrain (`03_sft.py`) → DPO (`05_dpo.py`) initialised from
the SFT model as both policy and reference.

**Evaluation.** Because new human votes cannot be collected offline, quality is
substantiated by (a) held-out human-pair preference accuracy of `π_DPO` vs
`π_ref`, and (b) the validated judge's score distribution of generated levels vs
baselines. A final human-SOTA claim is deferred to **redeployment** on the arena
(stage 8 produces the seed bundle for exactly that).

---

## 6. Hardware notes (NVIDIA GTX 2060, 6 GB)

- **Precision:** use `fp16: true` (configs already set this). Turing GPUs do not
  support `bf16`; the DPO script forces `bf16=False`. fp16 auto-disables on CPU.
- **Memory:** enable LoRA for DPO (`use_lora: true` in `configs/dpo.yaml`) so only
  adapter weights train and a single copy of the base weights is held (the
  reference is the base model with the adapter disabled). Keep
  `per_device_train_batch_size` at 1–2 and raise `gradient_accumulation_steps`.
- **Context:** the SFT window is 50 columns (≈800 tokens), comfortably under
  GPT-2's 1024. Generation grows levels by sliding-window continuation.
- **Throughput:** if you hit OOM, lower `max_length`, `win_cols`, or batch size,
  and/or enable gradient checkpointing.

---

## 7. Testing

```bash
pytest -q          # round-trip serialisation, converters, validation
```

---

## 8. Reproducibility & data provenance

- Primary data: `eda/data_10_05_2026/pcg-arena-votes-2026-05-10.json`
  (1000 of 1192 records on disk — the export is paginated; stage 0 warns and uses
  what is present; re-export with `offset=1000` for the remainder).
- The levels of the `mariodpo` generator deployed on the arena are stored in
  `generated_levels/`; they are not part of the arena seed corpus
  (`db/seed/levels`), so stage 1 resolves vote level IDs of the form
  `mariodpo::<file>.txt` to the files there.
- Seeds are fixed (`--seed`, default 42) throughout.
