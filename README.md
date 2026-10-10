# MarioDPO: Preference-Aligned Super Mario Bros Level Generation

This repository holds the code, data, analysis, and thesis for

> **PCG Arena: Evaluating and Improving Super Mario Bros Level Generators Through Player Preferences**
> Antoni Czapski, Master's thesis, University of Wrocław (advisor: dr Jakub Kowalski), 2026.

It uses blind pairwise votes collected on the **PCG Arena** platform
([pcg-arena.com](https://pcg-arena.com), separate `pcg-arena` repository) to:

1. analyse which generators players prefer and why: rankings, player
   consistency, static expressive range, gameplay trajectories, and tags
   (`eda/`);
2. build a **judge function** that labels extra preference pairs;
3. fine-tune the MarioGPT checkpoint with **Direct Preference Optimization**
   into a preference-aligned level generator, MarioDPO (`MarioDPO/`).

---

## Repository layout

```
mario-dpo/
├── MarioDPO/               MarioDPO pipeline (installable package `mariodpo`)
│   ├── src/mariodpo/    level I/O, features, judge, pairs, modeling, generation, validation
│   ├── scripts/            numbered stages 00_prepare_data ... 08_export_seed
│   ├── configs/            sft.yaml, dpo.yaml
│   ├── data/               raw vote export + processed feature / score tables
│   ├── models/             judge.pkl + DPO LoRA adapter (SFT base weights not included)
│   ├── outputs/            judge / evaluation plots, generated levels, arena seed bundle
│   ├── generated_levels/   levels of the `mariodpo` generator deployed on the arena
│   ├── tests/
│   └── README.md           full pipeline documentation
├── eda/                    exploratory analysis of arena data
│   ├── data_10_05_2026/    May 2026 export (votes, level stats, trajectories, player profiles)
│   ├── new_data/           earlier exports
│   ├── 00_data_preparation/ ... 07_replanned_analysis/   analysis stages
│   └── 06_judge_function_experiments/                    judge-function derivation
├── db/seed/levels/         copy of the arena seed corpus (13 baseline generators); input data
├── latex/
│   ├── masters_thesis/     thesis source (main.tex, bibliography.bib, img/, iithesis.cls)
│   ├── main_report/        earlier course report
│   ├── literature_review/  literature notes and background reports
│   ├── verification/       citation-verification scripts and report
│   └── render_level.py     renders ASCII levels to images
└── presentation/           thesis presentation (Beamer)
```

---

## MarioDPO pipeline (quick start)

All details, including version pins and GPU notes, are in
[MarioDPO/README.md](MarioDPO/README.md).

```bash
cd MarioDPO

python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or /cpu
pip install -r requirements.txt
pip install -e .
cp .env.example .env    # HF_TOKEN, WANDB_API_KEY (optional; WANDB_MODE=disabled works)

bash scripts/run_dummy_pipeline.sh     # CPU smoke test of every stage

# real run
python scripts/00_prepare_data.py
python scripts/01_extract_features.py
python scripts/02_train_judge.py
python scripts/03_sft.py
python scripts/04_build_pairs.py --n-synthetic 2000
python scripts/05_dpo.py
python scripts/06_generate.py  --model models/dpo --n 100
python scripts/07_evaluate.py  --dpo models/dpo --ref models/sft
python scripts/08_export_seed.py --model models/dpo --n 100
```

| Stage | What it does |
|---|---|
| 0–1 | Load the May 2026 vote export and extract 34 static structural features per level |
| 2 | Train a judge on the human votes and choose the model with the best held-out pairwise AUC |
| 3 | Continue-pretrain MarioGPT (`shyamsn97/Mario-GPT2-700-context-length`) in the arena's 16-row, column-major format (π_ref) |
| 4 | Build preference pairs: human votes oversampled 10×, plus synthetic pairs labelled by the judge |
| 5 | DPO with LoRA (β = 0.1) using `trl` |
| 6–8 | Generate levels with judge rejection sampling, evaluate them, and export an arena seed bundle |

The SFT base weights (`models/sft/`, about 220 MB) are not committed. Rebuild
them with `scripts/03_sft.py`. The DPO LoRA adapter in `models/dpo/` is
applied on top of these weights.

---

## Data provenance

- **Votes**: `eda/data_10_05_2026/` (also copied to
  `MarioDPO/data/raw/`) contains the PCG Arena export from May 2026:
  1,000 vote records, 77 anonymous player IDs, and 14 generators. Player
  IDs are anonymous browser IDs.
- **Seed levels**: `db/seed/levels/` is a copy of the arena's seed corpus.
  The analysis scripts and `mariodpo.constants` read it from this path.
- **`MarioDPO/generated_levels/`**: the levels of the `mariodpo` generator
  deployed on the arena, produced by the DPO-trained model. They live
  outside `db/seed/levels/`, so the analysis and feature-extraction
  scripts resolve vote level IDs of the form
  `mariodpo::mariodpo_nintendo_XXXX.txt` to these files.

---

## Thesis

Build the thesis from `latex/masters_thesis/`:

```bash
cd latex/masters_thesis
latexmk -pdf main.tex
```

`main.pdf` is included. The thesis figures are in
`latex/masters_thesis/img/`. The scripts that produce them are in
`eda/07_replanned_analysis/` and `latex/masters_thesis/scripts/`.

## Related

- **PCG Arena** (the platform that collected the votes): separate
  `pcg-arena` repository; live at [pcg-arena.com](https://pcg-arena.com).
- **MarioGPT**: Sudhakaran et al., 2023, the base checkpoint.
- **DPO**: Rafailov et al., 2023.

## License

MIT. See [LICENSE](LICENSE).
