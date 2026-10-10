"""Phase 3 — continue-pretrain (SFT) MarioGPT in the arena representation.

Loads the pretrained MarioGPT checkpoint (or a tiny dummy model for smoke
testing), builds a column-major corpus from the seed levels, and trains a
next-token objective so the model speaks the arena 16x37 column language. The
result is ``pi_ref`` for DPO.

Usage:
    python scripts/03_sft.py                          # full run (needs GPU+HF)
    python scripts/03_sft.py --dummy --max-steps 2    # CPU smoke test
    python scripts/03_sft.py --set num_train_epochs=1 learning_rate=3e-5
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from mariodpo.config import load_config
from mariodpo.constants import PROJECT_DIR
from mariodpo.modeling import (
    build_sft_texts,
    count_tile_chars_covered,
    load_model,
    load_tokenizer,
    resolve_fp16,
)
from mariodpo.tracking import LocalLogger, resolve_report_to, setup_wandb_env
from mariodpo.utils import load_dotenv, set_seed, setup_logging

log = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--dummy", action="store_true",
                    help="tiny model + tiny data + offline; CPU smoke test")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimisation steps (dummy runs use 2)")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[],
                    help="config overrides key=value")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(PROJECT_DIR / args.config, args.overrides)
    set_seed(int(cfg.get("seed", 42)))
    setup_wandb_env()

    from transformers import Trainer, TrainingArguments
    from transformers import DataCollatorForLanguageModeling

    # --- Data ---------------------------------------------------------------
    tokenizer = load_tokenizer(cfg["base_model"], dummy=args.dummy)
    covered = count_tile_chars_covered(tokenizer)
    log.info("Tokenizer covers %d/37 arena tile characters", covered)

    texts = build_sft_texts(
        win_cols=int(cfg.get("win_cols", 50)),
        stride=cfg.get("stride"),
        upweight_original=int(cfg.get("upweight_original", 3)),
        limit_per_generator=2 if args.dummy else cfg.get("limit_per_generator"),
    )
    if args.dummy:
        texts = texts[:64]
    log.info("SFT corpus: %d column-window examples", len(texts))

    from mariodpo.modeling import tokenize_texts

    dataset = tokenize_texts(texts, tokenizer, max_length=int(cfg.get("max_length", 256)))

    # --- Model --------------------------------------------------------------
    model = load_model(cfg["base_model"], dummy=args.dummy)
    model.resize_token_embeddings(len(tokenizer))

    if cfg.get("use_lora") and not args.dummy:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
                          lora_dropout=0.05, target_modules=["c_attn"])
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    # --- Trainer ------------------------------------------------------------
    out_dir = str(PROJECT_DIR / cfg.get("output_dir", "models/sft"))
    max_steps = args.max_steps if args.max_steps is not None else (2 if args.dummy else -1)
    fp16 = resolve_fp16(cfg.get("fp16", False))

    targs = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 5e-5)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=2,
        fp16=fp16,
        report_to=resolve_report_to(dummy=args.dummy),
        run_name="sft-dummy" if args.dummy else "sft",
        seed=int(cfg.get("seed", 42)),
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=targs, train_dataset=dataset, data_collator=collator,
    )

    local = LocalLogger("sft-dummy" if args.dummy else "sft", config=dict(cfg))
    log.info("Starting SFT (max_steps=%s, fp16=%s, device=%s)",
             max_steps, fp16, "cuda" if not args.dummy and resolve_fp16(True) else "cpu")
    result = trainer.train()
    log.info("SFT done: %s", result.metrics)
    local.log(phase="sft", **result.metrics)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Saved SFT model -> %s", out_dir)


if __name__ == "__main__":
    main()
