"""Phase 5 — DPO alignment of the SFT (continue-pretrained) model.

Loads the SFT model as both policy and (implicit) reference, then optimises the
DPO objective on the preference dataset from Phase 4. LoRA is recommended on a
6 GB GTX 2060 (policy = base + adapter; reference = base with the adapter
disabled, so only one set of base weights is held in memory).

Usage:
    python scripts/05_dpo.py                                  # full run
    python scripts/05_dpo.py --dummy --max-steps 2            # CPU smoke test
    python scripts/05_dpo.py --set beta=0.2 learning_rate=5e-6
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from mariodpo.config import load_config
from mariodpo.constants import PROJECT_DIR
from mariodpo.modeling import load_model, load_tokenizer, resolve_fp16
from mariodpo.tracking import LocalLogger, resolve_report_to, setup_wandb_env
from mariodpo.utils import load_dotenv, set_seed, setup_logging

log = setup_logging()


def _load_pairs(path, dummy: bool):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")
    keep = ["prompt", "chosen", "rejected"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    if dummy:
        ds = ds.select(range(min(16, len(ds))))
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(PROJECT_DIR / args.config, args.overrides)
    set_seed(int(cfg.get("seed", 42)))
    setup_wandb_env()

    from trl import DPOConfig, DPOTrainer

    # --- Model + tokenizer (from SFT) --------------------------------------
    sft_dir = PROJECT_DIR / cfg.get("sft_model", "models/sft")
    if args.dummy and not sft_dir.exists():
        # Build a tiny model on the fly if SFT wasn't run yet.
        tokenizer = load_tokenizer(dummy=True)
        model = load_model(dummy=True)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = load_tokenizer(str(sft_dir)) if _has_tok(sft_dir) \
            else load_tokenizer(dummy=args.dummy)
        model = load_model(checkpoint=str(sft_dir))

    pairs_path = PROJECT_DIR / cfg.get("pairs_file", "data/synthetic/preference_pairs.jsonl")
    dataset = _load_pairs(pairs_path, args.dummy)
    log.info("DPO dataset: %d preference records", len(dataset))

    # --- LoRA (recommended for the GTX 2060) -------------------------------
    peft_config = None
    use_lora = bool(cfg.get("use_lora", True)) and not args.dummy
    if use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=["c_attn"],
        )

    # --- DPO config --------------------------------------------------------
    out_dir = str(PROJECT_DIR / cfg.get("output_dir", "models/dpo"))
    max_steps = args.max_steps if args.max_steps is not None else (2 if args.dummy else -1)
    fp16 = resolve_fp16(cfg.get("fp16", False))

    dpo_args = DPOConfig(
        output_dir=out_dir,
        overwrite_output_dir=True,
        beta=float(cfg.get("beta", 0.1)),
        loss_type=str(cfg.get("loss_type", "sigmoid")),
        max_length=int(cfg.get("max_length", 256)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=2,
        fp16=fp16,
        bf16=False,  # GTX 2060 (Turing) lacks bf16; CPU dummy can't use it either
        remove_unused_columns=False,
        report_to=resolve_report_to(dummy=args.dummy),
        run_name="dpo-dummy" if args.dummy else "dpo",
        seed=int(cfg.get("seed", 42)),
    )

    trainer = _build_trainer(DPOTrainer, model, dpo_args, dataset, tokenizer, peft_config)

    local = LocalLogger("dpo-dummy" if args.dummy else "dpo", config=dict(cfg))
    log.info("Starting DPO (max_steps=%s, fp16=%s, lora=%s, beta=%s)",
             max_steps, fp16, use_lora, dpo_args.beta)
    result = trainer.train()
    log.info("DPO done: %s", result.metrics)
    local.log(phase="dpo", **result.metrics)

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Saved DPO model -> %s", out_dir)


def _has_tok(path) -> bool:
    from pathlib import Path

    return (Path(path) / "tokenizer_config.json").exists()


def _build_trainer(DPOTrainer, model, args, dataset, tokenizer, peft_config):
    """Construct DPOTrainer, tolerating the tokenizer/processing_class rename."""
    kwargs = dict(model=model, args=args, train_dataset=dataset,
                  peft_config=peft_config)
    try:
        return DPOTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **kwargs)


if __name__ == "__main__":
    main()
