"""Model & tokenizer loading, and dataset construction for SFT / DPO.

Heavy ML imports (torch, transformers) are done lazily inside functions so the
judge/feature phases stay importable on a CPU box without the full stack.

The base policy is the pretrained MarioGPT checkpoint
``shyamsn97/Mario-GPT2-700-context-length`` (a GPT-2 LM). We *continue-pretrain*
it in the arena column-major representation, then DPO-align it. A ``--dummy``
path builds a tiny randomly-initialised GPT-2 so the pipeline can be smoke-tested
on CPU with no network/download.
"""

from __future__ import annotations

import os
from pathlib import Path

from .constants import ALLOWED_TILES
from .level_io import iter_seed_levels, load_level
from .tokenizer_io import iter_column_windows, serialize, style_prompt

BASE_MODEL = "shyamsn97/Mario-GPT2-700-context-length"
TOKENIZER_FALLBACK = "distilgpt2"


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_fp16(requested: bool) -> bool:
    """fp16 only works on CUDA; silently disable it on CPU (e.g. the dummy run)."""
    return bool(requested) and has_cuda()


def load_tokenizer(model_name: str = BASE_MODEL, dummy: bool = False):
    """Load the GPT-2 BPE tokenizer; ensure a pad token exists.

    The GPT-2 BPE vocab already covers every single ASCII tile character, so no
    vocabulary surgery is needed — we simply continue-pretrain on the new
    column-major strings.
    """
    from transformers import AutoTokenizer

    name = TOKENIZER_FALLBACK if dummy else model_name
    token = os.environ.get("HF_TOKEN")
    try:
        tok = AutoTokenizer.from_pretrained(name, token=token)
    except Exception:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_name: str = BASE_MODEL,
    dummy: bool = False,
    checkpoint: str | Path | None = None,
):
    """Load a causal-LM policy.

    Priority: explicit ``checkpoint`` dir > tiny ``dummy`` config > the
    pretrained MarioGPT ``model_name`` (requires network + optional ``HF_TOKEN``).
    """
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    # Force safetensors: transformers>=4.57 refuses to torch.load pickle (.bin)
    # checkpoints unless torch>=2.6 (CVE-2025-32434). Loading the safetensors
    # weights avoids the blocked code path entirely and is the secure choice
    # (safetensors cannot execute arbitrary code on load).
    if checkpoint is not None:
        return AutoModelForCausalLM.from_pretrained(
            str(checkpoint), use_safetensors=True
        )
    if dummy:
        # Tiny model: proves wiring on CPU without downloading ~350 MB.
        cfg = GPT2Config(
            vocab_size=50257, n_positions=256, n_ctx=256,
            n_embd=64, n_layer=2, n_head=2,
        )
        return GPT2LMHeadModel(cfg)
    token = os.environ.get("HF_TOKEN")
    return AutoModelForCausalLM.from_pretrained(
        model_name, token=token, use_safetensors=True
    )


# --- SFT dataset (continue-pretraining corpus) -----------------------------
def build_sft_texts(
    win_cols: int = 50,
    stride: int | None = None,
    generators: list[str] | None = None,
    upweight_original: int = 3,
    limit_per_generator: int | None = None,
    style: str | None = None,
) -> list[str]:
    """Build the column-major text corpus for continue-pretraining.

    Trains on all seed generators (broad tile/height coverage); ``original``
    windows are repeated ``upweight_original`` times to bias the prior toward
    the human-authored style, as in the thesis.
    """
    prefix = style_prompt(style)
    texts: list[str] = []
    per_gen: dict[str, int] = {}
    for gen, fname, path in iter_seed_levels(generators):
        if limit_per_generator is not None:
            if per_gen.get(gen, 0) >= limit_per_generator:
                continue
            per_gen[gen] = per_gen.get(gen, 0) + 1
        rows = load_level(path)
        windows = list(iter_column_windows(rows, win_cols, stride))
        reps = upweight_original if gen == "original" else 1
        for _ in range(reps):
            texts.extend(prefix + w for w in windows)
    return texts


def tokenize_texts(texts: list[str], tokenizer, max_length: int = 256):
    """Tokenise a list of strings into a HF ``Dataset`` for causal-LM training."""
    from datasets import Dataset

    ds = Dataset.from_dict({"text": texts})

    def _tok(batch):
        out = tokenizer(
            batch["text"], truncation=True, max_length=max_length,
            padding="max_length",
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    return ds.map(_tok, batched=True, remove_columns=["text"])


def count_tile_chars_covered(tokenizer) -> int:
    """Sanity helper: how many of the 37 arena tiles encode without <unk>."""
    covered = 0
    for ch in ALLOWED_TILES:
        ids = tokenizer.encode(ch)
        if ids and tokenizer.unk_token_id not in ids:
            covered += 1
    return covered


def sequence_logprob(model, tokenizer, prompt: str, completion: str,
                     max_length: int | None = None) -> float:
    """Average per-token log-prob of ``completion`` given ``prompt`` under model.

    Used to evaluate preference: a model "prefers" the level with the higher
    conditional log-prob. Length-normalised so wide/narrow levels compare fairly.
    The sequence is capped to the model's positional context to avoid index
    errors on small models.
    """
    import torch

    device = next(model.parameters()).device
    ctx = int(getattr(getattr(model, "config", None), "n_positions", 1024) or 1024)
    cap = min(max_length or ctx, ctx)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False,
                         truncation=True, max_length=cap)["input_ids"]
    if len(full_ids) <= len(prompt_ids) or len(full_ids) < 2:
        return float("-inf")
    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = input_ids[0, 1:]
    token_lp = log_probs[range(len(targets)), targets]
    # Only score the completion tokens (skip the prompt region).
    start = min(len(prompt_ids) - 1, len(token_lp) - 1)
    comp_lp = token_lp[start:]
    return float(comp_lp.mean().item())
