"""Experiment tracking: Weights & Biases (optional) + always-on local JSON logs.

Tokens are read from the environment (``WANDB_API_KEY``, ``WANDB_PROJECT``,
``WANDB_MODE``) — never hard-coded. If W&B is unavailable or disabled, training
still runs and metrics are written locally under ``outputs/runs/<run_name>/``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .constants import OUTPUTS_DIR


def resolve_report_to(dummy: bool = False) -> list[str]:
    """Return the HF ``report_to`` list based on env and availability."""
    mode = os.environ.get("WANDB_MODE", "online").lower()
    if dummy or mode == "disabled" or not os.environ.get("WANDB_API_KEY"):
        # No key or explicitly disabled -> don't try to use W&B.
        os.environ.setdefault("WANDB_MODE", "disabled")
        return []
    try:
        import wandb  # noqa: F401
    except Exception:
        return []
    return ["wandb"]


class LocalLogger:
    """Append-only JSONL metric logger under ``outputs/runs/<run_name>/``."""

    def __init__(self, run_name: str, config: dict | None = None):
        self.dir = OUTPUTS_DIR / "runs" / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        if config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2))

    def log(self, **kv) -> None:
        kv["_t"] = time.time()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(kv) + "\n")


def setup_wandb_env() -> None:
    """Push W&B settings from the environment so the Trainer picks them up."""
    project = os.environ.get("WANDB_PROJECT", "pcg-mariodpo")
    os.environ["WANDB_PROJECT"] = project
