#!/usr/bin/env python3
"""Run one stable on-policy self-distillation experiment cell."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from firesd.config import apply_overrides, config_argument_parser, load_config, validate_config
from firesd.data import get_adapter, load_examples
from firesd.ema import LoRAEMA
from firesd.modeling import load_tokenizer_and_model
from firesd.trainer import StableSDTrainer
from firesd.utils import copy_file, ensure_dir, get_logger, seed_everything, write_json


def main() -> None:
    parser = config_argument_parser("Train one stable self-distillation experiment cell.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs/data setup without loading a model.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip any nonempty run directory, preserving its artifacts.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    validate_config(cfg)
    logger = get_logger("stablesd.train")

    seed = int(cfg["train"]["seed"])
    seed_everything(seed)
    cell = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cfg["experiment"]["name"]))
    run_path = Path(cfg["output"]["root"]) / cell / f"seed{seed}"
    if args.skip_existing and run_path.exists() and any(run_path.iterdir()):
        logger.info("Skipping existing run: %s (use a fresh output.root for a new run)", run_path)
        return
    run_dir = ensure_dir(run_path)
    write_json(run_dir / "resolved_config.json", cfg)
    copy_file(args.config, run_dir / "source_config.yaml")

    adapter = get_adapter(str(cfg["data"]["adapter"]))
    train_examples = load_examples(cfg["data"], "train", seed)
    eval_examples = load_examples(cfg["data"], "eval", seed + 10_000)
    logger.info("Loaded %d train and %d eval examples from %s", len(train_examples), len(eval_examples), cfg["data"]["adapter"])
    if args.dry_run:
        write_json(run_dir / "summary.json", {"status": "dry-run", "config": cfg, "train_examples": len(train_examples), "eval_examples": len(eval_examples)})
        logger.info("Dry run complete: %s", run_dir)
        return

    requested = str(cfg["runtime"].get("device", "cuda"))
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    if requested == "cuda" and device.type != "cuda":
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
    tokenizer, model = load_tokenizer_and_model(cfg["model"], cfg["lora"], device)
    ema = LoRAEMA(model, alpha=float(cfg["ema"]["alpha"]))
    trainer = StableSDTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        ema=ema,
        adapter=adapter,
        train_examples=train_examples,
        eval_examples=eval_examples,
        run_dir=run_dir,
    )
    result = trainer.train()
    write_json(run_dir / "summary.json", {"config": cfg, **result})
    logger.info("Completed run: %s", run_dir)


if __name__ == "__main__":
    main()
