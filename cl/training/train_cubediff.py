#!/usr/bin/env python
"""
CLI entry point that Accelerate will execute on every GPU rank.
It simply:
  • loads the YAML config,
  • instantiates CubeDiffTrainer,
  • calls trainer.train().
"""
import argparse, yaml, pathlib
from cl.training.trainer import CubeDiffTrainer           # <- the class you already have

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True,
                    help="Path to tiny_lora.yaml (or any YAML config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.cfg).read_text())
    print(f"train_cubediff.py - cfg is {cfg}")
    trainer = CubeDiffTrainer(
                config  = cfg,
                output_dir = cfg.get("output_dir", "outputs/cubediff_run"),
                mixed_precision = "bf16",
                gradient_accumulation_steps = cfg.get("gradient_accum_steps", 1))
    trainer.train()                       # ← generates samples & checkpoints

if __name__ == "__main__":
    main()

