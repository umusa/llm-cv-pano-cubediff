#!/usr/bin/env bash
# --------  CubeDiff tiny-LoRA launcher  --------------------

CONFIG_FILE="../config/tiny_lora.yaml"      # <-- edit path if you rename
ACCEL_FILE="../config/accelerate_l4.yaml"   # generated once below

# Generate accelerate config only the first time
if [ ! -f "$ACCEL_FILE" ]; then
  accelerate config default \
      --config_file "$ACCEL_FILE"           \
      --num_processes 4                     \
      --mixed_precision bf16
fi

# Launch 4-GPU job
accelerate launch --config_file "$ACCEL_FILE" \
  cl/training/train_cubediff.py --cfg "$CONFIG_FILE"
