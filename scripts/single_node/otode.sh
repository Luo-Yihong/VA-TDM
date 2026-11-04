#!/bin/bash
# OT-ODE warm-up for the VA-TDM student LoRA on PickScore prompts.
# Single-node, 8-GPU recipe (config: config/otode.py). The saved ema.ckpt can
# initialize VA-TDM via --config.train.lora_path.
set -euo pipefail
source scripts/vatdm_env.sh

CMD_OTODE="torchrun --standalone --nproc_per_node=8 scripts/train_otode_lora.py"

$CMD_OTODE --config config/otode.py \
    --config.otode.teacher_solver=euler \
    --config.otode.teacher_substeps=6 \
    --config.save_dir=logs/otode \
    "$@"
