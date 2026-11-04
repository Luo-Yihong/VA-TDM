#!/bin/bash
# Distill SD3.5-Medium into a 4-step student LoRA with VA-TDM on PickScore prompts.
# Single-node, 8-GPU recipe (config: pickscore_tdm2_8gpu_G64).
set -euo pipefail
source scripts/vatdm_env.sh

CMD_VATDM="torchrun --standalone --nproc_per_node=8 scripts/train_vatdm_lora_git.py"

$CMD_VATDM --config config/vatdm.py:pickscore_tdm2_8gpu_G64 \
    --config.mixed_precision=fp16 \
    --config.num_epochs=601 \
    --config.save_freq=601 \
    --config.save_dir=logs/vatdm \
    "$@"
