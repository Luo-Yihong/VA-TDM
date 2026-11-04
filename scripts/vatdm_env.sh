#!/usr/bin/env bash
# Source this file before running the single-node experiments.
VATDM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VATDM_WORKSPACE="$(dirname "$VATDM_ROOT")"
export VENV_PATH="${VENV_PATH:-$VATDM_WORKSPACE/.venvs/vatdm}"
source "$VENV_PATH/bin/activate"
if [[ -f "$VATDM_WORKSPACE/.config/vatdm/secrets.env" ]]; then
    source "$VATDM_WORKSPACE/.config/vatdm/secrets.env"
fi
export VATDM_MODEL_ROOT="${VATDM_MODEL_ROOT:-$VATDM_WORKSPACE/models/vatdm}"
export HF_HOME="${HF_HOME:-$VATDM_WORKSPACE/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_HOME="${VATDM_CUDA_HOME:-/cm/shared/apps/cuda12.4/toolkit/12.4.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MPLBACKEND=Agg
export VATDM_DEPS_PATH="${VATDM_DEPS_PATH:-$VATDM_WORKSPACE/.venvs/vatdm_python_deps}"
if [[ ! -d "$VATDM_DEPS_PATH/transformers" ]]; then
    echo "Missing VATDM dependency overlay: $VATDM_DEPS_PATH" >&2
    return 1
fi
export PYTHONPATH="$VATDM_ROOT:$VATDM_DEPS_PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$VATDM_ROOT"
