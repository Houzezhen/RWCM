#!/usr/bin/env bash
set -euo pipefail
PYTHON="/home/user/code_1/WCM/.venv/bin/python"
# Register-token 训练：与 2_run_train.sh 相同流程，仅换 config 与输出目录
GPUS=1
CUDA_VISIBLE_DEVICES=0
CONFIG="configs/train_1gpu_reg4.yaml"
DATASET_REPO_ID="data_toiletButton_merged_0814_with_return"
DATASET_ROOT="/home/user/code_1/WCM/data_toiletButton_merged_0814_with_return"
DATASET_REVISION=""
OUTPUT_DIR="outputs/wcm_toiletButton_reg4"
EPOCHS="30"
PER_DEVICE_BATCH_SIZE="8"
EVAL_BATCH_SIZE=""
NUM_WORKERS=""
PRECISION=""
RESUME=""
WANDB_MODE="offline"
ALLOW_CPU_SMOKE=0
# ----------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! [[ "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS must be a positive integer (got: $GPUS)" >&2
  exit 2
fi
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export WANDB_MODE
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WCM_DATASET_REPO_ID="$DATASET_REPO_ID"
export WCM_EXPECTED_WORLD_SIZE="$GPUS"
export WCM_OUTPUT_DIR="$OUTPUT_DIR"

set_optional_env() {
  local name="$1" value="$2"
  if [[ -n "$value" ]]; then
    export "$name=$value"
  else
    unset "$name" 2>/dev/null || true
  fi
}

set_optional_env WCM_DATASET_ROOT "$DATASET_ROOT"
set_optional_env WCM_DATASET_REVISION "$DATASET_REVISION"
set_optional_env WCM_EPOCHS "$EPOCHS"
set_optional_env WCM_PER_DEVICE_BATCH_SIZE "$PER_DEVICE_BATCH_SIZE"
set_optional_env WCM_EVAL_BATCH_SIZE "$EVAL_BATCH_SIZE"
set_optional_env WCM_NUM_WORKERS "$NUM_WORKERS"
set_optional_env WCM_PRECISION "$PRECISION"
set_optional_env WCM_RESUME "$RESUME"

if [[ "$ALLOW_CPU_SMOKE" == "1" ]]; then
  export WCM_ALLOW_CPU_DDP=1
  export WCM_FORCE_CPU=1
else
  unset WCM_ALLOW_CPU_DDP WCM_FORCE_CPU 2>/dev/null || true
  if [[ "$GPUS" -gt 1 ]]; then
    if ! "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.device_count() >= $GPUS else 1)"; then
      echo "GPUS=$GPUS requires at least $GPUS visible CUDA devices. Set ALLOW_CPU_SMOKE=1 only for a CPU smoke test." >&2
      exit 2
    fi
  fi
fi

if [[ "$GPUS" == "1" ]]; then
  "$PYTHON" -m world_critic.train --config "$CONFIG"
else
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$GPUS" \
    -m world_critic.train --config "$CONFIG"
fi
