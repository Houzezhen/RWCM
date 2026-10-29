#!/usr/bin/env bash
# Continue existing a training, start b/c concurrently, evaluate and summarize.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
BASE=/media/Data/user/WCM_outputs/module_replacement_20261027
OOD=/media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return
a_pid="$1"
trap 'echo "$?" > "$BASE/parallel_exit_code.txt"' EXIT

evaluate_variant() {
  local v="$1"
  bash run_scripts/evaluate.sh "$BASE/${v}_s42/deploy.pt" "$OOD" "$BASE/${v}_s42/eval_ood" \
    > "$BASE/${v}_s42/eval_ood.log" 2>&1
  date -Is > "$BASE/${v}_s42/completed_at.txt"
}

train_variant() {
  local v="$1"
  mkdir -p "$BASE/${v}_s42"
  bash run_scripts/train.sh "configs/module_replacement/${v}_s42.yaml" \
    > "$BASE/${v}_s42/train.log" 2>&1
  evaluate_variant "$v"
}

(
  tail --pid="$a_pid" -f /dev/null
  evaluate_variant replace_a_sampling
) &
a_job=$!
train_variant replace_b_encoding &
b_job=$!
train_variant replace_c_readout &
c_job=$!
wait "$a_job"
wait "$b_job"
wait "$c_job"
.venv/bin/python -m scripts.summarize_module_replacements --root "$BASE"
date -Is > "$BASE/completed_at.txt"
