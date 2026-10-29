#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
BASE=/media/Data/user/WCM_outputs/replacement_followup_20261027
OOD=/media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return
mkdir -p "$BASE"
trap 'echo "$?" > "$BASE/exit_code.txt"' EXIT
wait_gpu() {
  local free
  free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
  while [ "$free" -lt 24000 ]; do
    echo "$(date -Is) waiting for 24GB free on GPU1; free=${free}MB"
    sleep 30
    free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
  done
}
for v in replace_a_sampling replace_b_encoding replace_c_readout queries16; do
  wait_gpu
  mkdir -p "$BASE/${v}_s42"
  cfg="configs/replacement_followup/${v}_s42_full.yaml"
  .venv/bin/python -m scripts.smoke_test_module_replacements --config "$cfg"
  if [ "$v" = queries16 ]; then
    bash run_scripts/train.sh "configs/replacement_followup/${v}_s42_joint.yaml" \
      > "$BASE/${v}_s42/joint.log" 2>&1
  fi
  bash run_scripts/train.sh "$cfg" > "$BASE/${v}_s42/full.log" 2>&1
  bash run_scripts/evaluate.sh "$BASE/${v}_s42/full/deploy.pt" "$OOD" "$BASE/${v}_s42/eval_ood" \
    > "$BASE/${v}_s42/eval_ood.log" 2>&1
  if [ "$v" = queries16 ]; then
    rm -- "$BASE/${v}_s42/joint/checkpoints/last.pt" "$BASE/${v}_s42/joint/deploy.pt"
  fi
  date -Is > "$BASE/${v}_s42/completed_at.txt"
done
.venv/bin/python -m scripts.summarize_replacement_followup --root "$BASE"
date -Is > "$BASE/completed_at.txt"
