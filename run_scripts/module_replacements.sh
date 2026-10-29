#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
PYTHON=.venv/bin/python
BASE=/media/Data/user/WCM_outputs/module_replacement_20261027
OOD=/media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return
mkdir -p "$BASE"
date -Is > "$BASE/started_at.txt"
for v in replace_a_sampling replace_b_encoding replace_c_readout; do
  "$PYTHON" -m scripts.smoke_test_module_replacements --config "configs/module_replacement/${v}_s42.yaml"
done
# Reuse the paper's existing final H120 predictions; no reference training/eval.
mkdir -p "$BASE/reference_paper_s42/episode_curves"
cp outputs/eval_spacetime_t120_pair/ood/t120_s42/summary.json "$BASE/reference_paper_s42/summary.json"
cp outputs/eval_spacetime_t120_pair/ood/t120_s42/episode_curves/episode_curves.csv \
  "$BASE/reference_paper_s42/episode_curves/episode_curves.csv"
for v in replace_a_sampling replace_b_encoding replace_c_readout; do
  mkdir -p "$BASE/${v}_s42"
  bash run_scripts/train.sh "configs/module_replacement/${v}_s42.yaml" 2>&1 | tee "$BASE/${v}_s42/train.log"
  bash run_scripts/evaluate.sh "$BASE/${v}_s42/deploy.pt" "$OOD" "$BASE/${v}_s42/eval_ood" \
    2>&1 | tee "$BASE/${v}_s42/eval_ood.log"
  date -Is > "$BASE/${v}_s42/completed_at.txt"
done
"$PYTHON" -m scripts.summarize_module_replacements --root "$BASE"
date -Is > "$BASE/completed_at.txt"
