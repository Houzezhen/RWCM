#!/usr/bin/env bash
# Usage: bash run_scripts/module_ablation.sh no_temporal 42
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
variant="$1"
seed="$2"
base="${OUTPUT_ROOT:-/media/Data/user/WCM_outputs/module_ablation_20261026}"
run="$base/${variant}_s${seed}"
mkdir -p "$run"
bash run_scripts/train.sh "configs/module_ablation/${variant}_s${seed}_joint.yaml" 2>&1 | tee "$run/joint.log"
bash run_scripts/train.sh "configs/module_ablation/${variant}_s${seed}_full.yaml" 2>&1 | tee "$run/full.log"
# Full stage now contains the final model; joint artifacts are no longer needed.
rm -- "$run/joint/checkpoints/last.pt" "$run/joint/deploy.pt"
bash run_scripts/evaluate.sh "$run/full/deploy.pt" \
  /media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return "$run/eval_ood" \
  2>&1 | tee "$run/eval_ood.log"
date -Is > "$run/completed_at.txt"
