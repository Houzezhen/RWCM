#!/usr/bin/env bash
# 评估扩训组也纳入 6_eval_5cut.sh 的模型清单（追加条目）
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
F=6_eval_5cut.sh
if ! grep -q "baseline_exp_s3072" "$F"; then
  sed -i 's#  \[leak10\]="outputs/wcm_blockreg8_leak10/checkpoints/best.pt"#  [leak10]="outputs/wcm_blockreg8_leak10/checkpoints/best.pt"\n  [baseline_exp_s3072]="outputs/wcm_baseline_exp_s3072/checkpoints/best.pt"\n  [reg4_exp_s3072]="outputs/wcm_reg4_exp_s3072/checkpoints/best.pt"\n  [bs1_exp_s3072]="outputs/wcm_bs1_exp_s3072/checkpoints/best.pt"\n  [baseline_exp_s42]="outputs/wcm_baseline_exp_s42/checkpoints/best.pt"\n  [reg4_exp_s42]="outputs/wcm_reg4_exp_s42/checkpoints/best.pt"\n  [bs1_exp_s42]="outputs/wcm_bs1_exp_s42/checkpoints/best.pt"#' "$F"
  echo "patched 6_eval_5cut.sh with expanded models"
else
  echo "already patched"
fi
