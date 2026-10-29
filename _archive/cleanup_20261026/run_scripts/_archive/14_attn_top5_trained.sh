#!/usr/bin/env bash
# 训练后 top5 注意力分析：baseline vs D2（含多 seed）vs 单块版，与训练前结果对照
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

.venv/bin/python scripts/ablation_register_tokens_trained.py \
  --dataset-root /home/user/.cache/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return \
  --models \
    baseline=outputs/wcm_baseline_r2/checkpoints/best.pt \
    reg4_d2_s3072=outputs/wcm_reg4_r2/checkpoints/best.pt \
    reg4_d2_s42=outputs/wcm_reg4_r2_s42/checkpoints/best.pt \
    reg4_d2_s1337=outputs/wcm_reg4_r2_s1337/checkpoints/best.pt \
    bs1_s3072=outputs/wcm_blockreg8_bs1/checkpoints/best.pt \
  --num-batches 3 --batch-size 4 \
  --output-dir outputs/register_ablation_trained
