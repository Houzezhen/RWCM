#!/usr/bin/env bash
# metaworld button: LeJEPA(图像版, OK-AI lite_ssl 复刻 ViT-B/16 ep300) ± ST 前端对比。
# 官方 galilai-group/lejepa 只开源代码不放权重, 故用 OK-AI 复刻(用户 2026-11-18 确认)。
# 图像逐帧冻结前端, 与 LeVJEPA(视频版) 互补: 8 点表扩到 10 点。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CFG=configs/metaworld_button
OUT=/media/Data/user/WCM_outputs
DS=/home/user/code_1/WCM/dataset_trials/lerobot

for name in lejepa lejepa_st; do
  bash run_scripts/train.sh "$CFG/mw_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_${name}_s3072_test"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_${name}_s3072_ood"
done

# 配对 bootstrap: LeJEPA 有ST vs 无ST
for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/mw_eval_lejepa_s3072_${split}" "$OUT/mw_eval_lejepa_st_s3072_${split}" \
    --align common --seed 3072 \
    --output "$OUT/mw_compare_lejepa_nost_vs_st_${split}.json"
done

echo "lejepa pipeline done"
