#!/usr/bin/env bash
# metaworld button: 冻结 V-JEPA2 前端 × ST reducer 对比。
# 叙事: WCM/SpaceTime 本身是动作条件 JEPA; V-JEPA2 = 通用视频预测式 JEPA 前端(冻结),
#   对比 = 前端(自训 vs 通用视频预训练) × ST reducer 有/无。
# (VideoMAE 重构式对照已于 2026-11-06 按用户决定移除)
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CFG=configs/metaworld_button
OUT=/media/Data/user/WCM_outputs
DS=/home/user/code_1/WCM/dataset_trials/lerobot

for name in vjepa2 vjepa2_st; do
  bash run_scripts/train.sh "$CFG/mw_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_${name}_s3072_test"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_${name}_s3072_ood"
done

# 配对 bootstrap: V-JEPA2 有ST vs 无ST
for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/mw_eval_vjepa2_s3072_${split}" "$OUT/mw_eval_vjepa2_st_s3072_${split}" \
    --align common --seed 3072 \
    --output "$OUT/mw_compare_vjepa2_nost_vs_st_${split}.json"
done

echo "metaworld_button frontends pipeline done"
