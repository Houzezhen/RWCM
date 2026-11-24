#!/usr/bin/env bash
# v10 对称终点数据集上的 ST 基础模块消融(b/c 重设计版):
#   bmean = 时序注意力 -> 时间均值池化(保事件、丢顺序)
#   blast = 时序注意力 -> 只取当前帧(丢全部历史记忆)
#   cmean = 64-query perceiver -> 全 token 均值池化
# 对照 full(ST) = 已有 mw_button_spacetime_scratch_s3072(不重训)。
# 骨干 iJEPA 自训、seed 3072、协议与 mw_button 十点一致。
# 用 setsid/nohup 脱离会话运行。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

CFG=configs/metaworld_button
OUT=/media/Data/user/WCM_outputs
DS=/home/user/code_1/WCM/dataset_trials/lerobot

for name in st_bmean st_blast st_cmean; do
  bash run_scripts/train.sh "$CFG/mw_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_${name}_s3072_test"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_${name}_s3072_ood"
done

# 配对 bootstrap: 各消融 vs full(ST)
for split in test ood; do
  for name in st_bmean st_blast st_cmean; do
    .venv/bin/python scripts/compare_experiments.py \
      "$OUT/mw_eval_spacetime_scratch_s3072_${split}" "$OUT/mw_eval_${name}_s3072_${split}" \
      --align common --seed 3072 \
      --output "$OUT/mw_compare_full_vs_${name}_${split}.json"
  done
done

echo "st ablation pipeline done"
