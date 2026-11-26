#!/usr/bin/env bash
# v10 对称终点: c-query1 消融(perceiver 64 queries -> 单 query cross-attention)。
# 对照 full(ST) = 已有 mw_button_spacetime_scratch_s3072(不重训)。
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

bash run_scripts/train.sh "$CFG/mw_button_st_query1_s3072.yaml"
bash run_scripts/evaluate.sh "$OUT/mw_button_st_query1_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_st_query1_s3072_test"
bash run_scripts/evaluate.sh "$OUT/mw_button_st_query1_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_st_query1_s3072_ood"

for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/mw_eval_spacetime_scratch_s3072_${split}" "$OUT/mw_eval_st_query1_s3072_${split}" \
    --align common --seed 3072 \
    --output "$OUT/mw_compare_full_vs_st_query1_${split}.json"
done

echo "st query1 pipeline done"
