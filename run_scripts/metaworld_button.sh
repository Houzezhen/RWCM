#!/usr/bin/env bash
# metaworld button v2: 原始 WCM vs SpaceTime(scratch) 对照管线。
# 训练: baseline(从头) -> spacetime_scratch(从头)  [full 收尾阶段按需手动]
# 之后在 v2 test/ood 上评估并做 episode 级配对 bootstrap。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CFG=configs/metaworld_button
OUT=/media/Data/user/WCM_outputs
DS=/home/user/code_1/WCM/dataset_trials/lerobot

bash run_scripts/train.sh "$CFG/mw_button_baseline_s3072.yaml"
bash run_scripts/train.sh "$CFG/mw_button_spacetime_scratch_s3072.yaml"

for m in baseline spacetime_scratch; do
  bash run_scripts/evaluate.sh "$OUT/mw_button_${m}_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_${m}_s3072_test"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${m}_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_${m}_s3072_ood"
done

for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/mw_eval_baseline_s3072_${split}" "$OUT/mw_eval_spacetime_scratch_s3072_${split}" \
    --align common --seed 3072 \
    --output "$OUT/mw_compare_baseline_vs_spacetime_${split}.json"
done

echo "metaworld_button v2 pipeline done"
