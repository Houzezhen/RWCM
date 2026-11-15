#!/usr/bin/env bash
# 真机 toilet-button(0814 expanded): metaworld v2 同款六点实验。
# 与论文/stmemory 报告同一数据协议: train=data_toiletButton_0814_expanded_with_return,
# OOD=data_toiletButton_0722_with_return(125集), test=train 的 val 切分(10%)。
# baseline / spacetime(scratch) / 冻结前端 2x2(VideoMAE, V-JEPA2 x ST有无)。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

CFG=configs/real_button
OUT=/media/Data/user/WCM_outputs
OOD=/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return

for name in baseline spacetime_scratch vjepa2 vjepa2_st; do
  bash run_scripts/train.sh "$CFG/real_button_${name}_s3072.yaml"
  # OOD (0722, 与论文一致)
  bash run_scripts/evaluate.sh "$OUT/real_button_${name}_s3072/deploy.pt" "$OOD" "$OUT/real_eval_${name}_s3072_ood"
  # test = train 的 val 切分 (数据集/切分来自 checkpoint 配置与 episode_split.json)
  "${PYTHON:-.venv/bin/python}" -u -m world_critic.evaluate \
    --checkpoint "$OUT/real_button_${name}_s3072/deploy.pt" \
    --output-dir "$OUT/real_eval_${name}_s3072_test" \
    --split val --batch-size 4 --num-workers 4 --expected-world-size 1 \
    --no-episode-curves --episode-metrics
done

for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/real_eval_baseline_s3072_${split}" "$OUT/real_eval_spacetime_scratch_s3072_${split}" \
    --align common --seed 3072 --output "$OUT/real_compare_baseline_vs_spacetime_${split}.json"
  for fe in videomae vjepa2; do
    .venv/bin/python scripts/compare_experiments.py \
      "$OUT/real_eval_${fe}_s3072_${split}" "$OUT/real_eval_${fe}_st_s3072_${split}" \
      --align common --seed 3072 --output "$OUT/real_compare_${fe}_nost_vs_st_${split}.json"
  done
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/real_eval_videomae_s3072_${split}" "$OUT/real_eval_vjepa2_s3072_${split}" \
    --align common --seed 3072 --output "$OUT/real_compare_videomae_vs_vjepa2_${split}.json"
done

echo "real_button pipeline done"
