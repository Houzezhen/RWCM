#!/usr/bin/env bash
# 真机 toilet-button: LeJEPA(图像版, OK-AI 复刻) ± ST, 真机表 8 点扩 10 点。
# 协议与 real_button.sh 一致: train=0814_expanded, OOD=0722(125集), test=val 切分。
# 用 setsid/nohup 脱离会话运行。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

CFG=configs/real_button
OUT=/media/Data/user/WCM_outputs
OOD=/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return

for name in lejepa lejepa_st; do
  bash run_scripts/train.sh "$CFG/real_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/real_button_${name}_s3072/deploy.pt" "$OOD" "$OUT/real_eval_${name}_s3072_ood"
  "${PYTHON:-.venv/bin/python}" -u -m world_critic.evaluate \
    --checkpoint "$OUT/real_button_${name}_s3072/deploy.pt" \
    --output-dir "$OUT/real_eval_${name}_s3072_test" \
    --split val --batch-size 4 --num-workers 4 --expected-world-size 1 \
    --no-episode-curves --episode-metrics
done

for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/real_eval_lejepa_s3072_${split}" "$OUT/real_eval_lejepa_st_s3072_${split}" \
    --align common --seed 3072 --output "$OUT/real_compare_lejepa_nost_vs_st_${split}.json"
done

echo "real_button lejepa pipeline done"
