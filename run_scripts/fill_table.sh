#!/usr/bin/env bash
# 补齐八点表: 仿真v8四点 + 真机前端四点, 训练+评估+配对bootstrap。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

OUT=/media/Data/user/WCM_outputs
MW_DS=/home/user/code_1/WCM/dataset_trials/lerobot
REAL_OOD=/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return

# --- 仿真 v8 四点 ---
for name in baseline spacetime_scratch vjepa2 vjepa2_st; do
  bash run_scripts/train.sh "configs/metaworld_button/mw_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$MW_DS/metaworld_button_v2_test" "$OUT/mw_eval_${name}_s3072_test"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$MW_DS/metaworld_button_v2_ood" "$OUT/mw_eval_${name}_s3072_ood"
done

.venv/bin/python scripts/compare_experiments.py \
  "$OUT/mw_eval_baseline_s3072_test" "$OUT/mw_eval_spacetime_scratch_s3072_test" \
  --align common --seed 3072 --output "$OUT/mw_compare_baseline_vs_spacetime_test.json"
.venv/bin/python scripts/compare_experiments.py \
  "$OUT/mw_eval_baseline_s3072_ood" "$OUT/mw_eval_spacetime_scratch_s3072_ood" \
  --align common --seed 3072 --output "$OUT/mw_compare_baseline_vs_spacetime_ood.json"
for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/mw_eval_vjepa2_s3072_${split}" "$OUT/mw_eval_vjepa2_st_s3072_${split}" \
    --align common --seed 3072 --output "$OUT/mw_compare_vjepa2_nost_vs_st_${split}.json"
done

# --- 真机前端四点 ---
for name in vjepa21 vjepa21_st levjepa levjepa_st; do
  bash run_scripts/train.sh "configs/real_button/real_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/real_button_${name}_s3072/deploy.pt" "$REAL_OOD" "$OUT/real_eval_${name}_s3072_ood"
  "${PYTHON:-.venv/bin/python}" -u -m world_critic.evaluate \
    --checkpoint "$OUT/real_button_${name}_s3072/deploy.pt" \
    --output-dir "$OUT/real_eval_${name}_s3072_test" \
    --split val --batch-size 4 --num-workers 4 --expected-world-size 1 \
    --no-episode-curves --episode-metrics
done

for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/real_eval_vjepa21_s3072_${split}" "$OUT/real_eval_vjepa21_st_s3072_${split}" \
    --align common --seed 3072 --output "$OUT/real_compare_vjepa21_nost_vs_st_${split}.json"
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/real_eval_levjepa_s3072_${split}" "$OUT/real_eval_levjepa_st_s3072_${split}" \
    --align common --seed 3072 --output "$OUT/real_compare_levjepa_nost_vs_st_${split}.json"
done

echo "fill_table done"
