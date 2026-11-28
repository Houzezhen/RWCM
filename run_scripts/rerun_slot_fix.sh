#!/usr/bin/env bash
# 2026-11-28 槽位 bug 修复后重跑 LeVJEPA/LeJEPA 无 ST(仿真+真机共 4 个)。
# bug: 无 ST 前端曾误读最旧历史帧(model.py [:, -1]), 修正为当前帧([:, 0])。
# 旧结果归档为 *_slotlast; vjepa2/vjepa21 无 ST 按用户决定不重跑。
# 用 setsid/nohup 脱离会话运行。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WCM_EXPECTED_WORLD_SIZE=1

OUT=/media/Data/user/WCM_outputs
DS=/home/user/code_1/WCM/dataset_trials/lerobot
OOD_REAL=/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0722_with_return

# --- 仿真 ---
for name in levjepa lejepa; do
  bash run_scripts/train.sh "configs/metaworld_button/mw_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_${name}_s3072_test"
  bash run_scripts/evaluate.sh "$OUT/mw_button_${name}_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_${name}_s3072_ood"
done
for split in test ood; do
  for name in levjepa lejepa; do
    .venv/bin/python scripts/compare_experiments.py \
      "$OUT/mw_eval_${name}_s3072_${split}" "$OUT/mw_eval_${name}_st_s3072_${split}" \
      --align common --seed 3072 \
      --output "$OUT/mw_compare_${name}_nost_vs_st_${split}.json"
  done
done

# --- 真机 ---
for name in levjepa lejepa; do
  bash run_scripts/train.sh "configs/real_button/real_button_${name}_s3072.yaml"
  bash run_scripts/evaluate.sh "$OUT/real_button_${name}_s3072/deploy.pt" "$OOD_REAL" "$OUT/real_eval_${name}_s3072_ood"
  "${PYTHON:-.venv/bin/python}" -u -m world_critic.evaluate \
    --checkpoint "$OUT/real_button_${name}_s3072/deploy.pt" \
    --output-dir "$OUT/real_eval_${name}_s3072_test" \
    --split val --batch-size 4 --num-workers 4 --expected-world-size 1 \
    --no-episode-curves --episode-metrics
done
for split in test ood; do
  for name in levjepa lejepa; do
    .venv/bin/python scripts/compare_experiments.py \
      "$OUT/real_eval_${name}_s3072_${split}" "$OUT/real_eval_${name}_st_s3072_${split}" \
      --align common --seed 3072 \
      --output "$OUT/real_compare_${name}_nost_vs_st_${split}.json"
  done
done

echo "slot-fix rerun pipeline done"
