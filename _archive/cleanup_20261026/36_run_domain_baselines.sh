#!/usr/bin/env bash
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT="${OUT:-outputs/domain_baselines_20261022}"
OURS="${OURS:-/media/Data/user/WCM_outputs/wcm_spacetime_vit_t8_direct_full_t120_s42}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/user/.cache/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return}"
TEST_ROOT="${TEST_ROOT:-/media/Data/user/huggingface/lerobot/test/data_toiletButton_0722_with_return}"
OURS_EVAL="${OURS_EVAL:-outputs/eval_spacetime_t120_pair/ood/t120_s42}"
STEPS="${STEPS:-1000}"
SEEDS="${SEEDS:-42}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
date -Is > "$OUT/started_at.txt"
"$PYTHON" -m scripts.external_baselines.domain fetch --output "$OUT"
"$PYTHON" -m scripts.external_baselines.domain smoke --output "$OUT"
"$PYTHON" -m scripts.external_baselines.domain cache --output "$OUT" \
  --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" \
  --config "$OURS/resolved_config.json" --split-manifest "$OURS/episode_split.json"
for seed in $SEEDS; do
  for method in tdmpc2 vip iql; do
    "$PYTHON" -m scripts.external_baselines.domain train --output "$OUT" \
      --model "$method" --seed "$seed" --steps "$STEPS"
  done
done
"$PYTHON" -m scripts.external_baselines.domain summarize --output "$OUT" \
  --ours-eval "$OURS_EVAL" --seeds $SEEDS
date -Is > "$OUT/completed_at.txt"
