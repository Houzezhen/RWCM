#!/usr/bin/env bash
set -euo pipefail
# WCM 的 uv 环境装有 lerobot 0.5.1（脚本要求 >=0.5.1,<0.6），
# base conda 环境是 piper_lerobot 0.4.2，缺少 recompute_stats。
PYTHON="/home/user/code_1/WCM/.venv/bin/python"

"$PYTHON" scripts/add_returns_to_lerobot.py \
  --repo-id data_toiletButton_0722_2 \
  --root /media/Data/user/huggingface/lerobot/lerobotevo \
  --output-dir /media/Data/user/huggingface/lerobot/lerobotevo/data_toiletButton_0722_2_with_return_taskmax \
  --failure-penalty 300 \
  --normalization task_max \
  --success-labels /media/Data/user/huggingface/lerobot/lerobotevo/data_toiletButton_0722_2/success_labels.json
