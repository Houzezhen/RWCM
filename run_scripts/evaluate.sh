#!/usr/bin/env bash
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WCM_EXPECTED_WORLD_SIZE=1
"${PYTHON:-.venv/bin/python}" -u -m world_critic.evaluate \
  --checkpoint "$1" --dataset-root "$2" --output-dir "$3" \
  --split all --batch-size "${BATCH_SIZE:-4}" --num-workers 4 \
  --expected-world-size 1 --no-episode-curves --episode-metrics
