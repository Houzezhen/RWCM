#!/usr/bin/env bash
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE=offline
export WCM_EXPECTED_WORLD_SIZE=1
"${PYTHON:-.venv/bin/python}" -u -m world_critic.train --config "$1"
