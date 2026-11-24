#!/usr/bin/env bash
# lejepa_st 断点续训(会话退出被杀后从 last.pt 恢复)+ 剩余评估与 bootstrap。
# 用 setsid/nohup 脱离会话运行, 见启动命令。
set -eu
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE=offline
export WCM_EXPECTED_WORLD_SIZE=1

OUT=/media/Data/user/WCM_outputs
DS=/home/user/code_1/WCM/dataset_trials/lerobot

WCM_RESUME="$OUT/mw_button_lejepa_st_s3072/checkpoints/last.pt" \
  bash run_scripts/train.sh configs/metaworld_button/mw_button_lejepa_st_s3072.yaml

bash run_scripts/evaluate.sh "$OUT/mw_button_lejepa_st_s3072/deploy.pt" "$DS/metaworld_button_v2_test" "$OUT/mw_eval_lejepa_st_s3072_test"
bash run_scripts/evaluate.sh "$OUT/mw_button_lejepa_st_s3072/deploy.pt" "$DS/metaworld_button_v2_ood" "$OUT/mw_eval_lejepa_st_s3072_ood"

for split in test ood; do
  .venv/bin/python scripts/compare_experiments.py \
    "$OUT/mw_eval_lejepa_s3072_${split}" "$OUT/mw_eval_lejepa_st_s3072_${split}" \
    --align common --seed 3072 \
    --output "$OUT/mw_compare_lejepa_nost_vs_st_${split}.json"
done

echo "lejepa resume pipeline done"
