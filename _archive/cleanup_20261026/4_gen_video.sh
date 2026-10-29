PYTHON="/home/user/code_1/WCM/.venv/bin/python"
DATASET_REPO_ID="data_toiletButton_merged_0814_with_return"   # empty = use the checkpoint's dataset id
DATASET_ROOT="/home/user/code_1/WCM/data_toiletButton_merged_0814_with_return"
VISION_MODEL_NAME="google/vit-base-patch16-224-in21k"  # Local path / hf name supported
LANGUAGE_MODEL_NAME="openai/clip-vit-base-patch32"   # Local path / hf name supported

export WCM_DATASET_ROOT="$DATASET_ROOT"
export WCM_DATASET_REPO_ID="$DATASET_REPO_ID"
export WCM_VISION_MODEL_NAME="$VISION_MODEL_NAME"
export WCM_LANGUAGE_MODEL_NAME="$LANGUAGE_MODEL_NAME"
# huggingface.co 直连会被重置；走镜像站（已设置 HF_ENDPOINT 时尊重原值）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

"$PYTHON" -m episode_value_video pipeline \
  --checkpoint outputs/wcm_toiletButton_merged_0814/checkpoints/best.pt \
  --eval-output-dir outputs/wcm_toiletButton_merged_0814/eval_videos \
  --split all \
  --speed 2.0 \
  --overwrite \
  --output-dir outputs/wcm_toiletButton_merged_0814/eval_videos/episode_value_videos
