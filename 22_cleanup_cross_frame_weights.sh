#!/usr/bin/env bash
# Clean obsolete training weights after the Sparse4 ablation.
# Default: dry-run. Pass --apply to delete. Evaluation JSON/logs are untouched.
set -euo pipefail

BIG_ROOT="${WCM_OUTPUT_ROOT:-/media/Data/user/WCM_outputs}"
MODE=dry-run
if [[ "${1:-}" == "--apply" ]]; then
  MODE=apply
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--apply]" >&2
  exit 2
fi

[[ -d "$BIG_ROOT" ]] || { echo "[abort] output root does not exist: $BIG_ROOT" >&2; exit 1; }
ROOT_REAL=$(realpath "$BIG_ROOT")
[[ "$ROOT_REAL" != "/" && "$ROOT_REAL" != "/home" && "$ROOT_REAL" != "/media" ]] || {
  echo "[abort] refusing unsafe output root: $ROOT_REAL" >&2
  exit 1
}

# Main paper comparison: retain deploy and resume checkpoints.
KEEP_FULL=(
  wcm_baseline_exp_s3072
  wcm_baseline_exp_s42
  wcm_sparse4_image_exp_s3072
  wcm_sparse4_image_exp_s42
)

# Completed ablations: deploy.pt is sufficient for evaluation/reproduction of
# reported predictions. Remove optimizer-heavy full-resume checkpoints.
KEEP_DEPLOY_ONLY=(
  wcm_sparse4_exp_s3072
  wcm_sparse4_exp_s42
  wcm_sparse4_state_exp_s3072
  wcm_sparse4_state_exp_s42
)

# Rejected or superseded experiments. Names are exact: the script never scans
# arbitrary directories and decides to remove them by pattern.
DELETE_DIRS=(
  wcm_rank_exp_s3072
  wcm_rank_exp_s42
  wcm_reg4_early_exp_s3072
  wcm_reg4_early_exp_s42
  wcm_reg4_exp_s3072
  wcm_reg4_exp_s42
  wcm_reg4_r2
  wcm_reg4_r2_s42
  wcm_reg4_r2_s1337
  wcm_reg4_0814_bf16
  wcm_temporal_reg4_r1
  wcm_s4t4_r1
  wcm_blockreg8_r1
  wcm_blockreg8_bs1
  wcm_blockreg8_bs1_s42
  wcm_blockreg8_bs1_s1337
  wcm_bs1_exp_s3072
  wcm_bs1_exp_s42
  wcm_blockreg8_noleak
  wcm_blockreg8_noleak_s42
  wcm_blockreg8_noleak_s1337
  wcm_blockreg8_leak1
  wcm_blockreg8_leak10
  wcm_mix_c3
  wcm_mix_d3
)

checked_dir() {
  local name=$1
  local target="$ROOT_REAL/$name"
  local parent
  parent=$(realpath -m "$(dirname -- "$target")")
  [[ "$parent" == "$ROOT_REAL" && "$name" == wcm_* ]] || {
    echo "[abort] unsafe target: $target" >&2
    exit 1
  }
  printf '%s\n' "$target"
}

bytes_of() {
  du -sb -- "$1" 2>/dev/null | cut -f1 || printf '0\n'
}

remove_path() {
  local target=$1
  if [[ "$MODE" == apply ]]; then
    rm -rf -- "$target"
    echo "[deleted] $target"
  else
    echo "[would delete] $target"
  fi
}

echo "mode=$MODE"
echo "output_root=$ROOT_REAL"
echo
echo "== Keep full training state =="
for name in "${KEEP_FULL[@]}"; do
  target=$(checked_dir "$name")
  [[ -d "$target" ]] && echo "[keep full] $target" || echo "[missing] $target"
done

echo
echo "== Keep deploy.pt only for completed ablations =="
for name in "${KEEP_DEPLOY_ONLY[@]}"; do
  target=$(checked_dir "$name")
  [[ -d "$target" ]] || { echo "[missing] $target"; continue; }
  [[ -f "$target/deploy.pt" ]] || {
    echo "[abort] deploy.pt missing; refusing to trim $target" >&2
    exit 1
  }
  [[ -d "$target/checkpoints" ]] && remove_path "$target/checkpoints"
done

echo
echo "== Delete rejected/superseded training directories =="
reclaim=0
for name in "${DELETE_DIRS[@]}"; do
  target=$(checked_dir "$name")
  [[ -d "$target" ]] || { echo "[missing] $target"; continue; }
  size=$(bytes_of "$target")
  reclaim=$((reclaim + size))
  remove_path "$target"
done

echo
if [[ "$MODE" == apply ]]; then
  echo "cleanup complete; deleted rejected directories listed above"
else
  echo "dry-run complete; estimated rejected-directory reclaim: $((reclaim / 1024 / 1024 / 1024)) GiB"
  echo "review the exact list, then run: bash $0 --apply"
fi
echo "Evaluation artifacts under the repository outputs/ directory were not touched."
