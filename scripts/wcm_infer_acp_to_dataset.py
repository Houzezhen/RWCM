#!/usr/bin/env python
"""用 WCM（world critic）价值函数给 LeRobot v3 / v2.1 数据集逐帧推理价值，并生成 ACP 标注。

参考 evo_rl 中 pistar06 的离线价值标注流水线（lerobot_value_infer.py）：
  1. 用 WCM 模型对数据集逐帧（3 帧历史窗口 + 动作条件）推理价值；
  2. 用数据集的 return 列（真值 return-to-go）做帧间差分得到稠密 reward；
  3. 计算 n-step advantage（gamma=1）；
  4. 按任务内分位数把 advantage 二值化为 acp_indicator（默认 top 30% 为正）；
  5. 回写到数据集：complementary_info.value / advantage / acp_indicator。

下游 lerobot_train 的 ACP hook 只读 acp_indicator，无需改动。

用法示例：
  python scripts/wcm_infer_acp_to_dataset.py \
      --checkpoint outputs/wcm/checkpoints/best.pt \
      --dataset-root /media/Data/user/huggingface/lerobot/lerobotevo/data_toiletButton_0722_2_with_return \
      --batch-size 16 --num-workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# 允许从仓库根目录以外的位置运行（把本脚本的上级目录加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_critic.checkpoint import CHECKPOINT_SCHEMA_VERSION  # noqa: E402
from world_critic.config import apply_runtime_overrides  # noqa: E402
from world_critic.data import (  # noqa: E402
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_processor,
    episode_ids_from_dataset,
    load_lerobot_dataset,
)
from world_critic.model import WorldCriticModel  # noqa: E402
from world_critic.training import config_from_checkpoint_payload  # noqa: E402

# 与 evo_rl configs/value.py 的默认字段名保持一致
VALUE_FIELD = "complementary_info.value"
ADVANTAGE_FIELD = "complementary_info.advantage"
INDICATOR_FIELD = "complementary_info.acp_indicator"


# ---------------------------------------------------------------------------
# 下列 numpy 逻辑从 evo_rl/src/lerobot/scripts/lerobot_value_infer.py 移植
# （MIT 许可的 HuggingFace 代码），保持数值行为一致。
# ---------------------------------------------------------------------------

def compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """reward[i] = targets[i] - targets[i+1]（连续帧），末帧 = targets[i]."""
    rewards = np.zeros_like(targets, dtype=np.float32)
    n = targets.shape[0]
    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        if is_next_in_episode:
            rewards[i] = float(targets[i] - targets[i + 1])
        else:
            rewards[i] = float(targets[i])
    return rewards


def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    if n_step <= 0:
        raise ValueError("'n_step' must be > 0.")
    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ep_i = episode_indices[i]
        fi = frame_indices[i]
        discounted_sum = 0.0
        j = i
        steps = 0
        while steps < n_step and j < n:
            same_episode = episode_indices[j] == ep_i
            contiguous = frame_indices[j] == fi + steps
            if not same_episode or not contiguous:
                break
            discounted_sum += float(rewards[j])
            steps += 1
            j += 1
        if steps == n_step and j < n and episode_indices[j] == ep_i and frame_indices[j] == fi + n_step:
            bootstrap = float(values[j])
        else:
            bootstrap = 0.0
        advantages[i] = float(discounted_sum + bootstrap - values[i])
    return advantages


def compute_td_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """TD(0) advantage：A_t = r_t + gamma * V(s_{t+1}) - V(s_t)。

    用真实 trajectory 的 (s_t, a_t, r_t, s_{t+1}) 构造；episode 末帧无 s_{t+1}，bootstrap 置 0。
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("'gamma' must be within [0, 1].")
    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        next_value = float(values[i + 1]) if is_next_in_episode else 0.0
        advantages[i] = float(rewards[i]) + gamma * next_value - float(values[i])
    return advantages


def compute_task_thresholds(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    positive_ratio: float,
) -> dict[int, float]:
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("'positive_ratio' must be within [0, 1].")
    thresholds: dict[int, float] = {}
    quantile = 1.0 - positive_ratio
    for task_idx in np.unique(task_indices):
        task_adv = advantages[task_indices == task_idx]
        if task_adv.size == 0:
            thresholds[int(task_idx)] = float("inf")
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, quantile))
    return thresholds


def binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
    interventions: np.ndarray,
    force_intervention_positive: bool,
) -> np.ndarray:
    indicators = np.zeros_like(advantages, dtype=np.int64)
    for i in range(advantages.shape[0]):
        task_idx = int(task_indices[i])
        threshold = thresholds[task_idx]
        indicators[i] = 1 if float(advantages[i]) >= threshold else 0
    if force_intervention_positive:
        intervention_mask = interventions.astype(np.float32) > 0.5
        indicators[intervention_mask] = 1
    return indicators


def update_feature_metadata(dataset_root: Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": list(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")


def write_columns_in_place(
    dataset_root: Path,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> None:
    """在 data/chunk-*/file-*.parquet 中新增/覆盖列，并更新 meta/info.json 特征表."""
    if absolute_indices.ndim != 1:
        raise ValueError("'absolute_indices' must be rank-1.")
    max_index = int(np.max(absolute_indices))
    selected = np.zeros(max_index + 1, dtype=np.bool_)
    selected[absolute_indices] = True

    lookups: dict[str, np.ndarray] = {}
    for field, values in columns.items():
        lookup_dtype = np.float32 if feature_infos[field]["dtype"] == "float32" else np.int64
        lookup = np.zeros(max_index + 1, dtype=lookup_dtype)
        lookup[absolute_indices] = values.astype(lookup_dtype, copy=False)
        lookups[field] = lookup

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        # v2.1 layout keeps one parquet per episode instead of sharded files.
        data_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    for parquet_path in data_files:
        table = pq.read_table(parquet_path)
        idx_np = table["index"].to_numpy().astype(np.int64, copy=False)

        in_range = (idx_np >= 0) & (idx_np <= max_index)
        in_subset = np.zeros_like(in_range)
        in_subset[in_range] = selected[idx_np[in_range]]

        new_table = table
        for field, lookup in lookups.items():
            ftype = feature_infos[field]["dtype"]
            if ftype == "float32":
                default_value = np.nan
                target_dtype = np.float32
                pa_type = pa.float32()
            elif ftype == "int64":
                default_value = 0
                target_dtype = np.int64
                pa_type = pa.int64()
            else:
                raise ValueError(f"Unsupported annotation dtype '{ftype}' for field '{field}'.")

            if field in new_table.schema.names:
                current = new_table[field].to_numpy().astype(target_dtype, copy=True)
            else:
                current = np.full(idx_np.shape[0], default_value, dtype=target_dtype)

            if np.any(in_subset):
                subset_indices = idx_np[in_subset]
                current[in_subset] = lookup[subset_indices]

            array = pa.array(current, type=pa_type)
            if field in new_table.schema.names:
                col_idx = new_table.schema.names.index(field)
                new_table = new_table.set_column(col_idx, field, array)
            else:
                new_table = new_table.append_column(field, array)

        pq.write_table(new_table, parquet_path, compression="snappy")

    update_feature_metadata(dataset_root=dataset_root, feature_infos=feature_infos)
    if (dataset_root / "meta" / "episodes_stats.jsonl").is_file():
        update_v21_episodes_stats(
            dataset_root=dataset_root,
            data_files=data_files,
            absolute_indices=absolute_indices,
            columns=columns,
        )


def update_v21_episodes_stats(
    dataset_root: Path,
    data_files: list[Path],
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
) -> None:
    """Append per-episode stats for the newly written columns to v2.1 metadata.

    v2.1 keeps ``meta/episodes_stats.jsonl`` (one JSON object per episode);
    v3 stores the same statistics in parquet and is handled elsewhere.
    """
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    lines = [line for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or absolute_indices.size == 0:
        return

    index_to_episode = np.full(int(np.max(absolute_indices)) + 1, -1, dtype=np.int64)
    for parquet_path in data_files:
        table = pq.read_table(parquet_path, columns=["index", "episode_index"])
        index_to_episode[table["index"].to_numpy().astype(np.int64)] = (
            table["episode_index"].to_numpy().astype(np.int64)
        )

    per_episode: dict[int, dict[str, list[float]]] = {}
    for field, values in columns.items():
        values = np.asarray(values).reshape(-1)
        if values.size != absolute_indices.size:
            raise ValueError(
                f"Column {field!r} has {values.size} values but {absolute_indices.size} indices."
            )
        for abs_index, value in zip(absolute_indices.tolist(), values.tolist(), strict=True):
            episode = int(index_to_episode[int(abs_index)])
            if episode < 0:
                raise ValueError(f"Global index {abs_index} is not present in any data file.")
            per_episode.setdefault(episode, {}).setdefault(field, []).append(float(value))

    out_lines: list[str] = []
    for line in lines:
        entry = json.loads(line)
        episode = int(entry["episode_index"])
        if episode in per_episode:
            for field, bucket in per_episode[episode].items():
                arr = np.asarray(bucket, dtype=np.float64)
                entry["stats"][field] = {
                    "min": [float(arr.min())],
                    "max": [float(arr.max())],
                    "mean": [float(arr.mean())],
                    "std": [float(arr.std()) if arr.size > 1 else 0.0],
                    "count": [int(arr.size)],
                }
        out_lines.append(json.dumps(entry))
    stats_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="WCM checkpoint（deploy.pt 或 best.pt）")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="数据集根目录；留空则使用 checkpoint 中保存的 data.root",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="调试用：最多推理多少个 batch")
    parser.add_argument("--advantage-mode", choices=("n_step", "td"), default="n_step",
                        help="advantage 计算：n_step（默认，与 evo 一致）或 td（TD(0)：A_t = r_t + gamma*V(s_{t+1}) - V(s_t)）")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="TD advantage 的折扣因子（默认 1.0，与 return 定义一致）")
    parser.add_argument("--n-step", type=int, default=50, help="advantage 的 n 步窗口（默认 50，与 evo 一致）")
    parser.add_argument("--positive-ratio", type=float, default=0.3, help="每任务 top 比例为正（默认 0.3）")
    parser.add_argument("--force-intervention-positive", action="store_true", help="干预帧强制标记为正")
    parser.add_argument("--dry-run", action="store_true", help="只推理并打印统计，不写回数据集")
    parser.add_argument("--device", default="", help="如 'cuda:0'；留空自动选择")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    print(f"[wcm-acp] loading checkpoint: {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {payload.get('schema_version')!r}")
    if "model" not in payload or "config" not in payload:
        raise KeyError("Checkpoint artifact is missing model or config.")

    train_config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    if args.dataset_root:
        train_config.data.root = args.dataset_root

    print(f"[wcm-acp] dataset: repo={train_config.data.repo_id!r} root={train_config.data.root!r}", flush=True)
    dataset = load_lerobot_dataset(train_config.data)
    print(f"[wcm-acp] dataset rows: {len(dataset)}", flush=True)

    episode_ids = episode_ids_from_dataset(dataset)
    print(f"[wcm-acp] episodes: {len(episode_ids)}", flush=True)

    print("[wcm-acp] building temporal windows...", flush=True)
    eval_dataset = LeRobotWorldCriticDataset(dataset, train_config.data, episode_ids)
    if len(eval_dataset) == 0:
        raise ValueError("No temporal windows could be built (episodes shorter than the window?).")
    print(f"[wcm-acp] temporal windows: {len(eval_dataset)}", flush=True)

    print("[wcm-acp] loading image processor and tokenizer...", flush=True)
    processor = build_processor(train_config.model)
    collator = WorldCriticCollator(
        processor,
        train_config.model.vision.image_size,
        train_config.model.language.max_length,
    )
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    print("[wcm-acp] constructing WorldCriticModel (HF weights may load next)...", flush=True)
    model = WorldCriticModel(train_config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    if train_config.precision == "bf16" and device.type == "cuda":
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = nullcontext()

    # 逐窗口推理，取每个窗口 endpoint（最后有效历史位置）的价值，
    # 映射到 (episode_id, frame_index)。
    predictions: dict[tuple[int, int], float] = {}
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast:
                output = model(
                    images=batch["images"],
                    actions=batch["actions"],
                    instruction_input_ids=batch["instruction_input_ids"],
                    instruction_attention_mask=batch["instruction_attention_mask"],
                    valid_mask=batch["valid_mask"],
                )
            valid = output.valid_mask.bool()
            positions = torch.arange(valid.shape[1], device=valid.device).expand(valid.shape[0], -1)
            last_index = positions.masked_fill(~valid, -1).max(dim=1).values
            row_index = torch.arange(valid.shape[0], device=valid.device)
            endpoint_values = output.value[row_index, last_index].squeeze(-1)  # [B]

            episodes = batch["episode_id"].tolist()
            frames = batch["frame_indices"][row_index, last_index].tolist()
            values = endpoint_values.float().cpu().tolist()
            for ep, fr, val in zip(episodes, frames, values, strict=True):
                predictions[(int(ep), int(fr))] = float(val)

            if (batch_index + 1) % 20 == 0 or args.max_batches is not None and batch_index + 1 == args.max_batches:
                print(
                    f"[wcm-acp] inferred batches={batch_index + 1} "
                    f"frames={len(predictions)} elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break

    print(f"[wcm-acp] inference done: {len(predictions)} frames, elapsed_s={time.monotonic() - started:.1f}",
          flush=True)

    # 按数据集行序组装每帧 value；窗口起点位于帧 0/1 时无完整历史，
    # 用该 episode 内第一个有效值（帧 2）向前填充。
    raw = dataset.hf_dataset.with_format(None)
    n_rows = len(raw)
    absolute_indices = np.asarray(raw["index"], dtype=np.int64)
    episode_indices = np.asarray(raw["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw["frame_index"], dtype=np.int64)

    values_full = np.full(n_rows, np.nan, dtype=np.float32)
    for (ep, fr), val in predictions.items():
        row = np.flatnonzero((episode_indices == ep) & (frame_indices == fr))
        if row.size == 0:
            raise ValueError(f"Prediction references unknown frame: episode={ep} frame={fr}")
        values_full[row[0]] = val

    # 每个 episode 段内做边缘填充：
    #   开头（窗口起点早于 history_size）用第一个有效值前向填充；
    #   末尾（最后 history_size-1 帧不是任何窗口的 endpoint）用最后一个有效值后向填充。
    boundaries = np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1
    seg_starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries.astype(np.int64)))
    seg_ends = np.concatenate((boundaries.astype(np.int64), np.asarray([n_rows], dtype=np.int64)))
    for start, end in zip(seg_starts.tolist(), seg_ends.tolist(), strict=True):
        seg = values_full[start:end]
        valid_positions = np.flatnonzero(np.isfinite(seg))
        if valid_positions.size == 0:
            raise ValueError(f"Episode starting at row {start} has no predicted values at all.")
        seg[: valid_positions[0]] = seg[valid_positions[0]]
        seg[valid_positions[-1] + 1:] = seg[valid_positions[-1]]

    if not np.isfinite(values_full).all():
        raise ValueError("Some frames still lack predicted values after edge fill.")

    # 真值 return-to-go：用数据集的 return 列（归一化后的 return）做帧间差分
    if train_config.data.return_key not in raw.column_names:
        raise KeyError(f"Dataset is missing return column {train_config.data.return_key!r}. "
                       "Run add_returns_to_lerobot.py (v3) or add_returns_to_lerobot_v21.py (v2.1) first.")
    return_targets = np.asarray(raw[train_config.data.return_key], dtype=np.float32).reshape(-1)

    if "task_index" in raw.column_names:
        task_indices = np.asarray(raw["task_index"], dtype=np.int64).reshape(-1)
    else:
        task_indices = np.zeros(n_rows, dtype=np.int64)

    if "complementary_info.is_intervention" in raw.column_names:
        interventions = np.asarray(raw["complementary_info.is_intervention"], dtype=np.float32)
    else:
        interventions = np.zeros(n_rows, dtype=np.float32)

    rewards = compute_dense_rewards_from_targets(return_targets, episode_indices, frame_indices)
    if args.advantage_mode == "td":
        advantages = compute_td_advantages(
            rewards=rewards,
            values=values_full,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            gamma=args.gamma,
        )
    else:
        advantages = compute_n_step_advantages(
            rewards=rewards,
            values=values_full,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            n_step=args.n_step,
        )
    thresholds = compute_task_thresholds(task_indices, advantages, args.positive_ratio)
    indicators = binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=args.force_intervention_positive,
    )

    print(f"[wcm-acp] value    min={np.min(values_full):.4f} max={np.max(values_full):.4f} "
          f"mean={np.mean(values_full):.4f}", flush=True)
    print(f"[wcm-acp] advantage min={np.min(advantages):.4f} max={np.max(advantages):.4f} "
          f"mean={np.mean(advantages):.4f}", flush=True)
    print(f"[wcm-acp] thresholds: {json.dumps({str(k): round(float(v), 4) for k, v in thresholds.items()})}",
          flush=True)
    print(f"[wcm-acp] indicator positive ratio: {float(np.mean(indicators.astype(np.float32))):.4f}",
          flush=True)

    if args.dry_run:
        print("[wcm-acp] dry-run: not writing to the dataset.", flush=True)
        return

    columns = {
        VALUE_FIELD: values_full.astype(np.float32),
        ADVANTAGE_FIELD: advantages.astype(np.float32),
        INDICATOR_FIELD: indicators.astype(np.int64),
    }
    feature_infos = {
        VALUE_FIELD: {"dtype": "float32", "shape": (1,), "names": None},
        ADVANTAGE_FIELD: {"dtype": "float32", "shape": (1,), "names": None},
        INDICATOR_FIELD: {"dtype": "int64", "shape": (1,), "names": None},
    }
    root_candidates = [Path(train_config.data.root).expanduser().resolve()]
    if train_config.data.repo_id:
        root_candidates.append(root_candidates[0].joinpath(*train_config.data.repo_id.split("/")))
    dataset_root = next(
        (candidate for candidate in root_candidates if (candidate / "meta" / "info.json").is_file()),
        root_candidates[0],
    )
    print(f"[wcm-acp] writing columns in place: {dataset_root}", flush=True)
    write_columns_in_place(
        dataset_root=dataset_root,
        absolute_indices=absolute_indices,
        columns=columns,
        feature_infos=feature_infos,
    )
    print("[wcm-acp] done.", flush=True)


if __name__ == "__main__":
    main()
