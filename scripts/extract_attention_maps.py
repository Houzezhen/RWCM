#!/usr/bin/env python
"""提取 WCM context trunk 最后一层 self-attention 热力图，随机抽样数据集测试。

流程：
  1. 加载 deploy checkpoint（含模型权重 + 训练时配置）；
  2. 按 checkpoint 配置构建数据集与 DataLoader（与 wcm_infer_acp_to_dataset.py 相同的路径）；
  3. 在 context_trunk.transformer.layers[-1].self_attn 上挂 hook，
     以 average_attn_weights=True 抓取最后一层平均注意力矩阵；
  4. 随机抽取若干 batch 推理，保存热力图 PNG + 原始权重 NPZ 到输出目录。

用法：
  python scripts/extract_attention_maps.py \
      --checkpoint outputs/wcm_toiletButton_merged_0814/deploy.pt \
      --dataset-root data_toiletButton_merged_0814_with_return \
      --num-batches 4 --batch-size 8
"""
from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="", help="留空则用 checkpoint 里的 data.root")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=4, help="随机抽取多少个 batch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/attention_maps")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    print(f"[attn] loading checkpoint: {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {payload.get('schema_version')!r}")

    train_config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    if args.dataset_root:
        train_config.data.root = args.dataset_root

    print(f"[attn] dataset: repo={train_config.data.repo_id!r} root={train_config.data.root!r}", flush=True)
    dataset = load_lerobot_dataset(train_config.data)
    episode_ids = episode_ids_from_dataset(dataset)
    eval_dataset = LeRobotWorldCriticDataset(dataset, train_config.data, episode_ids)
    if len(eval_dataset) == 0:
        raise ValueError("No temporal windows could be built.")
    print(f"[attn] windows: {len(eval_dataset)}", flush=True)

    processor = build_processor(train_config.model)
    collator = WorldCriticCollator(
        processor,
        train_config.model.vision.image_size,
        train_config.model.language.max_length,
    )
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 随机抽样
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = WorldCriticModel(train_config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    # —— 挂 hook 抓最后一层 attention ——
    last_layer = model.context_trunk.transformer.layers[-1]
    # TransformerEncoderLayer 在 eval 下会走 fast path（绕过 self_attn 模块），
    # 因此直接 patch 最后一层的 forward，按 norm_first 结构重算并取注意力权重。
    captured: list[torch.Tensor] = []
    original_layer_forward = last_layer.forward

    def layer_forward(
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
        _layer=last_layer,
    ):
        # 显式传入完整因果 mask 时忽略 is_causal 标志（mask 本身已含因果性）。
        if src_mask is None and is_causal:
            src_mask = torch.triu(
                torch.ones(src.size(1), src.size(1), dtype=torch.bool, device=src.device), diagonal=1
            )
        x = src
        attended, weights = _layer.self_attn(
            _layer.norm1(x),
            _layer.norm1(x),
            _layer.norm1(x),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        captured.append(weights.detach().float().cpu())
        x = x + _layer.dropout1(attended)
        y = _layer.linear2(_layer.dropout(_layer.activation(_layer.linear1(_layer.norm2(x)))))
        x = x + _layer.dropout2(y)
        return x

    last_layer.forward = layer_forward
    try:
        # 逐 batch 推理
        results = []
        seen = 0
        loader_iter = iter(loader)
        while seen < args.num_batches:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            captured.clear()
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                model(
                    images=batch["images"],
                    actions=batch["actions"],
                    instruction_input_ids=batch["instruction_input_ids"],
                    instruction_attention_mask=batch["instruction_attention_mask"],
                    valid_mask=batch["valid_mask"],
                )
            if not captured:
                raise RuntimeError("No attention weights captured; fast path may have been used.")
            weights = captured[-1]  # [B, T, T] 平均头注意力
            results.append((batch, weights.clone()))
            seen += 1
            print(f"[attn] batch {seen}/{args.num_batches} attn shape={tuple(weights.shape)}", flush=True)
    finally:
        last_layer.forward = original_layer_forward

    # —— 可视化 + 保存 ——
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_npz = {}
    for b_idx, (batch, weights) in enumerate(results):
        for i in range(weights.shape[0]):
            valid = batch["valid_mask"][i].bool().cpu()
            n_valid = int(valid.sum())
            attn = weights[i, :n_valid, :n_valid].numpy()
            episode = int(batch["episode_id"][i])
            frames = batch["frame_indices"][i][valid].cpu().numpy()
            tag = f"batch{b_idx:02d}_sample{i:02d}_ep{episode:03d}_f{int(frames[-1]):05d}"

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(attn, cmap="viridis", aspect="auto")
            ax.set_xlabel("Key frame (history)")
            ax.set_ylabel("Query frame")
            ax.set_title(f"{tag}\nlast-layer avg attention (causal)")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out_dir / f"{tag}.png", dpi=150)
            plt.close(fig)
            saved_npz[tag] = attn
    np.savez_compressed(out_dir / "attention_weights.npz", **saved_npz)
    print(f"[attn] saved {len(saved_npz)} heatmaps to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
