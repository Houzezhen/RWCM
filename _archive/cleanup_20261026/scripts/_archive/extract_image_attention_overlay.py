#!/usr/bin/env python
"""提取 WCM 视觉编码器（ViT）最后一层注意力，平滑后叠加到原始图像上。

流程：
  1. 加载 deploy checkpoint 与数据管线（与 wcm_infer_acp_to_dataset.py 相同）；
  2. 在 ViT backbone 最后一层 encoder layer 上挂 forward hook，
     取 CLS token 对各 patch 的注意力（多头平均），reshape 成 patch 网格；
  3. 双三次插值到原图分辨率 + 高斯模糊，得到平滑过渡的热力图；
  4. 用透明渐变 colormap（jet）按 alpha 叠加到对应相机原始帧上保存。

用法：
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/extract_image_attention_overlay.py \
      --checkpoint outputs/wcm_toiletButton_merged_0814/deploy.pt \
      --dataset-root data_toiletButton_merged_0814_with_return \
      --num-batches 2 --batch-size 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/image_attention_overlay")
    parser.add_argument("--alpha", type=float, default=0.5, help="热力图叠加强度")
    parser.add_argument("--blur-sigma", type=float, default=4.0, help="高斯模糊 sigma（像素）")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-pretrained-vision",
        action="store_true",
        help="不加载 checkpoint 训练权重，ViT 用 HF 原始预训练权重（对照实验）",
    )
    parser.add_argument(
        "--image-keys",
        default="",
        help="覆盖 checkpoint 的 image_keys（逗号分隔），用于外部数据集如 LIBERO",
    )
    return parser.parse_args()


def gaussian_blur(heat: np.ndarray, sigma: float) -> np.ndarray:
    """对 [H,W] 热力图做可分离高斯模糊（scipy 不在依赖里，手写卷积）。"""
    if sigma <= 0:
        return heat
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    tensor = torch.from_numpy(heat).float()[None, None]
    k1 = torch.from_numpy(kernel).float().view(1, 1, -1, 1)
    k2 = torch.from_numpy(kernel).float().view(1, 1, 1, -1)
    tensor = F.conv2d(F.pad(tensor, (0, 0, radius, radius), mode="reflect"), k1)
    tensor = F.conv2d(F.pad(tensor, (radius, radius, 0, 0), mode="reflect"), k2)
    return tensor[0, 0].numpy()


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    """heat 归一化后上色，与 RGB 图像 [H,W,3]（0-255）按 alpha 混合。"""
    lo, hi = float(heat.min()), float(heat.max())
    norm = (heat - lo) / (hi - lo + 1e-8)
    cmap = plt.get_cmap("jet")
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * image).astype(np.uint8)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    print(f"[v-attn] loading checkpoint: {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {payload.get('schema_version')!r}")

    train_config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    if args.dataset_root:
        train_config.data.root = args.dataset_root
    if args.image_keys:
        train_config.data.image_keys = args.image_keys.split(",")
        train_config.data.normalize_action = False  # 外部数据集无训练统计

    print(f"[v-attn] dataset: repo={train_config.data.repo_id!r} root={train_config.data.root!r}", flush=True)
    dataset = load_lerobot_dataset(train_config.data)
    episode_ids = episode_ids_from_dataset(dataset)
    eval_dataset = LeRobotWorldCriticDataset(dataset, train_config.data, episode_ids)
    if len(eval_dataset) == 0:
        raise ValueError("No temporal windows could be built.")
    print(f"[v-attn] windows: {len(eval_dataset)}", flush=True)

    processor = build_processor(train_config.model)
    image_size = train_config.model.vision.image_size
    collator = WorldCriticCollator(
        processor,
        image_size,
        train_config.model.language.max_length,
    )
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = WorldCriticModel(train_config.model)
    if args.use_pretrained_vision:
        # 保留 HF 预训练初始化的 ViT，不覆盖 checkpoint 权重（对照实验）
        print("[v-attn] using ORIGINAL pretrained ViT weights (no checkpoint load)", flush=True)
    else:
        model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)

    # —— ViT backbone 最后一层：patch forward 重算注意力 ——
    vit_layers = model.vision_encoder.backbone.layers
    last_vit_layer = vit_layers[-1]
    captured: list[torch.Tensor] = []
    original_vit_forward = last_vit_layer.forward

    def vit_layer_forward(hidden_states, *_args, _layer=last_vit_layer):
        if _layer is None:
            raise RuntimeError("_layer binding lost.")
        qkv = _layer.attention  # ViTAttention：q/k/v_proj + num_attention_heads
        head_dim = qkv.q_proj.out_features // qkv.num_attention_heads
        n = hidden_states.size(0)
        q = qkv.q_proj(hidden_states).view(n, -1, qkv.num_attention_heads, head_dim).transpose(1, 2)
        k = qkv.k_proj(hidden_states).view(n, -1, qkv.num_attention_heads, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / (head_dim**0.5)
        probs = scores.softmax(dim=-1)
        captured.append(probs.mean(dim=1).detach().float().cpu())  # [N, tokens, tokens] 头平均
        return original_vit_forward(hidden_states)
    last_vit_layer.forward = vit_layer_forward

    # 原始帧获取：collator 的 images 已归一化，需另行从数据集取原图。
    # LeRobotWorldCriticDataset 的样本索引顺序与 collate 前一致，
    # 这里简单用 batch 内的 episode/frame 索引回数据集取原图。
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    loader_iter = iter(loader)
    for batch_index in range(args.num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        captured.clear()
        batch_dev = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            model(
                images=batch_dev["images"],
                actions=batch_dev["actions"],
                instruction_input_ids=batch_dev["instruction_input_ids"],
                instruction_attention_mask=batch_dev["instruction_attention_mask"],
                valid_mask=batch_dev["valid_mask"],
            )
        if not captured:
            raise RuntimeError("No ViT attention captured.")
        attn = captured[-1]  # [B*T*V, tokens, tokens]
        images = batch["images"]  # [B, T+1, V, C, H, W]（CPU，归一化前处理后）
        B, Tp1, V = images.shape[:3]
        if attn.shape[0] != B * Tp1 * V:
            raise RuntimeError(f"Unexpected attention batch size {attn.shape[0]} vs {B*Tp1*V}")

        # patch 网格：ViT 输出 tokens = 1(CLS) + grid*grid
        n_tokens = attn.shape[-1]
        grid = int(round((n_tokens - 1) ** 0.5))

        for b in range(B):
            valid = batch["valid_mask"][b].bool()
            n_valid = int(valid.sum())
            episode = int(batch["episode_id"][b])
            frames = batch["frame_indices"][b][valid].cpu().numpy()
            for t in range(n_valid):
                for v in range(V):
                    flat = (b * Tp1 + t) * V + v
                    # CLS 对 patch 的注意力 → 空间热力图
                    cls_attn = attn[flat, 0, 1:].reshape(grid, grid).numpy()
                    # 双三次上采样到模型输入分辨率，再放大回原图 480x640
                    heat = torch.from_numpy(cls_attn)[None, None]
                    heat = F.interpolate(heat, size=(image_size, image_size), mode="bicubic", align_corners=False)
                    heat = F.interpolate(
                        heat, size=(480, 640), mode="bicubic", align_corners=False
                    )[0, 0].numpy()
                    heat = gaussian_blur(heat, args.blur_sigma)

                    # 反归一化拿原始像素（用 processor 的均值方差）
                    img = images[b, t, v].permute(1, 2, 0).numpy()
                    mean = np.array(processor.image_processor.image_mean).reshape(1, 1, -1)
                    std = np.array(processor.image_processor.image_std).reshape(1, 1, -1)
                    img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
                    # 模型输入是正方形 resize，这里拉伸回原图 480x640
                    img_resized = np.array(
                        torch.nn.functional.interpolate(
                            torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None],
                            size=(480, 640),
                            mode="bilinear",
                            align_corners=False,
                        )[0].permute(1, 2, 0).numpy(),
                        dtype=np.uint8,
                    )

                    blended = overlay_heatmap(img_resized, heat, args.alpha)
                    tag = f"b{batch_index:02d}_s{b:02d}_ep{episode:03d}_f{int(frames[t]):05d}_cam{v}"
                    fig, ax = plt.subplots(figsize=(6.4, 4.8))
                    ax.imshow(blended)
                    ax.axis("off")
                    ax.set_title(tag)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"{tag}.png", dpi=150)
                    plt.close(fig)
                    saved += 1
            print(f"[v-attn] batch {batch_index + 1}/{args.num_batches} done", flush=True)

    print(f"[v-attn] saved {saved} overlay images to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
