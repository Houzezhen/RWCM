#!/usr/bin/env python
"""Register token 消融实验：K ∈ {0,1,4,8} 对比 ViT 最后一层注意力分布。

设计（Darcet et al. 2023, "Vision Transformers Need Registers"）：
  - 基线 K=0：原始 HF ViT 权重直接推理；
  - K>0：在第 0 层的 CLS+patch 序列后拼接 K 个随机初始化 register token，
    共同经过一遍 encoder（标准 early 语义）；
  - 同一批 LIBERO 样本，抓最后一层 CLS→patch 注意力（头平均），
    计算分布指标并输出热力图。

指标：
  - entropy：注意力分布熵（越低越集中）；
  - top5_ratio：前 5% patch 注意力质量（越高越集中）；
  - grid_var：14x14 网格方差（越高对比越强、分布越不均匀）；
  - high_attn_frac & 熵在 K 间的变化，用于判断 register 是否
    吸收了全局/异常注意力（Darcet 论文中 register 吸收高范数 token
    后，patch 注意力应更平滑地覆盖物体）。

用法：
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/ablation_register_tokens.py \
      --dataset-root libero_datasets/wcm_libero \
      --image-keys observation.images.image,observation.images.wrist_image
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_critic.config import DataConfig, ModelConfig, VisionConfig  # noqa: E402
from world_critic.data import (  # noqa: E402
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_processor,
    episode_ids_from_dataset,
    load_lerobot_dataset,
)
from world_critic.model import WorldCriticModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="libero_datasets/wcm_libero")
    parser.add_argument(
        "--image-keys",
        default="observation.images.image,observation.images.wrist_image",
    )
    parser.add_argument("--checkpoint", default="outputs/wcm_toiletButton_merged_0814/deploy.pt",
                        help="借用其模型结构配置（vision/language）与数据配置")
    parser.add_argument("--register-counts", default="0,1,4,8")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/register_token_ablation")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def attention_metrics(
    patch_attn: np.ndarray,
    register_attn: np.ndarray,
    grid: int,
) -> dict[str, float]:
    patch = patch_attn.astype(np.float64)
    register = register_attn.astype(np.float64)
    patch_mass = float(patch.sum())
    conditional = patch / max(patch_mass, 1e-12)
    k = max(1, int(conditional.size * 0.05))
    return {
        "patch_mass": patch_mass,
        "register_mass": float(register.sum()),
        "patch_entropy": float(-(conditional * np.log(conditional + 1e-12)).sum()),
        "patch_top5_ratio": float(np.sort(conditional)[-k:].sum()),
        "patch_grid_var": float(conditional.reshape(grid, grid).var()),
        "raw_patch_top5_mass": float(np.sort(patch)[-k:].sum()),
    }


def attention_projections(layer):
    attention = layer.attention
    candidates = (attention, getattr(attention, "attention", None))
    for module in candidates:
        if module is None:
            continue
        query = getattr(module, "q_proj", None)
        key = getattr(module, "k_proj", None)
        if query is None:
            query = getattr(module, "query", None)
        if key is None:
            key = getattr(module, "key", None)
        heads = getattr(module, "num_attention_heads", None)
        if query is not None and key is not None and heads is not None:
            return module, query, key, int(heads)
    raise TypeError(f"Unsupported attention layout in {type(layer).__name__}.")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model_cfg = payload["config"]["model"]

    data_cfg = DataConfig(
        repo_id="wcm_libero",
        root=args.dataset_root,
        image_keys=args.image_keys.split(","),
        action_key="action",
        state_key="observation.state",
        return_key="return",
        normalize_action=False,
    )
    dataset = load_lerobot_dataset(data_cfg)
    episode_ids = episode_ids_from_dataset(dataset)
    eval_dataset = LeRobotWorldCriticDataset(dataset, data_cfg, episode_ids)

    from world_critic.config import LanguageConfig  # noqa: E402
    model_cfg = ModelConfig(**{k: v for k, v in base_model_cfg.items() if k not in ("vision", "language")})
    model_cfg.language = LanguageConfig(**base_model_cfg["language"])
    vision_cfg = VisionConfig(**{k: v for k, v in base_model_cfg["vision"].items() if k != "num_register_tokens"})
    vision_cfg.pretrained = True
    vision_cfg.trainable = True
    vision_cfg.register_insert = "early"
    model_cfg.vision = vision_cfg
    model_cfg.action_dim = 7
    model_cfg.state_dim = 8

    processor = build_processor(model_cfg)
    collator = WorldCriticCollator(processor, vision_cfg.image_size, model_cfg.language.max_length)
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )
    batches = []
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        batches.append(batch)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, list[dict]] = {}

    for k in [int(x) for x in args.register_counts.split(",")]:
        vision_cfg.num_register_tokens = k
        model_cfg.vision = vision_cfg
        torch.manual_seed(args.seed)  # 每个 K 相同初始化种子（register 随机初始化）
        model = WorldCriticModel(model_cfg)
        model.to(device).eval().requires_grad_(False)

        # 抓最后一层 CLS→patch 注意力
        last_layer = model.vision_encoder.encoder_layers()[-1]
        attention_module, q_proj, k_proj, heads = attention_projections(last_layer)
        captured: list[torch.Tensor] = []
        original_forward = attention_module.forward

        def patched(
            hidden_states,
            *forward_args,
            _cap=captured,
            _original=original_forward,
            **forward_kwargs,
        ):
            head_dim = q_proj.out_features // heads
            n = hidden_states.size(0)
            q = q_proj(hidden_states).view(n, -1, heads, head_dim).transpose(1, 2)
            kk = k_proj(hidden_states).view(n, -1, heads, head_dim).transpose(1, 2)
            scores = q @ kk.transpose(-1, -2) / (head_dim**0.5)
            probs = scores.softmax(dim=-1)
            _cap.append(probs.mean(dim=1).detach().float().cpu())
            return _original(hidden_states, *forward_args, **forward_kwargs)

        attention_module.forward = patched
        try:
            per_sample = []
            for b_idx, batch in enumerate(batches):
                captured.clear()
                with torch.no_grad():
                    model(
                        images=batch["images"],
                        actions=batch["actions"],
                        instruction_input_ids=batch["instruction_input_ids"],
                        instruction_attention_mask=batch["instruction_attention_mask"],
                        valid_mask=batch["valid_mask"],
                    )
                attn = captured[-1]  # [B*(T+1)*V, tokens, tokens]（K>0 时含 register token）
                images = batch["images"]
                B, Tp1, V = images.shape[:3]
                n_tokens = attn.shape[-1]
                grid = int(round((n_tokens - 1 - k) ** 0.5))
                if (grid + 1 + k) ** 2 and (grid * grid + 1 + k) != n_tokens:
                    raise RuntimeError(f"token count {n_tokens} != 1+{grid}^2+{k}")

                for b in range(B):
                    n_valid = int(batch["valid_mask"][b].sum())
                    episode = int(batch["episode_id"][b])
                    frames = batch["frame_indices"][b][:n_valid].numpy()
                    for t in range(n_valid):
                        for v in range(V):
                            flat = (b * Tp1 + t) * V + v
                            # CLS 对 patch+register 的注意力；只取 patch 部分
                            cls_attn = attn[flat, 0].numpy()
                            patch_attn = cls_attn[1 : 1 + grid * grid]
                            register_attn = cls_attn[1 + grid * grid :]
                            m = attention_metrics(patch_attn, register_attn, grid)
                            m.update(episode=episode, frame=int(frames[t]), cam=v)
                            per_sample.append(m)

                            # 热力图（每个 K 只画前 6 个，避免文件爆炸）
                            if len([s for s in per_sample]) <= 6:
                                heat = torch.from_numpy(patch_attn.reshape(grid, grid))[None, None]
                                heat = F.interpolate(heat, size=(256, 256), mode="bicubic", align_corners=False)[0, 0].numpy()
                                img = images[b, t, v].permute(1, 2, 0).numpy()
                                mean = np.array(processor.image_processor.image_mean).reshape(1, 1, -1)
                                std = np.array(processor.image_processor.image_std).reshape(1, 1, -1)
                                img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)
                                img = np.array(
                                    F.interpolate(
                                        torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None],
                                        size=(256, 256), mode="bilinear", align_corners=False,
                                    )[0].permute(1, 2, 0).numpy(), dtype=np.uint8,
                                )
                                lo, hi = heat.min(), heat.max()
                                norm = (heat - lo) / (hi - lo + 1e-8)
                                overlay = (0.5 * (plt.get_cmap("jet")(norm)[..., :3] * 255) + 0.5 * img).astype(np.uint8)
                                tag = f"K{k}_b{b_idx}_ep{episode:03d}_f{int(frames[t]):05d}_cam{v}"
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.imshow(overlay)
                                ax.axis("off")
                                ax.set_title(tag)
                                fig.tight_layout()
                                fig.savefig(out_dir / f"{tag}.png", dpi=120)
                                plt.close(fig)
                print(f"[ablation] K={k}: batch {b_idx + 1}/{len(batches)}", flush=True)
        finally:
            attention_module.forward = original_forward
        results[k] = per_sample

    # —— 汇总指标 ——
    summary = {}
    for k, samples in results.items():
        if not samples:
            raise RuntimeError(f"No attention samples were collected for K={k}.")
        summary[k] = {
            "n": len(samples),
            **{
                f"{key}_mean": float(np.mean([sample[key] for sample in samples]))
                for key in samples[0]
                if key not in {"episode", "frame", "cam"}
            },
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[ablation] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
