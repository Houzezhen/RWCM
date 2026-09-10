#!/usr/bin/env python
"""训练后 top5 注意力分析：加载真实训练好的 checkpoint（含训练后的寄存 token 与 ViT 权重），
抓最后一层 CLS→patch 注意力，计算 entropy / top5_ratio / grid_var。
与训练前结果（outputs/register_token_ablation，HF 预训练权重 + 随机寄存）对照。

用法（GPU 空闲时；与训练并行时会很慢）：
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/ablation_register_tokens_trained.py \
      --dataset-root /home/user/.cache/huggingface/lerobot/test/data_toiletButton_0814_expanded_with_return \
      --models baseline=outputs/wcm_baseline_r2/checkpoints/best.pt \
               reg4_d2=outputs/wcm_reg4_r2/checkpoints/best.pt \
               reg4_s42=outputs/wcm_reg4_r2_s42/checkpoints/best.pt \
               bs1=outputs/wcm_blockreg8_bs1/checkpoints/best.pt \
      --num-batches 2 --output-dir outputs/register_ablation_trained
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=path/to/best.pt 列表")
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/register_ablation_trained")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def attention_metrics(cls_attn: np.ndarray, grid: int) -> dict[str, float]:
    a = cls_attn.astype(np.float64)
    entropy = float(-(a * np.log(a + 1e-12)).sum())
    k = max(1, int(a.size * 0.05))
    top5 = float(np.sort(a)[-k:].sum())
    grid_var = float(a.reshape(grid, grid).var())
    return {"entropy": entropy, "top5_ratio": top5, "grid_var": grid_var}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用第一个 checkpoint 的数据/模型配置构建 loader（所有模型同构）
    first_ckpt = args.models[0].split("=", 1)[1]
    payload = torch.load(first_ckpt, map_location="cpu", weights_only=False)
    full_cfg = payload["config"]
    base_model_cfg = full_cfg["model"]
    data_cfg_dict = full_cfg["data"]
    from world_critic.config import DataConfig  # noqa: E402
    data_cfg = DataConfig(**data_cfg_dict)
    data_cfg.root = args.dataset_root  # 强制用指定数据集

    dataset = load_lerobot_dataset(data_cfg)
    episode_ids = episode_ids_from_dataset(dataset)
    eval_dataset = LeRobotWorldCriticDataset(dataset, data_cfg, episode_ids)

    from world_critic.config import LanguageConfig, ModelConfig, VisionConfig  # noqa: E402
    model_cfg = ModelConfig(**{k: v for k, v in base_model_cfg.items() if k not in ("vision", "language")})
    model_cfg.language = LanguageConfig(**base_model_cfg["language"])
    vision_cfg = VisionConfig(**{k: v for k, v in base_model_cfg["vision"].items() if k != "num_register_tokens"})
    model_cfg.action_dim = full_cfg.get("model", {}).get("action_dim", 7)
    model_cfg.state_dim = full_cfg.get("model", {}).get("state_dim", 8)

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
    print(f"[data] {len(batches)} batches x {args.batch_size} loaded from {args.dataset_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary = {}

    for spec in args.models:
        name, ckpt_path = spec.split("=", 1)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mc = payload["config"]["model"]
        k = int(mc["vision"].get("num_register_tokens", 0))

        model_cfg_local = ModelConfig(**{kk: vv for kk, vv in mc.items() if kk not in ("vision", "language")})
        model_cfg_local.language = LanguageConfig(**mc["language"])
        model_cfg_local.vision = VisionConfig(**{kk: vv for kk, vv in mc["vision"].items()})
        model = WorldCriticModel(model_cfg_local).to(device).eval().requires_grad_(False)
        sd = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] {name}: K={k} missing={len(missing)} unexpected={len(unexpected)}")

        last_layer = model.vision_encoder.backbone.layers[-1]
        captured: list[torch.Tensor] = []
        original_forward = last_layer.forward

        def patched(hidden_states, *_args, _layer=last_layer, _cap=captured):
            qkv = _layer.attention
            n_tok_in = hidden_states.size(1)
            head_dim = qkv.q_proj.out_features // qkv.num_attention_heads
            n = hidden_states.size(0)
            q = qkv.q_proj(hidden_states).view(n, -1, qkv.num_attention_heads, head_dim).transpose(1, 2)
            kk = qkv.k_proj(hidden_states).view(n, -1, qkv.num_attention_heads, head_dim).transpose(1, 2)
            scores = q @ kk.transpose(-1, -2) / (head_dim ** 0.5)
            probs = scores.softmax(dim=-1)
            _cap.append(probs.mean(dim=1).detach().float().cpu())
            return original_forward(hidden_states)

        last_layer.forward = patched
        per_sample = []
        try:
            for b_idx, batch in enumerate(batches):
                captured.clear()
                b_dev = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
                with torch.no_grad():
                    model(
                        images=b_dev["images"],
                        actions=b_dev["actions"],
                        instruction_input_ids=b_dev["instruction_input_ids"],
                        instruction_attention_mask=b_dev["instruction_attention_mask"],
                        valid_mask=b_dev["valid_mask"],
                    )
                # 每帧最后一层被调用两次：第一程 197（无寄存）、第二程 1+patch+K（含寄存）。
                # captured 按帧序排列：K>0 时每帧 2 项 [f1_197, f1_201, f2_197, ...]，K=0 时每帧 1 项。
                # 注意每项的 batch 维 = B*Tp1*V（整个扁平 batch 一次前向只调用一次每程），
                # 所以帧 (b,t,v) 的第二程图 = captured[1][flat]（K>0）或 captured[0][flat]（K=0）
                images = batch["images"]
                B, Tp1, V = images.shape[:3]
                if k > 0:
                    second_pass = next((a for a in captured if a.shape[-1] == 1 + 196 + k), None)
                else:
                    second_pass = next((a for a in captured if a.shape[-1] == 197), None)
                if second_pass is None:
                    raise RuntimeError(f"no {'second-pass' if k>0 else ''} attention captured: sizes={[a.shape[-1] for a in captured]}")
                grid = int(round((second_pass.shape[-1] - 1 - k) ** 0.5))
                n_patch = second_pass.shape[-1] - 1 - k
                if grid * grid != n_patch:
                    raise RuntimeError(f"token count {second_pass.shape[-1]} patch={n_patch} 非完全平方")
                for b in range(B):
                    n_valid = int(batch["valid_mask"][b].sum())
                    for t in range(n_valid):
                        for v in range(V):
                            flat = (b * Tp1 + t) * V + v
                            # 帧索引 flat 直接在 batch 维上取（第二程图含全部帧）
                            cls_attn = second_pass[flat, 0, 1: 1 + grid * grid].numpy()
                            m = attention_metrics(cls_attn, grid)
                            per_sample.append(m)
                print(f"[{name}] batch {b_idx + 1}/{len(batches)} done", flush=True)
        finally:
            last_layer.forward = original_forward

        all_summary[name] = {
            "K": k,
            "n": len(per_sample),
            "entropy_mean": float(np.mean([s["entropy"] for s in per_sample])),
            "top5_ratio_mean": float(np.mean([s["top5_ratio"] for s in per_sample])),
            "grid_var_mean": float(np.mean([s["grid_var"] for s in per_sample])),
        }
        print(f"[{name}] top5_ratio_mean = {all_summary[name]['top5_ratio_mean']:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
