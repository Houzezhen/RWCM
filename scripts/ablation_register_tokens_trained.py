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


def attention_metrics(
    patch_attn: np.ndarray,
    register_attn: np.ndarray,
    grid: int,
) -> dict[str, float]:
    """Separate register capture from concentration within the patch distribution."""
    patch = patch_attn.astype(np.float64)
    register = register_attn.astype(np.float64)
    patch_mass = float(patch.sum())
    register_mass = float(register.sum())
    conditional = patch / max(patch_mass, 1e-12)
    k = max(1, int(conditional.size * 0.05))
    return {
        "patch_mass": patch_mass,
        "register_mass": register_mass,
        "patch_entropy": float(-(conditional * np.log(conditional + 1e-12)).sum()),
        "patch_top5_ratio": float(np.sort(conditional)[-k:].sum()),
        "patch_grid_var": float(conditional.reshape(grid, grid).var()),
        # Retain the historical quantity so old summaries remain comparable.
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
    model_cfg.vision = vision_cfg
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
        model.load_state_dict(sd, strict=True)
        print(f"[load] {name}: K={k} strict checkpoint load passed")

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
            scores = q @ kk.transpose(-1, -2) / (head_dim ** 0.5)
            probs = scores.softmax(dim=-1)
            _cap.append(probs.mean(dim=1).detach().float().cpu())
            return _original(hidden_states, *forward_args, **forward_kwargs)

        attention_module.forward = patched
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
                # early checkpoint 调用 encoder 一次；历史 late checkpoint 调用两次。
                # 按 token 数选出包含 K 个寄存器的真实 attention 图。
                images = batch["images"]
                B, Tp1, V = images.shape[:3]
                if k > 0:
                    register_attention = next(
                        (a for a in captured if a.shape[-1] == 1 + 196 + k), None
                    )
                else:
                    register_attention = next((a for a in captured if a.shape[-1] == 197), None)
                if register_attention is None:
                    raise RuntimeError(
                        f"no matching attention captured: sizes={[a.shape[-1] for a in captured]}"
                    )
                grid = int(round((register_attention.shape[-1] - 1 - k) ** 0.5))
                n_patch = register_attention.shape[-1] - 1 - k
                if grid * grid != n_patch:
                    raise RuntimeError(
                        f"token count {register_attention.shape[-1]} patch={n_patch} 非完全平方"
                    )
                for b in range(B):
                    n_valid = int(batch["valid_mask"][b].sum())
                    for t in range(n_valid):
                        for v in range(V):
                            flat = (b * Tp1 + t) * V + v
                            # 帧索引 flat 直接在 batch 维上取（第二程图含全部帧）
                            cls_attn = register_attention[flat, 0].numpy()
                            patch_attn = cls_attn[1: 1 + grid * grid]
                            register_attn = cls_attn[1 + grid * grid:]
                            m = attention_metrics(patch_attn, register_attn, grid)
                            per_sample.append(m)
                print(f"[{name}] batch {b_idx + 1}/{len(batches)} done", flush=True)
        finally:
            attention_module.forward = original_forward

        if not per_sample:
            raise RuntimeError(f"No attention samples were collected for {name}.")
        all_summary[name] = {
            "K": k,
            "n": len(per_sample),
            **{
                f"{key}_mean": float(np.mean([sample[key] for sample in per_sample]))
                for key in per_sample[0]
            },
        }
        print(
            f"[{name}] register_mass={all_summary[name]['register_mass_mean']:.4f} "
            f"patch_top5={all_summary[name]['patch_top5_ratio_mean']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
