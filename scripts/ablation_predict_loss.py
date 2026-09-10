#!/usr/bin/env python
"""预测 loss 消融：4 组模型在真机数据集（toiletButton）上对比 value / next-state 预测误差。

组别：
  A. original        ：HF 原始预训练权重（ViT+CLIP 均未训练）
  B. original+reg    ：HF 原始权重 + K 个随机初始化 register token
  C. finetuned       ：checkpoint 训练权重（deploy.pt）
  D. finetuned+reg   ：checkpoint 训练权重 + K 个随机 register token
                        （注：骨干没在 register 下重训，考察的是结构即时影响）

指标（与训练 loss 同口径）：
  value_mse   ：V(s) vs 数据集 return 列
  value_mae / pearson
  latent_mse  ：动作条件动力学预测的下一状态 latent vs 真实下一帧 latent（detach）

用法：
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/ablation_predict_loss.py \
      --checkpoint outputs/wcm_toiletButton_merged_0814/deploy.pt \
      --dataset-root data_toiletButton_merged_0814_with_return
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
    parser.add_argument("--checkpoint", default="outputs/wcm_toiletButton_merged_0814/deploy.pt")
    parser.add_argument("--dataset-root", default="data_toiletButton_merged_0814_with_return")
    parser.add_argument("--num-register", type=int, default=4, help="B/D 组的 register token 数")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="outputs/predict_loss_ablation/summary.json")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def evaluate(model, loader, device, autocast_ctx) -> dict[str, float]:
    se, ae = [], []
    pred_all, tgt_all = [], []
    latent_se = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            with autocast_ctx:
                out = model(
                    images=batch["images"],
                    actions=batch["actions"],
                    instruction_input_ids=batch["instruction_input_ids"],
                    instruction_attention_mask=batch["instruction_attention_mask"],
                    valid_mask=batch["valid_mask"],
                )
            mask = out.valid_mask.bool().unsqueeze(-1)
            v = out.value.float()[mask]
            t = batch["return_targets"].float()[mask]
            se.append(((v - t) ** 2).sum().item())
            ae.append((v - t).abs().sum().item())
            pred_all.append(v.cpu())
            tgt_all.append(t.cpu())
            # latent 一致性：detach 目标，与训练 next_state loss 同口径
            d = (out.next_state_pred.float() - out.target_next_state.float()) ** 2
            d = (d.sum(-1) * out.valid_mask.bool().float()).sum().item()
            latent_se.append(d)
    pred = torch.cat(pred_all)
    tgt = torch.cat(tgt_all)
    vp = pred.mean() - pred.mean()
    pearson = float(np.corrcoef(pred.numpy(), tgt.numpy())[0, 1]) if len(pred) > 2 else float("nan")
    n = len(pred)
    return {
        "n": n,
        "value_mse": float(np.sum(se) / n),
        "value_mae": float(np.sum(ae) / n),
        "value_pearson": pearson,
        "latent_mse": float(np.sum(latent_se) / n),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
    train_config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    train_config.data.root = args.dataset_root

    dataset = load_lerobot_dataset(train_config.data)
    episode_ids = episode_ids_from_dataset(dataset)
    eval_dataset = LeRobotWorldCriticDataset(dataset, train_config.data, episode_ids)
    processor = build_processor(train_config.model)
    collator = WorldCriticCollator(
        processor,
        train_config.model.vision.image_size,
        train_config.model.language.max_length,
    )
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if train_config.precision == "bf16" and device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    results = {}
    groups = [
        ("original", False, 0),
        (f"original+reg{args.num_register}", False, args.num_register),
        ("finetuned", True, 0),
        (f"finetuned+reg{args.num_register}", True, args.num_register),
    ]
    for name, load_ckpt, k_reg in groups:
        train_config.model.vision.num_register_tokens = k_reg
        model = WorldCriticModel(train_config.model)
        if load_ckpt:
            # finetuned+reg：加载全部训练权重，register_tokens 随机初始化保留
            state = dict(payload["model"])
            state.pop("vision_encoder.register_tokens", None)
            model.load_state_dict(state, strict=False)
        else:
            # 只加载非 vision 部分（language 冻结可直接用），
            # vision 保持 HF 原始权重 = 原始 backbone。
            missing, unexpected = model.load_state_dict(
                {kk: vv for kk, vv in payload["model"].items()
                 if not kk.startswith("vision_encoder.")},
                strict=False,
            )
            assert not unexpected, unexpected
            assert all("register" in m or m.startswith("vision_encoder.") for m in missing), missing
        model.to(device).eval().requires_grad_(False)
        started = time.monotonic()
        results[name] = evaluate(model, loader, device, autocast_ctx)
        results[name]["elapsed_s"] = round(time.monotonic() - started, 1)
        print(f"[loss-abl] {name}: {json.dumps(results[name])}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[loss-abl] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
