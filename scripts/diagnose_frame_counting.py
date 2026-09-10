"""帧号计数捷径诊断（纯评估，无训练，§7.5 方案）。

检验 value 模型到底在"看图判进度"还是在"数帧/用位置蒙答案"。
对已训练 checkpoint 在给定数据集上做两类探针：

A. 帧号分桶误差曲线（全量窗口，无扰动）
   按 (帧号占集长比例 f/L, 剩余步数 L-f, 集长 L, 绝对帧号 f) 分桶，
   输出每桶 mse/mae/bias/pearson/return 均值。
   若模型靠"计数/位置"，误差会集中在"时长-进度映射异常"的桶
   （例如集长偏离训练主流分布的集、离集尾/集首很远的位置）。

B. 内容-位置对照扰动（子集窗口，重算前向）
   保持窗口内每个位置的标签不动，只改喂给模型的画面内容：
   B1 视觉回拨：把最后历史帧的画面换成 k 帧前的画面（k=1 和最远）。
      若模型读视觉进度，预测值应随画面回拨向"更早帧的真值"移动
      （斜率 β≈1）；若靠计数/位置，预测值不动（β≈0）。
   B2 端点噪声：给最后历史帧画面加高斯噪声（内容破坏的对照）。
   B3 冻结运动：把整段历史画面都替换成端点帧画面（抹掉全部运动信息）。

用法示例（bash 15_diagnose_frame_counting.sh 封装）：
  CUDA_VISIBLE_DEVICES=0 WCM_EXPECTED_WORLD_SIZE=1 .venv/bin/python -m \
    scripts.diagnose_frame_counting --checkpoint outputs/wcm_baseline_r2/checkpoints/best.pt \
    --dataset-root /media/Data/user/huggingface/lerobot/test/toiletButton_0814_5cut_test_with_return \
    --output-dir outputs/frame_count_diag/baseline
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from world_critic.checkpoint import load_checkpoint_payload
from world_critic.config import apply_runtime_overrides, validate_train_config
from world_critic.data import (
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_processor,
    load_episode_split,
    load_lerobot_dataset,
    validate_action_normalization,
)
from world_critic.distributed import initialize_distributed
from world_critic.model import WorldCriticModel
from world_critic.training import (
    autocast_context,
    config_from_checkpoint_payload,
    move_batch_to_device,
    seed_everything,
    unwrap_model,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3:
        return None
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if math.isfinite(r) else None


def _bucket_table(bucket: np.ndarray, value: np.ndarray, target: np.ndarray) -> list[dict]:
    """每个桶：n / mse / mae / bias / pearson / 桶内真值均值。"""
    rows = []
    for key in np.unique(bucket):
        mask = bucket == key
        err = value[mask] - target[mask]
        rows.append(
            {
                "key": float(key) if np.issubdtype(bucket.dtype, np.number) else str(key),
                "n": int(mask.sum()),
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "pearson": _pearson(value[mask], target[mask]),
                "return_mean": float(np.mean(target[mask])),
            }
        )
    return rows


def _run_forward(model, batch, ctx, config):
    with torch.no_grad(), autocast_context(ctx.device, config.precision):
        return unwrap_model(model)(
            images=batch["images"],
            actions=batch["actions"],
            instruction_input_ids=batch["instruction_input_ids"],
            instruction_attention_mask=batch["instruction_attention_mask"],
            valid_mask=batch["valid_mask"],
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frame-counting shortcut diagnosis (evaluation only).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-root", default=None, help="覆盖 checkpoint 内存储的数据集根目录。")
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None, help="A 阶段最多处理的 batch 数。")
    p.add_argument("--probe-subset", type=int, default=2000, help="B 阶段随机抽样的窗口数（0=不跑扰动）。")
    p.add_argument("--probe-seed", type=int, default=3072)
    p.add_argument("--noise-sigma", type=float, default=0.1, help="B2 噪声强度（相对图像整体 std 的比例）。")
    p.add_argument("--log-every-batches", type=int, default=100)
    return p


def run() -> None:
    args = parser().parse_args()
    ctx = initialize_distributed(1)
    started = time.monotonic()

    def log(msg: str) -> None:
        print(f"[diag][+{time.monotonic() - started:8.1f}s] {msg}", flush=True)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    log(f"loading checkpoint: {checkpoint_path}")
    payload = load_checkpoint_payload(str(checkpoint_path), ctx)
    train_config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    if args.dataset_root:
        train_config.data.root = Path(args.dataset_root).expanduser().resolve()
    validate_train_config(train_config)
    seed_everything(train_config.seed, train_config.deterministic)
    history = train_config.data.history_size
    log(
        f"config ok: history_size={history}, vision={train_config.model.vision.model_name}, "
        f"dataset_root={train_config.data.root}"
    )

    log("loading dataset...")
    dataset = load_lerobot_dataset(train_config.data)
    validate_action_normalization(dataset, train_config.data)

    # 同 evaluate：split=all 时全量；否则按 checkpoint 附近 manifest 划 val/train。
    episode_ids = None
    if args.split != "all":
        checkpoint_dir = checkpoint_path.parent
        candidates = []
        if train_config.data.split_manifest:
            m = Path(train_config.data.split_manifest).expanduser()
            candidates.append(m)
            if not m.is_absolute():
                candidates += [checkpoint_dir / m, checkpoint_dir.parent / m]
        candidates += [checkpoint_dir.parent / "episode_split.json", checkpoint_dir / "episode_split.json"]
        manifest = next((c for c in candidates if c.exists()), None)
        if manifest is None:
            raise FileNotFoundError(f"No episode_split.json for split={args.split}.")
        split = load_episode_split(manifest)
        episode_ids = split.val if args.split == "val" else split.train

    eval_dataset = LeRobotWorldCriticDataset(dataset, train_config.data, episode_ids)
    log(f"temporal windows: {len(eval_dataset)}")
    processor = build_processor(train_config.model)
    collator = WorldCriticCollator(
        processor,
        train_config.model.vision.image_size,
        train_config.model.language.max_length,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=ctx.device.type == "cuda",
        collate_fn=collator,
    )
    model = WorldCriticModel(train_config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(ctx.device).eval().requires_grad_(False)
    log(f"model ready: params={sum(p.numel() for p in model.parameters())}")

    last_idx = history - 1  # value 端点 = 最后历史帧（images 列 index）

    # ---------- A 阶段：全量端点收集（无扰动） ----------
    episodes: dict[int, list] = {}
    ep_list: list[int] = []
    frame_list: list[int] = []
    val_list: list[float] = []
    ret_list: list[float] = []
    processed = 0
    max_batches = args.max_batches
    loader_iter = iter(loader)
    while max_batches is None or processed < max_batches:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        batch = move_batch_to_device(batch, ctx.device)
        out = _run_forward(model, batch, ctx, train_config)
        ret = canonical_ret(batch["return_targets"])  # [B,h]
        v = out.value[:, last_idx, 0]  # [B]
        r = ret[:, last_idx]
        eps = batch["episode_id"].detach().cpu().numpy().tolist()
        frames = batch["frame_indices"][:, last_idx].detach().cpu().numpy().tolist()
        for e, f, vv, rr in zip(
            eps,
            frames,
            v.float().detach().cpu().numpy().tolist(),
            r.float().detach().cpu().numpy().tolist(),
        ):
            ep_list.append(int(e))
            frame_list.append(int(f))
            val_list.append(float(vv))
            ret_list.append(float(rr))
            episodes.setdefault(int(e), []).append(int(f))
        processed += 1
        if processed % args.log_every_batches == 0:
            log(f"A-phase batches={processed} (endpoints={len(ep_list)})")
    n_all = len(ep_list)
    log(f"A-phase done: endpoints={n_all}")

    # 集长估计：端点最大帧号 + 2（端点帧 = start+h-1，start_max = L-(h+1)）
    ep_len = {e: int(np.max(fs)) + 2 for e, fs in episodes.items()}
    f_arr = np.asarray(frame_list, dtype=np.float64)
    v_arr = np.asarray(val_list, dtype=np.float64)
    r_arr = np.asarray(ret_list, dtype=np.float64)
    L_arr = np.asarray([float(ep_len[e]) for e in ep_list], dtype=np.float64)
    frac = f_arr / np.maximum(L_arr, 1.0)
    remaining = L_arr - 1.0 - f_arr
    err_all = v_arr - r_arr

    def bucket_records(key_arr: np.ndarray) -> list[dict]:
        return _bucket_table(key_arr, v_arr, r_arr)

    # 等频分桶（按窗口数分位数）
    def quantile_bins(values: np.ndarray, edges: int = 6) -> np.ndarray:
        qs = np.quantile(values, np.linspace(0, 1, edges + 1))
        qs = np.unique(qs)
        return np.digitize(values, qs[1:-1])

    frame_norm_bins = np.digitize(frac, np.linspace(frac.min(), frac.max(), 11)[1:-1])
    frame_abs_bins = np.digitize(f_arr, np.linspace(f_arr.min(), f_arr.max(), 11)[1:-1])
    remaining_bins = quantile_bins(remaining, 6)
    len_bins = quantile_bins(L_arr, 6)

    bucket_summary = {
        "frame_norm": bucket_records(frame_norm_bins),
        "frame_abs": bucket_records(frame_abs_bins),
        "remaining": bucket_records(remaining_bins),
        "ep_len": bucket_records(len_bins),
    }
    # 计数捷径的标量特征：误差是否与剩余步数 / 时长偏离度相关
    med_L = float(np.median(L_arr))
    corr_err_remaining = _pearson(np.abs(err_all), remaining)
    corr_err_len_dev = _pearson(np.abs(err_all), np.abs(L_arr - med_L))
    overall = {
        "n": n_all,
        "mse": float(np.mean(err_all**2)),
        "mae": float(np.mean(np.abs(err_all))),
        "bias": float(np.mean(err_all)),
        "pearson": _pearson(v_arr, r_arr),
        "episode_count": len(ep_len),
        "corr_abs_err_vs_remaining": corr_err_remaining,
        "corr_abs_err_vs_len_deviation": corr_err_len_dev,
        "median_episode_len": med_L,
        "min_episode_len": float(np.min(L_arr)),
        "max_episode_len": float(np.max(L_arr)),
    }

    # ---------- B 阶段：扰动探针 ----------
    probes: dict = {"n_windows": 0}
    if args.probe_subset > 0 and history >= 2:
        rng = np.random.default_rng(args.probe_seed)
        indices = rng.choice(len(eval_dataset), size=min(args.probe_subset, len(eval_dataset)), replace=False)
        log(f"B-phase: probing {len(indices)} windows with perturbations")
        probe_loader = DataLoader(
            Subset(eval_dataset, indices.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            collate_fn=collator,
        )
        # 探针容器（每窗口一行）
        rows = {k: [] for k in ("ep", "frame", "V0", "R0")}
        for probe_key in ("reg1", "regK", "noise", "freeze"):
            rows[f"dV_{probe_key}"] = []
            rows[f"dR_{probe_key}"] = []
        dR_keys = {1: "reg1", history - 1: "regK"}
        probe_batch = 0
        for batch in probe_loader:
            batch = move_batch_to_device(batch, ctx.device)
            base = batch["images"].clone()
            out0 = _run_forward(model, batch, ctx, train_config)
            ret = canonical_ret(batch["return_targets"])
            v0 = out0.value[:, last_idx, 0].float().detach().cpu()
            r0 = ret[:, last_idx].float().detach().cpu()
            img_std = float(base.std())
            variants = {}
            for k, pkey in dR_keys.items():
                img = base.clone()
                img[:, last_idx] = base[:, last_idx - k]
                variants[pkey] = (img, ret[:, last_idx - k].float().detach().cpu() - r0)
            img = base.clone()
            noise = torch.randn_like(base[:, last_idx]) * (args.noise_sigma * img_std)
            img[:, last_idx] = base[:, last_idx] + noise
            variants["noise"] = (img, torch.zeros_like(r0))
            img = base.clone()
            img[:, : history] = base[:, last_idx : last_idx + 1].expand(-1, history, -1, -1, -1, -1)
            variants["freeze"] = (img, torch.zeros_like(r0))
            for pkey, (img, dR) in variants.items():
                b2 = dict(batch)
                b2["images"] = img
                out_p = _run_forward(model, b2, ctx, train_config)
                vp = out_p.value[:, last_idx, 0].float().detach().cpu()
                rows[f"dV_{pkey}"].extend((vp - v0).numpy().tolist())
                rows[f"dR_{pkey}"].extend(dR.numpy().tolist())
            rows["ep"].extend(batch["episode_id"].detach().cpu().numpy().tolist())
            rows["frame"].extend(batch["frame_indices"][:, last_idx].detach().cpu().numpy().tolist())
            rows["V0"].extend(v0.numpy().tolist())
            rows["R0"].extend(r0.numpy().tolist())
            probe_batch += 1
            if probe_batch % args.log_every_batches == 0:
                log(f"B-phase batches={probe_batch}")
        probes["n_windows"] = len(rows["V0"])

        def probe_stats(pkey: str, *, use_true_diff: bool) -> dict:
            dV = np.asarray(rows[f"dV_{pkey}"])
            dR = np.asarray(rows[f"dR_{pkey}"])
            mask = np.abs(dR) > 1e-3 if use_true_diff else np.ones_like(dR, dtype=bool)
            sel = mask & np.isfinite(dV) & np.isfinite(dR)
            stats = {
                "mean_abs_dV": float(np.mean(np.abs(dV))) if dV.size else None,
                "std_dV": float(np.std(dV)) if dV.size else None,
            }
            if sel.sum() >= 3 and float(np.std(dR[sel])) > 1e-9:
                beta = float(np.dot(dV[sel], dR[sel]) / np.dot(dR[sel], dR[sel]))
                stats["slope_dV_on_dR_true"] = beta  # β≈1 视觉进度；β≈0 计数/位置
                stats["pearson_dV_dR"] = _pearson(dV[sel], dR[sel])
                stats["n_slope"] = int(sel.sum())
            return stats

        probes["reg1"] = probe_stats("reg1", use_true_diff=True)
        probes["regK"] = probe_stats("regK", use_true_diff=True)
        probes["noise"] = probe_stats("noise", use_true_diff=False)
        probes["freeze"] = probe_stats("freeze", use_true_diff=False)
        # B 阶段子集本身的整体质量（对照 A 阶段）
        v0 = np.asarray(rows["V0"])
        r0 = np.asarray(rows["R0"])
        probes["subset_overall"] = {
            "mse": float(np.mean((v0 - r0) ** 2)),
            "pearson": _pearson(v0, r0),
        }
        log(f"B-phase done: probes={probes['n_windows']}")

    result = {
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(train_config.data.root),
        "split": args.split,
        "history_size": history,
        "num_windows": n_all,
        "overall": overall,
        "buckets": bucket_summary,
        "probes": probes,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        _write_plots(out_dir / "buckets.png", bucket_summary, overall)
    except Exception as exc:  # matplotlib 可选
        print(f"[diag] plot skipped: {exc}", flush=True)
    log(f"summary written: {out_dir / 'summary.json'}")
    print(json.dumps(result["overall"], indent=2), flush=True)


def canonical_ret(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    return t[..., 0] if t.ndim == 3 else t


def _write_plots(path: Path, bucket_summary: dict, overall: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    titles = {"frame_norm": "frame / episode_len", "frame_abs": "absolute frame", "remaining": "remaining steps", "ep_len": "episode length"}
    for ax, (key, rows) in zip(axes.ravel(), bucket_summary.items()):
        keys = [r["key"] for r in rows]
        mse = [r["mse"] for r in rows]
        pear = [r["pearson"] if r["pearson"] is not None else np.nan for r in rows]
        n = [r["n"] for r in rows]
        ax2 = ax.twinx()
        ax.plot(keys, mse, "-o", label="mse")
        ax2.plot(keys, pear, "-s", color="tab:orange", label="pearson")
        ax.set_title(f"{titles[key]} (n={sum(n)})")
        ax.set_xlabel("bucket")
        ax.set_ylabel("mse", color="tab:blue")
        ax2.set_ylabel("pearson", color="tab:orange")
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle(f"frame-count diag: n={overall['n']}, mse={overall['mse']:.5f}, pearson={overall['pearson']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
