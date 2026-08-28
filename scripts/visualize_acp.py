#!/usr/bin/env python3
"""可视化数据集中每帧的 acp_indicator / advantage / value。

从 parquet 直接读取 complementary_info.* 列，为每条 episode 画一张曲线图。
用法:
    python scripts/visualize_acp.py --dataset-root data_toiletButton_merged_0814_with_return \
        --episodes 0 6 40 45            # 省略则画全部
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

VAL = "complementary_info.value"
ADV = "complementary_info.advantage"
ACP = "complementary_info.acp_indicator"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--episodes", nargs="*", type=int, default=None, help="episode ids；默认全部")
    p.add_argument("--output-dir", default=None, help="默认 <root>/acp_visualization")
    args = p.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.output_dir) if args.output_dir else root / "acp_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", VAL, ADV, ACP])
        rows.append((int(t.column("episode_index")[0].as_py()), t))
    rows.sort(key=lambda r: r[0])

    summary = []
    for ep, t in rows:
        if args.episodes is not None and ep not in args.episodes:
            continue
        val = t.column(VAL).to_numpy().ravel().astype(float)
        adv = t.column(ADV).to_numpy().ravel().astype(float)
        acp = t.column(ACP).to_numpy().ravel().astype(int)
        success = ep < 25
        label = "success" if success else "failure"
        frames = np.arange(len(val))

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]
        ax.plot(frames, val, color="#1565C0", lw=1.5, label="predicted value")
        ax.plot(frames, adv, color="#EF6C00", lw=1.0, alpha=0.85, label="n-step advantage")
        ax.axhline(0.0, color="gray", lw=0.6, ls="--")
        # acp=1 的帧用绿色底色高亮
        ax.scatter(frames[acp == 1], val[acp == 1], s=8, color="#2E7D32", zorder=3,
                   label=f"acp_indicator=1 ({acp.mean():.0%})")
        ax.set_ylabel("value / advantage")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title(f"episode {ep} ({label})  frames={len(val)}")
        ax2 = axes[1]
        ax2.fill_between(frames, 0, acp, step="mid", color="#2E7D32", alpha=0.6)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_ylabel("acp")
        ax2.set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(out_dir / f"episode-{ep:03d}-acp.png", dpi=110)
        plt.close(fig)
        summary.append({"episode": ep, "label": label, "frames": int(len(val)),
                        "acp_positive_ratio": float(acp.mean()),
                        "advantage_min": float(adv.min()), "advantage_max": float(adv.max()),
                        "value_min": float(val.min()), "value_max": float(val.max())})
        print(f"[acp-viz] episode {ep} ({label}): acp+={acp.mean():.2%} "
              f"adv=[{adv.min():.3f},{adv.max():.3f}] -> {out_dir}/episode-{ep:03d}-acp.png")

    (out_dir / "acp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[acp-viz] wrote {len(summary)} figures + acp_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
