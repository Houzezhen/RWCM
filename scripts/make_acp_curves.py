#!/usr/bin/env python3
"""从数据集 parquet 生成 acp/advantage 曲线 JSON，供 episode_value_video render 使用。

复用 world_critic.evaluate 的 curves 格式（list of {episode_id, frame_indices, values, returns, metrics}），
values 可选 advantage 或 acp_indicator。

用法:
    python scripts/make_acp_curves.py \
        --dataset-root data_toiletButton_merged_0814_with_return \
        --mode advantage \
        --output acp_curves.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

VAL = "complementary_info.value"
ADV = "complementary_info.advantage"
ACP = "complementary_info.acp_indicator"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--mode", choices=("advantage", "acp", "value"), default="advantage")
    p.add_argument("--output", required=True, help="输出 curves JSON 路径")
    args = p.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {root}/data")

    col = {"advantage": ADV, "acp": ACP, "value": VAL}[args.mode]
    curves = []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", VAL, ADV, ACP])
        episode = int(t.column("episode_index")[0].as_py())
        frames = t.column("frame_index").to_numpy().astype(int).tolist()
        values = t.column(col).to_numpy().ravel().astype(float).tolist()
        tgt = t.column(VAL).to_numpy().ravel().astype(float).tolist()
        pred = np.asarray(values)
        ref = np.asarray(tgt)
        metrics = {
            "count": len(values),
            "value_mse": float(np.mean((pred - ref) ** 2)) if args.mode == "value" else float("nan"),
            "value_rmse": float("nan"),
            "value_mae": float("nan"),
            "value_pearson": float("nan"),
        }
        curves.append({
            "episode_id": episode,
            "frame_indices": frames,
            "values": values,
            "returns": tgt,
            "metrics": metrics,
        })
        print(f"[acp-curves] episode {episode}: frames={len(values)} mode={args.mode}")

    out = Path(args.output).expanduser().resolve()
    out.write_text(json.dumps(curves), encoding="utf-8")
    print(f"[acp-curves] wrote {len(curves)} episodes to {out}")


if __name__ == "__main__":
    main()
