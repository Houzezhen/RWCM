#!/usr/bin/env python
"""LIBERO 数据集 → WCM 可读的 v2.1 布局（含 return 列）。

从 LIBERO parquet（image 为内嵌 bytes，256x256）读取 episodes，
解码首帧图像保存为 PNG、写出 v2.1 风格 meta 与数据引用，
并按"末帧成功=1、全帧 return 线性衰减"的 toy 约定加 return 列，
供 extract_image_attention_overlay.py 做注意力可视化测试。

产物布局（本脚本只做最小转换，图像存 PNG，不需要视频解码器）：
  libero_datasets/wcm_libero/
    meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
    data/chunk-000/episode_XXXXXX.parquet  （image 列改为 PNG bytes）
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="libero_datasets/datasets")
    parser.add_argument("--dest", default="libero_datasets/wcm_libero")
    parser.add_argument("--num-episodes", type=int, default=5)
    args = parser.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    (dest / "meta").mkdir(parents=True, exist_ok=True)
    (dest / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks = {
        json.loads(line)["task_index"]: json.loads(line)["task"]
        for line in (src / "meta" / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    }

    files = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))[: args.num_episodes]
    index_cursor = 0
    episode_meta = []
    stats_lines = []
    for ep_idx, file in enumerate(files):
        table = pq.read_table(file)
        n = table.num_rows
        task_idx = table["task_index"][0].as_py()

        # return 列：toy 约定 —— 假设每条 demo 都成功，return 从 1 线性衰减到 0
        returns = np.linspace(1.0, 0.0, n, dtype=np.float32)
        success = np.ones(n, dtype=np.int8)

        def decode(cell):
            img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return {"bytes": buf.getvalue(), "path": None}

        out = pa.table(
            {
                "observation.images.image": [decode(c) for c in table["image"].to_pylist()],
                "observation.images.wrist_image": [decode(c) for c in table["wrist_image"].to_pylist()],
                "observation.state": table["state"].to_pylist(),
                "action": table["actions"].to_pylist(),
                "timestamp": table["timestamp"].to_pylist(),
                "frame_index": table["frame_index"].to_pylist(),
                "episode_index": [ep_idx] * n,
                "index": list(range(index_cursor, index_cursor + n)),
                "task_index": [task_idx] * n,
                "return": returns.tolist(),
                "episode_success": success.tolist(),
            }
        )
        pq.write_table(out, dest / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        episode_meta.append(
            {"episode_index": ep_idx, "tasks": [tasks[task_idx]], "length": n}
        )
        stats_lines.append(json.dumps({"episode_index": ep_idx, "stats": {}}))
        index_cursor += n
        print(f"[convert] episode {ep_idx}: {n} frames, task={tasks[task_idx]!r}", flush=True)

    total_frames = index_cursor
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": len(files),
        "total_frames": total_frames,
        "total_tasks": len({m["tasks"][0] for m in episode_meta}),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": f"0:{len(files)}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [7], "names": None},
            "observation.state": {"dtype": "float32", "shape": [8], "names": None},
            "observation.images.image": {"dtype": "image", "shape": [256, 256, 3], "names": None},
            "observation.images.wrist_image": {"dtype": "image", "shape": [256, 256, 3], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "return": {"dtype": "float32", "shape": [1], "names": None},
            "episode_success": {"dtype": "int8", "shape": [1], "names": None},
        },
    }
    (dest / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (dest / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": k, "task": v}) for k, v in sorted(tasks.items())) + "\n"
    )
    (dest / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(m) for m in episode_meta) + "\n"
    )
    (dest / "meta" / "episodes_stats.jsonl").write_text("\n".join(stats_lines) + "\n")
    print(f"[convert] done: {len(files)} episodes, {total_frames} frames -> {dest}", flush=True)


if __name__ == "__main__":
    main()
