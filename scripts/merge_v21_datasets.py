#!/usr/bin/env python3
"""Merge multiple LeRobot v2.1 datasets into a single v2.1 dataset.

Episodes are renumbered (global ``index`` and ``episode_index``), videos are
hard-linked and renamed, per-chunk layout follows v2.1 conventions.  Success
datasets are given via --success (subset of --datasets); episodes of those
datasets get label 1, everything else label 0.  A ``success_labels.json`` is
written into the output root so the downstream scripts can consume it.

Usage:
    python scripts/merge_v21_datasets.py \
        --datasets /path/success_0717 /path/failure_0717_f_old \
        --success /path/success_0717 \
        --output /path/data_toiletButton_0717_merged
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# frames per chunk (matches the source datasets' chunks_size)
CHUNK_FRAME_CAP = 1000


def collect_episode_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))


def episode_index_from_path(path: Path) -> int:
    stem = path.stem  # episode_000042
    return int(stem[len("episode_"):])


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="v2.1 dataset roots, in merge order (success first)")
    parser.add_argument("--success", nargs="+", default=[],
                        help="subset of --datasets whose episodes are successes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunks-size", type=int, default=CHUNK_FRAME_CAP)
    args = parser.parse_args()

    if len(args.datasets) < 2:
        raise ValueError("Need at least two datasets to merge.")
    for s in args.success:
        if s not in args.datasets:
            raise ValueError(f"--success dataset {s} is not in --datasets")

    success_set = set(args.success)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    datasets = [Path(p) for p in args.datasets]
    episodes: list[tuple[Path, Path, bool]] = []  # (parquet, dataset_root, is_success)
    for ds in datasets:
        files = collect_episode_files(ds)
        if not files:
            raise ValueError(f"No v2.1 episode parquet files in {ds}")
        ok = str(ds) in success_set
        episodes.extend((f, ds, ok) for f in files)

    total_frames = sum(pq.read_metadata(f).num_rows for f, _, _ in episodes)
    print(f"Episodes: {len(episodes)}  Frames: {total_frames}")

    # --- data + videos -----------------------------------------------------
    # v2.1 chunk convention: chunk = episode_index // chunks_size (chunks_size
    # is the number of episode *indices* per chunk, not frames).
    new_episode_index = 0
    global_index = 0
    success_labels: dict[int, int] = {}
    info_counts = {"total_episodes": 0, "total_frames": 0, "total_videos": 0}

    for parquet_path, ds_root, is_success in episodes:
        table = pq.read_table(parquet_path)
        n = table.num_rows

        new_ep = new_episode_index
        chunk = new_ep // args.chunks_size
        table = (
            table
            .set_column(table.schema.get_field_index("index"),
                        "index", pa.array(range(global_index, global_index + n), type=pa.int64()))
            .set_column(table.schema.get_field_index("episode_index"),
                        "episode_index", pa.array([new_ep] * n, type=pa.int64()))
        )
        out_parquet = output / "data" / f"chunk-{chunk:03d}" / f"episode_{new_ep:06d}.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_parquet, compression="snappy")

        old_ep = episode_index_from_path(parquet_path)
        for cam_dir in sorted((ds_root / "videos").glob("chunk-*/*")):
            if not cam_dir.is_dir():
                continue
            src_mp4 = cam_dir / f"episode_{old_ep:06d}.mp4"
            if src_mp4.exists():
                hardlink_or_copy(
                    src_mp4,
                    output / "videos" / f"chunk-{chunk:03d}" / cam_dir.name / f"episode_{new_ep:06d}.mp4",
                )

        success_labels[new_ep] = 1 if is_success else 0
        global_index += n
        new_episode_index += 1
        info_counts["total_episodes"] += 1
        info_counts["total_frames"] += n
        info_counts["total_videos"] += sum(
            1 for _ in (ds_root / "videos").glob(f"chunk-*/*/episode_{old_ep:06d}.mp4")
        )
    total_chunks = (new_episode_index - 1) // args.chunks_size + 1

    # --- meta --------------------------------------------------------------
    meta_src = Path(args.datasets[0]) / "meta"
    out_meta = output / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)

    info = json.loads((meta_src / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = info_counts["total_episodes"]
    info["total_frames"] = info_counts["total_frames"]
    info["total_videos"] = info_counts["total_videos"]
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{info_counts['total_episodes']}"}
    (out_meta / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    # episodes.jsonl: keep each dataset's rows, renumber episode_index
    episodes_rows: list[dict] = []
    for ds in datasets:
        rows = [json.loads(line) for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines() if line.strip()]
        episodes_rows.extend(rows)
    for i, row in enumerate(episodes_rows):
        row["episode_index"] = i
    (out_meta / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in episodes_rows) + "\n", encoding="utf-8")

    # episodes_stats.jsonl: renumber episode_index, keep stats
    stats_rows: list[dict] = []
    for ds in datasets:
        rows = [json.loads(line) for line in (ds / "meta" / "episodes_stats.jsonl").read_text().splitlines() if line.strip()]
        stats_rows.extend(rows)
    for i, row in enumerate(stats_rows):
        row["episode_index"] = i
    (out_meta / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in stats_rows) + "\n", encoding="utf-8")

    # tasks.jsonl: all source datasets share the same task; take the first one
    shutil.copy2(meta_src / "tasks.jsonl", out_meta / "tasks.jsonl")

    # success labels for downstream scripts
    labels_path = output / "success_labels.json"
    labels_path.write_text(
        json.dumps({str(k): v for k, v in sorted(success_labels.items())}, indent=2) + "\n", encoding="utf-8")

    print(f"Done. Output: {output}")
    print(f"  episodes={info_counts['total_episodes']} frames={info_counts['total_frames']} "
          f"videos={info_counts['total_videos']} chunks={total_chunks}")
    print(f"  success_labels.json written ({sum(success_labels.values())} success / "
          f"{len(success_labels) - sum(success_labels.values())} failure)")


if __name__ == "__main__":
    main()
