#!/usr/bin/env python3
"""Add pi*0.6-style return fields directly to a LeRobot v2.1 dataset.

Unlike ``add_returns_to_lerobot.py`` (which always migrates to v3.0), this
tool keeps the dataset in v2.1: it writes ``return`` / ``return_raw`` /
``episode_success`` columns into each per-episode parquet file and syncs
``meta/info.json`` features and ``meta/episodes_stats.jsonl`` on a copy of the
source dataset.  Files that are modified are copied for real; read-only files
(videos, other parquet, ...) are hard-linked when possible to avoid duplicating
data on disk.

Reuses the canonical return computation from ``add_returns_to_lerobot`` so the
training values are identical regardless of the on-disk format.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .add_returns_to_lerobot import (
        AUTO_SUCCESS_KEYS,
        as_scalar,
        compute_pi06_returns,
        dataset_root,
        detect_lerobot_version,
        episode_success,
        load_success_labels,
        strict_bool,
    )
except ImportError:  # running directly as a script
    from add_returns_to_lerobot import (
        AUTO_SUCCESS_KEYS,
        as_scalar,
        compute_pi06_returns,
        dataset_root,
        detect_lerobot_version,
        episode_success,
        load_success_labels,
        strict_bool,
    )

# Field name -> (info.json dtype string, pyarrow type).  Matches the v3 writer.
RETURN_FIELDS = {
    "return": ("float32", pa.float32()),
    "return_raw": ("float32", pa.float32()),
    "episode_success": ("int8", pa.int8()),
}

_MODIFIABLE_NAMES = {"info.json", "episodes_stats.jsonl"}


def _copy_one(src: str | Path, dst: str | Path) -> None:
    """Copy a file; hard-link read-only files, really copy anything we edit."""
    src = Path(src)
    dst = Path(dst)
    name = src.name
    if name in _MODIFIABLE_NAMES or (name.startswith("episode_") and name.endswith(".parquet")):
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_dataset_tree(source: Path, output: Path) -> None:
    shutil.copytree(source, output, copy_function=_copy_one, dirs_exist_ok=False)


def load_episodes_jsonl(path: Path) -> list[dict]:
    episodes: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    return episodes


def load_tasks_jsonl(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            tasks[int(obj["task_index"])] = str(obj["task"])
    return tasks


def episode_to_task_index(episodes: list[dict], tasks: dict[int, str]) -> dict[int, int]:
    task_by_name = {task: idx for idx, task in tasks.items()}
    mapping: dict[int, int] = {}
    for ep in episodes:
        episode_index = int(ep["episode_index"])
        task_names = ep.get("tasks") or []
        if not task_names:
            raise ValueError(f"episode_index={episode_index} has no tasks entry.")
        name = str(task_names[0])
        if name not in task_by_name:
            raise ValueError(f"episode_index={episode_index} references unknown task {name!r}.")
        mapping[episode_index] = task_by_name[name]
    return mapping


def collect_episode_files(data_dir: Path) -> list[tuple[Path, int, int, int]]:
    """Return (file, episode_index, nrows, first_global_index) sorted by global order."""
    files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v2.1 episode parquet files under {data_dir}.")
    records: list[tuple[Path, int, int, int]] = []
    for f in files:
        table = pq.read_table(f, columns=["episode_index", "index"])
        episode = int(table.column("episode_index")[0].as_py())
        idx = table.column("index").to_numpy()
        records.append((f, episode, table.num_rows, int(idx[0])))
    records.sort(key=lambda rec: rec[3])
    return records


def load_columns(
    records: list[tuple[Path, int, int, int]], columns: list[str]
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for f, _ep, _n, _start in records:
        table = pq.read_table(f, columns=columns)
        for c in columns:
            result[c].append(table.column(c).to_numpy(zero_copy_only=False))
    return {c: np.concatenate(parts) for c, parts in result.items()}


def available_columns(records: list[tuple[Path, int, int, int]]) -> set[str]:
    names: set[str] = set()
    for f, _ep, _n, _start in records:
        names.update(pq.read_schema(f).names)
    return names


def load_success_column(
    records: list[tuple[Path, int, int, int]], key: str
) -> dict[int, bool]:
    values = load_columns(records, [key])[key]
    success_by_episode: dict[int, bool] = {}
    episode_index = load_columns(records, ["episode_index"])["episode_index"]
    for episode in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == episode)
        success_by_episode[int(episode)] = episode_success(
            values[rows], "last"
        )
    return success_by_episode


def write_columns(
    records: list[tuple[Path, int, int, int]],
    raw_returns: np.ndarray,
    returns: np.ndarray,
    success_rows: np.ndarray,
) -> None:
    offset = 0
    for f, _ep, nrows, _start in records:
        sl = slice(offset, offset + nrows)
        offset += nrows
        table = pq.read_table(f)
        new_table = (
            table.append_column(
                "return",
                pa.array(np.ascontiguousarray(returns[sl]).ravel(), type=pa.float32()),
            )
            .append_column(
                "return_raw",
                pa.array(np.ascontiguousarray(raw_returns[sl]).ravel(), type=pa.float32()),
            )
            .append_column(
                "episode_success",
                pa.array(success_rows[sl].astype(np.int8), type=pa.int8()),
            )
        )
        pq.write_table(new_table, f, compression="snappy")


def update_info_json(meta_dir: Path) -> None:
    path = meta_dir / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    for name, (dtype, _arrow) in RETURN_FIELDS.items():
        if name in features:
            raise ValueError(
                f"Source already contains feature {name!r}; choose a clean source dataset."
            )
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    info["features"] = features
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def update_episodes_stats(
    meta_dir: Path,
    records: list[tuple[Path, int, int, int]],
    raw_returns: np.ndarray,
    returns: np.ndarray,
    success_rows: np.ndarray,
) -> None:
    path = meta_dir / "episodes_stats.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    per_episode: dict[int, dict] = {}
    offset = 0
    for f, episode, nrows, _start in records:
        sl = slice(offset, offset + nrows)
        offset += nrows
        stats = per_episode.setdefault(episode, {})
        for key, values in (
            ("return", returns[sl]),
            ("return_raw", raw_returns[sl]),
            ("episode_success", success_rows[sl]),
        ):
            vals = np.asarray(values, dtype=np.float64).ravel()
            stats[key] = {
                "min": [float(vals.min())],
                "max": [float(vals.max())],
                "mean": [float(vals.mean())],
                "std": [float(vals.std()) if vals.size > 1 else 0.0],
                "count": [int(vals.size)],
            }
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        episode = int(obj["episode_index"])
        if episode in per_episode:
            obj["stats"] = {**obj["stats"], **per_episode[episode]}
        out_lines.append(json.dumps(obj))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def validate_output_dir(output: Path, source: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"Output directory {output} already exists. Use a completely new path."
        )
    try:
        output.resolve().relative_to(source.resolve())
        raise ValueError("Output directory must not be inside the source dataset.")
    except ValueError as exc:
        if str(exc).startswith("Output directory"):
            raise
        # not nested: fine


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add pi*0.6-style return fields to a LeRobot v2.1 dataset (output stays v2.1)."
    )
    p.add_argument("--root", required=True, help="dataset root, or parent when --repo-id is given")
    p.add_argument("--repo-id", default="", help="repository id resolved under --root")
    p.add_argument("--output-dir", required=True, help="new v2.1 dataset directory (must not exist)")
    p.add_argument("--success-labels", default=None, help="JSON mapping episode_index -> bool")
    p.add_argument("--success-key", default=None, help="per-frame success column name in the parquet")
    p.add_argument(
        "--success-reduction",
        default="last",
        choices=("last", "any", "constant"),
        help="how to reduce a per-frame success column to an episode label (default: last)",
    )
    p.add_argument("--failure-penalty", type=float, default=None)
    p.add_argument(
        "--normalization",
        default="task_max",
        choices=("task_max", "global_minmax", "none"),
    )
    p.add_argument("--step-scale", type=float, default=1.0)
    p.add_argument("--step-key", default="frame_index")
    return p


def main() -> None:
    args = parser().parse_args()
    source = dataset_root(args.root, args.repo_id)
    version = detect_lerobot_version(source)
    if version != "v2.1":
        raise ValueError(
            f"Expected a v2.1 dataset, got {version!r} at {source}. "
            "This tool only edits v2.1 layouts in place."
        )

    output = Path(args.output_dir).expanduser().resolve()
    validate_output_dir(output, source)

    meta_dir = source / "meta"
    data_dir = source / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    # 1. Copy the dataset; only files we will edit are really copied.
    copy_dataset_tree(source, output)

    # 2. Load v2.1 metadata from the output copy.
    out_meta = output / "meta"
    out_data = output / "data"
    episodes = load_episodes_jsonl(out_meta / "episodes.jsonl")
    tasks = load_tasks_jsonl(out_meta / "tasks.jsonl")
    ep_to_task = episode_to_task_index(episodes, tasks)
    records = collect_episode_files(out_data)

    # 3. Global row arrays (episode order follows the global index column).
    columns = load_columns(records, ["episode_index", args.step_key, "task_index"])
    episode_index = columns["episode_index"].astype(np.int64).reshape(-1)
    step_index = columns[args.step_key].astype(np.int64).reshape(-1)
    parquet_task_index = columns["task_index"].astype(np.int64).reshape(-1)

    # The v2.1 parquet already carries task_index per row; trust it when it
    # agrees with tasks.jsonl, otherwise use the metadata mapping.
    task_index = parquet_task_index.copy()
    for episode in np.unique(episode_index):
        expected = ep_to_task.get(int(episode))
        if expected is None:
            raise ValueError(f"episode_index={int(episode)} missing from episodes.jsonl.")
        rows = np.flatnonzero(episode_index == episode)
        if not np.all(parquet_task_index[rows] == expected):
            raise ValueError(
                f"episode_index={int(episode)}: parquet task_index disagrees with tasks.jsonl."
            )

    # 4. Success labels.
    success_by_episode: dict[int, bool]
    if args.success_labels:
        success_by_episode = load_success_labels(args.success_labels)
    else:
        candidates = [args.success_key] if args.success_key else list(AUTO_SUCCESS_KEYS)
        available = available_columns(records)
        success_key = next((c for c in candidates if c in available), None)
        if success_key is None:
            raise ValueError(
                "No trustworthy success label found. Pass --success-labels or --success-key; "
                "the script will not infer success from episode termination."
            )
        column_values = load_columns(records, [success_key])[success_key]
        success_by_episode = {}
        for episode in np.unique(episode_index):
            rows = np.flatnonzero(episode_index == episode)
            success_by_episode[int(episode)] = episode_success(
                column_values[rows], args.success_reduction
            )

    # 5. Returns (identical computation to the v3 writer).
    raw_returns, returns, task_max_length = compute_pi06_returns(
        episode_index,
        task_index,
        success_by_episode,
        args.failure_penalty,
        normalization=args.normalization,
        step_index=step_index,
        step_scale=args.step_scale,
    )

    success_rows = np.asarray(
        [success_by_episode[int(ep)] for ep in episode_index], dtype=bool
    )

    # 6. Write columns + metadata on the copy.
    write_columns(records, raw_returns, returns, success_rows)
    update_info_json(out_meta)
    update_episodes_stats(out_meta, records, raw_returns, returns, success_rows)

    metadata = {
        "schema_version": 2,
        "source": {"root": str(source), "detected_version": version},
        "canonical_training_format": "v2.1",
        "conversion_steps": [],
        "definition": {
            "nonterminal_reward": -float(args.step_scale),
            "success_terminal_reward": 0.0,
            "failure_terminal_reward": "-failure_penalty",
            "failure_penalty": (
                args.failure_penalty if args.failure_penalty is not None else "task_max_episode_length"
            ),
            "discount": 1.0,
            "normalization": args.normalization,
            "step_scale": args.step_scale,
        },
        "success_source": args.success_key
        or (str(Path(args.success_labels).resolve()) if args.success_labels else "auto-detected column"),
        "success_reduction": args.success_reduction if args.success_key else "episode_json",
        "task_max_episode_length": {str(key): value for key, value in task_max_length.items()},
        "num_success_episodes": int(sum(success_by_episode.values())),
        "num_failure_episodes": int(len(success_by_episode) - sum(success_by_episode.values())),
    }
    (out_meta / "return_definition.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # 7. Verify the output.
    verified = collect_episode_files(out_data)
    verified_names = available_columns(verified)
    missing = [name for name in RETURN_FIELDS if name not in verified_names]
    if missing:
        raise RuntimeError(f"Output is missing columns {missing}.")
    total_rows = sum(rec[2] for rec in verified)
    print(f"Source dataset   : {source}")
    print(f"Output dataset   : {output}")
    print(f"Episodes         : {len(records)} (success={metadata['num_success_episodes']}, "
          f"failure={metadata['num_failure_episodes']})")
    print(f"Total frames     : {total_rows}")
    print(f"Columns written  : {', '.join(RETURN_FIELDS)}")
    print(f"Normalization    : {args.normalization}")
    print(f"task_max_length  : {metadata['task_max_episode_length']}")
    sample = verified[0]
    row = pq.read_table(sample[0]).column("return").to_numpy()[0]
    print(f"Sample (ep {sample[1]}) first frame return = {row:.4f}")


if __name__ == "__main__":
    main()
