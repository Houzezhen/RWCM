#!/usr/bin/env python3
"""Convert a WCM-labeled LeRobot v2.1 dataset into the pistar RECAP standard form.

What it does (output is still a v2.1 dataset, pistar needs zero code changes):
  - Rename columns to the pistar convention:
      observation.images.primary -> image      (video feature)
      observation.images.wrist   -> wrist_image (video feature)
      observation.state          -> state
      action                     -> actions
    Videos are hard-linked and the per-camera folder renamed accordingly.
  - Add the pistar RECAP columns (metadata matches pistar's backfill_lerobot_columns):
      value_label   float32, from the WCM return column (clipped to [-1, 0]);
                    with --success-labels the last frame is forced to 0 (success) / -1 (fail)
      reward        float32, last frame 1.0 else 0.0 (all-zero for failed episodes)
      reward_label  float32, -1/T except last frame = value_label[-1]
      intervention  int64,   all 1 (demo) by default, configurable
      adv_ind       string,  per-frame "positive"/"negative": 优先取 WCM 的
                    complementary_info.acp_indicator（advantage 分位数二值化，1->"positive", 0->"negative"）；
                    无该列时退化为按成功/失败标记（--adv-ind-success/--adv-ind-fail）
  - Keep the original WCM columns (return / return_raw / episode_success /
    complementary_info.*) untouched.
  - Fix the parquet arrow-schema metadata: v2.1 "List" feature type is replaced
    by "Sequence" so pistar's newer `datasets` library can load the files.
  - Sync meta/info.json features and meta/episodes_stats.jsonl.

Usage:
    python scripts/convert_v21_to_pistar.py \
        --root /path/to/wcm_v21_dataset \
        --output /path/to/pistar_dataset \
        [--success-labels /path/to/success_labels.json]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Feature name -> pistar backfill_lerobot_columns metadata (info.json + dtype/arrow type).
# 与 pistar/control_your_robot/scripts/backfill_lerobot_columns.py 的 TARGET_FEATURES 对齐。
TARGET_FEATURES = {
    "adv_ind": {
        "dtype": "string",
        "shape": [1],
        "names": ["adv_ind"],
        "pa_type": pa.string(),
    },
    "value_label": {
        "dtype": "float32",
        "shape": [1],
        "names": ["value_label"],
        "pa_type": pa.float32(),
    },
    "reward": {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward"],
        "pa_type": pa.float32(),
    },
    "reward_label": {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward_label"],
        "pa_type": pa.float32(),
    },
    "intervention": {
        "dtype": "int64",
        "shape": [1],
        "names": ["intervention_flag"],
        "pa_type": pa.int64(),
    },
}
# 与 pistar 的 COLUMN_WRITE_ORDER 对齐
WRITE_ORDER = ("adv_ind", "value_label", "reward", "reward_label", "intervention")
NUMERIC_TARGETS = ("value_label", "reward", "reward_label", "intervention")
ACP_INDICATOR_COL = "complementary_info.acp_indicator"

# v2.1 column -> pistar column
RENAME_COLUMNS = {
    "observation.images.primary": "image",
    "observation.images.wrist": "wrist_image",
    "observation.state": "state",
    "action": "actions",
}
# Video folders to rename: (old folder name, new folder name)
VIDEO_FOLDER_RENAMES = [
    ("observation.images.primary", "image"),
    ("observation.images.wrist", "wrist_image"),
]

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


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def load_success_labels(path: Path | None) -> dict[int, bool] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"--success-labels must map episode_index -> 0/1, got {type(data)}")
    out: dict[int, bool] = {}
    for k, v in data.items():
        try:
            ep = int(k)
        except (TypeError, ValueError):
            continue  # skip non-episode keys if any
        if isinstance(v, (list, tuple, np.ndarray)):
            out[ep] = bool(np.asarray(v).reshape(-1)[0])
        else:
            out[ep] = bool(v)
    return out


def compute_column_values(
    num_rows: int,
    episode_index: int,
    return_values: np.ndarray,
    success_labels: dict[int, bool] | None,
    *,
    intervention: int,
    adv_ind_success: str,
    adv_ind_fail: str,
    acp_indicators: np.ndarray | None,
) -> dict[str, np.ndarray]:
    success = success_labels.get(episode_index, True) if success_labels is not None else True

    value_label = np.clip(np.asarray(return_values, dtype=np.float32), -1.0, 0.0)
    if success_labels is not None:
        value_label[-1] = 0.0 if success else -1.0

    reward = np.zeros(num_rows, dtype=np.float32)
    if success:
        reward[-1] = 1.0

    reward_label = np.full(num_rows, -1.0 / num_rows, dtype=np.float32)
    reward_label[-1] = value_label[-1]

    # adv_ind：优先用 WCM 的 acp_indicator（advantage 分位数二值化，1->"positive"，
    # 0->"negative"，即 pistar label_advantage_from_vlm 的逐帧语义）；
    # 无该列时退化为按成功/失败标记。
    if acp_indicators is not None:
        adv_ind = np.where(acp_indicators.reshape(-1) > 0.5, adv_ind_success, adv_ind_fail)
    else:
        adv_ind = np.full(num_rows, adv_ind_success if success else adv_ind_fail, dtype=object)

    return {
        "value_label": value_label.astype(np.float32),
        "reward": reward,
        "reward_label": reward_label,
        "intervention": np.full(num_rows, intervention, dtype=np.int64),
        "adv_ind": adv_ind.astype(object),
    }


def build_feature_meta_for_schema(table: pa.Table) -> dict:
    """Rebuild the arrow-schema "info.features" dict for the parquet columns
    (v2.1 "List" -> "Sequence" so newer `datasets` can parse the files)."""
    features: dict[str, dict] = {}
    for col_name in table.column_names:
        col = table.column(col_name)
        typ = col.type
        if pa.types.is_list(typ) or pa.types.is_fixed_size_list(typ):
            value_type = typ.value_type
            if pa.types.is_fixed_size_list(typ):
                length = typ.list_size
            else:
                length = len(col[0]) if len(col) else 0
            features[col_name] = {
                "feature": {"dtype": str(value_type), "_type": "Value"},
                "length": int(length),
                "_type": "Sequence",
            }
        elif pa.types.is_string(typ) or pa.types.is_large_string(typ):
            features[col_name] = {"dtype": "string", "_type": "Value"}
        else:
            features[col_name] = {"dtype": str(typ), "_type": "Value"}
    return features


def rewrite_parquet(
    src: Path,
    dst: Path,
    column_values: dict[str, np.ndarray],
) -> None:
    table = pq.read_table(src)
    table = table.rename_columns([RENAME_COLUMNS.get(c, c) for c in table.column_names])

    for col_name in WRITE_ORDER:
        pa_type = TARGET_FEATURES[col_name]["pa_type"]
        values = column_values[col_name]
        if col_name == "adv_ind":
            array = pa.array(values.tolist() if isinstance(values, np.ndarray) else list(values), type=pa_type)
        else:
            array = pa.array(values, type=pa_type)
        table = table.append_column(col_name, array)

    # Fix / rebuild the parquet arrow-schema metadata.
    meta = dict(table.schema.metadata) if table.schema.metadata else {}
    try:
        hf = json.loads(meta.get(b"huggingface", b"{}")) if meta.get(b"huggingface") else {}
    except (TypeError, json.JSONDecodeError):
        hf = {}
    info = hf.get("info") if isinstance(hf, dict) else None
    if not isinstance(info, dict):
        info = {}
    info["features"] = build_feature_meta_for_schema(table)
    hf["info"] = info
    meta[b"huggingface"] = json.dumps(hf).encode()

    schema = table.schema.with_metadata(meta)
    tmp = dst.with_name(dst.name + ".tmpwrite")
    pq.write_table(table.cast(schema), tmp, compression="snappy")
    tmp.replace(dst)


def numeric_stats(values: np.ndarray) -> dict[str, list]:
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [int(len(values))],
    }


def update_episodes_stats(stats_path: Path, parquet_files: list[Path]) -> None:
    rows = load_jsonl(stats_path)
    by_episode = {int(row["episode_index"]): row for row in rows}

    numeric = [c for c in NUMERIC_TARGETS]
    for f in parquet_files:
        episode_index = int(f.stem[len("episode_"):])
        row = by_episode[episode_index]
        stats = row["stats"]

        renamed = {}
        for key, value in stats.items():
            renamed[RENAME_COLUMNS.get(key, key)] = value
        stats = renamed

        table = pq.read_table(f, columns=numeric)
        for col_name in numeric:
            values = np.asarray(table.column(col_name).to_pylist(), dtype=np.float32)
            if col_name == "intervention":
                values = np.asarray(table.column(col_name).to_pylist(), dtype=np.int64)
            stats[col_name] = numeric_stats(values)

        row["stats"] = stats

    save_jsonl(stats_path, rows)


def update_info_json(info_path: Path) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features: dict = {}
    for key, value in info.get("features", {}).items():
        features[RENAME_COLUMNS.get(key, key)] = value

    for col_name, feature in TARGET_FEATURES.items():
        if col_name not in features:
            features[col_name] = {"dtype": feature["dtype"], "shape": list(feature["shape"]), "names": feature["names"]}

    info["features"] = features
    save_json(info_path, info)


def rename_video_folders(root: Path) -> None:
    for old_name, new_name in VIDEO_FOLDER_RENAMES:
        for folder in sorted((root / "videos").glob(f"chunk-*/{old_name}")):
            dst = folder.parent / new_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(folder), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Source WCM v2.1 dataset root")
    parser.add_argument("--output", required=True, help="Output dataset root (pistar form)")
    parser.add_argument("--success-labels", type=Path, default=None,
                        help="Optional JSON mapping episode_index -> 0/1")
    parser.add_argument("--value-col", default="return",
                        help="WCM column used to fill value_label (default: return)")
    parser.add_argument("--intervention", type=int, default=1, help="intervention value for these demos")
    parser.add_argument("--adv-ind-success", default="positive",
                        help="adv_ind value for positive frames (default: positive)")
    parser.add_argument("--adv-ind-fail", default="negative",
                        help="adv_ind value for negative frames (default: negative, pistar label_advantage 语义)")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    if not (root / "meta" / "info.json").exists():
        raise ValueError(f"Not a LeRobot dataset: {root}")
    if output.exists():
        raise ValueError(f"Output already exists, remove it first: {output}")

    success_labels = load_success_labels(args.success_labels)

    print(f"Copying dataset tree: {root} -> {output}")
    copy_dataset_tree(root, output)

    data_dir = output / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise ValueError(f"No v2.1 episode parquet files under {data_dir}")

    print(f"Rewriting {len(parquet_files)} parquet files ...")
    n_acp = 0
    for f in parquet_files:
        episode_index = int(f.stem[len("episode_"):])
        schema = pq.read_schema(f)
        has_acp = ACP_INDICATOR_COL in schema.names
        cols = [args.value_col] + ([ACP_INDICATOR_COL] if has_acp else [])
        ret = pq.read_table(f, columns=cols)
        return_values = np.asarray(ret.column(args.value_col).to_pylist(), dtype=np.float32).reshape(-1)
        num_rows = len(return_values)
        acp_indicators = ret.column(ACP_INDICATOR_COL).to_numpy() if has_acp else None
        if has_acp:
            n_acp += 1
        column_values = compute_column_values(
            num_rows,
            episode_index,
            return_values,
            success_labels,
            intervention=args.intervention,
            adv_ind_success=args.adv_ind_success,
            adv_ind_fail=args.adv_ind_fail,
            acp_indicators=acp_indicators,
        )
        rewrite_parquet(f, f, column_values)

    if n_acp:
        print(f"adv_ind: {n_acp}/{len(parquet_files)} episodes use WCM complementary_info.acp_indicator (1->positive, 0->negative)")
    else:
        print("adv_ind: no complementary_info.acp_indicator column found, falling back to success/failure labels")

    info_path = output / "meta" / "info.json"
    stats_path = output / "meta" / "episodes_stats.jsonl"
    update_info_json(info_path)
    print("Updating meta/episodes_stats.jsonl ...")
    update_episodes_stats(stats_path, parquet_files)

    print("Renaming video folders ...")
    rename_video_folders(output)

    print(f"Done. Output: {output}")
    print("Next: run pistar's scripts/add_value_labels.py? not needed (value_label already written).")
    print("      pistar's train_value.py / label_advantage_from_vlm.py can consume this dataset directly.")


if __name__ == "__main__":
    main()
