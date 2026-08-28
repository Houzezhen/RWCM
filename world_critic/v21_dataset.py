"""Minimal LeRobotDataset-compatible reader for LeRobot v2.1 datasets.

v2.1 stores one parquet file per episode (``data/chunk-*/episode_*.parquet``),
JSONL episode/task metadata, per-episode stats, and one MP4 per camera per
episode.  lerobot 0.5.x's ``LeRobotDataset`` only reads the v3.0 layout, so
world_critic gets this class as a drop-in replacement when the dataset root
carries the v2.1 markers (``meta/episodes.jsonl`` + ``meta/tasks.jsonl``).

It implements exactly the surface that ``world_critic/data.py`` and the
inference/labeling scripts consume: ``features``, ``hf_dataset`` (an
Arrow-backed view with ``data``/``column_names``/``[key]``/``with_format``),
``dataset[i]`` (including TorchCodec video decoding), ``meta.tasks`` and
``__len__``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _feature_dtype(feature: Any) -> str:
    if isinstance(feature, dict):
        value = feature.get("dtype", "")
    else:
        value = getattr(feature, "dtype", "")
    return str(value).strip().lower()


class _TableShim:
    """Expose the small Hugging-Face-Dataset surface world_critic touches."""

    def __init__(self, table: pa.Table) -> None:
        self.data = table

    @property
    def column_names(self) -> list[str]:
        return list(self.data.column_names)

    def __getitem__(self, key: str) -> Any:
        column = self.data.column(key)
        try:
            return column.to_numpy(zero_copy_only=False)
        except Exception:
            return column.to_pylist()

    def __len__(self) -> int:
        return self.data.num_rows

    def with_format(self, *args: Any, **kwargs: Any) -> "_TableShim":
        return self


class LeRobotV21Dataset:
    """Read a LeRobot v2.1 dataset with a LeRobotDataset-compatible interface."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta = self.root / "meta"

        info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
        features = info.get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError(f"No features declared in {meta / 'info.json'}.")
        self.features: dict[str, dict] = features

        task_rows = [
            json.loads(line)
            for line in (meta / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        name_to_index = {text: idx for idx, text in self._task_by_index.items()}

        episode_rows = [
            json.loads(line)
            for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._task_by_episode: dict[int, int] = {}
        for row in episode_rows:
            episode = int(row["episode_index"])
            names = row.get("tasks") or []
            if not names:
                raise ValueError(f"episode_index={episode} has no tasks entry.")
            name = str(names[0])
            if name not in name_to_index:
                raise ValueError(f"episode_index={episode} references unknown task {name!r}.")
            self._task_by_episode[episode] = name_to_index[name]

        self._records = self._collect_records(self.root / "data")
        self._offsets: dict[int, int] = {}
        num_rows = 0
        for _file, episode, nrows in self._records:
            self._offsets[episode] = num_rows
            num_rows += nrows
        self._num_rows = num_rows

        table = pa.concat_tables([pq.read_table(file) for file, _ep, _n in self._records])
        self.hf_dataset = _TableShim(table)
        self._row_episode = table.column("episode_index").to_numpy().astype(np.int64)

        tasks_frame = pd.DataFrame(
            [{"task_index": idx, "task": text} for idx, text in self._task_by_index.items()]
        ).set_index("task")
        self.meta = types.SimpleNamespace(tasks=tasks_frame)

        self._video_backend = "torchcodec"
        self._row_cache: dict[int, list[dict]] = {}
        self._decoders: dict[tuple[int, str], Any] = {}

    @staticmethod
    def _collect_records(data_dir: Path) -> list[tuple[Path, int, int]]:
        files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No v2.1 episode parquet files under {data_dir}.")
        records: list[tuple[Path, int, int, int]] = []
        for file in files:
            table = pq.read_table(file, columns=["episode_index", "index"])
            records.append(
                (
                    file,
                    int(table.column("episode_index")[0].as_py()),
                    table.num_rows,
                    int(table.column("index")[0].as_py()),
                )
            )
        records.sort(key=lambda rec: rec[3])
        return [(file, episode, nrows) for file, episode, nrows, _start in records]

    def _episode_rows(self, episode: int) -> list[dict]:
        rows = self._row_cache.get(episode)
        if rows is None:
            file = next((f for f, ep, _n in self._records if ep == episode), None)
            if file is None:
                raise KeyError(f"No parquet file for episode_index={episode}.")
            rows = pq.read_table(file).to_pylist()
            self._row_cache[episode] = rows
        return rows

    def _video_frame(self, episode: int, key: str, frame_index: int) -> np.ndarray:
        decoder = self._decoders.get((episode, key))
        if decoder is None:
            videos = sorted(
                self.root.glob(f"videos/chunk-*/{key}/episode_{episode:06d}.mp4")
            )
            if not videos:
                raise FileNotFoundError(f"No video for episode={episode}, key={key}.")
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(str(videos[0]))
            self._decoders[(episode, key)] = decoder
        frame = decoder.get_frame_at(index=frame_index)
        if hasattr(frame, "data"):  # torchcodec >= 0.4 returns a Frame namedtuple
            frame = frame.data
        frame = frame.cpu()
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = frame.permute(1, 2, 0)
        return frame.numpy()

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not (0 <= index < self._num_rows):
            raise IndexError(index)
        episode = int(self._row_episode[index])
        row_in_episode = int(index - self._offsets[episode])
        sample = dict(self._episode_rows(episode)[row_in_episode])
        frame_index = int(sample["frame_index"])
        for key, feature in self.features.items():
            if _feature_dtype(feature) == "video":
                sample[key] = self._video_frame(episode, key, frame_index)
        sample["task"] = self._task_by_index[self._task_by_episode[episode]]
        return sample
