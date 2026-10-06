"""Evaluate the WCM risk head as an episode-level success/failure classifier.

The model's risk target is failure (1 - episode_success).  Temporal evaluation
windows overlap, so each report selects one prediction per episode at a fixed
fraction of its available windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from world_critic.checkpoint import load_checkpoint_payload
from world_critic.config import apply_runtime_overrides, validate_train_config
from world_critic.data import (
    LeRobotWorldCriticDataset,
    WorldCriticCollator,
    build_processor,
    load_lerobot_dataset,
    load_episode_split,
    _scalar_column,
)
from world_critic.distributed import initialize_distributed
from world_critic.model import WorldCriticModel
from world_critic.training import config_from_checkpoint_payload, seed_everything


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    positive_recall = _safe_divide(tp, tp + fn)
    negative_recall = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    precision_success = _safe_divide(tn, tn + fn)
    recall_success = negative_recall
    f1_failure = _safe_divide(2.0 * precision * positive_recall, precision + positive_recall)
    f1_success = _safe_divide(
        2.0 * precision_success * recall_success,
        precision_success + recall_success,
    )
    return {
        "n_episodes": len(labels),
        "threshold": threshold,
        "accuracy": _safe_divide(tp + tn, len(labels)),
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "precision_failure": precision,
        "recall_failure": positive_recall,
        "f1_failure": f1_failure,
        "precision_success": precision_success,
        "recall_success": recall_success,
        "f1_success": f1_success,
        "macro_f1": 0.5 * (f1_failure + f1_success),
        "specificity_success": negative_recall,
        "confusion_matrix": {
            "true_failure_pred_failure": tp,
            "true_failure_pred_success": fn,
            "true_success_pred_failure": fp,
            "true_success_pred_success": tn,
        },
    }


def _binary_auroc(labels: list[int], probabilities: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(labels)), key=lambda index: probabilities[index])
    rank_sum = 0.0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and probabilities[order[end]] == probabilities[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        rank_sum += average_rank * sum(labels[order[index]] for index in range(position, end))
        position = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _calibrate_threshold(labels: list[int], scores: list[float]) -> float:
    """Choose a score threshold on calibration episodes only."""
    if not labels or len(set(labels)) < 2:
        raise ValueError("Threshold calibration requires both success and failure episodes.")
    unique = sorted(set(scores))
    candidates = [unique[0] - 1.0, unique[-1] + 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    best = None
    for threshold in candidates:
        metrics = _binary_metrics(labels, scores, threshold)
        key = (metrics["balanced_accuracy"], metrics["accuracy"], -abs(threshold))
        if best is None or key > best[0]:
            best = (key, threshold)
    assert best is not None
    return float(best[1])


def _resolve_episode_ids(config: Any, checkpoint: Path, split: str) -> list[int] | None:
    if split == "all":
        return None
    candidates: list[Path] = []
    if config.data.split_manifest:
        configured = Path(config.data.split_manifest).expanduser()
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.extend([checkpoint.parent / configured, checkpoint.parent.parent / configured])
    candidates.extend([checkpoint.parent / "episode_split.json", checkpoint.parent.parent / "episode_split.json"])
    manifest = next((path for path in candidates if path.exists()), None)
    if manifest is None:
        raise FileNotFoundError(
            f"Could not find episode_split.json for --split {split!r} next to {checkpoint}."
        )
    episode_split = load_episode_split(manifest)
    return getattr(episode_split, split)


def evaluate(args: argparse.Namespace) -> None:
    ctx = initialize_distributed(expected_world_size=1)
    if ctx.distributed:
        raise RuntimeError("Risk evaluation currently supports one process; omit torchrun.")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = load_checkpoint_payload(checkpoint_path, ctx)
    config = apply_runtime_overrides(config_from_checkpoint_payload(payload))
    if args.dataset_root:
        config.data.root = Path(args.dataset_root).expanduser().resolve()
    if args.mode == "value" and config.data.success_key is None:
        config.data.success_key = "episode_success"
    validate_train_config(config)
    if args.mode == "risk" and not config.model.predict_risk:
        raise ValueError("The checkpoint does not contain a risk head (model.predict_risk=false).")
    if args.mode == "value" and config.model.predict_risk:
        raise ValueError("Use --mode risk for a checkpoint with a risk head.")
    seed_everything(config.seed, config.deterministic)

    dataset = load_lerobot_dataset(config.data)
    processor = build_processor(config.model)
    collator = WorldCriticCollator(
        processor,
        config.model.vision.image_size,
        config.model.language.max_length,
    )
    model = WorldCriticModel(config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(ctx.device).eval().requires_grad_(False)

    def make_loader(split: str) -> DataLoader:
        episode_ids = _resolve_episode_ids(config, checkpoint_path, split)
        eval_dataset = LeRobotWorldCriticDataset(dataset, config.data, episode_ids)
        frame_by_row = (
            _scalar_column(dataset, "frame_index", np.int64).reshape(-1)
            if args.frame_index is not None
            else None
        )
        selected_rows = []
        for row_start, row_end in eval_dataset.episode_ranges.values():
            if row_end - row_start < eval_dataset.window:
                continue
            if episode_ids is not None and int(eval_dataset.episode_by_row[row_start]) not in episode_ids:
                continue
            if args.frame_index is None:
                selected_row = row_start + round(
                    (row_end - row_start - eval_dataset.window) * args.episode_fraction
                )
            else:
                assert frame_by_row is not None
                first_frame = int(frame_by_row[row_start])
                selected_row = (
                    row_start + args.frame_index - first_frame - (config.data.history_size - 1)
                )
                if not row_start <= selected_row <= row_end - eval_dataset.window:
                    raise ValueError(
                        f"Frame {args.frame_index} has no valid prediction in episode "
                        f"{int(eval_dataset.episode_by_row[row_start])}."
                    )
            selected_rows.append(selected_row)
        indices = np.searchsorted(eval_dataset.window_starts, selected_rows)
        if any(
            index >= len(eval_dataset.window_starts) or eval_dataset.window_starts[index] != row
            for index, row in zip(indices, selected_rows, strict=True)
        ):
            raise RuntimeError("Could not resolve the selected window for every episode.")
        selected_dataset = Subset(eval_dataset, indices.tolist())
        if len(selected_dataset) == 0:
            raise ValueError(f"Selected split {split!r} contains no evaluable episode endpoints.")
        return DataLoader(
            selected_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=ctx.device.type == "cuda",
            collate_fn=collator,
        )

    def collect_records(loader: DataLoader) -> list[dict[str, Any]]:
        endpoint_by_episode: dict[int, dict[str, Any]] = {}
        with torch.inference_mode():
          for batch in loader:
            device_batch = {
                key: value.to(ctx.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(
                images=device_batch["images"],
                actions=device_batch["actions"],
                instruction_input_ids=device_batch["instruction_input_ids"],
                instruction_attention_mask=device_batch["instruction_attention_mask"],
                valid_mask=device_batch["valid_mask"],
                state_vectors=device_batch.get("state_vectors"),
                history_images=device_batch.get("history_images"),
            )
            valid = output.valid_mask.bool()
            if args.mode == "risk":
                if output.risk_logits is None:
                    raise RuntimeError("Checkpoint forward returned no risk logits.")
                scores = torch.sigmoid(output.risk_logits.squeeze(-1))
            else:
                scores = -output.value.squeeze(-1)
            success_targets = device_batch["success_targets"].squeeze(-1)
            frame_indices = device_batch["frame_indices"]
            for row in range(scores.size(0)):
                valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                position = int(valid_positions[-1])
                episode_id = int(device_batch["episode_id"][row])
                frame_index = int(frame_indices[row, position])
                record = {
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "failure_score": float(scores[row, position]),
                    "raw_value": float(output.value[row, position]),
                    "true_failure": int(1.0 - float(success_targets[row, position])),
                }
                previous = endpoint_by_episode.get(episode_id)
                if previous is not None:
                    raise RuntimeError(f"Duplicate prediction for episode {episode_id}.")
                endpoint_by_episode[episode_id] = record
        return [endpoint_by_episode[key] for key in sorted(endpoint_by_episode)]

    calibration_records = (
        collect_records(make_loader(args.calibration_split)) if args.mode == "value" else []
    )
    records = collect_records(make_loader(args.split))
    labels = [record["true_failure"] for record in records]
    scores = [record["failure_score"] for record in records]
    if args.mode == "value":
        calibration_labels = [record["true_failure"] for record in calibration_records]
        calibration_scores = [record["failure_score"] for record in calibration_records]
        threshold = _calibrate_threshold(calibration_labels, calibration_scores)
    else:
        threshold = args.threshold
    metrics = _binary_metrics(labels, scores, threshold)
    metrics["auroc_failure"] = _binary_auroc(labels, scores)
    metrics["positive_failure_episodes"] = sum(labels)
    metrics["positive_success_episodes"] = len(labels) - sum(labels)

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(config.data.root),
        "split": args.split,
        "score_source": args.mode,
        "target_definition": "true_failure = 1 - episode_success",
        "score_definition": (
            "sigmoid(risk_logits) predicts failure probability"
            if args.mode == "risk"
            else "failure_score = -V(s); larger values indicate failure"
        ),
        "endpoint_definition": (
            "one prediction per episode at the specified frame index"
            if args.frame_index is not None
            else "one window per episode at the specified fraction of available windows"
        ),
        "episode_fraction": args.episode_fraction if args.frame_index is None else None,
        "frame_index": args.frame_index,
        "calibration_split": args.calibration_split if args.mode == "value" else None,
        "metrics": metrics,
        "episodes": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["risk", "value"], default="risk")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--calibration-split", default="train", choices=["train", "val", "all"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--episode-fraction", type=float, default=1.0)
    parser.add_argument("--frame-index", type=int)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1].")
    if not 0.0 <= args.episode_fraction <= 1.0:
        raise ValueError("--episode-fraction must be in [0,1].")
    if args.frame_index is not None and args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative.")
    evaluate(args)


if __name__ == "__main__":
    main()
