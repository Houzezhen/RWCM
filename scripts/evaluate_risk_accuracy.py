"""Evaluate the WCM risk head as an episode-level success/failure classifier.

The model's risk target is failure (1 - episode_success).  Temporal evaluation
windows overlap, so the primary report keeps the final available prediction per
episode.  This avoids counting the same episode repeatedly while preserving the
model's endpoint classification behavior.
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
    validate_train_config(config)
    if not config.model.predict_risk:
        raise ValueError("The checkpoint does not contain a risk head (model.predict_risk=false).")
    seed_everything(config.seed, config.deterministic)

    dataset = load_lerobot_dataset(config.data)
    episode_ids = _resolve_episode_ids(config, checkpoint_path, args.split)
    eval_dataset = LeRobotWorldCriticDataset(dataset, config.data, episode_ids)
    terminal_rows = [
        row_end - eval_dataset.window
        for row_start, row_end in eval_dataset.episode_ranges.values()
        if row_end - row_start >= eval_dataset.window
        and (episode_ids is None or int(eval_dataset.episode_by_row[row_start]) in episode_ids)
    ]
    indices = np.searchsorted(eval_dataset.window_starts, terminal_rows)
    if any(
        index >= len(eval_dataset.window_starts) or eval_dataset.window_starts[index] != row
        for index, row in zip(indices, terminal_rows, strict=True)
    ):
        raise RuntimeError("Could not resolve the final valid window for every selected episode.")
    terminal_dataset = Subset(eval_dataset, indices.tolist())
    if len(terminal_dataset) == 0:
        raise ValueError("Selected split contains no evaluable episode endpoints.")
    processor = build_processor(config.model)
    collator = WorldCriticCollator(
        processor,
        config.model.vision.image_size,
        config.model.language.max_length,
    )
    loader = DataLoader(
        terminal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=ctx.device.type == "cuda",
        collate_fn=collator,
    )

    model = WorldCriticModel(config.model)
    model.load_state_dict(payload["model"], strict=True)
    model.to(ctx.device).eval().requires_grad_(False)

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
            if output.risk_logits is None:
                raise RuntimeError("Checkpoint forward returned no risk logits.")
            valid = output.valid_mask.bool()
            probabilities = torch.sigmoid(output.risk_logits.squeeze(-1))
            success_targets = device_batch["success_targets"].squeeze(-1)
            frame_indices = device_batch["frame_indices"]
            for row in range(probabilities.size(0)):
                valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                position = int(valid_positions[-1])
                episode_id = int(device_batch["episode_id"][row])
                frame_index = int(frame_indices[row, position])
                record = {
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "failure_probability": float(probabilities[row, position]),
                    "true_failure": int(1.0 - float(success_targets[row, position])),
                }
                previous = endpoint_by_episode.get(episode_id)
                if previous is not None:
                    raise RuntimeError(f"Duplicate terminal prediction for episode {episode_id}.")
                endpoint_by_episode[episode_id] = record

    records = [endpoint_by_episode[key] for key in sorted(endpoint_by_episode)]
    labels = [record["true_failure"] for record in records]
    probabilities = [record["failure_probability"] for record in records]
    metrics = _binary_metrics(labels, probabilities, args.threshold)
    metrics["auroc_failure"] = _binary_auroc(labels, probabilities)
    metrics["positive_failure_episodes"] = sum(labels)
    metrics["positive_success_episodes"] = len(labels) - sum(labels)

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "dataset_root": str(config.data.root),
        "split": args.split,
        "target_definition": "true_failure = 1 - episode_success; risk sigmoid predicts failure probability",
        "endpoint_definition": "last available frame per episode",
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
    parser.add_argument("--dataset-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0,1].")
    evaluate(args)


if __name__ == "__main__":
    main()
