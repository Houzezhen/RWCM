"""Fail the staged experiment when alignment or OOD gates are not met."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return rows


def check_alignment(paths: list[Path], threshold: float) -> None:
    for path in paths:
        best = max(float(row["alignment_cosine"]) for row in read_jsonl(path))
        print(f"[alignment] {path}: best_cosine={best:.6f}")
        if best < threshold:
            raise SystemExit(f"Alignment gate failed: {best:.6f} < {threshold:.6f}")


def check_ood(paths: list[Path]) -> None:
    directions = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        pearson_delta = float(result["pearson_difference_candidate_minus_baseline"])
        mse_delta = float(result["mse_difference_candidate_minus_baseline"])
        base_centered = float(result["baseline_mse"]) - float(result["baseline_mean_bias"]) ** 2
        candidate_centered = float(result["candidate_mse"]) - float(result["candidate_mean_bias"]) ** 2
        pearson_ci_low = float(result["pearson_paired_episode_bootstrap_ci95"][0])
        directions.append(mse_delta)
        print(
            f"[ood] {path}: mse_delta={mse_delta:+.6f}, pearson_delta={pearson_delta:+.6f}, "
            f"centered_delta={candidate_centered - base_centered:+.6f}"
        )
        if pearson_delta < 0:
            raise SystemExit(f"OOD Pearson gate failed for {path}")
        if mse_delta < 0 and candidate_centered >= base_centered and pearson_ci_low <= 0:
            raise SystemExit(
                f"Apparent MSE gain is bias-only without significant ranking gain for {path}"
            )
    if directions[0] * directions[1] < 0:
        raise SystemExit("Two model seeds have opposite OOD MSE directions")


def check_checkpoint(path: Path) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model_config = config["model"]
    data_config = config["data"]
    gate = float(payload["model"]["spacetime_encoder.blend_gate"])
    print(
        f"[deploy] {path}: gate={gate}, teacher={model_config['spacetime_teacher_enabled']}, "
        f"mosaic={data_config['history_mosaic']}"
    )
    if gate != 1.0:
        raise SystemExit(f"Deploy gate must equal 1, got {gate}")
    if model_config["spacetime_teacher_enabled"] or data_config["history_mosaic"]:
        raise SystemExit("Deploy checkpoint still depends on the mosaic teacher")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", nargs="+", type=Path)
    parser.add_argument("--alignment-threshold", type=float, default=0.9)
    parser.add_argument("--ood", nargs=2, type=Path)
    parser.add_argument("--checkpoint", nargs="+", type=Path)
    args = parser.parse_args()
    if args.alignment:
        check_alignment(args.alignment, args.alignment_threshold)
    if args.ood:
        check_ood(args.ood)
    if args.checkpoint:
        for path in args.checkpoint:
            check_checkpoint(path)
    if not args.alignment and not args.ood and not args.checkpoint:
        parser.error("provide --alignment, --ood, or --checkpoint")


if __name__ == "__main__":
    run()
