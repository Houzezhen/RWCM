"""Paired episode-level comparison for two WCM evaluation directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("baseline", help="Baseline eval directory or episode_metrics.json")
    result.add_argument("candidate", help="Candidate eval directory or episode_metrics.json")
    result.add_argument("--bootstrap-samples", type=int, default=20_000)
    result.add_argument("--seed", type=int, default=3072)
    result.add_argument("--output")
    return result


def metric_path(value: str) -> Path:
    path = Path(value)
    candidates = (
        path,
        path / "episode_metrics.json",
        path / "episode_curves" / "episode_metrics.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No episode_metrics.json found under {path}.")


def load(path: Path) -> dict[int, tuple[float, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["episode_id"]): (float(row["value_mse"]), int(row["count"]))
        for row in rows
    }


def run() -> None:
    args = parser().parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")
    baseline_path = metric_path(args.baseline)
    candidate_path = metric_path(args.candidate)
    baseline = load(baseline_path)
    candidate = load(candidate_path)
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "Evaluation episode sets differ: "
            f"missing_candidate={missing_candidate[:10]}, missing_baseline={missing_baseline[:10]}"
        )

    episode_ids = sorted(baseline)
    base_mse = np.asarray([baseline[key][0] for key in episode_ids], dtype=np.float64)
    candidate_mse = np.asarray([candidate[key][0] for key in episode_ids], dtype=np.float64)
    counts = np.asarray([baseline[key][1] for key in episode_ids], dtype=np.float64)
    candidate_counts = np.asarray([candidate[key][1] for key in episode_ids], dtype=np.float64)
    if not np.array_equal(counts, candidate_counts):
        raise ValueError("Paired episodes contain different endpoint counts.")

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(episode_ids), size=(args.bootstrap_samples, len(episode_ids)))
    sampled_counts = counts[indices]
    base_boot = (base_mse[indices] * sampled_counts).sum(axis=1) / sampled_counts.sum(axis=1)
    candidate_boot = (
        (candidate_mse[indices] * sampled_counts).sum(axis=1) / sampled_counts.sum(axis=1)
    )
    differences = candidate_boot - base_boot
    base_pooled = float(np.average(base_mse, weights=counts))
    candidate_pooled = float(np.average(candidate_mse, weights=counts))
    result = {
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "episodes": len(episode_ids),
        "endpoints": int(counts.sum()),
        "baseline_mse": base_pooled,
        "candidate_mse": candidate_pooled,
        "mse_difference_candidate_minus_baseline": candidate_pooled - base_pooled,
        "relative_mse_change": (candidate_pooled - base_pooled) / base_pooled,
        "paired_episode_bootstrap_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "probability_candidate_lower_mse": float(np.mean(differences < 0)),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    run()
