"""Paired episode bootstrap for MSE, correlation, and calibration diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("baseline", help="Baseline eval directory or episode_metrics.json")
    result.add_argument("candidate", help="Candidate eval directory or episode_metrics.json")
    result.add_argument("--bootstrap-samples", type=int, default=20_000)
    result.add_argument("--seed", type=int, default=3072)
    result.add_argument(
        "--align",
        choices=["exact", "common"],
        default="exact",
        help="Endpoint alignment policy; common is required when history lengths differ.",
    )
    result.add_argument(
        "--endpoint-manifest",
        help=(
            "Optional JSON manifest containing an 'endpoints' list of "
            "{episode_id, frame_index} objects. Both inputs must contain every listed endpoint."
        ),
    )
    result.add_argument("--output")
    return result


def curve_path(value: str) -> Path:
    path = Path(value)
    if path.name == "episode_metrics.json":
        path = path.with_name("episode_curves.csv")
    candidates = (
        path,
        path / "episode_curves.csv",
        path / "episode_curves" / "episode_curves.csv",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "episode_curves.csv":
            return candidate
    raise FileNotFoundError(f"No episode_curves.csv found under {path}.")


def load(path: Path) -> dict[tuple[int, int], tuple[float, float]]:
    result: dict[tuple[int, int], tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["episode_id"]), int(row["frame_index"]))
            if key in result:
                raise ValueError(f"Duplicate episode/frame in {path}: {key}")
            result[key] = (float(row["value"]), float(row["return"]))
    if not result:
        raise ValueError(f"No episode curve rows found in {path}.")
    return result


def episode_statistics(
    rows: dict[tuple[int, int], tuple[float, float]],
    episode_ids: list[int],
) -> np.ndarray:
    """Return per-episode sufficient statistics for pooled MSE/Pearson."""

    index = {episode_id: position for position, episode_id in enumerate(episode_ids)}
    # n, squared error, sum target, sum prediction, target^2, prediction^2, product
    stats = np.zeros((len(episode_ids), 7), dtype=np.float64)
    for (episode_id, _), (prediction, target) in rows.items():
        position = index[episode_id]
        error = prediction - target
        stats[position] += (
            1.0,
            error * error,
            target,
            prediction,
            target * target,
            prediction * prediction,
            target * prediction,
        )
    return stats


def metrics(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = stats[..., 0]
    mse = stats[..., 1] / count
    target_variance = np.maximum(count * stats[..., 4] - stats[..., 2] ** 2, 0.0)
    prediction_variance = np.maximum(count * stats[..., 5] - stats[..., 3] ** 2, 0.0)
    denominator = np.sqrt(target_variance * prediction_variance)
    numerator = count * stats[..., 6] - stats[..., 2] * stats[..., 3]
    pearson = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    bias = (stats[..., 3] - stats[..., 2]) / count
    return mse, pearson, bias


def load_endpoint_manifest(path: Path) -> set[tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError(f"Endpoint manifest has no non-empty 'endpoints' list: {path}")
    result: set[tuple[int, int]] = set()
    for row in endpoints:
        try:
            key = (int(row["episode_id"]), int(row["frame_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed endpoint in {path}: {row!r}") from exc
        if key in result:
            raise ValueError(f"Duplicate endpoint in {path}: {key}")
        result.add(key)
    return result


def compare_rows(
    baseline: dict[tuple[int, int], tuple[float, float]],
    candidate: dict[tuple[int, int], tuple[float, float]],
    *,
    baseline_path: Path,
    candidate_path: Path,
    bootstrap_samples: int,
    seed: int,
    align: str = "exact",
    endpoint_keys: set[tuple[int, int]] | None = None,
    endpoint_manifest: Path | None = None,
) -> dict[str, Any]:
    """Compare two prediction sets on one explicit endpoint population."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    dropped_baseline = sorted(baseline_keys - candidate_keys)
    dropped_candidate = sorted(candidate_keys - baseline_keys)

    if endpoint_keys is not None:
        missing_baseline = sorted(endpoint_keys - baseline_keys)
        missing_candidate = sorted(endpoint_keys - candidate_keys)
        if missing_baseline or missing_candidate:
            raise ValueError(
                "Endpoint manifest is not covered by both evaluations: "
                f"missing_baseline={missing_baseline[:10]}, "
                f"missing_candidate={missing_candidate[:10]}"
            )
        baseline = {key: baseline[key] for key in endpoint_keys}
        candidate = {key: candidate[key] for key in endpoint_keys}
        dropped_baseline = sorted(baseline_keys - endpoint_keys)
        dropped_candidate = sorted(candidate_keys - endpoint_keys)
        effective_alignment = "manifest"
    elif dropped_baseline or dropped_candidate:
        if align == "exact":
            raise ValueError(
                "Evaluation endpoint sets differ: "
                f"missing_candidate={dropped_baseline[:10]}, "
                f"missing_baseline={dropped_candidate[:10]}"
            )
        common_keys = baseline_keys & candidate_keys
        if not common_keys:
            raise ValueError("Evaluation endpoint sets have no common endpoints.")
        baseline = {key: baseline[key] for key in common_keys}
        candidate = {key: candidate[key] for key in common_keys}
        effective_alignment = align
    else:
        effective_alignment = align

    mismatched_targets = [
        key
        for key in baseline
        if not math.isclose(baseline[key][1], candidate[key][1], rel_tol=0.0, abs_tol=1e-8)
    ]
    if mismatched_targets:
        raise ValueError(f"Paired endpoints contain different targets: {mismatched_targets[:10]}")

    episode_ids = sorted({episode_id for episode_id, _ in baseline})
    baseline_stats = episode_statistics(baseline, episode_ids)
    candidate_stats = episode_statistics(candidate, episode_ids)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(episode_ids), size=(bootstrap_samples, len(episode_ids)))
    baseline_boot = baseline_stats[indices].sum(axis=1)
    candidate_boot = candidate_stats[indices].sum(axis=1)
    base_boot_mse, base_boot_pearson, base_boot_bias = metrics(baseline_boot)
    candidate_boot_mse, candidate_boot_pearson, candidate_boot_bias = metrics(candidate_boot)
    mse_differences = candidate_boot_mse - base_boot_mse
    base_boot_centered = base_boot_mse - base_boot_bias**2
    candidate_boot_centered = candidate_boot_mse - candidate_boot_bias**2
    centered_differences = candidate_boot_centered - base_boot_centered
    pearson_differences = candidate_boot_pearson - base_boot_pearson
    finite_pearson_differences = pearson_differences[np.isfinite(pearson_differences)]
    base_mse, base_pearson, base_bias = metrics(baseline_stats.sum(axis=0))
    candidate_mse, candidate_pearson, candidate_bias = metrics(candidate_stats.sum(axis=0))
    base_centered = base_mse - base_bias**2
    candidate_centered = candidate_mse - candidate_bias**2
    relative_mse_change = (
        float((candidate_mse - base_mse) / base_mse) if base_mse != 0 else None
    )
    pearson_ci = (
        np.quantile(finite_pearson_differences, [0.025, 0.975]).tolist()
        if finite_pearson_differences.size
        else [None, None]
    )
    pearson_probability = (
        float(np.mean(finite_pearson_differences > 0))
        if finite_pearson_differences.size
        else None
    )
    return {
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "episodes": len(episode_ids),
        "endpoints": int(baseline_stats[:, 0].sum()),
        "alignment": effective_alignment,
        "endpoint_manifest": str(endpoint_manifest.resolve()) if endpoint_manifest else None,
        "dropped_baseline_endpoints": len(dropped_baseline),
        "dropped_candidate_endpoints": len(dropped_candidate),
        "target_consistency": {
            "checked_endpoints": len(baseline),
            "mismatched_endpoints": 0,
            "absolute_tolerance": 1e-8,
        },
        "baseline_mse": float(base_mse),
        "candidate_mse": float(candidate_mse),
        "mse_difference_candidate_minus_baseline": float(candidate_mse - base_mse),
        "relative_mse_change": relative_mse_change,
        "mse_paired_episode_bootstrap_ci95": np.quantile(
            mse_differences, [0.025, 0.975]
        ).tolist(),
        # Backward-compatible alias used by the first paired-register run.
        "paired_episode_bootstrap_ci95": np.quantile(
            mse_differences, [0.025, 0.975]
        ).tolist(),
        "probability_candidate_lower_mse": float(np.mean(mse_differences < 0)),
        "baseline_centered_mse": float(base_centered),
        "candidate_centered_mse": float(candidate_centered),
        "centered_mse_difference_candidate_minus_baseline": float(
            candidate_centered - base_centered
        ),
        "centered_mse_paired_episode_bootstrap_ci95": np.quantile(
            centered_differences, [0.025, 0.975]
        ).tolist(),
        "probability_candidate_lower_centered_mse": float(
            np.mean(centered_differences < 0)
        ),
        "baseline_pearson": float(base_pearson) if np.isfinite(base_pearson) else None,
        "candidate_pearson": (
            float(candidate_pearson) if np.isfinite(candidate_pearson) else None
        ),
        "pearson_difference_candidate_minus_baseline": (
            float(candidate_pearson - base_pearson)
            if np.isfinite(candidate_pearson - base_pearson)
            else None
        ),
        "pearson_paired_episode_bootstrap_ci95": pearson_ci,
        "probability_candidate_higher_pearson": pearson_probability,
        "baseline_mean_bias": float(base_bias),
        "candidate_mean_bias": float(candidate_bias),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "seed": seed,
    }


def run() -> None:
    args = parser().parse_args()
    baseline_path = curve_path(args.baseline)
    candidate_path = curve_path(args.candidate)
    baseline = load(baseline_path)
    candidate = load(candidate_path)
    manifest_path = Path(args.endpoint_manifest) if args.endpoint_manifest else None
    endpoint_keys = load_endpoint_manifest(manifest_path) if manifest_path else None
    result = compare_rows(
        baseline,
        candidate,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        align=args.align,
        endpoint_keys=endpoint_keys,
        endpoint_manifest=manifest_path,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    run()
