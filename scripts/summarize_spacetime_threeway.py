"""Run the preregistered SpaceTime three-way paired analysis and write reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.compare_experiments import compare_rows, curve_path, episode_statistics, load, metrics


MAIN_ROLES = ("baseline", "mosaic", "spacetime")
ALL_ROLES = (*MAIN_ROLES, "staged")
COMPARISONS = {
    "spacetime_vs_baseline": ("baseline", "spacetime"),
    "spacetime_vs_mosaic": ("mosaic", "spacetime"),
    "spacetime_vs_staged": ("staged", "spacetime"),
    "mosaic_vs_baseline": ("baseline", "mosaic"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--eval-root", type=Path, default=Path("outputs/eval_spacetime_threeway"))
    result.add_argument("--datasets", nargs="+", default=["5cut", "ood"])
    result.add_argument("--seeds", nargs="+", type=int, default=[3072, 42])
    result.add_argument("--bootstrap-samples", type=int, default=20_000)
    result.add_argument("--output-json", type=Path)
    result.add_argument("--output-markdown", type=Path)
    return result


def _curve(root: Path, dataset: str, role: str, seed: int) -> Path:
    return curve_path(str(root / dataset / f"{role}_s{seed}"))


def build_common_manifest(
    paths: dict[str, Path],
    *,
    dataset: str,
    seed: int,
    output: Path,
) -> tuple[dict[str, dict[tuple[int, int], tuple[float, float]]], set[tuple[int, int]], dict[str, Any]]:
    """Build the exact endpoint population shared by the three primary models."""

    rows = {role: load(path) for role, path in paths.items()}
    common = set.intersection(*(set(rows[role]) for role in MAIN_ROLES))
    if not common:
        raise ValueError(f"No common primary-model endpoints for {dataset} seed {seed}.")
    mismatches: list[tuple[int, int]] = []
    for key in sorted(common):
        targets = [rows[role][key][1] for role in MAIN_ROLES]
        if any(
            not math.isclose(targets[0], target, rel_tol=0.0, abs_tol=1e-8)
            for target in targets[1:]
        ):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            f"Primary models contain different targets for {dataset} seed {seed}: {mismatches[:10]}"
        )

    payload = {
        "dataset": dataset,
        "seed": seed,
        "roles": list(MAIN_ROLES),
        "source_endpoint_counts": {role: len(rows[role]) for role in MAIN_ROLES},
        "common_episodes": len({episode_id for episode_id, _ in common}),
        "common_endpoints": len(common),
        "dropped_endpoints": {
            role: len(rows[role]) - len(common) for role in MAIN_ROLES
        },
        "target_consistency": {
            "checked_endpoints": len(common),
            "mismatched_endpoints": 0,
            "absolute_tolerance": 1e-8,
        },
        "endpoints": [
            {"episode_id": episode_id, "frame_index": frame_index}
            for episode_id, frame_index in sorted(common)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return rows, common, payload


def absolute_metrics(
    rows: dict[tuple[int, int], tuple[float, float]], keys: set[tuple[int, int]]
) -> dict[str, Any]:
    selected = {key: rows[key] for key in keys}
    episode_ids = sorted({episode_id for episode_id, _ in selected})
    stat = episode_statistics(selected, episode_ids).sum(axis=0)
    mse, pearson, bias = metrics(stat)
    return {
        "episodes": len(episode_ids),
        "endpoints": len(selected),
        "mse": float(mse),
        "pearson": float(pearson) if math.isfinite(float(pearson)) else None,
        "mean_bias": float(bias),
        "centered_mse": float(mse - bias**2),
    }


def _metric_gate(comparisons: list[dict[str, Any]], prefix: str) -> bool:
    delta_key = f"{prefix}_difference_candidate_minus_baseline"
    ci_key = f"{prefix}_paired_episode_bootstrap_ci95"
    return (
        all(float(item[delta_key]) < 0 for item in comparisons)
        and any(float(item[ci_key][1]) < 0 for item in comparisons)
        and all(float(item[ci_key][0]) <= 0 for item in comparisons)
    )


def _pearson_values(comparisons: list[dict[str, Any]]) -> tuple[list[float], list[list[float]]]:
    deltas: list[float] = []
    intervals: list[list[float]] = []
    for item in comparisons:
        delta = item["pearson_difference_candidate_minus_baseline"]
        interval = item["pearson_paired_episode_bootstrap_ci95"]
        if delta is None or interval[0] is None or interval[1] is None:
            return [], []
        deltas.append(float(delta))
        intervals.append([float(interval[0]), float(interval[1])])
    return deltas, intervals


def baseline_gate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    raw_pass = _metric_gate(comparisons, "mse")
    centered_pass = _metric_gate(comparisons, "centered_mse")
    centered_directions = [
        float(item["centered_mse_difference_candidate_minus_baseline"])
        for item in comparisons
    ]
    pearson_deltas, pearson_intervals = _pearson_values(comparisons)
    pearson_pass = bool(pearson_deltas) and all(delta >= 0 for delta in pearson_deltas) and any(
        interval[0] > 0 for interval in pearson_intervals
    )
    not_bias_only = all(delta < 0 for delta in centered_directions)
    reasons = {
        "mse_or_centered_mse_two_seed_gate": raw_pass or centered_pass,
        "pearson_non_decrease_and_one_significant_gain": pearson_pass,
        "centered_mse_improves_both_seeds": not_bias_only,
    }
    return {"passed": all(reasons.values()), "checks": reasons}


def mosaic_gate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    raw_pass = _metric_gate(comparisons, "mse")
    centered_pass = _metric_gate(comparisons, "centered_mse")
    centered_directions = [
        float(item["centered_mse_difference_candidate_minus_baseline"])
        for item in comparisons
    ]
    pearson_deltas, pearson_intervals = _pearson_values(comparisons)
    pearson_non_decrease = bool(pearson_deltas) and all(delta >= 0 for delta in pearson_deltas)
    no_significant_decline = bool(pearson_intervals) and all(
        interval[1] >= 0 for interval in pearson_intervals
    )
    bias_only_fallback_needed = raw_pass and not all(delta < 0 for delta in centered_directions)
    bias_only_fallback_pass = (
        not bias_only_fallback_needed
        or (
            all(delta > 0 for delta in pearson_deltas)
            and any(interval[0] > 0 for interval in pearson_intervals)
        )
    )
    reasons = {
        "mse_or_centered_mse_two_seed_gate": raw_pass or centered_pass,
        "pearson_non_decrease_both_seeds": pearson_non_decrease,
        "no_significant_pearson_decline": no_significant_decline,
        "bias_only_requires_two_seed_pearson_gain": bias_only_fallback_pass,
    }
    return {"passed": all(reasons.values()), "checks": reasons}


def _has_direction_conflict(comparisons: list[dict[str, Any]]) -> bool:
    keys = (
        "mse_difference_candidate_minus_baseline",
        "centered_mse_difference_candidate_minus_baseline",
        "pearson_difference_candidate_minus_baseline",
    )
    for key in keys:
        values = [item[key] for item in comparisons]
        if any(value is None for value in values):
            return True
        if float(values[0]) * float(values[1]) < 0:
            return True
    for item in comparisons:
        error_directions = {
            math.copysign(1.0, float(item[key]))
            for key in keys[:2]
            if float(item[key]) != 0
        }
        if len(error_directions) > 1:
            return True
        pearson = float(item[keys[2]])
        mse = float(item[keys[0]])
        centered = float(item[keys[1]])
        if pearson != 0 and (mse < 0 or centered < 0) == (pearson < 0):
            return True
    return False


def choose_branch(
    baseline_result: dict[str, Any],
    mosaic_result: dict[str, Any],
    baseline_comparisons: list[dict[str, Any]],
    mosaic_comparisons: list[dict[str, Any]],
) -> dict[str, str]:
    conflict = _has_direction_conflict(baseline_comparisons) or _has_direction_conflict(
        mosaic_comparisons
    )
    if conflict:
        return {
            "branch": "D",
            "conclusion": "Seed 或指标方向冲突，证据不足；按预注册方案增加 seed。",
        }
    if baseline_result["passed"] and mosaic_result["passed"]:
        return {
            "branch": "A",
            "conclusion": "SpaceTime 同时超过原始 WCM 与 Image-B4 mosaic。",
        }
    if baseline_result["passed"]:
        return {
            "branch": "B",
            "conclusion": (
                "SpaceTime 超过原始 WCM，但未超过 Image-B4 mosaic；论文主叙事仅采用"
                "‘原始 WCM → SpaceTime’，mosaic 只作为实验强基线报告。"
            ),
        }
    return {
        "branch": "C",
        "conclusion": "SpaceTime 未稳定超过原始 WCM，主方案回退到 mosaic。",
    }


def _efficiency(root: Path, dataset: str, role: str, seed: int) -> dict[str, Any]:
    descriptors = {
        "baseline": {"input_frames": 1, "visual_tokens": 1},
        "mosaic": {"input_frames": 4, "visual_tokens": 1},
        "spacetime": {"input_frames": 8, "visual_tokens": 64},
        "staged": {"input_frames": 8, "visual_tokens": 64},
    }
    result = dict(descriptors[role])
    summary_path = root / dataset / f"{role}_s{seed}" / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result["parameters"] = summary.get("model", {}).get("parameters")
        result["forward_ms_per_sample"] = summary.get("timing", {}).get(
            "forward_ms_per_sample"
        )
        result["latency_batch_size"] = summary.get("timing", {}).get("batch_size")
    else:
        result.update(parameters=None, forward_ms_per_sample=None, latency_batch_size=None)
    return result


def _dataset_provenance(root: Path, dataset: str, seed: int) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ALL_ROLES:
        summary_path = root / dataset / f"{role}_s{seed}" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = summary.get("dataset")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"Evaluation summary lacks dataset provenance; rerun with schema v2: {summary_path}"
            )
        roles[role] = provenance
    roots = {str(item.get("root")) for item in roles.values()}
    revisions = {item.get("revision") for item in roles.values()}
    fingerprints = {item.get("fingerprint") for item in roles.values()}
    if len(roots) != 1 or len(revisions) != 1 or len(fingerprints) != 1:
        raise ValueError(
            f"Evaluation dataset provenance differs for {dataset} seed {seed}: {roles}"
        )
    return {
        "root": next(iter(roots)),
        "revision": next(iter(revisions)),
        "fingerprint": next(iter(fingerprints)),
        "roles": roles,
    }


def _fmt(value: Any, digits: int = 5) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SpaceTime 三方对照实验结果",
        "",
        f"预注册分支：**{result['decision']['branch']}**",
        "",
        result["decision"]["conclusion"],
        "",
        "## 三方共同 endpoint 绝对指标",
        "",
        "| 数据集 | seed | 原始 WCM MSE / cMSE / r / bias | mosaic MSE / cMSE / r / bias | SpaceTime MSE / cMSE / r / bias | episodes / endpoints |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in result["protocol"]["datasets"]:
        for seed in result["protocol"]["seeds"]:
            row = result["absolute_metrics"][dataset][str(seed)]
            coverage = row["baseline"]
            lines.append(
                f"| {dataset} | {seed} | {_fmt(row['baseline']['mse'])} / {_fmt(row['baseline']['centered_mse'])} / {_fmt(row['baseline']['pearson'])} / {_fmt(row['baseline']['mean_bias'])} "
                f"| {_fmt(row['mosaic']['mse'])} / {_fmt(row['mosaic']['centered_mse'])} / {_fmt(row['mosaic']['pearson'])} / {_fmt(row['mosaic']['mean_bias'])} "
                f"| {_fmt(row['spacetime']['mse'])} / {_fmt(row['spacetime']['centered_mse'])} / {_fmt(row['spacetime']['pearson'])} / {_fmt(row['spacetime']['mean_bias'])} "
                f"| {coverage['episodes']} / {coverage['endpoints']} |"
            )
    lines.extend(
        [
            "",
            "## OOD paired 增量",
            "",
            "| 对照 | seed | ΔMSE [95% CI] | Δcentered MSE [95% CI] | ΔPearson [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("spacetime_vs_baseline", "spacetime_vs_mosaic"):
        label = "原始 WCM" if name.endswith("baseline") else "mosaic"
        for seed in result["protocol"]["seeds"]:
            item = result["comparisons"]["ood"][str(seed)][name]
            mse_ci = item["mse_paired_episode_bootstrap_ci95"]
            centered_ci = item["centered_mse_paired_episode_bootstrap_ci95"]
            pearson_ci = item["pearson_paired_episode_bootstrap_ci95"]
            lines.append(
                f"| {label} | {seed} "
                f"| {_fmt(item['mse_difference_candidate_minus_baseline'])} [{_fmt(mse_ci[0])}, {_fmt(mse_ci[1])}] "
                f"| {_fmt(item['centered_mse_difference_candidate_minus_baseline'])} [{_fmt(centered_ci[0])}, {_fmt(centered_ci[1])}] "
                f"| {_fmt(item['pearson_difference_candidate_minus_baseline'])} [{_fmt(pearson_ci[0])}, {_fmt(pearson_ci[1])}] |"
            )
    lines.extend(["", "## 自动门禁", ""])
    for name, gate in result["gates"].items():
        lines.append(f"- {name}: {'PASS' if gate['passed'] else 'FAIL'}")
        for check, passed in gate["checks"].items():
            lines.append(f"  - {check}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 效率",
            "",
            "| seed | 模型 | 参数量 | 输入帧 | 视觉 token | forward ms / sample | batch |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in result["protocol"]["seeds"]:
        efficiency = result["absolute_metrics"]["ood"][str(seed)]["efficiency"]
        for role in MAIN_ROLES:
            item = efficiency[role]
            lines.append(
                f"| {seed} | {role} | {item['parameters'] if item['parameters'] is not None else 'NA'} "
                f"| {item['input_frames']} | {item['visual_tokens']} "
                f"| {_fmt(item['forward_ms_per_sample'], 3)} "
                f"| {item['latency_batch_size'] if item['latency_batch_size'] is not None else 'NA'} |"
            )
    lines.extend(
        [
            "",
            "注：全部绝对指标和 paired 增量均在每个数据集/seed 的三方共同 endpoint 上计算；OOD 决定分支，5cut 仅作支持性结果。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    root: Path,
    datasets: list[str],
    seeds: list[int],
    bootstrap_samples: int,
) -> dict[str, Any]:
    if "ood" not in datasets:
        raise ValueError("The preregistered decision requires the OOD dataset.")
    if len(seeds) != 2:
        raise ValueError("The preregistered gates require exactly two model seeds.")
    comparisons_dir = root / "comparisons"
    manifests_dir = root / "endpoint_manifests"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    absolute: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    for dataset in datasets:
        absolute[dataset] = {}
        comparisons[dataset] = {}
        manifests[dataset] = {}
        provenance[dataset] = {}
        for seed in seeds:
            paths = {role: _curve(root, dataset, role, seed) for role in ALL_ROLES}
            manifest_path = manifests_dir / f"{dataset}_threeway_s{seed}.json"
            rows, common, manifest = build_common_manifest(
                paths,
                dataset=dataset,
                seed=seed,
                output=manifest_path,
            )
            missing_staged = common - set(rows["staged"])
            if missing_staged:
                raise ValueError(
                    f"Staged diagnostic is missing shared endpoints for {dataset} seed {seed}: "
                    f"{sorted(missing_staged)[:10]}"
                )
            manifests[dataset][str(seed)] = {
                **{key: value for key, value in manifest.items() if key != "endpoints"},
                "path": str(manifest_path.resolve()),
            }
            provenance[dataset][str(seed)] = _dataset_provenance(root, dataset, seed)
            absolute[dataset][str(seed)] = {
                role: absolute_metrics(rows[role], common) for role in ALL_ROLES
            }
            absolute[dataset][str(seed)]["efficiency"] = {
                role: _efficiency(root, dataset, role, seed) for role in ALL_ROLES
            }
            comparisons[dataset][str(seed)] = {}
            for name, (baseline_role, candidate_role) in COMPARISONS.items():
                item = compare_rows(
                    rows[baseline_role],
                    rows[candidate_role],
                    baseline_path=paths[baseline_role],
                    candidate_path=paths[candidate_role],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                    endpoint_keys=common,
                    endpoint_manifest=manifest_path,
                )
                comparison_path = comparisons_dir / f"{dataset}_{name}_s{seed}.json"
                comparison_path.write_text(
                    json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                comparisons[dataset][str(seed)][name] = item

    baseline_comparisons = [
        comparisons["ood"][str(seed)]["spacetime_vs_baseline"] for seed in seeds
    ]
    mosaic_comparisons = [
        comparisons["ood"][str(seed)]["spacetime_vs_mosaic"] for seed in seeds
    ]
    baseline_result = baseline_gate(baseline_comparisons)
    mosaic_result = mosaic_gate(mosaic_comparisons)
    return {
        "protocol": {
            "datasets": datasets,
            "seeds": seeds,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed_policy": "model_seed",
            "primary_decision_dataset": "ood",
            "alignment": "three_primary_models_common_endpoints",
        },
        "endpoint_manifests": manifests,
        "dataset_provenance": provenance,
        "absolute_metrics": absolute,
        "comparisons": comparisons,
        "gates": {
            "spacetime_vs_baseline": baseline_result,
            "spacetime_vs_mosaic": mosaic_result,
        },
        "decision": choose_branch(
            baseline_result,
            mosaic_result,
            baseline_comparisons,
            mosaic_comparisons,
        ),
    }


def run() -> None:
    args = parser().parse_args()
    result = analyze(args.eval_root, args.datasets, args.seeds, args.bootstrap_samples)
    output_json = args.output_json or args.eval_root / "threeway_results.json"
    output_markdown = args.output_markdown or args.eval_root / "threeway_results.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    print(f"JSON: {output_json.resolve()}")
    print(f"Markdown: {output_markdown.resolve()}")


if __name__ == "__main__":
    run()
