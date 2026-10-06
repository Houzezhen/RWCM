"""Render episode-level SpaceTime H120 risk figures from validation outputs.

Usage: py -3.11 scripts/make_spacetime_risk_figures.py
The five checkpoint-validation predictions must exist for every seed. A missing
fraction is an error: the script never interpolates probabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/results_wcm_to_spacetime/risk_accuracy"
OUTPUT = ROOT / "assets/figures_wcm_to_spacetime/manuscript"
SEEDS = (3072, 42, 1337)
PROGRESS = (0.0, 0.25, 0.5, 0.75, 1.0)
SUFFIX = ("_f0", "_f025", "_f05", "_f075", "")
INK = "#243139"
MUTED = "#627177"
RULE = "#D6DFDF"
SUCCESS = "#087C79"
FAILURE = "#B54B43"
PALETTE = LinearSegmentedColormap.from_list(
    "failure_probability", ("#F5F8F7", "#F2D8CF", "#CB7562", "#963C40")
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def risk_score(row: dict, path: Path) -> float:
    """The existing reports and newer evaluation CLI use two field names."""
    values = [float(row[key]) for key in ("failure_probability", "failure_score") if key in row]
    if not values or (len(values) == 2 and not math.isclose(values[0], values[1], abs_tol=1e-8)):
        raise ValueError(f"Missing or inconsistent risk score in {path}")
    probability = values[0]
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"Invalid risk score in {path}")
    return probability


def load_risk() -> tuple[dict[int, dict[int, np.ndarray]], dict[int, int], dict[int, int]]:
    paths = {
        (seed, step): SOURCE / f"h120_s{seed}{suffix}.json"
        for seed in SEEDS
        for step, suffix in enumerate(SUFFIX)
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing measured H120 episode-progress results; generate/sync these files "
            "before rendering (no values are interpolated):\n  " + "\n  ".join(missing)
        )

    predictions: dict[int, dict[int, np.ndarray]] = {}
    labels: dict[int, int] = {}
    lengths: dict[int, int] = {}
    expected_ids: set[int] | None = None
    dataset_root: str | None = None
    for seed in SEEDS:
        by_step: dict[int, dict[int, dict]] = {}
        for step, fraction in enumerate(PROGRESS):
            path = paths[(seed, step)]
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("split") != "val" or report.get("metrics", {}).get("n_episodes") != 18:
                raise ValueError(f"Expected 18 checkpoint-validation episodes: {path}")
            current_root = report.get("dataset_root")
            if dataset_root is None:
                dataset_root = current_root
            elif current_root != dataset_root:
                raise ValueError(f"Dataset root differs from other risk reports: {path}")
            if step < 4 and not math.isclose(
                float(report.get("episode_fraction", -1)), fraction, abs_tol=1e-9
            ):
                raise ValueError(f"Unexpected episode fraction in {path}")
            rows = report["episodes"]
            row_by_id = {int(row["episode_id"]): row for row in rows}
            if len(rows) != 18 or len(row_by_id) != len(rows):
                raise ValueError(f"Missing or duplicate episode IDs in {path}")
            ids = set(row_by_id)
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"Episode IDs differ from other risk reports: {path}")
            for episode_id, row in row_by_id.items():
                label = int(row["true_failure"])
                if label not in (0, 1) or (episode_id in labels and labels[episode_id] != label):
                    raise ValueError(f"Inconsistent outcome for episode {episode_id}: {path}")
                labels[episode_id] = label
                risk_score(row, path)
            by_step[step] = row_by_id
        predictions[seed] = {}
        assert expected_ids is not None
        for episode_id in expected_ids:
            positions = [int(by_step[step][episode_id]["frame_index"]) for step in range(5)]
            if positions != sorted(positions) or positions[-1] < positions[0]:
                raise ValueError(f"Non-monotone sampled frame indices: seed {seed}, episode {episode_id}")
            episode_length = positions[-1] - positions[0] + 1
            if episode_id in lengths and lengths[episode_id] != episode_length:
                raise ValueError(f"Episode length differs across seeds: episode {episode_id}")
            lengths[episode_id] = episode_length
            predictions[seed][episode_id] = np.array(
                [risk_score(by_step[step][episode_id], paths[(seed, step)]) for step in range(5)]
            )

    if sorted(labels.values()).count(1) != 5 or len(labels) != 18:
        raise ValueError("Expected 13 successes and 5 failures in the checkpoint-validation split")
    return predictions, labels, lengths


def export(fig: plt.Figure, stem: str, exclude_axes: tuple[plt.Axes, ...] = ()) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_path = os.environ.get("WCM_FIGURE_AUDIT_SCRIPTS")
    if audit_path:
        sys.path.insert(0, audit_path)
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=str(OUTPUT / f"{stem}.alignment.json"),
            overlay_svg=str(OUTPUT / f"{stem}.alignment.svg"),
            exclude_axes=list(exclude_axes),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_trajectory(
    predictions: dict[int, dict[int, np.ndarray]], labels: dict[int, int]
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    fig.subplots_adjust(left=.11, right=.965, bottom=.235, top=.71)
    fig.text(.11, .925, "Episode risk across observed progress", color=INK,
             fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.text(.11, .865, "SpaceTime H120  |  checkpoint validation split  |  18 episodes, 3 seeds",
             color=MUTED, fontsize=7.2, ha="left", va="top")

    rows: list[list[object]] = []
    for true_failure, color, label in ((0, SUCCESS, "True success"), (1, FAILURE, "True failure")):
        ids = sorted(episode_id for episode_id, outcome in labels.items() if outcome == true_failure)
        per_episode = np.array([np.mean([predictions[seed][episode_id] for seed in SEEDS], axis=0)
                                for episode_id in ids])
        # Each translucent curve is one episode, averaged over seeds; n remains 13/5.
        for episode_id, values in zip(ids, per_episode):
            ax.plot(PROGRESS, values, color=color, alpha=.16, linewidth=.85, zorder=1)
            rows.extend((episode_id, true_failure, progress, float(value))
                        for progress, value in zip(PROGRESS, values))
        for seed in SEEDS:
            seed_mean = np.mean([predictions[seed][episode_id] for episode_id in ids], axis=0)
            ax.plot(PROGRESS, seed_mean, color=color, alpha=.34, linewidth=.95,
                    linestyle=(0, (2, 2)), zorder=2)
        group_mean = np.mean(per_episode, axis=0)
        ax.plot(PROGRESS, group_mean, color=color, linewidth=2.3,
                marker="o" if true_failure else "s", markersize=5.2,
                markeredgecolor="white", markeredgewidth=.65,
                label=f"{label} (n={len(ids)})", zorder=4)

    ax.axhline(.5, color="#A5B2B3", linestyle=(0, (3, 3)), linewidth=.8, zorder=0)
    ax.set(xlim=(-.025, 1.025), ylim=(-.055, 1.055),
           xticks=PROGRESS, xticklabels=("0", ".25", ".50", ".75", "1.0"),
           yticks=(0, .25, .5, .75, 1),
           xlabel="Episode progress", ylabel="Failure score")
    ax.grid(axis="y", color="#E6EBEA", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=5)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2,
              frameon=False, handlelength=2.0, columnspacing=3.0)
    fig.text(.11, .085,
             "Thick: outcome mean  |  dashed: seed means  |  faint: episode means (3 seeds)  |  gray: 0.5 threshold",
             color=MUTED, fontsize=6.4, ha="left", va="bottom")
    fig.text(.11, .045, "Collection source and outcome co-vary in this validation split.",
             color=MUTED, fontsize=6.4, ha="left", va="bottom")
    export(fig, "fig6_risk_trajectory")

    with (OUTPUT / "fig6_risk_trajectory.source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode_id", "true_failure", "episode_progress", "score_mean_across_3_seeds"))
        writer.writerows(rows)


def make_heatmap(
    predictions: dict[int, dict[int, np.ndarray]],
    labels: dict[int, int],
    lengths: dict[int, int],
) -> None:
    seed = 3072
    episode_order = sorted(labels, key=lambda episode_id: (-labels[episode_id], lengths[episode_id], episode_id))
    matrix = np.array([predictions[seed][episode_id] for episode_id in episode_order])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=.21, right=.87, bottom=.18, top=.79)
    fig.text(.21, .945, "Validation risk landscape", color=INK,
             fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.text(.21, .895, "SpaceTime H120  |  seed 3072  |  18 episodes ordered by outcome, then duration",
             color=MUTED, fontsize=7.2, ha="left", va="top")
    fig.legend(handles=(Patch(facecolor=FAILURE, label="True failure (5)"),
                        Patch(facecolor=SUCCESS, label="True success (13)")),
               loc="upper left", bbox_to_anchor=(.21, .85), ncol=2,
               frameon=False, handlelength=1.1, columnspacing=2.0, fontsize=7)

    image = ax.imshow(matrix, interpolation="nearest", aspect="auto", cmap=PALETTE,
                      norm=Normalize(vmin=0, vmax=1), extent=(-.125, 1.125, 17.5, -.5))
    for row_index, episode_id in enumerate(episode_order):
        ax.add_patch(Rectangle((-.19, row_index - .47), .075, .94,
                               color=FAILURE if labels[episode_id] else SUCCESS,
                               linewidth=0, clip_on=False))
    ax.axhline(4.5, color="white", linewidth=2.0, zorder=3)
    ax.set(xlim=(-.205, 1.125), ylim=(17.5, -.5),
           xticks=PROGRESS, xticklabels=("0", ".25", ".50", ".75", "1.0"),
           yticks=np.arange(18),
           yticklabels=[f"Episode {episode_id}" for episode_id in episode_order],
           xlabel="Episode progress", ylabel="Episode")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines[["left", "bottom"]].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, pad=.025, fraction=.045, ticks=(0, .25, .5, .75, 1))
    colorbar.set_label("Failure score", color=INK, fontsize=7.5, labelpad=8)
    colorbar.outline.set_edgecolor(RULE)
    colorbar.ax.tick_params(labelsize=6.8, length=0, pad=5)
    fig.text(.21, .047, "Checkpoint validation split; collection source and outcome co-vary.",
             color=MUTED, fontsize=6.4, ha="left", va="bottom")
    export(fig, "fig7_risk_heatmap", exclude_axes=(colorbar.ax,))

    with (OUTPUT / "fig7_risk_heatmap.source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("row", "episode_id", "true_failure", "episode_length_frames",
                         "episode_progress", "failure_score", "seed"))
        for row_index, episode_id in enumerate(episode_order):
            for progress, score in zip(PROGRESS, matrix[row_index]):
                writer.writerow((row_index + 1, episode_id, labels[episode_id], lengths[episode_id],
                                 progress, float(score), seed))


def load_frame2() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Return matched episode labels and scores, one row per seed and model."""
    model_pattern = {
        "Original WCM": "baseline_s{seed}_value_frame2.json",
        "SpaceTime H120": "h120_s{seed}_frame2.json",
    }
    expected_ids: list[int] | None = None
    expected_labels: np.ndarray | None = None
    scores: dict[str, list[np.ndarray]] = {model: [] for model in model_pattern}
    raw_scores: dict[str, dict[int, np.ndarray]] = {model: {} for model in model_pattern}
    for model, pattern in model_pattern.items():
        for seed in SEEDS:
            path = SOURCE / pattern.format(seed=seed)
            if not path.is_file():
                raise FileNotFoundError(f"Missing measured frame-2 result: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if (report.get("split") != "val" or report.get("frame_index") != 2 or
                    report.get("metrics", {}).get("n_episodes") != 18):
                raise ValueError(f"Unexpected population or frame index in {path}")
            rows = report["episodes"]
            row_by_id = {int(row["episode_id"]): row for row in rows}
            if len(row_by_id) != 18 or len(rows) != 18:
                raise ValueError(f"Missing or duplicate episode IDs in {path}")
            ids = sorted(row_by_id)
            labels = np.array([int(row_by_id[episode_id]["true_failure"]) for episode_id in ids])
            if expected_ids is None:
                expected_ids, expected_labels = ids, labels
            elif ids != expected_ids or not np.array_equal(labels, expected_labels):
                raise ValueError(f"Frame-2 episodes or outcomes differ in {path}")
            if model == "Original WCM":
                values = np.array([float(row_by_id[episode_id]["failure_score"])
                                   for episode_id in ids])
                if not all(math.isclose(float(row_by_id[episode_id]["failure_score"]),
                                        -float(row_by_id[episode_id]["raw_value"]),
                                        abs_tol=1e-6) for episode_id in ids):
                    raise ValueError(f"Original WCM score is not -V(s) in {path}")
            else:
                values = np.array([float(row_by_id[episode_id]["failure_score"])
                                   for episode_id in ids])
                if np.any((values < 0) | (values > 1)):
                    raise ValueError(f"Risk scores outside [0,1] in {path}")
            if not np.isfinite(values).all() or not np.isin(labels, (0, 1)).all():
                raise ValueError(f"Invalid scores or outcomes in {path}")
            scores[model].append(values)
            raw_scores[model][seed] = values
    assert expected_labels is not None
    if int(expected_labels.sum()) != 5:
        raise ValueError("Expected 5 failures among 18 checkpoint-validation episodes")
    return expected_labels, {model: np.array(values) for model, values in scores.items()}, raw_scores


def binary_curves(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, ...]:
    """Exact threshold ROC/PR coordinates, AUROC and average precision."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    threshold_ends = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(scores) - 1]
    tp = np.cumsum(sorted_labels)[threshold_ends].astype(float)
    fp = (threshold_ends + 1).astype(float) - tp
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC/PR require both outcomes")
    fpr = np.r_[0.0, fp / negatives]
    tpr = np.r_[0.0, tp / positives]
    recall = tpr
    precision = np.r_[1.0, tp / (tp + fp)]
    auc = float(np.trapz(tpr, fpr))
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return fpr, tpr, recall, precision, auc, ap


def curve_on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray, kind: str) -> np.ndarray:
    if kind == "roc":
        unique_x, first, counts = np.unique(x, return_index=True, return_counts=True)
        upper_y = np.array([np.max(y[start:start + count])
                            for start, count in zip(first, counts)])
        return np.interp(grid, unique_x, upper_y)
    if kind == "pr":
        return y[np.minimum(np.searchsorted(x, grid, side="left"), len(y) - 1)]
    raise ValueError(kind)


def pr_display_curve(recall: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the best precision at each recall; extra negatives add no recall."""
    unique_recall, first, counts = np.unique(recall, return_index=True, return_counts=True)
    visible_precision = np.array([np.max(precision[start:start + count])
                                  for start, count in zip(first, counts)])
    return unique_recall, visible_precision


def make_roc_pr() -> None:
    labels, scores, raw_scores = load_frame2()
    names = ("Original WCM", "SpaceTime H120")
    colors = ("#83939A", FAILURE)
    x_grid = np.linspace(0, 1, 101)
    n_boot = 3000
    rng = np.random.default_rng(120)
    success_indices = np.flatnonzero(labels == 0)
    failure_indices = np.flatnonzero(labels == 1)
    # One stratified episode draw is shared across models and seeds.
    boot_curves = np.empty((n_boot, 2, 2, len(x_grid)))
    boot_metrics = np.empty((n_boot, 2, 2))
    for sample_index in range(n_boot):
        draw = np.r_[rng.choice(success_indices, len(success_indices), replace=True),
                     rng.choice(failure_indices, len(failure_indices), replace=True)]
        sample_labels = labels[draw]
        for model_index, name in enumerate(names):
            seed_curves = np.empty((2, len(SEEDS), len(x_grid)))
            seed_metrics = np.empty((2, len(SEEDS)))
            for seed_index in range(len(SEEDS)):
                fpr, tpr, recall, precision, auc, ap = binary_curves(
                    sample_labels, scores[name][seed_index, draw]
                )
                seed_curves[0, seed_index] = curve_on_grid(fpr, tpr, x_grid, "roc")
                seed_curves[1, seed_index] = curve_on_grid(recall, precision, x_grid, "pr")
                seed_metrics[:, seed_index] = (auc, ap)
            boot_curves[sample_index, model_index] = seed_curves.mean(axis=1)
            boot_metrics[sample_index, model_index] = seed_metrics.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(left=.095, right=.975, bottom=.245, top=.73, wspace=.23)
    fig.text(.095, .945, "Frame-2 failure-score discrimination", fontsize=10.5,
             fontweight="bold", color=INK, ha="left", va="top")
    fig.text(.095, .885, "Checkpoint validation  |  n=18 episodes (5 failures)  |  3 matched seeds",
             fontsize=7.2, color=MUTED, ha="left", va="top")
    fig.legend(handles=[Line2D([], [], color=color, linewidth=2.2, label=name)
                        for name, color in zip(names, colors)],
               loc="upper left", bbox_to_anchor=(.095, .845), ncol=2,
               frameon=False, handlelength=2.5, columnspacing=2.7, fontsize=7)
    axes[0].set_title("a   ROC", loc="left", fontsize=8.5, fontweight="bold", pad=9)
    axes[1].set_title("b   Precision-recall", loc="left", fontsize=8.5, fontweight="bold", pad=9)
    axes[0].plot((0, 1), (0, 1), color="#C1CDCF", linewidth=.8,
                 linestyle=(0, (3, 3)), zorder=0)
    axes[1].axhline(5 / 18, color="#C1CDCF", linewidth=.8,
                    linestyle=(0, (3, 3)), zorder=0)
    summary = {}
    source_rows = []
    for model_index, (name, color) in enumerate(zip(names, colors)):
        observed = np.empty((2, len(SEEDS), len(x_grid)))
        observed_metrics = np.empty((2, len(SEEDS)))
        for seed_index, seed in enumerate(SEEDS):
            fpr, tpr, recall, precision, auc, ap = binary_curves(labels, raw_scores[name][seed])
            observed[0, seed_index] = curve_on_grid(fpr, tpr, x_grid, "roc")
            observed[1, seed_index] = curve_on_grid(recall, precision, x_grid, "pr")
            observed_metrics[:, seed_index] = (auc, ap)
            axes[0].plot(fpr, tpr, color=color, linewidth=.85, alpha=.42, zorder=2)
            visible_recall, visible_precision = pr_display_curve(recall, precision)
            axes[1].step(visible_recall, visible_precision, where="post", color=color,
                         linewidth=.85, alpha=.42, zorder=2)
        for curve_index, ax in enumerate(axes):
            lower, upper = np.percentile(boot_curves[:, model_index, curve_index],
                                         (2.5, 97.5), axis=0)
            mean_curve = observed[curve_index].mean(axis=0)
            ax.fill_between(x_grid, lower, upper, color=color, alpha=.13,
                            linewidth=0, zorder=1)
            ax.plot(x_grid, mean_curve, color=color, linewidth=2.2, zorder=3)
            source_rows.extend((name, "roc" if curve_index == 0 else "pr", float(x),
                                float(value), float(lo), float(hi))
                               for x, value, lo, hi in zip(x_grid, mean_curve, lower, upper))
        summary[name] = {}
        for metric_index, metric_name in enumerate(("AUROC", "average_precision")):
            summary[name][metric_name] = {
                "per_seed": dict(zip(map(str, SEEDS), observed_metrics[metric_index].tolist())),
                "mean_of_three_seeds": float(observed_metrics[metric_index].mean()),
                "paired_episode_bootstrap_ci95": np.percentile(
                    boot_metrics[:, model_index, metric_index], (2.5, 97.5)
                ).tolist(),
            }

    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, .25, .5, .75, 1),
               yticks=(0, .25, .5, .75, 1))
        ax.grid(color="#E8EEEE", linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=4)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(xlabel="Recall (failure)", ylabel="Precision (failure)")
    fig.text(.095, .10,
             "Thin: individual seeds; thick: seed mean; shading: 95% stratified paired episode bootstrap CI (3,000 draws).",
             fontsize=6.3, color=MUTED, ha="left", va="bottom")
    fig.text(.095, .06,
             "H120 separation is complete on this source-confounded checkpoint validation split; its bootstrap CI is degenerate.",
             fontsize=6.3, color=MUTED, ha="left", va="bottom")
    export(fig, "fig8_roc_pr")
    with (OUTPUT / "fig8_roc_pr.source.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "curve", "x", "mean_y_across_seeds", "ci95_low", "ci95_high"))
        writer.writerows(source_rows)
    (OUTPUT / "fig8_roc_pr.metrics.json").write_text(
        json.dumps({"population": {"split": "checkpoint_validation", "episodes": 18,
                                   "failures": 5, "seeds": SEEDS},
                    "bootstrap": {"unit": "episode", "paired_across_models_and_seeds": True,
                                  "stratified_by_outcome": True, "draws": n_boot,
                                  "seed": 120, "interval": "pointwise_percentile_95"},
                    "metrics": summary}, indent=2) + "\n", encoding="utf-8"
    )


def boundary_reflected_density(values: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian KDE on [0,1], with reflected kernels at both bounds."""
    distance = (grid[:, None] - values[None, :]) / bandwidth
    reflected_zero = (grid[:, None] + values[None, :]) / bandwidth
    reflected_one = (grid[:, None] - (2 - values[None, :])) / bandwidth
    return (np.exp(-.5 * distance ** 2) + np.exp(-.5 * reflected_zero ** 2) +
            np.exp(-.5 * reflected_one ** 2)).mean(axis=1)


def make_distribution() -> None:
    seed = 3072
    suffixes = ("_f0", "_f05", "_f075")
    times = ("0%", "50%", "75%")
    reports = []
    for suffix, fraction in zip(suffixes, (0, .5, .75)):
        path = SOURCE / f"h120_s{seed}{suffix}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing measured risk distribution: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("split") != "val" or
                not math.isclose(float(report.get("episode_fraction", -1)), fraction, abs_tol=1e-9)):
            raise ValueError(f"Unexpected split or progress in {path}")
        reports.append({int(row["episode_id"]): row for row in report["episodes"]})
    ids = sorted(reports[0])
    if len(ids) != 18 or any(sorted(report) != ids for report in reports):
        raise ValueError("Expected the same 18 episodes at all distribution checkpoints")
    labels = {episode_id: int(reports[0][episode_id]["true_failure"]) for episode_id in ids}
    if sum(labels.values()) != 5 or any(
        int(report[episode_id]["true_failure"]) != labels[episode_id]
        for report in reports for episode_id in ids
    ):
        raise ValueError("Outcome labels differ across distribution checkpoints")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35), sharey=True)
    fig.subplots_adjust(left=.095, right=.975, top=.73, bottom=.23, wspace=.14)
    fig.text(.095, .945, "Risk-score distributions over episode progress", color=INK,
             fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.text(.095, .885, "SpaceTime H120  |  seed 3072  |  checkpoint validation split",
             color=MUTED, fontsize=7.2, ha="left", va="top")
    fig.text(.095, .822, "Violin: fixed-bandwidth boundary-reflected density   |   points: all 18 episodes at exact scores",
             color=MUTED, fontsize=6.5, ha="left", va="top")
    grid = np.linspace(0, 1, 401)
    source_rows = []
    for panel_index, (ax, report, time) in enumerate(zip(axes, reports, times)):
        ax.set_title(f"{chr(97 + panel_index)}   {time}", loc="left",
                     fontsize=8.5, fontweight="bold", pad=9)
        for label, center, color, name in ((0, 0, SUCCESS, "Success"),
                                           (1, 1, FAILURE, "Failure")):
            group_ids = [episode_id for episode_id in ids if labels[episode_id] == label]
            scores = np.array([risk_score(report[episode_id], SOURCE / f"h120_s{seed}{suffixes[panel_index]}.json")
                               for episode_id in group_ids])
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Failure scores must be finite probabilities")
            density = boundary_reflected_density(scores, grid, bandwidth=.055)
            half_width = .23 * density / density.max()
            ax.fill_betweenx(grid, center - half_width, center + half_width,
                             facecolor=color, edgecolor=color, linewidth=.8,
                             alpha=.23, zorder=1)
            # Jitter only x; point y values are never altered.
            spread = .33 if len(scores) > 10 else .18
            jitter = np.linspace(-spread, spread, len(scores))
            ax.scatter(center + jitter, scores, s=17, facecolor=color,
                       edgecolor="white", linewidth=.45, zorder=3, clip_on=False)
            source_rows.extend((time, episode_id, label, float(score), seed)
                               for episode_id, score in zip(group_ids, scores))
        ax.axhline(.5, color="#AAB6B8", linestyle=(0, (3, 3)), linewidth=.75, zorder=0)
        ax.set(xlim=(-.42, 1.42), ylim=(0, 1), xticks=(0, 1),
               xticklabels=("Success\nn=13", "Failure\nn=5"), yticks=(0, .25, .5, .75, 1))
        ax.grid(axis="y", color="#E8EEEE", linewidth=.45)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.tick_params(axis="y", length=0, pad=4)
    axes[0].set_ylabel("Failure score")
    fig.text(.095, .09,
             "Violin smoothing bandwidth 0.055; failure scores saturate near 1 and success scores near 0.",
             color=MUTED, fontsize=6.3, ha="left", va="bottom")
    fig.text(.095, .05, "Collection source and outcome co-vary in this validation split.",
             color=MUTED, fontsize=6.3, ha="left", va="bottom")
    export(fig, "fig9_risk_distribution")
    with (OUTPUT / "fig9_risk_distribution.source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode_progress", "episode_id", "true_failure", "failure_score", "seed"))
        writer.writerows(source_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=("6", "7", "8", "9"),
                        default=("6", "7", "8", "9"),
                        help="Figure numbers to render (default: all).")
    requested = set(parser.parse_args().figures)
    if requested & {"6", "7"}:
        predictions, labels, lengths = load_risk()
        if "6" in requested:
            make_trajectory(predictions, labels)
        if "7" in requested:
            make_heatmap(predictions, labels, lengths)
    if "8" in requested:
        make_roc_pr()
    if "9" in requested:
        make_distribution()
    print(f"Wrote risk figures to {OUTPUT}")


if __name__ == "__main__":
    main()
