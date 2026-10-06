"""Render the WCM-to-SpaceTime manuscript figures from checked-in results."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

AUDIT_SCRIPTS = os.environ.get("WCM_FIGURE_AUDIT_SCRIPTS")
if AUDIT_SCRIPTS:
    sys.path.insert(0, AUDIT_SCRIPTS)
    from audit_panel_alignment import require_matplotlib_panel_alignment
else:
    require_matplotlib_panel_alignment = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "assets/results_wcm_to_spacetime"
OUTPUT = ROOT / "assets/figures_wcm_to_spacetime/manuscript"
VIDEOS = ROOT / "assets/videos_wcm_to_spacetime/t8_pair"
H120_VIDEOS = ROOT / "assets/videos_wcm_to_spacetime/t120"
CONCEPT_INSET = OUTPUT / "fig1_concept_inset.png"
SEEDS = (3072, 42, 1337)
INK = "#202B32"
QUIET = "#68757C"
RULE = "#CFD9DA"
BASE = "#829097"
H60 = "#147D77"
H120 = "#B9483F"
PAPER = "#FFFFFF"
SEED_MARKERS = {3072: "o", 42: "s", 1337: "^"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": PAPER,
})


def pair(name: str, seed: int) -> dict:
    if name == "original_to_h60":
        path = (
            RESULTS / "confirmatory_s1337/ood_h60_vs_baseline_s1337.json"
            if seed == 1337
            else RESULTS / f"original_to_h60/ood_spacetime_vs_baseline_s{seed}.json"
        )
        expected = 21832
    elif name == "h60_to_h120":
        path = (
            RESULTS / "confirmatory_s1337/ood_h120_vs_h60_s1337.json"
            if seed == 1337
            else RESULTS / f"temporal_span/ood_t120_vs_t60_s{seed}.json"
        )
        expected = 22082
    else:
        raise ValueError(name)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["seed"] != seed or result["episodes"] != 125 or result["endpoints"] != expected:
        raise ValueError(f"Unexpected paired population in {path}")
    if result["target_consistency"]["mismatched_endpoints"] != 0:
        raise ValueError(f"Target mismatch in {path}")
    return result


def label(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.035, text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom", color=INK)


def export(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(OUTPUT / f"{name}.alignment.json"),
            overlay_svg=str(OUTPUT / f"{name}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def block(ax: plt.Axes, x: float, y: float, width: float, height: float,
          title: str, subtitle: str, face: str, edge: str) -> None:
    ax.add_patch(Rectangle((x, y), width, height, facecolor=face,
                           edgecolor=edge, linewidth=1.1))
    ax.text(x + width / 2, y + height * .62, title, ha="center", va="center",
            fontsize=8, fontweight="bold", color=INK)
    ax.text(x + width / 2, y + height * .28, subtitle, ha="center", va="center",
            fontsize=6.4, color=QUIET)


def arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops={"arrowstyle": "->", "lw": 1.15, "color": QUIET})


def figure_system() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor(PAPER)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("auto")
    ax.axis("off")

    blue, purple, cyan, orange, red = "#387EE8", "#7055BB", "#19A6BA", "#EA8639", "#D95050"

    def panel(x: float, width: float) -> None:
        ax.add_patch(FancyBboxPatch((x, .075), width, .78,
                                   boxstyle="round,pad=0.006,rounding_size=0.014",
                                   facecolor="#FFFFFF", edgecolor="#D6E4F3",
                                   alpha=.93, linewidth=.9, zorder=1))

    def capsule(x: float, y: float, width: float, height: float,
                face: str, edge: str, radius: float = .011, lw: float = .8,
                zorder: int = 5) -> None:
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle=f"round,pad=0.002,rounding_size={radius}",
                                   facecolor=face, edgecolor=edge, linewidth=lw,
                                   zorder=zorder))

    def flow(start: tuple[float, float], end: tuple[float, float],
             linewidth: float = 4.0) -> None:
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        distance = np.linalg.norm(direction)
        unit = direction / distance
        shaft_end = np.asarray(end) - .014 * unit
        first, last = np.array(mpl.colors.to_rgb(blue)), np.array(mpl.colors.to_rgb(purple))
        for i in range(24):
            a, b = i / 24, (i + 1) / 24
            p0 = np.asarray(start) * (1 - a) + shaft_end * a
            p1 = np.asarray(start) * (1 - b) + shaft_end * b
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]), lw=linewidth,
                    color=first * (1 - b) + last * b, solid_capstyle="butt", zorder=4)
        normal = np.array((-unit[1], unit[0]))
        head = np.array((end, shaft_end + .008 * normal, shaft_end - .008 * normal))
        ax.add_patch(Polygon(head, closed=True, facecolor=purple,
                             edgecolor="none", zorder=4))

    def thin(start: tuple[float, float], end: tuple[float, float],
             color: str = "#A7B4C9", lw: float = 1.05) -> None:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "lw": lw, "mutation_scale": 7}, zorder=3)

    # The gradient and grid are one faint background image so grid lines never
    # compete with schematic labels or appear as process edges.
    yy, xx = np.mgrid[0:720, 0:1200]
    xx, yy = xx / 1199, yy / 719
    haze = np.exp(-(((xx - .83) / .7) ** 2 + ((yy - .72) / .8) ** 2))
    background = np.ones((720, 1200, 3))
    background -= haze[..., None] * np.array((.034, .018, .002))
    grid = ((np.mod(np.arange(720)[:, None], 36) == 0) |
            (np.mod(np.arange(1200)[None, :], 40) == 0))
    background[grid] *= .991
    ax.imshow(background, extent=(0, 1, 0, 1), aspect="auto", zorder=0)
    panel(.025, .555)
    panel(.606, .369)

    ax.text(.034, .962, "SpaceTime", fontsize=11, fontweight="bold",
            color="#24324C", va="center", zorder=6)
    ax.text(.184, .962, "Fixed-query visual history", fontsize=7.3,
            color="#60718E", va="center", zorder=6)
    ax.text(.05, .816, "01", fontsize=7.0, fontweight="bold", color=blue, va="center", zorder=6)
    ax.text(.083, .816, "TRAINING", fontsize=8.2, fontweight="bold",
            color="#24324C", va="center", zorder=6)
    ax.text(.628, .816, "02", fontsize=7.0, fontweight="bold", color=cyan, va="center", zorder=6)
    ax.text(.664, .816, "EPISODE-LEVEL EVALUATION", fontsize=8.0,
            fontweight="bold", color="#24324C", va="center", zorder=6)
    ax.plot((.05, .555), (.789, .789), color="#E3EAF4", lw=.8, zorder=2)
    ax.plot((.628, .95), (.789, .789), color="#E3EAF4", lw=.8, zorder=2)

    # Training input: eight temporal samples, each with two camera views.
    for dx, dy, fill in ((-.016, .025, "#E6F1FC"), (-.008, .012, "#DBEBFA"), (0, 0, "#CFE5F7")):
        capsule(.077 + dx, .594 + dy, .09, .113, fill, "#7FAED9", radius=.008, lw=.9)
    if not CONCEPT_INSET.exists():
        raise FileNotFoundError(CONCEPT_INSET)
    ax.imshow(Image.open(CONCEPT_INSET).convert("RGB"),
              extent=(.079, .165, .596, .705), aspect="auto", zorder=6)
    ax.text(.12, .563, "8 times × 2 views", ha="center", va="center",
            color="#2D506F", fontsize=6.8, fontweight="bold", zorder=6)
    for i in range(8):
        ax.add_patch(Circle((.086 + i * .0098, .531), .0027,
                            facecolor=blue, edgecolor="none", zorder=6))
    ax.text(.12, .482, "H60 / H120", ha="center", fontsize=6.1,
            color="#6B7F99", zorder=6)

    # Ordered capsule labels inside a single model glyph preserve architecture.
    hexagon = np.array(((.19, .602), (.233, .775), (.389, .775),
                        (.434, .602), (.389, .43), (.233, .43)))
    ax.add_patch(Polygon(hexagon, closed=True, facecolor="#F2EDFC",
                         edgecolor=purple, linewidth=1.5, zorder=3))
    ax.text(.312, .747, "SpaceTime", color="#4F3F87", fontsize=8.0,
            fontweight="bold", ha="center", va="center", zorder=7)
    for y, name in ((.676, "Shared ViT tokens"), (.616, "Temporal mixing"),
                    (.556, "64 queries · 384d"), (.496, "WCM trunk")):
        capsule(.232, y, .16, .043, "#FFFFFF", "#D5C8EF", radius=.014,
                lw=.7, zorder=5)
        ax.text(.312, y + .0215, name, fontsize=6.25, color="#4F3F87",
                ha="center", va="center", zorder=7)
    flow((.168, .645), (.191, .645), linewidth=3.9)
    flow((.434, .634), (.471, .634), linewidth=3.9)

    capsule(.479, .715, .083, .052, "#F8FAFD", "#D7DFEB", radius=.014)
    ax.text(.5205, .741, "V(s)", ha="center", va="center", fontsize=7.6,
            color="#4E6078", fontweight="bold", zorder=7)
    for halo, alpha in ((.066, .045), (.059, .07)):
        ax.add_patch(Circle((.521, .634), halo, color=orange, alpha=alpha,
                            ec="none", zorder=2))
    capsule(.473, .591, .096, .086, "#FFF1E3", orange,
            radius=.018, lw=1.35)
    ax.text(.521, .643, "Risk head", ha="center", va="center",
            fontsize=6.9, fontweight="bold", color="#9C4D16", zorder=7)
    ax.text(.521, .614, "p(fail)", ha="center", va="center",
            fontsize=6.4, color="#A4602D", zorder=7)
    capsule(.479, .492, .083, .052, "#F8FAFD", "#D7DFEB", radius=.014)
    ax.text(.5205, .518, "Q(s,a)", ha="center", va="center", fontsize=7.2,
            color="#4E6078", fontweight="bold", zorder=7)
    thin((.429, .715), (.478, .741))
    thin((.432, .526), (.478, .518))

    ax.text(.064, .368, "SUPERVISION", color="#BA6264", fontsize=6.6,
            fontweight="bold", zorder=6)
    ax.plot((.064, .555), (.35, .35), color="#F3D9DD", lw=.8, zorder=2)
    capsule(.077, .225, .207, .066, "#FFF9F9", "#F1CDD0", radius=.011)
    capsule(.311, .225, .246, .066, "#FFF9F9", "#F1CDD0", radius=.011)
    ax.text(.1805, .258, "return  →  V(s)", ha="center", va="center",
            fontsize=6.7, color="#A54851", fontweight="bold", zorder=7)
    ax.text(.434, .258, "1 − success  →  risk", ha="center", va="center",
            fontsize=6.7, color="#A54851", fontweight="bold", zorder=7)
    ax.plot((.18, .18, .239), (.294, .39, .44), color=red,
            lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.plot((.434, .434, .386), (.294, .39, .44), color=red,
            lw=1.0, ls=(0, (3, 3)), zorder=2)

    # Evaluation selects a fixed or terminal endpoint from overlapping windows.
    ax.text(.671, .743, "Selected windows", ha="center", va="center",
            fontsize=6.5, color="#416881", fontweight="bold", zorder=6)
    for dx, dy in ((-.012, .026), (-.006, .013), (0, 0)):
        capsule(.637 + dx, .604 + dy, .093, .087, "#E6F7FB",
                "#74C5D3", radius=.008, lw=.8)
    for x in (.651, .669, .687, .705):
        ax.plot((x, x + .010), (.644, .644), lw=2.0,
                color="#50ADCA", solid_capstyle="round", zorder=7)
    ax.text(.682, .572, "overlap", ha="center", fontsize=6.0,
            color="#748DA0", zorder=6)
    flow((.735, .647), (.785, .647), linewidth=3.8)
    capsule(.786, .606, .147, .083, "#F0EBFA", "#B7A4DD", radius=.018)
    ax.text(.8595, .652, "Deployed model", ha="center", va="center",
            fontsize=6.35, color="#60489D", fontweight="bold", zorder=7)
    ax.text(.8595, .624, "p(fail)", ha="center", va="center",
            fontsize=6.45, color="#60489D", zorder=7)
    flow((.859, .600), (.859, .557), linewidth=3.7)

    funnel = ((.784, .551), (.934, .551), (.887, .444),
              (.887, .416), (.831, .416), (.831, .444))
    ax.add_patch(Polygon(funnel, closed=True, facecolor="#DDF5F8",
                         edgecolor=cyan, linewidth=1.2, zorder=5))
    ax.text(.859, .503, "N → 1", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#18899D", zorder=7)
    ax.text(.859, .384, "fixed frame  /  terminal", ha="center", va="center",
            fontsize=6.1, color="#577C8D", zorder=7)
    flow((.859, .371), (.859, .344), linewidth=3.4)
    capsule(.744, .267, .225, .079, "#F1FAFC", "#A3DCE5", radius=.017)
    ax.text(.859, .316, "One episode prediction", ha="center", va="center",
            fontsize=7.1, fontweight="bold", color="#285E70", zorder=7)
    ax.text(.859, .286, "success / failure", ha="center", va="center",
            fontsize=6.0, color="#5D8390", zorder=7)

    thin((.812, .263), (.741, .215), color="#79AABB", lw=.95)
    thin((.906, .263), (.918, .215), color="#79AABB", lw=.95)
    capsule(.633, .117, .214, .100, "#FFFFFF", "#D8E7EB", radius=.012)
    ax.text(.740, .177, "Accuracy · balanced acc.", ha="center", va="center",
            fontsize=5.9, color="#3D6070", fontweight="bold", zorder=7)
    ax.text(.740, .147, "F1 · AUROC", ha="center", va="center",
            fontsize=6.0, color="#607A85", zorder=7)
    capsule(.867, .117, .094, .100, "#FFFFFF", "#D8E7EB", radius=.012)
    ax.add_patch(Rectangle((.906, .166), .016, .031, facecolor="#E5F4F7",
                           edgecolor=cyan, linewidth=.7, zorder=7))
    ax.text(.914, .139, "output.json", ha="center", va="center",
            fontsize=5.7, color="#3D6070", fontweight="bold", zorder=7)
    export(fig, "fig1_system")


def figure_original_to_h60() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    specs = (
        ("mse", "OOD MSE", (0, .17)),
        ("pearson", "OOD Pearson r", (0, 1.0)),
    )
    for ax, (metric, ylabel, limits) in zip(axes, specs):
        for seed in SEEDS:
            data = pair("original_to_h60", seed)
            before = data[f"baseline_{metric}"]
            after = data[f"candidate_{metric}"]
            ax.plot((0, 1), (before, after), color="#AFBDC0", lw=1.0, zorder=1)
            marker = SEED_MARKERS[seed]
            ax.scatter(0, before, s=48, marker=marker, facecolor=PAPER,
                       edgecolor=BASE, linewidth=1.4, zorder=3)
            ax.scatter(1, after, s=48, marker=marker, facecolor=H60,
                       edgecolor=H60, linewidth=1.1, zorder=3)
        ax.set(xticks=(0, 1), xticklabels=("Original WCM", "SpaceTime H60"),
               ylabel=ylabel, ylim=limits, xlim=(-.25, 1.25))
        ax.grid(axis="y", color="#E8EDEE", lw=.55)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_axisbelow(True)
    label(axes[0], "a")
    label(axes[1], "b")
    handles = [Line2D([], [], marker=SEED_MARKERS[s], linestyle="None",
                      markerfacecolor=PAPER, markeredgecolor=INK, markersize=5.5,
                      label=f"Seed {s}") for s in SEEDS]
    axes[1].legend(handles=handles, loc="lower right", ncol=1)
    export(fig, "fig2_ood_original_to_h60")


def figure_h60_to_h120() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, layout="constrained")
    specs = (
        ("mse", "MSE change: H120 − H60", (-.011, .0015), H120),
        ("pearson", "Pearson r change: H120 − H60", (-.014, .055), H60),
    )
    for ax, (metric, xlabel, limits, color) in zip(axes, specs):
        for index, seed in enumerate(SEEDS):
            data = pair("h60_to_h120", seed)
            delta = data[f"{metric}_difference_candidate_minus_baseline"]
            lower, upper = data[f"{metric}_paired_episode_bootstrap_ci95"]
            y = len(SEEDS) - 1 - index
            ax.plot((lower, upper), (y, y), color=color, lw=2.3,
                    solid_capstyle="round", zorder=2)
            ax.scatter(delta, y, s=43, marker=SEED_MARKERS[seed],
                       facecolor=color, edgecolor=PAPER, linewidth=.8, zorder=3)
        ax.axvline(0, color=QUIET, lw=.9, linestyle="--")
        ax.set(xlabel=xlabel, xlim=limits, ylim=(-.55, 2.55))
        ax.grid(axis="x", color="#E8EDEE", lw=.55)
        ax.set_axisbelow(True)
    axes[0].set(yticks=(2, 1, 0), yticklabels=("Seed 3072", "Seed 42", "Seed 1337"))
    axes[0].tick_params(axis="y", length=0, pad=6)
    axes[1].tick_params(axis="y", length=0)
    label(axes[0], "a")
    label(axes[1], "b")
    export(fig, "fig3_ood_span_effect")


def figure_failure_diagnostic() -> None:
    root = RESULTS / "risk_accuracy"
    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    models = (
        ("Original WCM", "baseline_s{seed}_value_frame2.json", BASE, -.24),
        ("SpaceTime H60", "h60_s{seed}_frame2.json", H60, 0),
        ("SpaceTime H120", "h120_s{seed}_frame2.json", H120, .24),
    )
    for index, (name, pattern, color, offset) in enumerate(models):
        for seed_index, seed in enumerate(SEEDS):
            data = json.loads((root / pattern.format(seed=seed)).read_text(encoding="utf-8"))
            if data["metrics"]["n_episodes"] != 18 or data["frame_index"] != 2:
                raise ValueError(f"Unexpected validation population for {name}, seed {seed}")
            correct = round(data["metrics"]["accuracy"] * 18)
            ax.scatter(index + offset + (seed_index - 1) * .085, correct,
                       s=55, marker=SEED_MARKERS[seed],
                       facecolor=color, edgecolor=PAPER, linewidth=.7, zorder=3)
    ax.set(xticks=range(3), xticklabels=[row[0] for row in models],
           ylabel="Correct episodes / 18", ylim=(9, 19), xlim=(-.5, 2.5),
           yticks=(10, 12, 14, 16, 18))
    ax.grid(axis="y", color="#E8EDEE", lw=.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=7)
    handles = [Line2D([], [], marker=SEED_MARKERS[s], linestyle="None",
                      markerfacecolor=PAPER, markeredgecolor=INK, markersize=5.5,
                      label=f"Seed {s}") for s in SEEDS]
    ax.legend(handles=handles, loc="upper left", ncol=3)
    export(fig, "fig4_validation_classification")


def video_frame(path: Path, seconds: float) -> Image.Image:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-ss", f"{seconds:.2f}", "-i", str(path), "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    image = subprocess.run(command, check=True, capture_output=True).stdout
    return Image.open(io.BytesIO(image)).convert("RGB")


def figure_video_examples() -> None:
    examples = (
        ("OOD episode 0", H120_VIDEOS / "t120_ood_s3072/episode-000000.mp4", (.10, 1.58, 3.05)),
        ("Failure episode 25", VIDEOS / "fail_spacetime_s3072/episode-000025.mp4", (.10, 1.71, 3.30)),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0))
    fig.subplots_adjust(left=.09, right=.995, bottom=.04, top=.86, wspace=.035, hspace=.13)
    for row, (case, video, times) in enumerate(examples):
        if not video.exists():
            raise FileNotFoundError(video)
        for col, (title, seconds) in enumerate(zip(("Start", "Midpoint", "End"), times)):
            ax = axes[row, col]
            ax.imshow(video_frame(video, seconds))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, color=INK, fontsize=8, fontweight="bold", pad=7)
            ax.text(.035, .035, f"{seconds:.2f} s", transform=ax.transAxes,
                    fontsize=7, color=PAPER, va="bottom", ha="left",
                    bbox={"facecolor": INK, "edgecolor": "none", "pad": 2.0})
        axes[row, 0].text(-.12, .5, f"{chr(97 + row)}  {case}",
                          transform=axes[row, 0].transAxes, color=INK,
                          fontsize=8.5, fontweight="bold", va="center", ha="center",
                          rotation=90, rotation_mode="anchor")
    export(fig, "fig5_video_examples")


def main() -> None:
    figure_system()
    figure_original_to_h60()
    figure_h60_to_h120()
    figure_failure_diagnostic()
    figure_video_examples()
    print(f"Figures written to {OUTPUT}")


if __name__ == "__main__":
    main()
