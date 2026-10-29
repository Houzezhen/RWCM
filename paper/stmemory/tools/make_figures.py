"""Data figures for the redesigned manuscript (Figs. 3-6 and Fig. A1).

Every value comes from data/results.json (built by tools/build_data.py), the archived
training-curve CSV and the extracted video frames. Run:  python -I tools/make_figures.py
"""
from pathlib import Path
import csv
import glob
import json
from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "data" / "results.json").read_text(encoding="utf-8"))
SEEDS = ("3072", "42", "1337")

for f in glob.glob(str(HERE / "assets" / "fonts" / "Inter-*.ttf")):  # TrueType copies made by tools/otf2ttf.py
    fm.fontManager.addfont(f)

MM = 1 / 25.4
W = 180 * MM
INK, SUB, RULE, PANEL = "#1B2433", "#5B6678", "#CBD3DE", "#F5F7FA"
CYAN, COBALT, CORAL = "#19A7CE", "#1F4E9C", "#E4572E"
COL = {"WCM": "#9AA5B4", "pi06": "#A9C0A0", "VIP": "#C9B79F", "IQL": "#B4A7CC", "STMemory-60": CYAN, "STMemory-120": COBALT}
SEED_MARK = {"3072": "o", "42": "s", "1337": "^"}

mpl.rcParams.update({
    "font.family": "Inter", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB,
    "axes.edgecolor": RULE, "axes.facecolor": PANEL, "axes.linewidth": 0,
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 0.9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 3, "ytick.major.pad": 3,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "cm",
})


def letter(ax, s, title=None, x=0.0, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="semibold", ha="left", va="bottom")
    if title:
        ax.text(x + 0.055, y, title, transform=ax.transAxes, fontsize=7, fontweight="medium",
                ha="left", va="bottom")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03, dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("wrote", stem)


def seed_vals(block, model, key):
    by = R[block][model]
    return [d[key] for d in by.values()]


# --------------------------------------------------------------------------- Fig. 3 main results
def fig_main():
    fig = plt.figure(figsize=(W, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.2], wspace=0.42, left=0.11, right=0.99, top=0.86, bottom=0.17)
    models = ["WCM", "pi06", "VIP", "IQL", "STMemory-60", "STMemory-120"]
    names = ["WCM$^\\dagger$", "$\\pi^{*}_{0.6}$ value$^\\dagger$", "VIP", "IQL critic", "STMemory-60", "STMemory-120"]
    for j, (key, xlab, fmt) in enumerate((("mse", "Value MSE, lower is better", "{:.3f}"),
                                          ("pearson", "Pearson correlation, higher is better", "{:.2f}"))):
        ax = fig.add_subplot(gs[0, j])
        for i, m in enumerate(models):
            block = "ood_wcm_manifest" if m in ("WCM", "pi06") else "ood"
            v = seed_vals(block, m, key)
            y = len(models) - 1 - i
            ax.barh(y, np.mean(v), height=0.62, color=COL[m], zorder=2, lw=0)
            if len(v) > 1:
                for s in SEEDS:
                    if s in R[block][m]:
                        ax.scatter(R[block][m][s][key], y + {"3072": -0.17, "42": 0, "1337": 0.17}[s], s=7,
                                   marker=SEED_MARK[s], color="white", edgecolor=INK, lw=0.45, zorder=3)
            ax.text(max(v) + (0.007 if key == "mse" else 0.05), y, fmt.format(np.mean(v)), va="center",
                    ha="left", fontsize=6, color=INK if m.startswith("ST") else SUB,
                    fontweight="semibold" if m == "STMemory-120" else "regular")
        ax.set_yticks(range(len(models))[::-1], names if j == 0 else [""] * len(models))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlab)
        if key == "mse":
            ax.set_xlim(0, 0.175)
        else:
            ax.set_xlim(0, 1.08)
        for t, m in zip(ax.get_yticklabels(), models):
            if m.startswith("ST"):
                t.set_color(INK); t.set_fontweight("semibold")
        letter(ax, "ab"[j], "Out-of-distribution test" if j == 0 else "Out-of-distribution ranking", x=-0.02 if j else -0.42)

    # c: error and ranking jointly, both test sets
    ax = fig.add_subplot(gs[0, 2])
    for block, mk, fill in (("ood", "o", True), ("indist", "D", False)):
        for m in ("WCM", "pi06", "VIP", "IQL", "STMemory-60", "STMemory-120"):
            src = "ood_wcm_manifest" if (m in ("WCM", "pi06") and block == "ood") else block
            if m not in R[src]:
                continue
            for s, d in R[src][m].items():
                ax.scatter(d["mse"], d["pearson"], s=18 if fill else 15, marker=mk,
                           color=COL[m] if fill else "white", edgecolor=COL[m], lw=0.9, zorder=3, alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlim(0.006, 0.25)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Value MSE, log scale")
    ax.set_ylabel("Pearson correlation")
    ax.annotate("better", xy=(0.0072, 0.95), xytext=(0.02, 0.72), fontsize=6, color=SUB,
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.6, mutation_scale=6))
    h = [Line2D([], [], ls="", marker="o", color=COL[m], ms=4, label=n) for m, n in
         zip(models, ["WCM", "$\\pi^{*}_{0.6}$ value", "VIP", "IQL critic", "STMemory-60", "STMemory-120"])]
    h += [Line2D([], [], ls="", marker="o", color=SUB, ms=4, label="OOD test"),
          Line2D([], [], ls="", marker="D", mfc="white", mec=SUB, ms=3.6, label="In-distribution test")]
    ax.legend(handles=h, loc="lower left", fontsize=5.6, handletextpad=0.2, labelspacing=0.3, borderaxespad=0.3)
    letter(ax, "c", "Error and ranking across test sets", x=-0.02)
    save(fig, "fig03_main_results")


# --------------------------------------------------------------------------- Fig. 4 memory span
def fig_span():
    fig = plt.figure(figsize=(W, 56 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.07, right=0.99, top=0.84, bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    for s in SEEDS:
        a, b = R["ood"]["STMemory-60"][s]["mse"], R["ood"]["STMemory-120"][s]["mse"]
        ax.plot([0, 1], [a, b], color=RULE, lw=1.0, zorder=1)
        ax.scatter([0], [a], s=22, marker=SEED_MARK[s], color=CYAN, ec="white", lw=0.5, zorder=3)
        ax.scatter([1], [b], s=22, marker=SEED_MARK[s], color=COBALT, ec="white", lw=0.5, zorder=3)
    ax.set_xticks([0, 1], ["Span 60", "Span 120"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0.01, 0.036)
    ax.set_ylabel("Value MSE")
    ax.grid(axis="x", visible=False)
    letter(ax, "a", "Error per seed", x=-0.2)

    ax = fig.add_subplot(gs[0, 1])
    ys = np.arange(3)[::-1]
    for y, s in zip(ys, SEEDS):
        d = R["span"][s]; base = R["ood"]["STMemory-60"][s]["mse"]
        rel = d["rel_mse_reduction_pct"]; lo, hi = [-100 * c / base for c in d["d_mse_ci"]][::-1]
        ax.barh(y, rel, height=0.55, color=COBALT, alpha=0.9, lw=0, zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=0.8, zorder=3)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.1, y + 0.1], color=INK, lw=0.8, zorder=3)
        ax.text(hi + 0.8, y, f"{rel:.1f}%", va="center", fontsize=6)
    ax.set_yticks(ys, [f"Seed {i}" for i in (1, 2, 3)])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 34)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("MSE reduction from wider span (%)")
    letter(ax, "b", "Relative error reduction", x=-0.2)

    ax = fig.add_subplot(gs[0, 2])
    for y, s in zip(ys, SEEDS):
        d = R["span"][s]
        lo, hi = d["d_pearson_ci"]
        ax.plot([lo, hi], [y, y], color=COBALT, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([d["d_pearson"]], [y], s=26, marker=SEED_MARK[s], color="white", ec=COBALT, lw=1.0, zorder=3)
    ax.axvline(0, color=SUB, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, [f"Seed {i}" for i in (1, 2, 3)])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.015, 0.05)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Pearson gain from wider span")
    letter(ax, "c", "Ranking gain with 95% CI", x=-0.2)
    save(fig, "fig04_memory_span")


# --------------------------------------------------------------------------- Fig. 5 outcome classification
def fig_cls():
    C = R["classification"]
    models = ["WCM", "STMemory-60", "STMemory-120"]
    fig = plt.figure(figsize=(W, 60 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.25, left=0.07, right=0.99, top=0.84, bottom=0.18)
    ax = fig.add_subplot(gs[0, 0])
    metrics = (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("auroc", "AUROC"))
    bw = 0.25
    for i, (k, _) in enumerate(metrics):
        for j, m in enumerate(models):
            v = [100 * C[m][s][k] for s in SEEDS]
            x = i + (j - 1) * bw
            ax.bar(x, np.mean(v), width=bw * 0.9, color=COL[m], lw=0, zorder=2)
            for s, val in zip(SEEDS, v):
                ax.scatter(x + {"3072": -0.06, "42": 0, "1337": 0.06}[s], val, s=6, marker=SEED_MARK[s],
                           color="white", ec=INK, lw=0.4, zorder=3)
    ax.plot([-0.45, 0.45], [100 * 13 / 18] * 2, color=SUB, lw=0.6, ls=(0, (3, 2)), zorder=4)
    ax.text(-0.45, 100 * 13 / 18 + 1.0, "always success", ha="left", va="bottom", fontsize=5.4, color=SUB, zorder=4)
    ax.plot([0.55, 2.45], [50, 50], color=SUB, lw=0.6, ls=(0, (1, 2)), zorder=4)
    ax.text(2.45, 51, "chance", ha="right", va="bottom", fontsize=5.4, color=SUB)
    ax.set_xticks(range(3), [n for _, n in metrics])
    ax.set_ylim(40, 116)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_ylabel("Score (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=COL[m], label=m) for m in models], loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.0, fontsize=5.8)
    letter(ax, "a", "Success and failure classification", x=-0.1)

    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(3)
    for j, m in enumerate(models):
        for s in SEEDS:
            mg = np.array(C[m][s]["margins"]); lab = np.array(C[m][s]["labels"])
            for val, y in zip(mg, lab):
                x = j + (-0.17 if y == 1 else 0.17) + rng.uniform(-0.09, 0.09)
                wrong = (val > 0) != (y == 1)
                c = COBALT if y == 1 else CORAL
                ax.scatter(x, val, s=9, marker="o", color="white" if wrong else c, ec=c, lw=0.7 if wrong else 0.2,
                           alpha=0.9, zorder=3)
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3), models)
    ax.set_ylabel(r"Held-out margin $\hat V-\tau$")
    ax.set_ylim(-0.75, 0.47)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=COBALT, ms=3.5, label="successful episode"),
                       Line2D([], [], ls="", marker="o", color=CORAL, ms=3.5, label="failed episode"),
                       Line2D([], [], ls="", marker="o", mfc="white", mec=SUB, ms=3.5, label="misclassified")],
              loc="lower right", fontsize=5.6, handletextpad=0.2)
    letter(ax, "b", "Value margins of every held-out episode", x=-0.1)
    save(fig, "fig05_outcome_classification")


# --------------------------------------------------------------------------- Fig. 6 qualitative
def fig_video():
    rows = (("ood_st120", "Successful out-of-distribution episode, STMemory-120"),
            ("fail_st60", "Failed episode, STMemory-60"))
    first = Image.open(HERE / "assets/video/ood_st120_0.png")
    ar = first.height / first.width
    gap = 1.0 * MM
    cw = (W - 4 * gap) / 5
    ch = cw * ar
    head = 4.2 * MM
    sub = 3.0 * MM
    H = 2 * (head + ch) + sub + 1.2 * MM
    fig = plt.figure(figsize=(W, H))
    prog = ("start", "25%", "50%", "75%", "end")
    for r, (key, title) in enumerate(rows):
        top = H - r * (head + ch + sub)
        fig.text(0, (top - head * 0.78) / H, "ab"[r], fontsize=8, fontweight="semibold", va="bottom")
        fig.text(3.6 * MM / W, (top - head * 0.78) / H, title, fontsize=7, fontweight="medium", va="bottom")
        for c in range(5):
            x0 = c * (cw + gap)
            ax = fig.add_axes([x0 / W, (top - head - ch) / H, cw / W, ch / H])
            ax.imshow(Image.open(HERE / f"assets/video/{key}_{c}.png"), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.text((x0 + cw / 2) / W, (top - head - ch - 0.6 * MM) / H, prog[c], ha="center", va="top",
                     fontsize=6, color=SUB)
    save(fig, "fig06_qualitative")


# --------------------------------------------------------------------------- Fig. A1 training curves
def fig_train():
    rows = list(csv.DictReader(open(HERE / "data/source/fig05_training_loss_curves.source.csv", encoding="utf-8-sig")))
    name = {"Original WCM": "WCM", "H60": "STMemory-60", "H120": "STMemory-120"}
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        data[r["panel"]][(name[r["model"]], r["seed"])][int(r["epoch"])] = float(r["value"])
    fig = plt.figure(figsize=(W, 92 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.22, left=0.07, right=0.99, top=0.9, bottom=0.09)
    spec = (("A", "Training value loss"), ("B", "Training latent loss"),
            ("C", "Validation value MSE"), ("D", "Validation latent MSE"))
    for i, (p, t) in enumerate(spec):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.axvspan(10.5, 13.4, color="#E9EDF3", lw=0, zorder=0)
        for m in ("WCM", "STMemory-60", "STMemory-120"):
            cur = [data[p][(m, s)] for s in SEEDS]
            ep = sorted(set.intersection(*[set(c) for c in cur]))
            arr = np.array([[c[e] for e in ep] for c in cur])
            ax.fill_between(ep, arr.min(0), arr.max(0), color=COL[m], alpha=0.18, lw=0)
            ax.plot(ep, arr.mean(0), color=COL[m], lw=1.4, marker="o", ms=2, mew=0)
        ax.set_yscale("log")
        ax.set_xlim(0.6, 13.4)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(which="minor", visible=False)
        letter(ax, "abcd"[i], t, x=-0.1)
    fig.legend(handles=[Line2D([], [], color=COL[m], lw=1.6, label=m) for m in ("WCM", "STMemory-60", "STMemory-120")]
               + [Patch(color="#E9EDF3", label="full fine-tuning")],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.2)
    save(fig, "figA1_training_curves")


if __name__ == "__main__":
    fig_main(); fig_span(); fig_cls(); fig_video(); fig_train()
