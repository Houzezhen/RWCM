"""Generate publication figures for the WCM -> SpaceTime narrative.

Narrative: original single-frame WCM -> SpaceTime (explicit spatiotemporal history
encoding, 8 frames -> 64 tokens), including a T60 vs T120 history-span comparison.
Mosaic is intentionally excluded.

Sources:
  baseline (original WCM): outputs/eval_spacetime_threeway/{ood,5cut}/baseline_s{seed}
  t60 (SpaceTime 60-step span): outputs/eval_spacetime_t120_pair/{ood,5cut}/t60_s{seed}
  t120 (SpaceTime 120-step span): outputs/eval_spacetime_t120_pair/{ood,5cut}/t120_s{seed}

Outputs: outputs/figures_wcm_to_spacetime/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THREEWAY = ROOT / "outputs/eval_spacetime_threeway"
T120 = ROOT / "outputs/eval_spacetime_t120_pair"
OUT = ROOT / "outputs/figures_wcm_to_spacetime"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [3072, 42]
ROLES = ["baseline", "t60", "t120"]
LABEL = {
    "baseline": "WCM (single-frame)",
    "t60": "SpaceTime-T60",
    "t120": "SpaceTime-T120 (ours)",
}
COLOR = {"baseline": "#888888", "t60": "#5b8db8", "t120": "#c0392b"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(dataset: str, role: str, seed: int) -> pd.DataFrame:
    base = THREEWAY if role == "baseline" else T120
    p = base / dataset / f"{role}_s{seed}" / "episode_curves" / "episode_curves.csv"
    return pd.read_csv(p).rename(columns={"episode_id": "episode", "frame_index": "frame"})


def common_eps(dataset: str) -> set:
    eps = None
    for role in ROLES:
        for seed in SEEDS:
            e = set(load(dataset, role, seed)["episode"].unique())
            eps = e if eps is None else (eps & e)
    return eps


def per_episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep, g in df.groupby("episode"):
        y = g["return"].to_numpy()
        p = g["value"].to_numpy()
        corr = float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 and np.std(y) > 0 else np.nan
        rows.append({"episode": ep, "mse": float(np.mean((p - y) ** 2)), "corr": corr})
    return pd.DataFrame(rows).set_index("episode")


# ---------------- Figure 1: OOD summary, three models, two seeds ----------------
def fig1_summary():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, ylabel in zip(axes, ["mse", "corr"], ["MSE", "Pearson r"]):
        for j, role in enumerate(ROLES):
            for i, seed in enumerate(SEEDS):
                m = per_episode_metrics(load("ood", role, seed))[metric].dropna()
                x = j + (-0.16 if seed == 3072 else 0.16)
                bp = ax.boxplot(
                    [m], positions=[x], widths=0.26, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"),
                )
                bp["boxes"][0].set_facecolor(COLOR[role] if seed == 3072 else "white")
                bp["boxes"][0].set_edgecolor(COLOR[role])
        ax.set_xticks(range(len(ROLES)))
        ax.set_xticklabels([LABEL[r] for r in ROLES], fontsize=9)
        ax.set_ylabel(ylabel)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.85", edgecolor="0.3"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3"),
    ]
    axes[0].legend(handles, ["seed 3072", "seed 42"], loc="upper right", frameon=False)
    fig.suptitle("OOD generalization: per-episode value-prediction quality", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ood_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 2: representative episode curves ----------------
def fig2_episode_curves():
    eps_ok = common_eps("ood")
    m0 = per_episode_metrics(load("ood", "baseline", 3072))
    m6 = per_episode_metrics(load("ood", "t60", 3072))
    m12 = per_episode_metrics(load("ood", "t120", 3072))
    j = m0.join(m6, rsuffix="_t60").join(m12, rsuffix="_t120").dropna()
    j = j[j.index.isin(eps_ok)]
    j["gain"] = j["corr_t120"] - j["corr"]
    picks = j.sort_values("gain", ascending=False).index[:2].tolist()

    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 2.4 * len(picks)), squeeze=False)
    for ax, ep in zip(axes[:, 0], picks):
        for role in ROLES:
            g = load("ood", role, 3072)
            g = g[g["episode"] == ep].sort_values("frame")
            ax.plot(g["frame"], g["value"], color=COLOR[role], lw=1.5,
                    label=f"{LABEL[role]} pred")
        g = load("ood", "baseline", 3072)
        g = g[g["episode"] == ep].sort_values("frame")
        ax.plot(g["frame"], g["return"], color="black", ls="--", lw=1.2, label="return (GT)")
        ax.set_xlabel("frame")
        ax.set_ylabel("value")
        ax.set_title(f"OOD episode {ep}", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_episode_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return picks


# ---------------- Figure 3: architecture schematic ----------------
def fig3_architecture():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    for i, t in enumerate(["t-120", "t-86", "t-51", "t"]):
        box(0.02 + i * 0.09, 0.64, 0.07, 0.22, t, fc="#eaf2f8")
    ax.text(0.02, 0.94, "8 history frames x 2 cameras (span 120 steps)", fontsize=9)

    box(0.13, 0.27, 0.13, 0.22, "shared ViT\n(per-frame)", fc="#d5e8d4")
    box(0.33, 0.27, 0.18, 0.22, "factorized temporal\nattention", fc="#d5e8d4")
    box(0.57, 0.27, 0.16, 0.22, "Perceiver\n1568 -> 64 tokens", fc="#f9e79f")
    box(0.79, 0.27, 0.15, 0.22, "WCM trunk\nvalue / risk / Q", fc="#fadbd8")
    for x0, x1 in [(0.20, 0.33), (0.46, 0.57), (0.73, 0.79)]:
        ax.annotate("", xy=(x1 + 0.01, 0.38), xytext=(x0 + 0.02, 0.38),
                    arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(0.14, 0.49), xytext=(0.10, 0.64),
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.06,
            "SpaceTime encoder: fixed 64 tokens regardless of history span; no mosaic input at deployment",
            ha="center", fontsize=10, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(OUT / "fig3_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 4: paired per-episode delta histograms ----------------
def fig4_paired_delta():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    eps_ok = common_eps("ood")
    for seed in SEEDS:
        m0 = per_episode_metrics(load("ood", "baseline", seed))
        for role in ["t60", "t120"]:
            mr = per_episode_metrics(load("ood", role, seed))
            j = m0.join(mr, lsuffix="_b", rsuffix="_r").dropna()
            j = j[j.index.isin(eps_ok)]
            axes[0].hist(j["mse_r"] - j["mse_b"], bins=40, alpha=0.5,
                         color=COLOR[role], label=f"{LABEL[role]} - WCM (s{seed})")
            axes[1].hist(j["corr_r"] - j["corr_b"], bins=40, alpha=0.5,
                         color=COLOR[role], label=f"{LABEL[role]} - WCM (s{seed})")
    axes[0].axvline(0, color="k", lw=1)
    axes[1].axvline(0, color="k", lw=1)
    axes[0].set_xlabel("per-episode MSE delta vs WCM")
    axes[1].set_xlabel("per-episode Pearson delta vs WCM")
    for ax in axes:
        ax.set_ylabel("episodes")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("OOD: paired per-episode improvement over original WCM", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_paired_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 5: T60 vs T120 span comparison ----------------
def fig5_span():
    eps_ok = common_eps("ood")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for seed in SEEDS:
        m6 = per_episode_metrics(load("ood", "t60", seed))
        m12 = per_episode_metrics(load("ood", "t120", seed))
        j = m6.join(m12, lsuffix="_t60", rsuffix="_t120").dropna()
        j = j[j.index.isin(eps_ok)]
        axes[0].scatter(j["mse_t60"], j["mse_t120"], s=10, alpha=0.6, label=f"seed {seed}")
        axes[1].scatter(j["corr_t60"], j["corr_t120"], s=10, alpha=0.6, label=f"seed {seed}")
    lim = axes[0].get_xlim()
    axes[0].plot(lim, lim, color="k", lw=1, ls="--")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("per-episode MSE, T60"); axes[0].set_ylabel("per-episode MSE, T120")
    lim = axes[1].get_xlim()
    axes[1].plot(lim, lim, color="k", lw=1, ls="--")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("per-episode Pearson, T60"); axes[1].set_ylabel("per-episode Pearson, T120")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("History span 60 -> 120 steps: same 64-token budget, better OOD", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_span_t60_vs_t120.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_summary()
    picks = fig2_episode_curves()
    fig3_architecture()
    fig4_paired_delta()
    fig5_span()
    summary = {
        "fig2_picked_episodes": picks,
        "roles": ROLES,
        "sources": {
            "baseline": str(THREEWAY),
            "t60_t120": str(T120),
        },
    }
    (OUT / "figures_summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", OUT)
