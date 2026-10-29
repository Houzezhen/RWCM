"""Write LaTeX table bodies from data/results.json into tables/*.tex."""
from pathlib import Path
import json

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R = json.loads((HERE / "data" / "results.json").read_text(encoding="utf-8"))
T = HERE / "tables"
T.mkdir(exist_ok=True)
SEEDS = ("3072", "42", "1337")


def ms(vals, nd=3, bold=False):
    v = np.asarray(vals, dtype=float)
    s = f"{v.mean():.{nd}f}" + (f" $\\pm$ {v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")
    return f"\\textbf{{{s}}}" if bold else s


def get(block, m, k):
    by = R[block].get(m, {})
    return [d[k] for d in by.values() if d.get(k) is not None]


# Table 1 main comparison
rows = [("WCM~\\citep{fei2026wcm}", "WCM", "single frame", "ood_wcm_manifest"),
        ("$\\pi^{*}_{0.6}$ value~\\citep{pistar06}", "pi06", "single frame", "ood_wcm_manifest"),
        ("VIP~\\citep{ma2023vip}", "VIP", "8 frames, span 120", "ood"),
        ("IQL critic~\\citep{kostrikov2022iql}", "IQL", "8 frames, span 120", "ood"),
        ("STMemory-60", "STMemory-60", "8 frames, span 60", "ood"),
        ("STMemory-120", "STMemory-120", "8 frames, span 120", "ood")]
lines = []
for label, m, ctx, block in rows:
    best = m == "STMemory-120"
    n = len(get(block, m, "mse"))
    idm, idr = get("indist", m, "mse"), get("indist", m, "pearson")
    marks = ["\\dagger"] * (block == "ood_wcm_manifest") + ["\\ast"] * (n == 1)
    cells = [label + (f"$^{{{','.join(marks)}}}$" if marks else ""), ctx,
             ms(get(block, m, "mse"), bold=best), ms(get(block, m, "pearson"), 2, bold=best),
             ms(get(block, m, "centered_mse"), bold=best),
             ms(idm, 4) + ("$^{\\ast}$" if len(idm) == 1 else "") if idm else "--",
             ms(idr, 2) if idr else "--"]
    lines.append(" & ".join(cells) + r" \\")
    if m == "IQL":
        lines.append(r"\midrule")
(T / "main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table 2 memory span, per seed
lines = []
for i, s in enumerate(SEEDS, 1):
    a, b = R["ood"]["STMemory-60"][s], R["ood"]["STMemory-120"][s]
    d = R["span"][s]
    lo, hi = d["d_pearson_ci"]
    lines.append(f"Seed {i} & {a['mse']:.4f} & \\textbf{{{b['mse']:.4f}}} & {d['rel_mse_reduction_pct']:.1f} "
                 f"& {a['pearson']:.3f} & \\textbf{{{b['pearson']:.3f}}} & [{lo:.3f}, {hi:.3f}] \\\\")
a = [R["ood"]["STMemory-60"][s]["mse"] for s in SEEDS]; b = [R["ood"]["STMemory-120"][s]["mse"] for s in SEEDS]
ra = [R["ood"]["STMemory-60"][s]["pearson"] for s in SEEDS]; rb = [R["ood"]["STMemory-120"][s]["pearson"] for s in SEEDS]
rel = np.mean([R["span"][s]["rel_mse_reduction_pct"] for s in SEEDS])
lines.append(r"\midrule")
lines.append(f"Mean & {np.mean(a):.4f} & \\textbf{{{np.mean(b):.4f}}} & {rel:.1f} & {np.mean(ra):.3f} & "
             f"\\textbf{{{np.mean(rb):.3f}}} & -- \\\\")
(T / "span.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table 3 outcome classification
lines = []
for m in ("WCM", "STMemory-60", "STMemory-120"):
    C = R["classification"][m]
    best = m == "STMemory-120"
    acc = [100 * C[s]["accuracy"] for s in SEEDS]
    bal = [100 * C[s]["balanced_accuracy"] for s in SEEDS]
    fr = [100 * C[s]["failures_detected"] / 5 for s in SEEDS]
    au = [C[s]["auroc"] for s in SEEDS]
    lines.append(" & ".join([m, ms(acc, 1, best), ms(bal, 1, best), ms(fr, 1, best), ms(au, 3, best)]) + r" \\")
(T / "classification.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Table A per seed OOD
lines = []
for m, block in (("WCM", "ood_wcm_manifest"), ("STMemory-60", "ood"), ("STMemory-120", "ood")):
    cells = [m]
    for s in SEEDS:
        d = R[block][m][s]
        cells += [f"{d['mse']:.4f}", f"{d['pearson']:.3f}"]
    lines.append(" & ".join(cells) + r" \\")
(T / "per_seed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# numbers quoted in the text
E = R["efficiency"]
q = {
    "wcm_to_60_pct": np.mean([R["wcm_vs_st60"][s]["rel_mse_reduction_pct"] for s in SEEDS]),
    "vip_pct": R["baselines_vs_st120"]["VIP"]["rel_mse_reduction_pct"],
    "iql_pct": R["baselines_vs_st120"]["IQL"]["rel_mse_reduction_pct"],
    "pi06_to_60_pct": [-R["pi06_vs"]["STMemory-60"][s]["rel_mse_change_pct"] for s in ("3072", "42")],
    "span_pct_min": min(R["span"][s]["rel_mse_reduction_pct"] for s in SEEDS),
    "span_pct_max": max(R["span"][s]["rel_mse_reduction_pct"] for s in SEEDS),
    "span_pct_mean": rel,
    "eff": E,
}
print(json.dumps(q, indent=1))
for f in sorted(T.glob("*.tex")):
    print("==", f.name); print(f.read_text(encoding="utf-8"))
