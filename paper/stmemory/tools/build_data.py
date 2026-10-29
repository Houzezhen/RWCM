"""Collect every number used in the redesigned manuscript into data/results.json.

Sources (read only, nothing is recomputed from model outputs):
  RWCM/assets/results_wcm_to_spacetime/**.json         paired OOD and in-distribution evaluations
  RWCM/outputs/eval_spacetime_threeway/threeway_results.md  in-distribution absolute metrics, efficiency
  RWCM/outputs/domain_baselines_20261022/comparisons/*.json  VIP and IQL critic on the 22,082-endpoint manifest
  data/source/fig03_value_diagnosis_margins.source.csv        per-episode value margins at frame 2
Run:  python -I tools/build_data.py
"""
from pathlib import Path
import csv
import json

import numpy as np

HERE = Path(__file__).resolve().parents[1]

RWCM = HERE.parents[1]
A = RWCM / "assets" / "results_wcm_to_spacetime"
DOM = RWCM / "outputs" / "domain_baselines_20261022" / "comparisons"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
SEEDS = (3072, 42, 1337)


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["_source"] = str(Path(path).relative_to(RWCM)).replace("\\", "/")
    return d


def side(d, which):
    """Absolute metrics of the baseline or candidate side of a paired report."""
    p = "baseline" if which == "baseline" else "candidate"
    return {"mse": d[f"{p}_mse"], "pearson": d[f"{p}_pearson"],
            "centered_mse": d.get(f"{p}_centered_mse"), "bias": d.get(f"{p}_bias"),
            "episodes": d["episodes"], "endpoints": d["endpoints"], "source": d["_source"]}


def delta(d):
    return {"d_mse": d["candidate_mse"] - d["baseline_mse"],
            "d_mse_ci": d.get("mse_paired_episode_bootstrap_ci95") or d["paired_episode_bootstrap_ci95"],
            "d_pearson": d["candidate_pearson"] - d["baseline_pearson"],
            "d_pearson_ci": d["pearson_paired_episode_bootstrap_ci95"],
            "rel_mse_reduction_pct": -100 * d["relative_mse_change"] if "relative_mse_change" in d
            else 100 * (1 - d["candidate_mse"] / d["baseline_mse"]),
            "episodes": d["episodes"], "endpoints": d["endpoints"], "source": d["_source"]}


res = {"ood": {}, "ood_wcm_manifest": {}, "span": {}, "wcm_vs_st60": {}, "wcm_vs_st120": {},
       "baselines_vs_st120": {}, "indist": {}, "classification": {}, "efficiency": {}}

# ---- OOD, 22,082-endpoint manifest: STMemory-60 and STMemory-120 (all three seeds)
span_files = {3072: A / "temporal_span" / "ood_t120_vs_t60_s3072.json",
              42: A / "temporal_span" / "ood_t120_vs_t60_s42.json",
              1337: A / "confirmatory_s1337" / "ood_h120_vs_h60_s1337.json"}
for s, f in span_files.items():
    d = load(f)
    res["ood"].setdefault("STMemory-60", {})[s] = side(d, "baseline")
    res["ood"].setdefault("STMemory-120", {})[s] = side(d, "candidate")
    res["span"][s] = delta(d)

# ---- External baselines (seed 42) against STMemory-120 on the same 22,082 endpoints
for name, key in (("VIP", "vip"), ("IQL", "iql")):
    d = load(DOM / f"ours_vs_{key}_s42.json")
    res["ood"].setdefault(name, {})[42] = side(d, "baseline")
    res["baselines_vs_st120"][name] = delta(d)
    assert abs(d["candidate_mse"] - res["ood"]["STMemory-120"][42]["mse"]) < 1e-9

# ---- Original WCM: defined on the 21,832-endpoint manifest (its own observation window)
wcm60 = {3072: A / "original_to_h60" / "ood_spacetime_vs_baseline_s3072.json",
         42: A / "original_to_h60" / "ood_spacetime_vs_baseline_s42.json",
         1337: A / "confirmatory_s1337" / "ood_h60_vs_baseline_s1337.json"}
for s, f in wcm60.items():
    d = load(f)
    res["ood_wcm_manifest"].setdefault("WCM", {})[s] = side(d, "baseline")
    res["ood_wcm_manifest"].setdefault("STMemory-60", {})[s] = side(d, "candidate")
    res["wcm_vs_st60"][s] = delta(d)
d = load(A / "confirmatory_s1337" / "ood_h120_vs_baseline_s1337.json")
res["ood_wcm_manifest"].setdefault("STMemory-120", {})[1337] = side(d, "candidate")
res["wcm_vs_st120"][1337] = delta(d)
d = load(RWCM / "outputs" / "domain_baselines_20261022" / "report" / "ours_vs_original_wcm_s42.json")
res["ood_wcm_manifest"]["STMemory-120"][42] = side(d, "candidate")
res["wcm_vs_st120"][42] = delta(d)

# ---- pi*0.6 value model: single trained model, scored on the same 21,832-endpoint manifest as WCM
P = json.loads((HERE / "data/source/pistar06_value_ood.json").read_text(encoding="utf-8"))
res["ood_wcm_manifest"]["pi06"] = {"single": dict(P["absolute"], source="data/source/pistar06_value_ood.json")}
res["pi06_vs"] = {}
for m, by in P["paired"].items():
    for s, p in by.items():
        other = res["ood_wcm_manifest"][m][int(s)]["mse"]
        assert abs(other + p["d_mse"] - P["absolute"]["mse"]) < 6e-4  # report deltas agree with saved metrics
        res["pi06_vs"].setdefault(m, {})[int(s)] = dict(p, rel_mse_change_pct=100 * (other / P["absolute"]["mse"] - 1))

# ---- In-distribution test set (145 held-out episodes)
# seeds 3072 and 42 from the three-way common manifest (27,727 endpoints)
three = {("WCM", 3072): (0.04054, 0.33480), ("WCM", 42): (0.03675, 0.40707),
         ("STMemory-60", 3072): (0.00999, 0.87337), ("STMemory-60", 42): (0.00944, 0.87058)}
for (m, s), (mse, r) in three.items():
    res["indist"].setdefault(m, {})[s] = {"mse": mse, "pearson": r, "episodes": 145, "endpoints": 27727,
                                          "source": "outputs/eval_spacetime_threeway/threeway_results.md"}
for m, f, which in (("WCM", "5cut_h60_vs_baseline_s1337.json", "baseline"),
                    ("STMemory-60", "5cut_h60_vs_baseline_s1337.json", "candidate"),
                    ("STMemory-120", "5cut_h120_vs_baseline_s1337.json", "candidate")):
    res["indist"].setdefault(m, {})[1337] = side(load(A / "confirmatory_s1337" / f), which)

# ---- Outcome classification at frame 2 from saved held-out value margins
rows = list(csv.DictReader(open(HERE / "data/source/fig03_value_diagnosis_margins.source.csv", encoding="utf-8-sig")))
name = {"Original WCM": "WCM", "SpaceTime H60": "STMemory-60", "SpaceTime H120": "STMemory-120"}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


for m in name.values():
    for s in SEEDS:
        ep = [r for r in rows if r["panel"] in "DEF" and name[r["model"]] == m and int(r["seed"]) == s]
        assert len(ep) == 18
        y = np.array([int(r["true_success"]) for r in ep])
        yhat = np.array([int(r["predicted_success"]) for r in ep])
        v = np.array([float(r["value"]) for r in ep])
        tp_s = int(((y == 1) & (yhat == 1)).sum()); tp_f = int(((y == 0) & (yhat == 0)).sum())
        res["classification"].setdefault(m, {})[s] = {
            "accuracy": float((y == yhat).mean()), "correct": int((y == yhat).sum()),
            "balanced_accuracy": 0.5 * (tp_s / 13 + tp_f / 5), "failures_detected": tp_f,
            "auroc": auroc(v[y == 1], v[y == 0]),
            "margins": [float(r["value_margin"]) for r in ep], "labels": y.tolist()}

# ---- Efficiency (three-way report, batch 4, mean over the two measured seeds)
res["efficiency"] = {"WCM": {"params_m": 170.604289, "frames": 1, "memory_tokens": 1, "ms": (6.947 + 6.985) / 2},
                     "STMemory": {"params_m": 176.126211, "frames": 8, "memory_tokens": 64, "ms": (12.924 + 12.787) / 2},
                     "source": "outputs/eval_spacetime_threeway/threeway_results.md"}


def summary(block):
    out = {}
    for m, by in block.items():
        out[m] = {}
        for k in ("mse", "pearson", "centered_mse"):
            v = [by[s][k] for s in by if by[s].get(k) is not None]
            if v:
                out[m][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                             "n": len(v)}
    return out


res["summary"] = {"ood": summary(res["ood"]), "ood_wcm_manifest": summary(res["ood_wcm_manifest"]),
                  "indist": summary(res["indist"])}
cls = {}
for m, by in res["classification"].items():
    cls[m] = {k: float(np.mean([by[s][k] for s in SEEDS])) for k in
              ("accuracy", "balanced_accuracy", "failures_detected", "auroc")}
res["summary"]["classification"] = cls

(OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res["summary"], indent=1))
for s in SEEDS:
    print("span", s, {k: res["span"][s][k] for k in ("d_mse", "d_mse_ci", "d_pearson", "d_pearson_ci", "rel_mse_reduction_pct")})
for s in SEEDS:
    print("wcm->60", s, round(res["wcm_vs_st60"][s]["rel_mse_reduction_pct"], 2))
for k, v in res["baselines_vs_st120"].items():
    print(k, round(v["rel_mse_reduction_pct"], 2), v["d_mse_ci"])
