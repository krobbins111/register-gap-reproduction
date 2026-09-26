"""Generate the paper's appendix tables (LaTeX) from the result files.

Reads results/master_arms.csv + master_baselines.csv (run
analysis/aggregate_results.py first), the per-cell adapter / decomposition /
symmetric-ablation JSONs, and the apriori-d' table. Writes
appendix/tab_*.tex so every number in the appendix is regenerated from
data, never retyped.

Usage:  python analysis\\make_paper_tables.py [--out appendix]
Env:    ICLR_EXTRA_ROOTS  extra ':'-separated roots holding <ds>/*.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_ROOT, RESULTS_ROOT  # noqa: E402

ROOTS = [CACHE_ROOT, RESULTS_ROOT] + [Path(p) for p in
                                      os.environ.get("ICLR_EXTRA_ROOTS", "").split(":") if p]
LABEL = {"default": "SigLIP2", "siglipv1-so400m-384": "SigLIP-so400m",
         "clip-vitl14-laion2b": "CLIP ViT-L/14", "metaclip2-ww-huge": "MetaCLIP-2",
         "medsiglip": "MedSigLIP", "biomedclip": "BiomedCLIP",
         "georsclip-vitb32": "GeoRSCLIP", "remoteclip-vitl14": "RemoteCLIP",
         "fashionclip": "FashionCLIP", "marqo-fashionsiglip": "Marqo-FashionSigLIP",
         "bioclip": "BioCLIP", "bioclip2": "BioCLIP-2"}
NATIVE = {"medsiglip", "biomedclip", "georsclip-vitb32", "remoteclip-vitl14",
          "fashionclip", "marqo-fashionsiglip", "bioclip", "bioclip2"}
DSNAME = {"nwpu": "NWPU", "rocov2": "ROCOv2", "skincap": "SkinCAP",
          "treeoflife": "TreeOfLife", "rsicd": "RSICD", "scimmir": "SciMMIR",
          "fashion200k": "Fashion200k", "semart": "SemArt", "coco": "COCO",
          "goodnews": "GoodNews", "facad": "FACAD"}


def collect(pattern):
    found = {}
    for root in ROOTS:
        for p in glob.glob(str(root / "*" / pattern)):
            r = json.load(open(p, encoding="utf-8"))
            found[(r["dataset"], r.get("mtag", "default"))] = r
    return found


def f3(v, sign=False):
    if v is None or v == "":
        return "--"
    v = float(v)
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def table(out, name, caption, label, header, rows, align):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {name}.tex ({len(rows)} rows)")


def longtable(out, name, caption, label, header, rows, align):
    lines = [r"\begin{center}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             f"\\begin{{longtable}}{{{align}}}",
             f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule",
             r"\endfirsthead", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\end{longtable}", r"\end{center}", ""]
    (out / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {name}.tex ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                          / "appendix"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    dp = {}
    if os.environ.get("ICLR_DPRIME", "perquery") == "perquery":
        srcs = (RESULTS_ROOT / "dprime_apriori_perquery.csv",)
    else:
        srcs = (CACHE_ROOT / "dprime_analysis.csv",
                RESULTS_ROOT / "dprime_apriori_extra.csv")
    for dpp in srcs:
        if dpp.exists():
            for r in csv.DictReader(open(dpp)):
                dp[(r["dataset"], r["backbone"])] = float(r["dprime_apriori_train"])

    nmp = RESULTS_ROOT / "nearmiss_apriori.csv"
    nm = ({(r["dataset"], r["backbone"]): r for r in csv.DictReader(open(nmp))}
          if nmp.exists() else {})

    symj = collect("symmetric_ablation_*.json")

    # ── A: full grid, all cells ──────────────────────────────────────────────
    arms = list(csv.DictReader(open(RESULTS_ROOT / "master_arms.csv")))
    arms.sort(key=lambda r: (r["dataset"], r["mtag"]))
    rows = []
    for r in arms:
        if not r.get("both_R50"):
            continue
        sd = symj.get((r["dataset"], r["mtag"]), {}).get("arms", {}).get("both", {}).get("R@50_std")
        rows.append([DSNAME[r["dataset"]], LABEL[r["mtag"]], r["n_train"],
                     f3(r["raw_R10"]), f3(r["raw_R50"]),
                     f3(r["both_R10"]), f3(r["both_R50"]),
                     f"{sd:.3f}" if sd is not None else "--",
                     f3(r["both_gain50"], True), f3(r["shared_gain50"], True),
                     f3(nm[(r["dataset"], r["mtag"])]["train_N50_x3"])
                     if (r["dataset"], r["mtag"]) in nm else "--",
                     f3(dp.get((r["dataset"], r["mtag"])))])
    longtable(out, "tab_grid_full",
              f"Full grid, {len(rows)} dataset$\\times$encoder cells. Raw and corrected "
              "(two maps, one per side) R@10 / R@50 with the across-seed standard "
              "deviation of corrected R@50; gains at R@50 for the two-map and "
              "tied-map arms; a-priori near-miss mass $N_{50}$ and $d'$ from the "
              "training catalog. Two seeds, val-R@50 model selection.",
              "tab:grid_full",
              ["Collection", "Encoder", "$n_{\\text{train}}$", "raw R@10",
               "raw R@50", "R@10", "R@50", "sd", "$\\Delta$R@50", "tied $\\Delta$R@50",
               "$N_{50}$", "$d'$"], rows, "llrrrrrrrrrr")

    # ── A2: main-text table, default encoder in full + mean over encoders ───
    nmv = lambda k: (float(nm[k]["train_N50_x3"]) if k in nm else None)
    by_ds = {}
    for r in arms:
        if r.get("both_R50"):
            by_ds.setdefault(r["dataset"], []).append(r)
    order = sorted(by_ds, key=lambda d: -np.mean([float(r["both_gain50"]) for r in by_ds[d]]))
    rows = []
    for d in order:
        dflt = next(r for r in by_ds[d] if r["mtag"] == "default")
        g10 = [float(r["both_R10"]) - float(r["raw_R10"]) for r in by_ds[d]]
        g50 = [float(r["both_gain50"]) for r in by_ds[d]]
        nn = sum(1 for x in g50 if x >= -1e-9)
        rows.append([DSNAME[d],
                     f"{float(dflt['raw_R10']):.3f} $\\to$ {float(dflt['both_R10']):.3f}",
                     f3(float(dflt["both_R10"]) - float(dflt["raw_R10"]), True),
                     f"{float(dflt['raw_R50']):.3f} $\\to$ {float(dflt['both_R50']):.3f}",
                     f3(dflt["both_gain50"], True),
                     f3(nmv((d, "default"))),
                     f"{len(by_ds[d])}", f"{min(g10):+.2f} / {np.mean(g10):+.2f} / {max(g10):+.2f}",
                     f"{min(g50):+.2f} / {np.mean(g50):+.2f} / {max(g50):+.2f}", f"{nn}/{len(by_ds[d])}"])
    table(out, "tab_main",
          "Per collection. Left: the default encoder (SigLIP2), raw $\\to$ corrected \\Rk{10} and \\Rk{50} "
          "with the two-map gain and the a-priori near-miss mass $N_{50}$ of the training catalog. "
          "Right: over all of the collection's $n$ encoders, the min / mean / max gain and the number of "
          "encoders with a non-negative \\Rk{50} gain. Rows sorted by mean $\\Delta$\\Rk{50}. COCO is the control.",
          "tab:main",
          ["Collection", "\\Rk{10}", "$\\Delta$", "\\Rk{50}", "$\\Delta$", "$N_{50}$",
           "$n$", "$\\Delta$\\Rk{10} min/mean/max", "$\\Delta$\\Rk{50} min/mean/max", "$\\ge0$"],
          rows, "lrrrrrrrrr")

    # ── A3: two collections in full: every encoder ─────────────────────────
    rows = []
    for d in ("nwpu", "rocov2"):
        cells_ = sorted(by_ds[d], key=lambda r: (r["mtag"] in NATIVE, -float(r["both_gain50"])))
        for k, r in enumerate(cells_):
            sd = symj.get((d, r["mtag"]), {}).get("arms", {}).get("both", {}).get("R@50_std")
            rows.append([DSNAME[d] if k == 0 else "", LABEL[r["mtag"]] + (" (specialist)" if r["mtag"] in NATIVE else ""),
                         f3(r["raw_R10"]), f3(r["both_R10"]), f3(float(r["both_R10"]) - float(r["raw_R10"]), True),
                         f3(r["raw_R50"]), f3(r["both_R50"]), f3(r["both_gain50"], True),
                         f3(nmv((d, r["mtag"])))])
        if d == "nwpu":
            rows.append(["\\midrule"])
    rows = [r if r != ["\\midrule"] else None for r in rows]
    lines_rows = []
    for r in rows:
        lines_rows.append(r)
    # write with a midrule between the two collections
    tex_rows = []
    for r in lines_rows:
        tex_rows.append(" & ".join(r) + " \\\\" if r is not None else "\\midrule")
    hdr = ["Collection", "Encoder", "raw", "corr.", "$\\Delta$\\Rk{10}", "raw", "corr.", "$\\Delta$\\Rk{50}", "$N_{50}$"]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             "\\caption{Two collections under every encoder tested. NWPU gains $+0.25$ to $+0.29$ \\Rk{50} on "
             "all six encoders, generalist and remote-sensing specialist alike, because its raw arrangement is "
             "poor everywhere; ROCOv2 spans $-0.005$ (BiomedCLIP, a biomedical specialist encoder already at raw "
             "\\Rk{50} $0.89$) to $+0.23$ (MedSigLIP), tracking how much each encoder leaves in the "
             "near-miss band.}",
             r"\label{tab:deepdive}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             " & \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\Rk{10}} & \\multicolumn{3}{c}{\\Rk{50}} & \\\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             " & ".join(hdr) + " \\\\", r"\midrule"] + tex_rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "tab_deepdive.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote tab_deepdive.tex ({sum(r is not None for r in lines_rows)} rows)")

    # ── B: training-free baselines ──────────────────────────────────────────
    bls = list(csv.DictReader(open(RESULTS_ROOT / "master_baselines.csv")))
    both = {(r["dataset"], r["mtag"]): r.get("both_gain50") for r in arms}
    rows = []
    for r in sorted(bls, key=lambda r: (r["dataset"], r["mtag"])):
        k = (r["dataset"], r["mtag"])
        rows.append([DSNAME[k[0]], LABEL[k[1]], f3(r["raw_R50"]),
                     f3(both.get(k), True), f3(r["csls_k10_gain50"], True),
                     f3(r["qbnorm_b20_gain50"], True),
                     f3(r["centerboth_gain50"], True),
                     f3(r["gapclose_sym_l0.5_gain50"], True),
                     f3(r["meanshift_gain50"], True)])
    table(out, "tab_baselines",
          "Training-free baselines, $\\Delta$R@50 over raw. CSLS ($k{=}10$), "
          "QB-Norm ($\\beta{=}20$, dynamic inverted softmax, train-caption "
          "querybank), symmetric centering, the exact symmetric $\\lambda{=}\\tfrac12$ "
          "gap-closure construction of \\citet{liang2022mind}, and query "
          "centroid shift. Ours = two maps, one per side.",
          "tab:baselines",
          ["Collection", "Encoder", "raw R@50", "Ours", "CSLS", "QB-Norm",
           "Center", "Liang $\\lambda{=}\\tfrac12$", "Centroid"], rows,
          "llrrrrrrr")

    # ── C: adapters ─────────────────────────────────────────────────────────
    ad = collect("adapter_baselines_*.json")
    rows = []
    for k, r in sorted(ad.items()):
        ca = r["clip_adapter"]
        rows.append([DSNAME[k[0]], LABEL[k[1]], f3(both.get(k), True),
                     f3(ca["text"]["gain@50"], True),
                     f3(ca["image"]["gain@50"], True),
                     f3(ca["both"]["gain@50"], True),
                     f3(r["tip_adapter"]["gain@50"], True),
                     f3(r["tip_adapter_f"]["gain@50"], True)])
    table(out, "tab_adapters",
          "Adapter baselines, $\\Delta$R@50 over raw, same recipe and budget. "
          "CLIP-Adapter (bottleneck MLP, residual ratio tuned on val) at three "
          "placements; Tip-Adapter and Tip-Adapter-F (cache of training "
          "pairs; $\\alpha,\\beta$ tuned on val).",
          "tab:adapters",
          ["Collection", "Encoder", "Ours", "CA-text", "CA-image", "CA-both",
           "Tip", "Tip-F"], rows, "llrrrrrr")

    # ── C2: compact factorisation table for the main text ───────────────────
    fac = collect("shared_factorization_*.json")
    dc = collect("both_decomposition_*.json")
    if fac:
        def frow(k):
            v = fac[k]["variants"]; e = dc.get(k, {}).get("edge", {})
            return dict(raw=v["raw"]["R@50"], gS=v["S"]["gain@50"], gR=v["R"]["gain@50"], gb=v["bias"]["gain@50"],
                        gfull=v["full"]["gain@50"], share=v["S"]["frac_of_full@50"],
                        edge=e.get("both_minus_shared@50"), rot=e.get("rel_rotation@50"))
        allrows = {k: frow(k) for k in fac}
        rows = []
        for k in sorted(allrows, key=lambda k: -allrows[k]["gfull"]):
            if k[1] != "default":
                continue
            r = allrows[k]
            rows.append([DSNAME[k[0]], f3(r["raw"]), f3(r["gfull"], True), f3(r["gS"], True), f3(r["gR"], True),
                         f3(r["gb"], True), f"{100 * r['share']:.0f}\\%",
                         f3(r["edge"], True) if r["edge"] is not None else "--",
                         f3(r["rot"], True) if r["rot"] is not None else "--"])
        med = lambda key: float(np.median([v[key] for v in allrows.values() if v[key] is not None]))
        rows.append([f"\\textit{{median, all {len(allrows)} cells}}", f3(med("raw")), f3(med("gfull"), True),
                     f3(med("gS"), True), f3(med("gR"), True), f3(med("gb"), True), f"{100 * med('share'):.0f}\\%",
                     f3(med("edge"), True), f3(med("rot"), True)])
        table(out, "tab_factor",
              "What each factor of the map contributes (SigLIP2 cells; last row the median over all cells with a "
              "factorisation, seed 0). The tied map $A=RS$ is split by polar decomposition and each factor is "
              "applied alone: the symmetric stretch $S$, the rotation $R$, and the bias $b$. $S$ carries the gain; "
              "$R$ alone changes nothing (Prop.~\\ref{prop:inert}); the bias alone is negligible. The last two "
              "columns give the extra gain of the two-map arm over the tied map and the part of that edge due to "
              "the relative rotation between the sides.",
              "tab:factor",
              ["Collection", "raw \\Rk{50}", "tied map", "$S$ alone", "$R$ alone", "$b$ alone", "$S$ share",
               "two-map edge", "of which rotation"], rows, "lrrrrrrrr")

    # ── D: both-vs-shared decomposition ─────────────────────────────────────
    dc = collect("both_decomposition_*.json")
    fac = collect("shared_factorization_*.json")
    rows = []
    for k, r in sorted(dc.items()):
        e = r["edge"]; rs = r["rel_rotation_stats"]
        sshare = (f"{100 * fac[k]['variants']['S']['frac_of_full@50']:.0f}\\%"
                  if k in fac else "--")
        rows.append([DSNAME[k[0]], LABEL[k[1]], sshare, f3(r["shared"]["R@50"]),
                     f3(r["twostretch"]["R@50"]), f3(r["both"]["R@50"]),
                     f3(e["both_minus_shared@50"], True),
                     f3(e["stretch_asym@50"], True),
                     f3(e["rel_rotation@50"], True),
                     f"{rs['mean_angle_deg']:.1f}",
                     f"{r['equiv_abs_err@50']:.0e}"])
    table(out, "tab_decomp",
          "Factorisation and decomposition (seed 0). Share of the tied map's "
          "R@50 gain recovered by its stretch $S$ alone; R@50 of the tied map, the "
          "two per-side stretches with rotations stripped, and the full "
          "two-map arm; the edge split into stretch asymmetry and relative "
          "rotation; mean rotation angle of $R_{\\text{rel}}$; and the exact "
          "equivalence error of $(S_t, R_{\\text{rel}}S_i)$ vs.\\ the trained pair.",
          "tab:decomp",
          ["Collection", "Encoder", "$S$ share", "tied", "2-stretch", "both", "edge",
           "asym.", "rel.\\ rot.", "angle$^\\circ$", "equiv.\\ err"], rows,
          "llrrrrrrrrr")

    # ── E: four arms at each collection's best encoder ─────────────────────
    sym = collect("symmetric_ablation_*.json")
    rows = []
    for k, r in sorted(sym.items()):
        a = r.get("arms", {})
        if not all(x in a for x in ("text", "image", "both", "shared")):
            continue
        rows.append([DSNAME[k[0]], LABEL[k[1]], f3(r["raw"]["R@50"]),
                     f3(a["text"]["gain@50"], True), f3(a["image"]["gain@50"], True),
                     f3(a["both"]["gain@50"], True), f3(a["shared"]["gain@50"], True),
                     f3(a["text"]["gain@10"], True), f3(a["image"]["gain@10"], True),
                     f3(a["both"]["gain@10"], True), f3(a["shared"]["gain@10"], True)])
    table(out, "tab_arms",
          "All four arms for every cell where the single-side arms were "
          "trained (the 14 original cells plus each collection's best-gain "
          "encoder). Gains over raw at R@50 and R@10.",
          "tab:arms",
          ["Collection", "Encoder", "raw R@50", "text$_{50}$", "image$_{50}$",
           "both$_{50}$", "tied$_{50}$", "text$_{10}$", "image$_{10}$",
           "both$_{10}$", "tied$_{10}$"], rows, "llrrrrrrrrr")

    # ── F: encoder-spread compression ───────────────────────────────────────
    byds = {}
    for r in arms:
        if r.get("both_R50"):
            byds.setdefault(r["dataset"], []).append(
                (float(r["raw_R50"]), float(r["both_R50"])))
    rows = []
    for ds, v in sorted(byds.items()):
        if len(v) < 3:
            continue
        raw_s = max(x for x, _ in v) - min(x for x, _ in v)
        cor_s = max(y for _, y in v) - min(y for _, y in v)
        rows.append([DSNAME[ds], str(len(v)), f3(raw_s), f3(cor_s),
                     f"{100 * (1 - cor_s / raw_s):+.0f}\\%"])
    table(out, "tab_compression",
          "Encoder spread (max$-$min R@50 across encoders) before and after "
          "correction, per collection; positive = compression.",
          "tab:compression",
          ["Collection", "encoders", "raw spread", "corrected spread",
           "compression"], rows, "lrrrr")
    # ── G: band sensitivity of the near-miss statistic ──────────────────────
    if nm:
        pass
        cells = [r for r in arms if r.get("both_R50") and (r["dataset"], r["mtag"]) in nm]
        y = np.array([float(r["both_gain50"]) for r in cells])
        dsv = np.array([r["dataset"] for r in cells])

        def lodo(x):
            pred = np.zeros_like(y)
            for d in set(dsv):
                m = dsv != d; a, b = np.polyfit(x[m], y[m], 1); pred[~m] = a * x[~m] + b
            e = y - pred
            return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.abs(e).mean(), np.abs(e[dsv == "nwpu"]).mean()
        rows = []
        for label, key in [("$d'$ (per-query, train)", None), ("catalog R@10 (train)", "train_R10"),
                           ("$N_{50}$, band $\\times1.5$", "train_N50_x1.5"),
                           ("$N_{50}$, band $\\times2$", "train_N50_x2"),
                           ("$N_{50}$, band $\\times3$ (used)", "train_N50_x3"),
                           ("$N_{50}$, band $\\times4$", "train_N50_x4"),
                           ("$N_{50}$, band $\\times6$", "train_N50_x6"),
                           ("$N_{50}$, band $\\times8$", "train_N50_x8"),
                           ("$N_{50}$, band $\\times3$, val split", "val_N50_x3")]:
            if key is None:
                x = np.array([dp.get((r["dataset"], r["mtag"]), np.nan) for r in cells])
                if np.isnan(x).any():
                    continue
            else:
                x = np.array([float(nm[(r["dataset"], r["mtag"])][key]) for r in cells])
            r2, mae, nw = lodo(x)
            pr = np.corrcoef(x, y)[0, 1]
            from scipy import stats as st
            rows.append([label, f"{pr:+.2f}", f"{st.spearmanr(x, y).statistic:+.2f}",
                         f"{r2:.2f}", f"{mae:.3f}", f"{nw:.3f}"])
        table(out, "tab_band",
              f"Pre-query predictors of the two-map $\\Delta$R@50 across all {len(cells)} "
              "cells: Pearson and Spearman correlation, leave-one-collection-out "
              "$R^2$ and MAE, and the held-out error on the six NWPU cells.",
              "tab:band",
              ["Predictor", "$r$", "$\\rho$", "LOCO $R^2$", "MAE", "NWPU MAE"], rows, "lrrrrr")

    # ── I: singular-value statistics of the trained two-map arm, every cell ──
    rows = []
    for k, r in sorted(symj.items()):
        b = r.get("arms", {}).get("both", {})
        t, i = b.get("map_svd_text"), b.get("map_svd_image")
        if not (t and i):
            continue
        rows.append([DSNAME[k[0]], LABEL[k[1]],
                     str(t["n_sv_gt_1.05"]), str(t["n_sv_lt_0.95"]), f"{t['sv_min']:.2f}", f"{t['sv_max']:.2f}",
                     str(i["n_sv_gt_1.05"]), str(i["n_sv_lt_0.95"]), f"{i['sv_min']:.2f}", f"{i['sv_max']:.2f}"])
    if rows:
        longtable(out, "tab_spectrum",
                  f"Singular values of the trained maps $A=I+W$ (two-map arm, seed 0), all {len(rows)} cells: "
                  "number of directions amplified ($\\sigma>1.05$) and suppressed ($\\sigma<0.95$), "
                  "and the smallest and largest singular value, for the text and image maps.",
                  "tab:spectrum",
                  ["Collection", "Encoder", "text amp.", "text sup.", "$\\sigma_{\\min}$", "$\\sigma_{\\max}$",
                   "image amp.", "image sup.", "$\\sigma_{\\min}$", "$\\sigma_{\\max}$"], rows, "llrrrrrrrr")

    # ── H: rank transport of the trained maps ───────────────────────────────
    tp = collect("rank_transport_*.json")
    if tp:
        bands = [(50, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 1000), (1000, 10 ** 9)]
        rows = []; pooled = {b: [0, 0.0] for b in bands}
        for k, r in sorted(tp.items()):
            bb = {tuple(b["raw_band"]): b for b in r["bands"]}
            cells_ = []
            for b in bands:
                e = bb.get(b)
                cells_.append(f"{e['p_cor_le50']:.2f}" if e else "--")
                if e:
                    pooled[b][0] += e["n"]; pooled[b][1] += e["n"] * e["p_cor_le50"]
            q = r["reach@50_x"]
            rows.append([f"{DSNAME[k[0]]} ({LABEL[k[1]]})"] + cells_ +
                        [f"{q['50']:.1f}", f"{q['90']:.1f}"])
        rows.append(["\\textbf{pooled}"] + [f"{v[1] / v[0]:.2f}" if v[0] else "--" for v in pooled.values()] + ["", ""])
        table(out, "tab_transport",
              "Rank transport of the trained two-map arm (seed 0): probability that a "
              "test query whose true image had the given raw rank ends inside the "
              "corrected top-50, by raw-rank band; and the raw rank of rescued pairs "
              "in multiples of the cutoff (median, 90th percentile).",
              "tab:transport",
              ["Cell", "51--100", "101--150", "151--200", "201--300", "301--500",
               "501--1k", "$>$1k", "reach p50", "p90"], rows, "lrrrrrrrrr")
    # ── I: LoRA fine-tuning comparator ──────────────────────────────────────
    lo = {}; seen = set()
    for root in ROOTS:
        for pth in glob.glob(str(root / "lora" / "*" / "lora_*.json")):
            r = json.load(open(pth, encoding="utf-8"))
            if (r["dataset"], r["n_train"]) in seen:      # same file reachable via two roots
                continue
            seen.add((r["dataset"], r["n_train"]))
            lo.setdefault(r["dataset"], []).append(r)
    if lo:
        rows = []
        for ds in sorted(lo, key=lambda d: -max(r["n_train"] for r in lo[d])):
            for r in sorted(lo[ds], key=lambda r: -r["n_train"]):
                raw = r["raw_cached"]; L = r["lora"]["test"]; M = r["maps"]["maps_batch256"]
                ML = r["maps"]["maps_on_lora"]; cf = r["coco_forgetting"]
                rows.append([DSNAME[ds], f"{r['n_train']:,}",
                             f"{raw['R@10']:.3f}/{raw['R@50']:.3f}",
                             f"{M['R@10']:.3f}/{M['R@50']:.3f}",
                             f"{L['R@10']:.3f}/{L['R@50']:.3f}",
                             f"{ML['R@10']:.3f}/{ML['R@50']:.3f}",
                             f"{M['wall_clock_s']:.1f}", f"{r['train']['wall_clock_s'] / 60:.0f}",
                             f"{r['train']['peak_gpu_mem_GB']:.1f}",
                             f"{cf['lora']['R@10'] - cf['raw']['R@10']:+.3f}/{cf['lora']['R@50'] - cf['raw']['R@50']:+.3f}"])
        table(out, "tab_lora",
              "LoRA fine-tuning against the learned maps on CLIP ViT-L/14 (LAION-2B), "
              "test \\Rk{10}/\\Rk{50}. LoRA: rank 16 on the q/k/v/out projections of both towers "
              "(4.3M trainable parameters), text$\\to$image InfoNCE at batch 64, bf16, epoch chosen on "
              "validation \\Rk{50}. Maps: the paper recipe on the same training pairs (batch 256). "
              "Cost is training wall-clock on one RTX~3080 and peak GPU memory; the last column is the "
              "change in COCO-5k \\Rk{10}/\\Rk{50} of the fine-tuned encoder (the maps leave it unchanged). "
              "One seed.",
              "tab:lora",
              ["Collection", "pairs", "raw", "maps", "LoRA", "maps on LoRA", "maps (s)", "LoRA (min)", "GB",
               "COCO $\\Delta$"], rows, "lrrrrrrrrr")

    print(f"[tables] -> {out}")


if __name__ == "__main__":
    main()
