"""All paper figures, regenerated from the result files, in paper style:
no in-figure titles or narrative text (the LaTeX captions carry that), only
axis labels, legends and a few data-point labels. Every n is computed from
the data so the figures stay correct as cells are added.

Reads:  results/master_arms.csv, results/master_baselines.csv
        <roots>/<ds>/{symmetric_ablation,adapter_baselines,shared_factorization,
                       dose_response}_*.json, <roots>/<ds>/shared_map_*.npz
        CACHE_ROOT/dprime_analysis.csv + results/dprime_apriori_extra.csv
Writes: figs/*.pdf + .png   (the files paper/figs/ that main.tex includes)
Prints: the headline numbers the paper text quotes (n, counts, correlations).

Usage:  python analysis\\make_paper_figs.py
Env:    ICLR_EXTRA_ROOTS  extra ':'-separated roots holding <ds>/*.json
"""
from __future__ import annotations

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_ROOT, RESULTS_ROOT  # noqa: E402

ROOTS = [CACHE_ROOT, RESULTS_ROOT] + [Path(p) for p in
                                      os.environ.get("ICLR_EXTRA_ROOTS", "").split(":") if p]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = ("#2a78d6", "#eb6834", "#1baf7a",
                                       "#eda100", "#e87ba4")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})
OUT = RESULTS_ROOT.parent / "figs"; OUT.mkdir(parents=True, exist_ok=True)

NATIVE = {"medsiglip", "biomedclip", "georsclip-vitb32", "remoteclip-vitl14",
          "fashionclip", "marqo-fashionsiglip", "bioclip", "bioclip2"}
LABEL = {"default": "SigLIP2", "siglipv1-so400m-384": "SigLIP-so400m",
         "clip-vitl14-laion2b": "CLIP-L", "metaclip2-ww-huge": "MetaCLIP-2",
         "medsiglip": "MedSigLIP", "biomedclip": "BiomedCLIP",
         "georsclip-vitb32": "GeoRSCLIP", "remoteclip-vitl14": "RemoteCLIP",
         "fashionclip": "FashionCLIP", "marqo-fashionsiglip": "Marqo-FSigLIP",
         "bioclip": "BioCLIP", "bioclip2": "BioCLIP-2"}
DSNAME = {"nwpu": "NWPU", "rocov2": "ROCOv2", "skincap": "SkinCAP",
          "treeoflife": "TreeOfLife", "rsicd": "RSICD", "scimmir": "SciMMIR",
          "fashion200k": "Fashion200k", "semart": "SemArt", "coco": "COCO",
          "goodnews": "GoodNews", "facad": "FACAD"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved", name)


def collect(pattern):
    found = {}
    for root in ROOTS:
        for p in glob.glob(str(root / "*" / pattern)):
            r = json.load(open(p, encoding="utf-8"))
            found[(r["dataset"], r.get("mtag", "default"))] = r
    return found


def cat(ds, mt):
    if ds == "coco": return "control"
    if mt in NATIVE: return "native"
    return "specialist"


GROUPS = {"specialist": (BLUE, "Specialist collection"),
          "control": (ORANGE, "General-domain control (COCO)"),
          "native": (AQUA, "Domain-specialist encoder")}


def load_dprime():
    """A-priori d' per cell. Default (ICLR_DPRIME=perquery): the paper's Eq. 1
    per-query definition on the full train split, from
    results/dprime_apriori_perquery.csv (analysis/apriori_dprime.py
    --definition perquery --all). ICLR_DPRIME=legacy: the original global
    2000-pair-subsample statistic from CACHE_ROOT/dprime_analysis.csv +
    results/dprime_apriori_extra.csv (kept for comparison only)."""
    dp = {}
    if os.environ.get("ICLR_DPRIME", "perquery") == "perquery":
        srcs = (RESULTS_ROOT / "dprime_apriori_perquery.csv",)
    else:
        srcs = (CACHE_ROOT / "dprime_analysis.csv",
                RESULTS_ROOT / "dprime_apriori_extra.csv")
    for p in srcs:
        if p.exists():
            for r in csv.DictReader(open(p)):
                dp[(r["dataset"], r["backbone"])] = float(r["dprime_apriori_train"])
    print(f"[dprime] {os.environ.get('ICLR_DPRIME', 'perquery')} definition, {len(dp)} cells")
    return dp


def load_lora():
    """LoRA comparator runs from <roots>/lora/<ds>/lora_<tag>[_n<k>].json
    (ablations/lora_finetune.py). Per dataset: 'full' = the run with the most
    training pairs, 'dose' = the 1,000-pair run when present."""
    runs = {}; seen = set()
    for root in ROOTS:
        for p in glob.glob(str(root / "lora" / "*" / "lora_*.json")):
            r = json.load(open(p, encoding="utf-8"))
            if (r["dataset"], r["n_train"]) in seen:      # same file reachable via two roots
                continue
            seen.add((r["dataset"], r["n_train"]))
            runs.setdefault(r["dataset"], []).append(r)
    out = {}
    for ds, rs in runs.items():
        full = max(rs, key=lambda r: r["n_train"])
        dose = [r for r in rs if r["n_train"] == 1000]
        out[ds] = {"full": full, "dose": dose[0] if dose else None}
    if out:
        print(f"[lora] {len(out)} collections with LoRA runs")
    return out


def load_nearmiss():
    """Near-miss mass per cell from results/nearmiss_apriori.csv
    (analysis/apriori_nearmiss.py --all). Returns {(ds, mt): row-dict}."""
    p = RESULTS_ROOT / "nearmiss_apriori.csv"
    if not p.exists():
        print("[nearmiss] results/nearmiss_apriori.csv missing; run analysis/apriori_nearmiss.py --all")
        return {}
    return {(r["dataset"], r["backbone"]): r for r in csv.DictReader(open(p))}


NM_MAIN = "train_N50_x3"      # N_50 = P(50 < rank <= 150) on a train-catalog sample of test-gallery size


def main() -> None:
    # ── data ────────────────────────────────────────────────────────────────
    dp = load_dprime()
    nm = load_nearmiss()
    rows = []
    for r in csv.DictReader(open(RESULTS_ROOT / "master_arms.csv")):
        if not r.get("both_R50"):
            continue
        rows.append({"ds": r["dataset"], "mt": r["mtag"],
                     "raw50": float(r["raw_R50"]), "raw10": float(r["raw_R10"]),
                     "b50": float(r["both_R50"]), "g50": float(r["both_gain50"]),
                     "g10": float(r["both_gain10"]) if r.get("both_gain10") else
                            float(r["both_R10"]) - float(r["raw_R10"]),
                     "s50": float(r["shared_gain50"]) if r.get("shared_gain50") else None,
                     "dp": dp.get((r["dataset"], r["mtag"])),
                     "nm": float(nm[(r["dataset"], r["mtag"])][NM_MAIN])
                           if (r["dataset"], r["mtag"]) in nm else None,
                     "nm_row": nm.get((r["dataset"], r["mtag"]))})
    n = len(rows)
    g = np.array([r["g50"] for r in rows])
    print(f"[grid] n={n}  nonneg={int((g >= -1e-9).sum())}  mean={g.mean():+.3f}  "
          f"median={np.median(g):+.3f}  max={g.max():+.3f}  min={g.min():+.3f}")
    wins = sum(1 for r in rows if r["s50"] is not None and r["g50"] > r["s50"])
    ns = sum(1 for r in rows if r["s50"] is not None)
    print(f"[grid] both>tied {wins}/{ns}")

    # ── Fig: the whole grid as a heatmap (ΔR@10 | ΔR@50), raw recall annotated ─
    # Columns: the four generalist encoders, then two "domain-specialist" slots
    # (each collection has at most two specialist encoders; the name goes in the cell).
    GEN = ["default", "siglipv1-so400m-384", "clip-vitl14-laion2b", "metaclip2-ww-huge"]
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["ds"], []).append(r)
    ds_order = sorted(by_ds, key=lambda d: -np.mean([r["g50"] for r in by_ds[d]]))
    cell = {(r["ds"], r["mt"]): r for r in rows}
    native_of = {d: sorted([r["mt"] for r in by_ds[d] if r["mt"] in NATIVE]) for d in ds_order}
    n_nat = max(len(v) for v in native_of.values())
    cols = GEN + [f"native{k}" for k in range(n_nat)]
    from matplotlib.colors import TwoSlopeNorm
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gain", [ORANGE, "white", BLUE])
    vmax = max(abs(r["g10"]) for r in rows + []) ; vmax = max(vmax, max(abs(r["g50"]) for r in rows))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, key, rawkey, lab in ((axes[0], "g10", "raw10", "ΔR@10"), (axes[1], "g50", "raw50", "ΔR@50")):
        M = np.full((len(ds_order), len(cols)), np.nan); who = {}
        for i, d in enumerate(ds_order):
            for j, e in enumerate(GEN):
                if (d, e) in cell:
                    M[i, j] = cell[(d, e)][key]; who[(i, j)] = cell[(d, e)]
            for k, e in enumerate(native_of[d]):
                M[i, len(GEN) + k] = cell[(d, e)][key]; who[(i, len(GEN) + k)] = cell[(d, e)]
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(ds_order)):
            for j in range(len(cols)):
                if (i, j) not in who:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0eb", edgecolor="white", lw=2))
                    continue
                r = who[(i, j)]; v = r[key]; raw = r[rawkey]
                dark = abs(v) > 0.55 * vmax
                fg = "white" if dark else INK; fg2 = "#e8eef8" if dark else MUTED
                if j >= len(GEN):
                    ax.text(j, i - 0.31, LABEL[r["mt"]].replace("Marqo-FSigLIP", "Marqo-FSL"), ha="center",
                            va="center", fontsize=5.2, color=fg2)
                ax.text(j, i + (0.02 if j >= len(GEN) else -0.08), f"{v:+.2f}", ha="center", va="center", fontsize=7.4,
                        color=fg, fontweight="bold")
                ax.text(j, i + 0.33, f"raw {raw:.2f}", ha="center", va="center", fontsize=5.2, color=fg2)
                if raw >= 0.9:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK2, lw=0.9, ls=(0, (2, 1.5))))
        ax.set_xticks(range(len(cols)), [LABEL[e] for e in GEN] + ["specialist" if n_nat == 1 else f"specialist {k + 1}" for k in range(n_nat)],
                      rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(ds_order)), [DSNAME[d] for d in ds_order], fontsize=7.5)
        ax.set_title(lab, fontsize=10, color=INK2, pad=5)
        ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(len(GEN) - 0.5, color=INK2, lw=0.8)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    cb.set_label("gain over raw", fontsize=7.5, color=INK2)
    save(fig, "grid_heatmap")
    print("[heatmap] rows:", ds_order, "native slots:", n_nat)

    # ── Fig: same grid, cell = share of the raw ERROR the maps recover ────────
    # recovered = gain / (1 - raw): 100% means every miss at that cutoff was
    # rescued. Makes fashion/web-register cells (raw 0.95+) comparable with
    # collapsed ones (raw 0.2-0.7). Rows sorted by mean recovered share at R@50.
    def recov(r, key, rawkey):
        err = 1.0 - r[rawkey]
        return r[key] / err if err > 1e-6 else float("nan")
    ds_order_r = sorted(by_ds, key=lambda d: -np.nanmean([recov(r, "g50", "raw50") for r in by_ds[d]]))
    cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list(
        "recov", [(0.0, ORANGE), (0.25, "white"), (1.0, BLUE)])
    norm_r = matplotlib.colors.Normalize(vmin=-0.25, vmax=0.75)   # white at 0, full blue at 75%+
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, key, rawkey, lab in ((axes[0], "g10", "raw10", "share of R@10 misses recovered"),
                                 (axes[1], "g50", "raw50", "share of R@50 misses recovered")):
        M = np.full((len(ds_order_r), len(cols)), np.nan); who = {}
        for i, d in enumerate(ds_order_r):
            for j, e in enumerate(GEN):
                if (d, e) in cell:
                    M[i, j] = recov(cell[(d, e)], key, rawkey); who[(i, j)] = cell[(d, e)]
            for k, e in enumerate(native_of[d]):
                M[i, len(GEN) + k] = recov(cell[(d, e)], key, rawkey); who[(i, len(GEN) + k)] = cell[(d, e)]
        ax.imshow(np.clip(M, -0.25, 0.75), cmap=cmap_r, norm=norm_r, aspect="auto")
        for i in range(len(ds_order_r)):
            for j in range(len(cols)):
                if (i, j) not in who:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0eb", edgecolor="white", lw=2))
                    continue
                r = who[(i, j)]; v = M[i, j]; err = 1.0 - r[rawkey]
                dark = v > 0.5
                fg = "white" if dark else INK; fg2 = "#e8eef8" if dark else MUTED
                if j >= len(GEN):
                    ax.text(j, i - 0.31, LABEL[r["mt"]].replace("Marqo-FSigLIP", "Marqo-FSL"), ha="center",
                            va="center", fontsize=5.2, color=fg2)
                ax.text(j, i + (0.02 if j >= len(GEN) else -0.08), f"{100 * v:+.0f}%", ha="center", va="center",
                        fontsize=7.4, color=fg, fontweight="bold")
                ax.text(j, i + 0.33, f"of {err:.2f}", ha="center", va="center", fontsize=5.2, color=fg2)
        ax.set_xticks(range(len(cols)), [LABEL[e] for e in GEN] + ["specialist" if n_nat == 1 else f"specialist {k + 1}" for k in range(n_nat)],
                      rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(ds_order_r)), [DSNAME[d] for d in ds_order_r], fontsize=7.5)
        ax.set_title(lab, fontsize=9.5, color=INK2, pad=5)
        ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(len(GEN) - 0.5, color=INK2, lw=0.8)
    sm = matplotlib.cm.ScalarMappable(norm=norm_r, cmap=cmap_r)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=[-0.25, 0, 0.25, 0.5, 0.75])
    cb.ax.set_yticklabels(["-25%", "0", "25%", "50%", "75%+"]); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    cb.set_label("misses recovered  (gain / (1 - raw recall))", fontsize=7.5, color=INK2)
    save(fig, "grid_heatmap_recovered")
    rec50 = [recov(r, "g50", "raw50") for r in rows]; rec10 = [recov(r, "g10", "raw10") for r in rows]
    print(f"[heatmap-recovered] R@50 share: median {np.nanmedian(rec50):.2f} mean {np.nanmean(rec50):.2f}; "
          f"R@10 share: median {np.nanmedian(rec10):.2f} mean {np.nanmean(rec10):.2f}")
    for d in ds_order_r:
        print(f"   {DSNAME[d]:14s} R@50 recovered mean {np.nanmean([recov(r,'g50','raw50') for r in by_ds[d]]):.2f}"
              f"  R@10 {np.nanmean([recov(r,'g10','raw10') for r in by_ds[d]]):.2f}")

    # ── Fig: corrected vs raw R@50 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, zorder=1)
    for k, (c, lab) in GROUPS.items():
        xs = [r["raw50"] for r in rows if cat(r["ds"], r["mt"]) == k]
        ys = [r["b50"] for r in rows if cat(r["ds"], r["mt"]) == k]
        ax.scatter(xs, ys, s=38, color=c, label=f"{lab} (n={len(xs)})",
                   edgecolors="white", linewidths=1.0, zorder=3)
    best = max(rows, key=lambda r: r["g50"])
    worst = min(rows, key=lambda r: r["g50"])
    for r, txt, xy, ha in ((best, f"{DSNAME[best['ds']]} / {LABEL[best['mt']]} "
                                  f"{best['g50']:+.2f}", (0.30, 0.44), "left"),
                           (worst, f"{DSNAME[worst['ds']]} / {LABEL[worst['mt']]} "
                                   f"{worst['g50']:+.3f}", (1.0, 0.62), "right")):
        ax.annotate(txt, (r["raw50"], r["b50"]), textcoords="data", xytext=xy,
                    fontsize=8.5, color=INK2, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8,
                                    shrinkA=0, shrinkB=4))
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("Raw R@50")
    ax.set_ylabel("R@50 with two learned maps")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    save(fig, "grid_never_hurts")

    # ── Fig: near-miss law (main diagnostic) ───────────────────────────────
    from scipy import stats as st
    nml = [r for r in rows if r["nm"] is not None]
    if nml:
        x = np.array([r["nm"] for r in nml]); y = np.array([r["g50"] for r in nml])
        dsl = np.array([r["ds"] for r in nml])
        preds = np.zeros_like(y)
        for d in set(dsl):
            tr = dsl != d; a, b = np.polyfit(x[tr], y[tr], 1); preds[~tr] = a * x[~tr] + b
        r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2_id = 1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        a, b = np.polyfit(x, y, 1)
        print(f"[nearmiss] n={len(x)} pearson={np.corrcoef(x, y)[0, 1]:+.3f} "
              f"spearman={st.spearmanr(x, y).statistic:+.3f} LODO R2={r2:.3f} "
              f"MAE={np.abs(y - preds).mean():.4f} | identity: R2={r2_id:.3f} "
              f"MAE={np.abs(y - x).mean():.4f} within0.05={np.mean(np.abs(y - x) <= 0.05):.2f} "
              f"| pooled fit y={a:.3f}x{b:+.4f} | NWPU LODO MAE="
              f"{np.abs(y - preds)[dsl == 'nwpu'].mean():.3f}")
        # band sensitivity + alternatives
        for key in ("train_N50_x1.5", "train_N50_x2", "train_N50_x3", "train_N50_x4",
                    "train_N50_x6", "train_N50_x8", "val_N50_x3", "train_R10"):
            xs_ = np.array([float(r["nm_row"][key]) for r in nml])
            pr_ = np.zeros_like(y)
            for d in set(dsl):
                tr = dsl != d; a_, b_ = np.polyfit(xs_[tr], y[tr], 1); pr_[~tr] = a_ * xs_[~tr] + b_
            print(f"[nearmiss:{key}] pearson={np.corrcoef(xs_, y)[0, 1]:+.3f} "
                  f"LODO R2={1 - ((y - pr_) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f} "
                  f"MAE={np.abs(y - pr_).mean():.3f} NWPU={np.abs(y - pr_)[dsl == 'nwpu'].mean():.3f}")
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        lim = max(x.max(), y.max()) * 1.08
        ax.plot([0, lim], [0, lim], color=INK2, lw=1.1, ls="--", zorder=2)
        for k, (c, lab) in GROUPS.items():
            xs = [r["nm"] for r in nml if cat(r["ds"], r["mt"]) == k]
            ys = [r["g50"] for r in nml if cat(r["ds"], r["mt"]) == k]
            ax.scatter(xs, ys, s=38, color=c, label=f"{lab} (n={len(xs)})",
                       edgecolors="white", linewidths=1.0, zorder=3)
        nw = [r for r in nml if r["ds"] == "nwpu"]
        ax.scatter([r["nm"] for r in nw], [r["g50"] for r in nw], s=120, facecolors="none",
                   edgecolors=INK, linewidths=0.9, zorder=4, label="NWPU cells")
        ax.set_xlim(0, lim); ax.set_ylim(min(-0.02, y.min() - 0.01), lim)
        ax.set_xlabel("Near-miss mass N₅₀ (catalog, before training)")
        ax.set_ylabel("Gain in R@50 from the learned maps")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        save(fig, "nearmiss_law")

    # ── Fig: d' law (appendix: predecessor diagnostic) ─────────────────────
    law = [r for r in rows if r["dp"] is not None]
    x = np.array([r["dp"] for r in law]); y = np.array([r["g50"] for r in law])
    pear = np.corrcoef(x, y)[0, 1]; rho = st.spearmanr(x, y).statistic
    # leave-one-dataset-out
    preds = np.zeros_like(y)
    for ds in set(r["ds"] for r in law):
        tr = np.array([r["ds"] != ds for r in law])
        a, b = np.polyfit(x[tr], y[tr], 1)
        preds[~tr] = a * x[~tr] + b
    r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = np.abs(y - preds).mean()
    raw = np.array([r["raw50"] for r in law])
    rx = x - np.polyval(np.polyfit(raw, x, 1), raw)
    ry = y - np.polyval(np.polyfit(raw, y, 1), raw)
    partial = np.corrcoef(rx, ry)[0, 1]
    print(f"[law] n={len(law)} pearson={pear:+.3f} spearman={rho:+.3f} "
          f"LODO R2={r2:.3f} MAE={mae:.4f} partial|raw={partial:+.3f}")
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
    ax.plot(xx, a * xx + b, color=INK2, lw=1.1, ls="--", zorder=2)
    for k, (c, lab) in GROUPS.items():
        xs = [r["dp"] for r in law if cat(r["ds"], r["mt"]) == k]
        ys = [r["g50"] for r in law if cat(r["ds"], r["mt"]) == k]
        ax.scatter(xs, ys, s=38, color=c, label=f"{lab} (n={len(xs)})",
                   edgecolors="white", linewidths=1.0, zorder=3)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("A-priori d′ (training catalog)")
    ax.set_ylabel("Gain in R@50")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    save(fig, "dprime_law")

    # ── Fig: encoder spread ─────────────────────────────────────────────────
    byds = {}
    for r in rows:
        byds.setdefault(r["ds"], []).append(r)
    spread = []
    for ds, rs in byds.items():
        if len(rs) < 3: continue
        raws = [r["raw50"] for r in rs]; cors = [r["b50"] for r in rs]
        spread.append((ds, max(raws) - min(raws), max(cors) - min(cors)))
    spread.sort(key=lambda t: -t[1])
    for ds, rs_, cs_ in spread:
        print(f"[spread] {ds:12s} raw {rs_:.3f} -> cor {cs_:.3f} "
              f"({100*(1-cs_/rs_):+.0f}%)")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ys = np.arange(len(spread))[::-1]
    for yy, (ds, rs_, cs_) in zip(ys, spread):
        ax.plot([rs_, cs_], [yy, yy], color=GRID, lw=2.2, zorder=1)
        ax.scatter([rs_], [yy], s=44, color=MUTED, zorder=3, edgecolors="white", linewidths=1.0)
        ax.scatter([cs_], [yy], s=44, color=(ORANGE if cs_ > rs_ else BLUE),
                   zorder=3, edgecolors="white", linewidths=1.0)
    ax.set_yticks(ys); ax.set_yticklabels([DSNAME[s[0]] for s in spread], color=INK2)
    ax.set_xlabel("Encoder spread on the collection (max − min R@50)")
    ax.scatter([], [], s=44, color=MUTED, label="raw", edgecolors="white")
    ax.scatter([], [], s=44, color=BLUE, label="corrected (compresses)", edgecolors="white")
    ax.scatter([], [], s=44, color=ORANGE, label="corrected (expands)", edgecolors="white")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    save(fig, "spread_compression")

    # ── Fig: training-free baselines / adapters ─────────────────────────────
    bl = {(r["dataset"], r["mtag"]): r
          for r in csv.DictReader(open(RESULTS_ROOT / "master_baselines.csv"))}
    both = {(r["ds"], r["mt"]): r["g50"] for r in rows}
    sel = sorted([k for k in bl if k[1] == "default" and k[0] not in ("coco", "facad")
                  and k in both], key=lambda k: -both[k])
    def blv(k, col):
        v = bl[k].get(col, ""); return float(v) if v not in ("", None) else 0.0
    series = [("Learned maps (one per side)", [both[k] for k in sel], BLUE),
              ("CSLS", [blv(k, "csls_k10_gain50") for k in sel], ORANGE),
              ("QB-Norm", [blv(k, "qbnorm_b20_gain50") for k in sel], AQUA),
              ("Centroid shift", [blv(k, "meanshift_gain50") for k in sel], YELLOW)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xpos = np.arange(len(sel)); w = 0.2
    for i, (lab, vals, c) in enumerate(series):
        ax.bar(xpos + (i - 1.5) * w, vals, width=w - 0.02, color=c, label=lab, zorder=3)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(xpos, [DSNAME[k[0]] for k in sel], fontsize=7.5)
    ax.set_xlim(-0.6, len(sel) - 0.4)
    ax.set_ylabel("ΔR@50 over raw")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, ncols=2)
    ax.grid(axis="x", visible=False)
    save(fig, "baselines")

    ad = collect("adapter_baselines_*.json")
    sel2 = sorted([k for k in ad if k[1] == "default" and k[0] not in ("coco", "facad")
                   and k in both], key=lambda k: -both[k])
    series2 = [("Learned maps (one per side)", [both[k] for k in sel2], BLUE),
               ("CLIP-Adapter (MLP, both sides)",
                [ad[k]["clip_adapter"]["both"]["gain@50"] for k in sel2], AQUA),
               ("Tip-Adapter (cache)", [ad[k]["tip_adapter"]["gain@50"] for k in sel2], ORANGE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xpos = np.arange(len(sel2)); w = 0.26
    for i, (lab, vals, c) in enumerate(series2):
        ax.bar(xpos + (i - 1) * w, vals, width=w - 0.02, color=c, label=lab, zorder=3)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(xpos, [DSNAME[k[0]] for k in sel2], fontsize=8)
    ax.set_ylabel("ΔR@50 over raw")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "adapters")

    # ── Fig: LoRA fine-tuning comparator (CLIP-L/14 LAION-2B) ───────────────
    lo = load_lora()
    if lo:
        ds_order = sorted(lo, key=lambda d: -(lo[d]["full"]["lora"]["test"]["R@50"]
                                               - lo[d]["full"]["raw_cached"]["R@50"]))
        def g(r, key, k="R@50"):
            base = r["raw_cached"][k]
            if key == "lora": return r["lora"]["test"][k] - base
            if key == "maps": return r["maps"]["maps_batch256"][k] - base
            if key == "maps_on_lora": return r["maps"]["maps_on_lora"][k] - base
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), width_ratios=[1.5, 1])
        xpos = np.arange(len(ds_order)); w = 0.26
        series3 = [("Learned maps (frozen encoder)", "maps", BLUE),
                   ("LoRA fine-tuning (both towers)", "lora", ORANGE),
                   ("Maps on the LoRA encoder", "maps_on_lora", AQUA)]
        top = 0.0
        for i, (lab, key, c) in enumerate(series3):
            vals = [g(lo[d]["full"], key) for d in ds_order]; top = max(top, max(vals))
            ax.bar(xpos + (i - 1) * w, vals, width=w - 0.02, color=c, label=lab, zorder=3)
        # 1,000-pair dose as open markers on the maps / LoRA bars
        for i, key in ((0, "maps"), (1, "lora")):
            vals = [g(lo[d]["dose"], key) if lo[d].get("dose") else np.nan for d in ds_order]
            ax.scatter(xpos + (i - 1) * w, vals, s=18, facecolors="white",
                       edgecolors=INK, linewidths=0.8, zorder=4,
                       label="same, trained on 1,000 pairs" if i == 0 else None)
        ax.axhline(0, color=AXIS, lw=1.0)
        ax.set_xticks(xpos, [DSNAME[d] for d in ds_order], fontsize=7.5)
        ax.set_xlim(-0.6, len(ds_order) - 0.4)
        ax.set_ylim(top=top * 1.5)
        ax.set_yticks([t for t in ax.get_yticks() if t <= top * 1.12])
        ax.set_ylabel("ΔR@50 over raw CLIP-L")
        ax.legend(loc="upper center", frameon=False, fontsize=7, ncols=2,
                  columnspacing=0.9, handlelength=1.4, handletextpad=0.5)
        ax.grid(axis="x", visible=False)
        # cost vs gain: seconds of training on one GPU, log axis
        for d in ds_order:
            r = lo[d]["full"]
            tm = r["maps"]["maps_batch256"]["wall_clock_s"]; tl = r["train"]["wall_clock_s"]
            gm, gl = g(r, "maps"), g(r, "lora")
            ax2.plot([tm, tl], [gm, gl], color=GRID, lw=1.0, zorder=2)
            ax2.scatter([tm], [gm], s=30, color=BLUE, edgecolors="white", zorder=3)
            ax2.scatter([tl], [gl], s=30, color=ORANGE, edgecolors="white", zorder=3)
            ax2.annotate(DSNAME[d], (tl, gl), xytext=(-6, 0), textcoords="offset points",
                         fontsize=6.8, color=INK2, ha="right", va="center")
        ax2.set_xscale("log")
        ax2.set_xlabel("training time, one RTX 3080 (s)")
        ax2.set_ylabel("ΔR@50 over raw CLIP-L")
        ax2.set_xlim(0.1, 8e4); ax2.set_xticks([1, 10, 100, 1e3, 1e4], ["1", "10", "100", "1k", "10k"])
        ax2.grid(axis="y", visible=False)
        ax2.scatter([], [], s=30, color=BLUE, label="maps")
        ax2.scatter([], [], s=30, color=ORANGE, label="LoRA")
        ax2.legend(loc="upper left", frameon=False, fontsize=7.5)
        fig.tight_layout(w_pad=1.5)
        save(fig, "lora")
        # headline numbers for the text
        fr = [g(lo[d]["full"], "maps") / g(lo[d]["full"], "lora") for d in ds_order]
        fr10 = [g(lo[d]["full"], "maps", "R@10") / g(lo[d]["full"], "lora", "R@10") for d in ds_order]
        ratio = [lo[d]["full"]["train"]["wall_clock_s"] / lo[d]["full"]["maps"]["maps_batch256"]["wall_clock_s"]
                 for d in ds_order]
        forget = [lo[d]["full"]["coco_forgetting"]["lora"]["R@50"] - lo[d]["full"]["coco_forgetting"]["raw"]["R@50"]
                  for d in ds_order]
        stack = [g(lo[d]["full"], "maps_on_lora") - g(lo[d]["full"], "lora") for d in ds_order]
        print("[lora] collections:", ds_order)
        print("[lora] maps/LoRA gain fraction R@50:", [f"{x:.2f}" for x in fr],
              " R@10:", [f"{x:.2f}" for x in fr10])
        print(f"[lora] wall-clock ratio LoRA/maps: min {min(ratio):.0f}x median {np.median(ratio):.0f}x max {max(ratio):.0f}x")
        print("[lora] COCO R@50 change after LoRA:", [f"{x:+.3f}" for x in forget])
        print("[lora] maps-on-LoRA extra R@50:", [f"{x:+.3f}" for x in stack])
        for d in ds_order:
            if lo[d].get("dose"):
                r = lo[d]["dose"]
                print(f"[lora] 1k pairs {d}: maps {g(r, 'maps'):+.3f} LoRA {g(r, 'lora'):+.3f} "
                      f"(LoRA best epoch {r['train']['best_epoch']}/{r['lora']['epochs']})")

    # ── Fig: four arms at each collection's best encoder ────────────────────
    sym = collect("symmetric_ablation_*.json")
    full = {k: r for k, r in sym.items()
            if all(a in r.get("arms", {}) for a in ("text", "image", "both", "shared"))}
    bestcell = {}
    for (ds, mt), r in sym.items():
        if ds == "coco" and mt != "default": continue
        gg = r["arms"].get("both", {}).get("gain@50")
        if gg is None: continue
        if ds not in bestcell or gg > bestcell[ds][1]:
            bestcell[ds] = ((ds, mt), gg)
    cells = [bestcell[d][0] for d in sorted(bestcell, key=lambda d: -bestcell[d][1])]
    missing = [k for k in cells if k not in full]
    if missing:
        print(f"[arms] WARNING best-encoder cells lacking single-side arms: {missing}")
    cells = [k for k in cells if k in full]
    ARMS = [("text", ORANGE, "Text side only"), ("image", YELLOW, "Image side only"),
            ("both", BLUE, "One map per side"), ("shared", AQUA, "One tied map")]
    fig, ax = plt.subplots(figsize=(0.8 * len(cells) + 1.5, 3.6))
    xpos = np.arange(len(cells)); w = 0.2
    for i, (arm, c, lab) in enumerate(ARMS):
        ax.bar(xpos + (i - 1.5) * w, [full[k]["arms"][arm]["gain@50"] for k in cells],
               width=w - 0.02, color=c, label=lab, zorder=3)
    ax.axhline(0, color=AXIS, lw=1.0)
    ax.set_xticks(xpos, [f"{DSNAME[k[0]]} ({LABEL[k[1]]})" for k in cells], fontsize=7.5,
                  rotation=28, ha="right", rotation_mode="anchor")
    ax.set_ylabel("ΔR@50 over raw")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, ncols=2)
    ax.grid(axis="x", visible=False)
    save(fig, "sym_ablation_full")

    # ── Fig: which side wins ────────────────────────────────────────────────
    side = []
    for k, r in full.items():
        gm = r.get("train_geometry", {})
        if gm.get("img_near_dup@0.9") is None: continue
        side.append((k, gm["img_near_dup@0.9"] - gm["txt_near_dup@0.9"],
                     r["arms"]["image"]["gain@10"] - r["arms"]["text"]["gain@10"]))
    sx = np.array([s[1] for s in side]); sy = np.array([s[2] for s in side])
    srho, sp = st.spearmanr(sx, sy)
    print(f"[side] n={len(side)} image wins {int((sy > 0).sum())}/{len(side)} "
          f"spearman={srho:+.3f} p={sp:.3f}")
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.axhline(0, color=AXIS, lw=0.9); ax.axvline(0, color=AXIS, lw=0.9)
    a, b = np.polyfit(sx, sy, 1)
    xx = np.linspace(sx.min(), sx.max(), 20)
    ax.plot(xx, a * xx + b, color=AXIS, lw=1.1, zorder=2)
    ax.scatter(sx, sy, s=38, color=BLUE, edgecolors="white", linewidths=1.0, zorder=3)
    for (k, xi, yi) in side:
        if abs(yi) > 0.04 or xi > 0.6 or xi < -0.25:
            ax.annotate(f"{DSNAME[k[0]]} ({LABEL[k[1]]})", (xi, yi),
                        textcoords="offset points",
                        xytext=(6, 5) if yi >= 0 else (-2, 9),
                        fontsize=7.5, color=INK2)
    ax.set_xlabel("Gallery collapse − text collapse (near-dup@0.9)")
    ax.set_ylabel("Image-side gain − text-side gain (ΔR@10)")
    save(fig, "which_side")

    # ── Fig: factorisation + spectrum ───────────────────────────────────────
    fac = collect("shared_factorization_*.json")
    if fac:
        cells_f = sorted(((f"{DSNAME[k[0]]} ({LABEL[k[1]]})",
                           100 * r["variants"]["S"]["frac_of_full@50"])
                          for k, r in fac.items()), key=lambda c: c[1])
        print(f"[fac] n={len(cells_f)} median S share={np.median([c[1] for c in cells_f]):.0f}%")
    for root in ROOTS:
        zp = root / "nwpu" / "shared_map_default.npz"
        if zp.exists():
            z = np.load(zp)
            A = np.eye(z["W"].shape[0]) + z["W"]
            sv = np.linalg.svd(A, compute_uv=False)
            print(f"[spectrum] nwpu/default: {int((sv > 1.05).sum())} amplified, "
                  f"{int((sv < 0.95).sum())} suppressed, {int((sv < 0.1).sum())} <0.1")
            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            idx = np.arange(1, len(sv) + 1)
            ax.axhline(1.0, color=AXIS, lw=1.0)
            ax.fill_between(idx, 1.0, np.maximum(sv, 1.0), color=BLUE, alpha=0.35, lw=0)
            ax.fill_between(idx, np.minimum(sv, 1.0), 1.0, color=ORANGE, alpha=0.35, lw=0)
            ax.plot(idx, sv, color=INK2, lw=1.3)
            ax.set_xlabel("Direction (singular-value rank)")
            ax.set_ylabel("Scale applied by the map")
            ax.set_xlim(1, len(sv)); ax.set_ylim(0, sv.max() * 1.06)
            save(fig, "spectrum")
            break

    # spectrum statistics over every cell (two-map arm, from the ablation reports)
    symall = collect("symmetric_ablation_*.json")
    amp, sup, small, rot = [], [], [], []
    for k, r in symall.items():
        b = r.get("arms", {}).get("both", {})
        for side in ("map_svd_text", "map_svd_image"):
            m = b.get(side)
            if m:
                amp.append(m["n_sv_gt_1.05"]); sup.append(m["n_sv_lt_0.95"])
                small.append(m["sv_min"]); rot.append(m["mean_rot_deg"])
    if amp:
        print(f"[spectrum:all] {len(amp)} maps: amplified median {np.median(amp):.0f} "
              f"(IQR {np.percentile(amp, 25):.0f}-{np.percentile(amp, 75):.0f}), suppressed median "
              f"{np.median(sup):.0f} (IQR {np.percentile(sup, 25):.0f}-{np.percentile(sup, 75):.0f}), "
              f"sv_min median {np.median(small):.2f}, maps with sv_min<0.1: {int((np.array(small) < 0.1).sum())}")

    # ── Fig: dose-response (main: gain vs training pairs; appendix: d' readout)
    dose_both = collect("dose_response_both_*.json")
    dose_tied = collect("dose_response_[!b]*.json")      # tied-map sweeps only
    dose = dose_both if dose_both else dose_tied
    if dose:
        print(f"[dose] using {'two-map' if dose_both else 'tied-map'} sweeps ({len(dose)} collections)")
    if dose:
        cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
        mks = ["o", "s", "^", "D", "v"]
        X_RAW = 40
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
        figA, (b1, b2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
        for i, (k, r) in enumerate(sorted(dose.items())):
            pts = r["points"]
            nn = np.array([p["n_train"] for p in pts], dtype=float)
            dpv = np.array([p["dprime"] for p in pts])
            r50 = np.array([p["R@50"] for p in pts]); r10 = np.array([p["R@10"] for p in pts])
            xs = np.where(nn == 0, X_RAW, nn)
            c, mk = cols[i % 5], mks[i % 5]
            lab = DSNAME[k[0]]
            ax1.plot(xs, r50 - r50[0], color=c, lw=1.8, marker=mk, ms=4.5, label=lab)
            ax2.plot(xs, r10 - r10[0], color=c, lw=1.8, marker=mk, ms=4.5, label=lab)
            b1.plot(xs, dpv, color=c, lw=1.8, marker=mk, ms=4.5, label=lab)
            b2.plot(dpv - dpv[0], r50 - r50[0], color=c, lw=1.8, marker=mk, ms=4.5,
                    label=f"{lab} (r={r['r_dprime_vs_R50_across_doses']:.2f})")
        for ax, yl in ((ax1, "ΔR@50 over raw"), (ax2, "ΔR@10 over raw"), (b1, "Test d′ after the map")):
            ax.set_xscale("log")
            ax.set_xticks([X_RAW, 100, 1000, 10000, 50000], ["raw", "100", "1k", "10k", "50k"])
            ax.tick_params(axis="x", which="minor", bottom=False)
            ax.set_xlabel("Training pairs"); ax.set_ylabel(yl)
        ax1.axhline(0, color=AXIS, lw=0.9); ax2.axhline(0, color=AXIS, lw=0.9)
        ax1.legend(frameon=False, fontsize=8)
        b1.legend(frameon=False, fontsize=8)
        b2.set_xlabel("Δd′ over raw"); b2.set_ylabel("ΔR@50 over raw")
        b2.legend(frameon=False, fontsize=8)
        fig.tight_layout(); figA.tight_layout()
        save(fig, "dose_response")
        save(figA, "dose_dprime")
    print(f"[figs] -> {OUT}")


if __name__ == "__main__":
    main()
