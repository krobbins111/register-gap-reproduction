"""R@50-only grid heatmap for the main text (half-width companion of the
near-miss panel). Reads the typeset full-grid table so it needs no result
JSONs; the style matches grid_heatmap in make_paper_figs.py.

Usage   python analysis\\make_fig_heatmap_r50.py [--table appendix/tab_grid_full.tex] [--out figs]
Output  <out>/grid_heatmap_r50.pdf/.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
GEN = ["SigLIP2", "SigLIP-so400m", "CLIP ViT-L/14", "MetaCLIP-2"]
SHORT = {"Marqo-FashionSigLIP": "Marqo-FSL", "CLIP ViT-L/14": "CLIP-L"}


def read_table(path: Path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if "&" not in line or "\\\\" not in line:
            continue
        c = [x.strip() for x in line.replace("\\\\", "").split("&")]
        if len(c) != 12 or not c[0] or c[0].startswith("\\") or c[0] == "Collection":
            continue
        rows.append({"ds": c[0], "enc": c[1], "raw50": float(c[4]), "g50": float(c[8].replace("$", "").replace("−", "-"))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="appendix/tab_grid_full.tex")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    rows = read_table(Path(a.table))
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF, "savefig.facecolor": SURF})
    by = {}
    for r in rows:
        by.setdefault(r["ds"], []).append(r)
    order = sorted(by, key=lambda d: -np.mean([r["g50"] for r in by[d]]))
    native = {d: sorted(r["enc"] for r in by[d] if r["enc"] not in GEN) for d in order}
    n_nat = max(len(v) for v in native.values())
    cols = GEN + [f"specialist {k + 1}" if n_nat > 1 else "specialist" for k in range(n_nat)]
    cell = {(r["ds"], r["enc"]): r for r in rows}
    vmax = max(abs(r["g50"]) for r in rows)
    cmap = LinearSegmentedColormap.from_list("gain", [ORANGE, "white", BLUE])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(3.9, 3.7))
    M = np.full((len(order), len(cols)), np.nan); who = {}
    for i, d in enumerate(order):
        for j, e in enumerate(GEN):
            if (d, e) in cell:
                M[i, j] = cell[(d, e)]["g50"]; who[(i, j)] = cell[(d, e)]
        for k, e in enumerate(native[d]):
            M[i, len(GEN) + k] = cell[(d, e)]["g50"]; who[(i, len(GEN) + k)] = cell[(d, e)]
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(order)):
        for j in range(len(cols)):
            if (i, j) not in who:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0eb", edgecolor="white", lw=2)); continue
            r = who[(i, j)]; v = r["g50"]; raw = r["raw50"]
            dark = abs(v) > 0.55 * vmax
            fg = "white" if dark else INK; fg2 = "#e8eef8" if dark else MUTED
            if j >= len(GEN):
                ax.text(j, i - 0.31, SHORT.get(r["enc"], r["enc"]), ha="center", va="center", fontsize=4.9, color=fg2)
            ax.text(j, i + (0.02 if j >= len(GEN) else -0.08), f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color=fg, fontweight="bold")
            ax.text(j, i + 0.33, f"raw {raw:.2f}", ha="center", va="center", fontsize=4.9, color=fg2)
            if raw >= 0.9:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK2, lw=0.9, ls=(0, (2, 1.5))))
    ax.set_xticks(range(len(cols)), [SHORT.get(e, e) for e in GEN] + cols[len(GEN):], rotation=35, ha="right", fontsize=6.8)
    ax.set_yticks(range(len(order)), order, fontsize=7.2)
    ax.set_title("Gain in R@50 from the two maps", fontsize=8.5, color=INK2, pad=5)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(len(GEN) - 0.5, color=INK2, lw=0.8)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "grid_heatmap_r50.pdf", bbox_inches="tight"); fig.savefig(out / "grid_heatmap_r50.png", dpi=300, bbox_inches="tight")
    print("saved", out / "grid_heatmap_r50.pdf")


if __name__ == "__main__":
    main()
