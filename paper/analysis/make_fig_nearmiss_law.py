"""Near-miss law panel for the main text: N50 (training catalog) against the
realised two-map gain, 58 cells, no NWPU circles. Reads the typeset full-grid
table so it needs no result JSONs; sized to sit beside grid_heatmap_r50 at
the same height (0.425 vs 0.555 linewidth).

Usage   python analysis\\make_fig_nearmiss_law.py [--table appendix/tab_grid_full.tex] [--out figs]
Output  <out>/nearmiss_law_shapes.pdf/.png  (category by marker shape, colour redundant)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
GEN = {"SigLIP2", "SigLIP-so400m", "CLIP ViT-L/14", "MetaCLIP-2"}


def read_table(path: Path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if "&" not in line or "\\\\" not in line:
            continue
        c = [x.strip() for x in line.replace("\\\\", "").split("&")]
        if len(c) != 12 or not c[0] or c[0].startswith("\\") or c[0] == "Collection":
            continue
        rows.append({"ds": c[0], "enc": c[1], "g50": float(c[8].replace("$", "")), "nm": float(c[10])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="appendix/tab_grid_full.tex")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    rows = read_table(Path(a.table))
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF, "savefig.facecolor": SURF})
    fig, ax = plt.subplots(figsize=(3.0, 3.72))
    # shape carries the category (colour is redundant): filled circle / open
    # triangle / open square are distinguishable without colour vision.
    groups = {"specialist": ([], BLUE, "o", True, "Specialist collection"),
              "native": ([], BLUE, "^", False, "Domain-specialist encoder"),
              "control": ([], ORANGE, "s", False, "General-domain control (COCO)")}
    for r in rows:
        k = "control" if r["ds"] == "COCO" else ("native" if r["enc"] not in GEN else "specialist")
        groups[k][0].append(r)
    for k, (rs, col, mk, filled, lab) in groups.items():
        ax.scatter([r["nm"] for r in rs], [r["g50"] for r in rs], s=18 if filled else 24, marker=mk,
                   facecolors=col if filled else "none", edgecolors=col, lw=0 if filled else 1.1,
                   label=f"{lab} (n={len(rs)})", zorder=3)
    lim = 0.32
    ax.plot([0, lim], [0, lim], ls=(0, (4, 3)), color=INK2, lw=0.9, zorder=2)
    ax.set_xlim(-0.005, lim); ax.set_ylim(-0.02, lim)
    ax.set_xlabel("Near-miss mass $N_{50}$ (catalog, before training)", fontsize=7, color=INK2)
    ax.set_ylabel("Gain in R@50 from the learned maps", fontsize=7, color=INK2)
    ax.tick_params(labelsize=6.3, colors=INK2, length=2)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(SURF)
    ax.legend(fontsize=5.6, frameon=False, loc="lower right", handletextpad=0.4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "nearmiss_law_shapes.pdf", bbox_inches="tight"); fig.savefig(out / "nearmiss_law_shapes.png", dpi=300, bbox_inches="tight")
    g = np.array([r["g50"] for r in rows]); nm = np.array([r["nm"] for r in rows])
    print("saved", out / "nearmiss_law_shapes.pdf", " R2 vs identity:", round(1 - ((g - nm) ** 2).sum() / ((g - g.mean()) ** 2).sum(), 3))


if __name__ == "__main__":
    main()
