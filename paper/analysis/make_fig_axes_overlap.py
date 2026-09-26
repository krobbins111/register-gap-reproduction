"""Redraw the cross-collection axis-overlap heatmaps from results/map_anatomy_<mtag>.json.

Two panels: overlap of the top-k displacement axes of the text map (left) and of
the image map (right) between every pair of collections.  Overlap of two
k-dimensional subspaces is ||P^T Q||_F^2 / k in [0, 1]; a random k-subspace of
R^d scores k/d.  No maps or embeddings are needed, only the saved JSON.

Outputs results/figs/map_axes_overlap_<mtag>.pdf/.png
Usage   python analysis\\make_fig_axes_overlap.py [--mtag default]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from config import RESULTS_ROOT  # noqa: E402

DSNAME = {"coco": "COCO", "facad": "FACAD", "fashion200k": "Fashion200k", "goodnews": "GoodNews", "semart": "SemArt",
          "scimmir": "SciMMIR", "skincap": "SkinCAP", "treeoflife": "TreeOfLife", "rocov2": "ROCOv2", "rsicd": "RSICD", "nwpu": "NWPU"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"


def draw(res: dict, out_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF, "savefig.facecolor": SURF})
    order = res["order"]; names = [DSNAME.get(d, d) for d in order]; n = len(order); k = res["k"]
    dim = res["per_collection"][order[0]]["dim"]; rand = k / dim
    XT = np.array(res["cross_text_axes_overlap"]); XI = np.array(res["cross_image_axes_overlap"])
    off = ~np.eye(n, dtype=bool)
    vmax = float(max(XT[off].max(), XI[off].max()))
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#2a78d6", "#0d3f86"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True, gridspec_kw={"wspace": 0.05})
    for ax, M, title in ((axes[0], XT, "Text-map axes"), (axes[1], XI, "Image-map axes")):
        im = ax.imshow(np.where(off, M, np.nan), cmap=cmap, vmin=0, vmax=vmax)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                            color="white" if M[i, j] > 0.6 * vmax else INK)
        # diagonal: a dot marks "same collection"
        ax.scatter(range(n), range(n), s=4, color=GRID, lw=0)
        ax.set_xticks(range(n), names, rotation=45, ha="right", rotation_mode="anchor", fontsize=6.4)
        ax.set_yticks(range(n), names, fontsize=6.4)
        ax.set_title(title, fontsize=8, color=INK, pad=6)
        ax.tick_params(length=0, colors=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor(SURF)
    fig.suptitle(f"Overlap of the top-{k} displacement axes between collections  (random subspace: {rand:.2f})",
                 fontsize=7.4, color=INK2, y=0.92)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=28)
    cb.ax.tick_params(labelsize=6, colors=INK2, length=2); cb.outline.set_visible(False)
    cb.set_label("subspace overlap", fontsize=6.6, color=INK2)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[axes-overlap] -> {out_stem.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtag", default="default")
    a = ap.parse_args()
    res = json.load(open(RESULTS_ROOT / f"map_anatomy_{a.mtag}.json", encoding="utf-8"))
    figs = RESULTS_ROOT / "figs"; figs.mkdir(exist_ok=True)
    draw(res, figs / f"map_axes_overlap_{a.mtag}")


if __name__ == "__main__":
    main()
