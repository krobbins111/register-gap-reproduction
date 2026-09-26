"""Qualitative showcase: what the two maps do to one query's view of the gallery.

One ROCOv2 caption (SigLIP2, default maps). Top: the five images the frozen
encoder ranks first, then the true image at its raw rank. Bottom: the five the
maps rank first (the true image now first), each tagged with the raw rank it
came from, so the whole neighbourhood is seen to re-sort, not just one point.
Optional right panel: every gallery image as a dot, raw rank vs rank after the
maps (log-log); dots below the diagonal moved up the list. The raw and mapped
top-5 and the true image are coloured.

Caption text and file names come from results/qual_figs/rocov2_test_index.json
(the HF test split in cache order, no-caption rows dropped); images from
<repo>/cache/rocov2/images (override with ICLR_ROCOV2_IMAGES).

Usage (from iclr2027\\):
    python analysis\\make_fig_showcase.py --q 4080 [--q 1754 ...] [--no-scatter] [--title] [--list]
Outputs results/qual_figs/fig_showcase_<q>[_noscatter].png/.pdf and slides/slide_showcase_<q>*.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
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
from PIL import Image

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from render_qualitative_panels import CACHE_ROOT, RESULTS_ROOT  # noqa: E402

OUT = RESULTS_ROOT / "qual_figs"
SLIDES = OUT / "slides"
IMAGES = Path(os.environ.get("ICLR_ROCOV2_IMAGES", CACHE_ROOT.parent / "raw" / "rocov2" / "images"))   # <ltg store>/raw/rocov2/images
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "figure.facecolor": SURF,
                     "savefig.facecolor": SURF})
K = 5


def nrm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load():
    z = np.load(CACHE_ROOT / "rocov2" / "test_embs.npz")
    c, g = nrm(z["caps"].astype(np.float32)), nrm(z["imgs"].astype(np.float32))
    m = np.load(RESULTS_ROOT / "maps" / "rocov2" / "maps_default.npz")
    c2 = nrm(c + c @ m["W_text"].T + m["b_text"])
    g2 = nrm(g + g @ m["W_image"].T + m["b_image"])
    idx = json.load(open(OUT / "rocov2_test_index.json", encoding="utf-8"))["rows"]
    assert len(idx) == len(c), f"index has {len(idx)} rows, cache {len(c)}"
    return c, g, c2, g2, idx


def query_view(q, c, g, c2, g2):
    """Ranks of every gallery image in caption q's list, raw and after."""
    s0, s1 = g @ c[q], g2 @ c2[q]

    def ranks(s):                      # 1 = highest score (ties broken by index, none occur in practice)
        order = np.argsort(-s, kind="stable")
        r = np.empty(len(s), int); r[order] = np.arange(1, len(s) + 1)
        return r
    return ranks(s0), ranks(s1), s0, s1


def thumb(path, size=340):
    im = Image.open(path).convert("RGB")
    f = size / max(im.width, im.height)
    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), SURF)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def frame(ax, color, lw):
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(lw)


def badge(ax, text, color, xy=(0.04, 0.96), fs=7.2, fg="white"):
    ax.text(*xy, text, transform=ax.transAxes, ha="left", va="top", fontsize=fs, color=fg, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=color, ec="none"))


def row(fig, sub, q, order, r0, r1, idx, mapped):
    """K thumbnails, then a sixth slot: the true image at its raw rank (frozen
    row) or the frozen #1 at the rank it fell to (mapped row)."""
    gs = sub.subgridspec(1, K + 2, wspace=0.06, width_ratios=[1] * K + [0.22, 1])
    axs = [None] * (K + 1)
    sp = plt.Subplot(fig, gs[K]); fig.add_subplot(sp); sp.axis("off")
    sp.text(0.5, 0.5, "…", transform=sp.transAxes, ha="center", va="center", fontsize=13, color=MUTED)
    extra = q if not mapped else int(order[-1])
    for k in range(K + 1):
        ax = fig.add_subplot(gs[k if k < K else K + 1]); axs[k] = ax
        j = order[k] if k < K else extra
        ax.imshow(thumb(IMAGES / idx[j]["file"])); ax.set_xticks([]); ax.set_yticks([])
        is_true = (j == q)
        col = AQUA if is_true else (BLUE if mapped else ORANGE)
        if k == K and mapped:
            col = ORANGE
        frame(ax, col if (is_true or k == K) else GRID, 2.2 if (is_true or k == K) else 0.8)
        rank = (r1 if mapped else r0)[j]
        badge(ax, f"#{rank}", col)
        if mapped:
            lab = "true image" if is_true else (f"was #{r0[j]}" if k < K else "frozen #1")
            ax.set_xlabel(lab, fontsize=6.6, color=col if (is_true or k == K) else INK2, labelpad=2)
        elif k == K:
            ax.set_xlabel("true image", fontsize=6.6, color=AQUA, labelpad=2)
    return axs


def scatter(ax, q, r0, r1, top0, top1):
    n = len(r0)
    ax.scatter(r0, r1, s=3, c=GRID, lw=0, rasterized=True, zorder=1)
    ax.plot([1, n], [1, n], ls=(0, (3, 3)), lw=0.7, color=MUTED, zorder=2)
    ax.scatter(r0[top0], r1[top0], s=22, c=ORANGE, lw=0, zorder=4, label="frozen top-5")
    ax.scatter(r0[top1], r1[top1], s=22, c=BLUE, lw=0, zorder=4, label="mapped top-5")
    ax.scatter([r0[q]], [r1[q]], s=95, marker="*", c=AQUA, ec="white", lw=0.5, zorder=5, label="true image")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, n * 1.3); ax.set_ylim(0.8, n * 1.3)
    ax.set_xlabel("rank, frozen encoder", fontsize=7.2, color=INK2)
    ax.set_ylabel("rank, after the maps", fontsize=7.2, color=INK2)
    ax.tick_params(labelsize=6.3, colors=INK2, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_facecolor(SURF)
    ax.annotate(f"{r0[q]} → {r1[q]}", (r0[q], r1[q]), xytext=(6, 8), textcoords="offset points",
                fontsize=6.8, color=AQUA, fontweight="bold")
    ax.set_title("each dot is one gallery image\nbelow the line: moved up the list",
                 fontsize=6.2, color=INK2, loc="left", pad=4)
    ax.legend(loc="upper left", fontsize=6.2, frameon=False, handletextpad=0.3, borderaxespad=0.2)


def fig_showcase(q, data, title=None, with_scatter=True):
    c, g, c2, g2, idx = data
    r0, r1, s0, s1 = query_view(q, c, g, c2, g2)
    top0 = np.argsort(r0)[:K]; top1 = np.argsort(r1)[:K]
    n = len(r0)
    W = 7.2 if with_scatter else 5.4
    fig = plt.figure(figsize=(W, 3.55))
    outer = fig.add_gridspec(1, 2 if with_scatter else 1, width_ratios=[3.05, 1.25] if with_scatter else [1],
                             left=0.012, right=0.985, top=0.87 if title else 0.97, bottom=0.05,
                             wspace=0.16)
    left = outer[0].subgridspec(3, 1, height_ratios=[0.62, 1, 1], hspace=0.42)
    # caption box
    axc = fig.add_subplot(left[0]); axc.axis("off")
    cap = " ".join(idx[q]["caption"].split())
    axc.text(0.0, 0.5, textwrap.fill("“" + cap + "”", 92, max_lines=3, placeholder=" …”"), ha="left", va="center",
             fontsize=7.0, color=INK, linespacing=1.35, transform=axc.transAxes,
             bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.8", fc="white", ec=GRID, lw=0.8))
    axc.text(0.0, 1.12, "Query caption:", ha="left", va="bottom", fontsize=7.2, color=INK2, transform=axc.transAxes)
    # rows
    raw_axes = row(fig, left[1], q, top0, r0, r1, idx, mapped=False)
    map_axes = row(fig, left[2], q, list(top1) + [int(top0[0])], r0, r1, idx, mapped=True)
    fig.text(raw_axes[0].get_position(fig).x0, raw_axes[0].get_position(fig).y1 + 0.012,
             "Frozen encoder: top five retrieved images and the true image at its rank",
             ha="left", va="bottom", fontsize=7.2, color=ORANGE, fontweight="bold")
    fig.text(map_axes[0].get_position(fig).x0, map_axes[0].get_position(fig).y1 + 0.012,
             "After the maps: top five retrieved images and where the frozen #1 landed",
             ha="left", va="bottom", fontsize=7.2, color=BLUE, fontweight="bold")
    if with_scatter:
        axs = fig.add_subplot(outer[1])
        scatter(axs, q, r0, r1, top0, top1)
    if title:
        fig.suptitle(title, fontsize=9, y=0.975)
    return fig, dict(q=int(q), file=idx[q]["file"], caption=cap, rank_raw=int(r0[q]), rank_after=int(r1[q]),
                     raw_top=[int(j) for j in top0], mapped_top=[int(j) for j in top1],
                     mapped_top_from=[int(r0[j]) for j in top1], n_gallery=int(n),
                     score_true_raw=float(s0[q]), score_true_after=float(s1[q]))


def save(fig, name, slide=True):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if slide:
        SLIDES.mkdir(parents=True, exist_ok=True)
        fig.savefig(SLIDES / f"slide_{name.replace('fig_', '')}.png", dpi=250, bbox_inches="tight")
    plt.close(fig); print(f"[fig] -> {OUT / name}.png/.pdf")


def list_candidates(data, lo=20, hi=400, top=30):
    """Queries the maps carry from raw rank lo..hi to rank 1 whose mapped top-5
    look like the true image (mean image cosine) more than the raw top-5 did."""
    c, g, c2, g2, idx = data
    n = len(c)
    S0, S1 = c @ g.T, c2 @ g2.T
    own0, own1 = S0[np.arange(n), np.arange(n)], S1[np.arange(n), np.arange(n)]
    R0, R1 = 1 + (S0 > own0[:, None]).sum(1), 1 + (S1 > own1[:, None]).sum(1)
    GG = g @ g.T; np.fill_diagonal(GG, -1)
    rows = []
    for q in np.where((R0 >= lo) & (R0 <= hi) & (R1 == 1))[0]:
        t0 = np.argsort(-S0[q])[:K]; t1 = [j for j in np.argsort(-S1[q])[:K + 1] if j != q][:K]
        rows.append((float(GG[q, t1].mean() - GG[q, t0].mean()), int(q), int(R0[q]), idx[q]["caption"][:90]))
    rows.sort(reverse=True)
    for jump, q, r, cap in rows[:top]:
        print(f"q={q:5d}  raw rank {r:3d} -> 1   coherence jump {jump:+.2f}   {cap}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, action="append", default=[])
    ap.add_argument("--no-scatter", action="store_true", help="also write the version without the rank panel")
    ap.add_argument("--title", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    data = load()
    if a.list:
        list_candidates(data)
    for q in a.q:
        fig, meta = fig_showcase(q, data, title="One query, before and after the maps" if a.title else None)
        save(fig, f"fig_showcase_{q}")
        json.dump(meta, open(OUT / f"showcase_{q}.json", "w", encoding="utf-8"), indent=1)
        if a.no_scatter:
            fig, _ = fig_showcase(q, data, title=None, with_scatter=False)
            save(fig, f"fig_showcase_{q}_noscatter")
