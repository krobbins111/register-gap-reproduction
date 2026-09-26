"""Register-gap illustration figures from the picked candidates.

Reads results/qual_figs/register_cands_<name>.json (analysis\\find_register_examples.py)
and renders, for the chosen rows:

  scans     (image side, ROCOv2)  two similar chest X-rays as column headers,
            their two captions as row headers, and the 2x2 matrix of the rank
            each X-ray gets in each caption's list, frozen and after the maps
            (cosines in the footer).
  goodnews  (text side)  a COCO photo whose web caption retrieves it at rank 1
            next to its GoodNews twin whose press caption does not, until the maps.
  fashion   (text side)  FACAD product copy retrieves its photo; the
            Fashion200k description of a look-alike photo does not, until the maps.

No in-figure titles unless --title (the LaTeX caption / slide title carries it).
Usage (from iclr2027\\):
    python analysis\\make_fig_register_pairs.py --rocov2 13 --goodnews 16 --fashion 2 --nwpu N --intro [--title]
Outputs results/qual_figs/fig_register_{scans,goodnews,fashion,nwpu,intro,intro_row}.png/.pdf
    --intro-row: the one-line page-1 form (four images, captions under, no ranks)
    --intro-split: the same as two files, intro_text / intro_image, for a subfigure pair,
                   each also saved as *_notext (register label only, no quoted caption)
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from render_qualitative_panels import RESULTS_ROOT  # noqa: E402

OUT = RESULTS_ROOT / "qual_figs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "figure.facecolor": SURF,
                     "savefig.facecolor": SURF})


def fix_path(p):
    """Optional path rewrite for rendering off-machine: ICLR_PATH_MAP='C:\\\\a\\\\b=/mnt/x'."""
    m = os.environ.get("ICLR_PATH_MAP")
    if m and "=" in m:
        src, dst = m.split("=", 1)
        if p.startswith(src):
            p = dst + p[len(src):].replace("\\", "/")
    return p


def thumb(path, size=340, crop=False):
    """Square thumbnail: letterboxed on the surface colour, or (crop=True) a
    centre crop with no padding."""
    im = Image.open(fix_path(path)).convert("RGB")
    if crop:
        s_ = min(im.width, im.height); x0 = (im.width - s_) // 2; y0 = (im.height - s_) // 2
        im = im.crop((x0, y0, x0 + s_, y0 + s_))
    f = size / max(im.width, im.height)
    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), SURF)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def tidy(t):
    t = " ".join(str(t).split())
    t = re.sub(r"^[A-Z ,&'-]{6,}\s", "", t)          # drop ALL-CAPS wire slug ("TEMPO AND DYNAMICS ")
    t = t.replace(" .", ".").replace(" ,", ",")
    while ".." in t:
        t = t.replace("..", ".")
    t = t[0].upper() + t[1:] if t else t
    if t and t[-1] not in ".!?":
        t += "."
    return t


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print(f"[fig] -> {OUT / name}.png/.pdf")


# ── scans: 2x2 rank matrices ────────────────────────────────────────────────
def cross_ranks(d):
    """Rank of each X-ray in each caption's list (raw and after the maps).
    Uses the keys written by find_register_examples.py; computes them from
    the caches if an older candidate JSON lacks them."""
    keys = ("rank_ci_gj", "rank_cj_gi", "rank_ci_gj_after", "rank_cj_gi_after")
    if all(k in d for k in keys):
        return d
    from render_qualitative_panels import CACHE_ROOT
    z = np.load(CACHE_ROOT / "rocov2" / "test_embs.npz")
    nrm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    c, g = nrm(z["caps"].astype(np.float32)), nrm(z["imgs"].astype(np.float32))
    m = np.load(RESULTS_ROOT / "maps" / "rocov2" / "maps_default.npz")
    c2, g2 = nrm(c + c @ m["W_text"].T + m["b_text"]), nrm(g + g @ m["W_image"].T + m["b_image"])
    rk = lambda cq, G, t: int(1 + ((G @ cq) > (G @ cq)[t]).sum())
    i, j = d["i"], d["j"]
    d.update(rank_ci_gj=rk(c[i], g, j), rank_cj_gi=rk(c[j], g, i),
             rank_ci_gj_after=rk(c2[i], g2, j), rank_cj_gi_after=rk(c2[j], g2, i))
    return d


def fig_scans(d, title):
    d = cross_ranks(d)
    fig = plt.figure(figsize=(7.2, 3.7))
    gs = fig.add_gridspec(3, 5, width_ratios=[2.6, 1, 1, 1, 1], height_ratios=[1.15, 1, 1],
                          left=0.01, right=0.99, top=0.84 if title else 0.9, bottom=0.15, wspace=0.08, hspace=0.06)
    raw = np.array([[d["rank_i"], d["rank_ci_gj"]], [d["rank_cj_gi"], d["rank_j"]]])
    aft = np.array([[d["rank_i_after"], d["rank_ci_gj_after"]], [d["rank_cj_gi_after"], d["rank_j_after"]]])
    imgs = (d["path_i"], d["path_j"])
    caps = (d["cap_i"], d["cap_j"])
    for blk, (c0, lab) in enumerate(((1, "frozen encoder"), (3, "after the learned maps"))):
        for k in range(2):
            ax = fig.add_subplot(gs[0, c0 + k]); ax.imshow(thumb(imgs[k], 300)); ax.axis("off")
            ax.set_title(f"X-ray {'AB'[k]}", fontsize=8, color=MUTED, pad=2)
            if k == 0:
                ax.text(1.04, 1.22, lab, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK2,
                        transform=ax.transAxes)
    for r in range(2):
        ax = fig.add_subplot(gs[1 + r, 0]); ax.axis("off")
        ax.text(0.02, 0.5, f"caption {'AB'[r]}:", fontsize=8.2, color=INK2, va="bottom", ha="left", transform=ax.transAxes)
        ax.text(0.02, 0.47, textwrap.fill(tidy(caps[r]), 46), fontsize=7.6, color=INK, va="top", ha="left",
                transform=ax.transAxes, style="italic", linespacing=1.15)
    n_gal = 9927
    for blk, (M, c0) in enumerate(((raw, 1), (aft, 3))):
        for r in range(2):
            for k in range(2):
                ax = fig.add_subplot(gs[1 + r, c0 + k]); ax.set_xticks([]); ax.set_yticks([])
                v = int(M[r, k]); own = (r == k)
                # shade by how close to the top the image lands (log scale)
                shade = 0.12 + 0.78 * (1 - np.log10(v) / np.log10(n_gal))
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor=BLUE if own else "#9a9a95", alpha=shade * (1.0 if own else 0.5), lw=0))
                for sp in ax.spines.values():
                    sp.set_edgecolor(GRID)
                ax.text(0.5, 0.5, f"#{v:,}", ha="center", va="center", fontsize=11.5, transform=ax.transAxes,
                        color="white" if (own and shade > 0.55) else INK, fontweight="bold" if own else "normal")
    cos = (f"cosine, caption × X-ray:  A·A {d['ci_gi']:.3f} → {d['ci_gi_after']:.3f}   A·B {d['ci_gj']:.3f} → {d['ci_gj_after']:.3f}   "
           f"B·B {d['cj_gj']:.3f} → {d['cj_gj_after']:.3f}   B·A {d['cj_gi']:.3f} → {d['cj_gi_after']:.3f}\n"
           f"X-ray·X-ray {d['img_sim']:.2f} → {d['img_sim_after']:.2f}     caption·caption {d['cap_sim']:.2f} → {d['cap_sim_after']:.2f}     "
           f"gallery of {n_gal:,} images")
    fig.text(0.63, 0.012, cos, ha="center", va="bottom", fontsize=7.6, color=INK2, linespacing=1.4)
    fig.text(0.02, 0.86 if not title else 0.8, "rank of the X-ray in the caption's list", fontsize=8, color=MUTED, va="bottom")
    if title:
        fig.suptitle("Two similar chest X-rays, two different captions", fontsize=12, fontweight="bold", x=0.01, ha="left")
    save(fig, "fig_register_scans")


# ── text side: web-register twin vs specialist caption ─────────────────────
def fig_text(d, name, head_a, head_b, title, note_a, note_b, show_rank=True):
    """show_rank=False: the same panel with the 'retrieved: rank ...' lines
    withheld (a build slide before the reveal); saved as <name>_norank."""
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.2, 1, 0.62], left=0.02, right=0.98,
                          top=0.9 if title else 0.98, bottom=0.02, wspace=0.08, hspace=0.05)
    cols = (("a", head_a, d["a_path"], d["a_cap"], f"rank #{d['a_rank']}", INK, note_a),
            ("b", head_b, d["b_path"], d["b_cap"], f"rank #{d['b_rank']} → #{d['b_rank_after']} with the learned maps", BLUE, note_b))
    for k, (side, head, path, cap, rk, col, note) in enumerate(cols):
        ax = fig.add_subplot(gs[0, k]); ax.axis("off")
        ax.text(0, 0.12, textwrap.fill(head, 46), fontsize=9.2, fontweight="bold", color=INK2, va="bottom",
                transform=ax.transAxes, linespacing=1.15)
        ax = fig.add_subplot(gs[1, k]); ax.imshow(thumb(path, 420)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRID); s.set_linewidth(1.2)
        ax = fig.add_subplot(gs[2, k]); ax.axis("off")
        ax.text(0, 0.95, textwrap.fill(f"“{tidy(cap)}”", 60), fontsize=7.9, style="italic", va="top", ha="left",
                transform=ax.transAxes, linespacing=1.2)
        if show_rank:
            ax.text(0, 0.16, f"retrieved: {rk}", fontsize=9, fontweight="bold", color=col, va="bottom", transform=ax.transAxes)
        ax.text(0, 0.0, note, fontsize=7.4, color=MUTED, va="bottom", transform=ax.transAxes)
    fig.text(0.5, 0.995 if not title else 0.93, "", ha="center")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", x=0.02, ha="left")
    save(fig, f"fig_register_{name}" + ("" if show_rank else "_norank"))


def norm_twin(d):
    """Accept both twin-JSON layouts: the generic a/b keys and the earlier
    GoodNews-only gn/coco keys."""
    if "gn" in d and "b" not in d:
        d = dict(d, b=d["gn"], a=d["coco"], b_rank=d["gn_rank"], b_rank_after=d["gn_rank_after"],
                 a_rank=d["coco_rank"], bcap_bimg=d["gncap_gnimg"], acap_bimg=d["cococap_gnimg"],
                 acap_aimg=d["cococap_cocoimg"], bcap_aimg=d["gncap_cocoimg"],
                 bcap_bimg_after=d["gncap_gnimg_after"], b_cap=d["gn_cap"], b_path=d["gn_path"],
                 a_cap=d["coco_cap"], a_path=d["coco_path"])
    return d


# ── intro: two rows, full width (text register / depiction register) ───────
def _pair_row(fig, sub, d, head_a, head_b, row_note, show_rank=True):
    """[img A][text A]   [img B][text B] inside one gridspec row, with one
    centred note under the row naming the pair."""
    g = sub.subgridspec(1, 4, width_ratios=[1, 1.55, 1, 1.55], wspace=0.06)
    cols = ((head_a, d["a_path"], d["a_cap"], f"rank #{d['a_rank']}", INK),
            (head_b, d["b_path"], d["b_cap"], f"rank #{d['b_rank']} → #{d['b_rank_after']} with the maps", BLUE))
    for k, (head, path, cap, rk, col) in enumerate(cols):
        axi = fig.add_subplot(g[0, 2 * k]); axi.imshow(thumb(path, 400)); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(1.0)
        axt = fig.add_subplot(g[0, 2 * k + 1]); axt.axis("off")
        axt.text(0.04, 0.98, textwrap.fill(head, 34), fontsize=8.3, fontweight="bold", color=INK2, va="top",
                 transform=axt.transAxes, linespacing=1.15)
        axt.text(0.04, 0.72, textwrap.fill(f"“{tidy(cap)}”", 40, max_lines=5, placeholder=" …"), fontsize=7.4,
                 style="italic", va="top", ha="left", transform=axt.transAxes, linespacing=1.2)
        if show_rank:
            axt.text(0.04, 0.14, textwrap.fill(f"retrieved: {rk}", 36), fontsize=8, fontweight="bold", color=col,
                     va="bottom", transform=axt.transAxes, linespacing=1.15)
    pos = sub.get_position(fig)
    fig.text((pos.x0 + pos.x1) / 2, pos.y0 - 0.025, row_note, ha="center", va="top", fontsize=7, color=MUTED)


def fig_intro(d_text, d_img, title, show_rank=True):
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 1, left=0.01, right=0.99, top=0.9 if title else 0.99, bottom=0.06, hspace=0.3)
    _pair_row(fig, gs[0], d_text,
              "Generic description: a web caption (COCO)", "Journalistic register: a press caption (GoodNews)",
              f"COCO and GoodNews pair: two photographs of the same kind of scene, cosine {d_text['img_sim']:.2f}, "
              f"captioned in two registers",
              show_rank=show_rank)
    _pair_row(fig, gs[1], d_img,
              "Generic depiction: a ground-level photograph (COCO)", "Remote-sensing register: an aerial tile (NWPU)",
              f"COCO and NWPU pair: two captions that say the same thing, cosine {d_img['cap_sim']:.2f}, "
              f"attached to two registers of image",
              show_rank=show_rank)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left")
    save(fig, "fig_register_intro" + ("" if show_rank else "_norank"))


# ── intro (row form): four images in one line, text under each, no ranks ───
def fig_intro_row(d_text, d_img, title):
    """One full-width row: [COCO photo][GoodNews photo]  [COCO photo][NWPU tile],
    each image with its register label and caption underneath. No rank text;
    the LaTeX caption carries the ranks."""
    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.14, 1, 1], left=0.005, right=0.995,
                          top=0.985 if not title else 0.9, bottom=0.075, wspace=0.05)
    cells = ((0, "Web caption (COCO)", d_text["a_path"], d_text["a_cap"], INK2),
             (1, "Press caption (GoodNews)", d_text["b_path"], d_text["b_cap"], BLUE),
             (3, "Photograph (COCO)", d_img["a_path"], d_img["a_cap"], INK2),
             (4, "Aerial tile (NWPU)", d_img["b_path"], d_img["b_cap"], BLUE))
    for col, head, path, cap, hc in cells:
        sub = gs[0, col].subgridspec(2, 1, height_ratios=[1, 0.5], hspace=0.03)
        axi = fig.add_subplot(sub[0]); axi.imshow(thumb(path, 420, crop=True)); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(1.0)
        axt = fig.add_subplot(sub[1]); axt.axis("off")
        axt.text(0.0, 0.97, head, fontsize=7.6, fontweight="bold", color=hc, va="top", ha="left", transform=axt.transAxes)
        axt.text(0.0, 0.74, textwrap.fill(f"\u201c{tidy(cap)}\u201d", 34, max_lines=4, placeholder=" \u2026"), fontsize=6.9,
                 style="italic", va="top", ha="left", transform=axt.transAxes, linespacing=1.2, color=INK)
    # pair braces: a thin rule over each pair with its note
    for c0, c1, note in ((0, 1, f"same kind of photograph (cosine {d_text['img_sim']:.2f}), two registers of caption"),
                         (3, 4, f"same caption (cosine {d_img['cap_sim']:.2f}), two registers of image")):
        p0 = gs[0, c0].get_position(fig); p1 = gs[0, c1].get_position(fig)
        fig.text((p0.x0 + p1.x1) / 2, 0.0, note, ha="center", va="bottom", fontsize=6.6, color=MUTED)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left")
    save(fig, "fig_register_intro_row")


# ── intro (split form): one pair per file, for a LaTeX subfigure pair ──────
def fig_intro_pair(d, name, head_a, head_b, with_caption=True):
    """[img A][img B] with the register label under each and, if
    with_caption, the quoted caption; no ranks, no note (the LaTeX
    subcaption names the side of the gap). Saved as
    fig_register_intro_<name>[_notext]."""
    fig = plt.figure(figsize=(3.5, 2.3 if with_caption else 1.85))
    gs = fig.add_gridspec(1, 2, left=0.005, right=0.995, top=0.99, bottom=0.005, wspace=0.05)
    for k, (head, path, cap, hc) in enumerate(((head_a, d["a_path"], d["a_cap"], INK2), (head_b, d["b_path"], d["b_cap"], BLUE))):
        sub = gs[0, k].subgridspec(2, 1, height_ratios=[1, 0.55 if with_caption else 0.11], hspace=0.03)
        axi = fig.add_subplot(sub[0]); axi.imshow(thumb(path, 420, crop=True)); axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(1.0)
        axt = fig.add_subplot(sub[1]); axt.axis("off")
        axt.text(0.0, 0.97 if with_caption else 0.9, head, fontsize=7.6, fontweight="bold", color=hc, va="top", ha="left", transform=axt.transAxes)
        if with_caption:
            axt.text(0.0, 0.72, textwrap.fill(f"\u201c{tidy(cap)}\u201d", 34, max_lines=4, placeholder=" \u2026"), fontsize=6.9,
                     style="italic", va="top", ha="left", transform=axt.transAxes, linespacing=1.2, color=INK)
    save(fig, f"fig_register_intro_{name}" + ("" if with_caption else "_notext"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocov2", type=int, default=None, help="row # in register_cands_rocov2.json")
    ap.add_argument("--goodnews", type=int, default=None)
    ap.add_argument("--fashion", type=int, default=None)
    ap.add_argument("--nwpu", type=int, default=None, help="row # in register_cands_nwpu.json (find_register_examples.py --only nwpu)")
    ap.add_argument("--intro", action="store_true", help="two-row page-1 figure: --goodnews row over --nwpu row")
    ap.add_argument("--intro-split", action="store_true", help="page-1 figure as two files (text side / image side) for a LaTeX subfigure pair")
    ap.add_argument("--intro-row", action="store_true", help="one-row page-1 figure: four images in a line, captions under, no ranks")
    ap.add_argument("--title", action="store_true", help="slide mode: in-figure titles")
    ap.add_argument("--no-rank", action="store_true", help="also save each text panel without the rank lines (<name>_norank)")
    a = ap.parse_args()
    if a.rocov2 is not None:
        d = json.load(open(OUT / "register_cands_rocov2.json", encoding="utf-8"))[a.rocov2]
        fig_scans(d, a.title)
    if a.goodnews is not None:
        d = norm_twin(json.load(open(OUT / "register_cands_goodnews.json", encoding="utf-8"))[a.goodnews])
        fig_text(d, "goodnews",
                 "Generic description: a web caption (COCO)", "Journalistic register: a press caption (GoodNews)",
                 "Same kind of photo, two registers of caption" if a.title else None,
                 f"COCO · photo·photo cosine {d['img_sim']:.2f} with its twin",
                 f"GoodNews · the COCO caption scores this photo {d['acap_bimg']:.3f}; its own caption {d['bcap_bimg']:.3f}")
    if a.fashion is not None:
        d = norm_twin(json.load(open(OUT / "register_cands_fashion.json", encoding="utf-8"))[a.fashion])
        fig_text(d, "fashion",
                 "Marketing register: retail product copy (FACAD)", "Descriptive register: a garment description (Fashion200k)",
                 "Same catalogue photography, two registers of caption" if a.title else None,
                 f"FACAD · photo·photo cosine {d['img_sim']:.2f} with its twin",
                 f"Fashion200k · caption·caption cosine {d['cap_sim']:.2f}")
        if a.no_rank:
            fig_text(d, "fashion",
                     "Marketing register: retail product copy (FACAD)", "Descriptive register: a garment description (Fashion200k)",
                     "Same catalogue photography, two registers of caption" if a.title else None,
                     f"FACAD · photo·photo cosine {d['img_sim']:.2f} with its twin",
                     f"Fashion200k · caption·caption cosine {d['cap_sim']:.2f}", show_rank=False)
    _main_extra(a)


def _main_extra(a):
    if a.nwpu is not None:
        d = norm_twin(json.load(open(OUT / "register_cands_nwpu.json", encoding="utf-8"))[a.nwpu])
        fig_text(d, "nwpu",
                 "Generic depiction: a ground-level photograph (COCO)", "Remote-sensing register: an aerial tile (NWPU)",
                 "Same kind of caption, two registers of image" if a.title else None,
                 f"COCO · caption·caption cosine {d['cap_sim']:.2f} with its twin",
                 f"NWPU · the web caption scores this tile {d['acap_bimg']:.3f}; its own caption {d['bcap_bimg']:.3f}")
    if a.intro:
        dt = norm_twin(json.load(open(OUT / "register_cands_goodnews.json", encoding="utf-8"))[a.goodnews])
        di = norm_twin(json.load(open(OUT / "register_cands_nwpu.json", encoding="utf-8"))[a.nwpu])
        fig_intro(dt, di, "The register gap, in action" if a.title else None)
        if a.no_rank:
            fig_intro(dt, di, "The register gap, in action" if a.title else None, show_rank=False)
    if a.intro_row:
        dt = norm_twin(json.load(open(OUT / "register_cands_goodnews.json", encoding="utf-8"))[a.goodnews])
        di = norm_twin(json.load(open(OUT / "register_cands_nwpu.json", encoding="utf-8"))[a.nwpu])
        fig_intro_row(dt, di, "The register gap" if a.title else None)
    if a.intro_split:
        dt = norm_twin(json.load(open(OUT / "register_cands_goodnews.json", encoding="utf-8"))[a.goodnews])
        di = norm_twin(json.load(open(OUT / "register_cands_nwpu.json", encoding="utf-8"))[a.nwpu])
        for wc in (True, False):
            fig_intro_pair(dt, "text", "Web caption (COCO)", "Press caption (GoodNews)", with_caption=wc)
            fig_intro_pair(di, "image", "Photograph (COCO)", "Aerial tile (NWPU)", with_caption=wc)


if __name__ == "__main__":
    main()
