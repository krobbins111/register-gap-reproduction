"""Render the qualitative figures from qualitative_<tag>.json + real images.

Runs in the repo venv on Windows (needs matplotlib + PIL + the legacy
`indexability/splits` loaders and their cached/materialized images). For each
cell it re-resolves the TEST split with the exact loader + limit that built the
embedding cache (mirrors symmetric_projection.load_split), hard-fails if the
count does not match the JSON's n_test (no silently wrong panels), then draws:

    <ds>_<tag>_flip{1..K}.png   rescued query: caption, raw top-5 row vs
                                corrected top-5 row, true match outlined,
                                rank R_raw -> R_cor
    <ds>_<tag>_regression.png   the worst regression (honesty panel)
    <ds>_<tag>_hubs.png         the images that answer every query: top hub
                                images with raw vs corrected top-10 counts
    <ds>_<tag>_axes.png         extremes of the top amplified image-side axes

Usage:  python analysis\\render_qualitative_panels.py --dataset nwpu --mtag metaclip2-ww-huge
Output: results/qual_figs/
"""
from __future__ import annotations

import argparse
import glob
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
ICLR = HERE.parents[1]
LEGACY = Path(os.environ.get("ICLR_LEGACY_PIPELINE", ICLR / "legacy"))
sys.path.insert(0, str(LEGACY))          # `splits` + LEGACY config (real data env)
from splits import load_split_pairs      # noqa: E402

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("iclr_roots", ICLR / "iclr_roots.py"); _roots = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_roots)
CACHE_ROOT = _roots.legacy_cache_root()
RESULTS_ROOT = Path(os.environ.get("ICLR_RESULTS_ROOT", ICLR / "results"))
OUT = RESULTS_ROOT / "qual_figs"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": False,
})
DSNAME = {"nwpu": "NWPU", "rocov2": "ROCOv2", "skincap": "SkinCAP",
          "treeoflife": "TreeOfLife", "rsicd": "RSICD", "scimmir": "SciMMIR",
          "fashion200k": "Fashion200k", "semart": "SemArt", "coco": "COCO",
          "goodnews": "GoodNews", "facad": "FACAD"}

# mtag -> model id (from grid_runner's matrix), for the embedding spot-check.
MTAG_MODEL = {
    "default": "google/siglip2-so400m-patch16-384",
    "medsiglip": "google/medsiglip-448",
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "clip-vitl14-laion2b": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "siglipv1-so400m-384": "google/siglip-so400m-patch14-384",
    "metaclip2-ww-huge": "facebook/metaclip-2-worldwide-huge-quickgelu",
    "remoteclip-vitl14": "chendelong/RemoteCLIP-ViT-L-14",
    "georsclip-vitb32": "Zilun/GeoRSCLIP-ViT-B-32",
    "fashionclip": "patrickjohncyh/fashion-clip",
    "marqo-fashionsiglip": "Marqo/marqo-fashionSigLIP",
    "bioclip": "imageomics/bioclip",
    "bioclip2": "imageomics/bioclip-2",
}


def verify_alignment(ds, mtag, paths, n_check=2):
    """End-to-end ordering proof: re-embed a few test images with the actual
    encoder and compare to the cached embedding rows. Catches any split-order
    or file drift that a length check cannot."""
    from backbones import make_embedder  # legacy pipeline, same env as caches
    tag = f"_{mtag}" if mtag else ""
    exact = CACHE_ROOT / ds / f"test_embs{tag}.npz"
    if not exact.exists():
        cands = sorted(glob.glob(str(CACHE_ROOT / ds / f"test_embs{tag}_n*.npz")),
                       key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
        if mtag == "":
            cands = [p for p in cands
                     if Path(p).name.split("_embs")[1].startswith("_n")]
        exact = Path(cands[0])
    cached = np.load(exact)["imgs"]
    idxs = [0, len(paths) // 2, len(paths) - 1][:max(1, n_check)]
    emb = make_embedder(MTAG_MODEL[mtag or "default"])
    fresh = emb.embed_images([paths[i] for i in idxs], batch_size=len(idxs))
    for k, i in enumerate(idxs):
        cos = float(np.dot(fresh[k], cached[i]))
        print(f"  [verify] test[{i}] re-embedded cos vs cache = {cos:.4f}")
        if cos < 0.98:
            raise RuntimeError(
                f"ALIGNMENT FAILURE: test index {i} re-embeds with cosine "
                f"{cos:.3f} to its cached row - the split ordering does NOT "
                f"match the cache. Refusing to render.")
    print("  [verify] image<->embedding alignment confirmed")


def resolve_split(ds: str, mtag: str, n_expected: int):
    """(texts, paths) for the test split in EXACTLY the cache's ordering.
    Mirrors symmetric_projection.load_split: exact full-split cache file first,
    else the largest _n<limit> variant (whose ordering is load_split_pairs with
    that limit, seed 0). Hard-fails on any count mismatch."""
    tag = f"_{mtag}" if mtag else ""
    exact = CACHE_ROOT / ds / f"test_embs{tag}.npz"
    limit = None
    if not exact.exists():
        cands = sorted(glob.glob(str(CACHE_ROOT / ds / f"test_embs{tag}_n*.npz")),
                       key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
        if mtag == "":
            cands = [p for p in cands
                     if Path(p).name.split("_embs")[1].startswith("_n")]
        if not cands:
            raise FileNotFoundError(f"no test cache for {ds}/{mtag or 'default'}")
        limit = int(cands[0].rsplit("_n", 1)[1].split(".")[0])
    texts, paths = load_split_pairs(ds, "test", limit=limit, seed=0)
    if len(texts) != n_expected:
        raise RuntimeError(
            f"ORDER MISMATCH: split loader gave {len(texts)} test pairs but the "
            f"qualitative JSON was computed on {n_expected}. Refusing to render "
            f"(indices would point at the wrong images).")
    return texts, paths


def thumb(path, size=340):
    """Square thumbnail that both down- AND up-scales so small sources (e.g.
    256px NWPU tiles) fill the frame instead of floating in padding."""
    im = Image.open(path).convert("RGB")
    f = size / max(im.width, im.height)
    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))),
                   Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), SURF)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def frame(ax, color, lw):
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])


def flip_panel(rec, texts, paths, title, fname):
    q = rec["query_idx"]
    rows = [("Raw ranking", rec["raw_top5"], rec["raw_top5_scores"], rec["raw_rank"]),
            ("With learned maps", rec["cor_top5"], rec["cor_top5_scores"], rec["cor_rank"])]
    fig, axes = plt.subplots(2, 5, figsize=(11.5, 5.9))
    cap = textwrap.fill(" ".join(str(texts[q]).split()), 118, max_lines=3,
                        placeholder=" ...")
    fig.suptitle(f"“{cap}”", x=0.045, y=0.985, ha="left", fontsize=10.5,
                 color=INK, style="italic", wrap=True)
    for ri, (lab, top5, scores, rank) in enumerate(rows):
        for ci in range(5):
            ax = axes[ri][ci]
            gi = top5[ci]
            ax.imshow(thumb(paths[gi]))
            is_true = (gi == q)
            frame(ax, BLUE if is_true else GRID, 3.5 if is_true else 1.0)
            ax.set_title(f"#{ci+1}  {scores[ci]:.3f}" + ("  TRUE" if is_true else ""),
                         fontsize=8.5, color=(BLUE if is_true else MUTED), pad=3)
        axes[ri][0].set_ylabel(lab, fontsize=10.5, color=INK2)
        axes[ri][4].text(1.06, 0.5, f"true match\nrank {rank}",
                         transform=axes[ri][4].transAxes, fontsize=9.5,
                         color=(BLUE if rank <= 5 else ORANGE), va="center")
    fig.text(0.045, 0.012, title, fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.055, right=0.9, top=0.86, bottom=0.055,
                        wspace=0.06, hspace=0.26)
    fig.savefig(OUT / fname, dpi=220, bbox_inches="tight")
    plt.close(fig); print("  saved", fname)


def hubs_panel(rep, paths, ds, tag):
    items = rep["hubs"]["items"][:5]
    n_q = rep["n_test"]
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.3))
    for ax, it in zip(axes, items):
        ax.imshow(thumb(paths[it["gallery_idx"]]))
        frame(ax, GRID, 1.0)
        ax.set_title(f"raw: top-10 for {it['raw_count']} of {n_q}\n"
                     f"fixed: {it['cor_count']}", fontsize=9,
                     color=INK2, pad=4)
    occ_r, occ_c = rep["hubs"]["top5_occupancy_raw"], rep["hubs"]["top5_occupancy_cor"]
    fig.suptitle(f"{DSNAME[ds]}: the images that answer every query",
                 x=0.045, ha="left", fontsize=13, color=INK)
    fig.text(0.045, 0.02, f"these 5 gallery images hold {occ_r:.0%} of all raw "
             f"top-10 slots; {occ_c:.0%} after the learned maps",
             fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.72, bottom=0.12, wspace=0.06)
    fig.savefig(OUT / f"{ds}_{tag}_hubs.png", dpi=220, bbox_inches="tight")
    plt.close(fig); print("  saved hubs")


def axes_panel(rep, paths, ds, tag, n_axes=2, per=6):
    ax_recs = rep["amplified_axes"][:n_axes]
    fig, axes = plt.subplots(2 * n_axes, per, figsize=(1.75 * per, 3.6 * n_axes))
    for ai, arec in enumerate(ax_recs):
        for side, ids in (("high", arec["top_pos"][:per]),
                          ("low", arec["top_neg"][:per])):
            r = 2 * ai + (0 if side == "high" else 1)
            for ci in range(per):
                ax = axes[r][ci]
                ax.imshow(thumb(paths[ids[ci]], 300))
                frame(ax, GRID, 1.0)
            axes[r][0].set_ylabel(f"axis {arec['axis']} {side}\n"
                                  f"(σ = {arec['sigma']:.2f})",
                                  fontsize=9.5, color=INK2)
    fig.suptitle(f"{DSNAME[ds]}: what the most-amplified learned axes separate",
                 x=0.045, ha="left", fontsize=13, color=INK)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.03,
                        wspace=0.05, hspace=0.12)
    fig.savefig(OUT / f"{ds}_{tag}_axes.png", dpi=220, bbox_inches="tight")
    plt.close(fig); print("  saved axes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--flips", type=int, default=5)
    ap.add_argument("--axes", type=int, default=2)
    ap.add_argument("--min-raw", type=int, default=50,
                    help="flip selection: true match's raw rank at least this")
    ap.add_argument("--max-raw", type=int, default=600,
                    help="flip selection: raw rank at most this (beyond a few "
                         "hundred the caption is usually too vague to read as "
                         "a fair example)")
    ap.add_argument("--min-caption-chars", type=int, default=60,
                    help="only show flips whose caption is at least this long "
                         "(descriptiveness proxy)")
    ap.add_argument("--verify", type=int, default=2,
                    help="re-embed this many test images and check them "
                         "against the cached embeddings (0 = skip)")
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    tag = mtag or "default"

    jpath = RESULTS_ROOT / ds / f"qualitative_{tag}.json"
    if not jpath.exists():
        jpath = CACHE_ROOT / ds / f"qualitative_{tag}.json"
    rep = json.load(open(jpath))
    print(f"[panels] {ds}/{tag}: n_test={rep['n_test']}")
    texts, paths = resolve_split(ds, mtag, rep["n_test"])
    OUT.mkdir(parents=True, exist_ok=True)
    if args.verify:
        verify_alignment(ds, mtag, paths, args.verify)

    # Select credible, descriptive flips: raw rank in the mis-scored-but-real
    # band, rescued into the top-5, caption long enough to read as a fair
    # query. Prefer corrected rank 1, then the longest captions.
    pool = rep.get("moderate_flips")
    if not pool:
        print("  WARNING: no moderate_flips in JSON - rerun "
              "qualitative_flips.py (updated) for band selection; "
              "falling back to extreme flips")
        pool = rep["flips"]
    # stale-panel guard: remove this cell's old flip/regression images so a
    # zero-candidate run can never leave earlier panels masquerading as new
    for old in glob.glob(str(OUT / f"{ds}_{tag}_flip*.png")) + \
            glob.glob(str(OUT / f"{ds}_{tag}_regression.png")):
        os.remove(old)
    in_band = [r for r in pool
               if args.min_raw <= r["raw_rank"] <= args.max_raw]
    cand, used_chars = [], args.min_caption_chars
    for chars in (args.min_caption_chars, 40, 25):
        cand = [r for r in in_band
                if len(" ".join(str(texts[r["query_idx"]]).split())) >= chars]
        used_chars = chars
        if len(cand) >= args.flips:
            break
    cand.sort(key=lambda r: (r["cor_rank"] != 1,
                             -len(str(texts[r["query_idx"]]))))
    print(f"  {len(cand)} candidate flips in band "
          f"[{args.min_raw}, {args.max_raw}] "
          f"(caption >= {used_chars} chars)")
    for k, rec in enumerate(cand[:args.flips], 1):
        flip_panel(rec, texts, paths,
                   f"{DSNAME[ds]} · true match rises from rank "
                   f"{rec['raw_rank']} to {rec['cor_rank']}",
                   f"{ds}_{tag}_flip{k}.png")
    if rep.get("regressions"):
        rec = rep["regressions"][0]
        flip_panel(rec, texts, paths,
                   f"{DSNAME[ds]} · honesty panel: a case the maps make "
                   f"worse (rank {rec['raw_rank']} to {rec['cor_rank']})",
                   f"{ds}_{tag}_regression.png")
    hubs_panel(rep, paths, ds, tag)
    axes_panel(rep, paths, ds, tag, n_axes=args.axes)
    print(f"[panels] done -> {OUT}")


if __name__ == "__main__":
    main()
