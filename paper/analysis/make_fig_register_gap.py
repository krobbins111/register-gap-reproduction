"""Front-page figures: the register gap, shown with two real catalog pairs.

Left: a COCO pair on the SAME THEME as the specialist collection, whose true
match ranks in the raw top-5 (verified from cached embeddings). Right: a
rescued specialist example. Two variants:

  --specialist rocov2   the TEXT register gap: a web caption about a medical
                        scene retrieves fine; a clinical caption that
                        describes what the image MEANS does not.
  --specialist nwpu     the VISUAL register gap: a nature landscape as people
                        photograph it retrieves fine; the same world as
                        aerial tiles depict it does not.

Runs in the repo venv (same loaders/caches as the panels).
Usage:  python analysis\\make_fig_register_gap.py --specialist nwpu
        [--spec-idx N]     override the specialist example (test index)
        [--coco-idx N]     override the web example (cache row)
        [--list-coco 12]   print candidate COCO pairs and exit (to pick)
Output: results/qual_figs/fig_register_gap[_nwpu].png/.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import re
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
from render_qualitative_panels import (resolve_split, thumb, CACHE_ROOT,  # noqa: E402
                                       RESULTS_ROOT, BLUE, ORANGE, INK, INK2,
                                       MUTED, GRID, SURF)

OUT = RESULTS_ROOT / "qual_figs"

_SENT_START = ("A", "An", "The", "There", "Many", "Some", "Several", "Two",
               "Three", "Four", "Five", "It", "In", "On", "Most", "Numerous",
               "Lots", "Green", "Yellow", "White", "Dense")


def tidy_caption(t: str) -> str:
    """Display-only cleanup for caption styles without sentence punctuation
    (NWPU): insert a period where a lowercase word is followed by a
    capitalized sentence-starter, collapse ' .' and guarantee a final period."""
    t = " ".join(str(t).split())
    pat = r"([a-z,]) (" + "|".join(_SENT_START) + r") "
    t = re.sub(pat, r"\1. \2 ", t)
    t = t.replace(" .", ".").replace(" ,", ",")
    if t and t[-1] not in ".!?":
        t += "."
    return t

SPEC = {
    "rocov2": {
        "mtag": "medsiglip",
        "qual_json": "qualitative_medsiglip.json",
        "coco_keywords": ("hospital", "doctor", "nurse", "dentist", "dental",
                          "x-ray", "xray", "medical", "surgery", "surgeon",
                          "stethoscope", "patient", "clinic", "cast ", " cast",
                          "wheelchair", "scrubs", "operating", "ambulance",
                          "exam ", "examination", "syringe", "bandage"),
        "head_web": "Web register - describes what the image looks like",
        "head_spec": "Specialist register - describes what the image means",
        "note_spec": "ROCOv2",
        "prefer": (54, 1),          # the muscle-index example, if present
        "prefer_word": None,
        "suffix": "",
    },
    "nwpu": {
        "mtag": "metaclip2-ww-huge",
        "qual_json": "qualitative_metaclip2-ww-huge.json",
        "coco_keywords": ("beach", "mountain", "field", "river", "forest",
                          "lake", "hill", "valley", "meadow", "shore",
                          "ocean", "countryside", "hillside", "grassy",
                          "trees", "island", "coast", "harbor", "bridge",
                          "farm"),
        "head_web": "Web register - a landscape as people photograph it",
        "head_spec": "Specialist register - the same world as aerial tiles depict it",
        "note_spec": "NWPU",
        "prefer": None,
        "prefer_word": "river",
        "fix_punct": True,          # NWPU captions lack sentence periods
        "suffix": "_nwpu",
    },
}


def load_embs(ds, mtag):
    """Mirror the ablation's cache resolution: exact full-split file, else the
    largest _n<limit> subset (whose ordering resolve_split reproduces)."""
    import glob as _glob
    tag = f"_{mtag}" if mtag else ""
    exact = CACHE_ROOT / ds / f"test_embs{tag}.npz"
    if not exact.exists():
        cands = sorted(_glob.glob(str(CACHE_ROOT / ds / f"test_embs{tag}_n*.npz")),
                       key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
        if mtag == "":
            cands = [p for p in cands
                     if Path(p).name.split("_embs")[1].startswith("_n")]
        exact = Path(cands[0])
    z = np.load(exact)
    return z["caps"], z["imgs"]


def ranks_and_dprime(caps, imgs):
    sims = caps @ imgs.T
    n = sims.shape[0]
    own = sims[np.arange(n), np.arange(n)]
    ranks = 1 + (sims > own[:, None]).sum(1)
    tot, tot2 = sims.sum(1), (sims ** 2).sum(1)
    m = (tot - own) / (n - 1)
    var = (tot2 - own ** 2) / (n - 1) - m ** 2
    dp = float(np.mean((own - m) / np.sqrt(np.maximum(var, 1e-12))))
    return ranks, dp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialist", default="rocov2", choices=sorted(SPEC))
    ap.add_argument("--spec-idx", "--rocov2-idx", type=int, default=None,
                    dest="spec_idx")
    ap.add_argument("--coco-idx", type=int, default=None)
    ap.add_argument("--spec-word", default=None,
                    help="pick the specialist flip whose caption contains this "
                         "word (default: 'river' for nwpu)")
    ap.add_argument("--list-coco", type=int, default=0)
    args = ap.parse_args()
    cfg = SPEC[args.specialist]
    ds, mtag = args.specialist, cfg["mtag"]

    # ── specialist half: a rescued example ──────────────────────────────────
    rep = json.load(open(RESULTS_ROOT / ds / cfg["qual_json"]))
    pool = rep.get("moderate_flips", []) + rep.get("flips", [])
    s_texts, s_paths = resolve_split(ds, mtag, rep["n_test"])
    s_caps, s_imgs = load_embs(ds, mtag)
    _, s_dp = ranks_and_dprime(s_caps, s_imgs)
    word = args.spec_word or cfg.get("prefer_word")
    if args.spec_idx is not None:
        rec = next(r for r in pool if r["query_idx"] == args.spec_idx)
    else:
        rec = None
        if cfg["prefer"]:
            rec = next((r for r in pool
                        if r["raw_rank"] == cfg["prefer"][0]
                        and r["cor_rank"] == cfg["prefer"][1]), None)
        if rec is None and word:
            themed = [r for r in pool
                      if 50 <= r["raw_rank"] <= 600 and r["cor_rank"] <= 3
                      and word in str(s_texts[r["query_idx"]]).lower()]
            themed.sort(key=lambda r: (r["cor_rank"],
                                       -len(str(s_texts[r["query_idx"]]))))
            for r in themed[:6]:
                print(f"  [spec candidate {r['query_idx']}] raw {r['raw_rank']}"
                      f" -> {r['cor_rank']}: "
                      f"{str(s_texts[r['query_idx']])[:90]}")
            rec = themed[0] if themed else None
            if rec is None:
                print(f"[fig1] NOTE: no '{word}' flip in band; using generic")
        if rec is None:
            rec = next(r for r in pool
                       if 50 <= r["raw_rank"] <= 300 and r["cor_rank"] == 1)
    qi = rec["query_idx"]

    # ── web half: an on-theme COCO pair whose true match ranks top-5 raw ────
    # The COCO loader can yield a slightly different count today than when the
    # cache was embedded, which shifts indices. Align by CONTENT: embed every
    # current caption once and match each cached row within a small window.
    from splits import load_split_pairs
    from backbones import make_embedder
    c_caps, c_imgs = load_embs("coco", "")
    c_ranks, c_dp = ranks_and_dprime(c_caps, c_imgs)
    c_texts, c_paths = load_split_pairs("coco", "test", limit=None, seed=0)
    emb = make_embedder("google/siglip2-so400m-patch16-384")
    fresh = emb.embed_texts([str(t) for t in c_texts])
    drift = len(c_texts) - len(c_caps)
    print(f"[fig1] coco: cache {len(c_caps)} rows, loader {len(c_texts)} "
          f"pairs (drift {drift:+d}); aligning by caption content")

    def match(r):  # cache row -> current pair index, or None
        lo, hi = max(0, r - 3), min(len(c_texts), r + 4 + abs(drift))
        cos = fresh[lo:hi] @ c_caps[r]
        j = int(np.argmax(cos)) + lo
        return j if cos.max() > 0.995 else None

    KEY = cfg["coco_keywords"]
    def on_theme(t):
        t = " " + str(t).lower() + " "
        return any(k in t for k in KEY)
    good, fallback = [], []
    for r in range(len(c_caps)):
        if c_ranks[r] > 5:
            continue
        j = match(r)
        if j is None:
            continue
        t = str(c_texts[j])
        if on_theme(t) and 30 <= len(t) <= 120:
            good.append((r, j))
        elif c_ranks[r] == 1 and 55 <= len(t) <= 95:
            fallback.append((r, j))
    good.sort(key=lambda rj: (c_ranks[rj[0]], -len(str(c_texts[rj[1]]))))
    if args.list_coco:
        pool_l = good if good else fallback
        for r, j in pool_l[:args.list_coco]:
            print(f"  [{r}] (rank {c_ranks[r]}) {c_texts[j]}")
        return
    if not good:
        print("[fig1] NOTE: no on-theme COCO caption ranked top-5; "
              "using a generic rank-1 pair")
        good = fallback
    if args.coco_idx is not None:
        r_pick = args.coco_idx
        ci = match(r_pick)
        if ci is None:
            raise RuntimeError(f"cache row {r_pick} has no caption match")
    else:
        r_pick, ci = good[0]
    # final proof: the matched image re-embeds onto this cache row
    img_cos = float(np.dot(emb.embed_images([c_paths[ci]], batch_size=1)[0],
                           c_imgs[r_pick]))
    print(f"[fig1] coco cache[{r_pick}] -> pair[{ci}] image cos {img_cos:.4f}")
    if img_cos < 0.98:
        raise RuntimeError("chosen COCO pair failed image verification")
    print(f"[fig1] coco '{str(c_texts[ci])[:60]}...' rank={c_ranks[r_pick]}")
    print(f"[fig1] {ds}[{qi}] raw {rec['raw_rank']} -> {rec['cor_rank']}")

    # ── draw: two compact vertical cards (text above/below, not beside) ─────
    fig = plt.figure(figsize=(8.6, 6.9))
    gs = fig.add_gridspec(1, 2, left=0.045, right=0.955, top=0.855,
                          bottom=0.03, wspace=0.14)

    def half(sub, img_path, caption, head, verdict, vcolor, note):
        g = sub.subgridspec(2, 1, height_ratios=[1.0, 0.52], hspace=0.05)
        axi = fig.add_subplot(g[0])
        axi.imshow(thumb(img_path, 460))
        for sp in axi.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(1.2)
        axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(textwrap.fill(head, 44), fontsize=10.5, color=INK2,
                      fontweight="bold", pad=8, loc="left")
        axt = fig.add_subplot(g[1]); axt.axis("off")
        cap = textwrap.fill(" ".join(str(caption).split()), 56, max_lines=6,
                            placeholder=" ...")
        axt.text(0, 0.97, f"\u201c{cap}\u201d", fontsize=9.8, color=INK,
                 va="top", style="italic", linespacing=1.4)
        axt.text(0, 0.24, textwrap.fill(verdict, 48), fontsize=10.5,
                 color=vcolor, va="top", fontweight="bold")
        axt.text(0, 0.06, note, fontsize=8.5, color=MUTED, va="bottom")

    half(gs[0], c_paths[ci], c_texts[ci],
         cfg["head_web"],
         f"retrieved: rank #{c_ranks[r_pick]}",
         INK2, f"COCO \u00b7 d' = {c_dp:.1f}")
    spec_caption = (tidy_caption(s_texts[qi]) if cfg.get("fix_punct")
                    else s_texts[qi])
    half(gs[1], s_paths[qi], spec_caption,
         cfg["head_spec"],
         f"retrieved: rank #{rec['raw_rank']} \u2192 #{rec['cor_rank']} "
         "with the learned maps",
         BLUE, f"{cfg['note_spec']} \u00b7 d' = {s_dp:.1f}")

    fig.text(0.045, 0.945, "The register gap, in action",
             fontsize=15, color=INK, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"fig_register_gap{cfg['suffix']}"
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] -> {OUT / (name + '.png')}")
    # the chosen pair, in the twin-JSON layout make_fig_register_pairs.py reads
    # (a = the web/COCO pair, b = the specialist pair)
    rec_out = dict(A="coco", B=ds, a=int(r_pick), b=int(qi),
                   a_cap=str(c_texts[ci]), a_path=str(c_paths[ci]), a_rank=int(c_ranks[r_pick]),
                   b_cap=str(s_texts[qi]), b_path=str(s_paths[qi]),
                   b_rank=int(rec["raw_rank"]), b_rank_after=int(rec["cor_rank"]),
                   bcap_bimg=float(np.dot(s_caps[qi], s_imgs[qi])),
                   acap_aimg=float(np.dot(c_caps[r_pick], c_imgs[r_pick])))
    if c_imgs.shape[1] == s_imgs.shape[1]:      # cross terms only make sense under one encoder
        rec_out.update(img_sim=float(np.dot(c_imgs[r_pick], s_imgs[qi])),
                       cap_sim=float(np.dot(c_caps[r_pick], s_caps[qi])),
                       acap_bimg=float(np.dot(c_caps[r_pick], s_imgs[qi])),
                       bcap_aimg=float(np.dot(s_caps[qi], c_imgs[r_pick])))
    else:
        rec_out.update(img_sim=None, cap_sim=None, acap_bimg=None, bcap_aimg=None,
                       note=f"specialist side uses {mtag or 'default'}; COCO side SigLIP2")
    pick_path = OUT / f"register_pick{cfg['suffix']}.json"
    json.dump(rec_out, open(pick_path, "w", encoding="utf-8"), indent=1)
    print(f"[fig1] pair -> {pick_path}")


if __name__ == "__main__":
    main()
