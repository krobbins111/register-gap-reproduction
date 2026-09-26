"""Candidate search for the two register-gap illustration figures.

Figure A (image side, ROCOv2): two scans that the frozen encoder sees as
near-identical (high image-image cosine) but whose captions describe
different findings (low caption-caption cosine). Under the raw encoder at
least one caption prefers the OTHER scan; after the learned maps each caption
prefers its own. The contact sheet prints every cosine before and after.

Figure B (text side, GoodNews vs COCO): a GoodNews press photo with a
near-twin in the COCO test set. The COCO caption retrieves its photo at rank
1; the GoodNews caption, describing who/where/why rather than what is in
frame, ranks its photo far down until the maps rescue it.

Both searches run on the cached SigLIP2 test embeddings and the exported
two-map weights (results/maps/<ds>/maps_default.npz, from run_transfer.bat).
Figure C (text side, catalogue register): FACAD product prose retrieves its
catalogue photo at rank 1; the Fashion200k attribute string of a look-alike
photo ranks it well down until the maps. Same image register, two text
registers.

Outputs, in results/qual_figs/:
    register_cands_rocov2.json / .png     (contact sheets, one row per pair)
    register_cands_goodnews.json / .png
    register_cands_fashion.json / .png
    register_cands_nwpu.json / .png       (twins by CAPTION: depiction register, SigLIP2)
Pick rows from the sheets; the final renderer takes the indices.

Usage (from iclr2027\):  python analysis\find_register_examples.py [--top 24] [--only rocov2|goodnews|fashion|nwpu]
"""
from __future__ import annotations

import argparse
import json
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

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from render_qualitative_panels import resolve_split, thumb, CACHE_ROOT, RESULTS_ROOT, INK, INK2, MUTED  # noqa: E402

OUT = RESULTS_ROOT / "qual_figs"; OUT.mkdir(parents=True, exist_ok=True)
MAPS = RESULTS_ROOT / "maps"


def nrm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def apply(x, W, b):
    return nrm(x + x @ W.T + b)


def ranks(q, gal, chunk=1024):
    out = np.zeros(len(q), int)
    for i in range(0, len(q), chunk):
        s = q[i:i + chunk] @ gal.T
        own = s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])]
        out[i:i + chunk] = 1 + (s > own[:, None]).sum(1)
    return out


def load(ds):
    z = np.load(CACHE_ROOT / ds / "test_embs.npz")
    c, g = nrm(z["caps"].astype(np.float32)), nrm(z["imgs"].astype(np.float32))
    m = np.load(MAPS / ds / "maps_default.npz")
    return c, g, apply(c, m["W_text"], m["b_text"]), apply(g, m["W_image"], m["b_image"])


def f(x):
    return f"{float(x):.3f}"


# ── Figure A: look-alike scans, different findings ──────────────────────────
def search_rocov2(top):
    c, g, c2, g2 = load("rocov2")
    n = len(c); r0, r1 = ranks(c, g), ranks(c2, g2)
    gg = g @ g.T; np.fill_diagonal(gg, -1)
    cc = c @ c.T
    nbrs = np.argsort(-gg, axis=1)[:, :3]
    cands = []
    for i in range(n):
        for j in nbrs[i]:
            if j < i:
                continue
            simg, scap = gg[i, j], cc[i, j]
            raw_i = (c[i] @ g[i], c[i] @ g[j]); raw_j = (c[j] @ g[j], c[j] @ g[i])
            cor_i = (c2[i] @ g2[i], c2[i] @ g2[j]); cor_j = (c2[j] @ g2[j], c2[j] @ g2[i])
            confused = raw_i[1] >= raw_i[0] or raw_j[1] >= raw_j[0]
            fixed = cor_i[0] > cor_i[1] + 0.03 and cor_j[0] > cor_j[1] + 0.03
            if simg > 0.88 and scap < 0.6 and confused and fixed and r1[i] <= 20 and r1[j] <= 20 \
                    and max(r0[i], r0[j]) >= 20:
                def rk(cq, G, t):        # rank of gallery item t in caption cq's list
                    sc = G @ cq; return int(1 + (sc > sc[t]).sum())
                cands.append(dict(i=int(i), j=int(j), img_sim=float(simg), cap_sim=float(scap),
                                  rank_ci_gj=rk(c[i], g, j), rank_cj_gi=rk(c[j], g, i),
                                  rank_ci_gj_after=rk(c2[i], g2, j), rank_cj_gi_after=rk(c2[j], g2, i),
                                  img_sim_after=float(g2[i] @ g2[j]), cap_sim_after=float(c2[i] @ c2[j]),
                                  rank_i=int(r0[i]), rank_j=int(r0[j]), rank_i_after=int(r1[i]), rank_j_after=int(r1[j]),
                                  ci_gi=float(raw_i[0]), ci_gj=float(raw_i[1]), cj_gj=float(raw_j[0]), cj_gi=float(raw_j[1]),
                                  ci_gi_after=float(cor_i[0]), ci_gj_after=float(cor_i[1]),
                                  cj_gj_after=float(cor_j[0]), cj_gi_after=float(cor_j[1])))
    cands.sort(key=lambda d: -((d["img_sim"] - d["cap_sim"]) - 0.01 * max(d["rank_i_after"], d["rank_j_after"])))
    cands = cands[:top]
    texts, paths = resolve_or_align("rocov2", c, g, sorted({d["i"] for d in cands} | {d["j"] for d in cands}))
    cands = [d for d in cands if d["i"] in texts and d["j"] in texts]
    for d in cands:
        d["cap_i"], d["cap_j"] = texts[d["i"]], texts[d["j"]]
        d["path_i"], d["path_j"] = str(paths[d["i"]]), str(paths[d["j"]])
    json.dump(cands, open(OUT / "register_cands_rocov2.json", "w", encoding="utf-8"), indent=1)
    sheet_rocov2(cands)
    print(f"[find] rocov2: {len(cands)} candidate pairs -> {OUT / 'register_cands_rocov2.png'}")


def sheet_rocov2(cands):
    rows = len(cands)
    fig = plt.figure(figsize=(14, 2.3 * rows))
    gs = fig.add_gridspec(rows, 3, width_ratios=[1, 1, 3.2], hspace=0.35, wspace=0.05)
    for r, d in enumerate(cands):
        for col, key in ((0, "path_i"), (1, "path_j")):
            ax = fig.add_subplot(gs[r, col]); ax.imshow(thumb(d[key], 260)); ax.axis("off")
            idx = d["i"] if col == 0 else d["j"]
            rk = (d["rank_i"], d["rank_i_after"]) if col == 0 else (d["rank_j"], d["rank_j_after"])
            ax.set_title(f"#{r}  test idx {idx}   rank {rk[0]} → {rk[1]}", fontsize=8, color=INK2, loc="left")
        ax = fig.add_subplot(gs[r, 2]); ax.axis("off")
        txt = (f"img·img {f(d['img_sim'])} → {f(d['img_sim_after'])}    cap·cap {f(d['cap_sim'])} → {f(d['cap_sim_after'])}\n"
               f"c_i: own {f(d['ci_gi'])} / other {f(d['ci_gj'])}  →  {f(d['ci_gi_after'])} / {f(d['ci_gj_after'])}      "
               f"c_j: own {f(d['cj_gj'])} / other {f(d['cj_gi'])}  →  {f(d['cj_gj_after'])} / {f(d['cj_gi_after'])}\n"
               f"i: {textwrap.shorten(d['cap_i'], 330)}\n"
               f"j: {textwrap.shorten(d['cap_j'], 330)}")
        ax.text(0, 1, textwrap.fill(txt, 120, replace_whitespace=False), fontsize=7.6, va="top", ha="left",
                color=INK, family="DejaVu Sans", transform=ax.transAxes, wrap=True)
    fig.savefig(OUT / "register_cands_rocov2.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def resolve_or_align(ds, c_caps, c_imgs, rows):
    """(texts, paths) for the requested cache rows of <ds>'s test split.
    Normally resolve_split gives the whole split in cache order. When the
    loader yields a different pair count today than when the SigLIP2 cache was
    embedded (COCO: 4999 cached rows vs 5000 pairs), indices shift, so fall
    back to CONTENT alignment as make_fig_register_gap.py does: re-embed the
    current captions, match each needed cache row inside a small window
    (caption cos > 0.995), then prove the matched image re-embeds onto the
    cached image row (cos > 0.98). Rows that fail are dropped with a note."""
    try:
        texts, paths = resolve_split(ds, "", len(c_caps))
        return {r: str(texts[r]) for r in rows}, {r: paths[r] for r in rows}
    except RuntimeError as e:
        print(f"[find] {ds}: {str(e).splitlines()[0]} -> aligning by content", flush=True)
    from splits import load_split_pairs
    from backbones import make_embedder
    texts, paths = load_split_pairs(ds, "test", limit=None, seed=0)
    drift = len(texts) - len(c_caps)
    emb = make_embedder("google/siglip2-so400m-patch16-384")
    fresh = emb.embed_texts([str(t) for t in texts])
    out_t, out_p = {}, {}
    for r in rows:
        lo, hi = max(0, r - 3 - max(0, -drift)), min(len(texts), r + 4 + max(0, drift))
        cos = fresh[lo:hi] @ c_caps[r]
        j = int(np.argmax(cos)) + lo
        if cos.max() < 0.995:
            print(f"[find]   {ds} row {r}: no caption match (best {cos.max():.3f}), dropped"); continue
        img_cos = float(np.dot(emb.embed_images([paths[j]], batch_size=1)[0], c_imgs[r]))
        if img_cos < 0.98:
            print(f"[find]   {ds} row {r} -> pair {j}: image cos {img_cos:.3f} < 0.98, dropped"); continue
        out_t[r], out_p[r] = str(texts[j]), paths[j]
        print(f"[find]   {ds} row {r} -> pair {j}: caption cos {cos.max():.4f}, image cos {img_cos:.4f}")
    return out_t, out_p


# ── Figure B: a press photo with a COCO twin ─────────────────────────────────
def search_twins(A, B, top, name, b_rank=(15, 300), b_after=5, a_rank=2, min_sim=0.62, nbrs=5, by="img"):
    """A pair in B with a near-twin in A: twin by IMAGE (by="img": same kind of
    picture, different caption register) or by CAPTION (by="cap": same kind of
    sentence, different image register). A's caption retrieves its photo at
    rank <= a_rank; B's caption ranks its photo in b_rank until the maps bring
    it to <= b_after. Prints the four cross cosines so the panel can show that
    A's caption scores B's photo about as well as B's own does."""
    bc, bi, bc2, bi2 = load(B)
    za = np.load(CACHE_ROOT / A / "test_embs.npz")
    ac, ai = nrm(za["caps"].astype(np.float32)), nrm(za["imgs"].astype(np.float32))
    rb0, rb1, ra0 = ranks(bc, bi), ranks(bc2, bi2), ranks(ac, ai)
    S = (bi @ ai.T) if by == "img" else (bc @ ac.T)
    cands = []
    for i in range(len(bc)):
        if not (b_rank[0] <= rb0[i] <= b_rank[1] and rb1[i] <= b_after):
            continue
        for j in np.argsort(-S[i])[:nbrs]:
            if ra0[j] <= a_rank and S[i, j] > min_sim:
                cands.append(dict(b=int(i), a=int(j), img_sim=float(bi[i] @ ai[j]), twin_by=by,
                                  b_rank=int(rb0[i]), b_rank_after=int(rb1[i]), a_rank=int(ra0[j]),
                                  bcap_bimg=float(bc[i] @ bi[i]), acap_bimg=float(ac[j] @ bi[i]),
                                  acap_aimg=float(ac[j] @ ai[j]), bcap_aimg=float(bc[i] @ ai[j]),
                                  cap_sim=float(bc[i] @ ac[j]), bcap_bimg_after=float(bc2[i] @ bi2[i])))
                break
    cands.sort(key=lambda d: -d["img_sim" if by == "img" else "cap_sim"])
    cands = cands[:top]
    btexts, bpaths = resolve_or_align(B, bc, bi, sorted({d["b"] for d in cands}))
    atexts, apaths = resolve_or_align(A, ac, ai, sorted({d["a"] for d in cands}))
    cands = [d for d in cands if d["b"] in btexts and d["a"] in atexts]
    for d in cands:
        d["b_cap"], d["b_path"] = btexts[d["b"]], str(bpaths[d["b"]])
        d["a_cap"], d["a_path"] = atexts[d["a"]], str(apaths[d["a"]])
        d["A"], d["B"] = A, B
    json.dump(cands, open(OUT / f"register_cands_{name}.json", "w", encoding="utf-8"), indent=1)
    sheet_twins(cands, A, B, name)
    print(f"[find] {name}: {len(cands)} candidate pairs -> {OUT / f'register_cands_{name}.png'}")


def sheet_twins(cands, A, B, name):
    rows = max(len(cands), 1)
    fig = plt.figure(figsize=(14, 2.3 * rows))
    gs = fig.add_gridspec(rows, 3, width_ratios=[1, 1, 3.2], hspace=0.35, wspace=0.05)
    for r, d in enumerate(cands):
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(thumb(d["b_path"], 260)); ax.axis("off")
        ax.set_title(f"#{r}  {B} {d['b']}   rank {d['b_rank']} → {d['b_rank_after']}", fontsize=8, color=INK2, loc="left")
        ax = fig.add_subplot(gs[r, 1]); ax.imshow(thumb(d["a_path"], 260)); ax.axis("off")
        ax.set_title(f"{A} {d['a']}   rank {d['a_rank']}", fontsize=8, color=INK2, loc="left")
        ax = fig.add_subplot(gs[r, 2]); ax.axis("off")
        txt = (f"img·img {f(d['img_sim'])}   cap·cap {f(d['cap_sim'])}   "
               f"s({B} cap, {B} img) {f(d['bcap_bimg'])} → {f(d['bcap_bimg_after'])}   s({A} cap, {B} img) {f(d['acap_bimg'])}   "
               f"s({A} cap, {A} img) {f(d['acap_aimg'])}   s({B} cap, {A} img) {f(d['bcap_aimg'])}\n"
               f"{B}: {textwrap.shorten(d['b_cap'], 330)}\n"
               f"{A}: {textwrap.shorten(d['a_cap'], 200)}")
        ax.text(0, 1, textwrap.fill(txt, 120, replace_whitespace=False), fontsize=7.6, va="top", ha="left",
                color=INK, family="DejaVu Sans", transform=ax.transAxes)
    fig.savefig(OUT / f"register_cands_{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--only", choices=["rocov2", "goodnews", "fashion", "nwpu"], default=None)
    a = ap.parse_args()
    if a.only in (None, "rocov2"):
        search_rocov2(a.top)
    if a.only in (None, "goodnews"):
        search_twins("coco", "goodnews", a.top, "goodnews")
    if a.only in (None, "fashion"):
        # FACAD product prose retrieves its catalogue photo; the Fashion200k
        # attribute string for a look-alike photo does not, until the maps.
        search_twins("facad", "fashion200k", a.top, "fashion", b_rank=(5, 300), b_after=3,
                     a_rank=1, min_sim=0.78, nbrs=3)
    if a.only in (None, "nwpu"):
        # depiction register: a COCO caption and an NWPU caption that say the
        # same kind of thing; the photograph is retrieved, the aerial tile is not.
        search_twins("coco", "nwpu", a.top, "nwpu", b_rank=(30, 300), b_after=3,
                     a_rank=1, min_sim=0.5, nbrs=5, by="cap")


if __name__ == "__main__":
    main()
