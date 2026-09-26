"""Mine the raw material for qualitative figures: the queries the fix rescues,
the hub images it dethrones, and the axes it amplifies.

Per cell (cached embeddings; both-arm, standard recipe, seed 0), writes
results/<ds>/qualitative_<tag>.json with test-split INDICES (resolve to
captions/image paths via the embedding pipeline's split ordering):

  flips       top-N queries by rank improvement: true item's raw rank vs
              corrected rank, plus raw top-5 and corrected top-5 gallery
              indices and scores  -> before/after retrieval panels
  regressions same, worst direction (honesty panel; usually near-empty)
  hubs        gallery items ranked in the raw top-10 for the most queries,
              with their corrected counts  -> "the images that answer every
              query" panel; includes top-10 occupancy concentration (share
              of all top-10 slots held by the 5 biggest hubs, raw vs fixed)
  axes        for the top-K amplified axes of the image-side stretch S_i:
              the 8 gallery indices with the largest positive and negative
              coordinate on each axis  -> "what the learned axes encode"

Usage:  python analysis/qualitative_flips.py --dataset nwpu [--mtag ...]
        (add --topn / --axes-k to taste)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "ablations"))
from config import out_path  # noqa: E402
from symmetric_projection import load_split, train_arm  # noqa: E402


def apply(module, x, dev):
    with torch.no_grad():
        return module(torch.tensor(x, device=dev)).cpu().numpy()


def ranks_of_true(sims):
    n = sims.shape[0]
    own = sims[np.arange(n), np.arange(n)]
    return 1 + (sims > own[:, None]).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topn", type=int, default=40)
    ap.add_argument("--axes-k", type=int, default=6)
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    f, g = train_arm("both", tr_caps, tr_imgs, va_caps, va_imgs,
                     epochs=30, lr=1e-4, batch=256, temp=0.05,
                     sym_loss=False, seed=args.seed, dev=dev)
    q_cor, g_cor = apply(f, te_caps, dev), apply(g, te_imgs, dev)

    sims_raw = te_caps @ te_imgs.T
    sims_cor = q_cor @ g_cor.T
    r_raw, r_cor = ranks_of_true(sims_raw), ranks_of_true(sims_cor)

    # ── flips & regressions ─────────────────────────────────────────────────
    imp = r_raw.astype(int) - r_cor.astype(int)
    def pack(qi):
        return {
            "query_idx": int(qi),
            "raw_rank": int(r_raw[qi]), "cor_rank": int(r_cor[qi]),
            "raw_top5": [int(j) for j in np.argsort(-sims_raw[qi])[:5]],
            "cor_top5": [int(j) for j in np.argsort(-sims_cor[qi])[:5]],
            "raw_top5_scores": [round(float(s), 4) for s in
                                np.sort(sims_raw[qi])[::-1][:5]],
            "cor_top5_scores": [round(float(s), 4) for s in
                                np.sort(sims_cor[qi])[::-1][:5]],
            "raw_true_score": round(float(sims_raw[qi, qi]), 4),
            "cor_true_score": round(float(sims_cor[qi, qi]), 4)}
    # best flips that end INSIDE the corrected top-5 (panel-worthy)
    good = [int(i) for i in np.argsort(-imp) if r_cor[i] <= 5][:args.topn]
    # moderate band: mis-scored but plausibly descriptive queries rescued into
    # the corrected top-5. The renderer's default window is raw rank 50-600,
    # so fill that band FIRST (a cell with thousands of rescues must not spend
    # the cap on rank 30-49), then pad with the near-band leftovers.
    order = np.argsort(r_raw)
    band = [int(i) for i in order
            if r_cor[i] <= 5 and 50 <= r_raw[i] <= 600][:100]
    extra = [int(i) for i in order
             if r_cor[i] <= 5 and (30 <= r_raw[i] < 50 or r_raw[i] > 600)][:40]
    mod = band + extra
    bad = [int(i) for i in np.argsort(imp) if imp[i] < 0][:10]

    # ── hubs: who owns the raw top-10, and after the fix ────────────────────
    k = 10
    top_raw = np.argsort(-sims_raw, axis=1)[:, :k]
    top_cor = np.argsort(-sims_cor, axis=1)[:, :k]
    cnt_raw = np.bincount(top_raw.ravel(), minlength=len(te_imgs))
    cnt_cor = np.bincount(top_cor.ravel(), minlength=len(te_imgs))
    hub_ids = np.argsort(-cnt_raw)[:15]
    occ = lambda c: round(float(np.sort(c)[::-1][:5].sum() / c.sum()), 4)
    hubs = {"k": k,
            "top5_occupancy_raw": occ(cnt_raw), "top5_occupancy_cor": occ(cnt_cor),
            "items": [{"gallery_idx": int(j), "raw_count": int(cnt_raw[j]),
                       "cor_count": int(cnt_cor[j])} for j in hub_ids]}

    # ── axes: what the image-side stretch amplifies ─────────────────────────
    W = f.net.weight.detach().cpu().numpy()  # noqa: F841 (text side, if wanted)
    Wg = g.net.weight.detach().cpu().numpy()
    A_i = np.eye(Wg.shape[0]) + Wg
    U, sig, Vt = np.linalg.svd(A_i.astype(np.float64))
    S_eigvec = Vt  # rows = axes of the stretch V Σ V^T
    axes = []
    for a in range(args.axes_k):
        v = S_eigvec[a]
        proj = te_imgs @ v
        axes.append({"axis": a, "sigma": round(float(sig[a]), 3),
                     "top_pos": [int(j) for j in np.argsort(-proj)[:8]],
                     "top_neg": [int(j) for j in np.argsort(proj)[:8]]})

    rep = {"dataset": ds, "mtag": mtag or "default", "seed": args.seed,
           "n_test": int(len(te_caps)),
           "note": "indices are positions in the TEST split ordering of "
                   "embedding_pipeline/query_projection.get_split_arrays",
           "flips": [pack(i) for i in good],
           "moderate_flips": [pack(i) for i in mod],
           "regressions": [pack(i) for i in bad],
           "hubs": hubs, "amplified_axes": axes}
    outp = out_path(f"qualitative_{mtag or 'default'}.json", ds)
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"[qual] {ds}/{mtag or 'default'}: {len(good)} flips "
          f"(best {int(imp[good[0]]) if good else 0} ranks), "
          f"{len(bad)} regressions; hub occupancy {hubs['top5_occupancy_raw']:.0%}"
          f" -> {hubs['top5_occupancy_cor']:.0%}; -> {outp}")


if __name__ == "__main__":
    main()
