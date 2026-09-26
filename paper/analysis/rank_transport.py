"""Rank transport: how far do the trained maps move a true pair?

Trains the two-map arm (seed 0, same recipe as the grid) for one cell, then
records raw and corrected rank of every test query's true image and reports

  * rescue probability by raw-rank band:  P(cor_rank <= 50 | raw_rank in band)
  * the reach: quantiles of raw_rank among rescued pairs (raw > 50, cor <= 50)
  * the same at R@10

This is the justification for the near-miss statistic in
analysis/apriori_nearmiss.py: it measures, from trained maps, the band of
raw ranks from which pairs are actually rescued. It is run on a handful of
cells for the appendix; it is NOT used to compute the statistic.

Writes results/<ds>/rank_transport_<mtag>.json and the full rank arrays as
results/<ds>/rank_transport_<mtag>.npz.

Usage: python analysis\rank_transport.py --dataset nwpu [--mtag ...]
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

BANDS = [(1, 10), (10, 50), (50, 100), (100, 150), (150, 200), (200, 300),
         (300, 500), (500, 1000), (1000, 10 ** 9)]


def apply(module, x, dev):
    with torch.no_grad():
        return module(torch.tensor(x, device=dev)).cpu().numpy()


def ranks_of_true(q, g, chunk=2048):
    n = len(q); out = []
    for i in range(0, n, chunk):
        s = q[i:i + chunk] @ g.T
        own = s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])]
        out.append(1 + (s > own[:, None]).sum(1))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_c, tr_i = load_split(ds, "train", mtag)
    va_c, va_i = load_split(ds, "val", mtag)
    te_c, te_i = load_split(ds, "test", mtag)
    f, g = train_arm("both", tr_c, tr_i, va_c, va_i, epochs=30, lr=1e-4,
                     batch=256, temp=0.05, sym_loss=False, seed=args.seed, dev=dev)
    r_raw = ranks_of_true(te_c, te_i)
    r_cor = ranks_of_true(apply(f, te_c, dev), apply(g, te_i, dev))
    n = len(r_raw)
    rep = {"dataset": ds, "mtag": mtag or "default", "seed": args.seed, "n_test": n,
           "raw_R10": float((r_raw <= 10).mean()), "raw_R50": float((r_raw <= 50).mean()),
           "cor_R10": float((r_cor <= 10).mean()), "cor_R50": float((r_cor <= 50).mean()),
           "bands": []}
    for lo, hi in BANDS:
        m = (r_raw > lo) & (r_raw <= hi) if lo > 1 else (r_raw <= hi)
        if m.sum() == 0:
            continue
        rep["bands"].append({"raw_band": [lo, hi], "n": int(m.sum()), "mass": float(m.mean()),
                             "p_cor_le50": float((r_cor[m] <= 50).mean()),
                             "p_cor_le10": float((r_cor[m] <= 10).mean()),
                             "median_cor_rank": float(np.median(r_cor[m]))})
    for k in (10, 50):
        resc = (r_raw > k) & (r_cor <= k); lost = (r_raw <= k) & (r_cor > k)
        rr = r_raw[resc]
        rep[f"rescued@{k}"] = {"n": int(resc.sum()), "share_of_test": float(resc.mean()),
                               "raw_rank_quantiles": {q: float(np.quantile(rr, q / 100)) if len(rr) else None
                                                      for q in (50, 75, 90, 95)},
                               "lost_n": int(lost.sum())}
        # ratio of raw rank to k for rescued pairs = the "reach" in multiples of k
        rep[f"reach@{k}_x"] = {q: float(np.quantile(rr / k, q / 100)) if len(rr) else None
                               for q in (50, 75, 90, 95)}
    tag = mtag or "default"
    jp = out_path(f"rank_transport_{tag}.json", ds)
    json.dump(rep, open(jp, "w"), indent=1)
    np.savez_compressed(out_path(f"rank_transport_{tag}.npz", ds), r_raw=r_raw, r_cor=r_cor)
    print(f"[transport] {ds}/{tag}: R@50 {rep['raw_R50']:.3f}->{rep['cor_R50']:.3f}")
    for b in rep["bands"]:
        print(f"   raw rank ({b['raw_band'][0]:>5},{b['raw_band'][1]:>6}]  mass {b['mass']:.3f}  "
              f"P(cor<=50) {b['p_cor_le50']:.2f}  P(cor<=10) {b['p_cor_le10']:.2f}")
    print(f"   reach@50 (raw/50 quantiles of rescued): {rep['reach@50_x']}")
    print(f"[transport] -> {jp}")


if __name__ == "__main__":
    main()
