"""Near-miss mass: a descriptive, a-priori statistic of the catalog's own
self-retrieval that predicts the trained two-map gain (candidate successor
to d' as the diagnostic).

For each catalog pair (q_i, g_i), rank g_i among all catalog images under the
frozen encoder. d' summarises the SCORE margin (mean/std over all
distractors); the near-miss mass summarises the RANK distribution:

    N_k(c) = P( k < rank_i <= c*k )            (default k=50, c=3)

i.e. the fraction of true pairs that sit just outside the cutoff, within a
bounded factor of it. A linear re-metric moves ranks by a bounded factor
(analysis/rank_transport.py measures that reach on trained maps), so the
recoverable mass at R@k is the mass in the reachable band. Deep misses
(rank in the thousands: semantically ambiguous captions) are not recoverable
and d' cannot tell them apart from near misses; on NWPU nearly all misses
are near misses (class clusters of a few hundred tiles), which is why d'
under-predicts it.

Gallery size matters for ranks, so the statistic is computed on a random
subsample of the TRAIN catalog of the same size as the test gallery (mean
over 3 seeds); the val split value is also reported. Test embeddings are
never touched.

Usage:  python analysis\apriori_nearmiss.py --all
Writes results\nearmiss_apriori.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_ROOT, RESULTS_ROOT  # noqa: E402

ROOTS = [CACHE_ROOT, RESULTS_ROOT] + [Path(p) for p in
                                      os.environ.get("ICLR_EXTRA_ROOTS", "").split(":") if p]
KS = (10, 50)
CS = (1.5, 2, 3, 4, 6, 8)
SEEDS = (0, 1, 2)


def find(ds, mtag, split):
    tag = "" if mtag == "default" else f"_{mtag}"
    for root in ROOTS:
        exact = root / ds / f"{split}_embs{tag}.npz"
        if exact.exists():
            return exact
        cands = sorted(glob.glob(str(root / ds / f"{split}_embs{tag}_n*.npz")),
                       key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
        if mtag == "default":
            cands = [p for p in cands if Path(p).name.split("_embs")[1].startswith("_n")]
        if cands:
            return Path(cands[0])
    raise FileNotFoundError(f"no {split} cache for {ds}/{mtag}")


def load(ds, mtag, split):
    z = np.load(find(ds, mtag, split))
    c, i = z["caps"].astype(np.float32), z["imgs"].astype(np.float32)
    return (c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8),
            i / np.maximum(np.linalg.norm(i, axis=1, keepdims=True), 1e-8))


def n_test_of(ds, mtag):
    for root in ROOTS:
        p = root / ds / f"symmetric_ablation_{mtag}.json"
        if p.exists():
            return int(json.load(open(p)).get("n_test") or 0)
    return 0


def ranks(caps, imgs, chunk=2048):
    n = len(caps); out = []
    for i in range(0, n, chunk):
        s = caps[i:i + chunk] @ imgs.T
        rows = np.arange(s.shape[0]); own = s[rows, np.arange(i, i + s.shape[0])]
        out.append(1 + (s > own[:, None]).sum(1))
    return np.concatenate(out)


def stats_of(rk):
    n = len(rk); m = {}
    for k in KS:
        m[f"R{k}"] = float((rk <= k).mean())
        for c in CS:
            m[f"N{k}_x{c:g}"] = float(((rk > k) & (rk <= int(c * k))).mean())
        lr = np.log(rk)
        m[f"dens_log{k}"] = float(np.mean(np.exp(-0.5 * ((lr - np.log(k)) / 0.5) ** 2)))
    m["mean_logrank"] = float(np.log10(rk).mean())
    m["median_rank_frac"] = float(np.median(rk) / n)
    return m


def cell(ds, mt):
    tc, ti = load(ds, mt, "train")
    nt = n_test_of(ds, mt)
    try:
        vc, vi = load(ds, mt, "val")
    except FileNotFoundError:
        vc = vi = None
    if not nt:
        nt = len(vc) if vc is not None else min(len(tc), 5000)
    n = min(nt, len(tc))
    acc = {}
    for s in SEEDS:
        idx = np.random.default_rng(s).permutation(len(tc))[:n]
        st = stats_of(ranks(tc[idx], ti[idx]))
        for k, v in st.items():
            acc.setdefault(k, []).append(v)
    m = {"dataset": ds, "backbone": mt, "n_train": len(tc), "n_gallery": n, "n_test": nt}
    m.update({f"train_{k}": float(np.mean(v)) for k, v in acc.items()})
    if vc is not None:
        m["n_val"] = len(vc)
        m.update({f"val_{k}": v for k, v in stats_of(ranks(vc, vi)).items()})
    m["N50"] = m["train_N50_x3"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cells = list(args.cells)
    if args.all:
        for r in csv.DictReader(open(RESULTS_ROOT / "master_arms.csv")):
            if r.get("both_gain50"):
                cells.append(f"{r['dataset']}:{r['mtag']}")
    out = RESULTS_ROOT / "nearmiss_apriori.csv"
    rows = {}
    if out.exists():
        for r in csv.DictReader(open(out)):
            rows[(r["dataset"], r["backbone"])] = r
    for c in cells:
        ds, mt = c.split(":")
        try:
            m = cell(ds, mt)
        except FileNotFoundError as e:
            print(f"[skip] {c}: {e}", flush=True); continue
        rows[(ds, mt)] = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in m.items()}
        print(f"[nearmiss] {c}: gallery {m['n_gallery']}  train R@50 {m['train_R50']:.3f}  "
              f"N50(x3) {m['N50']:.3f}  N50(x4) {m['train_N50_x4']:.3f}", flush=True)
    keys = sorted({k for r in rows.values() for k in r}, key=lambda k: (k not in ("dataset", "backbone"), k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"[nearmiss] -> {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
