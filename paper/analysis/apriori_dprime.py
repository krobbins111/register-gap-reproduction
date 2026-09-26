"""A-priori d' on the TRAIN split, two definitions.

--definition legacy   replicates analysis/compute_dprime.py exactly (kept only
    to reproduce the original analysis; the paper no longer uses it):
    cap the train split to a fixed 2000-pair subsample (seed-0 permutation;
    reproduces the legacy dprime_analysis.csv to 4 decimals), then
    d' = (mean matched sim - mean off-diagonal sim) / std(off-diagonal).
    Written to results/dprime_apriori_extra.csv (merged with the legacy CSV
    by the figure/table scripts; only cells missing there are needed).

--definition perquery  (default) the statistic as written in the paper (Eq. 1):
    for every training pair i, (s(q_i,g_i) - mu_i) / sigma_i with mu_i,
    sigma_i the mean/std of s(q_i,g_j) over j != i, averaged over i; computed
    on the FULL train split, chunked (60k x 60k fits). Written to
    results/dprime_apriori_perquery.csv, one row per cell in master_arms.csv.
    This is what the figure/table scripts read by default.

Usage:
  python analysis\apriori_dprime.py --cells nwpu:remoteclip-vitl14 semart:default
  python analysis\apriori_dprime.py --definition perquery --all
"""
from __future__ import annotations

import argparse
import csv
import glob
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

SUBSAMPLE = 2000          # the value used when dprime_analysis.csv was written
SEED = 0


def load_train(ds: str, mtag: str):
    tag = "" if mtag == "default" else f"_{mtag}"
    exact = CACHE_ROOT / ds / f"train_embs{tag}.npz"
    if not exact.exists():
        cands = sorted(glob.glob(str(CACHE_ROOT / ds / f"train_embs{tag}_n*.npz")),
                       key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
        if mtag == "default":
            cands = [p for p in cands
                     if Path(p).name.split("_embs")[1].startswith("_n")]
        if not cands:
            raise FileNotFoundError(f"no train cache for {ds}/{mtag}")
        exact = Path(cands[0])
    z = np.load(exact)
    return z["caps"].astype(np.float32), z["imgs"].astype(np.float32), exact.name


def subsample(caps, imgs):
    if len(caps) > SUBSAMPLE:
        idx = np.random.default_rng(SEED).permutation(len(caps))[:SUBSAMPLE]
        return caps[idx], imgs[idx]
    return caps, imgs


def dprime_global(caps, imgs):
    S = caps @ imgs.T
    m = len(S)
    matched = np.diag(S)
    off = S[~np.eye(m, dtype=bool)]
    return float((matched.mean() - off.mean()) / (off.std() + 1e-9))


def dprime_perquery(caps, imgs, chunk=2048):
    n = len(caps)
    vals = []
    for i in range(0, n, chunk):
        s = caps[i:i + chunk] @ imgs.T                       # (c, n)
        rows = np.arange(s.shape[0])
        own = s[rows, np.arange(i, i + s.shape[0])]
        tot, tot2 = s.sum(1, dtype=np.float64), (s.astype(np.float64) ** 2).sum(1)
        m = (tot - own) / (n - 1)
        var = (tot2 - own.astype(np.float64) ** 2) / (n - 1) - m ** 2
        vals.append((own - m) / np.sqrt(np.maximum(var, 1e-12)))
    return float(np.concatenate(vals).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[],
                    help="dataset:mtag pairs (mtag 'default' for SigLIP2)")
    ap.add_argument("--all", action="store_true",
                    help="every cell listed in results/master_arms.csv")
    ap.add_argument("--definition", choices=("legacy", "perquery"), default="perquery")
    args = ap.parse_args()
    cells = list(args.cells)
    if args.all:
        for r in csv.DictReader(open(RESULTS_ROOT / "master_arms.csv")):
            cells.append(f"{r['dataset']}:{r['mtag']}")
    if not cells:
        ap.error("give --cells or --all")
    name = ("dprime_apriori_extra.csv" if args.definition == "legacy"
            else "dprime_apriori_perquery.csv")
    out = RESULTS_ROOT / name
    rows = {}
    if out.exists():
        for r in csv.DictReader(open(out)):
            rows[(r["dataset"], r["backbone"])] = r
    for cell in cells:
        ds, mt = cell.split(":")
        caps, imgs, src = load_train(ds, mt)
        n_full = len(caps)
        if args.definition == "legacy":
            caps, imgs = subsample(caps, imgs)
            dp = dprime_global(caps, imgs)
        else:
            dp = dprime_perquery(caps, imgs)
        rows[(ds, mt)] = {"dataset": ds, "backbone": mt,
                          "dprime_apriori_train": f"{dp:.4f}",
                          "n_train": str(n_full), "source": src,
                          "definition": args.definition}
        print(f"[dprime/{args.definition}] {ds}/{mt}: d'={dp:.4f} "
              f"(n={n_full}, used={len(caps)}, {src})", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "backbone",
                                          "dprime_apriori_train", "n_train",
                                          "source", "definition"])
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"[dprime] -> {out}")


if __name__ == "__main__":
    main()
