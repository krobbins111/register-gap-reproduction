"""Train-size dose-response of the SHARED map: does more training raise d',
and does recall follow, across datasets?

Generalizes the old NWPU-only dose-response (r=0.99) to multiple collections.
For each train size n in a log sweep: subsample n pairs (seeded, sorted-index,
mirroring query_projection's loader), train the shared map with the standard
recipe, evaluate test d' and recall under the shared protocol (map applied to
both queries and gallery). Mean over seeds. n=0 row = raw anchor.

Usage:  python ablations/dose_response.py --dataset nwpu [--mtag ...]
Output: out/<ds>/dose_response_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import out_path                                    # noqa: E402
from symmetric_projection import (load_split, train_arm, recall_table,
                                  dprime, Identity)            # noqa: E402

SIZES = [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 50000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--arm", choices=("shared", "both"), default="both",
                    help="map form swept: 'both' (two maps, the paper's system; "
                         "writes dose_response_both_<tag>.json) or 'shared' "
                         "(tied map; writes dose_response_<tag>.json)")
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    tag = mtag or "default"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    target = np.arange(len(te_imgs))
    n_full = len(tr_caps)
    sizes = [s for s in SIZES if s < n_full] + [n_full]

    raw = recall_table(te_caps, te_imgs, target)
    raw_dp = dprime(te_caps, te_imgs, target)
    points = [{"n_train": 0, "R@50": raw["R@50"], "R@10": raw["R@10"],
               "dprime": round(raw_dp, 4), "seeds": 0}]
    print(f"[dose] {ds}/{tag} full={n_full} sizes={sizes}  "
          f"raw R@50 {raw['R@50']:.3f} d' {raw_dp:.3f}")

    for n in sizes:
        per = []
        for seed in args.seeds:
            if n < n_full:
                idx = np.sort(np.random.default_rng(seed)
                              .permutation(n_full)[:n])
                caps_n, imgs_n = tr_caps[idx], tr_imgs[idx]
            else:
                caps_n, imgs_n = tr_caps, tr_imgs
            f, g = train_arm(args.arm, caps_n, imgs_n, va_caps, va_imgs,
                             epochs=args.epochs, lr=1e-4, batch=256,
                             temp=0.05, sym_loss=False, seed=seed, dev=dev)
            with torch.no_grad():
                fq = f(torch.tensor(te_caps, device=dev)).cpu().numpy()
                gi = g(torch.tensor(te_imgs, device=dev)).cpu().numpy()
            t = recall_table(fq, gi, target)
            t["dprime"] = dprime(fq, gi, target)
            per.append(t)
        pt = {"n_train": int(n), "seeds": len(args.seeds)}
        for k in ("R@50", "R@10", "dprime"):
            vals = [p[k] for p in per]
            pt[k] = round(float(np.mean(vals)), 4)
            pt[k + "_std"] = round(float(np.std(vals)), 4)
        points.append(pt)
        print(f"[dose] n={n:6d}  d' {pt['dprime']:.3f}  R@50 {pt['R@50']:.3f}")

    dps = np.array([p["dprime"] for p in points])
    r50 = np.array([p["R@50"] for p in points])
    r_dose = float(np.corrcoef(dps, r50)[0, 1])
    rep = {"dataset": ds, "mtag": tag, "epochs": args.epochs,
           "seeds": args.seeds, "n_full": int(n_full),
           "r_dprime_vs_R50_across_doses": round(r_dose, 4),
           "points": points,
           "arm": args.arm,
           "note": f"{args.arm} map(s), standard recipe; test d'/recall; "
                   "n=0 row is the raw anchor"}
    stem = "dose_response" if args.arm == "shared" else "dose_response_both"
    p = out_path(f"{stem}_{tag}.json", ds)
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[dose] r(d', R@50) across doses = {r_dose:.3f} -> {p}")


if __name__ == "__main__":
    main()
