"""Train and SAVE the learned maps for every grid cell (for hand-off).

The grid runs never persisted the two-map weights (only the 14 tied maps of
factorize_shared.py). This retrains each arm with the exact grid recipe
(AdamW 1e-4 / wd 1e-4, batch 256, 30 epochs, tau 0.05, val-R@50 selection,
seed 0) and writes

    results/maps/<ds>/maps_<mtag>.npz
        W_text, b_text, W_image, b_image      two-map arm     phi(x) = normalize(x + W x + b)
        W_tied, b_tied                        tied-map arm    same map on both sides
        test_R10_raw, test_R50_raw, test_R10_both, test_R50_both, test_R10_tied, test_R50_tied

so the file is self-checking: the test numbers should match master_arms.csv
seed-0 values to a few thousandths (the grid averaged two seeds).

Usage:  python ablations\export_maps.py --all
        python ablations\export_maps.py --cells nwpu:default rocov2:medsiglip
"""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(HERE.parent))
from config import RESULTS_ROOT  # noqa: E402
from symmetric_projection import load_split, train_arm, recall_table, Identity  # noqa: E402


def weights(m):
    if isinstance(m, Identity):
        return None, None
    return (m.net.weight.detach().cpu().numpy().astype(np.float32),
            m.net.bias.detach().cpu().numpy().astype(np.float32))


def apply(m, x, dev):
    with torch.no_grad():
        return m(torch.tensor(x, device=dev)).cpu().numpy()


def export(ds: str, mtag: str, seed: int, dev: str, force: bool) -> None:
    tag = mtag or "default"
    out_dir = RESULTS_ROOT / "maps" / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"maps_{tag}.npz"
    if out.exists() and not force:
        print(f"[export] {ds}/{tag}: exists, skipping", flush=True)
        return
    tr_c, tr_i = load_split(ds, "train", mtag)
    va_c, va_i = load_split(ds, "val", mtag)
    te_c, te_i = load_split(ds, "test", mtag)
    target = np.arange(len(te_i))
    raw = recall_table(te_c, te_i, target)
    res = {"test_R10_raw": raw["R@10"], "test_R50_raw": raw["R@50"]}
    arrays = {}
    for arm in ("both", "shared"):
        f, g = train_arm(arm, tr_c, tr_i, va_c, va_i, epochs=30, lr=1e-4, batch=256,
                         temp=0.05, sym_loss=False, seed=seed, dev=dev)
        t = recall_table(apply(f, te_c, dev), apply(g, te_i, dev), target)
        key = "both" if arm == "both" else "tied"
        res[f"test_R10_{key}"], res[f"test_R50_{key}"] = t["R@10"], t["R@50"]
        if arm == "both":
            arrays["W_text"], arrays["b_text"] = weights(f)
            arrays["W_image"], arrays["b_image"] = weights(g)
        else:
            arrays["W_tied"], arrays["b_tied"] = weights(f)
    np.savez_compressed(out, **arrays, **{k: np.float32(v) for k, v in res.items()},
                        meta=json.dumps({"dataset": ds, "mtag": tag, "seed": seed,
                                         "recipe": "AdamW lr1e-4 wd1e-4 batch256 30ep tau0.05 "
                                                   "val-R@50 selection; phi(x)=normalize(x+Wx+b)",
                                         "n_train": int(len(tr_c)), "n_val": int(len(va_c)),
                                         "n_test": int(len(te_c))}))
    print(f"[export] {ds}/{tag}: raw R@50 {res['test_R50_raw']:.3f} -> both {res['test_R50_both']:.3f} "
          f"/ tied {res['test_R50_tied']:.3f} -> {out.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[], help="dataset:mtag (mtag 'default' for SigLIP2)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cells = [tuple(c.split(":")) for c in args.cells]
    if args.all:
        for r in csv.DictReader(open(RESULTS_ROOT / "master_arms.csv")):
            if r.get("both_R50"):
                cells.append((r["dataset"], r["mtag"]))
    for ds, mt in cells:
        export(ds, "" if mt == "default" else mt, args.seed, dev, args.force)


if __name__ == "__main__":
    main()
