"""Non-linear (bottleneck MLP) counterpart of the linear arms in symmetric_projection.py.

Same loader, same recipe (in-batch InfoNCE t2i, tau 0.05, AdamW 1e-4 / wd 1e-4, batch 256, 30 epochs, val-R@50
model selection, seeds 0-2), same report layout -- only the per-side map changes:

    linear (symmetric_projection):  phi(x) = normalize(x + W x + b)                      W zero-init
    mlp    (this script):           phi(x) = normalize(x + W2 GELU(W1 x + b1) + b2)      W2 zero-init, dropout 0.1
                                    (the QueryProjection 'mlp' arch of embedding_pipeline/query_projection.py, applied per side)

Arms: text (query side only), image (gallery side only), both (one MLP per side), shared (one MLP tied across sides).
Writes $LTG_RESULTS/<collection>/<encoder>/mlp_arms[_trN][_hH].json and prints the linear-vs-MLP comparison when
linear_arms.json exists for the cell.

usage (from iclr2027/):
    python -m ltg.maps.mlp --collection skincap --encoder siglip2-so400m-16-384                       # default SigLIP2, arms both shared, seeds 0 1 2
    python -m ltg.maps.mlp --collection skincap --encoder siglip2-so400m-16-384 --mtag clip-vitl14-laion2b --arms both --hidden 256
    run_mlp_arms.bat                                                     # every grid collection on the default encoder
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
from torch import nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))  # learn-the-gap/
from ltg.cache import result_path  # noqa: E402
from ltg.maps.linear import (Identity, dprime, encoder_key, load_split, near_dup_rate,  # noqa: E402
                             recall_table, side_geometry)

ARMS = ("text", "image", "both", "shared")


class ResidualMLP(nn.Module):
    """x -> normalize(x + MLP(x)); last layer zero-init so epoch 0 == raw baseline (same convention as ResidualLinear)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return nn.functional.normalize(x + self.net(x), dim=-1)


def make_arm(arm: str, dim: int, hidden: int, dropout: float, dev: str):
    mk = lambda: ResidualMLP(dim, hidden, dropout).to(dev)  # noqa: E731
    if arm == "text":
        f, g = mk(), Identity()
    elif arm == "image":
        f, g = Identity(), mk()
    elif arm == "both":
        f, g = mk(), mk()
    elif arm == "shared":
        f = mk(); g = f
    else:
        raise ValueError(arm)
    params = list(f.parameters()) + ([] if g is f else list(g.parameters()))
    return f, g, params


def apply(m, x, dev):
    if isinstance(m, Identity):
        return x
    with torch.no_grad():
        return m(torch.tensor(x, device=dev)).cpu().numpy()


def train_arm(arm, tr_caps, tr_imgs, va_caps, va_imgs, *, hidden, dropout, epochs, lr, batch, temp, seed, dev):
    """Identical loop to symmetric_projection.train_arm (t2i InfoNCE, val-R@50 selection); only the map differs."""
    torch.manual_seed(seed)
    dim = tr_caps.shape[1]
    f, g, params = make_arm(arm, dim, hidden, dropout, dev)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    Xtr, Ytr = torch.tensor(tr_caps, device=dev), torch.tensor(tr_imgs, device=dev)
    n = len(Xtr)
    best_r50, best = -1.0, None
    for ep in range(epochs):
        for m in (f, g):
            m.train()
        perm = torch.randperm(n, device=dev, generator=torch.Generator(dev).manual_seed(seed * 1000 + ep))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = f(Xtr[idx]) @ g(Ytr[idx]).T / temp
            loss = nn.functional.cross_entropy(logits, torch.arange(len(idx), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        for m in (f, g):
            m.eval()
        r50 = recall_table(apply(f, va_caps, dev), apply(g, va_imgs, dev), np.arange(len(va_imgs)))["R@50"]
        if r50 > best_r50:
            best_r50 = r50
            best = tuple(None if isinstance(m, Identity) else {k: v.detach().clone() for k, v in m.state_dict().items()}
                         for m in (f, g))
    for m, st in zip((f, g), best):
        if st is not None:
            m.load_state_dict(st)
    for m in (f, g):
        m.eval()
    return f, g, best_r50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", "--dataset", dest="dataset", required=True)
    ap.add_argument("--encoder", "--mtag", dest="mtag", default="", help="encoder key or legacy mtag")
    ap.add_argument("--arms", nargs="*", default=["both", "shared"], choices=ARMS)
    ap.add_argument("--hidden", type=int, default=1024, help="bottleneck width (query_projection.py default 1024)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--limit-train", type=int, default=None)
    args = ap.parse_args()
    ds, mtag = args.dataset, encoder_key(args.mtag)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    if args.limit_train and args.limit_train < len(tr_caps):
        idx = np.sort(np.random.default_rng(0).permutation(len(tr_caps))[:args.limit_train])
        tr_caps, tr_imgs = tr_caps[idx], tr_imgs[idx]
    target = np.arange(len(te_imgs))
    dim = tr_caps.shape[1]

    raw = recall_table(te_caps, te_imgs, target)
    raw_dp = dprime(te_caps, te_imgs, target)
    geom = side_geometry(tr_caps, tr_imgs)
    n_params = 2 * dim * args.hidden + args.hidden + dim
    print(f"[mlp] {ds} (mtag={mtag or 'default'})  train={len(tr_caps)} val={len(va_caps)} test={len(te_caps)}  dev={dev}  "
          f"d={dim} hidden={args.hidden} ({n_params:,} params/side vs {dim * dim + dim:,} linear)")
    print(f"[mlp] raw test: R@10 {raw['R@10']:.3f}  R@50 {raw['R@50']:.3f}  d' {raw_dp:.3f}")

    # the linear counterpart, for the side-by-side print
    lin_p = result_path(ds, mtag, "linear_arms.json")
    lin = json.load(open(lin_p, encoding="utf-8")).get("arms", {}) if lin_p.exists() else {}

    results = {}
    for arm in args.arms:
        per_seed, val_best = [], []
        for seed in args.seeds:
            f, g, vr = train_arm(arm, tr_caps, tr_imgs, va_caps, va_imgs, hidden=args.hidden, dropout=args.dropout,
                                 epochs=args.epochs, lr=args.lr, batch=args.batch, temp=args.temp, seed=seed, dev=dev)
            fq, gi = apply(f, te_caps, dev), apply(g, te_imgs, dev)
            t = recall_table(fq, gi, target)
            t["dprime"] = dprime(fq, gi, target)
            t["gal_near_dup@0.9_post"] = round(near_dup_rate(gi), 4) if not isinstance(g, Identity) else geom["img_near_dup@0.9"]
            per_seed.append(t); val_best.append(vr)
        agg = {}
        for k in per_seed[0]:
            vals = [p[k] for p in per_seed]
            agg[k] = round(float(np.mean(vals)), 4)
            agg[k + "_std"] = round(float(np.std(vals)), 4)
        agg["gain@50"] = round(agg["R@50"] - raw["R@50"], 4)
        agg["gain@10"] = round(agg["R@10"] - raw["R@10"], 4)
        agg["delta_dprime"] = round(agg["dprime"] - raw_dp, 4)
        agg["val_R@50"] = round(float(np.mean(val_best)), 4)
        agg["params_per_side"] = n_params
        if arm in lin:
            agg["linear_R@50"] = lin[arm].get("R@50")
            agg["linear_R@10"] = lin[arm].get("R@10")
            agg["mlp_minus_linear@50"] = round(agg["R@50"] - lin[arm]["R@50"], 4)
            agg["mlp_minus_linear@10"] = round(agg["R@10"] - lin[arm]["R@10"], 4)
        results[arm] = agg
        cmp = (f"  linear R@50 {lin[arm]['R@50']:.3f} -> mlp-linear {agg['mlp_minus_linear@50']:+.3f}" if arm in lin else "")
        print(f"[mlp] {arm:6s} R@10 {agg['R@10']:.3f}+-{agg['R@10_std']:.3f}  R@50 {agg['R@50']:.3f}+-{agg['R@50_std']:.3f}  "
              f"gain@50 {agg['gain@50']:+.3f}  d-dprime {agg['delta_dprime']:+.3f}{cmp}")

    report = {"dataset": ds, "collection": ds, "encoder": mtag, "mtag": mtag, "arch": "mlp", "hidden": args.hidden, "dropout": args.dropout,
              "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "temp": args.temp, "seeds": args.seeds,
              "n_train": int(len(tr_caps)), "n_val": int(len(va_caps)), "n_test": int(len(te_caps)), "dim": int(dim),
              "train_geometry": geom, "raw": {**raw, "dprime": round(raw_dp, 4)}, "arms": results,
              "linear_report": str(lin_p) if lin_p.exists() else None,
              "note": "residual bottleneck MLP per side (Linear-GELU-Dropout-Linear, last layer zero-init), the "
                      "query_projection.py 'mlp' arch applied symmetrically; recipe identical to symmetric_projection.py"}
    tag = (f"_tr{args.limit_train}" if args.limit_train else "") + (f"_h{args.hidden}" if args.hidden != 1024 else "")
    p = result_path(ds, mtag, f"mlp_arms{tag}.json")
    if p.exists():                       # merge partial-arm reruns, like symmetric_projection does
        try:
            old = json.load(open(p, encoding="utf-8"))
            kept = {a: v for a, v in old.get("arms", {}).items() if a not in results}
            if kept:
                report["arms"] = {**kept, **results}
        except Exception:
            pass
    json.dump(report, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[mlp] report -> {p}")


if __name__ == "__main__":
    main()
