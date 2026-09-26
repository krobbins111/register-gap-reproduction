"""Polar factorization of the SHARED register-gap map: rotation vs stretch vs bias.

The shared arm of the symmetric ablation showed that ONE tied linear operator
phi(x) = normalize(A x + b), A = I + W, applied to BOTH query and gallery
embeddings, captures ~95-100% of the best arm's gain. This script asks what
that operator actually does, by the (right) polar decomposition

    A = R . S      R = U V^T (orthogonal: pure rotation),
                   S = V Sigma V^T (symmetric PSD: pure anisotropic stretch
                                    along the input singular directions),
    where A = U Sigma V^T is the SVD.

Per (dataset, encoder) cell: train the shared map once (seed 0, the exact
recipe of ablations/symmetric_projection.py), save (W, b), then evaluate five
variants under the SAME shared protocol (variant applied to both test queries
and test gallery, then renormalized):

    raw      identity
    R        rotation component only (no bias)
    S        stretch component only (no bias)
    bias     x + b only (learned translation; the meanshift analogue)
    full     normalize(A x + b) as trained

Readout per variant: R@1/5/10/50, MRR, d'. If S alone recovers most of the
gain, the correction is spectrum repair (re-weighting directions); if R is
needed, it is orientation repair; bias ~ raw is expected (the centroid-shift
baseline already hurt everywhere).

Also computes the cell's A-PRIORI train-split d' (raw), for refitting the
d' -> gain law against shared-arm gains.

Usage (from indexability/):
    python ablations/factorize_shared.py --dataset rocov2 [--mtag ...]
Output: out/<ds>/shared_map_<tag>.npz  (W, b)
        out/<ds>/shared_factorization_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))       # ablations/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # indexability/
from config import out_path                                    # noqa: E402
from symmetric_projection import (load_split, train_arm, recall_table,
                                  dprime)                      # noqa: E402


def apply_variant(M: np.ndarray | None, b: np.ndarray | None,
                  x: np.ndarray) -> np.ndarray:
    y = x if M is None else x @ M.T
    if b is not None:
        y = y + b[None, :]
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def eval_variant(M, b, te_caps, te_imgs, target) -> dict:
    q = apply_variant(M, b, te_caps)
    g = apply_variant(M, b, te_imgs)
    t = recall_table(q, g, target)
    t["dprime"] = round(dprime(q, g, target), 4)
    return t


def apriori_train_dprime(tr_caps, tr_imgs, cap=5000, seed=0) -> float:
    if len(tr_caps) > cap:
        idx = np.random.default_rng(seed).permutation(len(tr_caps))[:cap]
        tr_caps, tr_imgs = tr_caps[idx], tr_imgs[idx]
    return round(dprime(tr_caps, tr_imgs, np.arange(len(tr_imgs))), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = mtag or "default"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    target = np.arange(len(te_imgs))

    f, _ = train_arm("shared", tr_caps, tr_imgs, va_caps, va_imgs,
                     epochs=args.epochs, lr=args.lr, batch=args.batch,
                     temp=args.temp, sym_loss=False, seed=args.seed, dev=dev)
    W = f.net.weight.detach().cpu().numpy().astype(np.float64)
    b = f.net.bias.detach().cpu().numpy().astype(np.float64)
    np.savez_compressed(out_path(f"shared_map_{tag}.npz", ds), W=W, b=b)

    A = np.eye(W.shape[0]) + W
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt                       # rotation (orthogonal)
    S = Vt.T @ np.diag(sv) @ Vt      # stretch (symmetric PSD); A = R @ S
    assert np.allclose(A, R @ S, atol=1e-8)

    variants = {
        "raw":  eval_variant(None, None, te_caps, te_imgs, target),
        "R":    eval_variant(R, None, te_caps, te_imgs, target),
        "S":    eval_variant(S, None, te_caps, te_imgs, target),
        "bias": eval_variant(None, b, te_caps, te_imgs, target),
        "full": eval_variant(A, b, te_caps, te_imgs, target),
    }
    raw50, full50 = variants["raw"]["R@50"], variants["full"]["R@50"]
    span = max(full50 - raw50, 1e-9)
    for k, t in variants.items():
        t["gain@50"] = round(t["R@50"] - raw50, 4)
        t["gain@10"] = round(t["R@10"] - variants["raw"]["R@10"], 4)
        t["frac_of_full@50"] = round((t["R@50"] - raw50) / span, 3)

    rep = {
        "dataset": ds, "mtag": tag, "seed": args.seed, "epochs": args.epochs,
        "apriori_train_dprime_raw": apriori_train_dprime(tr_caps, tr_imgs),
        "sv_max": round(float(sv[0]), 4), "sv_min": round(float(sv[-1]), 4),
        "sv_median": round(float(np.median(sv)), 4),
        "n_sv_gt_1.05": int((sv > 1.05).sum()),
        "n_sv_lt_0.95": int((sv < 0.95).sum()),
        "norm_b": round(float(np.linalg.norm(b)), 4),
        "det_sign_R": float(np.sign(np.linalg.det(R))),
        "variants": variants,
        "note": "shared protocol: variant applied to BOTH test queries and "
                "test gallery, then renormalized. A = R.S (right polar); "
                "R = U V^T rotation, S = V Sigma V^T stretch.",
    }
    p = out_path(f"shared_factorization_{tag}.json", ds)
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[fact] {ds}/{tag}  raw {raw50:.3f} -> full {full50:.3f}  | "
          f"R {variants['R']['R@50']:.3f} ({variants['R']['frac_of_full@50']:.0%}) "
          f"S {variants['S']['R@50']:.3f} ({variants['S']['frac_of_full@50']:.0%}) "
          f"bias {variants['bias']['R@50']:.3f}")
    print(f"[fact] report -> {p}")


if __name__ == "__main__":
    main()
