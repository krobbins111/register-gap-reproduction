"""Both-vs-shared decomposition — WHERE does the two-map edge come from?

Full grid: both beats shared in 48/55 cells (mean +0.006 R@50). The tied map
can express any symmetric stretch (rotation is provably inert in tied form),
so both's extra expressivity over shared is exactly two things:

    (1) stretch ASYMMETRY   — a different rescaling per side (S_t != S_i)
    (2) relative ROTATION   — re-aiming one cloud against the other (R_rel)

This is exact, not approximate: for any common orthogonal R_c,
(R_c A_t q)·(R_c A_i g) = (A_t q)·(A_i g), so with polar factors A_t = R_t S_t,
A_i = R_i S_i the both arm is score-equivalent to the pair
(S_t, R_rel S_i) with R_rel = R_t^T R_i (biases rotated by R_t^T).

Protocol per cell (cached embeddings; standard recipe, seed 0):
  train both + shared -> polar-decompose the affine maps A = I + W ->
  evaluate on test:
    shared        tied map (reference floor of the edge)
    two-stretch   (S_t, S_i), rotations stripped, biases rotated
    both          full two maps (reference ceiling)
  edge split:  d_asym  = twostretch - shared
               d_relrot = both - twostretch
  plus an EXACT equivalence check: (S_t, R_rel S_i) must reproduce both's
  recall to machine precision (validates the polar code), and R_rel stats
  (mean rotation angle over the top-energy subspace, ||R_rel - I||_F / sqrt(d)).

Usage:  python ablations/decompose_both.py --dataset nwpu [--mtag ...]
Output: out/<ds>/both_decomposition_<tag>.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import out_path  # noqa: E402
from symmetric_projection import (load_split, recall_table, dprime,  # noqa: E402
                                  train_arm, Identity)


def affine_of(module):
    """(A, b) with A = I + W from a trained ResidualLinear."""
    W = module.net.weight.detach().cpu().numpy().astype(np.float64)
    b = module.net.bias.detach().cpu().numpy().astype(np.float64)
    return np.eye(W.shape[0]) + W, b


def polar(A):
    """A = R S with R orthogonal (det-sign free), S symmetric PSD."""
    U, sig, Vt = np.linalg.svd(A)
    R = U @ Vt
    S = Vt.T @ np.diag(sig) @ Vt
    return R, S, sig


def apply_affine(x, A, b):
    y = x.astype(np.float64) @ A.T + b
    return (y / np.linalg.norm(y, axis=1, keepdims=True)).astype(np.float32)


def rot_stats(R_rel, sig_t, sig_i, top=64):
    """How far from identity is the relative rotation, overall and in the
    top-energy subspace (axes the stretches actually amplify)?"""
    d = R_rel.shape[0]
    fro = float(np.linalg.norm(R_rel - np.eye(d)) / np.sqrt(d))
    # rotation angles = phases of eigenvalues
    ang = np.abs(np.angle(np.linalg.eigvals(R_rel)))
    energy = np.argsort(-(sig_t + sig_i))[:top]
    sub = R_rel[np.ix_(energy, energy)]
    diag_mean = float(np.mean(np.diag(R_rel)))
    return {"fro_dist_per_sqrtd": round(fro, 4),
            "mean_angle_deg": round(float(np.degrees(ang.mean())), 3),
            "max_angle_deg": round(float(np.degrees(ang.max())), 3),
            "diag_mean": round(diag_mean, 5),
            f"top{top}_subblock_diag_mean": round(float(np.mean(np.diag(sub))), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    target = np.arange(len(te_imgs))
    raw = recall_table(te_caps, te_imgs, target)
    kw = dict(epochs=args.epochs, lr=1e-4, batch=256, temp=0.05,
              sym_loss=False, seed=args.seed, dev=dev)
    print(f"[decomp] {ds}/{mtag or 'default'} train={len(tr_caps)} dev={dev} "
          f"raw R@50={raw['R@50']:.4f}")

    # ── train the two reference arms ────────────────────────────────────────
    f_b, g_b = train_arm("both", tr_caps, tr_imgs, va_caps, va_imgs, **kw)
    f_s, _ = train_arm("shared", tr_caps, tr_imgs, va_caps, va_imgs, **kw)

    A_t, b_t = affine_of(f_b)
    A_i, b_i = affine_of(g_b)
    A_s, b_s = affine_of(f_s)
    R_t, S_t, sig_t = polar(A_t)
    R_i, S_i, sig_i = polar(A_i)
    R_rel = R_t.T @ R_i

    def ev(qA, qb, gA, gb):
        return recall_table(apply_affine(te_caps, qA, qb),
                            apply_affine(te_imgs, gA, gb), target)

    both = ev(A_t, b_t, A_i, b_i)
    shared = ev(A_s, b_s, A_s, b_s)
    twostretch = ev(S_t, R_t.T @ b_t, S_i, R_i.T @ b_i)
    # exact equivalence: strip only the COMMON rotation R_t
    equiv = ev(S_t, R_t.T @ b_t, R_rel @ S_i, R_t.T @ b_i)

    rep = {"dataset": ds, "mtag": mtag or "default", "seed": args.seed,
           "raw": raw, "both": both, "shared": shared,
           "twostretch": twostretch, "equiv_check": equiv,
           "edge": {
               "both_minus_shared@50": round(both["R@50"] - shared["R@50"], 4),
               "stretch_asym@50": round(twostretch["R@50"] - shared["R@50"], 4),
               "rel_rotation@50": round(both["R@50"] - twostretch["R@50"], 4),
               "both_minus_shared@10": round(both["R@10"] - shared["R@10"], 4),
               "stretch_asym@10": round(twostretch["R@10"] - shared["R@10"], 4),
               "rel_rotation@10": round(both["R@10"] - twostretch["R@10"], 4)},
           "equiv_abs_err@50": round(abs(equiv["R@50"] - both["R@50"]), 6),
           "rel_rotation_stats": rot_stats(R_rel, sig_t, sig_i),
           "stretch_asym_stats": {
               "S_diff_fro_rel": round(float(np.linalg.norm(S_t - S_i) /
                                             (0.5 * (np.linalg.norm(S_t - np.eye(len(S_t))) +
                                                     np.linalg.norm(S_i - np.eye(len(S_i))) + 1e-12))), 4),
               "sig_t_range": [round(float(sig_t.min()), 4), round(float(sig_t.max()), 4)],
               "sig_i_range": [round(float(sig_i.min()), 4), round(float(sig_i.max()), 4)]},
           "dprime": {"raw": round(dprime(te_caps, te_imgs, target), 4),
                      "both": round(dprime(apply_affine(te_caps, A_t, b_t),
                                           apply_affine(te_imgs, A_i, b_i),
                                           target), 4)}}

    outp = out_path(f"both_decomposition_{mtag or 'default'}.json", ds)
    json.dump(rep, open(outp, "w"), indent=1)
    e = rep["edge"]
    print(f"[decomp] shared {shared['R@50']:.4f}  twostretch "
          f"{twostretch['R@50']:.4f}  both {both['R@50']:.4f}  "
          f"(equiv err {rep['equiv_abs_err@50']:.6f})")
    print(f"[decomp] edge@50 {e['both_minus_shared@50']:+.4f} = asym "
          f"{e['stretch_asym@50']:+.4f} + relrot {e['rel_rotation@50']:+.4f}")
    print(f"[decomp] -> {outp}")


if __name__ == "__main__":
    main()
