"""Retrieval metrics shared by every experiment. Definitions are the paper's; the two source codebases agreed on them.

recall_table   R@1/5/10/50 + MRR + median rank, target = identity (caption i's image is gallery row i) unless given.
               ties='greater' counts strictly-better distractors (the grid pipeline's grid, iclr2027/ablations); ties='geq' also counts
               equal scores (the registergap pipeline's gap.py, pessimistic). They differ only on exact score ties (duplicates), and the
               grid numbers were produced with 'greater'.
near_miss_mass N_k = P[k < rank <= band*k] over a seeded sample of catalog pairs ranked against each other (the paper's N50:
               k=50, band=3, sample of gallery size drawn from the train split, mean over seeds). the registergap pipeline's near_miss_mass.
dprime         (own-sim - mean distractor sim) / distractor std, averaged over queries (the grid pipeline's compute_dprime semantics).
bootstrap_ci   percentile CI of a recall over query resampling (the registergap pipeline's tables).
"""
from __future__ import annotations

import numpy as np


def ranks_of_true(q: np.ndarray, gallery: np.ndarray, target: np.ndarray | None = None, ties: str = "greater") -> np.ndarray:
    sims = q @ gallery.T
    target = np.arange(len(q)) if target is None else target
    ts = sims[np.arange(len(target)), target]
    if ties == "greater":
        return 1 + (sims > ts[:, None]).sum(1)
    return (sims >= ts[:, None]).sum(1)          # includes self -> rank; pessimistic on ties


def recall_table(q: np.ndarray, gallery: np.ndarray, target: np.ndarray | None = None, ks=(1, 5, 10, 50),
                 ties: str = "greater") -> dict:
    r = ranks_of_true(q, gallery, target, ties)
    out = {f"R@{k}": float((r <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    return out


def bootstrap_ci(q, gallery, k: int = 10, n_boot: int = 200, seed: int = 0, target=None, ties="greater") -> list[float]:
    r = ranks_of_true(q, gallery, target, ties)
    hit = (r <= k).astype(np.float32)
    rng = np.random.default_rng(seed)
    vals = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def near_miss_mass(txt: np.ndarray, img: np.ndarray, k: int = 50, band: int = 3, n: int | None = None,
                   seeds=(0, 1, 2), ties: str = "geq") -> float:
    n = n or len(txt)
    vals = []
    for s in seeds:
        idx = np.random.RandomState(s).permutation(len(txt))[:n]      # RandomState, as in the registergap pipeline's gap.py, so N50 values match his files
        r = ranks_of_true(txt[idx], img[idx], ties=ties)
        vals.append(((r > k) & (r <= band * k)).mean())
    return float(np.mean(vals))


def dprime(q: np.ndarray, gallery: np.ndarray, target: np.ndarray | None = None) -> float:
    sims = q @ gallery.T
    n, g = sims.shape
    target = np.arange(n) if target is None else target
    own = sims[np.arange(n), target]
    tot, tot2 = sims.sum(1), (sims ** 2).sum(1)
    m = (tot - own) / (g - 1)
    var = (tot2 - own ** 2) / (g - 1) - m ** 2
    return float(np.mean((own - m) / np.sqrt(np.maximum(var, 1e-12))))


def near_dup_rate(x: np.ndarray, thr: float = 0.9, cap: int = 8000, seed: int = 0, chunk: int = 1024) -> float:
    """Fraction of points whose max cosine to another point >= thr (subsampled)."""
    if len(x) > cap:
        x = x[np.random.default_rng(seed).permutation(len(x))[:cap]]
    n, hits = len(x), 0
    for i in range(0, n, chunk):
        s = x[i:i + chunk] @ x.T
        s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])] = -1.0
        hits += int((s.max(1) >= thr).sum())
    return hits / n
