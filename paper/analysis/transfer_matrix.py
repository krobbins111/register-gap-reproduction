"""Cross-collection transfer of the learned maps (register-gap test).

Apply the maps learned on collection A to the test split of collection B, for
every ordered pair, and record the change in R@50 relative to B's raw
encoder. Prediction if the maps repair a COLLECTION-SPECIFIC register gap:
a dominant diagonal, positive off-diagonal entries only where registers are
shared (RSICD<->NWPU aerial scene sentences; Fashion200k<->FACAD catalogue
photos), and a COCO row near zero. Prediction if they were closing a GLOBAL
offset (the modality gap): a map learned anywhere would help everywhere.

As the global-offset comparison, the same matrix is computed for symmetric
centering learned on A's train split (subtract A's caption and image
centroids, renormalise), the one gap-style operation that helped in-domain.

Inputs   results/maps/<ds>/maps_<mtag>.npz     (ablations\export_maps.py)
         cached test / train embeddings
Outputs  results/transfer_<mtag>.json, results/transfer_<mtag>_{both,tied,center}.csv
         results/figs/transfer_matrix_<mtag>.pdf/.png

Usage:   python analysis\transfer_matrix.py            (default encoder, all collections with maps)
         python analysis\transfer_matrix.py --mtag clip-vitl14-laion2b
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
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

ORDER = ["coco", "facad", "fashion200k", "goodnews", "semart", "scimmir", "skincap",
         "treeoflife", "rocov2", "rsicd", "nwpu"]
DSNAME = {"nwpu": "NWPU", "rocov2": "ROCOv2", "skincap": "SkinCAP", "treeoflife": "TreeOfLife",
          "rsicd": "RSICD", "scimmir": "SciMMIR", "fashion200k": "Fashion200k", "semart": "SemArt",
          "coco": "COCO", "goodnews": "GoodNews", "facad": "FACAD"}
SAME_REGISTER = {("rsicd", "nwpu"), ("nwpu", "rsicd"), ("fashion200k", "facad"), ("facad", "fashion200k")}


def find(ds, mtag, split):
    tag = "" if mtag == "default" else f"_{mtag}"
    exact = CACHE_ROOT / ds / f"{split}_embs{tag}.npz"
    if exact.exists():
        return exact
    cands = sorted(glob.glob(str(CACHE_ROOT / ds / f"{split}_embs{tag}_n*.npz")),
                   key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
    if mtag == "default":
        cands = [p for p in cands if Path(p).name.split("_embs")[1].startswith("_n")]
    if not cands:
        raise FileNotFoundError(f"no {split} cache for {ds}/{mtag}")
    return Path(cands[0])


def load(ds, mtag, split):
    z = np.load(find(ds, mtag, split))
    return nrm(z["caps"].astype(np.float32)), nrm(z["imgs"].astype(np.float32))


def nrm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def apply(x, W, b):
    return nrm(x + x @ W.T + b)


def r50(q, g, chunk=2048):
    n = len(q); c = 0
    for i in range(0, n, chunk):
        s = q[i:i + chunk] @ g.T
        own = s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])]
        c += ((1 + (s > own[:, None]).sum(1)) <= 50).sum()
    return c / n


def heatmap(M, order, raw, path, title_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    n = len(order)
    fig, ax = plt.subplots(figsize=(0.62 * n + 1.8, 0.62 * n + 1.2))
    vmax = float(np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(range(n), [DSNAME[d] for d in order], rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(n), [DSNAME[d] for d in order], fontsize=8.5)
    ax.set_xlabel("applied to (test split)"); ax.set_ylabel("maps learned on")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.55 * vmax else "#0b0b0b",
                    fontweight="bold" if i == j else "normal")
    for (a, b) in SAME_REGISTER:
        if a in order and b in order:
            i, j = order.index(a), order.index(b)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#0b0b0b", lw=1.2, ls="--"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("ΔR@50 over raw")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarise(M, order):
    n = len(order); off = ~np.eye(n, dtype=bool)
    same = np.zeros_like(off)
    for a, b in SAME_REGISTER:
        if a in order and b in order:
            same[order.index(a), order.index(b)] = True
    pos = [(order[i], order[j], float(M[i, j])) for i in range(n) for j in range(n)
           if i != j and M[i, j] > 0.01]
    return {"diag_mean": float(np.diag(M).mean()),
            "offdiag_mean": float(M[off].mean()),
            "offdiag_max": float(M[off].max()),
            "offdiag_frac_gt_0.01": float((M[off] > 0.01).mean()),
            "same_register_mean": float(M[same].mean()) if same.any() else None,
            "other_offdiag_mean": float(M[off & ~same].mean()),
            "coco_row_mean_excl_self": float(M[order.index("coco"), :][[j for j in range(n) if order[j] != "coco"]].mean())
            if "coco" in order else None,
            "positive_offdiag": sorted(pos, key=lambda t: -t[2])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtag", default="default")
    args = ap.parse_args()
    mt = args.mtag
    order = [d for d in ORDER if (RESULTS_ROOT / "maps" / d / f"maps_{mt}.npz").exists()]
    missing = [d for d in ORDER if d not in order]
    if missing:
        print(f"[transfer] no exported maps for {missing} (run ablations\\export_maps.py); continuing with {len(order)}")
    maps = {d: np.load(RESULTS_ROOT / "maps" / d / f"maps_{mt}.npz") for d in order}
    tests = {d: load(d, mt, "test") for d in order}
    cents = {}
    for d in order:
        c, i = load(d, mt, "train")
        cents[d] = (c.mean(0), i.mean(0))
    raw = {d: r50(*tests[d]) for d in order}
    n = len(order)
    M = {k: np.zeros((n, n)) for k in ("both", "tied", "center")}
    for i, src in enumerate(order):
        m = maps[src]
        for j, tgt in enumerate(order):
            q, g = tests[tgt]
            M["both"][i, j] = r50(apply(q, m["W_text"], m["b_text"]), apply(g, m["W_image"], m["b_image"])) - raw[tgt]
            M["tied"][i, j] = r50(apply(q, m["W_tied"], m["b_tied"]), apply(g, m["W_tied"], m["b_tied"])) - raw[tgt]
            mc, mi = cents[src]
            M["center"][i, j] = r50(nrm(q - mc), nrm(g - mi)) - raw[tgt]
        print(f"[transfer] maps from {src:12s} " + " ".join(f"{M['both'][i, j]:+.3f}" for j in range(n)), flush=True)
    figs = RESULTS_ROOT.parent / "figs"; figs.mkdir(exist_ok=True)
    heatmap(M["both"], order, raw, figs / f"transfer_matrix_{mt}", "two maps")
    heatmap(M["center"], order, raw, figs / f"transfer_center_{mt}", "centering")
    rep = {"mtag": mt, "order": order, "raw_R50": raw,
           "matrices": {k: v.tolist() for k, v in M.items()},
           "summary": {k: summarise(v, order) for k, v in M.items()},
           "note": "rows: collection the correction was learned on; cols: collection it is applied to; "
                   "entries: R@50 after minus raw R@50 of the target. 'center' = symmetric centering with the "
                   "source's train centroids."}
    json.dump(rep, open(RESULTS_ROOT / f"transfer_{mt}.json", "w", encoding="utf-8"), indent=1)
    for k, v in M.items():
        with open(RESULTS_ROOT / f"transfer_{mt}_{k}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["learned_on \\ applied_to"] + order)
            for i, d in enumerate(order):
                w.writerow([d] + [f"{x:+.4f}" for x in v[i]])
    s = rep["summary"]["both"]
    print(f"[transfer] two maps: diag mean {s['diag_mean']:+.3f}, off-diag mean {s['offdiag_mean']:+.3f}, "
          f"same-register mean {s['same_register_mean']}, other off-diag mean {s['other_offdiag_mean']:+.3f}, "
          f"COCO row mean {s['coco_row_mean_excl_self']}")
    print(f"[transfer] positive off-diagonal (> +0.01): {s['positive_offdiag']}")
    sc = rep["summary"]["center"]
    print(f"[transfer] centering: diag mean {sc['diag_mean']:+.3f}, off-diag mean {sc['offdiag_mean']:+.3f}")
    print(f"[transfer] -> {RESULTS_ROOT / f'transfer_{mt}.json'}")


if __name__ == "__main__":
    main()
