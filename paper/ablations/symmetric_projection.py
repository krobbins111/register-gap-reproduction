"""Symmetric-projection ablation — WHICH SIDE of the register gap should move?

Reviewer 3 (WACV): "The projection is applied only to text?? Symmetric
alternatives are not compared empirically." This script settles it. Four arms,
identical training method (residual zero-init map(s), in-batch InfoNCE,
t2i cross-entropy, AdamW 1e-4/wd 1e-4, temp 0.05, 30 epochs, val-R@50 model
selection — byte-for-byte the recipe from query_projection.py):

    text    f on queries, gallery untouched          (the paper's method)
    image   g on gallery images, queries untouched   (index-side re-embed)
    both    f and g trained jointly                  (2x params)
    shared  ONE map applied to both modalities       (tied weights)

Why `shared` is a diagnostic and not just a fourth arm: applying the SAME
rotation to both clouds changes nothing about their RELATIVE orientation —
a pure cone-to-cone rotation is inexpressible with tied weights. So:
    shared ~ raw        => the gain is relative rotation (re-aiming);
    shared ~ text-only  => the gain is spectrum reshaping (axis spreading),
                           which a tied map CAN express.
Together with the SVD stats this decomposes the mechanism per dataset.

Hypothesis under test (grid-pipeline): the winning side tracks RELATIVE collapse —
a gallery far more collapsed than its text cloud (ROCOv2/SigLIP2: img nd@0.9
= 0.905 vs txt 0.189) may reward an image-side map (spread the gallery)
over the text-side default. The script therefore reports per-side train-split
geometry (near-dup@0.9, mean pair cos) and per-arm d-prime + SVD spectra, so
whichever arm wins, the geometry that predicted it is in the same JSON.

Everything runs from CACHED embeddings (out/<ds>/{split}_embs*.npz) — no
encoder, no GPU images; a full 4-arm x 3-seed run is minutes per dataset.

Suggested first wave (see report table in the ICLR gameplan):
    # image-side-collapsed, text scattered — the arm most likely to flip:
    python ablations/symmetric_projection.py --dataset rocov2
    # text cloud tighter than image cloud (slides: tighter text => bigger gain):
    python ablations/symmetric_projection.py --dataset skincap
    python ablations/symmetric_projection.py --dataset treeoflife
    # both sides collapsed (flagship gain), then the no-gap control:
    python ablations/symmetric_projection.py --dataset nwpu
    python ablations/symmetric_projection.py --dataset coco
    # second encoder geometry, same datasets:
    python ablations/symmetric_projection.py --dataset rocov2 --mtag clip-vitl14-laion2b

Output: out/<ds>/symmetric_ablation_<tag>.json + console table.
Note: semart's DEFAULT-backbone eval uses the prebuilt external gallery; this
script evaluates self-gallery datasets (test split = gallery), which covers
every dataset above. Extend load paths before adding semart/default or inquire.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Windows: redirected stdout defaults to cp1252 and dies on non-ASCII prints.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # indexability/
from config import out_path  # noqa: E402

ARMS = ("text", "image", "both", "shared")


# ── cached-embedding loader ──────────────────────────────────────────────────

def load_split(ds: str, split: str, mtag: str) -> tuple[np.ndarray, np.ndarray]:
    """(caps, imgs) from the cache written by query_projection.get_split_arrays.
    Resolution order: exact full-split file, then the largest _n<limit> variant
    (COCO caches are _n20000/_n3000). Refuses to guess across backbones."""
    tag = f"_{mtag}" if mtag else ""
    exact = out_path(f"{split}_embs{tag}.npz", ds)
    if exact.exists():
        z = np.load(exact)
        return z["caps"], z["imgs"]
    cands = sorted(glob.glob(str(out_path(f"{split}_embs{tag}_n*.npz", ds))),
                   key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
    if mtag == "" :  # default tag must not swallow e.g. train_embs_medsiglip.npz
        cands = [p for p in cands if Path(p).name.split("_embs")[1].startswith("_n")]
    if not cands:
        raise FileNotFoundError(
            f"no cached {split} embeddings for {ds} (mtag={mtag or 'default'}); "
            f"run query_projection.py once for this dataset/backbone first")
    z = np.load(cands[0])
    print(f"[sym] {ds}/{split}: using cached subset {Path(cands[0]).name}")
    return z["caps"], z["imgs"]


# ── model (identical to query_projection.QueryProjection, linear arm) ────────

class ResidualLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim, dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x):  # residual + renorm: epoch 0 == raw baseline
        return nn.functional.normalize(x + self.net(x), dim=-1)


class Identity(nn.Module):
    def forward(self, x):
        return x


def make_arm(arm: str, dim: int, dev: str) -> tuple[nn.Module, nn.Module, list]:
    """Return (f_text, g_image, trainable_params) for an arm."""
    if arm == "text":
        f, g = ResidualLinear(dim).to(dev), Identity()
        params = list(f.parameters())
    elif arm == "image":
        f, g = Identity(), ResidualLinear(dim).to(dev)
        params = list(g.parameters())
    elif arm == "both":
        f, g = ResidualLinear(dim).to(dev), ResidualLinear(dim).to(dev)
        params = list(f.parameters()) + list(g.parameters())
    elif arm == "shared":
        f = ResidualLinear(dim).to(dev)
        g = f                                   # tied weights
        params = list(f.parameters())
    else:
        raise ValueError(arm)
    return f, g, params


# ── metrics ──────────────────────────────────────────────────────────────────

def recall_table(q: np.ndarray, gallery: np.ndarray, target: np.ndarray) -> dict:
    sims = q @ gallery.T
    ts = sims[np.arange(len(target)), target]
    ranks = 1 + (sims > ts[:, None]).sum(1)
    out = {f"R@{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 50)}
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def dprime(q: np.ndarray, gallery: np.ndarray, target: np.ndarray) -> float:
    """Cross-modal discriminability: (own-sim − mean distractor sim) / distractor
    std, averaged over queries. Matches analysis/compute_dprime.py semantics."""
    sims = q @ gallery.T
    n, g = sims.shape
    own = sims[np.arange(n), target]
    tot, tot2 = sims.sum(1), (sims ** 2).sum(1)
    m = (tot - own) / (g - 1)
    var = (tot2 - own ** 2) / (g - 1) - m ** 2
    return float(np.mean((own - m) / np.sqrt(np.maximum(var, 1e-12))))


def near_dup_rate(x: np.ndarray, thr: float = 0.9, cap: int = 8000,
                  seed: int = 0, chunk: int = 1024) -> float:
    """Fraction of points whose max cosine to another point ≥ thr (subsampled)."""
    if len(x) > cap:
        idx = np.random.default_rng(seed).permutation(len(x))[:cap]
        x = x[idx]
    n, hits = len(x), 0
    for i in range(0, n, chunk):
        s = x[i:i + chunk] @ x.T
        s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])] = -1.0
        hits += int((s.max(1) >= thr).sum())
    return hits / n


def side_geometry(caps: np.ndarray, imgs: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    def mean_pair(x):
        k = min(len(x), 2000)
        i = rng.permutation(len(x))[:k]
        s = x[i] @ x[i].T
        return float((s.sum() - k) / (k * (k - 1)))
    return {"txt_near_dup@0.9": round(near_dup_rate(caps, seed=seed), 4),
            "img_near_dup@0.9": round(near_dup_rate(imgs, seed=seed), 4),
            "txt_mean_pair_cos": round(mean_pair(caps), 4),
            "img_mean_pair_cos": round(mean_pair(imgs), 4),
            "matched_pair_cos": round(float((caps * imgs).sum(1).mean()), 4)}


def map_svd_stats(mod: nn.Module, x: np.ndarray, dev: str) -> dict:
    """Spectrum of A = I + W and cone displacement on real inputs — ties the
    winner back to the rotate-vs-spread mechanism story."""
    if isinstance(mod, Identity):
        return {}
    W = mod.net.weight.detach().cpu().numpy()
    A = np.eye(W.shape[0], dtype=W.dtype) + W
    sv = np.linalg.svd(A, compute_uv=False)
    with torch.no_grad():
        y = mod(torch.tensor(x[:2000], device=dev)).cpu().numpy()
    disp_cos = float((x[:2000] * y).sum(1).mean())
    return {"sv_max": round(float(sv[0]), 3), "sv_min": round(float(sv[-1]), 3),
            "sv_median": round(float(np.median(sv)), 3),
            "frob_W": round(float(np.linalg.norm(W)), 3),
            "n_sv_gt_1.05": int((sv > 1.05).sum()),
            "n_sv_lt_0.95": int((sv < 0.95).sum()),
            "mean_cos_x_Ax": round(disp_cos, 4),
            "mean_rot_deg": round(float(np.degrees(np.arccos(
                np.clip(disp_cos, -1, 1)))), 2)}


# ── training-free baselines (QB-Norm, CSLS, centroid shift) ──────────────────
# All use ONLY the train split (querybank = train captions, centroids from
# train), so they are deployable pre-query exactly like the projection — no
# test-time transduction. They answer: how much of the projection's gain do
# standard hubness corrections / the generic modality-gap fix already capture?

def baseline_tables(tr_caps, tr_imgs, te_caps, te_imgs, target, *,
                    qb_size=5000, qb_beta=20.0, csls_k=10, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    qb = tr_caps[rng.permutation(len(tr_caps))[:qb_size]]
    S = te_caps @ te_imgs.T                       # raw scores (Q, G)
    B = qb @ te_imgs.T                            # querybank scores (Nb, G)
    out = {}

    def table_from_scores(sc):
        ts = sc[np.arange(len(target)), target]
        ranks = 1 + (sc > ts[:, None]).sum(1)
        t = {f"R@{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 50)}
        t["MRR"] = float((1.0 / ranks).mean())
        # d' on the score matrix (per-query z-score => scale-invariant)
        n, g = sc.shape
        own = sc[np.arange(n), target]
        m = (sc.sum(1) - own) / (g - 1)
        var = ((sc ** 2).sum(1) - own ** 2) / (g - 1) - m ** 2
        t["dprime"] = float(np.mean((own - m) / np.sqrt(np.maximum(var, 1e-12))))
        return t

    # centroid shift (Liang et al.-style modality-gap fix, train centroids).
    # NOTE: for ranking this is provably equivalent to adding the query-
    # independent bias s.g_j per gallery item (renormalization is a per-query
    # scale) -- i.e., it boosts central/hub gallery images for every query.
    shift = tr_imgs.mean(0) - tr_caps.mean(0)
    qs = te_caps + shift[None, :]
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    out["meanshift"] = table_from_scores(qs @ te_imgs.T)

    # symmetric gap closure — Liang et al.'s exact construction (NeurIPS'22):
    # text + lam*Delta, images - lam*Delta, renormalize; lam=0.5 closes the
    # centroid gap fully.
    lam = 0.5
    qg = te_caps + lam * shift[None, :]
    qg /= np.linalg.norm(qg, axis=1, keepdims=True)
    gg = te_imgs - lam * shift[None, :]
    gg /= np.linalg.norm(gg, axis=1, keepdims=True)
    out["gapclose_sym_l0.5"] = table_from_scores(qg @ gg.T)

    # symmetric variant: center EACH modality on its own train centroid and
    # renormalize -- the fairer "close the gap" baseline under a two-sided
    # protocol (this one genuinely changes angular geometry).
    qc = te_caps - tr_caps.mean(0)[None, :]
    qc /= np.linalg.norm(qc, axis=1, keepdims=True)
    gc = te_imgs - tr_imgs.mean(0)[None, :]
    gc /= np.linalg.norm(gc, axis=1, keepdims=True)
    out["centerboth"] = table_from_scores(qc @ gc.T)

    # CSLS (Conneau et al.): 2*S - r_gallery (querybank-side hub penalty).
    # The query-side term r_q is constant per row => rank-invariant; omitted.
    k = min(csls_k, len(qb))
    r_g = np.sort(B, axis=0)[-k:, :].mean(0)      # mean top-k sim per gallery item
    out[f"csls_k{k}"] = table_from_scores(2.0 * S - r_g[None, :])

    # QB-Norm with dynamic inverted softmax (Bogolin et al., CVPR'22).
    # IS log-score: beta*S_ij - logsumexp_b(beta*B_bj); applied per test query
    # only when its raw top-1 gallery item is in the querybank-activated set.
    col_max = B.max(0)
    den = col_max * qb_beta + np.log(
        np.exp(qb_beta * (B - col_max[None, :])).sum(0))   # (G,) logsumexp
    is_scores = qb_beta * S - den[None, :]
    activated = np.zeros(te_imgs.shape[0], dtype=bool)
    activated[np.unique(B.argmax(1))] = True
    gate = activated[S.argmax(1)]                 # per-query: top-1 activated?
    dyn = np.where(gate[:, None], is_scores,
                   qb_beta * S - float(den.mean()))  # unnormalized rows: keep
    out[f"qbnorm_b{int(qb_beta)}"] = table_from_scores(dyn)
    out["qbnorm_gate_frac"] = float(gate.mean())
    out["querybank_size"] = int(len(qb))
    return out


# ── training (one arm, one seed) ─────────────────────────────────────────────

def train_arm(arm: str, tr_caps, tr_imgs, va_caps, va_imgs, *, epochs, lr,
              batch, temp, sym_loss, seed, dev) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)                       # batch order; init is zeros
    dim = tr_caps.shape[1]
    f, g, params = make_arm(arm, dim, dev)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    Xtr = torch.tensor(tr_caps, device=dev)
    Ytr = torch.tensor(tr_imgs, device=dev)
    n = len(Xtr)
    best_r50, best = -1.0, None
    for ep in range(epochs):
        for m in (f, g):
            m.train()
        perm = torch.randperm(n, device=dev,
                              generator=torch.Generator(dev).manual_seed(
                                  seed * 1000 + ep))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            fq, gi = f(Xtr[idx]), g(Ytr[idx])
            logits = fq @ gi.T / temp
            tgt = torch.arange(len(idx), device=dev)
            loss = nn.functional.cross_entropy(logits, tgt)
            if sym_loss:                          # optional i2t direction
                loss = 0.5 * (loss + nn.functional.cross_entropy(logits.T, tgt))
            opt.zero_grad(); loss.backward(); opt.step()
        for m in (f, g):
            m.eval()
        with torch.no_grad():
            fv = f(torch.tensor(va_caps, device=dev)).cpu().numpy() \
                if not isinstance(f, Identity) else va_caps
            gv = g(torch.tensor(va_imgs, device=dev)).cpu().numpy() \
                if not isinstance(g, Identity) else va_imgs
        r50 = recall_table(fv, gv, np.arange(len(gv)))["R@50"]
        if r50 > best_r50:
            best_r50 = r50
            best = tuple({k: v.detach().clone() for k, v in m.state_dict().items()}
                         if not isinstance(m, Identity) else None for m in (f, g))
    for m, st in zip((f, g), best):
        if st is not None:
            m.load_state_dict(st)
    for m in (f, g):
        m.eval()
    return f, g


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=("rocov2", "skincap", "treeoflife", "nwpu", "coco",
                             "rsicd", "fashion200k", "facad", "scimmir",
                             "goodnews", "semart"))
    ap.add_argument("--mtag", default="",
                    help="backbone tag in the cache filenames ('' = default "
                         "SigLIP2, e.g. clip-vitl14-laion2b, medsiglip)")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS,
                    help="pass no values (--arms) to skip arm training, e.g. "
                         "for a baselines-only run")
    ap.add_argument("--baselines", action="store_true",
                    help="also compute training-free baselines (centroid "
                         "shift, CSLS, QB-Norm dynamic-IS) -> "
                         "retrieval_baselines_<tag>.json")
    ap.add_argument("--qb-size", type=int, default=5000)
    ap.add_argument("--qb-beta", type=float, default=20.0)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--sym-loss", action="store_true",
                    help="add the i2t InfoNCE direction (secondary check; "
                         "default keeps the paper's t2i-only loss)")
    ap.add_argument("--limit-train", type=int, default=None)
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    if args.limit_train and args.limit_train < len(tr_caps):
        idx = np.sort(np.random.default_rng(0).permutation(
            len(tr_caps))[:args.limit_train])
        tr_caps, tr_imgs = tr_caps[idx], tr_imgs[idx]
    target = np.arange(len(te_imgs))

    geom = side_geometry(tr_caps, tr_imgs)
    raw = recall_table(te_caps, te_imgs, target)
    raw_dp = dprime(te_caps, te_imgs, target)
    print(f"[sym] {ds} (mtag={mtag or 'default'})  train={len(tr_caps)} "
          f"val={len(va_caps)} test={len(te_caps)}  dev={dev}")
    print(f"[sym] train geometry: {json.dumps(geom)}")
    print(f"[sym] raw test: R@10 {raw['R@10']:.3f}  R@50 {raw['R@50']:.3f}  "
          f"d' {raw_dp:.3f}")

    if args.baselines:
        bl = baseline_tables(tr_caps, tr_imgs, te_caps, te_imgs, target,
                             qb_size=args.qb_size, qb_beta=args.qb_beta,
                             csls_k=args.csls_k)
        for name, t in bl.items():
            if isinstance(t, dict):
                t["gain@50"] = round(t["R@50"] - raw["R@50"], 4)
                t["gain@10"] = round(t["R@10"] - raw["R@10"], 4)
                print(f"[sym] BL {name:12s} R@10 {t['R@10']:.3f}  "
                      f"R@50 {t['R@50']:.3f}  gain@50 {t['gain@50']:+.3f}")
        blrep = {"dataset": ds, "mtag": mtag or "default",
                 "n_train": int(len(tr_caps)), "n_test": int(len(te_caps)),
                 "raw": {**raw, "dprime": round(raw_dp, 4)}, "baselines": bl,
                 "note": "querybank/centroids from train split only "
                         "(pre-query deployable, no test transduction)"}
        bltag = mtag or "default"
        blp = out_path(f"retrieval_baselines_{bltag}.json", ds)
        json.dump(blrep, open(blp, "w", encoding="utf-8"), indent=2)
        print(f"[sym] baselines -> {blp}")

    results = {}
    for arm in args.arms:
        per_seed = []
        svd_f, svd_g = {}, {}
        for seed in args.seeds:
            f, g = train_arm(arm, tr_caps, tr_imgs, va_caps, va_imgs,
                             epochs=args.epochs, lr=args.lr, batch=args.batch,
                             temp=args.temp, sym_loss=args.sym_loss,
                             seed=seed, dev=dev)
            with torch.no_grad():
                fq = f(torch.tensor(te_caps, device=dev)).cpu().numpy() \
                    if not isinstance(f, Identity) else te_caps
                gi = g(torch.tensor(te_imgs, device=dev)).cpu().numpy() \
                    if not isinstance(g, Identity) else te_imgs
            t = recall_table(fq, gi, target)
            t["dprime"] = dprime(fq, gi, target)
            t["gal_near_dup@0.9_post"] = round(near_dup_rate(gi), 4) \
                if not isinstance(g, Identity) else geom["img_near_dup@0.9"]
            per_seed.append(t)
            if seed == args.seeds[0]:
                svd_f = map_svd_stats(f, te_caps, dev)
                svd_g = map_svd_stats(g, te_imgs, dev) if g is not f else \
                    {"tied": True, **map_svd_stats(g, te_imgs, dev)}
        agg = {}
        for k in per_seed[0]:
            vals = [p[k] for p in per_seed]
            agg[k] = round(float(np.mean(vals)), 4)
            agg[k + "_std"] = round(float(np.std(vals)), 4)
        agg["gain@50"] = round(agg["R@50"] - raw["R@50"], 4)
        agg["gain@10"] = round(agg["R@10"] - raw["R@10"], 4)
        agg["delta_dprime"] = round(agg["dprime"] - raw_dp, 4)
        agg["map_svd_text"], agg["map_svd_image"] = svd_f, svd_g
        results[arm] = agg
        print(f"[sym] {arm:6s} R@10 {agg['R@10']:.3f}+-{agg['R@10_std']:.3f}  "
              f"R@50 {agg['R@50']:.3f}+-{agg['R@50_std']:.3f}  "
              f"gain@50 {agg['gain@50']:+.3f}  d-dprime {agg['delta_dprime']:+.3f}")

    if not results:            # baselines-only run: never overwrite arm reports
        return
    report = {"dataset": ds, "mtag": mtag or "default", "arch": "linear",
              "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
              "temp": args.temp, "sym_loss": bool(args.sym_loss),
              "seeds": args.seeds, "n_train": int(len(tr_caps)),
              "n_val": int(len(va_caps)), "n_test": int(len(te_caps)),
              "train_geometry": geom,
              "raw": {**raw, "dprime": round(raw_dp, 4)},
              "arms": results,
              "deploy_note": "text arm: per-query transform, index untouched. "
                             "image/both/shared arms: gallery must be re-"
                             "transformed and re-indexed whenever the map "
                             "changes."}
    tag = "_".join(p for p in (mtag, f"tr{args.limit_train}" if
                               args.limit_train else "") if p) or "default"
    if args.sym_loss:
        tag += "_symloss"
    p = out_path(f"symmetric_ablation_{tag}.json", ds)
    if p.exists():
        # Partial-arm reruns must never clobber arms from an earlier run
        # (e.g. adding text/image to a wave-3 both+shared report): merge.
        try:
            old = json.load(open(p, encoding="utf-8"))
            kept = {a: v for a, v in old.get("arms", {}).items()
                    if a not in results}
            if kept:
                report["arms"] = {**kept, **results}
                report["merged_arms_note"] = (
                    f"arms {sorted(kept)} kept from earlier run; "
                    f"arms {sorted(results)} from this run "
                    f"(seeds {args.seeds})")
                print(f"[sym] merged with existing report "
                      f"(kept {sorted(kept)})")
        except Exception as e:  # noqa: BLE001
            print(f"[sym] WARNING: could not merge existing report ({e}); "
                  f"overwriting")
    json.dump(report, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[sym] report -> {p}")


if __name__ == "__main__":
    main()
