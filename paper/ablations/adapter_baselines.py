"""Adapter baselines for the register-gap study: Tip-Adapter(-F) and
CLIP-Adapter, translated from few-shot classification to text-to-image
retrieval. Protocols verified against the papers (arXiv:2111.03930,
arXiv:2110.04544); the retrieval translation is ours and is stated in the
paper.

Tip-Adapter (training-free)  [Zhang et al., ECCV'22]
    Classification: keys = few-shot image features, values = one-hot labels;
    affinity A = exp(-beta (1 - q K^T)); logits = alpha A L + zero-shot.
    Retrieval translation: keys = TRAIN caption embeddings, values = their
    paired TRAIN image embeddings. The cache predicts an image vector
    v(q) = normalize(A(q) V) and the final score blends cache evidence with
    the raw score: s*(q,g) = q.g + alpha * v(q).g .  alpha, beta swept on the
    val split by R@50 (no test tuning).

Tip-Adapter-F  [same paper]
    Unfreezes the cache KEYS only (values fixed), fine-tuned with the paper's
    budget (20 epochs, SGD lr 1e-3) under our in-batch InfoNCE on the blended
    score; alpha,beta fixed to the training-free winner; val-R@50 selection.

CLIP-Adapter  [Gao et al., 2021]
    Bottleneck head A(f) = ReLU(f W1) W2 with hidden d//4, residual-ratio
    blend f* = normalize(r * A(f) + (1-r) * f). Placements: text / image /
    both (paper's best was visual; we report all three). Trained with OUR
    standard recipe (InfoNCE tau=0.05, AdamW 1e-4/wd 1e-4, 30 epochs,
    val-R@50 selection) so the only difference from our arms is the
    architecture; residual ratio r swept on val.

Usage:  python ablations/adapter_baselines.py --dataset skincap [--mtag ...]
Output: results/<ds>/adapter_baselines_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import out_path                                    # noqa: E402
from symmetric_projection import load_split, recall_table, dprime  # noqa: E402

TIP_ALPHAS = (0.25, 0.5, 1.0, 2.0)
TIP_BETAS = (1.0, 3.0, 5.5, 10.0)
CA_RATIOS = (0.2, 0.4, 0.6)


def table(q, g, target):
    t = recall_table(q, g, target)
    t["dprime"] = round(dprime(q, g, target), 4)
    return t


def score_table(scores, target):
    ts = scores[np.arange(len(target)), target]
    ranks = 1 + (scores > ts[:, None]).sum(1)
    out = {f"R@{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 50)}
    out["MRR"] = float((1.0 / ranks).mean())
    n, g = scores.shape
    own = scores[np.arange(n), target]
    m = (scores.sum(1) - own) / (g - 1)
    var = ((scores ** 2).sum(1) - own ** 2) / (g - 1) - m ** 2
    out["dprime"] = round(float(np.mean((own - m) /
                                        np.sqrt(np.maximum(var, 1e-12)))), 4)
    return out


# ── Tip-Adapter ──────────────────────────────────────────────────────────────

def tip_scores(q, K, V, G, alpha, beta):
    A = np.exp(-beta * (1.0 - q @ K.T))
    v = A @ V
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return q @ G.T + alpha * (v @ G.T)


def tip_free(tr_caps, tr_imgs, va_caps, va_imgs, te_caps, te_imgs, target):
    best = None
    va_t = np.arange(len(va_imgs))
    for a in TIP_ALPHAS:
        for b in TIP_BETAS:
            r = score_table(tip_scores(va_caps, tr_caps, tr_imgs, va_imgs,
                                       a, b), va_t)["R@50"]
            if best is None or r > best[0]:
                best = (r, a, b)
    _, a, b = best
    res = score_table(tip_scores(te_caps, tr_caps, tr_imgs, te_imgs, a, b),
                      target)
    res.update({"alpha": a, "beta": b})
    return res


def tip_finetuned(tr_caps, tr_imgs, va_caps, va_imgs, te_caps, te_imgs,
                  target, alpha, beta, dev, seed=0, epochs=20, lr=1e-3,
                  batch=256):
    torch.manual_seed(seed)
    K = nn.Parameter(torch.tensor(tr_caps, device=dev).clone())
    V = torch.tensor(tr_imgs, device=dev)
    opt = torch.optim.SGD([K], lr=lr)
    X = torch.tensor(tr_caps, device=dev)
    Y = torch.tensor(tr_imgs, device=dev)
    n = len(X)
    best_r, best_K = -1.0, None
    va_t = np.arange(len(va_imgs))
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev,
                              generator=torch.Generator(dev).manual_seed(
                                  seed * 1000 + ep))
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            q, g = X[idx], Y[idx]
            A = torch.exp(-beta * (1.0 - q @ K.T))
            v = nn.functional.normalize(A @ V, dim=-1)
            logits = (q @ g.T + alpha * (v @ g.T)) / 0.05
            loss = nn.functional.cross_entropy(
                logits, torch.arange(len(idx), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Kn = K.detach().cpu().numpy()
        r = score_table(tip_scores(va_caps, Kn, tr_imgs, va_imgs,
                                   alpha, beta), va_t)["R@50"]
        if r > best_r:
            best_r, best_K = r, Kn.copy()
    res = score_table(tip_scores(te_caps, best_K, tr_imgs, te_imgs,
                                 alpha, beta), target)
    res.update({"alpha": alpha, "beta": beta, "epochs": epochs, "lr": lr})
    return res


# ── CLIP-Adapter ─────────────────────────────────────────────────────────────

class BottleneckAdapter(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.w1 = nn.Linear(dim, dim // 4)
        self.w2 = nn.Linear(dim // 4, dim)
        self.r = ratio

    def forward(self, x):
        a = self.w2(torch.relu(self.w1(x)))
        return nn.functional.normalize(self.r * a + (1 - self.r) * x, dim=-1)


class MaybeAdapter(nn.Module):
    """Adapter or identity, so text/image/both placements share one loop."""
    def __init__(self, dim, ratio, active):
        super().__init__()
        self.net = BottleneckAdapter(dim, ratio) if active else None

    def forward(self, x):
        return self.net(x) if self.net is not None else x


def clip_adapter(placement, ratio, tr_caps, tr_imgs, va_caps, va_imgs,
                 te_caps, te_imgs, target, dev, seed=0, epochs=30):
    torch.manual_seed(seed)
    dim = tr_caps.shape[1]
    ft = MaybeAdapter(dim, ratio, placement in ("text", "both")).to(dev)
    fi = MaybeAdapter(dim, ratio, placement in ("image", "both")).to(dev)
    params = [p for m in (ft, fi) for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    X = torch.tensor(tr_caps, device=dev)
    Y = torch.tensor(tr_imgs, device=dev)
    n = len(X)
    best_r, best = -1.0, None
    va_t = np.arange(len(va_imgs))
    for ep in range(epochs):
        ft.train(); fi.train()
        perm = torch.randperm(n, device=dev,
                              generator=torch.Generator(dev).manual_seed(
                                  seed * 1000 + ep))
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            logits = ft(X[idx]) @ fi(Y[idx]).T / 0.05
            loss = nn.functional.cross_entropy(
                logits, torch.arange(len(idx), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        ft.eval(); fi.eval()
        with torch.no_grad():
            q = ft(torch.tensor(va_caps, device=dev)).cpu().numpy()
            g = fi(torch.tensor(va_imgs, device=dev)).cpu().numpy()
        r = recall_table(q, g, va_t)["R@50"]
        if r > best_r:
            best_r = r
            best = ([{k: v.detach().clone() for k, v in m.state_dict().items()}
                     for m in (ft, fi)])
    for m, st in zip((ft, fi), best):
        m.load_state_dict(st)
        m.eval()
    with torch.no_grad():
        q = ft(torch.tensor(te_caps, device=dev)).cpu().numpy()
        g = fi(torch.tensor(te_imgs, device=dev)).cpu().numpy()
    res = table(q, g, target)
    res.update({"placement": placement, "ratio": ratio, "epochs": epochs})
    return res, best_r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mtag", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--tip-epochs", type=int, default=20)
    ap.add_argument("--ca-epochs", type=int, default=30)
    args = ap.parse_args()
    ds, mtag = args.dataset, args.mtag
    tag = mtag or "default"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_caps, tr_imgs = load_split(ds, "train", mtag)
    va_caps, va_imgs = load_split(ds, "val", mtag)
    te_caps, te_imgs = load_split(ds, "test", mtag)
    target = np.arange(len(te_imgs))
    raw = table(te_caps, te_imgs, target)
    print(f"[adpt] {ds}/{tag} train={len(tr_caps)} dev={dev} "
          f"raw R@50 {raw['R@50']:.3f}")

    rep = {"dataset": ds, "mtag": tag, "n_train": int(len(tr_caps)),
           "raw": raw, "note": "retrieval translations of Tip-Adapter(-F) "
           "(arXiv:2111.03930) and CLIP-Adapter (arXiv:2110.04544); "
           "alpha/beta/ratio tuned on val R@50; CLIP-Adapter uses our "
           "standard training recipe for architecture-only comparison"}

    tf = tip_free(tr_caps, tr_imgs, va_caps, va_imgs, te_caps, te_imgs, target)
    tf["gain@50"] = round(tf["R@50"] - raw["R@50"], 4)
    rep["tip_adapter"] = tf
    print(f"[adpt] tip (a={tf['alpha']}, b={tf['beta']})  "
          f"R@50 {tf['R@50']:.3f}  gain {tf['gain@50']:+.3f}")

    per = [tip_finetuned(tr_caps, tr_imgs, va_caps, va_imgs, te_caps, te_imgs,
                         target, tf["alpha"], tf["beta"], dev, seed=s,
                         epochs=args.tip_epochs) for s in args.seeds]
    agg = {k: round(float(np.mean([p[k] for p in per])), 4)
           for k in ("R@1", "R@5", "R@10", "R@50", "MRR", "dprime")}
    agg["R@50_std"] = round(float(np.std([p["R@50"] for p in per])), 4)
    agg.update({k: per[0][k] for k in ("alpha", "beta", "epochs", "lr")})
    agg["gain@50"] = round(agg["R@50"] - raw["R@50"], 4)
    rep["tip_adapter_f"] = agg
    print(f"[adpt] tip-F  R@50 {agg['R@50']:.3f}±{agg['R@50_std']:.3f}  "
          f"gain {agg['gain@50']:+.3f}")

    ca = {}
    for placement in ("text", "image", "both"):
        best = None
        for ratio in CA_RATIOS:
            per, val_scores = [], []
            for s in args.seeds:
                r, vr = clip_adapter(placement, ratio, tr_caps, tr_imgs,
                                     va_caps, va_imgs, te_caps, te_imgs,
                                     target, dev, seed=s,
                                     epochs=args.ca_epochs)
                per.append(r); val_scores.append(vr)
            mv = float(np.mean(val_scores))
            if best is None or mv > best[0]:
                agg = {k: round(float(np.mean([p[k] for p in per])), 4)
                       for k in ("R@1", "R@5", "R@10", "R@50", "MRR",
                                 "dprime")}
                agg["R@50_std"] = round(float(np.std(
                    [p["R@50"] for p in per])), 4)
                agg.update({"ratio": ratio, "placement": placement,
                            "epochs": args.ca_epochs})
                agg["gain@50"] = round(agg["R@50"] - raw["R@50"], 4)
                best = (mv, agg)
        ca[placement] = best[1]
        print(f"[adpt] clip-adapter/{placement} (r={best[1]['ratio']})  "
              f"R@50 {best[1]['R@50']:.3f}  gain {best[1]['gain@50']:+.3f}")
    rep["clip_adapter"] = ca

    p = out_path(f"adapter_baselines_{tag}.json", ds)
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[adpt] report -> {p}")


if __name__ == "__main__":
    main()
