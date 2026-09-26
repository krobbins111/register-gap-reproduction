"""Published protocols (registergap-pipeline; ported from registergap story/sets/eval_protocols.py + SOTA_COMPARISON.md).

    skincap:  the Derm1M protocol (Yan et al., ICCV 2025, Table 4): every image of the collection as gallery, our held-out test captions
              as queries; R@1/5/10/50 text->image.
    rsicd:    the standard RSITR protocol: all five captions of each test image as queries (5,465 on 1,093), text->image and image->text
              (an image query hits at k if any of its captions is in the top k); mR = mean of the six recalls.
    semart:   the Text2Art test split is our test split (description-only queries): the ordinary test recall is the protocol number.
Reference rows live in configs/published.json. Models are the saved seed-0 checkpoints of ltg.maps.sets (--save); frozen is always reported.
Results: results/<collection>/<encoder>/protocol<tag>.json.  usage: python -m ltg.eval.protocols --collection skincap --encoder siglip2-so400m-16-384 --variants pooled,set,setw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import encoder_key, result_path  # noqa: E402
from ltg.maps.sets import DEV, SetMaps, load_all, load_model, pad_phrases, saved_grid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _scores(m, sub, P, Q, ph, phm, gidx, qchunk=64, gchunk=256):
    g = torch.tensor(sub["img"][gidx], device=DEV); S = np.zeros((len(Q), len(gidx)), np.float32)
    for gs in range(0, len(gidx), gchunk):
        gi = gidx[gs:gs + gchunk]; patches = torch.tensor(np.asarray(P[gi]).astype(np.float32), device=DEV)
        for qs in range(0, len(Q), qchunk): S[qs:qs + qchunk, gs:gs + gchunk] = m.score(Q[qs:qs + qchunk], ph[qs:qs + qchunk], phm[qs:qs + qchunk], g[gs:gs + gchunk], patches).cpu().numpy()
    return S


def protocol(collection: str, encoder: str, variants=("pooled", "set", "setw"), tag: str = "", cap: int = 32, grid: int | None = None) -> Path:
    key = encoder_key(encoder); grid = grid or saved_grid(collection, key, variants, tag); sub, P, emb, offsets, _ = load_all(collection, key, grid, cap, False); d = sub["txt"].shape[1]; te = np.where(sub["split"] == "test")[0]; n = len(sub["split"])
    models = {"frozen": SetMaps(d, train_maps=False).to(DEV).eval()}
    for v in variants:
        try: models[v] = load_model(collection, key, v, tag, d)
        except FileNotFoundError: print("no saved model for", v)
    R = {"collection": collection, "encoder": key, "cap": cap, "grid": int(round(np.sqrt(P.shape[1]))), "tag": tag}
    if collection == "skincap":
        gidx = np.arange(n); Q = torch.tensor(sub["txt"][te], device=DEV); ph, phm = pad_phrases(emb, offsets, te, cap); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
        R["protocol"] = {"name": "Derm1M whole-collection gallery", "gallery": int(n), "queries": int(len(te))}
        for v, m in models.items():
            S = _scores(m, sub, P, Q, ph, phm, gidx); r = (S >= S[np.arange(len(te)), te][:, None]).sum(1)
            R[v] = {f"R@{k}": float((r <= k).mean()) for k in (1, 5, 10, 50)} | {"median_rank": float(np.median(r))}; print(collection, v, {k: round(x, 3) for k, x in R[v].items()})
    elif collection == "rsicd":
        from ltg.data.images import rows as _rows
        from ltg.encoders.patches import get_embedder
        from ltg.encoders.phrases import phrase_sets
        by_id = {r["id"]: r.get("captions") or [r["caption"]] for r in _rows("rsicd")}; ids = [str(i) for i in sub["id"][te]]
        caps = [c for i in ids for c in by_id[i]]; owner = np.repeat(np.arange(len(te)), [len(by_id[i]) for i in ids])
        pe = get_embedder(key); Q = torch.tensor(pe.texts(caps, bs=256), device=DEV); sets = phrase_sets(caps, cap, False); flat = [t for s in sets for t in s]; PE = pe.texts(flat, bs=256)
        poff = np.cumsum([0] + [len(s) for s in sets]); pmax = min(cap, max(len(s) for s in sets)); ph = np.zeros((len(caps), pmax, d), np.float32); phm = np.zeros((len(caps), pmax), bool)
        for k in range(len(caps)): s, e = poff[k], min(poff[k + 1], poff[k] + pmax); ph[k, :e - s] = PE[s:e]; phm[k, :e - s] = True
        ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
        R["protocol"] = {"name": "standard RSITR: all captions per test image", "gallery": int(len(te)), "queries": int(len(caps))}
        for v, m in models.items():
            S = _scores(m, sub, P, Q, ph, phm, te); true = S[np.arange(len(caps)), owner]; r = (S >= true[:, None]).sum(1)
            t2i = {f"T2I R@{k}": float((r <= k).mean()) for k in (1, 5, 10)}
            rank_i = np.array([int((S[:, j] > S[:, j][owner == j].max()).sum()) + 1 for j in range(len(te))]); i2t = {f"I2T R@{k}": float((rank_i <= k).mean()) for k in (1, 5, 10)}
            R[v] = t2i | i2t | {"mR": float(np.mean(list(t2i.values()) + list(i2t.values())))}; print(collection, v, {k: round(x, 3) for k, x in R[v].items()})
    else:
        R["protocol"] = {"name": "test split = the published test set (Text2Art for semart)", "gallery": int(len(te)), "queries": int(len(te))}
        Q = torch.tensor(sub["txt"][te], device=DEV); ph, phm = pad_phrases(emb, offsets, te, cap); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
        for v, m in models.items():
            S = _scores(m, sub, P, Q, ph, phm, te); r = (S >= np.diag(S)[:, None]).sum(1); R[v] = {f"R@{k}": float((r <= k).mean()) for k in (1, 5, 10, 50)}; print(collection, v, {k: round(x, 3) for k, x in R[v].items()})
    pub = json.load(open(REPO / "configs" / "published.json")).get(collection); R["published"] = pub
    out = result_path(collection, key, f"protocol{tag}.json"); json.dump(R, open(out, "w"), indent=1); print("->", out); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--encoder", required=True); ap.add_argument("--variants", default="pooled,set,setw"); ap.add_argument("--tag", default=""); ap.add_argument("--cap", type=int, default=32); ap.add_argument("--grid", type=int, default=None, help="pool the store to this grid (default: the saved model's)"); a = ap.parse_args(argv)
    protocol(a.collection, a.encoder, a.variants.split(","), a.tag, a.cap, a.grid)


if __name__ == "__main__":
    main()
