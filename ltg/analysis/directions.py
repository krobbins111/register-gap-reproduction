"""Name the directions of the two linear maps with text (registergap-pipeline; ported from registergap story/directions/name_directions.py; after TextSpan,
Gandelsman et al. ICLR 2024). A residual map phi(x) = normalize(x + W x + b) with W = U S V^T: direction k reads the input along v_k and adds
s_k u_k; cos(u_k, v_k) near +1 is a stretch, near -1 a suppression, near 0 a rotation. Both ends are named by the phrases (the collection's own
spaCy phrases and COCO's, the web register) whose unit embeddings project most on them. Register specificity of a direction = mean projection
of the collection's train captions (text map) or train images (image map) on v_k minus the same for COCO.
    directions(collection, encoder, reference='coco', variants=('pooled','set','setw'))
      -> results/<collection>/<encoder>/directions.json (+ DIRECTIONS.md table). Pooled maps are retrained (seed 0, the grid pipeline's 'both' arm =
      the same recipe); set / gate maps are the saved seed-0 checkpoints of ltg.maps.sets when present.
usage: python -m ltg.analysis.directions --collection skincap --encoder siglip2-base-16-256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import encoder_key, load_cell, models_dir, phrases_path, result_path  # noqa: E402

TOPK, NP = 8, 6


def phrase_store(collection, encoder, min_count=2):
    key = encoder_key(encoder)
    for cap, verbs in ((32, True), (32, False), (12, False)):
        p = phrases_path(collection, key, cap, verbs)
        if not p.exists(): continue
        z = np.load(p, allow_pickle=True); texts = [str(t).lower().strip() for t in z["texts"]]; E = z["emb"].astype(np.float32); uniq = {}
        for i, t in enumerate(texts): uniq.setdefault(t, []).append(i)
        keys = [t for t, ix in uniq.items() if len(ix) >= min_count] or list(uniq)
        V = np.stack([E[uniq[t][0]] for t in keys]); V /= np.linalg.norm(V, axis=1, keepdims=True); return keys, V
    raise FileNotFoundError(f"no phrase store for {collection} x {key}")


def _top(vec, keys, V, n=NP, sign=1):
    p = V @ vec * sign; hi = np.argsort(-p)[:n]; return [(keys[i], round(float(p[i]), 3)) for i in hi]


def describe(W, own_keys, own_V, X_tr, X_ref, ref_keys, ref_V, caps_tr=None):
    U, S, Vt = np.linalg.svd(W); energy = (S ** 2).cumsum() / (S ** 2).sum(); rows = []
    for k in range(TOPK):
        v, u, s = Vt[k], U[:, k], float(S[k]); cos = float(u @ v); po, pc = X_tr @ v, X_ref @ v; spec = float(po.mean() - pc.mean())
        if spec < 0: v, u, po, pc, spec = -v, -u, -po, -pc, -spec
        row = {"k": k + 1, "s": round(s, 3), "cos_uv": round(cos, 2), "specificity": round(spec, 3), "own_mean": round(float(po.mean()), 3), "own_sd": round(float(po.std()), 3), "ref_mean": round(float(pc.mean()), 3),
               "v_own+": _top(v, own_keys, own_V), "v_ref+": _top(v, ref_keys, ref_V), "v_own-": _top(v, own_keys, own_V, sign=-1), "u_own+": _top(u, own_keys, own_V), "u_ref+": _top(u, ref_keys, ref_V), "u_own-": _top(u, own_keys, own_V, sign=-1), "u_ref-": _top(u, ref_keys, ref_V, sign=-1)}
        if caps_tr is not None:
            i = np.argsort(-po)[:4]; row["v_images+"] = [str(caps_tr[j])[:90] for j in i]; i = np.argsort(po)[:3]; row["v_images-"] = [str(caps_tr[j])[:90] for j in i]
        rows.append(row)
    return {"frob": round(float(np.linalg.norm(W)), 3), "energy_top8": round(float(energy[7]), 3), "energy_top16": round(float(energy[15]), 3), "top_s": [round(float(x), 3) for x in S[:16]], "directions": rows}


def maps_of(collection, encoder, variants):
    from ltg.maps.linear import train_arm
    key = encoder_key(encoder); c = load_cell(collection, key); tr, va = c.split == "train", c.split == "val"; out = {}
    if "pooled" in variants:
        f, g = train_arm("both", c.txt[tr], c.img[tr], c.txt[va], c.img[va], epochs=30, lr=1e-4, batch=256, temp=0.05, sym_loss=False, seed=0, dev="cuda" if torch.cuda.is_available() else "cpu")
        out["pooled"] = {"Wt": f.net.weight.detach().cpu().numpy(), "Wi": g.net.weight.detach().cpu().numpy()}
    for v in variants:
        if v == "pooled": continue
        p = models_dir(collection, key) / f"{v}.pt"
        if p.exists(): st = torch.load(p, map_location="cpu")["state"]; out[v] = {"Wt": st["Wt.weight"].numpy(), "Wi": st["Wi.weight"].numpy()}
    return c, tr, out


def directions(collection, encoder="siglip2-base-16-256", reference="coco", variants=("pooled", "set", "setw")) -> Path:
    key = encoder_key(encoder); ref_keys, ref_V = phrase_store(reference, key); rc = load_cell(reference, key)
    own_keys, own_V = phrase_store(collection, key); c, tr, maps = maps_of(collection, key, variants); txt_tr, img_tr, caps_tr = c.txt[tr], c.img[tr], c.caption[tr]
    ALL = {"collection": collection, "encoder": key, "reference": reference}; md = [f"# Directions of the maps: {collection} x {key} (reference {reference})", ""]
    fmt = lambda L: ", ".join(f"{t} ({p:+.2f})" for t, p in L)
    for variant, W in maps.items():
        for side, k_, X_tr, X_ref, caps in (("text map", "Wt", txt_tr, rc.txt, None), ("image map", "Wi", img_tr, rc.img, caps_tr)):
            desc = describe(W[k_], own_keys, own_V, X_tr, X_ref, ref_keys, ref_V, caps); ALL[f"{variant}/{k_}"] = desc
            md += [f"## {variant}, {side} (|W|_F {desc['frob']}, energy top-8 {desc['energy_top8']:.0%}, top-16 {desc['energy_top16']:.0%})", "", "| k | s | cos(u,v) | specificity | v_k reads (own) | v_k reads (ref) | u_k writes (own) | u_k writes (ref) | u_k opposes (own) |", "|---|---|---|---|---|---|---|---|---|"]
            md += [f"| {r['k']} | {r['s']} | {r['cos_uv']:+.2f} | {r['specificity']:+.3f} | {fmt(r['v_own+'])} | {fmt(r['v_ref+'])} | {fmt(r['u_own+'])} | {fmt(r['u_ref+'])} | {fmt(r['u_own-'])} |" for r in desc["directions"]]
            md.append("")
    out = result_path(collection, key, "directions.json"); json.dump(ALL, open(out, "w"), indent=1); open(out.with_name("DIRECTIONS.md"), "w").write("\n".join(md) + "\n"); print("->", out); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--encoder", default="siglip2-base-16-256"); ap.add_argument("--reference", default="coco"); ap.add_argument("--variants", default="pooled,set,setw"); a = ap.parse_args(argv)
    directions(a.collection, a.encoder, a.reference, a.variants.split(","))


if __name__ == "__main__":
    main()
