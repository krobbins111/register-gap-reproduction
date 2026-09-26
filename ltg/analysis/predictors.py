"""Can the gain be predicted before training? (registergap-pipeline; ported from registergap story/sets/predict_gain.py.)

For every (collection, encoder) with a sets result, pair the realised gains (pooled maps at R@50, the extra R@10 of sets over pooled) with
statistics that need no training: N50 (near-miss mass on train pairs, band 3 and band 2), the frozen test recall, caption-cloud and
image-cloud shift from the reference collection, the within-collection modality gap, spreads, words and phrases per caption.
    predictors(encoder, reference='coco') -> results/_analysis/<encoder>/predict_gain.json (+ correlations printed)
usage: python -m ltg.analysis.predictors --encoder siglip2-base-16-256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import cache_root, encoder_key, load_cell, phrases_path, results_root  # noqa: E402
from ltg.eval.metrics import near_miss_mass, recall_table  # noqa: E402


def predictors(encoder="siglip2-base-16-256", reference="coco", kind_of=None) -> Path:
    key = encoder_key(encoder); ref = load_cell(reference, key); rtr = ref.split == "train"; mu_t0, mu_i0 = ref.txt[rtr].mean(0), ref.img[rtr].mean(0); rows = []
    for rp in sorted(results_root().glob(f"*/{key}/sets*.json")):
        coll = rp.parent.parent.name; R = json.load(open(rp))
        if not all(k in R for k in ("frozen", "pooled", "set")): continue
        try: c = load_cell(coll, key)
        except FileNotFoundError: print("no cell for", coll); continue
        tr = c.split == "train"; Q, G = c.txt[tr], c.img[tr]; nmax = min(len(Q), 5000)
        n50, n50b2 = near_miss_mass(Q, G, k=50, band=3, n=nmax), near_miss_mass(Q, G, k=50, band=2, n=nmax)
        caps = [str(x) for x in c.caption[tr]]; words = float(np.mean([len(x.split()) for x in caps])) if any(caps) else float("nan")
        ph = next((p for p in (phrases_path(coll, key, 32, v) for v in (False, True)) if p.exists()), None); ppc = float(np.mean(np.diff(np.load(ph, allow_pickle=True)["offsets"]))) if ph else float("nan")
        rows.append(dict(cell=rp.stem, collection=coll, kind=(kind_of or {}).get(coll, ""), n_train=int(tr.sum()), frozen_R50=R["frozen"]["R@50"], frozen_R10=R["frozen"]["R@10"],
                         pooled_gain_R50=R["pooled"]["R@50"] - R["frozen"]["R@50"], pooled_gain_R10=R["pooled"]["R@10"] - R["frozen"]["R@10"], set_gain_R50=R["set"]["R@50"] - R["frozen"]["R@50"], set_gain_R10=R["set"]["R@10"] - R["frozen"]["R@10"],
                         set_minus_pooled_R10=R["set"]["R@10"] - R["pooled"]["R@10"], N50_b3=n50, N50_b2=n50b2, text_shift=float(np.linalg.norm(Q.mean(0) - mu_t0)), image_shift=float(np.linalg.norm(G.mean(0) - mu_i0)),
                         modality_gap=float(np.linalg.norm(Q.mean(0) - G.mean(0))), text_spread=float(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1).mean())), image_spread=float(np.sqrt(((G - G.mean(0)) ** 2).sum(1).mean())), words=words, phrases_per_caption=ppc, mean_pair_cos=float((Q * G).sum(1).mean())))
        print(f"{coll:14s} {rp.stem:12s} N50 {n50:.3f} frozen R@50 {rows[-1]['frozen_R50']:.3f} pooled gain {rows[-1]['pooled_gain_R50']:+.3f} set-pooled R@10 {rows[-1]['set_minus_pooled_R10']:+.3f}", flush=True)
    out = results_root() / "_analysis" / key / "predict_gain.json"; out.parent.mkdir(parents=True, exist_ok=True); json.dump(rows, open(out, "w"), indent=1)
    if len(rows) >= 3:
        from scipy.stats import pearsonr, spearmanr
        for target in ("pooled_gain_R50", "set_gain_R50", "set_minus_pooled_R10"):
            for pred in ("N50_b3", "N50_b2", "frozen_R50", "text_shift", "image_shift", "modality_gap", "words", "phrases_per_caption"):
                x, y = np.array([r[pred] for r in rows], float), np.array([r[target] for r in rows], float); m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= 3: print(f"  {pred:20s} -> {target:22s} rho {spearmanr(x[m], y[m])[0]:+.2f}  r {pearsonr(x[m], y[m])[0]:+.2f} (n={int(m.sum())})")
    print("->", out); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default="siglip2-base-16-256"); ap.add_argument("--reference", default="coco"); a = ap.parse_args(argv)
    predictors(a.encoder, a.reference)


if __name__ == "__main__":
    main()
