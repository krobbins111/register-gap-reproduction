"""Query projection — a learned residual map from query embeddings toward the
gallery's image manifold, trained from the gallery's own catalog.

Supervision is free: every TRAIN item pairs its query-register text (curator
caption / clinical figure caption / …) with its own image embedding. We learn

    f(query_emb) = normalize(query_emb + Net(query_emb))

with an InfoNCE loss (in-batch negatives) pulling f(orig) toward its own
image and away from the others, train on the train split, model-select on
val by R@50, and evaluate retrieval on the untouched test gallery (the same
gallery every other number for that dataset uses).

The map is residual + zero-init, so epoch 0 == the raw-query baseline and the
model can only learn a correction. "linear" fits a single affine map
(I+W)q+b; "mlp" adds a non-linear branch. On SemArt the linear map *beats*
the MLP — the curator→image register gap is, to first order, one matrix.

Datasets (--dataset)
--------------------
    semart   1,069-image test gallery; references: raw 0.554, G1 LLM rewrite
             0.815, oracle query->card 0.987.
    rocov2   ROCOv2-radiology test gallery (~9.9k figures), clinical-caption
             queries. A *harder* register gap than SemArt under generic
             SigLIP2 (vanilla R@50≈0.49) — but much of that is image-side
             collapse, so swap the backbone (below) and the projection rides
             whatever space the encoder gives it.

Backbones (--embed-model)
-------------------------
    default  google/siglip2-so400m-patch16-384  (generic; references apply)
    medical  google/medsiglip-448               (SigLIP-family, gated)
    open     microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
             (open_clip; trained on PMC-15M == ROCOv2's source)
Embeddings/checkpoints/reports are keyed by a per-backbone tag, so backbones
never collide and the default SigLIP2 artifacts are untouched.

Usage (3080-friendly; the one-time cost is embedding the train+val images):

    # SemArt (unchanged):
    python query_projection.py --arch linear

    # ROCOv2, generic SigLIP2:
    python query_projection.py --dataset rocov2 --arch linear --limit-train 20000 --limit-val 3000

    # ROCOv2, medical backbone (re-embeds under the new space, then cached):
    python query_projection.py --dataset rocov2 --arch linear --embed-model google/medsiglip-448
    python query_projection.py --dataset rocov2 --arch linear \
        --embed-model microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backbones import make_embedder, model_tag, resolve_encoder
from config import (EMBED_MODEL, INQUIRE_ENCODER, PAPER_DISCLAIMER, ROOT, DATA_DIR,
                    out_path, semart_dir_from_env)
from manifest import load_meta, load_queries
from splits import load_split_pairs

# Published reference points per dataset (R@50), printed inline so a run is
# self-explanatory. Only valid for the DEFAULT SigLIP2 backbone — omitted when
# a domain encoder is used (the run's own raw row is then the baseline).
REFERENCES: dict[str, dict] = {
    # The LLM-rewrite reference (0.815) is the DEPLOYED gallery-blind rewrite
    # system (with RRF/rerank); the frac-of-rewrite-gain field below lands on the
    # ~0.64 figure. The ceiling (0.987) is the self-retrieval ceiling (each image
    # from its own visual caption).
    "semart": {"reference_raw_R@50": 0.554,
               "reference_llm_rewrite_R@50": 0.815,
               "reference_self_retrieval_ceiling_R@50": 0.987},
    "rocov2": {"reference_raw_R@50": 0.525},  # vanilla SigLIP2, 1k-query probe
    # INQUIRE: no single trustworthy fullrank number to pin here — the paper's
    # best models score mAP@50 < 0.50 on the FULL 5M gallery, and our subset
    # gallery is easier (see splits/inquire.py). So the run's own raw mAP@50 row
    # is the baseline; we deliberately omit a reference to avoid a misleading
    # apples-to-oranges comparison.
}


# ── data ────────────────────────────────────────────────────────────────────

def resolve_meta(dataset: str) -> dict:
    """Embed config: from out/<ds>/meta.json if step1 built it, else the
    repo default (SigLIP2) — keeps every dataset in one embedding space."""
    p = out_path("meta.json", dataset)
    if p.exists():
        return load_meta(p)
    from config import TEXT_MAX_TOKENS
    return {"embed_model": EMBED_MODEL, "text_max_tokens": TEXT_MAX_TOKENS}


def _preview_texts(tag: str, texts, embs=None, n: int = 5) -> None:
    """Sanity print: the first few query texts actually being embedded (confirm
    the right register/field reaches the encoder), plus the embedding's shape and
    first vector's norm/dims. Only fires on the embedding path (cache miss), which
    is exactly when text -> embedding happens."""
    k = min(n, len(texts))
    print(f"[projection] PREVIEW {tag}: first {k} of {len(texts)} texts embedded:")
    for i, t in enumerate(texts[:k]):
        s = " ".join(str(t).split())
        print(f"  [{i}] {s[:200]}{'…' if len(s) > 200 else ''}")
    if embs is not None and len(embs):
        print(f"  -> emb shape {tuple(embs.shape)}  |v0|="
              f"{float(np.linalg.norm(embs[0])):.3f}  "
              f"v0[:4]={np.round(embs[0][:4], 4).tolist()}")


def _emb_cache_name(split: str, mtag: str, limit: int | None) -> str:
    """Backbone- and limit-keyed embedding cache filename."""
    name = f"{split}_embs"
    if mtag:
        name += f"_{mtag}"
    if limit is not None:
        name += f"_n{limit}"
    return name + ".npz"


def get_split_arrays(dataset: str, split: str, embedder, *,
                     limit: int | None = None, seed: int = 0, mtag: str = "",
                     semart_dir: Path | None = None,
                     cache_dir: Path | None = None,
                     embed_batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(caption_embs, image_embs) for a split, cached under out/<ds>/.

    Caching strategy that makes the train-size curve cheap:
      * full split (no --limit) caches to <split>_embs[_<mtag>].npz;
      * once that exists, any --limit just subsamples the cached arrays
        (no re-embedding) — so the curve is a seconds-level sweep;
      * a --limit run with no full cache yet embeds only that subset and
        caches it to <split>_embs[_<mtag>]_n<limit>.npz.
    """
    full_cache = out_path(_emb_cache_name(split, mtag, None), dataset)
    if full_cache.exists():
        z = np.load(full_cache)
        caps, imgs = z["caps"], z["imgs"]
        if limit is not None and limit < len(caps):
            idx = np.random.default_rng(seed).permutation(len(caps))[:limit]
            idx.sort()
            caps, imgs = caps[idx], imgs[idx]
            print(f"[projection] {split}: subsampled {len(caps)} of {len(z['caps'])} "
                  f"from cached full split")
        return caps, imgs

    if limit is not None:
        lim_cache = out_path(_emb_cache_name(split, mtag, limit), dataset)
        if lim_cache.exists():
            z = np.load(lim_cache)
            return z["caps"], z["imgs"]

    texts, paths = load_split_pairs(dataset, split, limit=limit, seed=seed,
                                    semart_dir=semart_dir, cache_dir=cache_dir)
    print(f"[projection] embedding {split}: {len(texts)} captions + images "
          f"(one-time, cached)")
    caps = embedder.embed_texts(texts)
    _preview_texts(f"{dataset}/{split}", texts, caps)
    imgs = embedder.embed_images(paths, batch_size=embed_batch)
    target = out_path(_emb_cache_name(split, mtag, None if limit is None
                                      else limit), dataset)
    np.savez_compressed(target, caps=caps, imgs=imgs)
    return caps, imgs


def load_test_eval(dataset: str, embedder, *, limit: int | None = None,
                   seed: int = 0, mtag: str = "", semart_dir: Path | None = None,
                   cache_dir: Path | None = None, embed_batch: int = 16
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (query_embs, gallery_embs, target_idx) for the test gallery.

    SemArt reuses the prebuilt out/semart artifacts so its numbers stay
    byte-identical to every other table in the paper. Self-gallery datasets
    (ROCOv2) build the gallery from the test split: gallery = test images,
    queries = their captions, target = identity. INQUIRE is the exception: an
    EXTERNAL, MULTI-POSITIVE gallery (iNat24) — `target` is then a list of
    per-query positive-index arrays (handled by multipos_table in main), and the
    gallery embeddings come from step1 (run it first).
    """
    if dataset == "inquire":
        from splits.inquire import embed_inquire_test
        # Returns (query_embs, gallery_embs, positives_list). positives_list is
        # ragged (1..1000+ positives/query), so the metric is multipos_table.
        return embed_inquire_test(embedder, mtag=mtag, cache_dir=cache_dir)

    if dataset == "semart":
        if mtag:
            # Non-default backbone: no prebuilt gallery exists in this embedding
            # space, so build the test self-gallery from the test split (each
            # description retrieves its own image) — exactly like ROCOv2/NWPU.
            # The DEFAULT SigLIP2 numbers still come from the byte-identical
            # prebuilt gallery in the branch below.
            caps, imgs = get_split_arrays(
                "semart", "test", embedder, limit=limit, seed=seed, mtag=mtag,
                semart_dir=semart_dir, cache_dir=cache_dir,
                embed_batch=embed_batch)
            return caps, imgs, np.arange(len(imgs))
        gallery = np.load(out_path("gallery_embeddings.npz",
                                    "semart"))["embeddings"]
        queries = [q for q in load_queries(out_path("queries.json", "semart"))
                   if q.text]
        qtexts = [q.text for q in queries]
        caps = embedder.embed_texts(qtexts)
        _preview_texts("semart/test-queries", qtexts, caps)
        target = np.array([q.positives[0] for q in queries])
        return caps, gallery, target

    caps, imgs = get_split_arrays(dataset, "test", embedder, limit=limit,
                                  seed=seed, mtag=mtag, semart_dir=semart_dir,
                                  cache_dir=cache_dir, embed_batch=embed_batch)
    return caps, imgs, np.arange(len(imgs))


# ── model + metric ───────────────────────────────────────────────────────────

class QueryProjection(nn.Module):
    def __init__(self, dim: int, hidden: int, arch: str):
        super().__init__()
        if arch == "linear":
            self.net = nn.Linear(dim, dim)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden, dim))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):  # residual: starts as identity
        return nn.functional.normalize(x + self.net(x), dim=-1)


def recall_table(q: np.ndarray, gallery: np.ndarray,
                 target: np.ndarray) -> dict:
    sims = q @ gallery.T
    ts = sims[np.arange(len(target)), target]
    ranks = 1 + (sims > ts[:, None]).sum(1)
    out = {f"R@{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 50)}
    out["MRR"] = float((1.0 / ranks).mean())
    return out


# ── multi-positive metric (INQUIRE) ──────────────────────────────────────────
# Single-positive recall_table can't score INQUIRE: each query has 1..1000+
# correct images. mAP@50 is INQUIRE's headline metric, computed exactly as its
# reference code (src/metrics.ap_at_k): the precision summed over the positives
# in the top-k, normalized by min(count_pos, k).

def _ap_at_k(rel_sorted: np.ndarray, count_pos: int, k: int) -> float:
    rel = rel_sorted[:k]
    if rel.sum() == 0:
        return 0.0
    cum = np.cumsum(rel)
    prec_at = cum / np.arange(1, len(rel) + 1)
    return float((prec_at * rel).sum() / min(count_pos, k))


def multipos_table(q: np.ndarray, gallery: np.ndarray,
                   positives: list[np.ndarray], ks=(1, 5, 10, 50)) -> dict:
    """Multi-positive retrieval metrics. mAP@k = mean over queries of INQUIRE's
    ap_at_k; R@k = mean fraction of a query's positives found in the top-k.
    Headline = mAP@50. O(Q.G log G) — fine for a subset gallery (≤~600k)."""
    sims = q @ gallery.T                            # (Q, G)
    order = np.argsort(-sims, axis=1)               # ranked gallery idx per query
    maxk = max(ks)
    ap = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    for i, pos in enumerate(positives):
        cp = int(len(pos))
        if cp == 0:
            continue
        posset = set(int(x) for x in pos)
        topk = order[i, :maxk]
        rel = np.fromiter((1.0 if int(g) in posset else 0.0 for g in topk),
                          dtype=np.float64, count=len(topk))
        for k in ks:
            ap[k].append(_ap_at_k(rel, cp, k))
            rec[k].append(float(rel[:k].sum() / cp))
    out = {f"mAP@{k}": round(float(np.mean(ap[k])), 4) for k in ks}
    out.update({f"R@{k}": round(float(np.mean(rec[k])), 4) for k in ks})
    out["n_queries_scored"] = int(sum(1 for p in positives if len(p)))
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="semart",
                    choices=("semart", "rocov2", "coco", "inquire", "nwpu",
                             "rsicd", "fashion200k", "scicap", "scimmir",
                             "deepeyenet", "skincap", "facad", "goodnews",
                             "recipe1m", "treeoflife"))
    ap.add_argument("--embed-model", default=None,
                    help="encoder id; default = dataset meta or SigLIP2. "
                         "Try google/medsiglip-448 (gated) or "
                         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_"
                         "patch16_224 (needs open_clip_torch).")
    ap.add_argument("--encoder", default=None,
                    help="friendly encoder name (resolved via "
                         "backbones.resolve_encoder), e.g. siglip-so400m-14-384. "
                         "For --dataset inquire this defaults to "
                         f"{INQUIRE_ENCODER} (SigLIP v1, to match the "
                         "pre-computed gallery shards). --embed-model overrides.")
    ap.add_argument("--text-max-tokens", type=int, default=None,
                    help="override the backbone's text context length")
    ap.add_argument("--semart-dir", type=Path, default=semart_dir_from_env())
    ap.add_argument("--cache-dir", type=Path, default=DATA_DIR,
                    help="HF datasets cache (where ROCOv2 parquet lives).")
    ap.add_argument("--arch", choices=("mlp", "linear"), default="mlp")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.05,
                    help="InfoNCE temperature")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --limit-* subsampling (train-size curve)")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="cap #train pairs (quick runs / train-size curve)")
    ap.add_argument("--limit-val", type=int, default=None,
                    help="cap #val pairs used for model selection")
    ap.add_argument("--limit-test", type=int, default=None,
                    help="cap #test pairs (smoke only; default = full gallery)")
    args = ap.parse_args()
    ds = args.dataset

    meta = resolve_meta(ds)
    # Encoder precedence: --embed-model > --encoder > inquire default > meta.
    # INQUIRE MUST use SigLIP v1 (the encoder of its pre-computed gallery shards),
    # not the repo-default SigLIP2 — otherwise queries and gallery live in
    # different spaces and the projection is meaningless.
    if args.embed_model:
        embed_model = args.embed_model
    elif args.encoder:
        embed_model = resolve_encoder(args.encoder)
    elif ds == "inquire":
        embed_model = resolve_encoder(INQUIRE_ENCODER)
    else:
        embed_model = meta["embed_model"]
    mtag = model_tag(embed_model)
    embedder = make_embedder(embed_model, text_max_tokens=args.text_max_tokens)

    common = dict(mtag=mtag, semart_dir=args.semart_dir,
                  cache_dir=args.cache_dir, embed_batch=args.embed_batch,
                  seed=args.seed)
    tr_caps, tr_imgs = get_split_arrays(ds, "train", embedder,
                                        limit=args.limit_train, **common)
    va_caps, va_imgs = get_split_arrays(ds, "val", embedder,
                                        limit=args.limit_val, **common)
    te_caps, te_gallery, te_target = load_test_eval(
        ds, embedder, limit=args.limit_test, **common)
    del embedder
    torch.cuda.empty_cache()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = QueryProjection(tr_caps.shape[1], args.hidden, args.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr = torch.tensor(tr_caps, device=dev)
    Ytr = torch.tensor(tr_imgs, device=dev)
    Xva = torch.tensor(va_caps, device=dev)

    print(f"[projection] dataset={ds} backbone={embed_model} (tag={mtag or 'default'}) "
          f"arch={args.arch} train={len(tr_caps)} val={len(va_caps)} "
          f"test_gallery={len(te_gallery)} queries={len(te_caps)}")
    print(f"[projection] baseline (raw query) on val: "
          f"{json.dumps(recall_table(va_caps, va_imgs, np.arange(len(va_imgs))))}")

    best_r50, best_state = -1.0, None
    n = len(Xtr)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.train_batch):
            idx = perm[i:i + args.train_batch]
            f = model(Xtr[idx])
            # InfoNCE within the batch: own image is the positive, other
            # batch images are negatives (mirrors retrieval).
            logits = f @ Ytr[idx].T / args.temp
            loss = nn.functional.cross_entropy(
                logits, torch.arange(len(idx), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            fva = model(Xva).cpu().numpy()
        val = recall_table(fva, va_imgs, np.arange(len(va_imgs)))
        if val["R@50"] > best_r50:
            best_r50 = val["R@50"]
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        print(f"[projection] ep{ep:02d} loss {tot / n:.4f} "
              f"val R@1 {val['R@1']:.3f} R@50 {val['R@50']:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fte = model(torch.tensor(te_caps, device=dev)).cpu().numpy()

    # INQUIRE is multi-positive (te_target is a list of per-query index arrays);
    # everything else is single-positive (te_target is a 1-D index array).
    if ds == "inquire":
        raw = multipos_table(te_caps, te_gallery, te_target)
        delta = multipos_table(fte, te_gallery, te_target)
    else:
        raw = recall_table(te_caps, te_gallery, te_target)
        delta = recall_table(fte, te_gallery, te_target)
    refs = REFERENCES.get(ds, {}) if not mtag else {}
    report = {
        "dataset": ds, "embed_model": embed_model, "arch": args.arch,
        "hidden": args.hidden, "epochs": args.epochs,
        "n_train": int(len(tr_caps)), "n_val": int(len(va_caps)),
        "gallery_size": int(len(te_gallery)),
        "n_test_queries": int(len(te_caps)),
        "raw_query": raw, "projected_query": delta, **refs,
    }
    if ds == "inquire":            # cross-encoder caveat travels with the numbers
        report = {"paper_disclaimer": PAPER_DISCLAIMER, **report}
    # Gain vs raw, and fraction of the LLM-rewrite gain recovered (if known).
    gain = delta["R@50"] - raw["R@50"]
    report["projection_gain_R@50"] = round(gain, 4)
    if ds == "inquire":                              # headline metric for INQUIRE
        report["projection_gain_mAP@50"] = round(
            delta["mAP@50"] - raw["mAP@50"], 4)
    rw_ref = refs.get("reference_llm_rewrite_R@50")
    if rw_ref is not None and rw_ref > raw["R@50"]:
        report["frac_of_rewrite_gain_recovered_R@50"] = round(
            gain / (rw_ref - raw["R@50"]), 4)

    parts = [p for p in (mtag, args.arch) if p]
    tag = "_".join(parts)
    if args.limit_train:
        tag += f"_tr{args.limit_train}"
    if args.seed != 0:
        tag += f"_s{args.seed}"
    if args.epochs != 30:
        tag += f"_ep{args.epochs}"
    with open(out_path(f"projection_report_{tag}.json", ds), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    torch.save(best_state, out_path(f"projection_{tag}.pt", ds))
    print(json.dumps(report, indent=2))
    print(f"[projection] model -> projection_{tag}.pt ; report -> "
          f"projection_report_{tag}.json  (under out/{ds}/)")


if __name__ == "__main__":
    main()
