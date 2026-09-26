"""Produce the embeddings for one dataset x encoder with the exact pipeline the paper's caches were built with.

Writes the legacy layout that every script in paper/ablations reads,
    <legacy cache>/<dataset>/{train,val,test}_embs[_<mtag>].npz     (caps, imgs: unit-norm float32)
under paper/iclr_roots.legacy_cache_root() (= $ICLR_CACHE_ROOT, else <ltg store>/legacy). The text side is the caption
(longest of five for RSICD, DESCRIPTION for SemArt, ...), the image side the raw image; loaders, split rules and seeds are
in splits/ (seed-0 manifests where the collection has one). Raw datasets are downloaded to $ICLR_DATA_DIR
(default <ltg store>/raw) on first use; the collections that ship as local zips (SkinCAP, FACAD, GoodNews, NWPU, SemArt)
need their folder placed there first — see splits/grid_local.py and splits/__init__.py docstrings for each layout.

    python paper/legacy/embed_cells.py --dataset skincap                       # SigLIP2-so400m/16-384 (mtag '' = default)
    python paper/legacy/embed_cells.py --dataset skincap --encoder medsiglip   # names: backbones.ENCODER_ALIASES / encoders.json id
    python paper/legacy/embed_cells.py --dataset rocov2 --limit-train 20000    # the grid's subsampled ROCOv2 cell

Then `python scripts/import_legacy.py --legacy <legacy cache>` moves the arrays into the store layout ltg/ reads, and
`python scripts/verify_cells.py` checks raw R@10/R@50 against the committed results to 4 decimals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from backbones import make_embedder, model_tag, resolve_encoder  # noqa: E402
from config import EMBED_MODEL, OUT_DIR, DATA_DIR, semart_dir_from_env  # noqa: E402
from query_projection import get_split_arrays, load_test_eval, resolve_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", default=None, help="friendly name (backbones.resolve_encoder) or HF id; default = SigLIP2-so400m/16-384")
    ap.add_argument("--embed-model", default=None, help="raw model id, overrides --encoder")
    ap.add_argument("--text-max-tokens", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--embed-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    meta = resolve_meta(a.dataset)
    model = a.embed_model or (resolve_encoder(a.encoder) if a.encoder else (meta.get("embed_model") or EMBED_MODEL))
    mtag = model_tag(model)
    print(f"[embed] {a.dataset} x {model} (mtag={mtag or 'default'}) -> {OUT_DIR / a.dataset}   raw data: {DATA_DIR}")
    emb = make_embedder(model, text_max_tokens=a.text_max_tokens)
    common = dict(mtag=mtag, semart_dir=semart_dir_from_env(), cache_dir=DATA_DIR, embed_batch=a.embed_batch, seed=a.seed)
    for split, lim in (("train", a.limit_train), ("val", a.limit_val)):
        c, i = get_split_arrays(a.dataset, split, emb, limit=lim, **common)
        print(f"[embed]   {split}: {len(c)} pairs, d={c.shape[1]}")
    q, g, _ = load_test_eval(a.dataset, emb, **common)
    print(f"[embed]   test: {len(q)} queries, {len(g)} gallery")


if __name__ == "__main__":
    main()
