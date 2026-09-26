"""Write store cells out in the legacy layout the paper-build scripts (paper/ablations, paper/analysis) read.

    <store>/<collection>/<encoder>.npz            ->   <legacy>/<collection>/{train,val,test}_embs[_<mtag>].npz  (caps, imgs)

<legacy> defaults to <store>/legacy (paper/iclr_roots.legacy_cache_root(); override with $ICLR_CACHE_ROOT). mtag is the
encoder's legacy_mtag in configs/encoders.json ('default' = no suffix). Cells not yet in the store are fetched from the
release first (ltg.cache.load_cell), so `python scripts/export_legacy_cache.py --cells skincap:siglip2-so400m-16-384` is
all a fresh machine needs before running any paper/ablations script on that cell. The grid trained a few cells on a
--limit-train subsample (paper/results/master_arms.csv, n_train); the ablation scripts take the same --limit-train flag,
so the export always writes the full split.

usage:  python scripts/export_legacy_cache.py [--cells collection:encoder ...] [--out DIR] [--force]
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ltg.cache import list_cells, load_cell  # noqa: E402

ENC = json.load(open(REPO / "configs" / "encoders.json", encoding="utf-8"))
KEY_TO_MTAG = {k: v.get("legacy_mtag", k) for k, v in ENC.items() if isinstance(v, dict)}


def legacy_root() -> Path:
    spec = ilu.spec_from_file_location("iclr_roots", REPO / "paper" / "iclr_roots.py")
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.legacy_cache_root()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", help="collection:encoder (default: every cell in the store)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = a.out or legacy_root()
    cells = [tuple(c.split(":")) for c in a.cells] if a.cells else list_cells()
    for coll, enc in cells:
        mtag = KEY_TO_MTAG.get(enc, enc)
        sfx = "" if mtag in ("default", "") else f"_{mtag}"
        d = out / coll; d.mkdir(parents=True, exist_ok=True)
        targets = {s: d / f"{s}_embs{sfx}.npz" for s in ("train", "val", "test")}
        if all(t.exists() for t in targets.values()) and not a.force:
            print(f"[export] {coll:12s} {enc:24s} exists, skip"); continue
        c = load_cell(coll, enc)
        for s, t in targets.items():
            txt, img = c.arrays(s)
            np.savez_compressed(t, caps=txt.astype(np.float32), imgs=img.astype(np.float32))
        print(f"[export] {coll:12s} {enc:24s} -> {d}/{{train,val,test}}_embs{sfx}.npz  n={c.sizes()}")
    print(f"[export] legacy cache: {out}")


if __name__ == "__main__":
    main()
