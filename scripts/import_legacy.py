"""Import the grid pipeline's iclr2027 / indexability caches ({train,val,test}_embs[_<mtag>][_n<limit>].npz) into the store.

Resolution per split mirrors iclr2027/ablations/symmetric_projection.load_split exactly (the file the grid numbers came
from): the exact full-split file first, else the largest _n<limit> subset; the default tag never swallows another tag's
file. Pair ids come from <ds>_split_seed0.json when it exists (SkinCAP, FACAD, Fashion200k, SciMMIR, GoodNews, TreeOfLife);
captions are not in the legacy caches and are left empty (meta.has_captions = false) until `ltg embed` rebuilds the cell
from pairs.jsonl. Encoders are renamed from legacy mtags to the keys in configs/encoders.json.

usage:
    python scripts/import_legacy.py --legacy ../indexability/out [--extra ../iclr2027/results] [--datasets skincap rsicd] [--encoders default medsiglip] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltg.cache import cache_root, cell_path, save_cell  # noqa: E402

ENC = json.load(open(Path(__file__).resolve().parents[1] / "configs" / "encoders.json", encoding="utf-8"))
MTAG_TO_KEY = {v["legacy_mtag"]: k for k, v in ENC.items() if isinstance(v, dict) and v.get("legacy_mtag")}
SPLITS = ("train", "val", "test")


def resolve(root: Path, ds: str, split: str, mtag: str) -> Path | None:
    tag = f"_{mtag}" if mtag and mtag != "default" else ""
    exact = root / ds / f"{split}_embs{tag}.npz"
    if exact.exists():
        return exact
    cands = sorted(glob.glob(str(root / ds / f"{split}_embs{tag}_n*.npz")), key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
    if not tag:
        cands = [p for p in cands if Path(p).name.split("_embs")[1].startswith("_n")]
    return Path(cands[0]) if cands else None


def legacy_cells(roots: list[Path]) -> dict[tuple[str, str], Path]:
    found = {}
    for root in roots:
        for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in ds_dir.glob("test_embs*.npz"):
                tag = re.sub(r"^test_embs_?", "", f.stem)
                tag = re.sub(r"_n\d+$", "", tag) or "default"
                found.setdefault((ds_dir.name, tag), root)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", required=True, type=Path, help="indexability/out (read-only)")
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="more roots with the same layout, e.g. iclr2027/results (LoRA embeddings, imported mtags)")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--encoders", nargs="*", help="legacy mtags or store keys")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    roots = [a.legacy] + list(a.extra)
    cells = legacy_cells(roots)
    want_enc = set(a.encoders or [])
    n_ok = 0
    for (ds, mtag), root in sorted(cells.items()):
        key = MTAG_TO_KEY.get(mtag, mtag)
        if a.datasets and ds not in a.datasets:
            continue
        if want_enc and mtag not in want_enc and key not in want_enc:
            continue
        dst = cell_path(ds, key)
        if dst.exists() and not a.force:
            print(f"[import] {ds:12s} {key:24s} exists, skip"); n_ok += 1; continue
        files = {s: resolve(root, ds, s, mtag) for s in SPLITS}
        if any(v is None for v in files.values()):
            print(f"[import] {ds:12s} {key:24s} MISSING splits {[s for s, v in files.items() if v is None]}"); continue
        man_p = root / ds / f"{ds}_split_seed0.json"
        man = json.load(open(man_p)) if man_p.exists() else None
        if a.dry_run:
            print(f"[import] {ds:12s} {key:24s} <- {[f.name for f in files.values()]}  ids={'manifest' if man else 'index'}"); continue
        t0 = time.time()
        txt, img, split, ids = [], [], [], []
        prov = {}
        for s in SPLITS:
            z = np.load(files[s])
            c, i = z["caps"], z["imgs"]
            assert len(c) == len(i), files[s]
            txt.append(c); img.append(i); split += [s] * len(c)
            mids = man.get(s) if man else None
            if mids is not None and len(mids) == len(c):
                ids += [str(x) for x in mids]
            else:
                if mids is not None:
                    print(f"[import]   {ds}/{s}: manifest has {len(mids)} ids, cache {len(c)} rows -> index ids")
                ids += [f"{s}_{j}" for j in range(len(c))]
            prov[s] = {"file": str(files[s]), "n": int(len(c)), "mtime": time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(files[s])))}
        txt, img = np.concatenate(txt), np.concatenate(img)
        info = ENC.get(key, {})
        if info.get("dim") and info["dim"] != txt.shape[1]:
            print(f"[import]   WARNING {ds} {key}: dim {txt.shape[1]} != encoders.json {info['dim']}")
        meta = {"encoder_id": info.get("id", mtag), "legacy_mtag": mtag, "source": "iclr2027/indexability legacy cache",
                "split_rule": "seed-0 manifest" if man else "cache order (no manifest)", "provenance": prov,
                "has_captions": False, "note": "captions absent in legacy caches; rebuild with `ltg embed` from pairs.jsonl to fill them"}
        p = save_cell(ds, key, txt, img, split, caption=None, ids=ids, meta=meta)
        n_ok += 1
        print(f"[import] {ds:12s} {key:24s} -> {p.relative_to(cache_root())}  n={len(split)} d={txt.shape[1]}  {time.time() - t0:.0f}s")
    print(f"[import] {n_ok} cells in {cache_root()}")


if __name__ == "__main__":
    main()
