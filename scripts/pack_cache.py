"""Prepare the embedding cache for anonymous hosting: one archive per cell (they are already one file each), a manifest with
sizes and sha256, and the download layout `fetch_cache.py` expects.

    python scripts/pack_cache.py --out ../learn-the-gap-cache-release [--cells skincap:siglip2-so400m-16-384 ...]

Writes <out>/<collection>/<encoder>.npz (copies; patch stores and phrase sets are included when present) and
<out>/cache_manifest.json = {"cells": [{"collection","encoder","path","bytes","sha256"}], "total_bytes", "note"}.
Upload the <out> folder as-is (OSF project storage, a Hugging Face dataset repo, or any static host); then set
"base_url" in configs/cache_release.json to where <out> is served and commit the manifest as configs/cache_manifest.json.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltg.cache import cache_root, list_cells, cell_path, patches_path

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--cells", nargs="*"); ap.add_argument("--no-copy", action="store_true", help="manifest only")
a = ap.parse_args()
cells = [tuple(c.split(":")) for c in a.cells] if a.cells else list_cells()
a.out.mkdir(parents=True, exist_ok=True)
rows, total = [], 0
for coll, enc in cells:
    srcs = [cell_path(coll, enc)]
    pp = patches_path(coll, enc)
    if pp.exists(): srcs.append(pp)
    srcs += sorted(cell_path(coll, enc).parent.glob(f"{enc}.phrases*.npz"))
    for src in srcs:
        if not src.exists(): continue
        rel = f"{coll}/{src.name}"
        dst = a.out / rel
        if not a.no_copy:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""): h.update(chunk)
        n = src.stat().st_size; total += n
        rows.append({"collection": coll, "encoder": enc, "path": rel, "bytes": n, "sha256": h.hexdigest()})
        print(f"[pack] {rel:60s} {n/2**20:8.1f} MB")
man = {"cells": rows, "total_bytes": total, "n_files": len(rows), "note": "one npz per (collection, encoder) cell, keys txt/img/split/caption/id/meta; see ltg/cache.py"}
json.dump(man, open(a.out / "cache_manifest.json", "w"), indent=1)
json.dump(man, open(Path(__file__).resolve().parents[1] / "configs" / "cache_manifest.json", "w"), indent=1)
print(f"[pack] {len(rows)} files, {total/2**30:.2f} GB -> {a.out}/cache_manifest.json (+ configs/cache_manifest.json)")
