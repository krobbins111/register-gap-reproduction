"""Download the released embedding cache into $LTG_CACHE (or --dest) and verify every file's sha256.

    python scripts/fetch_cache.py [--dest ../learn-the-gap-cache] [--cells skincap:siglip2-so400m-16-384 ...] [--base-url URL]

The base URL comes from configs/cache_release.json ({"base_url": "..."}); each file is fetched at <base_url>/<path> from
configs/cache_manifest.json. Files already present with the right hash are skipped, so the script is resumable. Needs only the
standard library (urllib); set HTTPS_PROXY if your network requires it.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltg.cache import cache_root
REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--dest", type=Path); ap.add_argument("--cells", nargs="*"); ap.add_argument("--base-url"); ap.add_argument("--verify-only", action="store_true")
a = ap.parse_args()
man = json.load(open(REPO / "configs" / "cache_manifest.json"))
rel = json.load(open(REPO / "configs" / "cache_release.json")) if (REPO / "configs" / "cache_release.json").exists() else {}
base = (a.base_url or rel.get("base_url") or "").rstrip("/")
dest = a.dest or cache_root()
want = set(a.cells or [])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()
ok = bad = 0
for row in man["cells"]:
    if want and f"{row['collection']}:{row['encoder']}" not in want: continue
    p = dest / row["path"]
    if p.exists() and sha(p) == row["sha256"]:
        ok += 1; print(f"[fetch] ok      {row['path']}"); continue
    if a.verify_only or not base:
        bad += 1; print(f"[fetch] MISSING {row['path']}" + ("" if base else "  (no base_url: set configs/cache_release.json or --base-url)")); continue
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base}/{row['path']}"; tmp = p.with_suffix(".part")
    print(f"[fetch] {row['bytes']/2**20:7.1f} MB  {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 22), b""): f.write(chunk)
    if sha(tmp) != row["sha256"]:
        tmp.unlink(); bad += 1; print(f"[fetch] HASH MISMATCH {row['path']}"); continue
    os.replace(tmp, p); ok += 1
print(f"[fetch] {ok} ok, {bad} missing/bad -> {dest}")
sys.exit(1 if bad else 0)
