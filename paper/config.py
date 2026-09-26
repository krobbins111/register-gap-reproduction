"""ICLR code-folder config: separates the (read-only) embedding caches from
this folder's (clean) results.

Reads resolve against CACHE_ROOT   (default: <ltg store>/legacy, see iclr_roots.py)
Writes resolve against RESULTS_ROOT (default: ./results)

Rule: a name containing a glob '*', or a file that already exists under the
cache, resolves to the cache; everything else (new reports, checkpoints)
resolves to results. Override with env vars ICLR_CACHE_ROOT / ICLR_RESULTS_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("iclr_roots", HERE / "iclr_roots.py")
_roots = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_roots)
CACHE_ROOT = _roots.legacy_cache_root()          # $ICLR_CACHE_ROOT, else <ltg store>/legacy (scripts/export_legacy_cache.py)
RESULTS_ROOT = Path(os.environ.get("ICLR_RESULTS_ROOT", HERE / "results"))


def out_path(name: str, dataset: str) -> Path:
    cache = CACHE_ROOT / dataset / name
    if "*" in name or cache.exists():
        return cache
    p = RESULTS_ROOT / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p / name
