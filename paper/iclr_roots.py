"""Where the paper-build scripts find embeddings: the legacy layout exported from the store.

    <legacy cache>/<dataset>/{train,val,test}_embs[_<mtag>].npz     arrays caps, imgs   (what ablations/*.py read)

<legacy cache> = $ICLR_CACHE_ROOT, else <store>/legacy where <store> is the ltg cache root ($LTG_CACHE, local.json["cache"],
../learn-the-gap-cache, ./cache — same order as ltg/cache.py). scripts/export_legacy_cache.py writes it from the store cells.
Loaded by file path (importlib) from paper/config.py and paper/legacy/config.py so the two modules named `config` never clash.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent


def store_root() -> Path:
    v = os.environ.get("LTG_CACHE")
    if not v:
        p = REPO / "local.json"
        try:
            v = json.load(open(p, encoding="utf-8")).get("cache") if p.exists() else None
        except Exception:
            v = None
    if v:
        return Path(v).expanduser()
    sib = REPO.parent / "learn-the-gap-cache"
    return sib if sib.exists() else REPO / "cache"


def legacy_cache_root() -> Path:
    v = os.environ.get("ICLR_CACHE_ROOT")
    return Path(v).expanduser() if v else store_root() / "legacy"


def _load_by_path(name: str = "iclr_roots"):
    """import this file under a fixed module name from anywhere (no sys.path games)."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(name, Path(__file__).resolve())
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
