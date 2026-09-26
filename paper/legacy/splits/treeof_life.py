"""TreeOfLife-10M-Captions training-pair loader for INQUIRE.

INQUIRE (iNat24) ships no training pairs, so the register-gap projection is
taught from a *separate* gallery-shaped source: BioCAP's TreeOfLife-10M-Captions
(imageomics/TreeOfLife-10M-Captions). Each pair is (trait-rich synthetic
caption, its image) — the same (query-register text, own image) contract every
other split obeys, so step8 treats INQUIRE train/val exactly like SemArt/ROCOv2
(single-positive, target = identity, InfoNCE in-batch negatives).

ENCODER (matters): these pairs are embedded with **SigLIP v1 SO400M-14@384**
(open_clip `ViT-SO400M-14-SigLIP-384` / webli), NOT SigLIP2 — because the INQUIRE
gallery embeddings we retrieve against are INQUIRE's own pre-computed v1 shards,
and query/training/gallery must share one space. This loader only returns
(text, image_path); the encoding happens in step8 (which defaults --encoder to
config.INQUIRE_ENCODER for --dataset inquire). config.INQUIRE_ENCODER is the
single source of truth for that default.

Leakage boundary
----------------
We filter to the **iNat21**-sourced rows of TreeOfLife-10M. iNat21 covers the
SAME 10k species as iNat24 (so the projection sees the right register/manifold)
but is a DISJOINT image set (iNat24 = observations 2021-2023 exported 2023-12-30;
iNat21 predates it) — so training never touches a test gallery image. The iNat21
filter is `catalog.csv.inat21_filename` being non-null.

Two HuggingFace pieces, joined by UUID
--------------------------------------
  captions : imageomics/TreeOfLife-10M-Captions  -> uuid_caption_description.parquet
             columns (uuid, caption, description); 9.56M rows; ~1.8 GB; loads via
             datasets.load_dataset directly. CAPTIONS ONLY — no pixels.
  catalog  : imageomics/TreeOfLife-10M  -> metadata/catalog.csv
             columns include (treeoflife_id, split, inat21_filename, inat21_cls_num,
             kingdom..species, common). Maps uuid==treeoflife_id -> iNat21 file.

Images are NOT in either HF dataset. You resolve them locally (see below). The
sampled (caption, relpath) pairs are cached to out/inquire/treeoflife_pairs.jsonl
so train/val are consistent across calls and re-runs are instant.

‼ AUTHOR — IMAGE SOURCE DECISION (one of these; set via env or CLI):
  (A) iNat21 tarball  [recommended, smallest]: download the iNat21 train images
      (CVPR-FGVC iNaturalist 2021) and point INAT21_DIR at the root that contains
      the per-class folders. Image path = INAT21_DIR / <inat21_filename>.
  (B) TreeOfLife-10M webdataset shards: extract some shards so you have
      <uuid>.jpg files, point TOL_IMAGE_DIR at that root. Image path =
      TOL_IMAGE_DIR / <uuid>.jpg. (Whole set is ~10M imgs in ~30 GB shards.)
You only need ~max-pairs images (default 100k), so you do NOT need the whole
shard set — fetch enough to cover the sample.

Env vars (all overridable; defaults under cache/treeoflife/):
  TOL_CAPTIONS  parquet path OR "imageomics/TreeOfLife-10M-Captions" (HF id)
  TOL_CATALOG   metadata/catalog.csv path
  INAT21_DIR    root of extracted iNat21 train images       (image source A)
  TOL_IMAGE_DIR root of extracted TreeOfLife <uuid>.jpg      (image source B)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from config import ROOT

# Disjoint, deterministic train/val partition of the sampled iNat21 pool. We
# build one big seeded ordering and hand split "train" the front and "val" a
# region far enough back that no plausible --limit-train overlaps it.
_VAL_OFFSET = 5_000_000          # > any plausible train size; pool is < this too
_POOL_CAP = 600_000              # cap rows we resolve+cache (train+val headroom)


def _tol_dir(cache_dir: Path) -> Path:
    d = cache_dir / "treeoflife"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _catalog_path(cache_dir: Path) -> Path:
    return Path(os.environ.get("TOL_CATALOG",
                               str(_tol_dir(cache_dir) / "catalog.csv")))


def _captions_source() -> str:
    return os.environ.get("TOL_CAPTIONS", "imageomics/TreeOfLife-10M-Captions")


def _resolve_image_relref(uuid: str, inat21_filename: str) -> str | None:
    """Local image path for a pair, honouring the image-source env vars.
    Returns a string path that exists, or None (pair is then skipped).

    Source A (iNat21 tarball)      : INAT21_DIR / inat21_filename
    Source B (TreeOfLife shards)   : TOL_IMAGE_DIR / <uuid>.jpg
    """
    inat21_dir = os.environ.get("INAT21_DIR")
    if inat21_dir and inat21_filename:
        p = Path(inat21_dir) / inat21_filename
        if p.exists():
            return str(p)
    tol_img = os.environ.get("TOL_IMAGE_DIR")
    if tol_img:
        p = Path(tol_img) / f"{uuid}.jpg"
        if p.exists():
            return str(p)
    return None


def _scan_inat21_catalog(catalog_path: Path) -> list[tuple[str, str]]:
    """Stream catalog.csv, return [(uuid, inat21_filename)] for iNat21 rows
    (inat21_filename non-empty). Stdlib csv — no pandas dep, ~10M rows is a
    one-time ~30-60 s scan."""
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"[treeoflife] catalog not found: {catalog_path}\n"
            f"  Download metadata/catalog.csv from "
            f"https://huggingface.co/datasets/imageomics/TreeOfLife-10M and "
            f"either place it there or set TOL_CATALOG.")
    rows: list[tuple[str, str]] = []
    # catalog.csv can have many columns; we only need the id + the iNat21 file.
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with open(catalog_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        id_col = "treeoflife_id" if "treeoflife_id" in (reader.fieldnames or []) \
            else ("uuid" if "uuid" in (reader.fieldnames or []) else None)
        if id_col is None or "inat21_filename" not in (reader.fieldnames or []):
            raise KeyError(
                f"[treeoflife] catalog.csv missing expected columns; got "
                f"{reader.fieldnames}. Need a uuid/treeoflife_id col and "
                f"inat21_filename.")
        for r in reader:
            fn = (r.get("inat21_filename") or "").strip()
            if fn:
                rows.append((r[id_col].strip(), fn))
    print(f"[treeoflife] catalog: {len(rows)} iNat21-sourced rows "
          f"(of the TreeOfLife-10M pool)")
    return rows


def _build_pair_manifest(cache_dir: Path, seed: int) -> Path:
    """Resolve up to _POOL_CAP (caption, image_path) iNat21 pairs once and
    cache them as out/inquire/treeoflife_pairs.jsonl. Deterministic given seed.
    Returns the manifest path. Cheap to re-call (returns immediately if cached).
    """
    from config import out_path  # local import: keeps module import side-effect free

    manifest = out_path(f"treeoflife_pairs_seed{seed}.jsonl", "inquire")
    if manifest.exists():
        return manifest

    catalog = _scan_inat21_catalog(_catalog_path(cache_dir))
    # Seeded order over the whole iNat21 pool, then take a capped working set.
    order = np.random.default_rng(seed).permutation(len(catalog))
    take = order[:min(_POOL_CAP, len(catalog))]
    want = {catalog[i][0]: catalog[i][1] for i in take}     # uuid -> inat21_filename
    want_order = [catalog[i][0] for i in take]              # preserve seeded order
    print(f"[treeoflife] resolving captions+images for {len(want)} sampled uuids…")

    # Pull captions for the sampled uuids only (avoid holding all 9.56M in RAM).
    caps = _load_captions_for(set(want), cache_dir)

    n_written = 0
    n_no_cap = 0
    n_no_img = 0
    with open(manifest, "w", encoding="utf-8") as out:
        for uuid in want_order:
            cap = caps.get(uuid)
            if not cap:
                n_no_cap += 1
                continue
            ref = _resolve_image_relref(uuid, want[uuid])
            if ref is None:
                n_no_img += 1
                continue
            out.write(json.dumps({"uuid": uuid, "text": cap, "path": ref}) + "\n")
            n_written += 1
    print(f"[treeoflife] manifest: wrote {n_written} pairs "
          f"({n_no_cap} no-caption, {n_no_img} image-not-on-disk) -> "
          f"{manifest.name}")
    if n_written == 0:
        raise RuntimeError(
            "[treeoflife] 0 resolvable pairs. Almost certainly the images "
            "aren't on disk: set INAT21_DIR (iNat21 tarball) or TOL_IMAGE_DIR "
            "(extracted TreeOfLife <uuid>.jpg shards). See module docstring.")
    return manifest


def _load_captions_for(uuids: set[str], cache_dir: Path) -> dict[str, str]:
    """uuid -> caption for the requested uuids only. Reads the captions parquet
    (HF id or local path) once; keeps only rows we need."""
    src = _captions_source()
    caps: dict[str, str] = {}
    local = Path(src)
    if local.exists() and local.suffix == ".parquet":
        # Local parquet via pyarrow (datasets is also fine; pyarrow ships with it).
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(columns=["uuid", "caption"], batch_size=65536):
            d = batch.to_pydict()
            for u, c in zip(d["uuid"], d["caption"]):
                if u in uuids and c:
                    caps[u] = str(c)
    else:
        from datasets import load_dataset

        ds = load_dataset(src, split="train")
        # Iterate once; column names are uuid/caption per the dataset card.
        for row in ds:
            u = row.get("uuid")
            if u in uuids:
                c = row.get("caption")
                if c:
                    caps[u] = str(c)
    print(f"[treeoflife] captions matched {len(caps)}/{len(uuids)} sampled uuids")
    return caps


def load_treeoflife_pairs(split: str, *, limit: int | None = None,
                          seed: int = 0, cache_dir: Path | None = None
                          ) -> tuple[list[str], list[Path]]:
    """(captions, image_paths) for INQUIRE supervision.

    split == "train"  -> the front of the seeded iNat21 pool.
    split == "val"    -> a disjoint region (offset _VAL_OFFSET) for model
                         selection (single-positive R@50), as in the other
                         datasets' held-out val.
    `limit` caps the returned count (default 100k for train when None — see
    step8 --limit-train). Pairs are read from the cached manifest, so this is
    fast after the first build.
    """
    cache_dir = cache_dir or (ROOT / "cache")
    manifest = _build_pair_manifest(cache_dir, seed)
    with open(manifest, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    if split == "val":
        # Disjoint tail region; modular so it works even if the pool is small.
        if len(pairs) > _VAL_OFFSET:
            pairs = pairs[_VAL_OFFSET:]
        else:
            cut = int(len(pairs) * 0.9)
            pairs = pairs[cut:]                 # last 10% reserved for val
        default_limit = 5_000
    else:                                       # "train"
        if len(pairs) > _VAL_OFFSET:
            pairs = pairs[:_VAL_OFFSET]
        else:
            cut = int(len(pairs) * 0.9)
            pairs = pairs[:cut]
        default_limit = 100_000                 # the grid pipeline's target train size

    lim = limit if limit is not None else default_limit
    pairs = pairs[:lim]
    texts = [p["text"] for p in pairs]
    paths = [Path(p["path"]) for p in pairs]
    print(f"[treeoflife] split={split}: {len(texts)} pairs "
          f"(limit={lim}, seed={seed})")
    return texts, paths
