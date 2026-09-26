"""Dataset-agnostic split loader for the register-gap projection (step8/9).

The learned-projection experiment needs, for each split, a set of
(query-register text, its own gallery image) pairs — that's the free,
gallery-specific supervision that teaches the projection. This module
returns those pairs uniformly across datasets so step8/step9 don't need to
know how any one dataset stores its data:

    load_split_pairs(dataset, split) -> (texts, image_paths)

    texts        list[str]    the i-th query (curator prose / clinical
                              caption / …) whose single ground-truth image
                              is the i-th gallery image.
    image_paths  list[Path]   local image file for each item.

The i-th text and i-th image are a matched pair, so target = identity:
the supervision is always (caption_i, image_i). Returning *paths* (not PIL
objects) keeps embedder.embed_images streaming one batch at a time, so a
60k-image split never has to sit in RAM.

Datasets
--------
    semart   reads the split CSVs (semart_<split>.csv); curator DESCRIPTION
             is the query, painting is the image.
    rocov2   eltorio/ROCOv2-radiology on HuggingFace (parquet already cached
             under cache/). PubMed figure caption is the query, radiology
             figure is the image. Each figure is written once to a shared
             jpg cache (cache/rocov2/images/<image_id>.jpg, so the ~9.9k test
             images already on disk are reused).
    coco     yerevann/coco-karpathy (Karpathy train/val/test). One caption per
             image is the query, the image is its own positive (self-gallery,
             like rocov2). Images download once from their COCO url into
             cache/coco/images/. The CONTROL: literal visual captions, so the
             projection's gain should be ≈ 0.
    inquire  SUPERVISION ONLY here (train/val): TreeOfLife-10M-Captions pairs
             over the iNat21 species (splits/treeof_life.py). INQUIRE has no
             native train pairs — its 250-query, multi-positive iNat24 test
             gallery is a *different* dataset and is built in splits/inquire.py
             + adapters.build_inquire, NOT through load_split_pairs("inquire",
             "test"). See splits/inquire.py and splits/treeof_life.py.

Adding a dataset = add one branch that returns (texts, image_paths) (or, for
a multi-positive external-gallery dataset like INQUIRE, a sibling module +
an adapters.build_<ds> + a step8 load_test_eval branch).

NOTE (file layout): this used to be a single module splits.py. It is now a
package so INQUIRE/TreeOfLife can live in their own files (splits/inquire.py,
splits/treeof_life.py) without bloating this one — the public API
(load_split_pairs, stream_rocov2_train, _rocov2_pairs) is unchanged, so every
`from splits import …` site keeps working.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import ROOT, DATA_DIR, semart_dir_from_env

# Datasets whose test gallery == "every item in the test split, retrieved by
# its own caption" (single-positive, target = identity). SemArt is handled
# separately in step8 because its test artifacts (gallery_embeddings.npz +
# queries.json) are already built by step1 and we want byte-identical numbers.
# INQUIRE is NOT here: its test gallery is external (iNat24) and multi-positive.
SELF_GALLERY_DATASETS = (
    "rocov2", "coco", "nwpu",
    # Grid datasets (Jul 2026) — all single-positive self-gallery.
    "rsicd", "fashion200k", "scicap", "scimmir",
    "deepeyenet", "skincap", "facad", "goodnews", "recipe1m", "treeoflife",
)


def _norm_split(dataset: str, split: str) -> str:
    """Map our internal split names onto each dataset's own naming."""
    if dataset in ("rocov2", "coco"):
        # HF split is spelled "validation".
        return {"val": "validation", "valid": "validation"}.get(split, split)
    if dataset == "semart":
        return {"validation": "val"}.get(split, split)
    if dataset == "nwpu":
        # NWPU splits are spelled "train" / "val" / "test".
        return {"validation": "val", "valid": "val"}.get(split, split)
    if dataset == "rsicd":
        # RSICD native HF split for validation is spelled "valid".
        return {"val": "valid", "validation": "valid"}.get(split, split)
    # Grid datasets that own their split logic (stratified or native-in-loader)
    # keep our internal names train/val/test unchanged.
    return split


def _subsample(texts: list[str], paths: list[Path], limit: int | None,
               seed: int) -> tuple[list[str], list[Path]]:
    if limit is None or limit >= len(texts):
        return texts, paths
    idx = np.random.default_rng(seed).permutation(len(texts))[:limit]
    idx.sort()
    return [texts[i] for i in idx], [paths[i] for i in idx]


# ── SemArt ─────────────────────────────────────────────────────────────────

def _semart_pairs(split: str, semart_dir: Path) -> tuple[list[str], list[Path]]:
    from semart_data import image_path, load_manifest

    recs = [r for r in load_manifest(semart_dir, split) if r.description]
    texts = [r.description for r in recs]
    paths = [image_path(semart_dir, r) for r in recs]
    return texts, paths


# ── NWPU-Captions (aerial/satellite; single-positive self-gallery) ───────────

def _nwpu_pairs(split: str, json_path=None, images_dir=None
                ) -> tuple[list[str], list[Path]]:
    """(longest-caption, image_path) pairs for one NWPU split. Images are
    auto-extracted from NWPU_images.tar.gz on first use; text_i's ground-truth
    image is image_i (single-positive self-gallery, like rocov2)."""
    from nwpu_data import (ensure_images_extracted, image_path,
                           load_nwpu_manifest, nwpu_images_dir_from_env,
                           nwpu_json_path, nwpu_tar_path)
    from config import NWPU_IMAGES_TAR

    jp = json_path or nwpu_json_path()
    idir = images_dir or nwpu_images_dir_from_env()
    ensure_images_extracted(idir, NWPU_IMAGES_TAR)

    s = {"validation": "val", "valid": "val"}.get(split, split)
    recs = load_nwpu_manifest(jp, split=s)
    texts, paths = [], []
    n_missing = 0
    for r in recs:
        p = image_path(idir, r)
        if not p.exists():
            n_missing += 1
            continue
        texts.append(r.caption)
        paths.append(p)
    if n_missing:
        print(f"[nwpu] WARNING: skipped {n_missing} records with missing image "
              f"files in {idir} (archive may be incomplete)")
    return texts, paths


# ── ROCOv2 ─────────────────────────────────────────────────────────────────

def _safe_id(image_id: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in image_id)


def _rocov2_pairs(split: str, cache_dir: Path,
                  max_load_px: int = 512) -> tuple[list[str], list[Path]]:
    """Caption→image pairs for one ROCOv2 split, caching each figure to a
    local jpg (shared with benchmarks/rocov2.py). No-caption rows are dropped,
    so every gallery image has exactly one query and vice-versa."""
    from datasets import load_dataset

    image_dir = cache_dir / "rocov2" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[splits] loading ROCOv2 split={split!r} from HF cache "
          f"({cache_dir})…", flush=True)
    ds = load_dataset("eltorio/ROCOv2-radiology", split=split,
                      cache_dir=str(cache_dir))

    texts: list[str] = []
    paths: list[Path] = []
    n_no_cap = 0
    n = len(ds)
    for i, row in enumerate(ds):
        caption = str(row.get("caption", "") or "").strip()
        if not caption:
            n_no_cap += 1
            continue
        image_id = str(row["image_id"])
        local = image_dir / f"{_safe_id(image_id)}.jpg"
        if not local.exists():
            img = row["image"].convert("RGB")
            img.thumbnail((max_load_px, max_load_px))
            img.save(local, format="JPEG", quality=90)
        texts.append(caption)
        paths.append(local)
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"\r[splits] rocov2 {split}: materialized "
                  f"{len(paths)}/{i + 1} (cache {image_dir.name})",
                  end="", flush=True)
    print()
    if n_no_cap:
        print(f"[splits] rocov2 {split}: dropped {n_no_cap} figures with no "
              f"caption; {len(paths)} usable pairs")
    return texts, paths


def stream_rocov2_train(n: int, cache_dir: Path, max_load_px: int = 512
                        ) -> tuple[list[str], list[Path]]:
    """Stream the first n ROCOv2 TRAIN (caption, image_path) pairs WITHOUT
    downloading the whole train split — for the few-shot card pool on Colab
    (the few-shot M only needs ~k examples). Uses HF streaming + islice."""
    import itertools

    from datasets import load_dataset

    image_dir = cache_dir / "rocov2" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True)
    texts: list[str] = []
    paths: list[Path] = []
    for row in itertools.islice(ds, n * 3):          # over-fetch to skip empties
        cap = str(row.get("caption", "") or "").strip()
        if not cap:
            continue
        local = image_dir / f"{_safe_id(str(row['image_id']))}.jpg"
        if not local.exists():
            img = row["image"].convert("RGB")
            img.thumbnail((max_load_px, max_load_px))
            img.save(local, format="JPEG", quality=90)
        texts.append(cap)
        paths.append(local)
        if len(texts) >= n:
            break
    print(f"[splits] streamed {len(texts)} ROCOv2 train pairs (no full download)")
    return texts, paths


# ── COCO (Karpathy splits; the control — projection gain ≈ 0) ────────────────

def _coco_pairs(split: str, cache_dir: Path, captions_per_image: int = 1,
                max_items: int | None = None) -> tuple[list[str], list[Path]]:
    """(caption, image_path) pairs for one COCO-Karpathy split, mirroring the
    ROCOv2 self-gallery contract: one query per gallery image
    (captions_per_image=1), so text_i's ground-truth image is image_i. Images
    download once from their COCO url into a shared jpg cache
    (cache/coco/images/<filename>). Rows with no caption or unresolvable image
    are dropped, so texts[i] <-> paths[i] stay aligned.

    max_items early-stops after that many pairs (first-N). The Karpathy train
    split is ~113k images, so for the control we cap rather than download them
    all; COCO has no prior cached run to keep byte-identical, so first-N is fine.

    COCO is the paper's CONTROL: the queries are already literal visual
    descriptions (no curator/clinical register gap), so the learned projection
    is expected to add ~0 — that ~0 gain IS the result."""
    import urllib.request

    from datasets import load_dataset

    image_dir = cache_dir / "coco" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[splits] loading COCO-Karpathy split={split!r}"
          + (f" (cap {max_items})" if max_items else "") + "…", flush=True)
    ds = load_dataset("yerevann/coco-karpathy", split=split)

    texts: list[str] = []
    paths: list[Path] = []
    n_skip = 0
    n = len(ds)
    for i, row in enumerate(ds):
        if max_items is not None and len(texts) >= max_items:
            break
        fname = (row.get("filename") or row.get("file_name")
                 or Path(str(row.get("url") or row.get("coco_url") or "")).name)
        url = row.get("url") or row.get("coco_url")
        if not fname or not url:
            n_skip += 1
            continue
        local = image_dir / fname
        if not local.exists():
            try:
                tmp = local.with_suffix(local.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(local)
            except Exception:  # noqa: BLE001 — network/url miss; drop the row
                n_skip += 1
                continue
        caps = row.get("sentences") or row.get("captions") or [row.get("caption")]
        if isinstance(caps, str):
            caps = [caps]
        caps = [c["raw"] if isinstance(c, dict) and "raw" in c else str(c)
                for c in caps if c]
        caps = [c.strip() for c in caps if c and str(c).strip()][:captions_per_image]
        if not caps:
            n_skip += 1
            continue
        for cap in caps:
            texts.append(cap)
            paths.append(local)
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"\r[splits] coco {split}: materialized {len(paths)}/{i + 1}",
                  end="", flush=True)
    print()
    if n_skip:
        print(f"[splits] coco {split}: dropped {n_skip} rows (no caption/url or "
              f"download failed); {len(paths)} usable pairs")
    return texts, paths


# ── RSICD (remote sensing; native splits, 5 captions -> longest) ─────────────

def _rsicd_pairs(split: str, cache_dir: Path) -> tuple[list[str], list[Path]]:
    """(longest-caption, image_path) pairs for one RSICD split.

    arampacha/rsicd ships native train/valid/test with `captions` (a list of 5
    human scene descriptions) and an embedded `image`. We take the LONGEST
    caption per image (NWPU precedent — the longest tends to carry the most
    spatial detail) so it's single-positive self-gallery like NWPU/ROCOv2.
    Native splits are already the right scale (8.7k/1.1k/1.1k), so no resplit.
    """
    from datasets import load_dataset

    image_dir = cache_dir / "rsicd" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"[splits] loading RSICD split={split!r} from HF…", flush=True)
    ds = load_dataset("arampacha/rsicd", split=split)

    texts, paths = [], []
    n_skip = 0
    for i, row in enumerate(ds):
        caps = row.get("captions") or []
        if isinstance(caps, str):
            caps = [caps]
        caps = [str(c).strip() for c in caps if c and str(c).strip()]
        if not caps:
            n_skip += 1
            continue
        text = max(caps, key=len)                     # longest of the 5
        fname = row.get("filename") or f"rsicd_{i}.jpg"
        local = image_dir / f"{_safe_id(Path(fname).name)}.jpg"
        if not local.exists():
            img = row["image"].convert("RGB")
            img.thumbnail((512, 512))
            img.save(local, format="JPEG", quality=90)
        texts.append(text)
        paths.append(local)
    if n_skip:
        print(f"[splits] rsicd {split}: dropped {n_skip} caption-less rows")
    print(f"[splits] rsicd {split}: {len(paths)} pairs")
    return texts, paths


# ── Fashion200k (e-commerce; single pool -> stratified 3-way split) ───────────

def _fashion200k_pairs(split: str, cache_dir: Path, seed: int,
                       limit: int | None = None
                       ) -> tuple[list[str], list[Path]]:
    """(product-description, image_path) pairs for a stratified Fashion200k split.

    Marqo/fashion200k is one undivided pool (~202k). We build an image-disjoint,
    category2-stratified train/val/test (50k/5k/5k) with splits.strata, cached so
    the three calls agree. Query = the literal product `text` (the gap-NO /
    collapse-YES corner: products look alike, text is already literal). rid = row
    index (stable in the parquet, one image per row → image-disjoint)."""
    from datasets import load_dataset

    from config import out_path
    from splits.strata import DEFAULT_SIZES, assign_split

    image_dir = cache_dir / "fashion200k" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Marqo/fashion200k", split="data")

    # Lightweight records (no image decode) for the split assignment. Fashion200k
    # has several images per product (item_ID like "51727804_0", "_1", …), all
    # sharing one product description — so we DEDUP to one image per product
    # (strip the trailing "_N") to keep text_i <-> image_i a distinctive pair.
    meta = ds.select_columns(["text", "category2", "item_ID"])
    records, seen = [], set()
    n_dup = 0
    for i, r in enumerate(meta):
        text = (r["text"] or "").strip()
        if not text:
            continue
        product = str(r.get("item_ID") or i).rsplit("_", 1)[0]
        if product in seen:
            n_dup += 1
            continue
        seen.add(product)
        records.append({"rid": str(i), "text": text,
                        "key": r.get("category2") or "unknown"})
    print(f"[splits] fashion200k: {len(records)} unique products "
          f"(deduped {n_dup} extra product images)")

    assign = assign_split(
        records, key_fn=lambda r: r["key"], rid_fn=lambda r: r["rid"],
        cache_path=out_path(f"fashion200k_split_seed{seed}.json", "fashion200k"),
        sizes=DEFAULT_SIZES, seed=seed)

    by_rid = {r["rid"]: r for r in records}
    rids = assign[split]
    if limit is not None and limit < len(rids):
        idx = np.random.default_rng(seed).permutation(len(rids))[:limit]
        idx.sort()
        rids = [rids[i] for i in idx]
    texts, paths = [], []
    for rid in rids:
        rec = by_rid[rid]
        local = image_dir / f"f200k_{rid}.jpg"
        if not local.exists():
            img = ds[int(rid)]["image"].convert("RGB")
            img.thumbnail((512, 512))
            img.save(local, format="JPEG", quality=90)
        texts.append(rec["text"])
        paths.append(local)
    print(f"[splits] fashion200k {split}: {len(paths)} pairs (stratified by "
          f"category2, seed={seed})")
    return texts, paths


# ── SciCap (scientific figures; native splits, stratified subsample to target) ─

def _scicap_pairs(split: str, cache_dir: Path, seed: int,
                  limit: int | None = None) -> tuple[list[str], list[Path]]:
    """(figure-caption, image_path) pairs for a SciCap split.

    CrowdAILab/scicap is a COCO-style release: image folders + annotation JSON,
    NOT a flat table (its HF viewer is broken). We snapshot the repo, read the
    per-split annotations, and use `caption_no_index` (the caption with the
    "Figure N:" prefix stripped) as the query. Native splits are ~400k, so we
    stratified-subsample each to the target (50k/5k/5k) by `figure_type`.

    NOTE: the exact annotation-file names / image-dir layout are discovered by
    globbing (the release has shifted layouts); if 0 records are found the error
    prints what was on disk so the globs can be adjusted.
    """
    import json as _json
    import os

    from huggingface_hub import snapshot_download

    from splits.strata import DEFAULT_SIZES, stratified_subsample

    # Default: keep the full `caption` INCLUDING the "Figure N:" prefix — it's
    # semantically part of the figure caption and, being near-constant boilerplate,
    # pushes query embeddings closer together (aids the text-collapse story). Set
    # SCICAP_STRIP_INDEX=1 to use `caption_no_index` (prefix removed) instead.
    _primary = "caption_no_index" if os.environ.get("SCICAP_STRIP_INDEX") \
        else "caption"

    root = Path(snapshot_download("CrowdAILab/scicap", repo_type="dataset",
                                  cache_dir=str(cache_dir / "hf")))
    # Canonical annotation file per split: train.json / val.json / public-test.json
    # (NOT train-acl.json = the small ACL-figures subset, NOT *-metadata.json).
    hf_split = {"val": "val", "validation": "val", "test": "public-test"}.get(
        split, split)
    ann_path = next(iter(sorted(root.rglob(f"{hf_split}.json"))), None)
    if ann_path is None:
        cands = [p for p in root.rglob("*.json")
                 if p.stem.lower().startswith(hf_split)
                 and "metadata" not in p.stem.lower()
                 and "acl" not in p.stem.lower()]
        ann_path = cands[0] if cands else None
    if ann_path is None:
        raise FileNotFoundError(
            f"[scicap] no {hf_split}.json under {root}. Found: "
            f"{[p.name for p in root.rglob('*.json')]}")
    print(f"[splits] scicap: reading {ann_path.name}", flush=True)
    ann = _json.loads(ann_path.read_text(encoding="utf-8"))

    images = {im["id"]: im for im in ann.get("images", [])}
    records = []
    for a in ann.get("annotations", []):
        im = images.get(a.get("image_id"))
        if im is None:
            continue
        text = (a.get(_primary) or a.get("caption")
                or a.get("caption_no_index") or "").strip()
        if not text:
            continue
        records.append({"file_name": im.get("file_name"),
                        "text": text,
                        "key": im.get("figure_type") or "unknown"})
    if not records:
        raise RuntimeError(
            f"[scicap] parsed 0 (caption, figure) records from {ann_path.name}. "
            f"Keys seen on an annotation: "
            f"{list(ann.get('annotations', [{}])[0].keys())}; on an image: "
            f"{list(ann.get('images', [{}])[0].keys())}.")

    target = dict(zip(("train", "val", "test"), DEFAULT_SIZES)).get(hf_split, 5000)
    if limit is not None:
        target = min(target, limit)
    keep = stratified_subsample(records, lambda r: r["key"], target, seed=seed)

    # Resolve image files (PNGs on disk once the spanned zip is extracted).
    # file_name may be bare; index all images once by basename for a robust lookup.
    img_index = {p.name: p for p in root.rglob("*.png")}
    if not img_index:
        raise FileNotFoundError(
            "[scicap] no .png images found. SciCap ships images as a MULTI-PART "
            "spanned zip (img-split.zip + img-split.z01..z10, ~22 GB) that is NOT "
            "auto-extracted. Extract it ONCE with 7-Zip (open img-split.zip with "
            f"all .z0* parts present -> Extract Here) under\n  {root}\nthen re-run. "
            "If disk is tight, consider dropping SciCap from the grid.")
    texts, paths, n_missing = [], [], 0
    for i in keep:
        rec = records[i]
        fn = rec["file_name"]
        p = img_index.get(fn) or img_index.get(Path(fn).name) if fn else None
        if p is None:
            n_missing += 1
            continue
        texts.append(rec["text"])
        paths.append(p)
    if n_missing:
        print(f"[splits] scicap {hf_split}: {n_missing} records had no image file "
              f"on disk (skipped)")
    print(f"[splits] scicap {hf_split}: {len(paths)} pairs (stratified subsample "
          f"of ~{len(records)} by figure_type, target {target}, seed={seed})")
    return texts, paths


# ── SciMMIR (scientific figures+tables; parquet, turnkey) — SciCap replacement ─

def _scimmir_pairs(split: str, cache_dir: Path, seed: int,
                   limit: int | None = None) -> tuple[list[str], list[Path]]:
    """(figure/table caption, image) pairs for a stratified SciMMIR split.

    m-a-p/SciMMIR is a clean parquet dataset (image + text + class/super_class/
    sub_class) — no spanned-zip extraction like SciCap. We pool from
    validation+test (~33k, ~3.7 GB; the 59 GB train is skipped) and build an
    image-disjoint, sub_class-stratified 50k/5k/5k split.

    FIGURE SUBSET ONLY by default: SciMMIR's "Table" subset is rendered tables,
    where retrieval is really text-in-image reading, not the visual register-gap
    task — so we keep only super_class=Figure (sub_class = architecture /
    illustration / result, which is also the stratification key). Override with
    SCIMMIR_SUBSET=table|all, or the pooled splits with SCIMMIR_SPLIT."""
    import os

    from datasets import load_dataset

    from config import out_path
    from splits.strata import DEFAULT_SIZES, assign_split

    src = os.environ.get("SCIMMIR_SPLIT", "validation+test")
    subset = os.environ.get("SCIMMIR_SUBSET", "fig").lower()
    image_dir = cache_dir / "scimmir" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("m-a-p/SciMMIR", split=src)

    meta = ds.select_columns(["text", "sub_class", "super_class"])
    records, sup_seen = [], set()
    for i, r in enumerate(meta):
        text = (r["text"] or "").strip()
        sup = str(r.get("super_class") or "").lower()
        sup_seen.add(sup)
        # SciMMIR spells these 'fig' / 'table'; match either direction so
        # SCIMMIR_SUBSET=fig or =figure both work.
        if subset != "all" and not (sup.startswith(subset)
                                    or subset.startswith(sup)):
            continue
        if text:
            records.append({"rid": str(i), "text": text,
                            "key": r.get("sub_class") or "unknown"})
    if not records:
        raise RuntimeError(
            f"[scimmir] 0 rows after SCIMMIR_SUBSET={subset!r} filter. "
            f"super_class values present: {sorted(sup_seen)}. Set SCIMMIR_SUBSET "
            f"to one of those (or 'all').")
    print(f"[splits] scimmir: {len(records)} rows in subset={subset!r} "
          f"(super_class values seen: {sorted(sup_seen)})")
    assign = assign_split(
        records, key_fn=lambda r: r["key"], rid_fn=lambda r: r["rid"],
        cache_path=out_path(f"scimmir_split_seed{seed}.json", "scimmir"),
        sizes=DEFAULT_SIZES, seed=seed)
    by = {r["rid"]: r for r in records}
    rids = assign[split]
    if limit is not None and limit < len(rids):
        idx = np.random.default_rng(seed).permutation(len(rids))[:limit]
        idx.sort()
        rids = [rids[i] for i in idx]
    texts, paths = [], []
    for rid in rids:
        rec = by[rid]
        local = image_dir / f"scimmir_{rid}.jpg"
        if not local.exists():
            img = ds[int(rid)]["image"].convert("RGB")
            img.thumbnail((512, 512))
            img.save(local, format="JPEG", quality=90)
        texts.append(rec["text"])
        paths.append(local)
    print(f"[splits] scimmir {split}: {len(paths)} pairs (pool={src}, "
          f"stratified by sub_class, seed={seed})")
    return texts, paths


# ── Public API ──────────────────────────────────────────────────────────────

def load_split_pairs(dataset: str, split: str, *,
                     limit: int | None = None, seed: int = 0,
                     semart_dir: Path | None = None,
                     cache_dir: Path | None = None,
                     ) -> tuple[list[str], list[Path]]:
    """Return aligned (texts, image_paths) for a dataset split.

    limit subsamples deterministically (seeded) AFTER loading the manifest —
    use it for quick runs and the train-size curve. The i-th text's
    ground-truth image is always the i-th path.
    """
    s = _norm_split(dataset, split)
    if dataset == "semart":
        texts, paths = _semart_pairs(s, semart_dir or semart_dir_from_env())
    elif dataset == "rocov2":
        texts, paths = _rocov2_pairs(s, cache_dir or (DATA_DIR))
    elif dataset == "nwpu":
        # NWPU self-gallery: longest caption per image; images auto-extracted
        # from NWPU_images.tar.gz. Uses env/default json + images dir.
        texts, paths = _nwpu_pairs(s)
    elif dataset == "coco":
        # COCO is the control with no prior cache to match — cap downloads at the
        # requested limit (first-N) instead of materializing the full ~113k train.
        texts, paths = _coco_pairs(s, cache_dir or (DATA_DIR),
                                   max_items=limit)
        return texts, paths
    # ── Grid datasets (Jul 2026) ────────────────────────────────────────────
    elif dataset == "rsicd":
        # Native splits, longest of 5 captions; small enough to subsample after.
        texts, paths = _rsicd_pairs(s, cache_dir or (DATA_DIR))
        return _subsample(texts, paths, limit, seed)
    elif dataset == "fashion200k":
        return _fashion200k_pairs(s, cache_dir or (DATA_DIR), seed,
                                  limit=limit)
    elif dataset == "scicap":
        return _scicap_pairs(s, cache_dir or (DATA_DIR), seed, limit=limit)
    elif dataset == "scimmir":
        return _scimmir_pairs(s, cache_dir or (DATA_DIR), seed, limit=limit)
    elif dataset == "deepeyenet":
        from splits.grid_local import deepeyenet_pairs
        texts, paths = deepeyenet_pairs(s)          # native splits
        return _subsample(texts, paths, limit, seed)
    elif dataset == "skincap":
        from splits.grid_local import skincap_pairs
        return skincap_pairs(s, seed, limit=limit)
    elif dataset == "facad":
        from splits.grid_local import facad_pairs
        return facad_pairs(s, seed, limit=limit)
    elif dataset == "goodnews":
        from splits.grid_local import goodnews_pairs
        return goodnews_pairs(s, seed, limit=limit)
    elif dataset == "recipe1m":
        from splits.grid_local import recipe1m_pairs
        return recipe1m_pairs(s, seed, limit=limit)
    elif dataset == "treeoflife":
        from splits.grid_local import treeoflife_ds_pairs
        return treeoflife_ds_pairs(s, seed, cache_dir or (DATA_DIR),
                                   limit=limit)
    elif dataset == "inquire":
        # INQUIRE has no native train pairs — supervision is TreeOfLife-10M
        # (caption, image) over the iNat21 species (same 10k species as iNat24,
        # zero image overlap). The single-positive "self-pair" contract holds:
        # text_i's ground-truth image is image_i. The multi-positive INQUIRE
        # *test* gallery is NOT loaded here (see splits/inquire.py).
        if s == "test":
            raise ValueError(
                "load_split_pairs('inquire','test') is undefined: INQUIRE's "
                "test gallery is the external, multi-positive iNat24 set. Use "
                "splits.inquire.load_inquire_eval / adapters.build_inquire "
                "instead (query_projection.load_test_eval routes there "
                "automatically).")
        from splits.treeof_life import load_treeoflife_pairs
        texts, paths = load_treeoflife_pairs(
            s, limit=limit, seed=seed, cache_dir=cache_dir or (DATA_DIR))
        # treeof_life already honours `limit`; skip the generic _subsample so
        # we don't re-truncate (its sampling is over the iNat21 pool, not here).
        return texts, paths
    else:
        raise ValueError(
            f"unknown dataset {dataset!r}; known: semart, rocov2, coco, nwpu, "
            f"inquire (add a branch in splits.load_split_pairs)")
    return _subsample(texts, paths, limit, seed)
