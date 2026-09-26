"""Loaders for grid datasets staged locally under splits/<NAME>/ (or an env-var
root). Wired to the ACTUAL download formats grid-pipeline staged Jul 2026.

Contract (identical to the rest of the pipeline): each loader returns aligned
(texts, image_paths) for split in {train, val, test}, single-positive
self-gallery (text_i's ground-truth image is image_i), image-level disjoint.
Datasets with no native split get a deterministic, image-disjoint,
metadata-stratified 50k/5k/5k split via splits.strata (clamped to the pool).
Images are materialized to a jpg cache ONLY for the requested split (so a smoke
run or a small split never unpacks the whole source).

Default data root = this package's sibling dir, e.g. splits/SkinCAP, splits/FACAD,
splits/GoodNews, splits/TOL — overridable via the env vars below.

Staged formats (real):
  skincap    SKINCAP_DIR/skincap_v240623.csv  (id, skincap_file_path,
             caption_zh_polish_en [English caption], disease, …). Images come
             from the HF webdataset joshuachou/SkinCAP (ungated after accepting
             the KAUST terms); we join CSV text <-> HF image and cache the jpgs.
  facad      FACAD_DIR/ECCV_data/<PART>_{CAPTIONS,CATES}.json + WORDMAP.json
             (captions are WORDMAP-encoded int sequences) and
             FACAD_DIR/<PART>_IMAGES-*.hdf5 (images[i] = (3,H,W) uint8), aligned
             by index. PART defaults to TEST (FACAD_PART) — TEST is 101,225
             images (~20 GB); VAL is only ~20k. TRAIN (174 GB) is unused.
  goodnews   GOODNEWS_DIR/captioning_dataset.json {article_id: {images:{ix:cap},
             headline, article}} + resized.tar.gz (members resized/<id>_<ix>.jpg).
             One (caption, image) pair per article-image; needed members are
             extracted from the tar in a single pass.
  treeoflife TOL_DIR/catalog.csv (treeoflife_id + taxonomy) for the stratum/
             dedup; REAL Wikipedia `description` from imageomics/
             TreeOfLife-10M-Captions; images streamed from the HF webdataset
             imageomics/TreeOfLife-10M (bounded — see treeoflife_ds_pairs).
  recipe1m   RECIPE1M_DIR/layer1.json (+ layer2.json) native MIT format; images
             in the nested MIT layout. Query = ingredients. (Awaiting the image
             download — see recipe1m_pairs.)
  deepeyenet DEEPEYENET_DIR/DeepEyeNet_{train,valid,test}.json (skipped for now).

Env vars: SKINCAP_DIR FACAD_DIR (FACAD_PART) GOODNEWS_DIR TOL_DIR RECIPE1M_DIR
DEEPEYENET_DIR.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tarfile
from pathlib import Path

from config import ROOT, DATA_DIR, out_path

_SPLITS_DIR = Path(__file__).resolve().parent


# ── shared helpers ───────────────────────────────────────────────────────────

def _dataset_dir(env: str, name: str) -> Path:
    """Local root for a dataset: $ENV if set, else <data dir>/<name> ($ICLR_DATA_DIR). Must exist."""
    root = Path(os.environ.get(env, str(DATA_DIR / name)))
    if not root.exists():
        raise FileNotFoundError(
            f"[{name}] data root not found: {root}. Set {env} or place the files "
            f"under {DATA_DIR}/{name}/ (see grid_local.py docstring for the layout).")
    return root


def _cache_img_dir(dataset: str) -> Path:
    d = DATA_DIR / dataset / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


def _pick(cols, candidates):
    """First candidate present in cols (case-insensitive), else None."""
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _finish_split(dataset: str, split: str, records: list[dict], *, seed: int,
                  materialize, limit: int | None = None
                  ) -> tuple[list[str], list[Path]]:
    """records = [{"rid","text","key", ...}] (NO path yet). `materialize(rec) ->
    Path | None` is called ONLY for the requested split's records (after the
    split assignment + limit subsample), so we never unpack images we won't use.
    Records whose materialize returns None are dropped."""
    import numpy as np

    from splits.strata import DEFAULT_SIZES, assign_split

    records = [r for r in records if r.get("text")]
    if not records:
        raise RuntimeError(f"[{dataset}] 0 records with text — check the staged "
                           f"metadata (see grid_local.py).")
    assign = assign_split(
        records, key_fn=lambda r: r["key"], rid_fn=lambda r: r["rid"],
        cache_path=out_path(f"{dataset}_split_seed{seed}.json", dataset),
        sizes=DEFAULT_SIZES, seed=seed)
    rids = assign[split]
    if limit is not None and limit < len(rids):
        idx = np.random.default_rng(seed).permutation(len(rids))[:limit]
        idx.sort()
        rids = [rids[i] for i in idx]
    by = {r["rid"]: r for r in records}
    texts, paths, n_miss = [], [], 0
    for rid in rids:
        rec = by.get(rid)
        if rec is None:
            continue
        p = materialize(rec)
        if p is None:
            n_miss += 1
            continue
        texts.append(rec["text"])
        paths.append(Path(p))
    if n_miss:
        print(f"[grid_local] {dataset}/{split}: {n_miss} records had no "
              f"materializable image (skipped)")
    print(f"[grid_local] {dataset}/{split}: {len(paths)} pairs")
    return texts, paths


# ── SkinCAP (dermatology) — CSV text + HF-webdataset images ───────────────────

def skincap_pairs(split: str, seed: int, limit: int | None = None
                  ) -> tuple[list[str], list[Path]]:
    """English caption (caption_zh_polish_en) + disease label from the local CSV,
    joined to images pulled from the HF repo's `skincap/` folder by filename
    (skincap_file_path). Filename join = no ordering ambiguity."""
    from huggingface_hub import snapshot_download

    root = _dataset_dir("SKINCAP_DIR", "SkinCAP")
    csv_path = next((p for p in root.glob("*.csv")), None)
    if csv_path is None:
        raise FileNotFoundError(f"[skincap] no metadata .csv under {root}")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    cols = list(rows[0].keys()) if rows else []
    ccap = _pick(cols, ("caption_zh_polish_en", "caption_en", "caption"))
    cfp = _pick(cols, ("skincap_file_path", "id"))
    cdis = _pick(cols, ("disease", "nine_partition_label", "three_partition_label"))
    if not (ccap and cfp):
        raise KeyError(f"[skincap] caption/file columns not found in {cols}")

    # The HF dataset is an imagefolder with only an `image` column (no filename),
    # so load_dataset can't be joined to the CSV by name. Instead pull the repo's
    # `skincap/` image FOLDER and resolve each image by skincap_file_path — an
    # unambiguous filename join (no ordering risk).
    snap = Path(snapshot_download("joshuachou/SkinCAP", repo_type="dataset",
                                  allow_patterns=["skincap/*"]))
    img_index = {p.name: p for p in snap.rglob("*") if p.is_file()}
    if not img_index:
        raise FileNotFoundError(
            f"[skincap] no image files under the HF 'skincap/' folder ({snap}).")

    img_dir = _cache_img_dir("skincap")
    records, n_miss = [], 0
    for r in rows:
        fp = str(r[cfp]).strip()
        t = (r.get(ccap) or "").strip()
        src = img_index.get(fp) or img_index.get(Path(fp).name)
        if not t or src is None:
            n_miss += 1
            continue
        records.append({"rid": _safe(fp), "text": t, "src": str(src),
                        "key": (r.get(cdis) or "unknown") if cdis else "all"})
    print(f"[skincap] {len(records)} records "
          f"({n_miss} rows without caption or matching image)")

    def materialize(rec):
        from PIL import Image
        p = img_dir / f"{rec['rid']}.jpg"
        if not p.exists():
            Image.open(rec["src"]).convert("RGB").save(p, "JPEG", quality=90)
        return p

    return _finish_split("skincap", split, records, seed=seed, limit=limit,
                         materialize=materialize)


# ── FACAD (fashion captioning) — HDF5 images + WORDMAP-decoded captions ────────

def _facad_decode(seq, inv, specials):
    words = [inv.get(i) for i in seq if i and i in inv]
    return " ".join(w for w in words if w and w not in specials)


def facad_pairs(split: str, seed: int, limit: int | None = None
                ) -> tuple[list[str], list[Path]]:
    """FACAD marketing caption (decoded from WORDMAP-encoded int sequences) with
    its HDF5 image, aligned by index within a partition. PART defaults to TEST
    (env FACAD_PART) — TEST has 101,225 images; set FACAD_PART=VAL for the
    smaller ~20k pool."""
    import h5py
    import numpy as np
    from PIL import Image

    root = _dataset_dir("FACAD_DIR", "FACAD")
    part = os.environ.get("FACAD_PART", "TEST").upper()
    # ECCV_data may sit under a nested extract dir (ECCV_data-<ts>/ECCV_data/…),
    # so locate the JSONs by search rather than a fixed path.
    caps_p = next(iter(sorted(root.rglob(f"{part}_CAPTIONS.json"))), None)
    if caps_p is None:
        raise FileNotFoundError(
            f"[facad] {part}_CAPTIONS.json not found anywhere under {root} — "
            f"unzip ECCV_data-*.zip so the ECCV_data/ JSONs exist.")
    ecc = caps_p.parent
    cate_p = ecc / f"{part}_CATES.json"
    wmap_p = ecc / "WORDMAP.json"
    for p in (cate_p, wmap_p):
        if not p.exists():
            raise FileNotFoundError(
                f"[facad] missing {p.name} next to {caps_p.name} in {ecc}")
    # HDF5 image file for this partition (name varies: TEST_IMAGES.hdf5,
    # VAL_IMAGES-013.hdf5, …); search so a nested/renamed file is still found.
    h5_files = sorted(root.rglob(f"{part}_IMAGES*.hdf5")) or \
        sorted(root.rglob(f"{part}_IMAGES*.h5"))
    if not h5_files:
        raise FileNotFoundError(
            f"[facad] no {part}_IMAGES*.hdf5 under {root}.")

    caps = json.loads(caps_p.read_text(encoding="utf-8"))
    cates = json.loads(cate_p.read_text(encoding="utf-8"))
    wmap = json.loads(wmap_p.read_text(encoding="utf-8"))
    inv = {v: k for k, v in wmap.items()}
    specials = {"<start>", "<end>", "<pad>", "<unk>", "<null>", "<sos>", "<eos>"}

    # FACAD stores several near-duplicate images per product, all sharing ONE
    # caption. Keep one image per unique caption so text_i <-> image_i is a
    # distinctive pair (else InfoNCE gets false negatives and the gallery is
    # padded with duplicate positives).
    n = min(len(caps), len(cates))
    records, seen = [], set()
    n_dup = 0
    for i in range(n):
        text = _facad_decode(caps[i], inv, specials)
        if not text:
            continue
        if text in seen:
            n_dup += 1
            continue
        seen.add(text)
        records.append({"rid": f"{part.lower()}_{i}", "text": text,
                        "key": str(cates[i]), "_i": i})
    print(f"[facad] {len(records)} unique-caption products "
          f"(deduped {n_dup} duplicate-caption images)")

    h5 = h5py.File(str(h5_files[0]), "r")
    dset_key = "images" if "images" in h5 else list(h5.keys())[0]
    dset = h5[dset_key]
    img_dir = _cache_img_dir("facad")

    def materialize(rec):
        i = rec["_i"]
        p = img_dir / f"{rec['rid']}.jpg"
        if not p.exists():
            arr = np.asarray(dset[i])
            if arr.ndim == 3 and arr.shape[0] in (1, 3):     # (C,H,W) -> (H,W,C)
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = arr.astype("uint8")
            Image.fromarray(arr).convert("RGB").save(p, "JPEG", quality=90)
        return p

    try:
        return _finish_split("facad", split, records, seed=seed, limit=limit,
                             materialize=materialize)
    finally:
        h5.close()


# ── GoodNews (news) — captioning_dataset.json + resized.tar.gz ────────────────

def goodnews_pairs(split: str, seed: int, limit: int | None = None
                   ) -> tuple[list[str], list[Path]]:
    """One (photo caption, article image) pair per article-image. Query = the NYT
    caption. Images are extracted from resized.tar.gz (members
    resized/<article_id>_<ix>.jpg) in a SINGLE pass, only for the chosen split."""
    import numpy as np

    from splits.strata import DEFAULT_SIZES, assign_split

    root = _dataset_dir("GOODNEWS_DIR", "GoodNews")
    jpath = next((p for p in (root / "captioning_dataset.json",) if p.exists()),
                 None) or next(iter(sorted(root.glob("*.json"))), None)
    if jpath is None:
        raise FileNotFoundError(f"[goodnews] no captioning json under {root}")
    tar_path = next((p for p in (root / "resized.tar.gz",) if p.exists()), None) \
        or next(iter(sorted(root.glob("*.tar.gz"))), None)
    if tar_path is None:
        raise FileNotFoundError(f"[goodnews] no resized.tar.gz under {root}")

    data = json.loads(jpath.read_text(encoding="utf-8"))
    records = []
    for aid, art in data.items():
        imgs = (art or {}).get("images") or {}
        for ix, cap in imgs.items():
            cap = " ".join(str(cap or "").split())
            if not cap:
                continue
            rid = f"{aid}_{ix}"
            records.append({"rid": rid, "text": cap, "key": "all",
                            "member": f"resized/{rid}.jpg"})

    assign = assign_split(
        records, key_fn=lambda r: r["key"], rid_fn=lambda r: r["rid"],
        cache_path=out_path(f"goodnews_split_seed{seed}.json", "goodnews"),
        sizes=DEFAULT_SIZES, seed=seed)
    rids = assign[split]
    if limit is not None and limit < len(rids):
        idx = np.random.default_rng(seed).permutation(len(rids))[:limit]
        idx.sort()
        rids = [rids[i] for i in idx]

    by = {r["rid"]: r for r in records}
    img_dir = _cache_img_dir("goodnews")
    want = {}                                   # member name -> dest path
    for rid in rids:
        rec = by[rid]
        dest = img_dir / f"{_safe(rid)}.jpg"
        if not dest.exists():
            want[rec["member"]] = dest
    if want:
        print(f"[goodnews] extracting {len(want)} images from {tar_path.name} "
              f"(one pass)…", flush=True)
        with tarfile.open(tar_path, "r:gz") as t:
            for m in t:
                dest = want.get(m.name)
                if dest is None:
                    continue
                src = t.extractfile(m)
                if src is not None:
                    dest.write_bytes(src.read())
                want.pop(m.name, None)
                if not want:
                    break

    texts, paths, n_miss = [], [], 0
    for rid in rids:
        rec = by[rid]
        p = img_dir / f"{_safe(rid)}.jpg"
        if not p.exists():
            n_miss += 1
            continue
        texts.append(rec["text"])
        paths.append(p)
    if n_miss:
        print(f"[goodnews] {split}: {n_miss} images not found in tar (skipped)")
    print(f"[grid_local] goodnews/{split}: {len(paths)} pairs")
    return texts, paths


# ── Recipe1M (food) — native MIT layer1/layer2 (awaiting image download) ──────

def recipe1m_pairs(split: str, seed: int, limit: int | None = None
                   ) -> tuple[list[str], list[Path]]:
    """Query = the INGREDIENTS list. Native MIT format:
        RECIPE1M_DIR/layer1.json  [{id, title, ingredients:[{text}], partition}]
        RECIPE1M_DIR/layer2.json  [{id, images:[{id, url}]}]   (image filenames)
        RECIPE1M_DIR/images/<partition>/<a>/<b>/<c>/<d>/<imgid>.jpg  (MIT layout)
    One (ingredients, first-available-image) pair per recipe (dedup -> single
    positive). Download only ONE image partition (val or test) to save disk."""
    root = _dataset_dir("RECIPE1M_DIR", "Recipe1M")
    # layer1/layer2 ship inside recipe1M_layers.tar.gz — auto-extract them once
    # (small) if the JSONs aren't already unpacked.
    # Extract the layer JSONs to the LOCAL cache, never into the (DRIVE-synced)
    # data dir — layer1.json is ~1.3 GB uncompressed and would thrash sync.
    work = DATA_DIR / "recipe1m"
    work.mkdir(parents=True, exist_ok=True)
    l1 = next((p for p in (work / "layer1.json", root / "layer1.json")
               if p.exists()), None)
    if l1 is None:
        layers_tar = next(iter(root.glob("*layers*.tar.gz")), None)
        if layers_tar is not None:
            want = {"layer1.json", "layer2.json", "det_ingrs.json"}
            print(f"[recipe1m] extracting {sorted(want)} from {layers_tar.name} "
                  f"-> {work} (a few GB, one time)…", flush=True)
            with tarfile.open(layers_tar, "r:gz") as t:
                for m in t:
                    base = os.path.basename(m.name)
                    if base in want:
                        m.name = base
                        t.extract(m, work)
                        want.discard(base)
                        print(f"[recipe1m]   extracted {base}", flush=True)
                        if not want:
                            break            # stop scanning the rest of the tar
            l1 = next((p for p in (work / "layer1.json",) if p.exists()), None)
    if l1 is None:
        raise FileNotFoundError(
            f"[recipe1m] layer1.json not found under {root} (and no *layers*.tar.gz "
            f"to extract it from). Need layer1 + a layer2/layer2+ + the extracted "
            f"val image tree.")
    gb = l1.stat().st_size / 1e9
    if gb > 0.4:
        print(f"[recipe1m] parsing {l1.name} ({gb:.1f} GB) — this needs several GB "
              f"of RAM and a few minutes; results are cached after the first run.",
              flush=True)
    layer1 = json.loads(l1.read_text(encoding="utf-8"))
    # Prefer the SMALL layer2.json over the 2.6 GB layer2+.json (Recipe1M+).
    l2p = next((p for p in (work / "layer2.json", root / "layer2.json",
                            root / "layer2+.json") if p.exists()), None)
    imgs_by_id = {}
    if l2p is not None:
        for rec in json.loads(l2p.read_text(encoding="utf-8")):
            imgs_by_id[rec["id"]] = [im["id"] for im in rec.get("images", [])]

    # The image tar (recipe1M_images_val.tar) extracts to a <part>/a/b/c/d/ tree;
    # search both RECIPE1M_DIR/<part> and RECIPE1M_DIR/images/<part>.
    img_roots = [root / "images", root]

    def _resolve(imgid: str):
        for base in img_roots:
            for part in ("val", "test", "train"):
                p = base / part / imgid[0] / imgid[1] / imgid[2] / imgid[3] / imgid
                if p.exists():
                    return p
        for base in img_roots:
            if base.exists():
                hits = list(base.rglob(imgid))
                if hits:
                    return hits[0]
        return None

    records = []
    for rec in layer1:
        ings = rec.get("ingredients") or []
        text = ", ".join(i.get("text", "") if isinstance(i, dict) else str(i)
                         for i in ings).strip()
        rid = rec.get("id", "")
        cand = imgs_by_id.get(rid) or []
        records.append({"rid": rid, "text": text, "key": rec.get("partition", "all"),
                        "_imgids": cand})

    def materialize(rec):
        for imgid in rec["_imgids"]:
            p = _resolve(imgid)
            if p is not None:
                return p
        return None

    return _finish_split("recipe1m", split, records, seed=seed, limit=limit,
                         materialize=materialize)


# ── DeepEyeNet (ophthalmology; native splits) — skipped for now, kept wired ───

def deepeyenet_pairs(split: str) -> tuple[list[str], list[Path]]:
    root = _dataset_dir("DEEPEYENET_DIR", "DeepEyeNet")
    fname = {"val": "valid", "validation": "valid"}.get(split, split)
    jpath = root / f"DeepEyeNet_{fname}.json"
    if not jpath.exists():
        raise FileNotFoundError(f"[deepeyenet] missing {jpath.name} in {root}")
    entries = json.loads(jpath.read_text(encoding="utf-8"))
    texts, paths = [], []
    for entry in entries:
        for rel, fields in entry.items():
            desc = (fields.get("clinical-description")
                    or fields.get("clinical_description") or "").strip()
            p = root / rel
            if desc and p.exists():
                texts.append(desc)
                paths.append(p)
    print(f"[grid_local] deepeyenet {fname}: {len(paths)} pairs (clinical desc)")
    return texts, paths


# ── TreeOfLife (biology) — real Wikipedia description + bounded webdataset pull ─

def treeoflife_ds_pairs(split: str, seed: int, cache_dir: Path,
                        limit: int | None = None
                        ) -> tuple[list[str], list[Path]]:
    """Query = REAL Wikipedia `description` (imageomics/TreeOfLife-10M-Captions),
    deduped to one representative image per species (single-positive). Images are
    streamed from the HF webdataset imageomics/TreeOfLife-10M and matched by
    treeoflife_id. BOUNDED: we stream until we've collected TOL_MAX_SPECIES
    species (env, default 12000) or the stream ends, so we never read all 10M.

    Requires TOL_DIR/catalog.csv (treeoflife_id + taxonomy). The species reps +
    descriptions + cached images are written once to a manifest so re-runs are
    instant.
    """
    manifest = out_path(f"treeoflife_ds_seed{seed}.jsonl", "treeoflife")
    if not manifest.exists():
        _build_treeoflife_manifest(manifest, seed)

    from splits.strata import DEFAULT_SIZES, assign_split
    import numpy as np

    recs = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    records = [{"rid": r["rid"], "text": r["text"], "key": r["key"],
                "path": r["path"]} for r in recs]
    assign = assign_split(
        records, key_fn=lambda r: r["key"], rid_fn=lambda r: r["rid"],
        cache_path=out_path(f"treeoflife_split_seed{seed}.json", "treeoflife"),
        sizes=DEFAULT_SIZES, seed=seed)
    rids = assign[split]
    if limit is not None and limit < len(rids):
        idx = np.random.default_rng(seed).permutation(len(rids))[:limit]
        idx.sort()
        rids = [rids[i] for i in idx]
    by = {r["rid"]: r for r in records}
    sel = [by[rid] for rid in rids if rid in by]
    print(f"[grid_local] treeoflife/{split}: {len(sel)} pairs")
    return [r["text"] for r in sel], [Path(r["path"]) for r in sel]


def _build_treeoflife_manifest(manifest: Path, seed: int) -> None:
    """Stream TOL, collect one image per species that has a Wikipedia description,
    up to TOL_MAX_SPECIES. Writes {rid, text, key, path} lines."""
    import pyarrow.parquet as pq
    from datasets import load_dataset

    tol_dir = Path(os.environ.get("TOL_DIR", str(_SPLITS_DIR / "TOL")))
    catalog = tol_dir / "catalog.csv"
    if not catalog.exists():
        raise FileNotFoundError(f"[treeoflife] catalog.csv not found at {catalog} "
                                f"(set TOL_DIR).")
    max_species = int(os.environ.get("TOL_MAX_SPECIES", "12000"))

    # 1) uuid -> (species, class) from the catalog (stream; keep only what we need).
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    uuid_taxon: dict[str, tuple[str, str]] = {}
    with open(catalog, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        idc = "treeoflife_id" if "treeoflife_id" in (r.fieldnames or []) else "uuid"
        for row in r:
            sp = (row.get("species") or "").strip()
            if sp:
                uuid_taxon[row[idc].strip()] = (sp, (row.get("class") or "unknown"))
    print(f"[treeoflife] catalog: {len(uuid_taxon)} ided rows with a species")

    # 2) uuid -> Wikipedia description (real). Prefer a local parquet if provided.
    src = os.environ.get("TOL_CAPTIONS_PARQUET",
                         "imageomics/TreeOfLife-10M-Captions")
    desc_by_uuid: dict[str, str] = {}
    p = Path(src)
    if p.exists() and p.suffix == ".parquet":
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(columns=["uuid", "description"], batch_size=65536):
            d = b.to_pydict()
            for u, desc in zip(d["uuid"], d["description"]):
                if desc and u in uuid_taxon:
                    desc_by_uuid.setdefault(u, str(desc))
    else:
        ds = load_dataset(src, data_files="uuid_caption_description.parquet",
                          split="train", streaming=True)
        for row in ds:
            u, desc = row.get("uuid"), row.get("description")
            if desc and u in uuid_taxon:
                desc_by_uuid.setdefault(u, str(desc))
    print(f"[treeoflife] {len(desc_by_uuid)} uuids carry a Wikipedia description")

    # 3) stream the image webdataset; keep one image per species until the budget.
    from io import BytesIO

    import PIL.Image
    from datasets import Image as HFImage

    # These are trusted scientific scans (some are 90M+ pixels), so lift PIL's
    # decompression-bomb cap that would otherwise raise on the big herbarium
    # sheets.
    PIL.Image.MAX_IMAGE_PIXELS = None

    img_dir = _cache_img_dir("treeoflife")
    seen_species: set[str] = set()
    wds = load_dataset("imageomics/TreeOfLife-10M", split="train", streaming=True)
    # Disable datasets' auto image-decode: a single corrupt EXIF among 10M images
    # otherwise crashes the whole stream (getexif -> TIFF parse) inside __next__,
    # before any of our code runs. We decode each image ourselves below, skipping
    # any that fail. cast_column to a raw (undecoded) Image on every plausible
    # image column; casts of absent columns are ignored.
    feats = getattr(wds, "features", None) or {}
    img_cols = [n for n, f in feats.items() if type(f).__name__ == "Image"] \
        or ["jpg", "image", "png", "jpeg", "webp"]
    for c in img_cols:
        try:
            wds = wds.cast_column(c, HFImage(decode=False))
        except Exception:      # noqa: BLE001 - column absent / not an Image
            pass

    n = 0
    with open(manifest, "w", encoding="utf-8") as out:
        for row in wds:
            uid = (row.get("treeoflife_id") or row.get("__key__")
                   or row.get("uuid") or "")
            uid = str(uid)
            if uid not in desc_by_uuid:
                continue
            species, klass = uuid_taxon[uid]
            if species in seen_species:
                continue
            raw = next((row.get(c) for c in img_cols if row.get(c)), None)
            if raw is None:
                continue
            data = raw.get("bytes") if isinstance(raw, dict) else raw
            if not data:
                continue
            dest = img_dir / f"{_safe(uid)}.jpg"
            try:
                PIL.Image.open(BytesIO(data)).convert("RGB").save(
                    dest, "JPEG", quality=90)
            except Exception:      # noqa: BLE001 - corrupt/undecodable image
                continue
            seen_species.add(species)
            out.write(json.dumps({"rid": uid, "text": desc_by_uuid[uid],
                                  "key": klass, "path": str(dest)}) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"[treeoflife] collected {n}/{max_species} species", flush=True)
            if n >= max_species:
                break
    print(f"[treeoflife] manifest: {n} species reps -> {manifest.name}")
    if n == 0:
        raise RuntimeError("[treeoflife] collected 0 species — the webdataset row "
                           "keys may differ; inspect one row's keys and adjust "
                           "_build_treeoflife_manifest.")
