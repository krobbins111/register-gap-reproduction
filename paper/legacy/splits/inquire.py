"""INQUIRE test gallery + 250 expert queries (multi-positive) over iNat24.

This is the INQUIRE *evaluation* side — distinct from training, which comes from
TreeOfLife (splits/treeof_life.py). INQUIRE is the paper's hardest register gap:
expert ecology prose ("godwit performing distal rhynchokinesis") against a
natural-world gallery, and unlike SemArt/ROCOv2 it is MULTI-POSITIVE — each query
has 1..1000+ correct images (~33k total) and the headline metric is mAP@50
(INQUIRE's own metric; see metrics.multipos_table / query_projection).

What's loadable, what isn't (see the Step-1 recon):
  * queries + labels : 3 small CSVs in the public GitHub repo
        data/inquire/inquire_queries_val.csv   (index,query_id,query_text,super…)
        data/inquire/inquire_queries_test.csv
        data/inquire/inquire_annotations.csv   (query_id,image_id,image_path)
    Every annotation row is a POSITIVE (relevance is implicit). We auto-download
    them from raw.githubusercontent if --inquire-dir doesn't have them.
  * gallery embeddings: we do NOT embed the 441 GB of iNat24 images. Instead we
    load INQUIRE's OWN pre-computed SigLIP v1 (SO400M-14@384) shards — the user
    downloads the embedding folder from Google Drive (set INQUIRE_EMBS_DIR):
        embs/siglip-so400m-14-384/img_emb/img_emb_*.npy   float16 [n, 1152]
        embs/siglip-so400m-14-384/metadata/*.parquet      per-row image_id
    See load_inquire_gallery_embeddings + build_inquire_queries below. The whole
    image-path / distractor-sampling machinery (build_inquire_gallery) is LEGACY,
    only for an embed-from-scratch run; the default flow is embedding-based.

‼ AUTHOR — GALLERY SUBSET DECISION (defensible-paper choice; see recon Q7):
  Do NOT embed all 5M. The gallery here is
        all ~33k query-positive images  ∪  a controlled distractor pool
  so every query's FULL positive set is present (mAP labels stay complete) while
  the index fits in numpy RAM and embeds in <2 h. Knobs:
     --max-gallery-images N   total gallery size (positives + distractors).
                              None = positives only (smoke; trivially easy).
     --species-filter K       restrict distractors to the query species + K
                              random other species (a focused, harder-than-random
                              distractor set). Omit = distractors sampled from all
                              species.
  REPORT HONESTLY: a reduced-distractor gallery is *easier* than 5M fullrank, so
  the subset mAP@50 is an UPPER BOUND, not leaderboard-comparable. Call it
  "INQUIRE-subset (Nk-distractor) fullrank".
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

from config import ROOT, inquire_embs_dir_from_env, out_path
from manifest import ImageRec, QueryRec

_GDRIVE = ("https://drive.google.com/drive/folders/"
           "1remNGZdc08B7i-Xg3oAaY68fnJ3QXyWm")
_ID_COL_CANDIDATES = ("image_id", "inat24_image_id", "id", "uid")

_RAW_BASE = ("https://raw.githubusercontent.com/inquire-benchmark/INQUIRE/"
             "main/data/inquire/")
_CSV_FILES = ("inquire_queries_val.csv", "inquire_queries_test.csv",
              "inquire_annotations.csv")


def inquire_dir_from_env(cache_dir: Path | None = None) -> Path:
    d = Path(os.environ.get("INQUIRE_DIR",
                            str((cache_dir or (ROOT / "cache")) / "inquire")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def inat24_dir_from_env(cache_dir: Path | None = None) -> Path:
    return Path(os.environ.get("INAT24_DIR",
                               str((cache_dir or (ROOT / "cache")) / "inat24")))


def _ensure_csv(name: str, inquire_dir: Path) -> Path:
    """Return the local CSV path, fetching it from the public repo if absent."""
    local = inquire_dir / name
    if local.exists():
        return local
    url = _RAW_BASE + name
    print(f"[inquire] downloading {name} from {url}")
    tmp = local.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(local)
    return local


def _species_of(image_path: str) -> str:
    """Species key = the taxonomy dir token of an iNat24 path, e.g.
    'train/04686_Animalia_..._Mungos_mungo/<uuid>.jpg' -> the middle dir."""
    parts = image_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


def load_queries_labels(inquire_dir: Path, which=("val", "test")
                        ) -> tuple[list[dict], dict[int, list[tuple[int, str]]]]:
    """Parse the INQUIRE CSVs.

    Returns
      queries : [{"query_id": int, "text": str, "supercategory", "category",
                  "iconic_group", "split"}], in (val then test) order.
      labels  : query_id -> [(image_id:int, image_path:str)]  (all positives).
    """
    queries: list[dict] = []
    for split in which:
        qcsv = _ensure_csv(f"inquire_queries_{split}.csv", inquire_dir)
        with open(qcsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                queries.append({
                    "query_id": int(r["query_id"]),
                    "text": r["query_text"].strip(),
                    "supercategory": r.get("supercategory", "").strip(),
                    "category": r.get("category", "").strip(),
                    "iconic_group": r.get("iconic_group", "").strip(),
                    "split": split,
                })

    acsv = _ensure_csv("inquire_annotations.csv", inquire_dir)
    labels: dict[int, list[tuple[int, str]]] = {}
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with open(acsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            qid = int(r["query_id"])
            labels.setdefault(qid, []).append(
                (int(r["image_id"]), r["image_path"].strip()))
    n_pos = sum(len(v) for v in labels.values())
    print(f"[inquire] {len(queries)} queries ({'/'.join(which)}), "
          f"{n_pos} positive annotations over {len(labels)} labelled queries")
    return queries, labels


# ── pre-computed gallery embeddings (the default path) ───────────────────────

def _shard_num(p: Path) -> tuple[int, str]:
    """Sort key: trailing integer in the filename (img_emb_12 -> 12)."""
    m = re.findall(r"(\d+)", p.stem)
    return (int(m[-1]) if m else -1, p.stem)


def load_inquire_gallery_embeddings(embs_dir: Path | None = None, *,
                                    encoder: str = "siglip-so400m-14-384",
                                    id_col: str | None = None
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Load INQUIRE's pre-computed gallery shards, returned AS-IS (float16).

    Layout under embs_dir (downloaded from the INQUIRE Google Drive):
        embs/<encoder>/img_emb/img_emb_*.npy      float16 [n_shard, 1152]
        embs/<encoder>/metadata/*.parquet         per-shard row metadata (image_id)

    Shards are concatenated in sorted numeric order; the parquet metadata gives
    each row's iNat24 image_id, so labels map to gallery rows. Each .npy is
    memmapped; np.concatenate then materializes the gallery (≈11.5 GB for the
    full 5M — expected; cap with step1 --max-gallery-images for a lighter run).

    Returns (embs[N, 1152] float16, image_ids[N] int64).
    """
    embs_dir = Path(embs_dir) if embs_dir else inquire_embs_dir_from_env()
    base = embs_dir / "embs" / encoder
    img_dir, meta_dir = base / "img_emb", base / "metadata"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"[inquire] pre-computed embeddings not found at {img_dir}.\n"
            f"  Download the SigLIP SO400M-14@384 embedding folder from INQUIRE's "
            f"Google Drive:\n    {_GDRIVE}\n"
            f"  and set INQUIRE_EMBS_DIR to the local path (expected layout "
            f"embs/{encoder}/img_emb/img_emb_*.npy + metadata/*.parquet).")

    npys = sorted(img_dir.glob("img_emb_*.npy"), key=_shard_num)
    metas = sorted(meta_dir.glob("*.parquet"), key=_shard_num)
    if not npys:
        raise FileNotFoundError(f"[inquire] no img_emb_*.npy under {img_dir}")
    if not metas:
        raise FileNotFoundError(f"[inquire] no *.parquet under {meta_dir}")

    import pyarrow.parquet as pq

    def _ids_from(parquet_path: Path) -> np.ndarray:
        t = pq.read_table(parquet_path)
        col = id_col or next((c for c in _ID_COL_CANDIDATES
                              if c in t.column_names), None)
        if col is None:
            raise KeyError(
                f"[inquire] no image-id column in {parquet_path.name}; columns "
                f"= {t.column_names}. Pass id_col=… for the right one.")
        return np.asarray(t.column(col).to_pylist(), dtype=np.int64)

    shards = [np.load(p, mmap_mode="r") for p in npys]
    lengths = [s.shape[0] for s in shards]
    if len(metas) == len(npys):                       # one parquet per shard
        id_list = [_ids_from(m) for m in metas]
    elif len(metas) == 1:                             # single metadata table
        all_ids = _ids_from(metas[0])
        b = np.cumsum([0] + lengths)
        id_list = [all_ids[b[i]:b[i + 1]] for i in range(len(lengths))]
    else:
        raise ValueError(
            f"[inquire] {len(npys)} img_emb shards but {len(metas)} parquet "
            f"files — expected one-per-shard or a single metadata table.")

    for p, s, idc in zip(npys, shards, id_list):
        if len(idc) != s.shape[0]:
            raise ValueError(
                f"[inquire] {p.name}: {s.shape[0]} rows but {len(idc)} image_ids "
                f"in its metadata — shard/metadata misalignment.")

    embs = np.concatenate([np.asarray(s) for s in shards], axis=0)
    image_ids = np.concatenate(id_list).astype(np.int64)
    print(f"[inquire] gallery embeddings: {embs.shape} {embs.dtype} from "
          f"{len(npys)} shard(s), encoder={encoder}")
    return embs, image_ids


def build_inquire_queries(image_ids: np.ndarray,
                          inquire_dir: Path | None = None) -> list[QueryRec]:
    """Multi-positive queries against a gallery laid out as `image_ids` (row r ==
    gallery index of image_ids[r]). Each query's positives = the gallery rows of
    its relevant image_ids that are present in the gallery."""
    inquire_dir = inquire_dir or inquire_dir_from_env()
    queries, labels = load_queries_labels(inquire_dir)
    id2row = {int(iid): r for r, iid in enumerate(image_ids.tolist())}

    qrecs: list[QueryRec] = []
    n_absent = 0
    for q in queries:
        ann = labels.get(q["query_id"], [])
        rows = sorted({id2row[iid] for iid, _ in ann if iid in id2row})
        n_absent += len({iid for iid, _ in ann}) - len(rows)
        if rows:
            qrecs.append(QueryRec(f"inquire_{q['query_id']}", q["text"], rows))
    print(f"[inquire] {len(qrecs)} queries mapped to gallery rows "
          f"({n_absent} positive annotations absent from the embedding set)")
    return qrecs


def _scan_inat24_catalog(inat24_dir: Path) -> list[tuple[int, str]]:
    """All iNat24 images as [(image_id, image_path)] from train.json (COCO-style:
    images[].id / images[].file_name). Used only for distractor sampling; if the
    catalog is absent we fall back to a positives-only gallery."""
    cat = inat24_dir / "train.json"
    if not cat.exists():
        print(f"[inquire] WARNING: {cat} not found — no distractor catalog. "
              f"Gallery will be positives-only (download train.json.tar.gz from "
              f"the iNat2024 S3 bucket to add distractors).")
        return []
    print(f"[inquire] loading iNat24 catalog {cat} (this is a few GB of JSON)…")
    with open(cat, encoding="utf-8") as f:
        data = json.load(f)
    imgs = data["images"] if isinstance(data, dict) else data
    out = [(int(im["id"]), im["file_name"]) for im in imgs]
    print(f"[inquire] catalog: {len(out)} iNat24 images")
    return out


def build_inquire_gallery(*, inquire_dir: Path | None = None,
                          inat24_dir: Path | None = None,
                          max_gallery_images: int | None = None,
                          species_filter: int | None = None,
                          seed: int = 0, cache_dir: Path | None = None
                          ) -> tuple[list[ImageRec], list[QueryRec]]:
    """Build the (gallery images, multi-positive queries) for INQUIRE.

    Gallery = unique positives (always) + distractors up to max_gallery_images.
    Positives come first so their gallery indices are stable across distractor
    choices. Each query's `positives` = gallery indices of its relevant images
    present in the gallery (always its full set, since positives are never
    dropped). image local path = <inat24_dir>/<image_path>.
    """
    inquire_dir = inquire_dir or inquire_dir_from_env(cache_dir)
    inat24_dir = inat24_dir or inat24_dir_from_env(cache_dir)
    queries, labels = load_queries_labels(inquire_dir)

    # ── positives: unique image_id -> (image_path, species) ──────────────────
    pos_path: dict[int, str] = {}
    for qid, items in labels.items():
        for image_id, image_path in items:
            pos_path.setdefault(image_id, image_path)
    query_species = {_species_of(p) for p in pos_path.values()}
    print(f"[inquire] {len(pos_path)} unique positive images across "
          f"{len(query_species)} species")

    # ── distractors: sample from the iNat24 catalog (optional species filter) ─
    distractors: list[tuple[int, str]] = []
    n_pos = len(pos_path)
    n_want_distract = (None if max_gallery_images is None
                       else max(0, max_gallery_images - n_pos))
    if n_want_distract:
        catalog = _scan_inat24_catalog(inat24_dir)
        rng = np.random.default_rng(seed)
        if species_filter is not None:
            all_species = sorted({_species_of(p) for _, p in catalog})
            extra_pool = [s for s in all_species if s not in query_species]
            k = min(species_filter, len(extra_pool))
            chosen = set(rng.choice(extra_pool, size=k, replace=False).tolist()) \
                if k else set()
            allowed = query_species | chosen
            cand = [(i, p) for i, p in catalog
                    if i not in pos_path and _species_of(p) in allowed]
            print(f"[inquire] species-filter: {len(allowed)} species "
                  f"({len(query_species)} query + {k} distractor) -> "
                  f"{len(cand)} candidate distractors")
        else:
            cand = [(i, p) for i, p in catalog if i not in pos_path]
        if cand:
            take = rng.permutation(len(cand))[:n_want_distract]
            distractors = [cand[j] for j in take]
        print(f"[inquire] sampled {len(distractors)} distractors "
              f"(wanted {n_want_distract})")

    # ── assemble gallery (positives first) + multi-positive queries ──────────
    images: list[ImageRec] = []
    id2idx: dict[int, int] = {}

    def _add(image_id: int, image_path: str, role: str) -> None:
        idx = len(images)
        id2idx[image_id] = idx
        local = str(inat24_dir / image_path)
        images.append(ImageRec(idx, local,
                               {"image_id": image_id,
                                "species": _species_of(image_path),
                                "role": role}))

    for image_id, image_path in pos_path.items():
        _add(image_id, image_path, "positive")
    for image_id, image_path in distractors:
        if image_id not in id2idx:
            _add(image_id, image_path, "distractor")

    qrecs: list[QueryRec] = []
    for q in queries:
        pos_idx = sorted(id2idx[iid] for iid, _ in labels.get(q["query_id"], [])
                         if iid in id2idx)
        if not pos_idx:
            continue                    # query whose positives aren't in gallery
        qrecs.append(QueryRec(f"inquire_{q['query_id']}", q["text"], pos_idx))

    print(f"[inquire] gallery {len(images)} images "
          f"({n_pos} positives + {len(images) - n_pos} distractors), "
          f"{len(qrecs)} usable queries")
    return images, qrecs


# ── step8/step10 entry point: embed the test side, multi-positive ───────────

def embed_inquire_test(embedder, *, mtag: str = "",
                       cache_dir: Path | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(query_embs, gallery_embs, positives_list) for the INQUIRE test gallery.

    Reuses the gallery embeddings step1 already produced (gallery_embeddings
    [_tag].npz) and the manifest/queries it wrote — so the heavy gallery embed
    happens once, in step1. Only the 250 query texts are embedded here. Also
    writes the step10 cache (delta_test_embs[_tag].npz with caps=queries,
    imgs=gallery + test_positives[_tag].json) so geometry can read it.

    positives_list[i] = np.ndarray of gallery indices that satisfy query i.
    """
    from manifest import load_images, load_queries

    gname = "gallery_embeddings.npz" if not mtag \
        else f"gallery_embeddings_{mtag}.npz"
    gpath = out_path(gname, "inquire")
    if not gpath.exists():
        raise SystemExit(
            f"[inquire] {gpath} missing — run step1 first:\n"
            f"  python embed_gallery.py --dataset inquire "
            f"--max-gallery-images N [--species-filter K] "
            f"[--embed-model …]")
    gallery = np.load(gpath)["embeddings"]

    queries = load_queries(out_path("queries.json", "inquire"))
    # sanity: manifest length must match the embedded gallery
    images = load_images(out_path("manifest.json", "inquire"))
    if len(images) != len(gallery):
        raise SystemExit(
            f"[inquire] manifest ({len(images)}) vs gallery_embeddings "
            f"({len(gallery)}) length mismatch — re-run step1 so they agree.")

    q_texts = [q.text for q in queries]
    q_embs = embedder.embed_texts(q_texts)
    positives_list = [np.asarray(q.positives, dtype=np.int64) for q in queries]

    # step10 cache (caps/imgs keys, like the other datasets' test cache) + labels
    np.savez_compressed(out_path(
        f"delta_test_embs{('_' + mtag) if mtag else ''}.npz", "inquire"),
        caps=q_embs, imgs=gallery)
    with open(out_path(f"test_positives{('_' + mtag) if mtag else ''}.json",
                       "inquire"), "w", encoding="utf-8") as f:
        json.dump([p.tolist() for p in positives_list], f)

    print(f"[inquire] test: {len(q_embs)} queries vs {len(gallery)} gallery "
          f"images, multi-positive (mAP@50 eval)")
    return q_embs, gallery, positives_list


def load_test_positives(mtag: str = "") -> list[np.ndarray]:
    """Read the cached multi-positive label sets (for step10)."""
    p = out_path(f"test_positives{('_' + mtag) if mtag else ''}.json", "inquire")
    with open(p, encoding="utf-8") as f:
        return [np.asarray(x, dtype=np.int64) for x in json.load(f)]
