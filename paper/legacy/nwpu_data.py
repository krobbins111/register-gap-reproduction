"""NWPU-Captions manifest + image extraction.

Reads dataset_nwpu.json (a dict of 45 scene-category keys -> list of image
records) and produces an ordered list of records. The order of this list
defines gallery index order for the whole pipeline — every later artifact
(embeddings, captions, indexability scores) refers to images by their
position here.

Caption strategy: each image ships 5 captions (raw, raw_1..raw_4); we select
the LONGEST by word count (tiebreak = raw, the first one). Longer NWPU
captions consistently add spatial relational language specific to the
aerial/satellite viewpoint (mean 18.2 words vs 15.2 for caption-0).

Images come from NWPU_images.tar.gz (a Git LFS object). They are extracted
once into NWPU_IMAGES_DIR; the tar may store jpgs flat or under
category subfolders — either way they are placed directly in images_dir.
"""
from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

# Repo root = parent of the indexability dir (matches config.ROOT).
ROOT = Path(__file__).resolve().parent.parent
from config import DATA_DIR  # noqa: E402


@dataclass
class NWPURecord:
    idx: int                 # gallery index (position in this manifest)
    imgid: int               # unique image id across the full dataset
    filename: str            # image filename (no category subfolder prefix)
    category: str            # scene category key (airplane, airport, …)
    split: str               # "train" | "val" | "test"
    caption: str             # the LONGEST of the 5 captions
    all_captions: list[str]  # all 5 captions in order (raw, raw_1..raw_4)


_CAP_KEYS = ("raw", "raw_1", "raw_2", "raw_3", "raw_4")


def _all_captions(item: dict) -> list[str]:
    return [str(item.get(k, "") or "").strip() for k in _CAP_KEYS]


def longest_caption(item: dict) -> str:
    """Pick the longest caption (by word count) among raw/raw_1..raw_4.
    Tiebreak = raw (the first one), which is why we compare with strict >."""
    caps = _all_captions(item)
    best = caps[0]
    best_len = len(best.split())
    for cap in caps[1:]:
        n = len(cap.split())
        if n > best_len:
            best, best_len = cap, n
    return best


def nwpu_json_path() -> Path:
    return Path(__file__).resolve().parent / "dataset_nwpu.json"


def nwpu_tar_path() -> Path:
    return Path(__file__).resolve().parent / "NWPU_images.tar.gz"


def nwpu_images_dir_from_env() -> Path:
    """Read NWPU_IMAGES_DIR env var; default to ROOT/cache/nwpu_images."""
    return Path(os.environ.get("NWPU_IMAGES_DIR",
                               str(DATA_DIR / "nwpu_images")))


def load_nwpu_manifest(json_path: Path,
                       split: str | None = None) -> list[NWPURecord]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. dataset_nwpu.json must live in the "
            f"indexability root alongside the source files.")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records: list[NWPURecord] = []
    for category in sorted(data.keys()):
        for item in data[category]:
            if split is not None and item.get("split") != split:
                continue
            records.append(NWPURecord(
                idx=len(records),
                imgid=int(item["imgid"]),
                filename=str(item["filename"]),
                category=category,
                split=str(item.get("split", "")),
                caption=longest_caption(item),
                all_captions=_all_captions(item),
            ))
    return records


def ensure_images_extracted(images_dir: Path, tar_path: Path) -> None:
    """Make sure the extracted NWPU jpgs are present under images_dir.

    If images_dir already has >=100 .jpg files, do nothing. Otherwise, if
    tar_path is a real archive (size > 1 KB, i.e. not a Git LFS pointer),
    extract it — placing every jpg directly in images_dir whether the tar
    stores them flat or under category subfolders. Else raise a clear error.
    """
    images_dir = Path(images_dir)
    if images_dir.exists():
        n_jpg = sum(1 for _ in images_dir.glob("*.jpg"))
        if n_jpg >= 100:
            return

    tar_path = Path(tar_path)
    if not tar_path.exists() or tar_path.stat().st_size <= 1024:
        raise FileNotFoundError(
            f"NWPU images not extracted at {images_dir} and {tar_path} is not "
            f"a real archive (missing or a Git LFS pointer, "
            f"{tar_path.stat().st_size if tar_path.exists() else 0} bytes). "
            f"Run `git lfs pull` to fetch NWPU_images.tar.gz (~407 MB), or set "
            f"NWPU_IMAGES_DIR to a directory of already-extracted jpgs.")

    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"[nwpu_data] extracting {tar_path} -> {images_dir} …", flush=True)
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.lower().endswith(".jpg"):
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            dst = images_dir / name           # flatten any subfolder layout
            with open(dst, "wb") as out:
                out.write(src.read())
            n += 1
    print(f"[nwpu_data] extracted {n} jpgs to {images_dir}")
    if n < 100:
        raise FileNotFoundError(
            f"NWPU extraction produced only {n} jpgs in {images_dir}; the "
            f"archive {tar_path} may be incomplete. Re-fetch with `git lfs pull`.")


def image_path(images_dir: Path, rec: NWPURecord) -> Path:
    return Path(images_dir) / rec.filename
