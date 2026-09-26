"""Generic dataset schema for the audit pipeline.

Three artifacts per dataset, all under out/<dataset>/:

    manifest.json   ordered image list — position = gallery index everywhere
                    [{"idx": 0, "ref": "<local path or https URL>", "meta": {...}}]
    queries.json    [{"qid": "...", "text": "...", "positives": [idx, ...]}]
                    (single-GT datasets just have one positive per query)
    meta.json       {"dataset", "embed_model", "text_max_tokens", ...}

Images referenced by URL are downloaded once into out/<dataset>/images/
and reused (captioning is the only step that needs pixels for
precomputed-embedding datasets like ILIAS).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class ImageRec:
    idx: int
    ref: str                      # local path or https:// URL
    meta: dict = field(default_factory=dict)


@dataclass
class QueryRec:
    qid: str
    text: str
    positives: list[int]          # gallery idxs that satisfy this query


def save_json(objs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(o) if hasattr(o, "__dataclass_fields__") else o
                   for o in objs] if isinstance(objs, list) else objs,
                  f, ensure_ascii=False, indent=1)


def load_images(path: Path) -> list[ImageRec]:
    with open(path, encoding="utf-8") as f:
        return [ImageRec(**d) for d in json.load(f)]


def load_queries(path: Path) -> list[QueryRec]:
    with open(path, encoding="utf-8") as f:
        return [QueryRec(**d) for d in json.load(f)]


def load_meta(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def local_name(ref: str) -> str:
    """Cache filename for a remote ref — stable under manifest reindexing."""
    return hashlib.md5(ref.encode()).hexdigest()[:12] + "_" + Path(ref).name


def resolve_image(rec: ImageRec, download_dir: Path) -> Image.Image:
    """Open a local image, or download-and-cache a remote one."""
    if not rec.ref.startswith(("http://", "https://")):
        return Image.open(rec.ref).convert("RGB")
    download_dir.mkdir(parents=True, exist_ok=True)
    local = download_dir / local_name(rec.ref)
    if not local.exists():
        tmp = local.with_suffix(".part")
        urllib.request.urlretrieve(rec.ref, tmp)
        tmp.rename(local)
    return Image.open(local).convert("RGB")
