"""Rows and images of a collection, in the row order of its cells (the registergap pipeline's registergap/embed.py loaders, ported).

Two sources, tried in order:
  1. data/<collection>/pairs.jsonl  (id, image, caption, split): `image` is a local path or an http(s) URL. Canonical (outline step 1).
  2. The loaders below, which reproduce registergap/embed.load_collection row for row (the order every existing cell, patch store and
     phrase store was built in): HF datasets for the benchmarks, the SemArt zip, the public webcam repositories for duluth / popocatepetl.
`rows(collection)` yields dicts {id, image, caption, split} where image is a path, URL or PIL image; `image_iter(collection)` yields PIL
images in the same order and decodes lazily (the big HF collections do not fit decoded in RAM). `export_pairs(collection)` writes source 1
from source 2 for the collections whose images are files on disk.
"""
from __future__ import annotations

import csv
import glob
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SPLITS = ("train", "val", "test")


def _pil(x):
    from PIL import Image
    if isinstance(x, (str, Path)):
        s = str(x)
        if s.startswith("http://") or s.startswith("https://"):
            import urllib.request
            return Image.open(io.BytesIO(urllib.request.urlopen(s, timeout=60).read()))
        return Image.open(s)
    return x


def _imgcol(d):
    from datasets import Image as HFImage
    return [c for c in d.column_names if isinstance(d.features[c], HFImage)][0]


def _hf_rows(collection: str):
    """registergap/embed.load_collection, verbatim row selection (seeds, orders, caption fields)."""
    from datasets import load_dataset
    rows = []
    if collection == "rsicd":
        ds = load_dataset("arampacha/rsicd")
        for split in ["train", "valid", "test"]:
            d = ds[split]; capcol = [c for c in d.column_names if "caption" in c.lower()][0]; imgcol = _imgcol(d)
            idcol = [c for c in d.column_names if "file" in c.lower() or c == "id"]
            for i, r in enumerate(d):
                caps = r[capcol]; caps = [caps] if isinstance(caps, str) else caps
                rows.append(dict(image=r[imgcol], caption=max(caps, key=len), split={"valid": "val"}.get(split, split), id=str(r[idcol[0]] if idcol else f"{split}_{i}"), captions=list(caps)))
    elif collection == "coco":                                  # the registergap pipeline's COCO-2017-val build (3.5k/0.5k/1k); retired in favour of Karpathy, kept for the existing cells
        d = load_dataset("lmms-lab/COCO-Caption2017", split="val"); imgcol = _imgcol(d)
        capcol = "answer" if "answer" in d.column_names else [c for c in d.column_names if "caption" in c.lower()][0]
        perm = np.random.RandomState(0).permutation(len(d)); split_of = {int(k): ("test" if j < 1000 else ("val" if j < 1500 else "train")) for j, k in enumerate(perm)}
        for i, r in enumerate(d):
            c = r[capcol]; c = c[0] if isinstance(c, list) else str(c)
            rows.append(dict(image=r[imgcol], caption=c, split=split_of[i], id=str(r.get("question_id", i)) if isinstance(r, dict) else str(i)))
    elif collection == "skincap":
        d = load_dataset("sercetexam9/skincap", split="train"); imgcol = _imgcol(d)
        perm = np.random.RandomState(0).permutation(len(d)); split_of = {int(k): ("test" if j < 600 else ("val" if j < 1200 else "train")) for j, k in enumerate(perm)}
        for i, r in enumerate(d):
            rows.append(dict(image=r[imgcol], caption=str(r["text"]), split=split_of[i], id=str(i)))
    elif collection == "semart":
        z = _semart_dir()
        base = os.path.dirname(glob.glob(f"{z}/**/semart_train.csv", recursive=True)[0]); imgdir = glob.glob(f"{base}/Images")[0]
        for split in ["train", "val", "test"]:
            with open(f"{base}/semart_{split}.csv", encoding="latin-1") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    rows.append(dict(image=os.path.join(imgdir, r["IMAGE_FILE"]), caption=r["DESCRIPTION"].strip(), split=split, id=r["IMAGE_FILE"]))
    elif collection in ("facad", "fashion200k"):
        if collection == "facad":
            d = load_dataset("Luna288/image-captioning-FACAD-base", split="train"); ntr, nva, nte = 8754, 1874, 1874
        else:
            d = load_dataset("Marqo/fashion200k", split="data"); seen, keep = set(), []
            for i, it in enumerate(d["item_ID"]):
                if it not in seen: seen.add(it); keep.append(i)
            d = d.select(keep); ntr, nva, nte = 50000, 4999, 4999
        imgcol = _imgcol(d); p = np.random.RandomState(0).permutation(len(d)); idx = {"train": p[:ntr], "val": p[ntr:ntr + nva], "test": p[ntr + nva:ntr + nva + nte]}
        for sp, ii in idx.items():
            for j in ii:
                r = d[int(j)]; rows.append(dict(image=r[imgcol], caption=str(r["text"]), split=sp, id=str(int(j))))
    elif collection == "rocov2":                                # the registergap pipeline's partial-download cell (6 of 27 train shards); the grid pipeline's decision: full train for new cells
        files = {"train": [f"data/train-{i:05d}-of-00027.parquet" for i in range(6)], "validation": ["data/validation-00000-of-00006.parquet"], "test": [f"data/test-{i:05d}-of-00006.parquet" for i in range(6)]}
        ds = load_dataset("eltorio/ROCOv2-radiology", data_files=files, verification_mode="no_checks")
        for split, sp in [("train", "train"), ("validation", "val"), ("test", "test")]:
            for r in ds[split]: rows.append(dict(image=r["image"], caption=str(r["caption"]), split=sp, id=str(r["image_id"])))
    elif collection == "scimmir":
        ds = load_dataset("m-a-p/SciMMIR", data_files={"validation": "data/validation-*", "test": "data/test-*"}, verification_mode="no_checks")
        rng = np.random.RandomState(0); tr = ds["validation"]; p = rng.permutation(len(tr)); te = ds["test"]; p2 = rng.permutation(len(te))
        for j in p[:16184]: r = tr[int(j)]; rows.append(dict(image=r["image"], caption=str(r["text"]), split="train", id=f"val_{int(j)}"))
        for k, j in enumerate(p2[:6934]): r = te[int(j)]; rows.append(dict(image=r["image"], caption=str(r["text"]), split="val" if k < 3467 else "test", id=f"test_{int(j)}"))
    elif collection == "flickr":
        d = load_dataset("lmms-lab/flickr30k", split="test"); imgcol = _imgcol(d)   # Plummer split ids from data/flickr30k_entities/pairs.jsonl
        raise NotImplementedError("flickr rows come from data/flickr30k_entities/pairs.jsonl (export from registergap); HF fallback not wired")
    elif collection in ("duluth", "popocatepetl", "ospreys"):
        repo = {"ospreys": "uk-osprey-cam-captions", "duluth": "duluth-harbor-cam-captions", "popocatepetl": "popocatepetl-cam-captions"}[collection]
        base = DATA / repo
        if not base.is_dir():
            subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/ANONYMIZED/{repo}.git", str(base)], check=True)
        for p in (json.loads(l) for l in open(base / "pairs.jsonl", encoding="utf-8")):
            if not p.get("is_cam", True): continue
            rows.append(dict(image=str(base / p["image"]), caption=p["caption"], split=p["split"], id=str(p["id"])))
    else:
        raise ValueError(f"no loader for {collection}; provide data/{collection}/pairs.jsonl")
    return rows


def _semart_dir() -> Path:
    root = DATA / "semart"
    if not glob.glob(f"{root}/**/semart_train.csv", recursive=True):
        from huggingface_hub import hf_hub_download
        z = hf_hub_download("leo20000306/SemArt", "SemArt.zip", repo_type="dataset"); root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf: zf.extractall(root)
    return root


def rows(collection: str) -> list[dict]:
    p = DATA / collection / "pairs.jsonl"
    if p.exists():
        out = []
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); img = r["image"]
            if not (str(img).startswith("http") or os.path.isabs(str(img))): img = str((p.parent / img).resolve())
            out.append(dict(image=img, caption=r["caption"], split=r["split"], id=str(r["id"]), **{k: v for k, v in r.items() if k not in ("image", "caption", "split", "id")}))
        return out
    return _hf_rows(collection)


def image_iter(collection: str):
    for r in rows(collection):
        yield _pil(r["image"])


def export_pairs(collection: str) -> Path:
    """Write data/<collection>/pairs.jsonl from the loaders (only for collections whose images are files: semart, the webcams, cityflow, flickr)."""
    rs = rows(collection); out = DATA / collection / "pairs.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rs:
            if not isinstance(r["image"], str): raise ValueError(f"{collection}: rows carry decoded images, not paths; export needs the HF index instead")
            f.write(json.dumps({"id": r["id"], "image": r["image"], "caption": r["caption"], "split": r["split"]}, ensure_ascii=False) + "\n")
    return out
