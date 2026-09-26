"""The embedding store: the one interface every experiment in this repository reads.

    $LTG_CACHE/<collection>/<encoder>.npz
        txt      (n, d) float32, unit-norm   caption embeddings
        img      (n, d) float32, unit-norm   image embeddings (row i is the image of caption i)
        split    (n,)   str                  'train' | 'val' | 'test'
        caption  (n,)   str                  the caption text ('' when the cell was imported from a caption-less legacy cache)
        id       (n,)   str                  the pair id (image file name / dataset id)
        meta     json str                    encoder id, dim, provenance, split rule, has_captions

    $LTG_CACHE/<collection>/<encoder>.patches.npy           (n, G*G, d) float16 unit-norm per-patch joint-space embeddings
    $LTG_CACHE/<collection>/<encoder>.phrases_<cap>[v].npz  emb (P, d) fp16, offsets (n+1,), texts (P,) object

Results go to $LTG_RESULTS/<collection>/<encoder>/<experiment>.json (committed). The cache is never inside the repo:
see cache_root() for the lookup order (env var, local.json, ../learn-the-gap-cache). A cell that is missing locally is fetched on
demand from the released cache (configs/cache_release.json -> Hugging Face dataset RegisterGap/LearnTheRegisterGapCache) and
sha256-checked against configs/cache_manifest.json; LTG_OFFLINE=1 (or local.json {"offline": true}) disables that, so a
reproduction from your own cache folder is `LTG_CACHE=/your/folder LTG_OFFLINE=1 ...`.

The layout is the registergap pipeline's (registergap/cache/<collection>_<model>.npz) with the encoder moved into a subfolder so a collection's
cells sit together; `legacy_cell_path` still finds his flat files. the grid pipeline's {split}_embs_<mtag>.npz caches are imported once by
scripts/import_legacy.py and never read directly (see configs/encoders.json for the mtag -> encoder key map).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


REPO = Path(__file__).resolve().parents[1]


def _local() -> dict:
    """local.json at the repo root (git-ignored): {"cache": "...", "results": "..."} for this machine. Env vars win over it."""
    p = REPO / "local.json"
    try:
        return json.load(open(p, encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def cache_root() -> Path:
    """Embeddings live OUTSIDE the repository (they are 10+ GB and never committed):
    $LTG_CACHE, else local.json["cache"], else ../learn-the-gap-cache next to the repo, else ./cache."""
    v = os.environ.get("LTG_CACHE") or _local().get("cache")
    if v:
        return Path(v).expanduser()
    sib = REPO.parent / "learn-the-gap-cache"
    return sib if sib.exists() else REPO / "cache"


def results_root() -> Path:
    v = os.environ.get("LTG_RESULTS") or _local().get("results")
    return Path(v).expanduser() if v else REPO / "results"


def release() -> dict:
    """configs/cache_release.json: where the released cache is served (Hugging Face dataset RegisterGap/LearnTheRegisterGapCache)."""
    p = REPO / "configs" / "cache_release.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def manifest() -> dict:
    """configs/cache_manifest.json: {path: {bytes, sha256}} for every released file."""
    p = REPO / "configs" / "cache_manifest.json"
    if not p.exists():
        return {}
    return {r["path"]: r for r in json.load(open(p, encoding="utf-8"))["cells"]}


def offline() -> bool:
    return os.environ.get("LTG_OFFLINE", "").lower() in ("1", "true", "yes") or bool(_local().get("offline"))


def sha256_of(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_file(rel: str, root: Path | None = None, verify: bool = True) -> Path:
    """Download one released file (<collection>/<encoder>.npz) into the cache and check its sha256. Standard library only."""
    import urllib.request
    base = release().get("base_url", "").rstrip("/")
    if not base:
        raise FileNotFoundError(f"{rel}: no base_url in configs/cache_release.json")
    row = manifest().get(rel)
    dst = (root or cache_root()) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = f"{base}/{rel}"
    print(f"[cache] fetching {rel}" + (f" ({row['bytes'] / 2**20:.0f} MB)" if row else "") + f" from {base}", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 22), b""):
            f.write(chunk)
    if verify and row and sha256_of(tmp) != row["sha256"]:
        tmp.unlink()
        raise IOError(f"{rel}: sha256 mismatch after download (manifest {row['sha256'][:12]}...)")
    os.replace(tmp, dst)
    return dst


def ensure_cell(collection: str, encoder: str, root: Path | None = None) -> Path:
    """Path of a cell, downloading it from the released cache if it is absent locally (and not offline)."""
    key = encoder_key(encoder)
    p = cell_path(collection, key, root)
    if p.exists():
        return p
    rel = f"{collection}/{key}.npz"
    if offline() or rel not in manifest():
        return p
    return fetch_file(rel, root)


def _encoders() -> dict:
    """configs/encoders.json (cached)."""
    global _ENC
    try:
        return _ENC
    except NameError:
        _ENC = json.load(open(REPO / "configs" / "encoders.json", encoding="utf-8"))
        return _ENC


def encoder_key(name: str) -> str:
    """Store key for an encoder given the key itself, the grid pipeline's legacy mtag or the registergap pipeline's registergap model name."""
    enc = _encoders()
    if name in enc:
        return name
    for k, v in enc.items():
        if isinstance(v, dict) and (v.get("legacy_model") == name or v.get("legacy_mtag") == name):
            return k
    return name


def legacy_names(encoder: str) -> list[str]:
    """Names a cell for this encoder may carry in the registergap pipeline's flat cache: the key itself and its registergap model name."""
    key = encoder_key(encoder)
    v = _encoders().get(key, {})
    out = [encoder, key]
    if isinstance(v, dict) and v.get("legacy_model"):
        out.append(v["legacy_model"])
    return list(dict.fromkeys(out))


def cell_path(collection: str, encoder: str, root: Path | None = None) -> Path:
    return (root or cache_root()) / collection / f"{encoder}.npz"


def legacy_cell_path(collection: str, encoder: str, root: Path | None = None) -> Path:
    """the registergap pipeline's flat layout: <root>/<collection>_<encoder>.npz."""
    return (root or cache_root()) / f"{collection}_{encoder}.npz"


def result_path(collection: str, encoder: str, name: str, root: Path | None = None) -> Path:
    p = (root or results_root()) / collection / encoder
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


@dataclass
class Cell:
    collection: str
    encoder: str
    txt: np.ndarray
    img: np.ndarray
    split: np.ndarray
    caption: np.ndarray
    id: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.txt.shape[1])

    def mask(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return self.split == split

    def arrays(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(txt, img) of one split, in stored order; row i of both is the same pair."""
        m = self.mask(split)
        return self.txt[m], self.img[m]

    def rows(self, split: str) -> dict:
        m = self.mask(split)
        return {"txt": self.txt[m], "img": self.img[m], "caption": self.caption[m], "id": self.id[m]}

    def sizes(self) -> dict:
        return {s: int((self.split == s).sum()) for s in SPLITS}

    def subsample(self, split: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Seeded subsample of a split (sorted indices, like symmetric_projection --limit-train)."""
        txt, img = self.arrays(split)
        if n is None or n >= len(txt):
            return txt, img
        idx = np.sort(np.random.default_rng(seed).permutation(len(txt))[:n])
        return txt[idx], img[idx]


def load_cell(collection: str, encoder: str, root: Path | None = None, mmap: bool = False) -> Cell:
    key = encoder_key(encoder)
    tried = [cell_path(collection, key, root), cell_path(collection, encoder, root)]
    tried += [legacy_cell_path(collection, n, root) for n in legacy_names(encoder)]
    p = next((t for t in tried if t.exists()), None)
    if p is None and not offline() and f"{collection}/{key}.npz" in manifest():
        p = fetch_file(f"{collection}/{key}.npz", root)
    if p is None:
        raise FileNotFoundError(f"no cell {collection} x {encoder} under {root or cache_root()} (looked for {[str(t) for t in tried]}); "
                                f"run `ltg embed {collection} {encoder}`, scripts/import_legacy.py, scripts/import_registergap.py, or set LTG_OFFLINE=0 to fetch it from the released cache")
    encoder = key
    z = np.load(p, allow_pickle=True, mmap_mode="r" if mmap else None)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    n = len(z["split"])
    caption = z["caption"] if "caption" in z.files else np.array([""] * n, dtype=object)
    ids = z["id"] if "id" in z.files else np.array([f"{s}_{i}" for i, s in enumerate(z["split"])], dtype=object)
    return Cell(collection, encoder, np.asarray(z["txt"], np.float32), np.asarray(z["img"], np.float32),
                np.asarray(z["split"]).astype(str), np.asarray(caption).astype(object), np.asarray(ids).astype(object), meta)


def save_cell(collection: str, encoder: str, txt, img, split, caption=None, ids=None, meta: dict | None = None,
              root: Path | None = None, compress: bool = False) -> Path:
    txt, img = unit(txt), unit(img)
    n = len(split)
    assert len(txt) == len(img) == n, "txt/img/split length mismatch"
    caption = np.array([""] * n, dtype=object) if caption is None else np.asarray(caption, dtype=object)
    ids = np.array([f"{s}_{i}" for i, s in enumerate(split)], dtype=object) if ids is None else np.asarray(ids, dtype=object)
    meta = {"collection": collection, "encoder": encoder, "dim": int(txt.shape[1]), "n": int(n),
            "sizes": {s: int((np.asarray(split) == s).sum()) for s in SPLITS},
            "has_captions": bool((caption != "").any()), **(meta or {})}
    p = cell_path(collection, encoder, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")                              # atomic: a killed write never leaves a half cell behind
    (np.savez_compressed if compress else np.savez)(tmp, txt=txt, img=img, split=np.asarray(split).astype(str),
                                                    caption=caption, id=ids, meta=json.dumps(meta))
    os.replace(tmp, p)
    return p


def list_cells(root: Path | None = None) -> list[tuple[str, str]]:
    root = root or cache_root()
    out = []
    if not root.exists():
        return out
    for coll in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(coll.glob("*.npz")):
            if ".phrases_" in f.name:
                continue
            out.append((coll.name, f.stem))
    known = []                                                   # the registergap pipeline's flat files: <collection>_<model>.npz, both sides may hold underscores
    for k, v in _encoders().items():
        if isinstance(v, dict):
            known += [k] + ([v["legacy_model"]] if v.get("legacy_model") else [])
    known = sorted(set(known), key=len, reverse=True)
    for f in sorted(root.glob("*_*.npz")):
        stem = f.stem
        enc = next((k for k in known if stem.endswith("_" + k)), None)
        if enc is None:
            continue                                             # concatenations, tsne dumps and other non-cells are skipped
        out.append((stem[: -len(enc) - 1], enc))
    return out


def patches_path(collection: str, encoder: str, root: Path | None = None) -> Path:
    return (root or cache_root()) / collection / f"{encoder_key(encoder)}.patches.npy"


def phrases_path(collection: str, encoder: str, cap: int = 12, verbs: bool = False, root: Path | None = None) -> Path:
    sfx = (f"_p{cap}" if cap != 12 else "") + ("v" if verbs else "")
    return (root or cache_root()) / collection / f"{encoder_key(encoder)}.phrases{sfx}.npz"


def models_dir(collection: str, encoder: str, root: Path | None = None) -> Path:
    """Saved seed-0 map checkpoints (.pt, git-ignored) next to the results of the cell."""
    p = (root or results_root()) / collection / encoder_key(encoder) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p
