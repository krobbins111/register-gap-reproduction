"""Shared defaults for the indexability pipeline.

Everything is overridable via CLI flags; this file just centralizes paths
and model IDs so the four step scripts agree with each other.
"""
from __future__ import annotations

import os
import sys as _sys
from pathlib import Path

# Windows consoles (and redirected log files) default to cp1252, which raises
# UnicodeEncodeError on the non-Latin-1 characters that appear in some captions
# and species names (e.g. the Hawaiian okina U+02BB, en-dashes, accented Latin).
# Every pipeline script imports config, so forcing UTF-8 with replacement here
# means a stray print can never abort a run. No-op where stdout can't reconfigure.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - redirected/closed stream, etc.
        pass

# Some galleries contain legitimately huge scans (TreeOfLife herbarium sheets run
# to 180M+ pixels), which trip PIL's decompression-bomb guard when the embedder
# OPENS them (before it can downscale). These are trusted, so disable the cap
# process-wide. Every pipeline script imports config, so this covers them all.
try:
    import PIL.Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = None
except Exception:  # noqa: BLE001 - PIL missing (non-image path)
    pass

# Repo root = parent of this directory.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("iclr_roots", Path(__file__).resolve().parents[1] / "iclr_roots.py")
_roots = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_roots)
ROOT = Path(__file__).resolve().parent.parent
# raw datasets (HF downloads, materialised images, local zips): $ICLR_DATA_DIR, else <ltg store>/raw. Never inside the repo.
DATA_DIR = Path(os.environ.get("ICLR_DATA_DIR", str(_roots.store_root() / "raw")))

# SemArt data dir (the folder containing semart_test.csv and Images/).
# The ZIP extracts as SemArt/SemArt/, hence the nested default.
SEMART_DIR = DATA_DIR / "SemArt" / "SemArt"

# All pipeline outputs live here.
OUT_DIR = _roots.legacy_cache_root()      # embeddings written here: <ltg store>/legacy/<dataset>/<split>_embs[_<mtag>].npz

# Embedding model — matches the rest of the repo so numbers are comparable.
EMBED_MODEL = "google/siglip2-so400m-patch16-384"

# INQUIRE is the exception: its gallery embeddings are INQUIRE's OWN pre-computed
# SigLIP **v1** SO400M-14@384 shards (so we never touch the 441 GB of images), so
# its training pairs and queries must be embedded with the SAME v1 encoder — NOT
# SigLIP2. This short name resolves to an open_clip id in backbones.resolve_encoder.
INQUIRE_ENCODER = "siglip-so400m-14-384"

# Where the user downloads INQUIRE's pre-computed embedding folder (Google Drive,
# https://drive.google.com/drive/folders/1remNGZdc08B7i-Xg3oAaY68fnJ3QXyWm). The
# expected layout under it is embs/<encoder>/img_emb/img_emb_*.npy (+ metadata/).
INQUIRE_EMBS_DIR = DATA_DIR / "inquire_embs"

# NWPU-Captions: 31,500 aerial/satellite images, 45 scene categories.
# dataset_nwpu.json lives in the indexability root alongside the source files.
# Images are extracted from NWPU_images.tar.gz (Git LFS) into NWPU_IMAGES_DIR.
NWPU_IMAGES_DIR = DATA_DIR / "nwpu_images"
NWPU_IMAGES_TAR = Path(__file__).resolve().parent / "NWPU_images.tar.gz"

# Stamped at the top of every INQUIRE eval/geometry report so the cross-encoder
# caveat travels with the numbers (INQUIRE = v1, SemArt/ROCOv2 = SigLIP2).
PAPER_DISCLAIMER = (
    "Gallery embeddings use SigLIP SO400M-14@384 (v1) pre-computed by [INQUIRE "
    "benchmark]; training pairs embedded with the same encoder. SemArt and "
    "ROCOv2 experiments use SigLIP2-SO400M-patch16-384.")

# Captioner. Anything loadable via AutoModelForImageTextToText works.
# Hardware guide:
#   - Colab A100 40GB: Qwen3.5-9B in fp16 (default); Qwen3.5-27B with
#     --load-in-4bit (or fp16 if you land an 80GB A100)
#   - 3080 10GB:       Qwen3.5-4B fp16, or Qwen3-VL-8B with --load-in-4bit
CAPTION_MODEL = "Qwen/Qwen3.5-9B"

# SigLIP text towers truncate at 64 tokens. Captions longer than this get
# cut off at audit time — which is correct: the audit must see exactly what
# the index sees. We also pass this to the captioner prompt as guidance.
TEXT_MAX_TOKENS = 64

# K values for indexability@K and for failure labels at the gate.
K_VALUES = (1, 5, 10, 50)

# The two caption styles. Style name -> prompt.
CAPTION_PROMPTS = {
    "generic": (
        "Describe this image in one short sentence, the way a casual "
        "viewer would."
    ),
    "specific": (
        "Write a search-index caption for this image in AT MOST 40 "
        "words. Lead with the most distinctive identifying detail, then "
        "main subject, setting, and dominant colors. Be concrete and "
        "visual; no filler phrases, no speculation about who made it."
    ),
    # Probes findability under an interpretive, non-visual query register
    # (the SemArt situation). Still GT-free — the VLM sees only the image.
    # Off by default; enable with step2 --styles generic,specific,curatorial
    "curatorial": (
        "Write one sentence (AT MOST 40 words) about this image in the "
        "style of an art catalogue commentary: interpretive, allusive, "
        "focused on mood, theme, and significance rather than literal "
        "visual description. Do not name an artist or title."
    ),
    # Radiology counterpart of 'curatorial': the contextual register a
    # clinician would use. Used as the ROCOv2 'direct' card (cardgen.py).
    "clinical": (
        "Write a one-sentence radiology figure caption (AT MOST 40 words) as "
        "it would appear in a journal article: name the imaging modality (CT, "
        "MRI, X-ray, ultrasound, angiography, …), the anatomy, and the key "
        "finding. Be concise and clinical; no speculation beyond what is visible."
    ),
    # NWPU-Captions counterpart: a top-down aerial/satellite register. Used as
    # the NWPU 'direct' caption card.
    "aerial": (
        "Write a search-index caption for this aerial or satellite image in AT MOST 40 words. "
        "Describe the scene from directly above: identify the main land-use category, the key "
        "objects and their spatial layout (relative positions, densities, patterns). Be concrete "
        "and top-down; no filler phrases, no ground-level perspective."
    ),
}
DEFAULT_STYLES = ("generic", "specific")


def out_path(name: str, dataset: str = "semart") -> Path:
    d = OUT_DIR / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def semart_dir_from_env() -> Path:
    """Allow SEMART_DIR override via env var (useful on Colab)."""
    return Path(os.environ.get("SEMART_DIR", str(SEMART_DIR)))


def nwpu_images_dir_from_env() -> Path:
    """Allow NWPU_IMAGES_DIR override via env var."""
    return Path(os.environ.get("NWPU_IMAGES_DIR", str(NWPU_IMAGES_DIR)))


def inquire_embs_dir_from_env() -> Path:
    """Root of INQUIRE's downloaded pre-computed embedding folder (override via
    INQUIRE_EMBS_DIR env var)."""
    return Path(os.environ.get("INQUIRE_EMBS_DIR", str(INQUIRE_EMBS_DIR)))
