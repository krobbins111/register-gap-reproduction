"""Embedding backbones — one factory so step8/step9 can swap the encoder.

The projection method is backbone-agnostic: `QueryProjection` operates on whatever
vector the encoder emits, so swapping the encoder *is* the generality test.
The same free linear map is learned on top of a generic SigLIP2 space, a
medical SigLIP space (MedSigLIP), or a PMC-pretrained CLIP space (BiomedCLIP);
whatever the encoder can separate, the projection then re-aligns the query
register onto. ROCOv2's weak vanilla numbers under generic SigLIP2 are largely
an *image-side* collapse (radiology sits in a tiny cone), which a domain
encoder fixes directly — so the headline becomes "the method rides the
embedding space."

Backends
--------
    hf         transformers AutoModel with get_image_features/get_text_features
               — SigLIP2 (default), SigLIP v1, MedSigLIP, CLIP, MetaCLIP-2.
               Reuses the existing embedder.SigLIP2Embedder unchanged.
    open_clip  open_clip models. Two sub-modes:
                 * hub package  — create_model_from_pretrained("hf-hub:ID")
                   (BiomedCLIP, laion CLIP, BioCLIP/BioCLIP-2, timm SigLIP v1).
                 * state_dict   — build a base arch then load a bare .pt from a
                   HF repo that ships raw checkpoints, not an open_clip package
                   (RemoteCLIP, GeoRSCLIP). See OPENCLIP_STATEDICT below.
               Needs `pip install open_clip_torch`. Lazily imported, so
               SigLIP/MedSigLIP users never need it.

Known ids resolve via REGISTRY -> (backend, text_max_tokens, short_tag). The
short_tag is "" for the default backbone (so existing caches / SemArt numbers /
delta_<arch>.pt names are untouched) and a slug otherwise, used to key the
embedding caches, checkpoints, and reports per backbone. Unknown ids fall back
to the hf backend; override token length with --text-max-tokens.

Setup notes
-----------
    MedSigLIP  gated: accept the Health AI Developer Foundations terms at
               https://huggingface.co/google/medsiglip-448 and
               `huggingface-cli login` once. 0.9B params, fp16 fits a 3080.
    BiomedCLIP open (MIT): `pip install open_clip_torch`. ViT-B/16 + PubMedBERT,
               256-token context; trained on PMC-15M (ROCOv2's own source).

Grid backbones added Jul 2026 (all `pip install open_clip_torch`; friendly
--encoder aliases in ENCODER_ALIASES):
    laion CLIP-L/14  open_clip hub, 768-d, 77 tok. Generic LAION-2B anchor.
    SigLIP v1 so400m google/siglip-so400m-patch14-384 via hf, 1152-d, 64 tok.
                     (transformers SigLIP v1; distinct tag from the timm/INQUIRE
                     open_clip v1 so caches never collide.)
    MetaCLIP-2 ww    facebook/metaclip-2-worldwide-huge-quickgelu via hf, 1024-d,
                     77 tok. REQUIRES transformers >= 4.56 (MetaClip2 arch); if
                     AutoModel can't load it, upgrade transformers. CC-BY-NC.
    BioCLIP / -2     imageomics/bioclip (512-d) and bioclip-2 (768-d), open_clip
                     hub, 77 tok. Taxonomy-trained; ecology/biology galleries.
    RemoteCLIP-L/14  state_dict: base ViT-L-14 + chendelong/RemoteCLIP
                     -> RemoteCLIP-ViT-L-14.pt (clean full load), 768-d, 77 tok.
    GeoRSCLIP-B/32   state_dict: base ViT-B-32 (openai init) + Zilun/GeoRSCLIP
                     -> ckpt/RS5M_ViT-B-32.pt (strict=False per authors), 512-d,
                     77 tok. RS5M-pretrained ONLY (no RSICD/RSITMD finetune) so
                     it stays leakage-safe if those datasets join the grid.
    (Art backbone intentionally deferred — no mature art dual-encoder on HF;
     art galleries run generic-only for now.)
"""
from __future__ import annotations

import re

import numpy as np
import torch

from config import EMBED_MODEL, TEXT_MAX_TOKENS

# embed_model id -> (backend, text_max_tokens, short_tag)
REGISTRY: dict[str, tuple[str, int, str]] = {
    EMBED_MODEL: ("hf", TEXT_MAX_TOKENS, ""),               # default, no tag
    "google/siglip2-so400m-patch16-384": ("hf", 64, ""),
    "google/medsiglip-448": ("hf", 64, "medsiglip"),
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        ("open_clip", 256, "biomedclip"),
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        ("open_clip", 256, "biomedclip"),
    # SigLIP **v1** SO400M-14@384 (open_clip / timm webli weights) — the encoder
    # INQUIRE used for its pre-computed gallery shards. Distinct space from the
    # default SigLIP2, so it carries its own tag (artifacts never collide).
    "hf-hub:timm/ViT-SO400M-14-SigLIP-384": ("open_clip", 64, "siglipv1so400m"),
    "hf-hub:timm/ViT-SO400M-14-SigLIP": ("open_clip", 64, "siglipv1so400m"),

    # --- Grid backbones (Jul 2026) -------------------------------------------
    # Generic:
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": ("open_clip", 77, "clip-vitl14-laion2b"),
    "google/siglip-so400m-patch14-384": ("hf", 64, "siglipv1-so400m-384"),
    "facebook/metaclip-2-worldwide-huge-quickgelu": ("hf", 77, "metaclip2-ww-huge"),
    # Biology / ecology (open_clip hub packages):
    "imageomics/bioclip": ("open_clip", 77, "bioclip"),
    "imageomics/bioclip-2": ("open_clip", 77, "bioclip2"),
    # Remote sensing (open_clip state_dict — see OPENCLIP_STATEDICT). The synthetic
    # ids below pick one checkpoint out of each multi-checkpoint HF repo.
    "chendelong/RemoteCLIP-ViT-L-14": ("open_clip", 77, "remoteclip-vitl14"),
    "Zilun/GeoRSCLIP-ViT-B-32": ("open_clip", 77, "georsclip-vitb32"),
    # Fashion / e-commerce (added with the fashion datasets). FashionCLIP is a
    # transformers CLIP fine-tune; Marqo-FashionSigLIP is an open_clip SigLIP
    # fine-tune (64-tok SigLIP text tower, comparable to the SigLIP2 default).
    "patrickjohncyh/fashion-clip": ("hf", 77, "fashionclip"),
    "Marqo/marqo-fashionSigLIP": ("open_clip", 64, "marqo-fashionsiglip"),
}

# open_clip backbones whose HF repo ships a BARE .pt state_dict (not an
# open_clip package), so they must be built from a base arch then loaded.
#   id -> (open_clip_arch, hf_repo, ckpt_filename, init_pretrained, strict)
# init_pretrained=None => random-init base then full load (RemoteCLIP: clean).
# init_pretrained="openai"/"laion..." => init the base with those weights then
#   overlay the checkpoint (GeoRSCLIP: the authors' recipe, strict=False).
OPENCLIP_STATEDICT: dict[str, tuple[str, str, str, str | None, bool]] = {
    "chendelong/RemoteCLIP-ViT-L-14": (
        "ViT-L-14", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt",
        None, True),
    "Zilun/GeoRSCLIP-ViT-B-32": (
        "ViT-B-32", "Zilun/GeoRSCLIP", "ckpt/RS5M_ViT-B-32.pt",
        "openai", False),
}

# Friendly short names (e.g. --encoder siglip-so400m-14-384) -> loadable id. An
# unknown name is returned unchanged, so --encoder also accepts a full model id.
ENCODER_ALIASES: dict[str, str] = {
    "siglip-so400m-14-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
    "ViT-SO400M-14-SigLIP-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
    "ViT-SO400M-14-SigLIP": "hf-hub:timm/ViT-SO400M-14-SigLIP",
    # Grid backbones (Jul 2026) — short names for --encoder.
    "clip-l-14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "siglip-v1-so400m": "google/siglip-so400m-patch14-384",
    "metaclip2": "facebook/metaclip-2-worldwide-huge-quickgelu",
    "bioclip": "imageomics/bioclip",
    "bioclip-2": "imageomics/bioclip-2",
    "remoteclip": "chendelong/RemoteCLIP-ViT-L-14",
    "georsclip": "Zilun/GeoRSCLIP-ViT-B-32",
    "fashionclip": "patrickjohncyh/fashion-clip",
    "marqo-fashionsiglip": "Marqo/marqo-fashionSigLIP",
}


def resolve_encoder(name: str) -> str:
    """Map a short --encoder name onto a make_embedder-loadable model id."""
    return ENCODER_ALIASES.get(name, name)


def _slug(model_id: str) -> str:
    """Short filesystem-safe tag from a model id's last path component."""
    base = model_id.split("hf-hub:")[-1].rstrip("/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-").lower()
    return base[:24]


def resolve_backbone(embed_model: str) -> tuple[str, int, str]:
    if embed_model in REGISTRY:
        return REGISTRY[embed_model]
    low = embed_model.lower()
    if "biomedclip" in low or embed_model.startswith("hf-hub:") or \
            "open_clip" in low:
        return ("open_clip", 256, _slug(embed_model))
    tag = "" if embed_model == EMBED_MODEL else _slug(embed_model)
    return ("hf", TEXT_MAX_TOKENS, tag)


def model_tag(embed_model: str) -> str:
    """Cache/report key for a backbone ("" for the default SigLIP2)."""
    return resolve_backbone(embed_model)[2]


def make_embedder(embed_model: str, *, text_max_tokens: int | None = None,
                  device: str | None = None):
    """Return an encoder exposing embed_texts(list)->ndarray and
    embed_images(paths, batch_size)->ndarray, both L2-normalized — the same
    interface step8/step9 already rely on."""
    backend, default_toks, _ = resolve_backbone(embed_model)
    toks = text_max_tokens or default_toks
    if backend == "open_clip":
        return OpenCLIPEmbedder(embed_model, text_max_tokens=toks, device=device)
    from embedder import SigLIP2Embedder
    return SigLIP2Embedder(embed_model, device=device, text_max_tokens=toks)


def _load_state_dict(repo: str, filename: str) -> dict:
    """Download a bare checkpoint from a HF repo and return a clean state_dict.

    Handles the two things that trip up open_clip state_dict repos: a possible
    {'state_dict': ...} / {'model': ...} wrapper, and a 'module.' DataParallel
    prefix. torch>=2.6 defaults weights_only=True; these are plain tensor dicts
    so that path is preferred, with a trusted-source fallback for older pickles.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # pragma: no cover - checkpoint carries non-tensor objects
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


class OpenCLIPEmbedder:
    """open_clip backend (BiomedCLIP and friends). Mirrors SigLIP2Embedder:
    L2-normalized float32 embeddings, paths streamed one batch at a time."""

    def __init__(self, model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-"
                 "vit_base_patch16_224", device: str | None = None,
                 text_max_tokens: int = 256):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = text_max_tokens

        spec = OPENCLIP_STATEDICT.get(model_id)
        if spec is not None:
            # Bare-state_dict repo (RemoteCLIP / GeoRSCLIP): build the base arch,
            # optionally init it with public weights, then overlay the checkpoint.
            arch, repo, ckpt_file, init_pretrained, strict = spec
            print(f"[embedder] building open_clip {arch} for {model_id} "
                  f"(state_dict, init={init_pretrained}) on {self.device}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=init_pretrained)
            self.tokenizer = open_clip.get_tokenizer(arch)
            state = _load_state_dict(repo, ckpt_file)
            msg = self.model.load_state_dict(state, strict=strict)
            missing = getattr(msg, "missing_keys", None)
            if missing is not None:
                print(f"[embedder] loaded {ckpt_file}: "
                      f"{len(msg.missing_keys)} missing / "
                      f"{len(msg.unexpected_keys)} unexpected keys")
        else:
            hub = model_id if model_id.startswith("hf-hub:") else \
                f"hf-hub:{model_id}"
            print(f"[embedder] loading {hub} via open_clip on {self.device}")
            self.model, self.preprocess = \
                open_clip.create_model_from_pretrained(hub)
            self.tokenizer = open_clip.get_tokenizer(hub)

        self.model.to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    @torch.no_grad()
    def embed_images(self, paths, batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        chunks = []
        for i in range(0, len(paths), batch_size):
            batch = [self.preprocess(Image.open(p).convert("RGB"))
                     for p in paths[i:i + batch_size]]
            x = torch.stack(batch).to(self.device)
            if self.device == "cuda":
                x = x.half()
            feats = self.model.encode_image(x)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy())
            print(f"\r[embedder] images {min(i + batch_size, len(paths))}/"
                  f"{len(paths)}", end="", flush=True)
        print()
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def embed_texts(self, texts, batch_size: int = 64) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(texts[i:i + batch_size],
                                  context_length=self.context_length
                                  ).to(self.device)
            feats = self.model.encode_text(toks)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy())
        return np.concatenate(chunks, axis=0)
