"""SigLIP2 embedding wrapper (image + text towers).

Notes that matter for correctness:
  * SigLIP-family text encoders require padding="max_length" (64 tokens).
    Long captions are truncated — deliberately. The audit must measure
    findability under the SAME text encoding the live index uses.
  * Embeddings are L2-normalized, so cosine similarity == dot product.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import EMBED_MODEL, TEXT_MAX_TOKENS


def _to_tensor(out) -> torch.Tensor:
    """get_*_features returns a raw tensor in some transformers versions
    and a ModelOutput (with pooler_output) in others. Normalize that."""
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    return out.last_hidden_state[:, 0]


class SigLIP2Embedder:
    """Despite the name, works for any dual-encoder with get_*_features
    (SigLIP2, CLIP ViT-L, ...). Pass text_max_tokens to match the model:
    64 for SigLIP family, 77 for CLIP."""

    def __init__(self, model_id: str = EMBED_MODEL, device: str | None = None,
                 text_max_tokens: int = TEXT_MAX_TOKENS):
        from transformers import AutoModel, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_max_tokens = text_max_tokens
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[embedder] loading {model_id} on {self.device} ({dtype})")
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def embed_images(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        chunks = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                img = Image.open(p).convert("RGB")
                img.thumbnail((512, 512))  # model resizes to 384 anyway
                batch.append(img)
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = _to_tensor(self.model.get_image_features(**inputs))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy())
            print(f"\r[embedder] images {min(i + batch_size, len(paths))}/{len(paths)}",
                  end="", flush=True)
        print()
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=texts[i:i + batch_size],
                padding="max_length",
                max_length=self.text_max_tokens,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            feats = _to_tensor(self.model.get_text_features(**inputs))
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy())
        return np.concatenate(chunks, axis=0)


def rank_of_target(query_embs: np.ndarray, gallery_embs: np.ndarray,
                   target_idx: np.ndarray) -> np.ndarray:
    """For each query, the 1-indexed rank of its target gallery image.

    rank = 1 + (number of gallery images scoring strictly higher than the
    target). Ties resolve optimistically; with float similarities exact
    ties are vanishingly rare.
    """
    sims = query_embs @ gallery_embs.T                  # (Q, G)
    target_sims = sims[np.arange(len(target_idx)), target_idx]
    return 1 + (sims > target_sims[:, None]).sum(axis=1)
