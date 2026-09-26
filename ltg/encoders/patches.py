"""Per-patch joint-space embeddings (registergap-pipeline; ported from registergap story/viz/patchemb.py and story/sets/extract.py).

SigLIP2: the image embedding is an attention-pooling head over the patch tokens, pooled = a + MLP(LN(a)), a = sum_i alpha_i out_proj(V tok_i).
Applying the head to one token alone gives e_i = a_i + MLP(LN(a_i)), the embedding the image would have had if the head attended only to
patch i; e_i lives in the joint space, so <e_i, text> is a query-conditioned heatmap. TIPSv2: the patch tokens are trained to align with
text directly, so they are used as they are. Either way the per-patch unit vectors are averaged over k x k blocks to the requested grid
(24 x 24 -> 12 x 12 for so400m/16-384, 32 x 32 -> 16 x 16 for TIPSv2-B/14, 16 x 16 as is for SigLIP2 base) and renormalised, which is the
pooling train_sets.py applied to the base stores.

    extract(collection, encoder, grid) -> ltg.cache.patches_path(collection, encoder): (n, G*G, d) fp16 unit-norm, rows in cell order,
    with the pooled-vs-cell cosine check (>= 0.999) that guards the ordering, like extract.py.
usage: python -m ltg.encoders.patches --collection skincap --encoder siglip2-so400m-16-384 --grid 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import encoder_key, load_cell, patches_path  # noqa: E402

ENC = json.load(open(Path(__file__).resolve().parents[2] / "configs" / "encoders.json", encoding="utf-8"))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _pool(tok: torch.Tensor, src: int, grid: int) -> torch.Tensor:
    """(B, src*src, d) unit patch vectors -> (B, grid*grid, d), k x k block means renormalised (identity when grid == src)."""
    if grid >= src:
        return torch.nn.functional.normalize(tok, dim=-1)
    if src % grid: raise ValueError(f"grid {grid} does not divide the native grid {src}")
    B, _, D = tok.shape; k = src // grid
    tok = torch.nn.functional.normalize(tok, dim=-1).reshape(B, grid, k, grid, k, D).mean((2, 4)).reshape(B, grid * grid, D)
    return torch.nn.functional.normalize(tok, dim=-1)


class PatchEmbedder:
    """SigLIP (2) family: pooling head applied per token. images() -> (E (n,G*G,d) fp16 unit, A (n, src*src) head attention, P (n,d) pooled unit)."""
    def __init__(self, mid: str, grid: int | None = None, dev: str = DEV):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.dev = dev; self.model = AutoModel.from_pretrained(mid).to(dev).eval()
        self.proc = AutoProcessor.from_pretrained(mid); self.tok = AutoTokenizer.from_pretrained(mid)
        self.vm = self.model.vision_model; self.src = self.vm.config.image_size // self.vm.config.patch_size; self.grid = grid or self.src

    @torch.no_grad()
    def images(self, ims, bs: int = 16):
        from PIL import Image
        E, A, P = [], [], []
        for s in range(0, len(ims), bs):
            chunk = [(Image.open(x) if isinstance(x, str) else x).convert("RGB") for x in ims[s:s + bs]]
            pi = self.proc(images=chunk, return_tensors="pt").to(self.dev)
            out = self.vm(pixel_values=pi["pixel_values"]); tok = out.last_hidden_state; B, N, D = tok.shape
            head = self.vm.head; probe = head.probe.repeat(B, 1, 1)
            _, w = head.attention(probe, tok, tok, need_weights=True, average_attn_weights=True)
            e = head(tok.reshape(B * N, 1, D)).reshape(B, N, D)
            E.append(_pool(e, self.src, self.grid).half().cpu().numpy()); A.append(w[:, 0].float().cpu().numpy())
            P.append(torch.nn.functional.normalize(out.pooler_output, dim=-1).float().cpu().numpy())
        return np.concatenate(E), np.concatenate(A), np.concatenate(P)

    @torch.no_grad()
    def texts(self, caps, bs: int = 128) -> np.ndarray:
        out = []
        for s in range(0, len(caps), bs):
            pt = self.tok(list(caps[s:s + bs]), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
            f = self.model.get_text_features(**pt); f = f if torch.is_tensor(f) else f.pooler_output
            out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class TipsPatchEmbedder:
    """TIPSv2: text-aligned patch tokens (after the class and register tokens), pooled to the grid; the class token is the image embedding."""
    def __init__(self, mid: str, grid: int | None = None, dev: str = DEV):
        from transformers import AutoModel, AutoProcessor
        self.dev = dev; self.model = AutoModel.from_pretrained(mid).to(dev).eval(); self.proc = AutoProcessor.from_pretrained(mid)
        self.vm = self.model.vision_model; self.n_extra = 1 + self.vm.config.num_register_tokens
        self.src = self.vm.config.image_size // self.vm.config.patch_size; self.grid = grid or 16

    @torch.no_grad()
    def images(self, ims, bs: int = 16):
        from PIL import Image
        E, A, P = [], [], []
        for s in range(0, len(ims), bs):
            chunk = [(Image.open(x) if isinstance(x, str) else x).convert("RGB") for x in ims[s:s + bs]]
            pi = self.proc(images=chunk, return_tensors="pt").to(self.dev); hs = self.vm(pixel_values=pi["pixel_values"]).last_hidden_state; B = hs.shape[0]
            E.append(_pool(hs[:, self.n_extra:], self.src, self.grid).half().cpu().numpy()); A.append(np.full((B, self.grid * self.grid), 1.0 / (self.grid * self.grid), np.float32))
            P.append(torch.nn.functional.normalize(hs[:, 0], dim=-1).float().cpu().numpy())
        return np.concatenate(E), np.concatenate(A), np.concatenate(P)

    @torch.no_grad()
    def texts(self, caps, bs: int = 128) -> np.ndarray:
        out = []
        for s in range(0, len(caps), bs):
            pt = self.proc(text=list(caps[s:s + bs]), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
            f = self.model.get_text_features(**pt); f = f.pooler_output if not torch.is_tensor(f) else f
            out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


DEFAULT_GRID = {"siglip2-base-16-256": 16, "siglip2-so400m-16-384": 12, "siglip2-so400m-14-384": 9, "tipsv2-b14": 16}


def get_embedder(encoder: str, grid: int | None = None, dev: str = DEV):
    key = encoder_key(encoder); info = ENC[key]; grid = grid or DEFAULT_GRID.get(key)
    return (TipsPatchEmbedder if info.get("family") == "tips" else PatchEmbedder)(info["id"], grid, dev)


def extract(collection: str, encoder: str, grid: int | None = None, bs: int = 48, force: bool = False) -> Path:
    from ltg.data.images import image_iter
    key = encoder_key(encoder); out = patches_path(collection, key)
    if out.exists() and not force:
        print("have", out); return out
    cell = load_cell(collection, key); n = len(cell.split); pe = get_embedder(key, grid)
    G = pe.grid; out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(".tmp.npy")
    E = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, G * G, cell.dim)); P = np.zeros((n, cell.dim), np.float32)
    t0 = time.time(); s = 0; buf = []
    def flush():
        nonlocal s, buf
        e, _, p = pe.images(buf, bs=16); E[s:s + len(buf)] = e; P[s:s + len(buf)] = p; s += len(buf); buf = []
    for im in image_iter(collection):
        buf.append(im)
        if len(buf) == bs:
            flush()
            if (s // bs) % 40 == 0: print(f"  {collection} {s}/{n}  {time.time() - t0:.0f}s  pooled-vs-cell cos {float((P[s - bs:s] * cell.img[s - bs:s]).sum(1).mean()):.4f}", flush=True)
    if buf: flush()
    E.flush(); del E
    cos = float((P[:s] * cell.img[:s]).sum(1).mean())
    if s != n or cos < 0.999:
        raise RuntimeError(f"{collection} x {key}: rows {s}/{n}, mean pooled-vs-cell cosine {cos:.4f}; the image order does not match the cell (left {tmp})")
    os.replace(tmp, out); print("saved", out, "grid", G, "mean pooled-vs-cell cos", round(cos, 4))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--encoder", required=True)
    ap.add_argument("--grid", type=int, default=None, help="pooled grid side (default: 16 for base and TIPSv2, 12 for so400m/16-384)")
    ap.add_argument("--force", action="store_true"); a = ap.parse_args(argv)
    extract(a.collection, a.encoder, a.grid, force=a.force)


if __name__ == "__main__":
    main()
