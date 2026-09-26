"""LoRA fine-tuning of the encoder as the comparator for the learned maps.

Fine-tunes CLIP ViT-L/14 (LAION-2B; one of the grid's four generic encoders)
with LoRA on both towers, on the SAME catalog pairs, SAME text->image
in-batch InfoNCE at tau=0.05, SAME val-R@50 model selection and SAME test
gallery as the maps, then reports side by side:

  * raw encoder, LoRA encoder, linear maps on the raw embeddings (at the LoRA
    batch size and at the grid's 256), and maps on top of the LoRA embeddings;
  * trainable parameters, wall-clock and peak GPU memory for each;
  * forgetting: COCO test R@k under the raw and the LoRA encoder.

Sized for a 10 GB card: bf16 autocast, gradient checkpointing, batch 64.
Images are loaded from the dataset paths by the pipeline's own split loader,
so the pairs are exactly the grid's pairs.

Outputs
  results/lora/<ds>/lora_<tag>[_n<limit>].json           all numbers
  results/lora/<ds>/adapter_<tag>[_n<limit>]/            the LoRA adapter (peft)
  results/<ds>/{train,val,test}_embs_<tag>_lora[_n<limit>].npz   LoRA embeddings, in
      the cache format, so ablations/symmetric_projection.py --mtag <tag>_lora works on top

Usage (from the iclr2027 folder):
  python ablations/lora_finetune.py --dataset nwpu
  python ablations/lora_finetune.py --dataset rocov2 --limit-train 20000
  python ablations/lora_finetune.py --dataset nwpu --limit-train 1000 --epochs 30
Requires: transformers, peft (pip install peft), torch with CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve()
# Two packages both called `config`: the legacy pipeline's (needed by `splits`
# and by symmetric_projection's out_path) is the one on sys.path; the
# iclr2027 one is loaded under its own name from its file.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("iclr_config", HERE.parents[1] / "config.py")
iclr_config = _ilu.module_from_spec(_spec); _spec.loader.exec_module(iclr_config)
RESULTS_ROOT, out_path = iclr_config.RESULTS_ROOT, iclr_config.out_path
LEGACY = Path(os.environ.get("ICLR_LEGACY_PIPELINE", HERE.parents[1] / "legacy"))
# Order matters: `splits` must be imported FIRST so that the module named
# `config` in sys.modules is the legacy one (symmetric_projection later does
# `from config import out_path`, which the legacy config also provides).
sys.path.insert(0, str(LEGACY))                          # legacy pipeline: splits, config
from splits import load_split_pairs                       # noqa: E402
sys.path.insert(0, str(HERE.parent))                     # iclr2027/ablations (symmetric_projection)
from symmetric_projection import train_arm, recall_table  # noqa: E402


def load_split(ds, split, mtag):
    """Cached embeddings (unit-norm caps, imgs) via the iclr2027 cache resolution."""
    import glob as _glob
    tag = f"_{mtag}" if mtag else ""
    exact = iclr_config.CACHE_ROOT / ds / f"{split}_embs{tag}.npz"
    if not exact.exists():
        cands = sorted(_glob.glob(str(iclr_config.CACHE_ROOT / ds / f"{split}_embs{tag}_n*.npz")),
                       key=lambda p: -int(p.rsplit("_n", 1)[1].split(".")[0]))
        if not cands:
            raise FileNotFoundError(f"no cached {split} embeddings for {ds}/{mtag or 'default'}")
        exact = Path(cands[0])
    z = np.load(exact)
    return z["caps"].astype(np.float32), z["imgs"].astype(np.float32)

MODEL_ID = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
TAG = "clip-vitl14-laion2b"
TAU = 0.05


# ── data ─────────────────────────────────────────────────────────────────────

class PairSet(Dataset):
    def __init__(self, paths, token_ids, attn, processor):
        self.paths, self.ids, self.attn, self.proc = paths, token_ids, attn, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px, self.ids[i], self.attn[i]


class ImageSet(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.proc = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


def rocov2_pairs_fast(split):
    """Same pairs, same order and same drop rule as splits._rocov2_pairs, but
    without decoding every figure: rows are read with the image column
    undecoded and a figure is decoded only if its cached jpg is missing."""
    from datasets import load_dataset, Image as HFImage
    import io
    from splits import _safe_id, _norm_split
    import config as legacy_config
    cache_dir = legacy_config.ROOT / "cache"
    image_dir = cache_dir / "rocov2" / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    hf_split = _norm_split("rocov2", split)          # 'val' -> 'validation', as splits does
    ds = load_dataset("eltorio/ROCOv2-radiology", split=hf_split, cache_dir=str(cache_dir))
    ds = ds.cast_column("image", HFImage(decode=False))
    texts, paths = [], []
    for row in ds:
        caption = str(row.get("caption", "") or "").strip()
        if not caption:
            continue
        local = image_dir / f"{_safe_id(str(row['image_id']))}.jpg"
        if not local.exists():
            raw = row["image"]
            img = Image.open(io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]).convert("RGB")
            img.thumbnail((512, 512)); img.save(local, format="JPEG", quality=90)
        texts.append(caption); paths.append(local)
    print(f"[lora] rocov2 {split}: {len(texts)} pairs (cached figures)", flush=True)
    return texts, paths


def pairs(ds, split, **kw):
    return rocov2_pairs_fast(split) if ds == "rocov2" else load_split_pairs(ds, split, **kw)


def subsample_idx(n, limit, seed):
    """Same rule as query_projection.get_split_arrays: seeded permutation, sorted."""
    if limit is None or limit >= n:
        return np.arange(n)
    idx = np.random.default_rng(seed).permutation(n)[:limit]
    return np.sort(idx)


# ── model ────────────────────────────────────────────────────────────────────

def build(rank, alpha, dropout, towers, dev):
    from transformers import CLIPModel, CLIPProcessor
    from peft import LoraConfig, get_peft_model
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad_(False)
    scope = {"both": r".*(text_model|vision_model).*\.(q_proj|k_proj|v_proj|out_proj)$",
             "text": r".*text_model.*\.(q_proj|k_proj|v_proj|out_proj)$",
             "image": r".*vision_model.*\.(q_proj|k_proj|v_proj|out_proj)$"}[towers]
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=scope)
    model = get_peft_model(model, cfg).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return model, processor, n_train, n_total


def text_feats(model, ids, attn):
    """Projected text features, explicit so it works across transformers versions
    (get_text_features returns a tensor in some versions and a ModelOutput in others)."""
    pooled = model.text_model(input_ids=ids, attention_mask=attn).pooler_output
    return model.text_projection(pooled)


def image_feats(model, px):
    pooled = model.vision_model(pixel_values=px).pooler_output
    return model.visual_projection(pooled)


@torch.no_grad()
def embed_texts(model, ids, attn, dev, bs=256):
    out = []
    model.eval()
    for i in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = text_feats(model, ids[i:i + bs].to(dev), attn[i:i + bs].to(dev))
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embed_images(model, paths, processor, dev, bs=128, workers=4):
    out = []
    model.eval()
    dl = DataLoader(ImageSet(paths, processor), batch_size=bs, num_workers=workers, pin_memory=True)
    for px in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = image_feats(model, px.to(dev, non_blocking=True))
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def r_at(q, g):
    t = recall_table(q, g, np.arange(len(g)))
    return {k: round(float(t[k]), 4) for k in ("R@1", "R@10", "R@50")}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--towers", choices=("both", "text", "image"), default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--coco-limit", type=int, default=5000)
    ap.add_argument("--no-coco", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    suffix = f"_n{args.limit_train}" if args.limit_train else ""
    out_dir = RESULTS_ROOT / "lora" / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {"dataset": ds, "encoder": TAG, "model_id": MODEL_ID, "seed": args.seed,
           "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "towers": args.towers,
                    "targets": "q,k,v,out projections", "epochs": args.epochs, "batch": args.batch,
                    "lr": args.lr, "tau": TAU, "loss": "text->image in-batch InfoNCE",
                    "precision": "bf16 autocast + gradient checkpointing"}}

    # pairs: exactly the grid's pairs (same loader, same seeded subsample rule)
    tr_texts, tr_paths = pairs(ds, "train")
    va_texts, va_paths = pairs(ds, "val")
    te_texts, te_paths = pairs(ds, "test")
    idx = subsample_idx(len(tr_texts), args.limit_train, args.seed)
    tr_texts = [tr_texts[i] for i in idx]; tr_paths = [tr_paths[i] for i in idx]
    rep["n_train"], rep["n_val"], rep["n_test"] = len(tr_texts), len(va_texts), len(te_texts)
    print(f"[lora] {ds}: train {len(tr_texts)} val {len(va_texts)} test {len(te_texts)}", flush=True)

    model, processor, n_trainable, n_total = build(args.rank, args.alpha, args.dropout, args.towers, dev)
    rep["params_trainable"], rep["params_total"] = n_trainable, n_total
    tok = lambda texts: processor.tokenizer(texts, padding="max_length", truncation=True,  # noqa: E731
                                            max_length=77, return_tensors="pt")
    tr_tok, va_tok, te_tok = tok(tr_texts), tok(va_texts), tok(te_texts)

    # raw encoder (HF weights) on the test gallery, for the in-script baseline
    t0 = time.time()
    raw_te_c = embed_texts(model, te_tok["input_ids"], te_tok["attention_mask"], dev)
    raw_te_i = embed_images(model, te_paths, processor, dev, workers=args.workers)
    rep["raw_hf"] = r_at(raw_te_c, raw_te_i)
    print(f"[lora] raw (HF weights, LoRA at zero) test {rep['raw_hf']}", flush=True)

    # ── train ────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    dl = DataLoader(PairSet(tr_paths, tr_tok["input_ids"], tr_tok["attention_mask"], processor),
                    batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=args.workers > 0)
    va_c_ids, va_c_attn = va_tok["input_ids"], va_tok["attention_mask"]
    best_r50, best_state, best_epoch = -1.0, None, -1
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time(); curve = []
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for px, ids, attn in dl:
            px, ids, attn = px.to(dev, non_blocking=True), ids.to(dev), attn.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ft = text_feats(model, ids, attn)
                fi = image_feats(model, px)
            ft = torch.nn.functional.normalize(ft.float(), dim=-1)
            fi = torch.nn.functional.normalize(fi.float(), dim=-1)
            logits = ft @ fi.T / TAU
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ft), device=dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        va_c = embed_texts(model, va_c_ids, va_c_attn, dev)
        va_i = embed_images(model, va_paths, processor, dev, workers=args.workers)
        r50 = float(recall_table(va_c, va_i, np.arange(len(va_i)))["R@50"])
        curve.append({"epoch": ep + 1, "loss": round(tot / max(nb, 1), 4), "val_R50": round(r50, 4)})
        print(f"[lora] epoch {ep + 1}/{args.epochs} loss {tot / max(nb, 1):.3f} val R@50 {r50:.4f}", flush=True)
        if r50 > best_r50:
            best_r50, best_epoch = r50, ep + 1
            best_state = {k: v.detach().clone().cpu() for k, v in get_peft_model_state_dict(model).items()}
    rep["train"] = {"wall_clock_s": round(time.time() - t_train, 1),
                    "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "best_epoch": best_epoch, "best_val_R50": round(best_r50, 4), "curve": curve}
    set_peft_model_state_dict(model, best_state)
    model.save_pretrained(out_dir / f"adapter_{TAG}{suffix}")

    # ── LoRA embeddings for every split, in the cache format ─────────────
    t1 = time.time()
    embs = {}
    for split, texts_tok, paths in (("train", tr_tok, tr_paths), ("val", va_tok, va_paths), ("test", te_tok, te_paths)):
        c = embed_texts(model, texts_tok["input_ids"], texts_tok["attention_mask"], dev)
        i = embed_images(model, paths, processor, dev, workers=args.workers)
        embs[split] = (c, i)
        np.savez_compressed(out_path(f"{split}_embs_{TAG}_lora{suffix}.npz", ds), caps=c, imgs=i)
    rep["reembed_wall_clock_s"] = round(time.time() - t1, 1)
    rep["lora"]["test"] = r_at(*embs["test"])
    print(f"[lora] LoRA test {rep['lora']['test']}", flush=True)

    # ── the maps, same pairs, for comparison ─────────────────────────────
    tr_c_raw, tr_i_raw = load_split(ds, "train", TAG)
    va_c_raw, va_i_raw = load_split(ds, "val", TAG)
    te_c_raw, te_i_raw = load_split(ds, "test", TAG)
    rep["raw_cached"] = r_at(te_c_raw, te_i_raw)
    sub_c, sub_i = tr_c_raw[idx], tr_i_raw[idx]
    rep["maps"] = {}
    for label, batch in (("maps_batch%d" % args.batch, args.batch), ("maps_batch256", 256)):
        torch.cuda.reset_peak_memory_stats(); t2 = time.time()
        f, g = train_arm("both", sub_c, sub_i, va_c_raw, va_i_raw, epochs=30, lr=1e-4, batch=batch,
                         temp=TAU, sym_loss=False, seed=args.seed, dev=dev)
        with torch.no_grad():
            q = f(torch.tensor(te_c_raw, device=dev)).cpu().numpy()
            gg = g(torch.tensor(te_i_raw, device=dev)).cpu().numpy()
        rep["maps"][label] = {**r_at(q, gg), "wall_clock_s": round(time.time() - t2, 1),
                              "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                              "params_trainable": int(2 * (te_c_raw.shape[1] ** 2 + te_c_raw.shape[1]))}
        print(f"[lora] {label}: test {rep['maps'][label]}", flush=True)
    # maps on top of the LoRA embeddings
    t3 = time.time()
    f, g = train_arm("both", embs["train"][0], embs["train"][1], embs["val"][0], embs["val"][1],
                     epochs=30, lr=1e-4, batch=256, temp=TAU, sym_loss=False, seed=args.seed, dev=dev)
    with torch.no_grad():
        q = f(torch.tensor(embs["test"][0], device=dev)).cpu().numpy()
        gg = g(torch.tensor(embs["test"][1], device=dev)).cpu().numpy()
    rep["maps"]["maps_on_lora"] = {**r_at(q, gg), "wall_clock_s": round(time.time() - t3, 1)}
    print(f"[lora] maps on top of LoRA: test {rep['maps']['maps_on_lora']}", flush=True)

    # ── forgetting: COCO under raw vs LoRA ───────────────────────────────
    if not args.no_coco:
        co_texts, co_paths = load_split_pairs("coco", "test", limit=args.coco_limit, seed=0)
        co_tok = tok(co_texts)
        lo_c = embed_texts(model, co_tok["input_ids"], co_tok["attention_mask"], dev)
        lo_i = embed_images(model, co_paths, processor, dev, workers=args.workers)
        # raw: disable the adapter
        with model.disable_adapter():
            ra_c = embed_texts(model, co_tok["input_ids"], co_tok["attention_mask"], dev)
            ra_i = embed_images(model, co_paths, processor, dev, workers=args.workers)
        rep["coco_forgetting"] = {"n": len(co_texts), "raw": r_at(ra_c, ra_i), "lora": r_at(lo_c, lo_i)}
        print(f"[lora] COCO raw {rep['coco_forgetting']['raw']} -> LoRA {rep['coco_forgetting']['lora']}", flush=True)

    rep["total_wall_clock_s"] = round(time.time() - t0, 1)
    p = out_dir / f"lora_{TAG}{suffix}.json"
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=1)
    print(f"[lora] -> {p}")


if __name__ == "__main__":
    main()
