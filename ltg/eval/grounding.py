"""Grounding without boxes (registergap-pipeline; ported from registergap story/sets/ground.py, ground_flickr.py, ground_coco.py).

For every phrase, the argmax patch of <T(phrase), I(patch_j)> on the store's grid; a hit if the patch centre lies in the annotated box.
    box_hit(collection='cityflow'|'cityflow_cam', encoder, variants, tag)  CityFlow-NL: subject / other noun / verb phrases of the test
        captions against the described vehicle's box (data/cityflow_nl/pairs.jsonl: uuid, size, box, caption); chance = box area share.
    pointing_flickr(encoder, variants, tag)   Flickr30k Entities test: every phrase with a box, all five sentences
        (data/flickr30k_entities/entities_test.json); centre-patch and box-union chance baselines; accuracy by phrase type.
    pointing_coco(encoder, variants, tag)     COCO: caption noun phrases naming a category present in the image, against that category's
        instance boxes (data/coco_ann/annotations/instances_val2017.json).
Models are the saved seed-0 checkpoints of ltg.maps.sets (--save). Results: results/<collection>/<encoder>/grounding<tag>.json in the
ground_*.json schema. usage: python -m ltg.eval.grounding --task box_hit --collection cityflow --encoder siglip2-base-16-256 --variants set,setw
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import encoder_key, result_path  # noqa: E402
from ltg.maps.sets import DEV, SetMaps, load_all, load_model, pad_phrases, saved_grid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]; DATA = REPO / "data"


def _grid_mask(boxes_xywh, W, H, g):
    cx = (np.arange(g) + 0.5) * W / g; cy = (np.arange(g) + 0.5) * H / g; M = np.zeros((g, g), bool)
    for x, y, w, h in boxes_xywh: M |= (cy[:, None] >= y) & (cy[:, None] <= y + h) & (cx[None, :] >= x) & (cx[None, :] <= x + w)
    return M.reshape(-1)


def _models(collection, encoder, variants, tag, d):
    out = {"frozen": SetMaps(d, train_maps=False).to(DEV).eval()}
    for v in variants:
        try: out[v] = load_model(collection, encoder, v, tag, d)
        except FileNotFoundError: print("no saved model for", v, "(run ltg.maps.sets --save)")
    return out


def box_hit(collection="cityflow", encoder="siglip2-base-16-256", variants=("set", "setw"), tag="", cap=32, verbs=True, grid=None) -> Path:
    from ltg.encoders.phrases import nlp
    key = encoder_key(encoder); grid = grid or saved_grid(collection, key, variants, tag); sub, P, emb, offsets, texts = load_all(collection, key, grid, cap, verbs); d = sub["txt"].shape[1]; g = int(round(np.sqrt(P.shape[1])))
    pairs = {p["uuid"]: p for p in (json.loads(l) for l in open(DATA / "cityflow_nl" / "pairs.jsonl"))}; ids = [str(i) for i in sub["id"]]; te = np.where(sub["split"] == "test")[0]
    def kind(phrase, doc):
        chunks = [c.text.strip(" .,;:!#'\"") for c in doc.noun_chunks]
        if chunks and phrase == chunks[0]: return "subject"
        toks = [t for t in doc if t.text == phrase.split()[0]]
        return "verb" if toks and toks[0].pos_ == "VERB" else "other"
    docs = {i: nlp()(pairs[ids[i]]["caption"]) for i in te}
    @torch.no_grad()
    def hits(m):
        out = {k: [] for k in ("subject", "other", "verb")}; gate = {k: [] for k in out}; chance = []
        ph, phm = pad_phrases(emb, offsets, te, cap); ph_t, phm_t = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
        Tp = m.Tp(ph_t); w = m.phrase_weights(Tp, phm_t).cpu().numpy() if m.use_gate else None
        for b, i in enumerate(te):
            p = pairs[ids[i]]; W, H = p["size"]; M = _grid_mask([p["box"]], W, H, g); chance.append(M.mean())
            Ip = m.Ip(torch.tensor(np.asarray(P[i]).astype(np.float32), device=DEV)); sim = (Tp[b] @ Ip.T).cpu().numpy()
            s, e = offsets[i], min(offsets[i + 1], offsets[i] + ph.shape[1])
            for k in range(e - s):
                kd = kind(str(texts[s + k]), docs[i]); out[kd].append(float(M[int(sim[k].argmax())]))
                if w is not None: gate[kd].append(float(w[b, k]))
        res = {k: {"hit": float(np.mean(v)), "n": len(v)} for k, v in out.items() if v}
        if w is not None:
            for k in res: res[k]["gate_mean_weight"] = float(np.mean(gate[k]))
        res["chance_box_area"] = float(np.mean(chance)); return res
    R = {"collection": collection, "encoder": key, "grid": g, "task": "box_hit"}
    for name, m in _models(collection, key, variants, tag, d).items(): R[name] = hits(m); print(name, {k: round(v["hit"], 3) for k, v in R[name].items() if isinstance(v, dict)})
    out = result_path(collection, key, f"grounding{tag}.json"); json.dump(R, open(out, "w"), indent=1); print("->", out); return out


def pointing_flickr(encoder="tipsv2-b14", variants=("pooled", "set", "setw"), tag="", cap=32, grid=None) -> Path:
    from ltg.encoders.patches import get_embedder
    key = encoder_key(encoder); grid = grid or saved_grid("flickr", key, variants, tag); sub, P, emb, offsets, _ = load_all("flickr", key, grid, cap, False); d = sub["txt"].shape[1]; g = int(round(np.sqrt(P.shape[1])))
    ids = [str(i) for i in sub["id"]]; row_of = {i: k for k, i in enumerate(ids)}
    E = [e for e in json.load(open(DATA / "flickr30k_entities" / "entities_test.json")) if e["id"] in row_of and sub["split"][row_of[e["id"]]] == "test"]
    def mask_xyxy(boxes, W, H): return _grid_mask([(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes], W, H, g)
    items = []
    for e in E:
        W, H = e["size"]; allm = mask_xyxy([b for s in e["sentences"] for p in s["phrases"] for b in p["boxes"]], W, H)
        for s in e["sentences"]:
            for p in s["phrases"]:
                if p["boxes"]: items.append((row_of[e["id"]], p["text"], (p["types"] or ["?"])[0], mask_xyxy(p["boxes"], W, H), allm))
    pe = get_embedder(key); Tt = torch.tensor(pe.texts([t for _, t, _, _, _ in items], bs=256), device=DEV)
    types = [t for _, _, t, _, _ in items]; rows = np.array([r for r, *_ in items]); masks = np.stack([m for *_, m, _ in items]); allmasks = np.stack([m for *_, m in items]); c = g * (g // 2) + g // 2
    @torch.no_grad()
    def point(m):
        Tp = m.Tp(Tt); J = np.zeros(len(items), int)
        for r in np.unique(rows):
            Ip = m.Ip(torch.tensor(np.asarray(P[r]).astype(np.float32), device=DEV)); k = np.where(rows == r)[0]; J[k] = (Tp[k] @ Ip.T).argmax(1).cpu().numpy()
        hit = masks[np.arange(len(items)), J]; anyhit = allmasks[np.arange(len(items)), J]
        return {"pointing_acc": float(hit.mean()), "any_box_acc": float(anyhit.mean()), "by_type": {t: {"acc": float(hit[[tt == t for tt in types]].mean()), "n": int(sum(tt == t for tt in types))} for t in sorted(set(types))}}
    R = {"collection": "flickr", "encoder": key, "grid": g, "task": "pointing_game", "n_images": len(E), "n_phrases": len(items), "chance_box_area": float(masks.mean()), "center_patch_acc": float(masks[:, c].mean()), "center_patch_any_box": float(allmasks[:, c].mean())}
    for name, m in _models("flickr", key, variants, tag, d).items(): R[name] = point(m); print(name, round(R[name]["pointing_acc"], 3))
    out = result_path("flickr", key, f"grounding{tag}.json"); json.dump(R, open(out, "w"), indent=1); print("->", out); return out


def pointing_coco(encoder="siglip2-base-16-256", variants=("pooled", "set", "setw"), tag="", cap=32, split="test", grid=None) -> Path:
    from ltg.encoders.phrases import nlp as _nlp
    key = encoder_key(encoder); grid = grid or saved_grid("coco", key, variants, tag); sub, P, emb, offsets, texts = load_all("coco", key, grid, cap, False); d = sub["txt"].shape[1]; g = int(round(np.sqrt(P.shape[1]))); ids = [str(i) for i in sub["id"]]
    A = json.load(open(DATA / "coco_ann" / "annotations" / "instances_val2017.json")); cat = {c["id"]: c["name"] for c in A["categories"]}
    img = {im["file_name"]: im for im in A["images"]}; inst = collections.defaultdict(list)
    for an in A["annotations"]: inst[an["image_id"]].append((cat[an["category_id"]], an["bbox"]))
    SYN = json.load(open(REPO / "configs" / "coco_synonyms.json")); LEX = {w: c for c, ws in SYN.items() for w in ws}
    def category_of(phrase):
        t = phrase.lower().strip()
        for k in sorted(LEX, key=len, reverse=True):
            if " " in k and k in t: return LEX[k]
        toks = [x for x in _nlp()(t) if x.pos_ in ("NOUN", "PROPN")]; head = (toks[-1].lemma_ if toks else t.split()[-1]).lower()
        return LEX.get(head, LEX.get(head + "s"))
    rows_, items = np.where(sub["split"] == split)[0], []
    for i in rows_:
        im = img.get(ids[i]); present = inst.get(im["id"], []) if im else []
        if not present: continue
        W, H = im["width"], im["height"]; allm = _grid_mask([b for _, b in present], W, H, g); cats = collections.defaultdict(list)
        for c, b in present: cats[c].append(b)
        for k in range(offsets[i], min(offsets[i + 1], offsets[i] + cap)):
            c = category_of(str(texts[k]))
            if c in cats: items.append((i, k, c, _grid_mask(cats[c], W, H, g), allm))
    rows = np.array([r for r, *_ in items]); ks = np.array([k for _, k, *_ in items]); masks = np.stack([m for _, _, _, m, _ in items]); allmasks = np.stack([m for *_, m in items]); c0 = g * (g // 2) + g // 2
    @torch.no_grad()
    def point(m):
        Tp = m.Tp(torch.tensor(emb[ks].astype(np.float32), device=DEV)); J = np.zeros(len(items), int)
        for r in np.unique(rows):
            Ip = m.Ip(torch.tensor(np.asarray(P[r]).astype(np.float32), device=DEV)); sel = np.where(rows == r)[0]; J[sel] = (Tp[sel] @ Ip.T).argmax(1).cpu().numpy()
        hit = masks[np.arange(len(items)), J]; small = masks.mean(1) < 0.1; person = np.array([c == "person" for _, _, c, _, _ in items])
        return {"pointing_acc": float(hit.mean()), "any_instance_acc": float(allmasks[np.arange(len(items)), J].mean()), "small_object_acc": float(hit[small].mean()), "n_small": int(small.sum()), "person_acc": float(hit[person].mean()), "nonperson_acc": float(hit[~person].mean())}
    R = {"collection": "coco", "encoder": key, "grid": g, "task": "pointing_game", "n_images": int(len(rows_)), "n_phrases": len(items), "chance_box_area": float(masks.mean()), "center_patch_acc": float(masks[:, c0].mean())}
    for name, m in _models("coco", key, variants, tag, d).items(): R[name] = point(m); print(name, round(R[name]["pointing_acc"], 3))
    out = result_path("coco", key, f"grounding{tag}.json"); json.dump(R, open(out, "w"), indent=1); print("->", out); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=["box_hit", "pointing_flickr", "pointing_coco"], required=True)
    ap.add_argument("--collection", default="cityflow"); ap.add_argument("--encoder", required=True); ap.add_argument("--variants", default="pooled,set,setw"); ap.add_argument("--tag", default=""); ap.add_argument("--cap", type=int, default=32); ap.add_argument("--grid", type=int, default=None); a = ap.parse_args(argv)
    v = a.variants.split(",")
    if a.task == "box_hit": box_hit(a.collection, a.encoder, v, a.tag, a.cap, grid=a.grid)
    elif a.task == "pointing_flickr": pointing_flickr(a.encoder, v, a.tag, a.cap, a.grid)
    else: pointing_coco(a.encoder, v, a.tag, a.cap, grid=a.grid)


if __name__ == "__main__":
    main()
