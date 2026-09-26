"""Caption -> set of phrases (registergap-pipeline; ported from registergap story/sets/phrases.py).

spaCy noun chunks, optionally verb phrases ('drives forward', 'turns left'), with a split on '&' / ' and ' / commas for terse captions
and a whole-caption fallback; capped at `cap` phrases; embedded with the encoder's text tower (the same tower that embedded the captions).
    phrases(collection, encoder, cap=32, verbs=False) -> ltg.cache.phrases_path(...): emb (P, d) fp16 unit, offsets (n+1,), texts (P,) object,
    rows in cell order (read from cell.caption; legacy cells imported without captions must be re-embedded first).
usage: python -m ltg.encoders.phrases --collection skincap --encoder siglip2-so400m-16-384 --cap 32 [--verbs]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import encoder_key, load_cell, phrases_path  # noqa: E402

DROP = {"it", "this", "that", "which", "these", "those", "there", "the patient", "patients", "the description", "this description", "the image",
        "the photo", "the picture", "this photo", "this image", "this picture", "the photograph"}
_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def phrases_of(caption: str, doc, cap: int = 32, verbs: bool = False) -> list[str]:
    out = []
    for ch in doc.noun_chunks:
        t = ch.text.strip(" .,;:!#'\"")
        if len(t) < 3 or t.lower() in DROP or ch.root.pos_ == "PRON": continue
        out.append(t)
    if verbs:
        for tok in doc:
            if tok.pos_ != "VERB": continue
            end = tok.i
            for ch in tok.children:
                if ch.i > tok.i and ch.dep_ in ("prt", "advmod", "acomp", "oprd", "prep") and ch.i <= end + 1: end = ch.i
            out.append(doc[tok.i:end + 1].text.strip(" .,;:!#'\""))
    if len(out) <= 1 and not verbs:
        parts = [p.strip(" .,;:!#'\"") for p in re.split(r"\s*(?:&|\band\b|,|/|\+)\s*", caption) if len(re.findall(r"[A-Za-z]", p)) >= 3]
        if len(parts) > 1: out = parts
    if not out or (len(out) == 1 and len(caption.split()) <= 5 and len(out[0]) < len(caption.strip())): out = [caption.strip()[:200]]
    seen, ded = set(), []
    for t in out:
        if t.lower() not in seen: seen.add(t.lower()); ded.append(t)
    return ded[:cap]


def phrase_sets(caps, cap: int = 32, verbs: bool = False) -> list[list[str]]:
    return [phrases_of(c, doc, cap, verbs) for c, doc in zip(caps, nlp().pipe([c[:1000] for c in caps], batch_size=256))]


def phrases(collection: str, encoder: str, cap: int = 32, verbs: bool = False, force: bool = False, embedder=None) -> Path:
    key = encoder_key(encoder); out = phrases_path(collection, key, cap, verbs)
    if out.exists() and not force:
        print("have", out); return out
    cell = load_cell(collection, key); caps = [str(c) for c in cell.caption]
    if not any(caps): raise RuntimeError(f"{collection} x {key}: the cell has no captions (legacy import); re-embed from pairs.jsonl first")
    sets = phrase_sets(caps, cap, verbs); flat = [t for s in sets for t in s]; offsets = np.cumsum([0] + [len(s) for s in sets])
    if embedder is None:
        from ltg.encoders.patches import get_embedder
        embedder = get_embedder(key)
    E = embedder.texts(flat, bs=256).astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True); np.savez(out, emb=E, offsets=offsets, texts=np.array(flat, dtype=object))
    L = np.array([len(s) for s in sets]); print(f"{collection} x {key}: {len(caps)} captions, {len(flat)} phrases, per caption mean {L.mean():.1f} median {np.median(L):.0f} max {L.max()} -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--encoder", required=True)
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--verbs", action="store_true"); ap.add_argument("--force", action="store_true"); a = ap.parse_args(argv)
    phrases(a.collection, a.encoder, a.cap, a.verbs, a.force)


if __name__ == "__main__":
    main()
