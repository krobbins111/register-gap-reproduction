"""API-only embedders: Gemini Embedding 2 (registergap-pipeline; ported from registergap story/api/embed_gemini.py and contamination_probe.py, 22 Sep).

No weights, so no LoRA: the two maps are the whole adaptation toolkit. Captions are sent as "task: search result | query: <caption>",
images as bare JPEG parts (long side 512), 3072-d output, unit norm. Requests run in an ordered thread pool with a hard 90 s deadline per
call (a hung socket would otherwise block the pool) and are checkpointed under $LTG_CACHE/<collection>/parts/ so a rate-limit stop resumes.
    embed_api(collection, encoder='gemini-embedding-2', ...) -> save_cell(collection, encoder, ...) with meta.encoder_id, dim, date, prefix.
    probe(collection, encoder, reference='siglip2-base-16-256') -> results/<collection>/<encoder>/probe.json: frozen recall on the original
        against LLM-paraphrased test captions (gpt-5-mini, cached under data/<collection>/paraphrase_test.json), for the API model and a
        reference open encoder, plus the API model's two maps on both query sets.
Keys: GEMINI_API_KEY (and OPENAI_API_KEY for the probe's paraphrases) in the environment; never in files. Do not send CityFlow-NL (NVIDIA
licence) or non-redistributable webcam frames to any API. usage: python -m ltg.encoders.api --collection skincap [--limit 40]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import cache_root, encoder_key, load_cell, result_path, save_cell  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL = "gemini-embedding-2"; TEXT_PREFIX = "task: search result | query: "; CALL_TIMEOUT = 90


def _client():
    from google import genai
    from google.genai import types
    return genai.Client(http_options=types.HttpOptions(timeout=120_000)), types


def _jpeg(im, max_side=512):
    from PIL import Image
    if isinstance(im, str): im = Image.open(im)
    im = im.convert("RGB"); w, h = im.size; s = max_side / max(w, h)
    if s < 1: im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=90); return b.getvalue()


def embed_api(collection: str, encoder: str = "gemini-embedding-2", dim: int = 3072, limit: int = 0, workers: int = 4, img_batch: int = 6, txt_batch: int = 32, max_side: int = 512, force: bool = False) -> Path:
    from ltg.data.images import rows as _rows
    from ltg.cache import cell_path
    client, types = _client(); key = encoder_key(encoder); out = cell_path(collection, key)
    if out.exists() and not force: print("have", out); return out
    rows = _rows(collection)
    if limit: rows = [r for s in ("train", "val", "test") for r in [x for x in rows if x["split"] == s][:limit]]
    n = len(rows); part = cache_root() / collection / "parts" / f"{key}{'_lim%d' % limit if limit else ''}"; part.mkdir(parents=True, exist_ok=True)
    print(f"{collection}: {n} rows", {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}, flush=True)
    timed = ThreadPoolExecutor(64)
    def call(contents, tries=14):
        for t in range(tries):
            fut = timed.submit(client.models.embed_content, model=MODEL, contents=contents, config=types.EmbedContentConfig(output_dimensionality=dim))
            try:
                r = fut.result(timeout=CALL_TIMEOUT); E = np.array([e.values for e in r.embeddings], np.float32)
                if len(E) != len(contents): raise RuntimeError(f"got {len(E)} embeddings for {len(contents)} contents")
                return E
            except FutTimeout: print(f"  retry {t + 1}: call timed out after {CALL_TIMEOUT}s (abandoned)", flush=True)
            except Exception as e: wait = min(90, 2 ** t); print(f"  retry {t + 1}: {str(e)[:120]} (sleep {wait}s)", flush=True); time.sleep(wait)
        raise RuntimeError("gave up")
    def run(kind, make, bs):
        f = part / f"{kind}.npy"; done = np.load(f) if f.exists() else np.zeros((0, dim), np.float32); s0 = len(done); t0 = time.time(); chunks = [done]; starts = list(range(s0, n, bs))
        def one(s): return call([types.Content(parts=make(r)) for r in rows[s:s + bs]])
        with ThreadPoolExecutor(workers) as ex:
            for k, E in enumerate(ex.map(one, starts)):
                chunks.append(E)
                if k % 20 == 19 or k == len(starts) - 1: np.save(f, np.concatenate(chunks)); print(f"  {kind} {min(starts[k] + bs, n)}/{n}  {time.time() - t0:.0f}s", flush=True)
        return np.concatenate(chunks)
    T = run("txt", lambda r: [types.Part.from_text(text=TEXT_PREFIX + r["caption"])], txt_batch)
    I = run("img", lambda r: [types.Part.from_bytes(data=_jpeg(r["image"], max_side), mime_type="image/jpeg")], img_batch)
    meta = {"encoder_id": MODEL, "dim": dim, "text_prefix": TEXT_PREFIX, "image_max_side": max_side, "date": time.strftime("%Y-%m-%d"), "source": "Gemini API (google-genai)", "limit": limit}
    p = save_cell(collection, key, T, I, [r["split"] for r in rows], caption=[r["caption"] for r in rows], ids=[str(r["id"]) for r in rows], meta=meta)
    print("saved", p, T.shape, I.shape); return p


def _recall(Q, G):
    S = Q @ G.T; r = (S >= np.diag(S)[:, None]).sum(1); return {f"R@{k}": float((r <= k).mean()) for k in (1, 10, 50)}


def probe(collection: str, encoder: str = "gemini-embedding-2", reference: str = "siglip2-base-16-256", bs: int = 5) -> Path:
    """Paraphrase probe: frozen recall on original against paraphrased test captions, API model and reference encoder, same gallery."""
    from ltg.encoders.patches import get_embedder
    from ltg.maps.linear import train_arm
    import openai
    key, ref = encoder_key(encoder), encoder_key(reference); G = load_cell(collection, key); S = load_cell(collection, ref); assert (S.caption == G.caption).all(), "row order differs"
    te = S.split == "test"; ids = [str(i) for i in S.id[te]]; caps = [str(c) for c in S.caption[te]]
    path = REPO / "data" / collection / "paraphrase_test.json"; path.parent.mkdir(parents=True, exist_ok=True); done = json.load(open(path)) if path.exists() else {}
    oa = openai.OpenAI(); PROMPT = "Rewrite each caption below in different words, keeping exactly the same content, the same technical vocabulary and register, and about the same length. Do not add, remove or reorder any detail."
    def parse(txt, k):
        if not txt: return None
        t = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
        try: o = json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, re.S)
            if not m: return None
            try: o = json.loads(m.group(0))
            except Exception: return None
        out = o.get("out") if isinstance(o, dict) else o
        return [str(x).strip() for x in out] if isinstance(out, list) and len(out) == k else None
    def run(b):
        items = "\n".join(f"{k + 1}. {caps[i]}" for k, i in enumerate(b)); msg = f"{PROMPT}\n\nReturn a JSON object {{\"out\": [...]}} with exactly {len(b)} strings, one per item, in the same order.\n\nItems:\n{items}"
        for _ in range(3):
            for t in range(5):
                try: txt = oa.chat.completions.create(model="gpt-5-mini", reasoning_effort="minimal", max_completion_tokens=6000, messages=[{"role": "user", "content": msg}]).choices[0].message.content; break
                except Exception: time.sleep(min(60, 2 ** t)); txt = None
            res = parse(txt, len(b))
            if res: return b, res
        return b, None
    todo = [i for i in range(len(ids)) if ids[i] not in done]; batches = [todo[s:s + bs] for s in range(0, len(todo), bs)]
    with ThreadPoolExecutor(8) as ex:
        for b, res in ex.map(run, batches):
            if res:
                for i, r in zip(b, res): done[ids[i]] = r
    json.dump(done, open(path, "w"), indent=1); para = [done.get(i, caps[j]) for j, i in enumerate(ids)]
    client, types = _client(); pf = cache_root() / collection / "parts" / f"{key}_paraphrase.npy"; pf.parent.mkdir(parents=True, exist_ok=True)
    if pf.exists(): Tg = np.load(pf)
    else:
        out = []
        for s in range(0, len(para), 32):
            for t in range(8):
                try: r = client.models.embed_content(model=MODEL, contents=[types.Content(parts=[types.Part.from_text(text=TEXT_PREFIX + x)]) for x in para[s:s + 32]], config=types.EmbedContentConfig(output_dimensionality=G.dim)); out.append(np.array([e.values for e in r.embeddings], np.float32)); break
                except Exception: time.sleep(min(60, 2 ** t))
        Tg = np.concatenate(out); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True); np.save(pf, Tg)
    Ts = get_embedder(ref).texts(para, bs=128)
    res = {"collection": collection, "encoder": key, "reference": ref, "n_paraphrased": sum(i in done for i in ids),
           ref: {"original": _recall(S.txt[te], S.img[te]), "paraphrase": _recall(Ts, S.img[te])}, key: {"original": _recall(G.txt[te], G.img[te]), "paraphrase": _recall(Tg, G.img[te])}}
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"; tr, va = G.split == "train", G.split == "val"
    f, g = train_arm("both", G.txt[tr], G.img[tr], G.txt[va], G.img[va], epochs=30, lr=1e-4, batch=256, temp=0.05, sym_loss=False, seed=0, dev=dev)
    import torch
    with torch.no_grad():
        ap_ = lambda m, x: m(torch.tensor(x, device=dev)).cpu().numpy()
        res[key]["two_maps_original"] = _recall(ap_(f, G.txt[te]), ap_(g, G.img[te])); res[key]["two_maps_paraphrase"] = _recall(ap_(f, Tg), ap_(g, G.img[te]))
    out = result_path(collection, key, "probe.json"); json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1)); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--encoder", default="gemini-embedding-2"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="run the paraphrase probe instead of embedding"); ap.add_argument("--reference", default="siglip2-base-16-256"); ap.add_argument("--force", action="store_true"); a = ap.parse_args(argv)
    (probe(a.collection, a.encoder, a.reference) if a.probe else embed_api(a.collection, a.encoder, limit=a.limit, force=a.force))


if __name__ == "__main__":
    main()
