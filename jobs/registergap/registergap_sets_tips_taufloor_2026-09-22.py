"""Job: TIPSv2-B/14 sets on SciMMIR and ROCOv2 with a temperature floor (tau >= 0.01) and the full 16 x 16 grid (2026-09-22), to test whether the small
set gain on these two (TIPS.md) was the temperature collapse and the 8 x 8 pooling. Otherwise the 2026-09-21 job: SciMMIR, ROCOv2, Fashion200k.
Same row selection as registergap/embed.py and the 2026-09-19 SigLIP2 job. TIPSv2 (google/tipsv2-b14, 448 px, 14 px patches): the image
embedding is the class token, the 32 x 32 text-aligned patch tokens are averaged 4 x 4 to an 8 x 8 grid, spaCy noun-phrase
sets (cap 32), then the paper's two pooled maps against two maps over sets, with ablations (phrases only, patches only) and the learned phrase
gate. Writes results/<RUN>/results.json after every collection. Env: JOB_COLLECTIONS (default scimmir,rocov2,fashion200k), JOB_EPOCHS (20)."""
import os, sys, subprocess, json, time, re
RUN = os.environ.get("JOB_NAME", "registergap_sets_tips_taufloor_2026-09-22")
REPO_DIR = os.environ.get("SCRATCH_DIR", os.getcwd()); OUT = os.path.join(REPO_DIR, "results", RUN); os.makedirs(OUT, exist_ok=True)
WORK = "/content/sets_work_tips_g16"; os.makedirs(WORK, exist_ok=True)
COLLECTIONS = os.environ.get("JOB_COLLECTIONS", "scimmir,rocov2").split(",")
EPOCHS = int(os.environ.get("JOB_EPOCHS", 20)); GRID = int(os.environ.get("JOB_GRID", 16)); PMAX = 32; BS = 128; TAU_FLOOR = float(os.environ.get("JOB_TAU_FLOOR", 0.01))
VARIANTS = ["pooled", "set", "setw"]
def push_results():
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "results"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-qm", f"results: {RUN}"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "push", "-q"], check=False)
    except Exception as e: print("push_results failed:", e)
LOGF = open(f"{OUT}/log.txt", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a); print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()
RESULTS_PATH = f"{OUT}/results.json"; RES = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
def save(): json.dump(RES, open(RESULTS_PATH, "w"), indent=1); push_results()

# ---------------- setup ----------------
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "transformers>=5.10", "datasets", "spacy", "sentencepiece", "protobuf"], check=False)
subprocess.run([sys.executable, "-m", "spacy", "download", "-q", "en_core_web_sm"], check=False)
import torch, numpy as np, spacy
from datasets import load_dataset
from datasets import Image as HFImage
from transformers import AutoModel, AutoProcessor
torch.backends.cuda.matmul.allow_tf32 = True; DEV = "cuda" if torch.cuda.is_available() else "cpu"; log(torch.__version__, DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
MID = "google/tipsv2-b14"; model = AutoModel.from_pretrained(MID).to(DEV).eval(); proc = AutoProcessor.from_pretrained(MID)
vm = model.vision_model; N_EXTRA = 1 + vm.config.num_register_tokens; SRC = vm.config.image_size // vm.config.patch_size; DIM = vm.config.hidden_size
import transformers; log("transformers", transformers.__version__, MID, "grid", SRC, "->", GRID, "dim", DIM); nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

@torch.no_grad()
def embed_text(caps, bs=256):
    out = []
    for s in range(0, len(caps), bs):
        pt = proc(text=list(caps[s:s + bs]), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(DEV)
        f = model.get_text_features(**pt); f = f if torch.is_tensor(f) else f.pooler_output
        out.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)

@torch.no_grad()
def embed_images(ims):
    """TIPSv2: class token (B,D) fp32 and 8 x 8 region embeddings (B,64,D) fp16 from the text-aligned patch tokens (4 x 4 averages of the 32 x 32 grid), both unit norm."""
    pi = proc(images=[im.convert("RGB") for im in ims], return_tensors="pt").to(DEV)
    hs = vm(pixel_values=pi["pixel_values"]).last_hidden_state; B, N, D = hs.shape
    tok = hs[:, N_EXTRA:].reshape(B, SRC, SRC, D); k = SRC // GRID
    r = torch.nn.functional.normalize(tok.reshape(B, GRID, k, GRID, k, D).mean((2, 4)).reshape(B, GRID * GRID, D), dim=-1)
    return torch.nn.functional.normalize(hs[:, 0], dim=-1).float().cpu().numpy(), r.half().cpu().numpy()

# ---------------- collections: same row selection as registergap/embed.py ----------------
def _imgcol(d): return [c for c in d.column_names if isinstance(d.features[c], HFImage)][0]
def collection(name):
    """Returns caps (list), splits (np array of 'train'/'val'/'test'), and image_iter() yielding PIL images in the same order."""
    if name == "coco":
        d = load_dataset("lmms-lab/COCO-Caption2017", split="val"); col = _imgcol(d); ans = d["answer"]
        rng = np.random.RandomState(0); perm = rng.permutation(len(d)); sp = np.empty(len(d), object)
        for j, k in enumerate(perm): sp[int(k)] = "test" if j < 1000 else ("val" if j < 1500 else "train")
        caps = [a[0] if isinstance(a, list) else str(a) for a in ans]
        return caps, sp, lambda: (d[i][col] for i in range(len(d)))
    if name in ("facad", "fashion200k"):
        if name == "facad": d = load_dataset("Luna288/image-captioning-FACAD-base", split="train"); ntr, nva, nte = 8754, 1874, 1874
        else:
            d = load_dataset("Marqo/fashion200k", split="data"); seen, keep = set(), []
            for i, it in enumerate(d["item_ID"]):
                if it not in seen: seen.add(it); keep.append(i)
            d = d.select(keep); ntr, nva, nte = 50000, 4999, 4999
        col = _imgcol(d); texts = d["text"]; p = np.random.RandomState(0).permutation(len(d))
        order = np.concatenate([p[:ntr], p[ntr:ntr + nva], p[ntr + nva:ntr + nva + nte]]); sp = np.array(["train"] * ntr + ["val"] * nva + ["test"] * nte, object)
        return [str(texts[int(j)]) for j in order], sp, lambda: (d[int(j)][col] for j in order)
    if name == "rocov2":
        files = {"train": [f"data/train-{i:05d}-of-00027.parquet" for i in range(6)], "validation": ["data/validation-00000-of-00006.parquet"], "test": [f"data/test-{i:05d}-of-00006.parquet" for i in range(6)]}
        ds = load_dataset("eltorio/ROCOv2-radiology", data_files=files, verification_mode="no_checks"); caps, sp = [], []
        for split, s in (("train", "train"), ("validation", "val"), ("test", "test")): caps += [str(c) for c in ds[split]["caption"]]; sp += [s] * len(ds[split])
        def it():
            for split in ("train", "validation", "test"):
                d = ds[split]
                for i in range(len(d)): yield d[i]["image"]
        return caps, np.array(sp, object), it
    if name == "scimmir":
        ds = load_dataset("m-a-p/SciMMIR", data_files={"validation": "data/validation-*", "test": "data/test-*"}, verification_mode="no_checks")
        rng = np.random.RandomState(0); tr = ds["validation"]; p = rng.permutation(len(tr)); te = ds["test"]; p2 = rng.permutation(len(te))
        ttr, tte = tr["text"], te["text"]; caps = [str(ttr[int(j)]) for j in p[:16184]] + [str(tte[int(j)]) for j in p2[:6934]]
        sp = np.array(["train"] * 16184 + ["val"] * 3467 + ["test"] * 3467, object)
        def it():
            for j in p[:16184]: yield tr[int(j)]["image"]
            for j in p2[:6934]: yield te[int(j)]["image"]
        return caps, sp, it
    raise ValueError(name)

# ---------------- phrases ----------------
DROP = {"it", "this", "that", "which", "these", "those", "there", "the patient", "patients", "the description", "this description", "the image", "the photo", "the picture", "this photo", "this image", "this picture", "the photograph"}
def phrases_of(caption, doc):
    out = []
    for ch in doc.noun_chunks:
        t = ch.text.strip(" .,;:!#'\"")
        if len(t) < 3 or t.lower() in DROP or ch.root.pos_ == "PRON": continue
        out.append(t)
    if len(out) <= 1:
        parts = [p.strip(" .,;:!#'\"") for p in re.split(r"\s*(?:&|\band\b|,|/|\+)\s*", caption) if len(re.findall(r"[A-Za-z]", p)) >= 3]
        if len(parts) > 1: out = parts
    if not out or (len(out) == 1 and len(caption.split()) <= 5 and len(out[0]) < len(caption.strip())): out = [caption.strip()[:200]]
    seen, ded = set(), []
    for t in out:
        if t.lower() not in seen: seen.add(t.lower()); ded.append(t)
    return ded[:PMAX]

def pad_phrases(emb, offsets, idx):
    pmax = int(min(PMAX, max(offsets[i + 1] - offsets[i] for i in idx))); B = len(idx); X = np.zeros((B, pmax, emb.shape[1]), np.float32); M = np.zeros((B, pmax), bool)
    for b, i in enumerate(idx):
        s, e = offsets[i], min(offsets[i + 1], offsets[i] + pmax); k = e - s; X[b, :k] = emb[s:e]; M[b, :k] = True
    return X, M

# ---------------- model ----------------
class SetMaps(torch.nn.Module):
    def __init__(self, d=None, lam=1.0, tau=0.05, use_pool=True, use_set=True, gate=False):
        d = d or DIM
        super().__init__(); self.Wt, self.Wi = torch.nn.Linear(d, d), torch.nn.Linear(d, d); self.use_gate = gate
        if gate: self.gate = torch.nn.Linear(d, 1)
        for L in [self.Wt, self.Wi] + ([self.gate] if gate else []): torch.nn.init.zeros_(L.weight); torch.nn.init.zeros_(L.bias)
        self.log_lam = torch.nn.Parameter(torch.tensor(float(np.log(lam)))); self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(tau)))); self.use_pool, self.use_set = use_pool, use_set
    def T(self, x): return torch.nn.functional.normalize(x + self.Wt(x), dim=-1)
    def I(self, x): return torch.nn.functional.normalize(x + self.Wi(x), dim=-1)
    def weights(self, Tp, phm):
        if self.use_gate: return torch.softmax(self.gate(Tp).squeeze(-1).masked_fill(~phm, float("-inf")), dim=1).nan_to_num(0.0)
        return phm.float() / phm.float().sum(1, keepdim=True).clamp(min=1)
    def score(self, q, ph, phm, g, patches):
        S = torch.zeros(q.shape[0], g.shape[0], device=q.device)
        if self.use_pool: S = S + self.T(q) @ self.I(g).T
        if self.use_set:
            tau = self.log_tau.exp().clamp(min=TAU_FLOOR); Tp = self.T(ph); Ip = self.I(patches); B, Pn, d = Tp.shape; C, J, _ = Ip.shape
            sim = (Tp.reshape(B * Pn, d) @ Ip.reshape(C * J, d).T).reshape(B, Pn, C, J).permute(0, 2, 1, 3)
            soft = tau * torch.logsumexp(sim / tau, dim=3) - tau * float(np.log(J))
            S = S + self.log_lam.exp() * torch.einsum("bcp,bp->bc", soft, self.weights(Tp, phm))
        return S

def ranks(S): true = np.diag(S); return (S >= true[:, None]).sum(1)
def rec(r, boot=1000):
    out = {f"R@{k}": float((r <= k).mean()) for k in (1, 5, 10, 50)}; out["median_rank"] = float(np.median(r))
    idx = np.random.RandomState(0).randint(0, len(r), (boot, len(r))); b = (r[idx] <= 10).mean(1); out["R@10_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]; return out

@torch.no_grad()
def evaluate(m, D, split, variant, vmax=500):
    idx = np.where(D["split"] == split)[0]
    if split == "val" and len(idx) > vmax: idx = idx[np.random.RandomState(0).permutation(len(idx))[:vmax]]
    n = len(idx); q = torch.tensor(D["txt"][idx], device=DEV); g = torch.tensor(D["img"][idx], device=DEV)
    ph, phm = pad_phrases(D["phr"], D["off"], idx); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
    if variant == "patches_only": ph, phm = q[:, None, :], torch.ones(n, 1, dtype=torch.bool, device=DEV)
    S = np.zeros((n, n), np.float32); gch, qch = 512, 128
    for gs in range(0, n, gch):
        gi = idx[gs:gs + gch]; patches = g[gs:gs + gch][:, None, :] if variant in ("phrases_only", "phrases_only_w") else torch.tensor(np.asarray(D["P"][gi]).astype(np.float32), device=DEV)
        for qs in range(0, n, qch): S[qs:qs + qch, gs:gs + gch] = m.score(q[qs:qs + qch], ph[qs:qs + qch], phm[qs:qs + qch], g[gs:gs + gch], patches).cpu().numpy()
    return S

def train(D, variant, seed=0, epochs=EPOCHS, lr=1e-4, wd=1e-4, temp=0.05):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); tr = np.where(D["split"] == "train")[0]
    cfg = {"pooled": dict(use_set=False), "set": dict(), "phrases_only": dict(use_pool=False), "patches_only": dict(use_pool=False), "setw": dict(gate=True), "phrases_only_w": dict(use_pool=False, gate=True)}[variant]
    m = SetMaps(**cfg).to(DEV)
    groups = [{"params": [m.Wt.weight, m.Wt.bias, m.Wi.weight, m.Wi.bias], "lr": lr, "weight_decay": wd}, {"params": [m.log_lam, m.log_tau], "lr": 1e-2, "weight_decay": 0.0}]
    if m.use_gate: groups.append({"params": list(m.gate.parameters()), "lr": 1e-3, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups); best, best_state = -1, None; t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(tr); m.train()
        for s in range(0, len(perm), BS):
            bi = np.sort(perm[s:s + BS])
            if len(bi) < 8: continue
            q = torch.tensor(D["txt"][bi], device=DEV); g = torch.tensor(D["img"][bi], device=DEV)
            ph, phm = pad_phrases(D["phr"], D["off"], bi); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
            if variant == "patches_only": ph, phm = q[:, None, :], torch.ones(len(bi), 1, dtype=torch.bool, device=DEV)
            patches = g[:, None, :] if variant in ("phrases_only", "phrases_only_w") else torch.tensor(np.asarray(D["P"][bi]).astype(np.float32), device=DEV)
            S = m.score(q, ph, phm, g, patches) / temp; y = torch.arange(len(bi), device=DEV)
            loss = 0.5 * (torch.nn.functional.cross_entropy(S, y) + torch.nn.functional.cross_entropy(S.T, y)); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); r10 = float((ranks(evaluate(m, D, "val", variant)) <= 10).mean())
        if r10 > best: best, best_state = r10, {k: v.detach().clone() for k, v in m.state_dict().items()}
        log(f"    {variant} ep{ep:02d} loss {loss.item():.3f} val R@10 {r10:.4f} lam {m.log_lam.exp().item():.2f} tau {m.log_tau.exp().item():.4f} [{time.time()-t0:.0f}s]")
    m.load_state_dict(best_state); m.eval(); return m, best

@torch.no_grad()
def gate_report(m, D, texts):
    te = np.where(D["split"] == "test")[0]; ph, phm = pad_phrases(D["phr"], D["off"], te); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
    Tp = m.T(ph); w = m.weights(Tp, phm); logits = m.gate(Tp).squeeze(-1).cpu().numpy(); agg = {}
    for b, i in enumerate(te):
        s, e = D["off"][i], min(D["off"][i + 1], D["off"][i] + ph.shape[1])
        for k in range(e - s): agg.setdefault(str(texts[s + k]).lower(), []).append(float(logits[b, k]))
    rows = sorted(((float(np.mean(v)), len(v), t) for t, v in agg.items() if len(v) >= 5), key=lambda x: x[0])
    return {"effective_phrases": float((1 / (w ** 2).sum(1)).mean()), "mean_phrases": float(phm.sum(1).float().mean()), "lowest": [(t, round(l, 2), c) for l, c, t in rows[:20]], "highest": [(t, round(l, 2), c) for l, c, t in rows[-20:][::-1]]}

# ---------------- main ----------------
for name in COLLECTIONS:
    if name in RES and all(v in RES[name] for v in VARIANTS): log("have", name); continue
    t0 = time.time(); log("==== collection", name); caps, sp, image_iter = collection(name); n = len(caps); log(name, "rows", n, {s: int((sp == s).sum()) for s in ("train", "val", "test")})
    # embeddings (cached on the Colab disk)
    tpath, ipath, ppath = f"{WORK}/{name}_txt.npy", f"{WORK}/{name}_img.npy", f"{WORK}/{name}_patches.npy"
    if not os.path.exists(tpath): np.save(tpath, embed_text(caps)); log("text embedded", f"{time.time()-t0:.0f}s")
    if not os.path.exists(ppath):
        P = np.lib.format.open_memmap(ppath + ".tmp.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, DIM)); G = np.zeros((n, DIM), np.float32); s = 0; buf = []
        for im in image_iter():
            buf.append(im)
            if len(buf) == 64:
                g, r = embed_images(buf); G[s:s + 64] = g; P[s:s + 64] = r; s += 64; buf = []
                if (s // 64) % 100 == 0: log(f"  {name} images {s}/{n} {time.time()-t0:.0f}s")
        if buf: g, r = embed_images(buf); G[s:s + len(buf)] = g; P[s:s + len(buf)] = r; s += len(buf)
        assert s == n, (s, n); P.flush(); del P; os.rename(ppath + ".tmp.npy", ppath); np.save(ipath, G); log("images embedded", f"{time.time()-t0:.0f}s")
    # phrases
    hpath = f"{WORK}/{name}_phrases.npz"
    if not os.path.exists(hpath):
        sets = [phrases_of(c, doc) for c, doc in zip(caps, nlp.pipe([c[:1000] for c in caps], batch_size=256))]; flat = [t for st in sets for t in st]; off = np.cumsum([0] + [len(st) for st in sets])
        np.savez(hpath, emb=embed_text(flat, bs=512).astype(np.float16), offsets=off, texts=np.array(flat, dtype=object)); log("phrases", len(flat), "mean per caption %.1f" % np.mean([len(st) for st in sets]), f"{time.time()-t0:.0f}s")
    H = np.load(hpath, allow_pickle=True)
    D = {"txt": np.load(tpath), "img": np.load(ipath), "P": np.load(ppath, mmap_mode="r"), "split": sp, "phr": H["emb"], "off": H["offsets"]}
    RES.setdefault(name, {})["n"] = {s: int((sp == s).sum()) for s in ("train", "val", "test")}; RES[name]["phrases_per_caption"] = float(np.mean(np.diff(H["offsets"])))
    if "frozen" not in RES[name]:
        m0 = SetMaps(use_set=False).to(DEV).eval(); RES[name]["frozen"] = rec(ranks(evaluate(m0, D, "test", "pooled"))); log(name, "frozen", RES[name]["frozen"]); save()
    for variant in VARIANTS:
        if variant in RES[name]: continue
        m, bv = train(D, variant); r = rec(ranks(evaluate(m, D, "test", variant))); r["val_R@10"] = bv; r["lam"], r["tau"] = float(m.log_lam.exp()), float(m.log_tau.exp())
        if m.use_gate: r["gate"] = gate_report(m, D, H["texts"])
        RES[name][variant] = r; log(name, variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("gate",)}); save()
    log("==== done", name, f"{time.time()-t0:.0f}s")
log("ALL DONE"); save()
