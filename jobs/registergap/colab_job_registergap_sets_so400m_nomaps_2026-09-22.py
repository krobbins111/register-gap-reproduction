"""Ablation job: the set score WITHOUT maps on SigLIP2 so400m/16-384 (frozen phrase and patch embeddings; only lambda and tau chosen on val),
for SkinCAP, RSICD and SemArt, reusing the stores the main job left in /content (rebuilt if the runtime was recycled). Mirrors zeroshot_set in
story/sets/train_sets.py. Original header of the main job follows.
Job: two maps over sets on SigLIP2 so400m/16-384, the draft's default encoder, for SkinCAP, SemArt and RSICD (2026-09-22).
Patch stores: the SigLIP2 attention-pooling head applied to each of the 24 x 24 patch tokens alone (patchemb.py), unit-normalised, averaged
2 x 2 to a 12 x 12 grid and renormalised (the same pooling train_sets.py applies to the base stores). spaCy noun-phrase sets (cap 32), then the
paper's pooled two maps against two maps over sets and the learned phrase gate; 30 epochs, batch 128, tau floor 0.01, two seeds.
Published protocols computed in-job on the seed-0 models: SkinCAP with the whole collection as gallery (Derm1M protocol), RSICD with all
five captions per test image (standard RSITR protocol, T2I and I2T), SemArt = the Text2Art test split (our test split).
Writes results/<RUN>/results.json after every stage. Env: JOB_COLLECTIONS (default skincap,rsicd,semart), JOB_EPOCHS (30), JOB_SEEDS (2)."""
import os, sys, subprocess, json, time, re, glob, zipfile, csv
RUN = os.environ.get("JOB_NAME", "registergap_sets_so400m_2026-09-22_nomaps")
REPO_DIR = os.environ.get("SCRATCH_DIR", os.getcwd()); OUT = os.path.join(REPO_DIR, "results", RUN); os.makedirs(OUT, exist_ok=True)
WORK = "/content/sets_work_so400m_g12"; os.makedirs(WORK, exist_ok=True)
COLLECTIONS = os.environ.get("JOB_COLLECTIONS", "skincap,rsicd,semart").split(",")
EPOCHS = int(os.environ.get("JOB_EPOCHS", 30)); SEEDS = int(os.environ.get("JOB_SEEDS", 2)); GRID = 12; PMAX = 32; BS = 128; TAU_FLOOR = 0.01
VARIANTS = ["pooled", "set", "setw"]
MID = "google/siglip2-so400m-patch16-384"
def push_results():
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "results"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-qm", f"results: {RUN}"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase", "-q"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "push", "-q"], check=False)
    except Exception as e: print("push_results failed:", e)
LOGF = open(f"{OUT}/log.txt", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a); print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()
RESULTS_PATH = f"{OUT}/results.json"; RES = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
def save(): json.dump(RES, open(RESULTS_PATH, "w"), indent=1); push_results()

# ---------------- setup ----------------
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "transformers>=5.10", "datasets", "spacy", "sentencepiece", "protobuf", "huggingface_hub"], check=False)
subprocess.run([sys.executable, "-m", "spacy", "download", "-q", "en_core_web_sm"], check=False)
import torch, numpy as np, spacy
from datasets import load_dataset
from datasets import Image as HFImage
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor
from PIL import Image
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; DEV = "cuda" if torch.cuda.is_available() else "cpu"
log(torch.__version__, DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
model = AutoModel.from_pretrained(MID).to(DEV).eval(); proc = AutoProcessor.from_pretrained(MID)
vm = model.vision_model; SRC = vm.config.image_size // vm.config.patch_size; DIM = vm.config.hidden_size
import transformers; log("transformers", transformers.__version__, MID, "grid", SRC, "->", GRID, "dim", DIM); nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
RES["_config"] = {"model": MID, "src_grid": SRC, "grid": GRID, "dim": DIM, "pmax": PMAX, "epochs": EPOCHS, "batch": BS, "tau_floor": TAU_FLOOR, "seeds": SEEDS, "lr": 1e-4, "wd": 1e-4, "loss_temp": 0.05}

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
    """SigLIP2: pooled image embedding (B,D) fp32 and 12 x 12 per-region embeddings (B,144,D) fp16: the attention-pooling head applied to each
    patch token alone (the embedding the image would have had if the head attended only to that patch), unit-normalised, averaged 2 x 2, renormalised."""
    pi = proc(images=[(Image.open(im) if isinstance(im, str) else im).convert("RGB") for im in ims], return_tensors="pt").to(DEV)
    out = vm(pixel_values=pi["pixel_values"]); tok = out.last_hidden_state; B, N, D = tok.shape
    e = vm.head(tok.reshape(B * N, 1, D)).reshape(B, N, D)
    e = torch.nn.functional.normalize(e, dim=-1).reshape(B, SRC, SRC, D); k = SRC // GRID
    r = torch.nn.functional.normalize(e.reshape(B, GRID, k, GRID, k, D).mean((2, 4)).reshape(B, GRID * GRID, D), dim=-1)
    return torch.nn.functional.normalize(out.pooler_output, dim=-1).float().cpu().numpy(), r.half().cpu().numpy()

# ---------------- collections: same rows and splits as registergap/embed.py ----------------
def _imgcol(d): return [c for c in d.column_names if isinstance(d.features[c], HFImage)][0]
def collection(name):
    """Returns caps (list), splits (np array of 'train'/'val'/'test'), image_iter() yielding PIL images or paths in the same order, and allcaps (per row list of captions, or None)."""
    if name == "skincap":
        d = load_dataset("sercetexam9/skincap", split="train"); col = _imgcol(d); texts = [str(t) for t in d["text"]]
        perm = np.random.RandomState(0).permutation(len(d)); sp = np.empty(len(d), object)
        for j, k in enumerate(perm): sp[int(k)] = "test" if j < 600 else ("val" if j < 1200 else "train")
        return texts, sp, lambda: (d[i][col] for i in range(len(d))), None
    if name == "rsicd":
        ds = load_dataset("arampacha/rsicd"); caps, sp, allcaps = [], [], []
        for split, s in (("train", "train"), ("valid", "val"), ("test", "test")):
            d = ds[split]; capcol = [c for c in d.column_names if "caption" in c.lower()][0]
            for c in d[capcol]:
                cs = [c] if isinstance(c, str) else [str(x) for x in c]; caps.append(max(cs, key=len)); allcaps.append(cs); sp.append(s)
        def it():
            for split in ("train", "valid", "test"):
                d = ds[split]; col = _imgcol(d)
                for i in range(len(d)): yield d[i][col]
        return caps, np.array(sp, object), it, allcaps
    if name == "semart":
        z = hf_hub_download("leo20000306/SemArt", "SemArt.zip", repo_type="dataset"); os.makedirs("/content/semart", exist_ok=True)
        if not glob.glob("/content/semart/**/semart_train.csv", recursive=True):
            with zipfile.ZipFile(z) as zf: zf.extractall("/content/semart")
        base = os.path.dirname(glob.glob("/content/semart/**/semart_train.csv", recursive=True)[0]); imgdir = glob.glob(f"{base}/Images")[0]
        caps, sp, paths = [], [], []
        for split in ["train", "val", "test"]:
            with open(f"{base}/semart_{split}.csv", encoding="latin-1") as f:
                for r in csv.DictReader(f, delimiter="\t"): paths.append(os.path.join(imgdir, r["IMAGE_FILE"])); caps.append(r["DESCRIPTION"].strip()); sp.append(split)
        return caps, np.array(sp, object), lambda: (p for p in paths), None
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

def phrase_store(caps):
    sets = [phrases_of(c, doc) for c, doc in zip(caps, nlp.pipe([c[:1000] for c in caps], batch_size=256))]; flat = [t for st in sets for t in st]
    return embed_text(flat, bs=512).astype(np.float16), np.cumsum([0] + [len(st) for st in sets]), np.array(flat, dtype=object)

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
def score_matrix(m, D, qidx, gidx, variant, Q=None, ph=None, phm=None):
    """S of shape (queries, len(gidx)). Queries are the rows qidx of D unless Q / ph / phm (external captions) are given."""
    if Q is None:
        Q = torch.tensor(D["txt"][qidx], device=DEV); ph, phm = pad_phrases(D["phr"], D["off"], qidx); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
    g = torch.tensor(D["img"][gidx], device=DEV); nq, ng = Q.shape[0], len(gidx); S = np.zeros((nq, ng), np.float32); gch, qch = 512, 128
    for gs in range(0, ng, gch):
        gi = gidx[gs:gs + gch]; patches = g[gs:gs + gch][:, None, :] if variant in ("phrases_only", "phrases_only_w") else torch.tensor(np.asarray(D["P"][gi]).astype(np.float32), device=DEV)
        for qs in range(0, nq, qch): S[qs:qs + qch, gs:gs + gch] = m.score(Q[qs:qs + qch], ph[qs:qs + qch], phm[qs:qs + qch], g[gs:gs + gch], patches).cpu().numpy()
    return S

def evaluate(m, D, split, variant, vmax=500):
    idx = np.where(D["split"] == split)[0]
    if split == "val" and len(idx) > vmax: idx = idx[np.sort(np.random.RandomState(0).permutation(len(idx))[:vmax])]
    return score_matrix(m, D, idx, idx, variant)

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
        log(f"    {variant} s{seed} ep{ep:02d} loss {loss.item():.3f} val R@10 {r10:.4f} lam {m.log_lam.exp().item():.2f} tau {m.log_tau.exp().item():.4f} [{time.time()-t0:.0f}s]")
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

# ---------------- published protocols ----------------
@torch.no_grad()
def protocol(name, m, D, variant, allcaps):
    te = np.where(D["split"] == "test")[0]
    if name == "skincap":                                   # Derm1M protocol: gallery = every image in the collection, queries = our held-out test captions
        gidx = np.arange(len(D["split"])); S = score_matrix(m, D, te, gidx, variant); r = (S >= S[np.arange(len(te)), te][:, None]).sum(1)
        return {f"R@{k}": float((r <= k).mean()) for k in (1, 5, 10, 50)} | {"median_rank": float(np.median(r)), "gallery": int(len(gidx)), "queries": int(len(te)), "note": "whole collection as gallery (Derm1M, ICCV 2025, Table 4 protocol)"}
    if name == "rsicd":                                     # standard RSITR protocol: all five captions of each test image as queries
        if "rsicd5" not in D:
            caps5 = [c for i in te for c in allcaps[i]]; owner = np.repeat(np.arange(len(te)), [len(allcaps[i]) for i in te])
            Q5 = embed_text(caps5); E5, off5, _ = phrase_store(caps5); ph5, phm5 = pad_phrases(E5, off5, np.arange(len(caps5)))
            D["rsicd5"] = (torch.tensor(Q5, device=DEV), torch.tensor(ph5, device=DEV), torch.tensor(phm5, device=DEV), owner, len(caps5))
        Q5, ph5, phm5, owner, nq = D["rsicd5"]; S = score_matrix(m, D, None, te, variant, Q=Q5, ph=ph5, phm=phm5)
        true = S[np.arange(nq), owner]; r = (S >= true[:, None]).sum(1); t2i = {f"T2I R@{k}": float((r <= k).mean()) for k in (1, 5, 10)}
        rank_i = np.array([int((S[:, j] > S[:, j][owner == j].max()).sum()) + 1 for j in range(len(te))]); i2t = {f"I2T R@{k}": float((rank_i <= k).mean()) for k in (1, 5, 10)}
        return t2i | i2t | {"mR": float(np.mean(list(t2i.values()) + list(i2t.values()))), "gallery": int(len(te)), "queries": int(nq), "note": "standard RSITR protocol: all 5 captions per test image; I2T hit if any of the image's captions is in the top k"}
    return {"note": "test split = the published test set (Text2Art); see the main numbers"}

# ---------------- main: no-maps set score ----------------
GRID_LAM = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0); GRID_TAU = (0.01, 0.02, 0.05, 0.1)
def build_stores(name, caps, image_iter):
    t0 = time.time(); n = len(caps); tpath, ipath, ppath, hpath = f"{WORK}/{name}_txt.npy", f"{WORK}/{name}_img.npy", f"{WORK}/{name}_patches.npy", f"{WORK}/{name}_phrases.npz"
    if not os.path.exists(tpath): np.save(tpath, embed_text(caps)); log("text embedded", f"{time.time()-t0:.0f}s")
    if not os.path.exists(ppath):
        P = np.lib.format.open_memmap(ppath + ".tmp.npy", mode="w+", dtype=np.float16, shape=(n, GRID * GRID, DIM)); G = np.zeros((n, DIM), np.float32); s = 0; buf = []
        for im in image_iter():
            buf.append(im)
            if len(buf) == 48:
                g, r = embed_images(buf); G[s:s + 48] = g; P[s:s + 48] = r; s += 48; buf = []
                if (s // 48) % 50 == 0: log(f"  {name} images {s}/{n} {time.time()-t0:.0f}s")
        if buf: g, r = embed_images(buf); G[s:s + len(buf)] = g; P[s:s + len(buf)] = r; s += len(buf)
        assert s == n, (s, n); P.flush(); del P; os.rename(ppath + ".tmp.npy", ppath); np.save(ipath, G); log("images embedded", f"{time.time()-t0:.0f}s")
    if not os.path.exists(hpath):
        E, off, texts = phrase_store(caps); np.savez(hpath, emb=E, offsets=off, texts=texts); log("phrases", len(texts), f"{time.time()-t0:.0f}s")
    H = np.load(hpath, allow_pickle=True)
    return {"txt": np.load(tpath), "img": np.load(ipath), "P": np.load(ppath, mmap_mode="r"), "phr": H["emb"], "off": H["offsets"]}

for name in COLLECTIONS:
    if name in RES and "nomaps_set" in RES[name]: log("have", name); continue
    t0 = time.time(); log("==== collection", name); caps, sp, image_iter, allcaps = collection(name)
    D = build_stores(name, caps, image_iter); D["split"] = sp; RES.setdefault(name, {})["n"] = {s: int((sp == s).sum()) for s in ("train", "val", "test")}; RES[name].setdefault("protocol", {})
    m0 = SetMaps(use_set=False).to(DEV).eval(); RES[name]["frozen"] = rec(ranks(evaluate(m0, D, "test", "pooled"))); RES[name]["protocol"]["frozen"] = protocol(name, m0, D, "pooled", allcaps); log(name, "frozen", RES[name]["frozen"])
    best = None; grid = {}
    for lam in GRID_LAM:
        for tau in GRID_TAU:
            m = SetMaps(lam=lam, tau=tau).to(DEV).eval(); r10 = float((ranks(evaluate(m, D, "val", "set")) <= 10).mean()); grid[f"lam{lam}_tau{tau}"] = r10
            if best is None or r10 > best[0]: best = (r10, lam, tau)
    log(name, "val grid best", best)
    m = SetMaps(lam=best[1], tau=best[2]).to(DEV).eval(); r = rec(ranks(evaluate(m, D, "test", "set"))); r["lam"], r["tau"], r["val_R@10"], r["val_grid"] = best[1], best[2], best[0], grid
    RES[name]["nomaps_set"] = r; RES[name]["protocol"]["nomaps_set"] = protocol(name, m, D, "set", allcaps)
    log(name, "nomaps_set", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "val_grid"}, RES[name]["protocol"]["nomaps_set"]); save()
    log("==== done", name, f"{time.time()-t0:.0f}s")
log("ALL DONE"); save()
