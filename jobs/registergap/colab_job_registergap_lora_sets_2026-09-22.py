"""Job: is grounding a property of the set objective or of the parameterization? (2026-09-22)
Same data, same phrases, same loss, two parameterizations: (a) two residual linear maps on frozen SigLIP2-B/256 features ('setmaps', the
paper's method) and (b) LoRA r=16 on the attention projections of both towers ('lora_set'), both trained with the two-maps-over-sets objective
  score(q, g) = <q, g> + lambda * mean_p tau * logsumexp_j <phrase_p, patch_j> / tau      (lambda, tau learned; 'setw' adds a learned phrase gate),
plus LoRA with the plain pooled contrastive loss ('lora_pooled', the 14 Sep baseline) and the frozen encoder. No boxes are used in training.
Collections: Flickr30k (8k train captions; retrieval on the 1k test gallery; pointing game on Flickr30k Entities test, 14k phrases with boxes)
and SkinCAP (retrieval on the 600 test gallery). Patches = SigLIP2's attention-pooling head applied to each patch token alone (patchemb.py),
phrases = spaCy noun chunks (phrases.py rules, cap JOB_PMAX). Writes results/<JOB_NAME>/results.json after every variant.
Env: JOB_COLLECTIONS (flickr,skincap), JOB_VARIANTS (setmaps,setmapsw,lora_pooled,lora_set,lora_setw), JOB_EPOCHS_LORA (5), JOB_BS (32), JOB_PMAX (32), JOB_SEED (0)."""
import os, sys, subprocess, json, time, math, re, copy
NAME = os.environ.get("JOB_NAME", "registergap_lora_sets_2026-09-22"); REPO_DIR = os.environ.get("SCRATCH_DIR", os.getcwd()); OUT = os.path.join(REPO_DIR, "results", NAME); os.makedirs(OUT, exist_ok=True)
WORK = "/content/lora_sets_work"; os.makedirs(WORK, exist_ok=True)
def push_results():
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "results"], check=True); subprocess.run(["git", "-C", REPO_DIR, "commit", "-qm", f"results: {NAME}"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase", "-q", "origin", "main"], check=False); subprocess.run(["git", "-C", REPO_DIR, "push", "-q"], check=False)
    except Exception as e: print("push_results failed:", e)
LOGF = open(f"{OUT}/joblog.txt", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a); print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()
RESULTS_PATH = f"{OUT}/results.json"; RES = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
def save(): json.dump(RES, open(RESULTS_PATH, "w"), indent=1); push_results()
subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"])
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "transformers", "datasets", "peft", "accelerate", "huggingface_hub", "spacy"], check=False)
subprocess.run([sys.executable, "-m", "spacy", "download", "-q", "en_core_web_sm"], check=False)
import numpy as np, torch, spacy
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image
torch.backends.cuda.matmul.allow_tf32 = True; dev = "cuda"; log(torch.__version__, torch.cuda.get_device_name(0))
COLLECTIONS = os.environ.get("JOB_COLLECTIONS", "flickr,skincap").split(","); VARIANTS = os.environ.get("JOB_VARIANTS", "setmaps,setmapsw,lora_pooled,lora_set,lora_setw").split(",")
EPOCHS_LORA = int(os.environ.get("JOB_EPOCHS_LORA", 5)); BS = int(os.environ.get("JOB_BS", 32)); PMAX = int(os.environ.get("JOB_PMAX", 32)); SEED = int(os.environ.get("JOB_SEED", 0)); VAL_MAX = 500; TEMP = 0.05
MID = "google/siglip2-base-patch16-256"; proc = AutoProcessor.from_pretrained(MID); base = AutoModel.from_pretrained(MID).to(dev).eval()
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# ---------------- data ----------------
HF = {}
def load_collection(name):
    rows = []
    if name == "flickr":
        d = load_dataset("lmms-lab/flickr30k", split="test"); HF["flickr"] = d; fn2row = {fn: i for i, fn in enumerate(d["filename"])}
        for p in (json.loads(l) for l in open(f"{REPO_DIR}/data/flickr30k_entities/pairs.jsonl")):
            fn = os.path.basename(p["image"])
            if fn in fn2row: rows.append(dict(image=("flickr", fn2row[fn]), caption=p["caption"], split=p["split"], id=p["id"]))
    elif name == "skincap":
        d = load_dataset("sercetexam9/skincap", split="train"); HF["skincap"] = d; p = np.random.RandomState(0).permutation(len(d))
        sp = {int(k): ("test" if j < 600 else ("val" if j < 1200 else "train")) for j, k in enumerate(p)}
        for i in range(len(d)): rows.append(dict(image=("skincap", i), caption=str(d[i]["text"]), split=sp[i], id=str(i)))
    return rows
def pil(x): return HF[x[0]][x[1]]["image"].convert("RGB")
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

# ---------------- encoders: pooled + per-patch (head applied to each token alone), phrases ----------------
def vision_of(model): return model.base_model.model.vision_model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model.vision_model
def text_feats(model, texts, bs=256, grad=False):
    out = []
    for s in range(0, len(texts), bs):
        pt = proc(text=list(texts[s:s + bs]), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.get_text_features(**pt); f = f if torch.is_tensor(f) else f.pooler_output
        out.append(torch.nn.functional.normalize(f.float(), dim=-1))
    return torch.cat(out)
def image_feats(model, ims, grad=False, patches=True):
    """pooled (B,d) and per-patch (B,N,d) unit embeddings in the joint space."""
    vm = vision_of(model); pi = proc(images=ims, return_tensors="pt").to(dev)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        o = vm(pixel_values=pi["pixel_values"]); tok = o.last_hidden_state; B, N, D = tok.shape
        pooled = torch.nn.functional.normalize(o.pooler_output.float(), dim=-1)
        if not patches: return pooled, None
        e = vm.head(tok.reshape(B * N, 1, D)).reshape(B, N, D)
    return pooled, torch.nn.functional.normalize(e.float(), dim=-1)
def pad(embs_list, d):
    P = max(1, max(len(e) for e in embs_list)); X = torch.zeros(len(embs_list), P, d, device=dev); M = torch.zeros(len(embs_list), P, dtype=torch.bool, device=dev)
    for b, e in enumerate(embs_list):
        if len(e): X[b, :len(e)] = e; M[b, :len(e)] = True
    return X, M

class SetHead(torch.nn.Module):
    """lambda, tau and the optional phrase gate of the set score; with maps=True also the two residual linear maps (the frozen-feature parameterization)."""
    def __init__(self, d=768, lam=1.0, tau=0.05, gate=False, maps=False, use_set=True):
        super().__init__(); self.log_lam = torch.nn.Parameter(torch.tensor(math.log(lam))); self.log_tau = torch.nn.Parameter(torch.tensor(math.log(tau))); self.use_set = use_set
        self.gate = torch.nn.Linear(d, 1) if gate else None; self.maps = maps
        if maps: self.Wt, self.Wi = torch.nn.Linear(d, d), torch.nn.Linear(d, d)
        for L in ([self.gate] if gate else []) + ([self.Wt, self.Wi] if maps else []): torch.nn.init.zeros_(L.weight); torch.nn.init.zeros_(L.bias)
    def T(self, x): return torch.nn.functional.normalize(x + self.Wt(x), dim=-1) if self.maps else x
    def I(self, x): return torch.nn.functional.normalize(x + self.Wi(x), dim=-1) if self.maps else x
    def weights(self, Tp, phm):
        if self.gate is not None: return torch.softmax(self.gate(Tp).squeeze(-1).masked_fill(~phm, float("-inf")), 1).nan_to_num(0.0)
        return phm.float() / phm.float().sum(1, keepdim=True).clamp(min=1)
    def score(self, q, ph, phm, g, patches):
        S = self.T(q) @ self.I(g).T
        if not self.use_set: return S
        tau = self.log_tau.exp(); Tp = self.T(ph); Ip = self.I(patches); B, Pn, d = Tp.shape; C, J, _ = Ip.shape
        sim = (Tp.reshape(B * Pn, d) @ Ip.reshape(C * J, d).T).reshape(B, Pn, C, J).permute(0, 2, 1, 3)
        soft = tau * torch.logsumexp(sim / tau, dim=3) - tau * math.log(J)
        return S + self.log_lam.exp() * torch.einsum("bcp,bp->bc", soft, self.weights(Tp, phm))

def ranks(S):
    true = np.diag(S); return (S >= true[:, None]).sum(1)
def rec(r): return {"R@1": float((r <= 1).mean()), "R@5": float((r <= 5).mean()), "R@10": float((r <= 10).mean()), "R@50": float((r <= 50).mean()), "median_rank": float(np.median(r))}

@torch.no_grad()
def encode_rows(model, rows, phr, bs=64, patches=True):
    """Encode a list of rows: q (n,d), g (n,d), patch store (n,N,d) fp16 on CPU, phrase embeddings per row (list of (P_i,d) on CPU)."""
    Q, G, PT, PH = [], [], [], []
    for s in range(0, len(rows), bs):
        ch = rows[s:s + bs]; g, p = image_feats(model, [pil(r["image"]) for r in ch], patches=patches); q = text_feats(model, [r["caption"] for r in ch])
        Q.append(q.cpu()); G.append(g.cpu()); PT.append(p.half().cpu() if patches else None)
        flat = [t for r in ch for t in phr[r["id"]]]; e = text_feats(model, flat).cpu() if flat else torch.zeros(0, q.shape[1]); k = 0
        for r in ch: n = len(phr[r["id"]]); PH.append(e[k:k + n]); k += n
    return torch.cat(Q), torch.cat(G), (torch.cat(PT) if patches else None), PH
@torch.no_grad()
def retrieval(head, enc, use_set, qchunk=64, gchunk=256):
    Q, G, PT, PH = enc; n = len(Q); S = np.zeros((n, n), np.float32); head.eval(); head.use_set, prev = use_set, head.use_set
    q = Q.to(dev); g = G.to(dev); ph, phm = pad([e.to(dev) for e in PH], Q.shape[1])
    for gs in range(0, n, gchunk):
        patches = PT[gs:gs + gchunk].to(dev).float() if use_set else None
        for qs in range(0, n, qchunk): S[qs:qs + qchunk, gs:gs + gchunk] = head.score(q[qs:qs + qchunk], ph[qs:qs + qchunk], phm[qs:qs + qchunk], g[gs:gs + gchunk], patches).cpu().numpy()
    head.use_set = prev; return S

# ---------------- Flickr30k Entities pointing game ----------------
def mask_of(boxes, W, H, g=16):
    cx = (np.arange(g) + 0.5) * W / g; cy = (np.arange(g) + 0.5) * H / g; M = np.zeros((g, g), bool)
    for x0, y0, x1, y1 in boxes: M |= (cy[:, None] >= y0) & (cy[:, None] <= y1) & (cx[None, :] >= x0) & (cx[None, :] <= x1)
    return M.reshape(-1)
def pointing_items(rows_te):
    row_of = {r["id"]: k for k, r in enumerate(rows_te)}; E = [e for e in json.load(open(f"{REPO_DIR}/data/flickr30k_entities/entities_test.json")) if e["id"] in row_of]; items = []
    for e in E:
        W, H = e["size"]; allm = mask_of([b for s in e["sentences"] for p in s["phrases"] for b in p["boxes"]], W, H)
        for s in e["sentences"]:
            for p in s["phrases"]:
                if p["boxes"]: items.append((row_of[e["id"]], p["text"], (p["types"] or ["?"])[0], mask_of(p["boxes"], W, H), allm))
    return items
@torch.no_grad()
def pointing(model, head, items, PT):
    """argmax patch of <T(phrase), I(patch)> with the model's own features (and the head's maps if it has them)."""
    T = text_feats(model, [t for _, t, _, _, _ in items]); Tp = head.T(T); rows = np.array([r for r, *_ in items]); J = np.zeros(len(items), int)
    for r in np.unique(rows):
        Ip = head.I(PT[r].to(dev).float()); k = np.where(rows == r)[0]; J[k] = (Tp[k] @ Ip.T).argmax(1).cpu().numpy()
    masks = np.stack([m for *_, m, _ in items]); allm = np.stack([m for *_, m in items]); hit = masks[np.arange(len(items)), J]; anyhit = allm[np.arange(len(items)), J]; types = [t for _, _, t, _, _ in items]
    return {"pointing_acc": float(hit.mean()), "any_box_acc": float(anyhit.mean()), "n_phrases": len(items), "by_type": {t: {"acc": float(hit[[tt == t for tt in types]].mean()), "n": int(sum(tt == t for tt in types))} for t in sorted(set(types))}}

# ---------------- training ----------------
def train_lora(rows_tr, rows_va, phr, variant, tag):
    """LoRA on both towers, pooled or set objective; best-val checkpoint by the variant's own score."""
    torch.manual_seed(SEED); lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
    lm = get_peft_model(copy.deepcopy(base), lcfg); use_set = variant != "lora_pooled"; head = SetHead(gate=variant.endswith("w"), use_set=use_set).to(dev)
    params = [p for p in lm.parameters() if p.requires_grad]; log(f"{tag}: {sum(p.numel() for p in params)/1e6:.2f}M LoRA params, set={use_set}")
    groups = [{"params": params, "lr": 1e-4, "weight_decay": 1e-4}, {"params": [head.log_lam, head.log_tau], "lr": 1e-2, "weight_decay": 0.0}] + ([{"params": head.gate.parameters(), "lr": 1e-3, "weight_decay": 0.0}] if head.gate is not None else [])
    opt = torch.optim.AdamW(groups); spe = math.ceil(len(rows_tr) / BS); epochs = max(EPOCHS_LORA, math.ceil(1000 / spe)); best, best_state, steps, t0 = -1, None, 0, time.time()
    for ep in range(epochs):
        lm.train(); head.train(); perm = np.random.RandomState(SEED * 100 + ep).permutation(len(rows_tr))
        for s in range(0, len(perm), BS):
            ch = [rows_tr[i] for i in perm[s:s + BS]]
            if len(ch) < 4: continue
            g, patches = image_feats(lm, [pil(r["image"]) for r in ch], grad=True, patches=use_set); q = text_feats(lm, [r["caption"] for r in ch], grad=True)
            if use_set:
                flat = [t for r in ch for t in phr[r["id"]]]; e = text_feats(lm, flat, grad=True); k = 0; embs = []
                for r in ch: n = len(phr[r["id"]]); embs.append(e[k:k + n]); k += n
                ph, phm = pad(embs, q.shape[1])
            else: ph, phm, patches = q[:, None, :], torch.ones(len(ch), 1, dtype=torch.bool, device=dev), None
            S = head.score(q, ph, phm, g, patches) / TEMP; y = torch.arange(len(ch), device=dev)
            loss = 0.5 * (torch.nn.functional.cross_entropy(S, y) + torch.nn.functional.cross_entropy(S.T, y)); opt.zero_grad(); loss.backward(); opt.step(); steps += 1
        lm.eval(); enc = encode_rows(lm, rows_va, phr, patches=use_set); r10 = float((ranks(retrieval(head, enc, use_set)) <= 10).mean())
        log(f"  {tag} ep{ep} step{steps} loss {loss.item():.3f} val R@10 {r10:.4f} lam {head.log_lam.exp().item():.2f} tau {head.log_tau.exp().item():.4f} [{time.time()-t0:.0f}s]")
        if r10 > best: best, best_state = r10, ({k: v.detach().clone() for k, v in lm.state_dict().items()}, {k: v.detach().clone() for k, v in head.state_dict().items()})
    lm.load_state_dict(best_state[0]); head.load_state_dict(best_state[1]); lm.eval(); head.eval(); return lm, head, {"steps": steps, "epochs": epochs, "val_R@10": best}

def train_setmaps(enc_tr, enc_va, variant, tag, epochs=30, bs=128):
    """The paper's two residual linear maps on frozen features with the same set objective (SetMaps in story/sets/train_sets.py)."""
    torch.manual_seed(SEED); head = SetHead(gate=variant.endswith("w"), maps=True).to(dev); Q, G, PT, PH = enc_tr
    groups = [{"params": [head.Wt.weight, head.Wt.bias, head.Wi.weight, head.Wi.bias], "lr": 1e-4, "weight_decay": 1e-4}, {"params": [head.log_lam, head.log_tau], "lr": 1e-2, "weight_decay": 0.0}] + ([{"params": head.gate.parameters(), "lr": 1e-3, "weight_decay": 0.0}] if head.gate is not None else [])
    opt = torch.optim.AdamW(groups); rng = np.random.RandomState(SEED); best, best_state, t0 = -1, None, time.time()
    for ep in range(epochs):
        head.train(); perm = rng.permutation(len(Q))
        for s in range(0, len(perm), bs):
            bi = np.sort(perm[s:s + bs])
            if len(bi) < 8: continue
            q = Q[bi].to(dev); g = G[bi].to(dev); patches = PT[bi].to(dev).float(); ph, phm = pad([PH[i].to(dev) for i in bi], Q.shape[1])
            S = head.score(q, ph, phm, g, patches) / TEMP; y = torch.arange(len(bi), device=dev); loss = 0.5 * (torch.nn.functional.cross_entropy(S, y) + torch.nn.functional.cross_entropy(S.T, y)); opt.zero_grad(); loss.backward(); opt.step()
        r10 = float((ranks(retrieval(head, enc_va, True)) <= 10).mean())
        if ep % 5 == 4 or ep == epochs - 1: log(f"  {tag} ep{ep} loss {loss.item():.3f} val R@10 {r10:.4f} lam {head.log_lam.exp().item():.2f} tau {head.log_tau.exp().item():.4f} [{time.time()-t0:.0f}s]")
        if r10 > best: best, best_state = r10, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval(); return head, {"epochs": epochs, "val_R@10": best}

# ---------------- main ----------------
for name in COLLECTIONS:
    log("==== collection", name); rows = load_collection(name); tr = [r for r in rows if r["split"] == "train"]; va = [r for r in rows if r["split"] == "val"]; te = [r for r in rows if r["split"] == "test"]
    va = [va[i] for i in np.random.RandomState(0).permutation(len(va))[:VAL_MAX]]; RES.setdefault(name, {"n_train": len(tr), "n_val": len(va), "n_test": len(te)}); log(name, "train", len(tr), "val", len(va), "test", len(te))
    caps = {r["id"]: r["caption"] for r in rows}; ids = list(caps); phr = {i: phrases_of(caps[i], d) for i, d in zip(ids, nlp.pipe([caps[i][:1000] for i in ids], batch_size=256))}
    RES[name]["phrases_per_caption"] = float(np.mean([len(v) for v in phr.values()])); items = pointing_items(te) if name == "flickr" else None
    enc_te = encode_rows(base, te, phr); enc_va = encode_rows(base, va, phr)
    if "frozen" not in RES[name]:
        h0 = SetHead().to(dev).eval(); r = {"retrieval_pooled": rec(ranks(retrieval(h0, enc_te, False))), "retrieval_set_lam1_tau05": rec(ranks(retrieval(h0, enc_te, True)))}
        if items: r["pointing"] = pointing(base, h0, items, enc_te[2])
        RES[name]["frozen"] = r; log(name, "frozen", json.dumps({k: (v.get("R@10") if "R@10" in v else v.get("pointing_acc")) for k, v in r.items()})); save()
    enc_tr = None
    for variant in VARIANTS:
        if variant in RES[name]: log("have", name, variant); continue
        t0 = time.time()
        if variant.startswith("setmaps"):
            if enc_tr is None: enc_tr = encode_rows(base, tr, phr); log(name, "train features cached", tuple(enc_tr[2].shape))
            head, info = train_setmaps(enc_tr, enc_va, variant, f"{name}/{variant}"); model = base; enc = enc_te
        else:
            model, head, info = train_lora(tr, va, phr, variant, f"{name}/{variant}"); enc = encode_rows(model, te, phr)
        r = {"retrieval_pooled": rec(ranks(retrieval(head, enc, False))), "retrieval_set": rec(ranks(retrieval(head, enc, True))) if head.use_set else None, "lam": float(head.log_lam.exp()), "tau": float(head.log_tau.exp()), "seconds": time.time() - t0} | info
        if items: r["pointing"] = pointing(model, head, items, enc[2])
        RES[name][variant] = r; log(name, variant, json.dumps({k: (v.get("R@10") if isinstance(v, dict) and "R@10" in v else (v.get("pointing_acc") if isinstance(v, dict) else v)) for k, v in r.items()})); save()
        if variant.startswith("lora"): del model; torch.cuda.empty_cache()
log("ALL DONE")
