"""Job: data-scaling curve, two linear maps versus LoRA (2026-09-22).
For each collection and each training-set size n in JOB_SIZES (default 50,100,300,1000,all), draw the same random subset of the train
pairs for every method (seeded), then: (a) the paper's two residual linear maps on frozen SigLIP2 embeddings, (b) LoRA r=16 on the attention
projections of both towers (the 14 Sep recipe), and optionally (c) projection-head fine-tuning. Same val split for model selection (capped at
JOB_VAL_CAP rows), same test gallery for R@k. Training budget is matched in optimizer steps, not epochs: epochs = clip(ceil(400 / steps_per_epoch), 5, 100),
batch = min(64, n) for the fine-tunes and min(256, n) for the maps, so a 50-pair run still takes ~100 steps. Seeds: JOB_SEEDS (default 3; 1 for n=all).
Writes results/<RUN>/results.json after every (collection, n, seed) point:
  RES[c] = {"n_train": N, "n_test": M, "raw": R@k, "points": {"50": {"0": {"maps": R@k, "lora": R@k, "head": R@k, "lora_steps": s, ...}}}}
Env: JOB_COLLECTIONS (default skincap,rsicd,duluth,semart), JOB_SIZES, JOB_SEEDS, JOB_METHODS (default maps,lora), JOB_VAL_CAP (500).
Run by the Colab launcher (results/<RUN>/ inside the scratch repo)."""
import os, sys, subprocess, json, time, math
RUN = "registergap_scaling_2026-09-22"
REPO_DIR = os.environ.get("SCRATCH_DIR", os.getcwd()); OUT = os.path.join(REPO_DIR, "results", RUN); os.makedirs(OUT, exist_ok=True)
def push_results():
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "results"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "commit", "-qm", f"results: {RUN}"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "push", "-q"], check=False)
    except Exception as e: print("push_results failed:", e)
LOGF = open(f"{OUT}/log.txt", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a); print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()
subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"])          # Colab's torchao 0.10 breaks peft
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "transformers", "datasets", "peft", "accelerate", "huggingface_hub"])
import torch; log(torch.__version__, torch.cuda.get_device_name(0))
# ---------------- toolkit ----------------
# the toolkit from the laptop, verbatim
GAP_SRC = r'''"""Register-gap toolkit: raw recall, near-miss mass N_k, two linear maps, polar decomposition. Works on cached unit-norm embeddings."""
import numpy as np, torch, warnings
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

def load(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    return d

def split(d, name):
    m = d["split"] == name
    return d["txt"][m], d["img"][m]

def ranks(Q, G):
    """Rank (1-based) of the true pair for each query i against gallery G (same ordering). Ties pessimistic."""
    S = Q @ G.T
    true = np.diag(S)
    return 1 + (S >= true[:, None]).sum(1) - 1  # count of scores >= true, includes self -> rank

def recall(Q, G, ks=(1, 10, 50)):
    r = ranks(Q, G)
    return {f"R@{k}": float((r <= k).mean()) for k in ks}

def near_miss_mass(Q, G, k=50, band=3, n=None, seeds=(0, 1, 2)):
    """N_k = P[k < rank <= band*k] over a random sample of size n of the catalog (default: all)."""
    n = n or len(Q)
    vals = []
    for s in seeds:
        idx = np.random.RandomState(s).permutation(len(Q))[:n]
        r = ranks(Q[idx], G[idx])
        vals.append(((r > k) & (r <= band * k)).mean())
    return float(np.mean(vals))

class Maps(torch.nn.Module):
    """phi(x) = normalize(x + W x + b), W,b zero-init. tied=True shares one map for both modalities."""
    def __init__(self, d, form="two"):
        super().__init__()
        self.form = form
        self.Wt = torch.nn.Linear(d, d); self.Wi = torch.nn.Linear(d, d)
        for L in (self.Wt, self.Wi):
            torch.nn.init.zeros_(L.weight); torch.nn.init.zeros_(L.bias)
    def text(self, q):
        if self.form == "image-only": return q
        L = self.Wt
        return torch.nn.functional.normalize(q + L(q), dim=-1)
    def image(self, g):
        if self.form == "text-only": return g
        L = self.Wt if self.form == "tied" else self.Wi
        return torch.nn.functional.normalize(g + L(g), dim=-1)

def train_maps(Qtr, Gtr, Qva, Gva, form="two", epochs=30, bs=256, lr=1e-4, wd=1e-4, tau=0.05, seed=0, verbose=False):
    torch.manual_seed(seed)
    d = Qtr.shape[1]
    m = Maps(d, form)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Qt, Gt = torch.tensor(Qtr), torch.tensor(Gtr)
    best, best_state = -1, None
    for ep in range(epochs):
        perm = torch.randperm(len(Qt))
        m.train()
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]
            q, g = m.text(Qt[idx]), m.image(Gt[idx])
            logits = q @ g.T / tau
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(idx)))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        r50 = recall(*apply(m, Qva, Gva))["R@50"]
        if verbose: print(f"  ep{ep:02d} loss {loss.item():.3f} val R@50 {r50:.4f}")
        if r50 > best:
            best, best_state = r50, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m

@torch.no_grad()
def apply(m, Q, G):
    return m.text(torch.tensor(Q)).numpy(), m.image(torch.tensor(G)).numpy()

def polar(m, side="t"):
    """A = I + W  ->  A = R S (R orthogonal, S sym. PSD). Returns R, S, singular values of S."""
    L = m.Wt if side == "t" else m.Wi
    A = np.eye(L.weight.shape[0]) + L.weight.detach().numpy()
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt
    S = Vt.T @ np.diag(s) @ Vt
    return R, S, s
'''
open(os.path.join(REPO_DIR, 'gap.py'),'w').write(GAP_SRC); sys.path.insert(0, REPO_DIR)
from gap import *
print('gap.py ok')
# ---------------- loaders ----------------
# ---- data loaders (same rules as the laptop runs) ----
import numpy as np, io, zipfile, glob, csv
from datasets import load_dataset, Image as HFImage
from huggingface_hub import hf_hub_download
def load_collection(name):
    rows = []
    if name == "rsicd":
        ds = load_dataset("arampacha/rsicd")
        for split in ["train", "valid", "test"]:
            for i, r in enumerate(ds[split]):
                caps = r["captions"]; caps = [caps] if isinstance(caps, str) else caps
                rows.append(dict(image=r["image"], caption=max(caps, key=len), split={"valid":"val"}.get(split, split), id=str(r["filename"])))
    elif name == "skincap":
        d = load_dataset("sercetexam9/skincap", split="train"); p = np.random.RandomState(0).permutation(len(d))
        sp = {int(k): ("test" if j < 600 else ("val" if j < 1200 else "train")) for j, k in enumerate(p)}
        for i, r in enumerate(d): rows.append(dict(image=r["image"], caption=str(r["text"]), split=sp[i], id=str(i)))
    elif name == "semart":
        z = hf_hub_download("leo20000306/SemArt", "SemArt.zip", repo_type="dataset"); os.makedirs("/content/semart", exist_ok=True)
        if not glob.glob("/content/semart/**/semart_train.csv", recursive=True):
            with zipfile.ZipFile(z) as zf: zf.extractall("/content/semart")
        base = os.path.dirname(glob.glob("/content/semart/**/semart_train.csv", recursive=True)[0]); imgdir = glob.glob(f"{base}/Images")[0]
        for split in ["train", "val", "test"]:
            with open(f"{base}/semart_{split}.csv", encoding="latin-1") as f:
                for r in csv.DictReader(f, delimiter="\t"): rows.append(dict(image=os.path.join(imgdir, r["IMAGE_FILE"]), caption=r["DESCRIPTION"].strip(), split=split, id=r["IMAGE_FILE"]))
    elif name in ("ospreys", "duluth", "popocatepetl"):
        import json, hashlib, subprocess
        repo = {"ospreys": "uk-osprey-cam-captions", "duluth": "duluth-harbor-cam-captions", "popocatepetl": "popocatepetl-cam-captions"}[name]
        base = f"/content/{repo}"
        if not os.path.isdir(base): subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/ANONYMIZED/{repo}.git", base], check=True)
        for i, p in enumerate(json.loads(l) for l in open(f"{base}/pairs.jsonl")):
            if not p.get("is_cam", True): continue
            cap = " ".join(p["caption"].split())
            if len(cap) < 3: continue
            key = p["url"] if name == "ospreys" else (p.get("date") or "")[:10]
            h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
            img = os.path.join(base, p["image"])
            if not os.path.exists(img): continue
            rows.append(dict(image=img, caption=cap, split="train" if h < 70 else ("val" if h < 85 else "test"), id=f"{key}#{i}"))
    return rows
from PIL import Image
def pil(x): return (Image.open(x) if isinstance(x, str) else x).convert("RGB")
# ---------------- model / training ----------------
# ---- model, embedding, InfoNCE training loop (shared by projection-head and LoRA fine-tunes) ----
import json, time, copy
from transformers import AutoModel, AutoProcessor
MID = "google/siglip2-base-patch16-256"; dev = "cuda"
proc = AutoProcessor.from_pretrained(MID)
def feats(model, out): return out if torch.is_tensor(out) else out.pooler_output
@torch.no_grad()
def embed(model, rows, bs=128):
    model.eval(); I, T = [], []
    for s in range(0, len(rows), bs):
        ch = rows[s:s+bs]
        pi = proc(images=[pil(r["image"]) for r in ch], return_tensors="pt").to(dev)
        pt = proc(text=[r["caption"] for r in ch], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fi = feats(model, model.get_image_features(**pi)); ft = feats(model, model.get_text_features(**pt))
        I.append(torch.nn.functional.normalize(fi.float(), dim=-1).cpu().numpy()); T.append(torch.nn.functional.normalize(ft.float(), dim=-1).cpu().numpy())
    return np.concatenate(I), np.concatenate(T)
def evaluate(model, rows_te):
    I, T = embed(model, rows_te); return recall(T, I)
def finetune(model, rows_tr, rows_va, rows_te, epochs=5, bs=64, lr=1e-5, tau=0.05, tag=""):
    params = [p for p in model.parameters() if p.requires_grad]; print(f"{tag}: {sum(p.numel() for p in params)/1e6:.2f}M trainable params")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); best, best_state = -1, None; t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = np.random.RandomState(ep).permutation(len(rows_tr))
        for s in range(0, len(perm), bs):
            ch = [rows_tr[i] for i in perm[s:s+bs]]
            pi = proc(images=[pil(r["image"]) for r in ch], return_tensors="pt").to(dev)
            pt = proc(text=[r["caption"] for r in ch], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fi = torch.nn.functional.normalize(feats(model, model.get_image_features(**pi)).float(), dim=-1); ft = torch.nn.functional.normalize(feats(model, model.get_text_features(**pt)).float(), dim=-1)
            logits = ft @ fi.T / tau; loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ch), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        r = evaluate(model, rows_va)["R@50"]; print(f"  {tag} ep{ep} loss {loss.item():.3f} val R@50 {r:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if r > best: best, best_state = r, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return evaluate(model, rows_te)


# ---------------- scaling loop ----------------
from peft import LoraConfig, get_peft_model
COLLECTIONS = os.environ.get("JOB_COLLECTIONS", "skincap,rsicd,duluth,semart").split(",")
SIZES = os.environ.get("JOB_SIZES", "50,100,300,1000,all").split(",")
NSEEDS = int(os.environ.get("JOB_SEEDS", 3)); METHODS = os.environ.get("JOB_METHODS", "maps,lora").split(","); VAL_CAP = int(os.environ.get("JOB_VAL_CAP", 500))
RESULTS_PATH = f"{OUT}/results.json"; RES = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
def save(): json.dump(RES, open(RESULTS_PATH, "w"), indent=1); push_results()
def budget(n, bs, target=400):
    """epochs so that the run has about `target` optimizer steps, clipped to [5, 100]"""
    spe = math.ceil(n / bs); return max(5, min(100, math.ceil(target / spe))), spe

def finetune_steps(model, rows_tr, rows_va, rows_te, epochs, bs, lr, tag, eval_every):
    """the 14 Sep finetune loop with a step budget and periodic val (best-val checkpoint), returns (test R@k, steps)"""
    params = [p for p in model.parameters() if p.requires_grad]; log(f"{tag}: {sum(p.numel() for p in params)/1e6:.2f}M trainable, {epochs} ep x {math.ceil(len(rows_tr)/bs)} steps")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4); best, best_state, steps, t0 = -1, None, 0, time.time()
    for ep in range(epochs):
        model.train(); perm = np.random.RandomState(ep).permutation(len(rows_tr))
        for s in range(0, len(perm), bs):
            ch = [rows_tr[i] for i in perm[s:s+bs]]
            if len(ch) < 2: continue
            pi = proc(images=[pil(r["image"]) for r in ch], return_tensors="pt").to(dev)
            pt = proc(text=[r["caption"] for r in ch], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fi = torch.nn.functional.normalize(feats(model, model.get_image_features(**pi)).float(), dim=-1); ft = torch.nn.functional.normalize(feats(model, model.get_text_features(**pt)).float(), dim=-1)
            logits = ft @ fi.T / 0.05; loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ch), device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            r = evaluate(model, rows_va)["R@50"]; log(f"  {tag} ep{ep} step{steps} loss {loss.item():.3f} val R@50 {r:.4f} [{time.time()-t0:.0f}s]")
            if r > best: best, best_state = r, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); out = evaluate(model, rows_te); out["steps"] = steps; out["val_R@50"] = best; return out

def train_maps_budget(Qtr, Gtr, Qva, Gva, n, seed):
    bs = min(256, n); epochs, _ = budget(n, bs, target=600)
    return train_maps(Qtr, Gtr, Qva, Gva, form="two", epochs=epochs, bs=bs, seed=seed)

for name in COLLECTIONS:
    log("==== collection", name)
    rows = load_collection(name); tr = [r for r in rows if r["split"] == "train"]; va = [r for r in rows if r["split"] == "val"]; te = [r for r in rows if r["split"] == "test"]
    va = [va[i] for i in np.random.RandomState(0).permutation(len(va))[:VAL_CAP]]
    RES.setdefault(name, {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "points": {}}); log(name, "train", len(tr), "val", len(va), "test", len(te))
    base = AutoModel.from_pretrained(MID).to(dev)
    Itr, Ttr = embed(base, tr); Iva, Tva = embed(base, va); Ite, Tte = embed(base, te)
    if "raw" not in RES[name]: RES[name]["raw"] = recall(Tte, Ite); save()
    log(name, "raw", RES[name]["raw"])
    for size in SIZES:
        n = len(tr) if size == "all" else min(int(size), len(tr))
        if size != "all" and n < int(size): log("skip", size, "train has only", len(tr)); continue
        nseeds = 1 if size == "all" else NSEEDS
        for seed in range(nseeds):
            key = f"{size}"; P = RES[name]["points"].setdefault(key, {}).setdefault(str(seed), {"n": n})
            if all(m in P for m in METHODS): log("have", name, key, seed); continue
            idx = np.arange(len(tr)) if size == "all" else np.random.RandomState(1000 + seed).permutation(len(tr))[:n]
            sub = [tr[i] for i in idx]
            if "maps" in METHODS and "maps" not in P:
                m = train_maps_budget(Ttr[idx], Itr[idx], Tva, Iva, n, seed); P["maps"] = recall(*apply(m, Tte, Ite)); log(name, key, seed, "maps", P["maps"]); save()
            bs = min(64, n); epochs, spe = budget(n, bs); eval_every = max(1, epochs // 8)
            if "lora" in METHODS and "lora" not in P:
                torch.manual_seed(seed)
                lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
                try:
                    lm = get_peft_model(copy.deepcopy(base), lcfg); P["lora"] = finetune_steps(lm, sub, va, te, epochs, bs, 1e-4, f"{name}/{key}/s{seed}/lora", eval_every); del lm
                except Exception as e:
                    import traceback; traceback.print_exc(); P["lora_error"] = repr(e)[:500]
                torch.cuda.empty_cache(); log(name, key, seed, "lora", P.get("lora")); save()
            if "head" in METHODS and "head" not in P:
                torch.manual_seed(seed); ph = copy.deepcopy(base)
                for p in ph.parameters(): p.requires_grad = False
                for pn, p in ph.named_parameters():
                    if "head" in pn or "projection" in pn: p.requires_grad = True
                P["head"] = finetune_steps(ph, sub, va, te, epochs, bs, 1e-4, f"{name}/{key}/s{seed}/head", eval_every); del ph; torch.cuda.empty_cache(); log(name, key, seed, "head", P["head"]); save()
    del base; torch.cuda.empty_cache()
    log(name, json.dumps({k: {s: {m: round(v["R@10"], 3) for m, v in d.items() if isinstance(v, dict) and "R@10" in v} for s, d in sd.items()} for k, sd in RES[name]["points"].items()}))
log("ALL DONE")
