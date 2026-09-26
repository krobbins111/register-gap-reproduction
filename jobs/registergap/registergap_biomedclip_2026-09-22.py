"""Job: two maps on BiomedCLIP (2026-09-22). The draft's grid has BiomedCLIP on ROCOv2 as the one cell where the maps do nothing (raw R@50 0.89,
N50 0.04): a specialist encoder that has seen PubMed figures has no register gap left to repair there. This reruns that cell on our split
(13,326 / 1,651 / 9,927, the memo protocol every other ROCOv2 number uses) so it sits in the same table as SigLIP2, TIPSv2 and Gemini, and adds
SkinCAP (where the draft found BiomedCLIP gains least of all encoders). BiomedCLIP = microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
via open_clip (ViT-B/16 at 224 px, PubMedBERT text, context 256). Reports frozen R@k, N50 (band 3), two maps (3 seeds), and the text- and
image-only arms. Env: JOB_COLLECTIONS (rocov2,skincap)."""
import os, sys, subprocess, json, time
NAME = os.environ.get("JOB_NAME", "registergap_biomedclip_2026-09-22"); REPO_DIR = os.environ.get("SCRATCH_DIR", os.getcwd()); OUT = os.path.join(REPO_DIR, "results", NAME); os.makedirs(OUT, exist_ok=True)
def push_results():
    try:
        subprocess.run(["git", "-C", REPO_DIR, "add", "results"], check=True); subprocess.run(["git", "-C", REPO_DIR, "commit", "-qm", f"results: {NAME}"], check=False)
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--rebase", "-q", "origin", "main"], check=False); subprocess.run(["git", "-C", REPO_DIR, "push", "-q"], check=False)
    except Exception as e: print("push_results failed:", e)
LOGF = open(f"{OUT}/joblog.txt", "a")
def log(*a):
    line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a); print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "open_clip_torch", "datasets", "huggingface_hub"], check=False)
import numpy as np, torch, open_clip
from datasets import load_dataset
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
dev = "cuda"; MID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, preprocess = open_clip.create_model_from_pretrained(MID); tok = open_clip.get_tokenizer(MID); model = model.to(dev).eval(); log("loaded", MID)
def collection(name):
    rows = []
    if name == "rocov2":
        files = {"train": [f"data/train-{i:05d}-of-00027.parquet" for i in range(6)], "validation": ["data/validation-00000-of-00006.parquet"], "test": [f"data/test-{i:05d}-of-00006.parquet" for i in range(6)]}
        ds = load_dataset("eltorio/ROCOv2-radiology", data_files=files, verification_mode="no_checks")
        for split, s in (("train", "train"), ("validation", "val"), ("test", "test")):
            d = ds[split]; caps = d["caption"]
            for i in range(len(d)): rows.append(dict(image=(d, i), caption=str(caps[i]), split=s))
    elif name == "skincap":
        d = load_dataset("sercetexam9/skincap", split="train"); p = np.random.RandomState(0).permutation(len(d)); sp = {int(k): ("test" if j < 600 else ("val" if j < 1200 else "train")) for j, k in enumerate(p)}; texts = d["text"]
        for i in range(len(d)): rows.append(dict(image=(d, i), caption=str(texts[i]), split=sp[i]))
    return rows
@torch.no_grad()
def embed(rows, bs=128):
    I, T = [], []
    for s in range(0, len(rows), bs):
        ch = rows[s:s + bs]; ims = torch.stack([preprocess(r["image"][0][r["image"][1]]["image"].convert("RGB")) for r in ch]).to(dev); tx = tok([r["caption"] for r in ch], context_length=256).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): fi = model.encode_image(ims); ft = model.encode_text(tx)
        I.append(torch.nn.functional.normalize(fi.float(), dim=-1).cpu().numpy()); T.append(torch.nn.functional.normalize(ft.float(), dim=-1).cpu().numpy())
        if (s // bs) % 20 == 0: log(f"  {s + len(ch)}/{len(rows)}")
    return np.concatenate(I), np.concatenate(T)
RESULTS_PATH = f"{OUT}/results.json"; RES = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
def save(): json.dump(RES, open(RESULTS_PATH, "w"), indent=1); push_results()
for name in os.environ.get("JOB_COLLECTIONS", "rocov2,skincap").split(","):
    if name in RES and "two_maps_seed2" in RES[name]: log("have", name); continue
    log("==== collection", name); rows = collection(name); sp = np.array([r["split"] for r in rows]); tr, va, te = sp == "train", sp == "val", sp == "test"; log(name, {s: int((sp == s).sum()) for s in ("train", "val", "test")})
    I, T = embed(rows); R = RES.setdefault(name, {"n_train": int(tr.sum()), "n_test": int(te.sum())})
    R["frozen"] = recall(T[te], I[te]); R["N50_b3"] = near_miss_mass(T[tr], I[tr], band=3, n=int(te.sum())); log(name, "frozen", R["frozen"], "N50", round(R["N50_b3"], 3)); save()
    for seed in range(3):
        m = train_maps(T[tr], I[tr], T[va], I[va], form="two", seed=seed); R[f"two_maps_seed{seed}"] = recall(*apply(m, T[te], I[te])); log(name, "two maps seed", seed, R[f"two_maps_seed{seed}"]); save()
    for form in ("text-only", "image-only", "tied"):
        m = train_maps(T[tr], I[tr], T[va], I[va], form=form, seed=0); R[form] = recall(*apply(m, T[te], I[te])); log(name, form, R[form])
    np.savez_compressed(f"{OUT}/{name}_biomedclip_test.npz", img=I[te].astype(np.float16), txt=T[te].astype(np.float16)); save()
log("ALL DONE")
