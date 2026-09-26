"""Two maps over sets + phrase gate (registergap-pipeline; ported from registergap story/sets/train_sets.py, the memos' code path).

The same two residual linear maps (text T, image I) act on every element of a caption's phrase set and an image's patch set:
    score(q, g) = <T(q), I(g)> + lambda * sum_p w_p * tau * logsumexp_j <T(p), I(e_j)> / tau      (soft max over patches per phrase)
with w_p uniform over the caption's phrases or a learned gate (masked softmax of a linear gate on the mapped phrase, zero-init = uniform).
Variants: pooled (lambda = 0, the paper's maps) | set | setw (gate) | set4 / set4w (separate phrase and patch maps) | phrases_only |
patches_only | set_tmap / set_imap / pooled_tmap / pooled_imap (one side frozen). Also: frozen, zeroshot_set (no maps; lambda, tau on val),
pooled_maps_set_score (pooled-trained maps, set score at test). InfoNCE both directions, temperature 0.05, AdamW 1e-4 / wd 1e-4 on the
maps (1e-2 on lambda and tau, 1e-3 on the gate), batch 128, best val R@10; bootstrap CI over test queries.

Reads the cell, patch store and phrase store through ltg.cache; the patch store's grid is read from its shape and pooled further with
--grid when asked. Writes $LTG_RESULTS/<collection>/<encoder>/sets<tag>.json in the results_<name>.json schema (keys frozen, zeroshot_set,
pooled, pooled_maps_set_score, set, setw, ... each with R@1/5/10/50, median_rank, R@10_ci, val_R@10, lam, tau); --save stores the seed-0
model of each variant under results/<collection>/<encoder>/models/ (git-ignored).
usage: python -m ltg.maps.sets --collection skincap --encoder siglip2-base-16-256 --cap 32 --variants pooled,set,setw --seeds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ltg.cache import encoder_key, load_cell, models_dir, patches_path, phrases_path, result_path  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TAU_FLOOR = 0.0            # set by main(); the paper's so400m runs used 0.01, the base memo numbers 0 (see ABLATIONS.md)
VAL_MAX = 500


def load_all(collection: str, encoder: str, grid: int | None = None, cap: int = 32, verbs: bool = False):
    """-> (sub dict with txt/img/split/caption/id, P (n, G*G, d) fp16 [memmap or pooled in RAM], phrase emb, offsets, texts)."""
    key = encoder_key(encoder); cell = load_cell(collection, key)
    sub = {"txt": cell.txt, "img": cell.img, "split": cell.split, "caption": cell.caption, "id": cell.id}
    P = np.load(patches_path(collection, key), mmap_mode="r"); ph = np.load(phrases_path(collection, key, cap, verbs), allow_pickle=True)
    assert len(P) == len(sub["split"]) == len(ph["offsets"]) - 1, "patch store, cell and phrase store disagree on the row count"
    src = int(round(np.sqrt(P.shape[1])))
    if grid and grid < src:                                        # average k x k blocks of unit patch embeddings, renormalise; kept in RAM as fp16
        assert src % grid == 0, f"grid {grid} does not divide {src}"
        k = src // grid; n = len(P); R = np.zeros((n, grid * grid, P.shape[2]), np.float16)
        for s0 in range(0, n, 512):
            X = np.asarray(P[s0:s0 + 512]).astype(np.float32).reshape(-1, src, src, P.shape[2]).reshape(-1, grid, k, grid, k, P.shape[2]).mean((2, 4)).reshape(-1, grid * grid, P.shape[2])
            R[s0:s0 + 512] = (X / np.linalg.norm(X, axis=-1, keepdims=True)).astype(np.float16)
        P = R
    return sub, P, ph["emb"], ph["offsets"], ph["texts"]


def pad_phrases(emb, offsets, idx, pmax: int = 32):
    """(len(idx), P, d) float32 + mask, P = longest phrase set in the batch (<= pmax)."""
    B = len(idx); pmax = int(min(pmax, max(offsets[i + 1] - offsets[i] for i in idx))); X = np.zeros((B, pmax, emb.shape[1]), np.float32); M = np.zeros((B, pmax), bool)
    for b, i in enumerate(idx):
        s, e = offsets[i], min(offsets[i + 1], offsets[i] + pmax); k = e - s; X[b, :k] = emb[s:e]; M[b, :k] = True
    return X, M


class SetMaps(torch.nn.Module):
    def __init__(self, d: int, lam=1.0, tau=0.05, use_pool=True, use_set=True, train_maps=True, separate=False, gate=False, train_t=True, train_i=True):
        super().__init__()
        self.Wt, self.Wi = torch.nn.Linear(d, d), torch.nn.Linear(d, d); self.separate, self.use_gate = separate, gate
        if separate: self.Wtp, self.Wip = torch.nn.Linear(d, d), torch.nn.Linear(d, d)
        if gate: self.gate = torch.nn.Linear(d, 1)
        for L in [self.Wt, self.Wi] + ([self.Wtp, self.Wip] if separate else []) + ([self.gate] if gate else []):
            torch.nn.init.zeros_(L.weight); torch.nn.init.zeros_(L.bias); L.weight.requires_grad_(train_maps); L.bias.requires_grad_(train_maps)
        for L, on in ((self.Wt, train_t), (self.Wi, train_i)):
            if not on: L.weight.requires_grad_(False); L.bias.requires_grad_(False)
        self.log_lam = torch.nn.Parameter(torch.tensor(float(np.log(lam)))); self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(tau))))
        self.use_pool, self.use_set = use_pool, use_set
    def T(self, x): return torch.nn.functional.normalize(x + self.Wt(x), dim=-1)
    def I(self, x): return torch.nn.functional.normalize(x + self.Wi(x), dim=-1)
    def Tp(self, x): return torch.nn.functional.normalize(x + (self.Wtp if self.separate else self.Wt)(x), dim=-1)
    def Ip(self, x): return torch.nn.functional.normalize(x + (self.Wip if self.separate else self.Wi)(x), dim=-1)
    def phrase_weights(self, Tp, phm):
        if self.use_gate:
            logits = self.gate(Tp).squeeze(-1).masked_fill(~phm, float("-inf")); return torch.softmax(logits, dim=1).nan_to_num(0.0)
        return phm.float() / phm.float().sum(1, keepdim=True).clamp(min=1)
    def score(self, q, ph, phm, g, patches):
        S = torch.zeros(q.shape[0], g.shape[0], device=q.device)
        if self.use_pool: S = S + self.T(q) @ self.I(g).T
        if self.use_set:
            tau = self.log_tau.exp(); tau = tau.clamp(min=TAU_FLOOR) if TAU_FLOOR > 0 else tau
            Tp = self.Tp(ph); Ip = self.Ip(patches); B, Pn, d = Tp.shape; C, J, _ = Ip.shape
            sim = (Tp.reshape(B * Pn, d) @ Ip.reshape(C * J, d).T).reshape(B, Pn, C, J).permute(0, 2, 1, 3)
            soft = tau * torch.logsumexp(sim / tau, dim=3) - tau * float(np.log(J))
            S = S + self.log_lam.exp() * torch.einsum("bcp,bp->bc", soft, self.phrase_weights(Tp, phm))
        return S


VARIANT_CFG = {"pooled": dict(use_pool=True, use_set=False), "set": dict(use_pool=True, use_set=True), "phrases_only": dict(use_pool=False, use_set=True), "patches_only": dict(use_pool=False, use_set=True),
               "set4": dict(use_pool=True, use_set=True, separate=True), "setw": dict(use_pool=True, use_set=True, gate=True), "set4w": dict(use_pool=True, use_set=True, separate=True, gate=True),
               "phrases_only_w": dict(use_pool=False, use_set=True, gate=True), "set_tmap": dict(use_pool=True, use_set=True, train_i=False), "set_imap": dict(use_pool=True, use_set=True, train_t=False),
               "pooled_tmap": dict(use_pool=True, use_set=False, train_i=False), "pooled_imap": dict(use_pool=True, use_set=False, train_t=False)}
SEEDED = ("pooled", "set", "set4", "setw", "set4w")


def ranks_from_scores(S):
    true = np.diag(S); return (S >= true[:, None]).sum(1)          # ties = geq (pessimistic), as in the memos


def rec(r, boot=1000, seed=0):
    out = {f"R@{k}": float((r <= k).mean()) for k in (1, 5, 10, 50)}; out["median_rank"] = float(np.median(r))
    idx = np.random.RandomState(seed).randint(0, len(r), (boot, len(r))); b = (r[idx] <= 10).mean(1); out["R@10_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out["ties"] = "geq"; return out


@torch.no_grad()
def evaluate(m, sub, P, emb, offsets, split, pmax=32, qchunk=64, gchunk=256):
    idx = np.where(sub["split"] == split)[0]
    if split == "val" and len(idx) > VAL_MAX: idx = idx[np.random.RandomState(0).permutation(len(idx))[:VAL_MAX]]
    n = len(idx); q = torch.tensor(sub["txt"][idx], device=DEV); g = torch.tensor(sub["img"][idx], device=DEV)
    ph, phm = pad_phrases(emb, offsets, idx, pmax); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
    S = np.zeros((n, n), np.float32)
    for gs in range(0, n, gchunk):
        gi = idx[gs:gs + gchunk]; patches = torch.tensor(np.asarray(P[gi]).astype(np.float32), device=DEV)
        for qs in range(0, n, qchunk):
            S[qs:qs + qchunk, gs:gs + gchunk] = m.score(q[qs:qs + qchunk], ph[qs:qs + qchunk], phm[qs:qs + qchunk], g[gs:gs + gchunk], patches).cpu().numpy()
    return S


@torch.no_grad()
def evaluate_variant(m, sub, P, emb, offsets, split, variant, pmax=32):
    if variant not in ("phrases_only", "phrases_only_w", "patches_only"): return evaluate(m, sub, P, emb, offsets, split, pmax)
    idx = np.where(sub["split"] == split)[0]
    if split == "val" and len(idx) > VAL_MAX: idx = idx[np.random.RandomState(0).permutation(len(idx))[:VAL_MAX]]
    n = len(idx); q = torch.tensor(sub["txt"][idx], device=DEV); g = torch.tensor(sub["img"][idx], device=DEV)
    if variant in ("phrases_only", "phrases_only_w"):
        ph, phm = pad_phrases(emb, offsets, idx, pmax); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
        return m.score(q, ph, phm, g, g[:, None, :]).cpu().numpy()
    S = np.zeros((n, n), np.float32); phm = torch.ones(n, 1, dtype=torch.bool, device=DEV)
    for gs in range(0, n, 256):
        gi = idx[gs:gs + 256]; patches = torch.tensor(np.asarray(P[gi]).astype(np.float32), device=DEV)
        for qs in range(0, n, 64): S[qs:qs + 64, gs:gs + 256] = m.score(q[qs:qs + 64], q[qs:qs + 64, None, :], phm[qs:qs + 64], g[gs:gs + 256], patches).cpu().numpy()
    return S


def train(sub, P, emb, offsets, variant, seed=0, epochs=30, bs=128, lr=1e-4, wd=1e-4, temp=0.05, pmax=32, init=None, verbose=False):
    torch.manual_seed(seed); rng = np.random.RandomState(seed); tr = np.where(sub["split"] == "train")[0]; d = sub["txt"].shape[1]
    m = SetMaps(d, **VARIANT_CFG[variant]).to(DEV)
    if init is not None: m.load_state_dict(init, strict=False)
    maps = [p for n_, p in m.named_parameters() if p.requires_grad and n_.startswith("W")]; scal = [m.log_lam, m.log_tau]; gate = [p for n_, p in m.named_parameters() if n_.startswith("gate")]
    groups = [{"params": maps, "lr": lr, "weight_decay": wd}, {"params": scal, "lr": 1e-2, "weight_decay": 0.0}] + ([{"params": gate, "lr": 1e-3, "weight_decay": 0.0}] if gate else [])
    opt = torch.optim.AdamW(groups); best, best_state = -1, None; t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(tr); m.train()
        for s in range(0, len(perm), bs):
            bi = np.sort(perm[s:s + bs])
            if len(bi) < 8: continue
            q = torch.tensor(sub["txt"][bi], device=DEV); g = torch.tensor(sub["img"][bi], device=DEV)
            ph, phm = pad_phrases(emb, offsets, bi, pmax); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
            if variant == "patches_only": ph, phm = q[:, None, :], torch.ones(len(bi), 1, dtype=torch.bool, device=DEV)
            patches = torch.tensor(np.asarray(P[bi]).astype(np.float32), device=DEV)
            if variant in ("phrases_only", "phrases_only_w"): patches = g[:, None, :]
            S = m.score(q, ph, phm, g, patches) / temp; y = torch.arange(len(bi), device=DEV)
            loss = 0.5 * (torch.nn.functional.cross_entropy(S, y) + torch.nn.functional.cross_entropy(S.T, y)); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); r10 = float((ranks_from_scores(evaluate_variant(m, sub, P, emb, offsets, "val", variant, pmax)) <= 10).mean())
        if verbose: print(f"  {variant} s{seed} ep{ep:02d} loss {loss.item():.3f} val R@10 {r10:.4f} lam {m.log_lam.exp().item():.3f} tau {m.log_tau.exp().item():.4f} [{time.time()-t0:.0f}s]", flush=True)
        if r10 > best: best, best_state = r10, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval(); return m, best


@torch.no_grad()
def diagnostics(m, sub, emb, offsets, texts=None, pmax=32):
    out = {}
    if m.separate:
        Wt, Wtp, Wi, Wip = m.Wt.weight, m.Wtp.weight, m.Wi.weight, m.Wip.weight
        out["text_map_vs_phrase_map_relF"] = float(((Wt - Wtp).norm() / Wt.norm().clamp(min=1e-9)).cpu()); out["image_map_vs_patch_map_relF"] = float(((Wi - Wip).norm() / Wi.norm().clamp(min=1e-9)).cpu())
    if m.use_gate:
        te = np.where(sub["split"] == "test")[0]; ph, phm = pad_phrases(emb, offsets, te, pmax); ph, phm = torch.tensor(ph, device=DEV), torch.tensor(phm, device=DEV)
        Tp = m.Tp(ph); w = m.phrase_weights(Tp, phm); n = phm.sum(1).float()
        out["gate"] = {"mean_max_weight": float(w.max(1).values.mean()), "mean_uniform_weight": float((1 / n).mean()), "effective_phrases": float((1 / (w ** 2).sum(1)).mean()), "mean_phrases": float(n.mean())}
        if texts is not None:
            logits = m.gate(Tp).squeeze(-1).cpu().numpy(); agg = {}
            for b, i in enumerate(te):
                s, e = offsets[i], min(offsets[i + 1], offsets[i] + ph.shape[1])
                for k in range(e - s): agg.setdefault(str(texts[s + k]).lower(), []).append(float(logits[b, k]))
            rows = sorted(((np.mean(v), len(v), t) for t, v in agg.items() if len(v) >= 5), key=lambda x: x[0])
            out["gate"]["lowest"] = [(t, round(float(l), 2), c) for l, c, t in rows[:15]]; out["gate"]["highest"] = [(t, round(float(l), 2), c) for l, c, t in rows[-15:][::-1]]
    return out


def zeroshot_set(sub, P, emb, offsets, pmax=32, lams=(0.25, 0.5, 1.0, 2.0, 4.0), taus=(0.01, 0.02, 0.05, 0.1)):
    d = sub["txt"].shape[1]; best = None
    for lam in lams:
        for tau in taus:
            m = SetMaps(d, lam=lam, tau=tau, train_maps=False).to(DEV).eval(); r = float((ranks_from_scores(evaluate(m, sub, P, emb, offsets, "val", pmax)) <= 10).mean())
            if best is None or r > best[0]: best = (r, lam, tau)
    _, lam, tau = best; return SetMaps(d, lam=lam, tau=tau, train_maps=False).to(DEV).eval(), {"lam": lam, "tau": tau, "val_R@10": best[0]}


def saved_grid(collection: str, encoder: str, variants, tag: str = "") -> int | None:
    """The patch grid the first available saved model of these variants was trained at (None if no model is saved)."""
    for v in variants:
        p = models_dir(collection, encoder) / f"{v}{tag}.pt"
        if p.exists():
            g = torch.load(p, map_location="cpu").get("grid"); return int(g) if g else None
    return None


def load_model(collection: str, encoder: str, variant: str, tag: str = "", d: int | None = None):
    """The saved seed-0 model of a variant (results/<c>/<e>/models/<variant><tag>.pt)."""
    ck = torch.load(models_dir(collection, encoder) / f"{variant}{tag}.pt", map_location=DEV)
    d = d or ck["state"]["Wt.weight"].shape[0]; m = SetMaps(d, **VARIANT_CFG[variant]).to(DEV); m.load_state_dict(ck["state"]); return m.eval()


def train_sets(collection: str, encoder: str, grid=None, cap=32, verbs=False, variants=("pooled", "set", "setw"), seeds=2, epochs=30, tau_floor=0.01, tag="", save=False, verbose=False) -> Path:
    global TAU_FLOOR
    TAU_FLOOR = tau_floor; key = encoder_key(encoder); pmax = cap
    sub, P, emb, offsets, texts = load_all(collection, key, grid, cap, verbs); d = sub["txt"].shape[1]; G = int(round(np.sqrt(P.shape[1])))
    out = result_path(collection, key, f"sets{tag}.json"); res = json.load(open(out)) if out.exists() else {}
    res |= {"collection": collection, "encoder": key, "grid": G, "pmax": pmax, "verbs": verbs, "tau_floor": tau_floor, "epochs": epochs, "n_train": int((sub["split"] == "train").sum()), "n_test": int((sub["split"] == "test").sum()), "device": DEV, "ties": "geq"}
    if "frozen" not in res:
        m0 = SetMaps(d, train_maps=False, use_set=False).to(DEV).eval(); res["frozen"] = rec(ranks_from_scores(evaluate(m0, sub, P, emb, offsets, "test", pmax))); print(collection, key, "frozen", res["frozen"], flush=True)
    if "zeroshot_set" not in res and "set" in variants:
        mz, prm = zeroshot_set(sub, P, emb, offsets, pmax); res["zeroshot_set"] = rec(ranks_from_scores(evaluate(mz, sub, P, emb, offsets, "test", pmax))) | prm; print(collection, key, "zeroshot_set", res["zeroshot_set"], flush=True)
    json.dump(res, open(out, "w"), indent=1)
    for variant in variants:
        if variant in res: print("have", variant); continue
        runs = []; t0 = time.time(); diag = None
        for s in range(seeds if variant in SEEDED else 1):
            m, bv = train(sub, P, emb, offsets, variant, seed=s, epochs=epochs, pmax=pmax, verbose=verbose); r = rec(ranks_from_scores(evaluate_variant(m, sub, P, emb, offsets, "test", variant, pmax))); r["val_R@10"] = bv
            r["lam"], r["tau"] = float(m.log_lam.exp().detach()), float(m.log_tau.exp().detach()); runs.append(r)
            if s == 0 and (m.separate or m.use_gate): diag = diagnostics(m, sub, emb, offsets, texts, pmax)
            if s == 0 and save: torch.save({"state": m.state_dict(), "variant": variant, "grid": G, "pmax": pmax, "verbs": verbs, "encoder": key, "tau_floor": tau_floor}, models_dir(collection, key) / f"{variant}{tag}.pt")
            if variant == "pooled" and s == 0:
                best = None
                for lam in (0.25, 0.5, 1.0, 2.0):
                    for tau in (0.01, 0.02, 0.05, 0.1):
                        m2 = SetMaps(d, lam=lam, tau=tau).to(DEV).eval(); m2.load_state_dict({k: v for k, v in m.state_dict().items() if k.startswith("W")}, strict=False)
                        rv = float((ranks_from_scores(evaluate(m2, sub, P, emb, offsets, "val", pmax)) <= 10).mean())
                        if best is None or rv > best[0]: best = (rv, lam, tau, m2)
                res["pooled_maps_set_score"] = rec(ranks_from_scores(evaluate(best[3], sub, P, emb, offsets, "test", pmax))) | {"lam": best[1], "tau": best[2]}
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k not in ("R@10_ci", "ties")}; agg["R@10_ci"] = runs[0]["R@10_ci"]; agg["ties"] = "geq"
        agg["R@10_seed_sd"] = float(np.std([r["R@10"] for r in runs])); agg["seeds"] = len(runs); agg["seconds"] = time.time() - t0
        if diag: agg["diagnostics"] = diag
        res[variant] = agg; print(collection, key, variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items() if k != "diagnostics"}, flush=True)
        json.dump(res, open(out, "w"), indent=1)
    print("done ->", out); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--encoder", required=True)
    ap.add_argument("--grid", type=int, default=None, help="pool the patch store further to this grid (default: as stored)"); ap.add_argument("--cap", type=int, default=32); ap.add_argument("--verbs", action="store_true")
    ap.add_argument("--variants", default="pooled,set,setw"); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tau-floor", type=float, default=0.01, help="floor on the learned set temperature (paper: 0.01; 0 = none, as in the base-encoder memos)")
    ap.add_argument("--tag", default=""); ap.add_argument("--save", action="store_true"); ap.add_argument("--verbose", action="store_true"); a = ap.parse_args(argv)
    train_sets(a.collection, a.encoder, a.grid, a.cap, a.verbs, a.variants.split(","), a.seeds, a.epochs, a.tau_floor, a.tag, a.save, a.verbose)


if __name__ == "__main__":
    main()
