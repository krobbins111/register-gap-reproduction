"""Reproduce the paper from the released embedding cache in one command.

    python scripts/reproduce.py                 # fetch the 58 cells (12.8 GB, resumable), verify raw recall to 4 dp, timing table
    python scripts/reproduce.py --train         # + retrain the two-map and tied-map arms of every cell on your GPU (~30 min on an RTX 3080)
                                                #   and compare with the committed results within --tol (default 0.01)
    python scripts/reproduce.py --cells skincap:siglip2-so400m-16-384 rsicd:siglip2-so400m-16-384 --train    # a subset
    LTG_CACHE=/your/embeddings LTG_OFFLINE=1 python scripts/reproduce.py --train                             # your own cache, no download

Stages: 1 fetch (skipped when the cells are present and hash-correct)  2 verify_cells  3 [--train] ltg.maps.linear per cell
        4 [--train] compare with results/<c>/<e>/linear_arms.json  5 timing_table. Exit status is non-zero if any check fails.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltg.cache import cache_root, ensure_cell, manifest, offline, results_root, result_path, sha256_of

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--cells", nargs="*", help="collection:encoder (default: every released cell)")
ap.add_argument("--train", action="store_true"); ap.add_argument("--tol", type=float, default=0.01)
ap.add_argument("--seeds", nargs="+", default=["0", "1"]); ap.add_argument("--no-baselines", action="store_true")
a = ap.parse_args()
py = sys.executable
man = manifest()
cells = [tuple(c.split(":")) for c in a.cells] if a.cells else sorted({(r["collection"], r["encoder"]) for r in man.values()})

print(f"[repro] cache {cache_root()}  offline={offline()}  cells={len(cells)}")
# 1. fetch / check
for c, e in cells:
    rel = f"{c}/{e}.npz"; p = cache_root() / rel
    if p.exists() and rel in man and sha256_of(p) == man[rel]["sha256"]:
        continue
    if offline():
        if not p.exists():
            sys.exit(f"[repro] {rel} missing and LTG_OFFLINE is set")
        print(f"[repro] {rel}: present, hash not checked against the release (own cache)")
        continue
    ensure_cell(c, e)
# 2. verify raw numbers
r = subprocess.run([py, str(REPO / "scripts" / "verify_cells.py")] + (["--cells"] + [f"{c}:{e}" for c, e in cells] if a.cells else []))
if r.returncode:
    sys.exit("[repro] raw-recall verification failed")
# 3-4. retrain + compare
bad = 0
if a.train:
    for c, e in cells:
        ref_p = result_path(c, e, "linear_arms.json")
        ref = json.load(open(ref_p, encoding="utf-8")) if ref_p.exists() else None
        # never overwrite the committed report: write the rerun next to it
        env = dict(__import__("os").environ, LTG_RESULTS=str(results_root() / "_reproduce"))
        cmd = [py, "-m", "ltg.maps.linear", "--collection", c, "--encoder", e, "--arms", "both", "shared", "--seeds", *a.seeds] + ([] if a.no_baselines else ["--baselines"])
        print("[repro]", " ".join(cmd[2:]))
        if subprocess.run(cmd, cwd=REPO, env=env).returncode:
            bad += 1; print(f"[repro] {c} x {e}: training failed"); continue
        new = json.load(open(results_root() / "_reproduce" / c / e / "linear_arms.json", encoding="utf-8"))
        if not ref:
            print(f"[repro] {c} x {e}: no committed report to compare with"); continue
        for arm in ("both", "shared"):
            if arm in ref["arms"] and arm in new["arms"]:
                d10 = abs(new["arms"][arm]["R@10"] - ref["arms"][arm]["R@10"]); d50 = abs(new["arms"][arm]["R@50"] - ref["arms"][arm]["R@50"])
                ok = d10 <= a.tol and d50 <= a.tol; bad += not ok
                print(f"[repro] {c:12s} {e:24s} {arm:6s} R@10 {new['arms'][arm]['R@10']:.4f} vs {ref['arms'][arm]['R@10']:.4f}  R@50 {new['arms'][arm]['R@50']:.4f} vs {ref['arms'][arm]['R@50']:.4f}  {'ok' if ok else 'DIFF'}")
    subprocess.run([py, str(REPO / "scripts" / "timing_table.py")], env=dict(__import__("os").environ, LTG_RESULTS=str(results_root() / "_reproduce")))
print(f"[repro] done: {'all checks passed' if not bad else f'{bad} problems'}")
sys.exit(1 if bad else 0)
