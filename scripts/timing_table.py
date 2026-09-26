"""Per-cell wall-clock of training the linear maps, from results/<collection>/<encoder>/linear_arms.json (hardware block +
train_seconds_per_seed). Writes results/timing_cells.csv and prints the summary the paper's setup section needs.

usage:  python scripts/timing_table.py [--arm both]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltg.cache import results_root

ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="both"); ap.add_argument("--report", default="linear_arms.json"); ap.add_argument("--tex", help="also write a per-collection LaTeX table here")
a = ap.parse_args()
rows = []
for p in sorted(results_root().glob(f"*/*/{a.report}")):
    r = json.load(open(p, encoding="utf-8")); arm = r.get("arms", {}).get(a.arm)
    if not arm or "train_seconds_per_seed" not in arm:
        continue
    hw = r.get("hardware", {})
    rows.append({"collection": r["collection"], "encoder": r["encoder"], "n_train": r["n_train"], "dim": r.get("dim") or len(arm.get("map_svd_text", {}).get("singular_values", [])) or "",
                 "epochs": r["epochs"], "batch": r["batch"], "seeds": len(r["seeds"]),
                 "seconds_per_seed_mean": arm["train_seconds_mean"], "seconds_per_seed": " ".join(str(s) for s in arm["train_seconds_per_seed"]),
                 "seconds_shared": r["arms"].get("shared", {}).get("train_seconds_mean", ""),
                 "peak_gpu_gb": arm.get("peak_gpu_gb", ""), "wall_seconds_total": hw.get("wall_seconds_total", ""), "gpu": hw.get("gpu", ""), "torch": hw.get("torch", "")})
if not rows:
    sys.exit("no timed reports yet: run jobs/run_linear_grid.bat (or one ltg.maps.linear cell) first")
out = results_root() / "timing_cells.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
s = np.array([r["seconds_per_seed_mean"] for r in rows]); n = np.array([r["n_train"] for r in rows], float)
print(f"[timing] {len(rows)} cells on {rows[0]['gpu']} (torch {rows[0]['torch']}), arm={a.arm}, {rows[0]['epochs']} epochs, batch {rows[0]['batch']}")
print(f"[timing] seconds per training run: median {np.median(s):.1f}  min {s.min():.1f}  max {s.max():.1f}  mean {s.mean():.1f}")
print(f"[timing] per 1,000 training pairs: median {np.median(s / n * 1000):.2f} s  (min {(s/n*1000).min():.2f}, max {(s/n*1000).max():.2f})")
for r in sorted(rows, key=lambda r: r["n_train"]):
    print(f"   {r['collection']:12s} {r['encoder']:24s} n={r['n_train']:6d} d={r['dim']!s:>5}  {r['seconds_per_seed_mean']:7.1f} s/seed   peak {r['peak_gpu_gb']} GB")
print(f"[timing] -> {out}")
if a.tex:
    NAME = {"nwpu": "NWPU", "rocov2": "ROCOv2", "skincap": "SkinCAP", "treeoflife": "TreeOfLife", "rsicd": "RSICD", "scimmir": "SciMMIR",
            "fashion200k": "Fashion200k", "semart": "SemArt", "coco": "COCO", "goodnews": "GoodNews", "facad": "FACAD"}
    by = {}
    for r in rows:
        by.setdefault(r["collection"], []).append(r)
    L = ["\\begin{table}[h]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         f"\\caption{{Training cost of the two-map arm per collection: seconds for one training run (30 epochs, per-epoch validation, embeddings resident on the GPU; test evaluation excluded), mean and range over the collection's encoders, and peak GPU memory. {rows[0]['gpu']}, PyTorch {rows[0]['torch']}, batch {rows[0]['batch']}. Every cell: \\texttt{{results/timing\\_cells.csv}}.}}",
         "\\label{tab:timing}", "\\begin{tabular}{lrrrrr}", "\\toprule", "Collection & train pairs & encoders & s / run & range & peak GB \\\\", "\\midrule"]
    for c, rs in sorted(by.items(), key=lambda kv: int(kv[1][0]["n_train"])):
        ss = [float(r["seconds_per_seed_mean"]) for r in rs]; pk = max(float(r["peak_gpu_gb"] or 0) for r in rs)
        L.append(f"{NAME.get(c, c)} & {int(rs[0]['n_train']):,} & {len(rs)} & {np.mean(ss):.1f} & {min(ss):.1f}--{max(ss):.1f} & {pk:.2f} \\\\")
    tot = sum(2 * float(r["seconds_per_seed_mean"]) + 2 * float(r["seconds_shared"] or 0) for r in rows) / 60
    L += ["\\midrule", f"all 58 cells & & & median {np.median(s):.1f}, max {s.max():.1f} & {np.median(s / n * 1000):.2f} s per 1,000 pairs & {tot:.0f} min for the grid \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open(a.tex, "w", encoding="utf-8").write("\n".join(L)); print(f"[timing] -> {a.tex}")
