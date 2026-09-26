"""Check that the store reproduces the paper: raw test R@10/R@50 recomputed from every cell against the committed
results/<collection>/<encoder>/linear_arms.json (the run behind the tables), to 4 decimals. --master compares against the
iclr2027 master_arms.csv instead (internal). A missing cell is fetched from the released cache first (ltg.cache.ensure_cell). Also recomputes N50 (paper
protocol) and writes results/verification_cells.csv.

usage:  python scripts/verify_cells.py --master ../iclr2027/results/master_arms.csv [--cells skincap:siglip2-so400m-16-384 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ltg.cache import cache_root, list_cells, load_cell, results_root  # noqa: E402
from ltg.eval.metrics import near_miss_mass, recall_table  # noqa: E402

ENC = json.load(open(Path(__file__).resolve().parents[1] / "configs" / "encoders.json", encoding="utf-8"))
KEY_TO_MTAG = {k: v["legacy_mtag"] for k, v in ENC.items() if isinstance(v, dict) and v.get("legacy_mtag")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, help="internal: iclr2027/results/master_arms.csv; default = committed linear_arms.json")
    ap.add_argument("--cells", nargs="*", help="collection:encoder")
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--n50", action="store_true", help="also compute N50 (slower on big train splits)")
    a = ap.parse_args()
    if a.master:
        master = {(r["dataset"], r["mtag"]): r for r in csv.DictReader(open(a.master, encoding="utf-8"))}
    else:
        master = {}
        for p in sorted(results_root().glob("*/*/linear_arms.json")):
            r = json.load(open(p, encoding="utf-8"))
            master[(r["collection"], r["encoder"])] = {"raw_R10": r["raw"]["R@10"], "raw_R50": r["raw"]["R@50"], "n_train": r["n_train"]}
    cells = [tuple(c.split(":")) for c in a.cells] if a.cells else (list_cells() or [(c, e) for (c, e) in master])
    if not a.cells and not list_cells():
        print(f"[verify] no local cells under {cache_root()}: fetching the {len(cells)} released cells first")
    rows, bad = [], 0
    for coll, enc in cells:
        c = load_cell(coll, enc)
        q, g = c.arrays("test")
        r = recall_table(q, g)
        mt = KEY_TO_MTAG.get(enc, enc) if a.master else enc
        m = master.get((coll, mt))
        row = {"collection": coll, "encoder": enc, "n_train": c.sizes()["train"], "n_test": len(q), "dim": c.dim,
               "R@10": round(r["R@10"], 4), "R@50": round(r["R@50"], 4)}
        if m:
            d10, d50 = abs(r["R@10"] - float(m["raw_R10"])), abs(r["R@50"] - float(m["raw_R50"]))
            row.update({"master_R@10": round(float(m["raw_R10"]), 4), "master_R@50": round(float(m["raw_R50"]), 4),
                        "master_n_train": int(m["n_train"]), "ok": d10 <= a.tol and d50 <= a.tol})
            if int(m["n_train"]) != row["n_train"]:
                row["note"] = f"grid trained with --limit-train {m['n_train']} (store holds the full train split)"
            bad += not row["ok"]
        else:
            row.update({"master_R@10": "", "master_R@50": "", "master_n_train": "", "ok": ""})
        row.setdefault("note", "")
        if a.n50:
            tr_q, tr_g = c.arrays("train")
            row["N50"] = round(near_miss_mass(tr_q, tr_g, k=50, band=3, n=min(len(tr_q), len(q))), 4)
        rows.append(row)
        flag = "" if row["ok"] == "" else ("ok" if row["ok"] else "MISMATCH")
        print(f"[verify] {coll:12s} {enc:24s} n_train={row['n_train']:6d} test={len(q):5d} raw R@10 {r['R@10']:.4f} R@50 {r['R@50']:.4f}"
              + (f"  master {row['master_R@10']:.4f} {row['master_R@50']:.4f}" if m else "  (not in master)") + f"  {flag}")
    out = results_root() / "verification_cells.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[verify] {len(rows)} cells, {bad} mismatches -> {out}   (cache: {cache_root()})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
