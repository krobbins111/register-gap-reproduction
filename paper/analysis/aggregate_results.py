"""Aggregate all symmetric-ablation + baseline reports into the paper's master
tables. Scans BOTH the clean results folder and the legacy cache folder
(results take precedence when a cell exists in both), so the 14 cloud-run
cells and the wave-3 GPU cells land in one table.

Usage:  python analysis/aggregate_results.py
Output: results/master_arms.csv, results/master_baselines.csv,
        results/master_tables.md
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_ROOT, RESULTS_ROOT  # noqa: E402

ARMS = ("text", "image", "both", "shared")
BL_KEYS = ("csls_k10", "qbnorm_b20", "centerboth", "gapclose_sym_l0.5",
           "meanshift")


def collect(pattern: str) -> dict:
    found = {}
    for root in (CACHE_ROOT, RESULTS_ROOT):        # results overwrite cache
        for p in glob.glob(str(root / "*" / pattern)):
            r = json.load(open(p))
            found[(r["dataset"], r.get("mtag", "default"))] = r
    return found


def main() -> None:
    arms = collect("symmetric_ablation_*.json")
    bls = collect("retrieval_baselines_*.json")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_ROOT / "master_arms.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["dataset", "mtag", "n_train", "raw_R10", "raw_R50",
               "txt_nd", "img_nd"]
        for a in ARMS:
            hdr += [f"{a}_R10", f"{a}_R50", f"{a}_gain50", f"{a}_ddprime"]
        w.writerow(hdr)
        for (ds, mt), r in sorted(arms.items()):
            g = r.get("train_geometry", {})
            row = [ds, mt, r.get("n_train"), r["raw"]["R@10"], r["raw"]["R@50"],
                   g.get("txt_near_dup@0.9"), g.get("img_near_dup@0.9")]
            for a in ARMS:
                v = r["arms"].get(a, {})
                row += [v.get("R@10"), v.get("R@50"), v.get("gain@50"),
                        v.get("delta_dprime")]
            w.writerow(row)

    with open(RESULTS_ROOT / "master_baselines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "mtag", "raw_R50"] +
                   [f"{k}_gain50" for k in BL_KEYS])
        for (ds, mt), r in sorted(bls.items()):
            b = r["baselines"]
            w.writerow([ds, mt, r["raw"]["R@50"]] +
                       [b.get(k, {}).get("gain@50") for k in BL_KEYS])

    lines = ["# Master tables", "",
             f"arm cells: {len(arms)} · baseline cells: {len(bls)}", "",
             "| dataset | encoder | raw R@50 | both Δ50 | shared Δ50 | text Δ50 | image Δ50 |",
             "|---|---|--:|--:|--:|--:|--:|"]
    for (ds, mt), r in sorted(arms.items(),
                              key=lambda kv: -(kv[1]["arms"].get("both", {})
                                               .get("gain@50") or -9)):
        a = r["arms"]
        def g(k):
            v = a.get(k, {}).get("gain@50")
            return f"{v:+.3f}" if v is not None else "—"
        lines.append(f"| {ds} | {mt} | {r['raw']['R@50']:.3f} | {g('both')} | "
                     f"{g('shared')} | {g('text')} | {g('image')} |")
    (RESULTS_ROOT / "master_tables.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    print(f"[agg] {len(arms)} arm cells, {len(bls)} baseline cells -> "
          f"{RESULTS_ROOT}/master_arms.csv, master_baselines.csv, "
          f"master_tables.md")


if __name__ == "__main__":
    main()
