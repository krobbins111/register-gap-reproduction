"""Tables of the phrase/patch, grounding, published-protocol and API sections (paper Tables 3, 6, 7, 11 and the table half of
Figure 3), regenerated from the store-layout result JSONs (results/<collection>/<encoder>/*.json, written by ltg.maps.sets,
ltg.eval.grounding, ltg.eval.protocols, ltg.encoders.api). Reference rows (published methods) come from configs/published.json.

    python paper/analysis/make_casestudy_tables.py [--out paper/appendix] [--print]

Writes  tab_sets.tex        Table 3   SigLIP2 so400m/16-384: frozen | set score, no maps | pooled two maps | two maps over sets | + gate
        tab_grounding.tex   Figure 3  TIPSv2-B/14: CityFlow-NL box hit (random split, new cameras) and Flickr30k pointing accuracy
        tab_protocols.tex   Table 6   published protocols (Derm1M / RSITR / Text2Art) on so400m
        tab_pointing.tex    Table 7   Flickr30k Entities pointing game against the published weakly supervised methods
        tab_gemini.tex      Table 11  Gemini Embedding 2: frozen | linear two maps | MLP h=1024 | MLP h=4096
Every number is the mean over the seeds present in the JSON (keys <variant>, <variant>_s1, ...), as in the paper.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PAPER = HERE.parents[1]
REPO = PAPER.parent
RES = REPO / "results"
PUB = json.load(open(REPO / "configs" / "published.json", encoding="utf-8"))
NAME = {"skincap": "SkinCAP", "rsicd": "RSICD", "semart": "SemArt", "scimmir": "SciMMIR", "rocov2": "ROCOv2", "fashion200k": "Fashion200k",
        "facad": "FACAD", "coco": "COCO (control)", "flickr": "Flickr30k (control)"}
SO400M, TIPS, GEM = "siglip2-so400m-16-384", "tipsv2-b14", "gemini-embedding-2"


def load(collection, encoder, name):
    """results/<c>/<e>/<name>.json, or the newest <name>_*.json import of it."""
    d = RES / collection / encoder
    p = d / f"{name}.json"
    if not p.exists():
        cands = sorted(glob.glob(str(d / f"{name}_*.json")))
        if name == "sets":                                   # sets_<import date>.json, not the sets_nomaps / _g16 / _p32 ... variants
            cands = [c for c in cands if not any(x in Path(c).name for x in ("nomaps", "multiquery", "taufloor", "_s3", "_save",
                                                                                "_repro", "_g4", "_g16", "_p24", "_p32", "_all"))]
        if not cands:
            return None
        p = Path(cands[-1])
    return json.load(open(p, encoding="utf-8"))


def seed_mean(r: dict, variant: str, key: str):
    vals = [r[k][key] for k in r if (k == variant or k.startswith(variant + "_s")) and isinstance(r[k], dict) and key in r[k]]
    return sum(vals) / len(vals) if vals else None


def f3(v):
    return "--" if v is None else f"{v:.3f}"


def table(out: Path, name: str, caption: str, label: str, header: str, rows: list[str], align: str, star=False):
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{align}}}", "\\toprule", header + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    (out / f"{name}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {name}.tex ({len(rows)} rows)")
    return L


# ── Table 3: sets on so400m ────────────────────────────────────────────────────────────────────────────────────────────
def tab_sets(out):
    rows = []
    for c in ("skincap", "rsicd", "semart"):
        r, nm = load(c, SO400M, "sets"), load(c, SO400M, "sets_nomaps")
        if r is None:
            print(f"  [sets] no results/{c}/{SO400M}/sets*.json — run ltg.maps.sets"); continue
        gal = r["n"]["test"]
        lines = [("frozen encoder", r, "frozen"), ("set score, no maps", nm, "nomaps_set"), ("pooled two maps", r, "pooled"),
                 ("two maps over sets", r, "set"), ("sets + phrase gate", r, "setw")]
        for i, (lab, src, var) in enumerate(lines):
            if src is None:
                continue
            first = f"{NAME[c]} ({gal:,} gallery)" if i == 0 else ""
            rows.append(f"{first} & {lab} & " + " & ".join(f3(seed_mean(src, var, k)) for k in ("R@1", "R@5", "R@10", "R@50")) + " \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    return table(out, "tab_sets", "Phrase- and patch-level scoring improves retrieval beyond pooled embeddings. Text-to-image recall using "
                 "SigLIP2 so400m/16-384, mean of two seeds. ``Set score, no maps'' uses the frozen encoder and selects only the global/local "
                 "weight and set-score temperature on validation data. ``Pooled two maps'' uses global embeddings only; the final two rows "
                 "train and score with phrase and patch features, with or without the learned phrase gate.",
                 "tab:sets", "collection & maps & R@1 & R@5 & R@10 & R@50", rows, "llrrrr")


# ── Figure 3 table + Table 7: grounding on TIPSv2 ─────────────────────────────────────────────────────────────────────
def tab_grounding(out):
    cf, cam, fl = load("cityflow", TIPS, "grounding"), load("cityflow_cam", TIPS, "grounding"), load("flickr", TIPS, "grounding")
    if cf is None or fl is None:
        print("  [grounding] missing results/cityflow|flickr/tipsv2-b14/grounding.json — run ltg.eval.grounding"); return
    pc = lambda v: "--" if v is None else f"{100 * v:.1f}"
    hit = lambda d, var: None if d is None or var not in d else d[var]["subject"]["hit"]
    rows = [f"chance (box area) & {pc(cf['frozen']['chance_box_area'])} & {pc(cam['frozen']['chance_box_area'] if cam else None)} & {pc(fl['chance_box_area'])} \\\\",
            f"center patch & -- & -- & {pc(fl['center_patch_acc'])} \\\\"]
    for lab, var in (("frozen encoder", "frozen"), ("pooled two maps", "pooled"), ("two maps over sets", "set"), ("sets + phrase gate", "setw")):
        rows.append(f"{lab} & {pc(hit(cf, var))} & {pc(hit(cam, var))} & {pc(fl[var]['pointing_acc'] if var in fl else None)} \\\\")
    L = table(out, "tab_grounding", "Grounding without box supervision (TIPSv2-B/14 maps trained from caption--image pairs only; boxes used for "
              "evaluation). CityFlow-NL subject-phrase localization on the random split and on cameras unseen during training, and "
              "pointing-game accuracy on Flickr30k Entities.", "tab:grounding",
              "method & CityFlow-NL random & CityFlow-NL new cameras & Flickr30k pointing (\\%)", rows, "lrrr")
    # Table 7: Flickr pointing against the published methods
    rows7 = [f"chance (box area) & {pc(fl['chance_box_area'])} \\\\", f"center patch & {pc(fl['center_patch_acc'])} \\\\"]
    for lab, var in (("frozen encoder", "frozen"), ("pooled two maps", "pooled"), ("two maps over sets", "set"), ("sets + phrase gate", "setw")):
        if var in fl:
            rows7.append(f"{lab} & {pc(fl[var]['pointing_acc'])} \\\\")
    pub = PUB.get("flickr", {}).get("rows", {})
    if pub:
        rows7.append("\\midrule")
        for k, v in pub.items():
            acc = v.get("pointing_acc")
            acc = max(acc) if isinstance(acc, list) else acc          # a range -> the best reported number
            rows7.append(f"{k} & {pc(acc)} \\\\")
    table(out, "tab_pointing", f"Pointing game on Flickr30k Entities (test split, {fl['n_phrases']:,} phrases on {fl['n_images']:,} images). A phrase "
          "counts as found when its best patch lies in the phrase's box. Published rows are weakly supervised methods at their best reported number.",
          "tab:pointing", "method & pointing accuracy (\\%)", rows7, "lr")


# ── Table 6: published protocols on so400m ────────────────────────────────────────────────────────────────────────────
def tab_protocols(out):
    rows = []
    sk = load("skincap", SO400M, "sets")
    if sk and "protocol" in sk:
        P = sk["protocol"]
        for lab, v in PUB["skincap"]["rows"].items():
            if "best" in lab:
                rows.append(f"SkinCAP, Derm1M & {lab.split(' (')[0]} (best published, zero-shot) & -- & -- & {f3(v.get('R@10'))} & {f3(v.get('R@50'))} \\\\")
        for lab, var in (("so400m, frozen encoder", "frozen"), ("so400m, pooled two maps", "pooled"), ("so400m, sets + phrase gate", "setw")):
            r = P[var]; rows.append(f" & {lab} & {f3(r['R@1'])} & {f3(r['R@5'])} & {f3(r['R@10'])} & {f3(r['R@50'])} \\\\")
        rows.append("\\midrule")
    rs = load("rsicd", SO400M, "sets")
    if rs and "protocol" in rs and "T2I R@1" in rs["protocol"]["frozen"]:
        P = rs["protocol"]
        for lab, v in PUB["rsicd"]["rows"].items():
            if "mR" in v:
                rows.append(f"RSICD, RSITR (T2I) & {lab} & {f3(v['T2I R@1'])} & {f3(v['T2I R@5'])} & {f3(v['T2I R@10'])} & mR {f3(v['mR'])} \\\\")
        for lab, var in (("so400m, frozen encoder", "frozen"), ("so400m, pooled two maps", "pooled"), ("so400m, sets + phrase gate", "setw")):
            r = P[var]; rows.append(f" & {lab} & {f3(r['T2I R@1'])} & {f3(r['T2I R@5'])} & {f3(r['T2I R@10'])} & mR {f3(r['mR'])} \\\\")
        rows.append("\\midrule")
    se = load("semart", SO400M, "sets")
    if se:
        for lab, v in PUB["semart"]["rows"].items():
            if "best row" in lab:
                rows.append(f"SemArt, Text2Art & ContextNet, best row (uses title and author) & {f3(v.get('R@1'))} & {f3(v.get('R@5'))} & {f3(v.get('R@10'))} & -- \\\\")
        for lab, var in (("so400m, frozen encoder", "frozen"), ("so400m, pooled two maps", "pooled"), ("so400m, sets + phrase gate", "setw")):
            rows.append(f" & {lab} & " + " & ".join(f3(seed_mean(se, var, k)) for k in ("R@1", "R@5", "R@10", "R@50")) + " \\\\")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    table(out, "tab_protocols", "Under the published protocols, on the default encoder. SkinCAP: the whole collection as gallery and the held-out "
          "test captions as queries (Derm1M). RSICD: all five captions of each test image as queries; mR is the mean of the six recalls. "
          "SemArt: the Text2Art test split with description-only queries.", "tab:protocols",
          "protocol & method & R@1 & R@5 & R@10 & R@50", rows, "llrrrr")


# ── Table 11: Gemini Embedding 2, linear vs MLP ───────────────────────────────────────────────────────────────────────
def tab_gemini(out):
    rows = []
    for c in ("skincap", "semart", "rsicd", "scimmir", "rocov2", "fashion200k", "facad", "coco", "flickr"):
        r = load(c, GEM, "mlp")
        if r is None:
            continue
        cell = lambda k: f"{r[k]['R@10']:.3f} / {r[k]['R@50']:.3f}"
        rows.append(f"{NAME[c]} & {r['n']['test']:,} & {cell('frozen')} & {cell('linear_two_maps')} & {cell('mlp_h1024')} & {cell('mlp_h4096')} \\\\")
    table(out, "tab_gemini", "Linear two maps against a residual MLP per side on Gemini Embedding 2 (3072-d; nine public collections). Same recipe "
          "for all (InfoNCE $\\tau$=0.05, AdamW $10^{-4}$, batch 256, 30 epochs, val-R@50 selection); mean of three seeds; hidden widths 1024 "
          "and 4096. R@10 / R@50 on each test gallery.", "tab:gemini",
          "collection & gallery & frozen & linear two maps & MLP $h$=1024 & MLP $h$=4096", rows, "lrcccc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PAPER / "appendix")
    ap.add_argument("--print", action="store_true", help="also print the .tex to stdout")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for fn in (tab_sets, tab_grounding, tab_protocols, tab_gemini):
        fn(a.out)
    if a.print:
        for n in ("tab_sets", "tab_grounding", "tab_protocols", "tab_pointing", "tab_gemini"):
            p = a.out / f"{n}.tex"
            if p.exists():
                print(f"\n% ---- {n}.tex\n" + p.read_text(encoding="utf-8"))
    print(f"[casestudy-tables] -> {a.out}")


if __name__ == "__main__":
    main()
