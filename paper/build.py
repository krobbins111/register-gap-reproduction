"""Regenerate every table and every no-cache figure of the submission from the committed result files.

    python paper/build.py            # ~1 min, needs numpy, scipy, matplotlib; writes paper/appendix/*.tex and paper/figs/*.pdf

Order matters: the appendix tables come first (make_paper_tables), the main-text tables are assembled from them
(make_main_tables), and two main-text figures are drawn from the typeset full-grid table (heatmap, near-miss law).
Figures that need embeddings or images (Fig. 1, showcase, qualitative panels, transfer) are listed in README.md
with their own commands.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ["analysis/make_paper_tables.py"],                       # appendix/tab_*.tex (13 tables)
    ["analysis/make_main_tables.py"],                        # appendix/tab_grid_main.tex, tab_alternatives.tex
    ["analysis/make_casestudy_tables.py"],                   # appendix/tab_sets, tab_grounding, tab_protocols, tab_pointing, tab_gemini (Tables 3, 6, 7, 11, Fig. 3)
    ["analysis/make_paper_figs.py"],                         # figs/{grid_never_hurts,spread_compression,which_side,spectrum,dose_response,lora,baselines,adapters,...}
    ["analysis/make_fig_heatmap_r50.py"],                    # figs/grid_heatmap_r50.pdf
    ["analysis/make_fig_nearmiss_law.py"],                   # figs/nearmiss_law_shapes.pdf
    ["analysis/make_concept.py", "--variant", "collapse", "--clean", "--no-gif"],   # figs/concept_maps_collapse_clean.pdf
]
for s in STEPS:
    print(f"\n$ python {' '.join(s)}", flush=True)
    r = subprocess.run([sys.executable, *s], cwd=HERE)
    if r.returncode:
        sys.exit(f"[build] step failed: {s}")
print("\n[build] done -> paper/appendix, paper/figs")
