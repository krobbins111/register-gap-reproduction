# Learn the (Register) Gap — reproduction

Code and result files to reproduce the numbers in every table and figure of the ICLR 2027 submission
*Learn the (Register) Gap: Making Effective Specialist Text-to-Image Retrieval with Generalist Encoders*.

The pipeline has three stages. Every number in the paper is a function of stage 1's embedding caches and nothing else.

```
1. embeddings   one cache file per (collection, encoder) cell        scripts/fetch_cache.py  |  paper/legacy/embed_cells.py
2. experiments  each writes a result JSON per cell                     ltg/maps/*, paper/ablations/*, paper/analysis/*
3. tables       LaTeX tables and figures from the result JSONs        python paper/build.py
```

Stage 3 runs in about a minute on a CPU from the committed result files, so every table can be checked against the paper
before running anything else. Stage 2 recomputes those result files from the embeddings (the whole grid retrains in
~30 min on one RTX 3080; LoRA is the only multi-hour step). Stage 1 rebuilds the embeddings from the raw datasets.

```
python -m pip install -r requirements.txt
python paper/build.py                          # stage 3: paper/appendix/tab_*.tex, paper/figs/*.pdf from the committed results
python scripts/reproduce.py                    # stage 1: fetch the 58 caches (12.8 GB), verify their raw R@10/R@50 to 4 dp
python scripts/reproduce.py --train            # stage 2 for the grid: retrain every cell, compare with the paper within 0.01
```

## 1. Embeddings

A cell is `<cache>/<collection>/<encoder>.npz`: unit-norm caption and image embeddings, split labels, captions, pair ids
(`ltg/cache.py`). The cache lives outside the repository (`$LTG_CACHE`, else `local.json {"cache": ...}`, else
`../learn-the-gap-cache`). Collections and split rules: `configs/collections.json` (paper Appendix A, Table 4); encoders
and their HF ids: `configs/encoders.json`.

**Download.** The 58 grid cells are a Hugging Face dataset (`configs/cache_release.json`, sha256 per file in
`configs/cache_manifest.json`). Any experiment that needs a missing cell fetches it; `LTG_OFFLINE=1` turns that off.

```
python scripts/fetch_cache.py [--cells skincap:siglip2-so400m-16-384 ...]
python scripts/verify_cells.py                 # raw R@10 / R@50 of every cell == Table 15, 4 dp  (paper Appendix B)
python scripts/export_legacy_cache.py          # the flat {train,val,test}_embs[_<mtag>].npz layout that paper/ablations/* read
```

**Rebuild from raw data.** `paper/legacy/` is the pipeline the caches were built with: dataset loaders and split rules in
`paper/legacy/splits/`, encoders in `paper/legacy/backbones.py`. HF-hosted collections download themselves into
`$ICLR_DATA_DIR` (default `<cache>/raw`); SkinCAP, FACAD, GoodNews, NWPU and SemArt must be placed there first (layouts at
the top of `paper/legacy/splits/grid_local.py` and `splits/__init__.py`).

```
python paper/legacy/embed_cells.py --dataset skincap                       # SigLIP2-so400m/16-384 (the default encoder)
python paper/legacy/embed_cells.py --dataset skincap --encoder medsiglip   # --encoder <legacy_mtag> or --embed-model <HF id>
python scripts/import_legacy.py --legacy <cache>/legacy                    # into the store layout; then scripts/verify_cells.py
```

Patch stores and phrase sets (Table 3, Figure 3, Tables 6–7) sit next to the cell:
`python -m ltg.encoders.patches --collection skincap --encoder siglip2-so400m-16-384 --grid 12` and
`python -m ltg.encoders.phrases --collection skincap --encoder siglip2-so400m-16-384 --cap 32` (raw images and spaCy needed).
Gemini Embedding 2 cells (Table 11): `python -m ltg.encoders.api --collection skincap` with `GEMINI_API_KEY` set.

## 2. Experiments → result files

Legacy-layout scripts (`paper/ablations`, `paper/analysis`) run from `paper/` and write `paper/results/<dataset>/<experiment>_<mtag>.json`
(`mtag` = `legacy_mtag` in `configs/encoders.json`; `default` = SigLIP2). Store-layout modules (`ltg/...`) run from the repo root and
write `results/<collection>/<encoder>/<experiment>.json`. All result files are committed, so stage 3 works without this stage.

| experiment | command (one cell shown) | result file | feeds |
|---|---|---|---|
| two maps, tied, text-only, image-only; training-free baselines | `python -m ltg.maps.linear --collection nwpu --encoder siglip2-so400m-16-384 --seeds 0 1 --baselines` — or the original `python ablations/symmetric_projection.py --dataset nwpu --seeds 0 1 --baselines`; every cell: `jobs/run_linear_grid.bat` / `scripts/reproduce.py --train` | `results/<c>/<e>/linear_arms.json`, `baselines.json` — `paper/results/<ds>/symmetric_ablation_<mtag>.json`, `retrieval_baselines_<mtag>.json` | Tables 1, 2, 10, 12, 15, 16; Fig. 2a, 5, 9 |
| aggregate the grid | `python analysis/aggregate_results.py` | `paper/results/master_arms.csv`, `master_baselines.csv` | every table |
| near-miss mass N50, band sweep, val-split variant | `python analysis/apriori_nearmiss.py --all` | `paper/results/nearmiss_apriori.csv` | Table 1 (N50), Fig. 2b, Table 8 |
| d′ and catalog-R@10 predictors | `python analysis/apriori_dprime.py --all` | `paper/results/dprime_apriori_perquery.csv` | Table 8, Table 15 |
| rank transport of rescued pairs | `python analysis/rank_transport.py --dataset nwpu [--mtag georsclip-vitb32]` (the 11 cells: `paper/jobs/run_nearmiss.bat`) | `paper/results/<ds>/rank_transport_<mtag>.json` | Table 9 |
| CLIP-Adapter (text / image / both), Tip-Adapter(-F) | `python ablations/adapter_baselines.py --dataset skincap --seeds 0 1` (`paper/jobs/run_adapters.bat`) | `paper/results/<ds>/adapter_baselines_<mtag>.json` | Table 2, Table 13 |
| LoRA r=16 on CLIP ViT-L/14, maps on LoRA (GPU, raw images, `peft`) | `python ablations/lora_finetune.py --dataset nwpu [--limit-train 1000 --epochs 30]` (`paper/jobs/run_lora.bat`) | `paper/results/lora/<ds>/lora_clip-vitl14-laion2b_n<N>.json` | Table 2 (bottom), Table 14 |
| polar factorisation of the tied map; two-map vs tied edge | `python ablations/factorize_shared.py --dataset nwpu`; `python ablations/decompose_both.py --dataset nwpu` (`paper/jobs/run_decomp.bat`) | `paper/results/<ds>/shared_factorization_<mtag>.json`, `both_decomposition_<mtag>.json`, `shared_map_<mtag>.npz` | Table 5, Fig. 7 |
| training-set size sweep | `python ablations/dose_response.py --dataset skincap --seeds 0 1` (goodnews nwpu rocov2 skincap treeoflife) | `paper/results/<ds>/dose_response_both_<mtag>.json` | Fig. 6 |
| cross-collection transfer of the maps and of centering | `python ablations/export_maps.py --cells coco:default ... nwpu:default` then `python analysis/transfer_matrix.py` (`paper/jobs/run_transfer.bat`) | `paper/results/maps/<ds>/maps_default.npz` (not committed; seconds to rebuild), `paper/results/transfer_default.json` | Fig. 8 |
| two maps over phrase × patch sets, phrase gate, set score without maps | `python -m ltg.maps.sets --collection skincap --encoder siglip2-so400m-16-384 --variants frozen,zeroshot_set,pooled,set,setw --seeds 2 --save` | `results/<c>/<e>/sets.json` | Table 3 |
| published protocols (Derm1M, RSITR, Text2Art) | `python -m ltg.eval.protocols --collection skincap --encoder siglip2-so400m-16-384 --variants pooled,setw` | `results/<c>/<e>/protocol.json` | Table 6 |
| grounding: CityFlow-NL box hit, Flickr30k pointing game | `python -m ltg.eval.grounding --task box_hit --collection cityflow --encoder tipsv2-b14 --variants frozen,pooled,set,setw` (also `--collection cityflow_cam`); `--task pointing_flickr` | `results/cityflow[_cam]/tipsv2-b14/grounding.json`, `results/flickr/tipsv2-b14/grounding.json` | Fig. 3 (table), Table 7 |
| linear maps vs residual MLP on Gemini Embedding 2 | `python -m ltg.maps.linear --collection skincap --encoder gemini-embedding-2 --seeds 0 1 2`; `python -m ltg.maps.mlp --collection skincap --encoder gemini-embedding-2 --hidden 1024` (and `4096`) | `results/<c>/gemini-embedding-2/mlp.json` | Table 11 |
| residual MLP on SigLIP2 (§4.2 text) | `python -m ltg.maps.mlp --collection skincap --encoder siglip2-so400m-16-384` | `results/<c>/<e>/mlp_arms.json` | text |
| wall-clock per cell | `python scripts/reproduce.py --train` then `python scripts/timing_table.py --tex` | `results/timing_cells.csv` | §4.1 timing sentence |

Recipe for every map (paper §3.5): zero-initialised residual maps, in-batch InfoNCE τ=0.05, AdamW 1e-4 / wd 1e-4, batch 256,
30 epochs, epoch chosen on validation R@50, mean of two seeds. `ltg.maps.linear` is a verbatim port of
`symmetric_projection.py` reading the store; `scripts/reproduce.py --train` checks each arm against the committed JSON.

## 3. Tables and figures from the result files

`python paper/build.py` runs everything in this section (from `paper/`). Tables land in `paper/appendix/`, figures in `paper/figs/`.
The tables regenerate byte-identical to the submission's LaTeX; the figures carry the same numbers with the paper's styling.

| paper | script (from `paper/`) | output | reads |
|---|---|---|---|
| Table 1 grid summary, default encoder + best encoder | `analysis/make_main_tables.py` | `appendix/tab_grid_main.tex` | `tab_grid_full.tex` |
| Table 2 alternatives (baselines, adapters, LoRA on CLIP-L) | `analysis/make_main_tables.py` | `appendix/tab_alternatives.tex` | `tab_baselines/adapters/lora.tex` |
| Table 3 sets on so400m | `analysis/make_casestudy_tables.py` | `appendix/tab_sets.tex` | `results/<c>/siglip2-so400m-16-384/sets*.json` |
| Table 4 collections and split sizes | typeset by hand from `scripts/verify_cells.py` (n_train / n_test per cell → `results/verification_cells.csv`) | — | cells |
| Table 5 factor contributions | `analysis/make_paper_tables.py` | `appendix/tab_factor.tex` | `shared_factorization_*.json`, `both_decomposition_*.json` |
| Table 6 published protocols | `analysis/make_casestudy_tables.py` | `appendix/tab_protocols.tex` | `results/<c>/<e>/sets*.json` (`protocol` block) / `protocol.json`, `configs/published.json` |
| Table 7 Flickr30k pointing game | `analysis/make_casestudy_tables.py` | `appendix/tab_pointing.tex` | `results/flickr/tipsv2-b14/grounding.json`, `configs/published.json` |
| Table 8 pre-query predictors | `analysis/make_paper_tables.py` | `appendix/tab_band.tex` | `nearmiss_apriori.csv`, `dprime_apriori_perquery.csv`, `master_arms.csv` |
| Table 9 rank transport | `analysis/make_paper_tables.py` | `appendix/tab_transport.tex` | `rank_transport_*.json` |
| Table 10 encoder spread | `analysis/make_paper_tables.py` | `appendix/tab_compression.tex` | `master_arms.csv` |
| Table 11 Gemini linear vs MLP | `analysis/make_casestudy_tables.py` | `appendix/tab_gemini.tex` | `results/<c>/gemini-embedding-2/mlp.json` |
| Table 12 training-free baselines | `analysis/make_paper_tables.py` | `appendix/tab_baselines.tex` | `master_baselines.csv` |
| Table 13 adapters | `analysis/make_paper_tables.py` | `appendix/tab_adapters.tex` | `adapter_baselines_*.json` |
| Table 14 LoRA | `analysis/make_paper_tables.py` | `appendix/tab_lora.tex` | `lora/<ds>/lora_*.json` |
| Table 15 full grid | `analysis/make_paper_tables.py` | `appendix/tab_grid_full.tex` | `master_arms.csv`, `nearmiss_apriori.csv` |
| Table 16 all four arms | `analysis/make_paper_tables.py` | `appendix/tab_arms.tex` | `symmetric_ablation_*.json` |
| Figure 2a gain heatmap | `analysis/make_paper_figs.py` | `figs/grid_heatmap.pdf` | `master_arms.csv` |
| Figure 2b near-miss law | `analysis/make_paper_figs.py`; marker-by-category form `analysis/make_fig_nearmiss_law.py` | `figs/nearmiss_law.pdf`; `figs/nearmiss_law_shapes.pdf` | `master_arms.csv`, `nearmiss_apriori.csv` |
| Figure 3 grounding (table half) | `analysis/make_casestudy_tables.py` | `appendix/tab_grounding.tex` | `results/cityflow[_cam]/tipsv2-b14/grounding.json`, `results/flickr/tipsv2-b14/grounding.json` |
| Figure 5 encoder spread | `analysis/make_paper_figs.py` | `figs/spread_compression.pdf` | `master_arms.csv` |
| Figure 6 dose-response | `analysis/make_paper_figs.py` | `figs/dose_response.pdf` | `dose_response_both_*.json` |
| Figure 7 singular-value spectrum (NWPU, SigLIP2) | `analysis/make_paper_figs.py` | `figs/spectrum.pdf` | `paper/results/nwpu/shared_map_default.npz` (committed) |
| Figure 8 transfer matrix | `analysis/transfer_matrix.py` (needs the exported maps, stage 2) | `figs/transfer_matrix_default.pdf` | `paper/results/maps/*/maps_default.npz` |
| Figure 9 which side wins | `analysis/make_paper_figs.py` | `figs/which_side.pdf` | `symmetric_ablation_*.json` (single-side arms, near-dup rates) |

Figures that show real images need the raw collection on disk (stage 1) and the maps (`ablations/export_maps.py`):

| paper | script | notes |
|---|---|---|
| Figure 1 register gap (GoodNews/COCO, NWPU/COCO pairs) | `analysis/find_register_examples.py --top 24`, then `analysis/make_fig_register_pairs.py --goodnews 16 --nwpu 12 --intro-split --no-rank` | the chosen rows are in the committed `paper/results/qual_figs/register_cands_*.json`; output `paper/results/qual_figs/fig_register_intro_{text,image}_notext.pdf` |
| Figure 3 heat maps (CityFlow-NL frames) | `ltg.eval.grounding` gives the per-phrase patch scores; the panels are drawn on CityFlow-NL frames, which cannot be redistributed (NVIDIA licence) — obtain them from the AI City Challenge; boxes and captions are in `data/cityflow_nl/pairs.jsonl` | |
| Figure 4 one ROCOv2 query before/after | `analysis/make_fig_showcase.py --q 1754 --no-scatter` | needs `rocov2 × siglip2-so400m-16-384`, its maps, the ROCOv2 images (`$ICLR_ROCOV2_IMAGES`) and the committed `paper/results/qual_figs/rocov2_test_index.json` |

The committed grounding results hold the TIPSv2 random-split run and the SigLIP2-B/256 camera-split run; the TIPSv2 "new cameras"
column and "pooled two maps" row of Figure 3 come from the `ltg.eval.grounding` commands in stage 2 on `cityflow_cam` / `cityflow`.

## Layout

```
ltg/          store (cache.py), metrics, maps (linear, mlp, sets, lora), grounding, protocols, patch / phrase / API encoders
scripts/      fetch_cache, verify_cells, reproduce, export_legacy_cache, import_legacy, timing_table
configs/      collections.json, encoders.json, grid.json, published.json, cache_release.json, cache_manifest.json
results/      store-layout results: results/<collection>/<encoder>/*.json
paper/        analysis/ (tables + figures), ablations/ (experiments), legacy/ (embedding pipeline), results/ (legacy-layout
              results + master CSVs), jobs/ (batch drivers), figs/ and appendix/ (outputs), build.py
jobs/         grid driver, Colab jobs for the LoRA / scaling runs
```
