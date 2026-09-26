@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM 1) NWPU candidates: COCO caption twins under SigLIP2 (depiction register). Pick a row from the sheet.
if not exist results\qual_figs\register_cands_nwpu.json %PY% analysis\find_register_examples.py --only nwpu --top 24
REM 2) render the register-gap figures from the chosen candidate rows (edit --nwpu N after picking)
%PY% analysis\make_fig_register_pairs.py --rocov2 13 --goodnews 16 --fashion 10 --nwpu 12 --intro --no-rank %*
echo done - see results\qual_figs\fig_register_*.png  (add --title for slide versions)
pause
