@echo off
REM Qualitative showcase figures (ROCOv2, SigLIP2, default maps): one query before/after the maps.
REM Needs results\qual_figs\rocov2_test_index.json (already written) and cache\rocov2\images.
REM   run_showcase.bat            -> renders the three picked queries, with and without the rank panel
REM   run_showcase.bat --list     -> prints 30 more candidate queries (raw rank 20..400 -> 1)
cd /d "%~dp0"
set PY=..\venv\Scripts\python.exe
if "%~1"=="--list" (
  %PY% analysis\make_fig_showcase.py --list
) else (
  %PY% analysis\make_fig_showcase.py --q 1754 --q 4080 --q 9824 --no-scatter %*
)
