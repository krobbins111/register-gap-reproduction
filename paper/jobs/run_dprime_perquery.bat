@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM A-priori d' for all 58 cells with the paper's per-query definition (Eq. 1),
REM on the full train split. CPU only; a few minutes. Writes
REM results\dprime_apriori_perquery.csv, which the figure/table scripts read.
%PY% analysis\apriori_dprime.py --definition perquery --all
pause
