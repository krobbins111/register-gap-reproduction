@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM Candidate search + contact sheets for the two register-gap figures.
REM Needs results\maps\rocov2, goodnews, coco (run_transfer.bat already exported them).
%PY% analysis\find_register_examples.py --top 24 %*
echo done - open results\qual_figs\register_cands_rocov2.png and register_cands_goodnews.png
pause
