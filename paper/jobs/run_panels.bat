@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM One button: re-mine flips (adds the moderate band) then render panels.
REM Mining retrains the both-arm per cell (GPU, ~minutes each); rendering
REM verifies image<->embedding alignment before drawing anything.
for %%C in ("nwpu metaclip2-ww-huge" "rocov2 medsiglip" "skincap siglipv1-so400m-384" "treeoflife bioclip" "fashion200k metaclip2-ww-huge" "semart clip-vitl14-laion2b" "scimmir clip-vitl14-laion2b") do (
  for /f "tokens=1,2" %%a in (%%C) do (
    %PY% analysis\qualitative_flips.py --dataset %%a --mtag %%b >> results\log_qual.txt 2>&1
    %PY% analysis\render_qualitative_panels.py --dataset %%a --mtag %%b >> results\log_panels.txt 2>&1
  )
)
echo done - see results\qual_figs
pause
