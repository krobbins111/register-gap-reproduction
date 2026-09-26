@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM Qualitative mining on the flagship cells (one per domain story).
%PY% analysis\qualitative_flips.py --dataset nwpu --mtag metaclip2-ww-huge >> results\log_qual.txt 2>&1
%PY% analysis\qualitative_flips.py --dataset rocov2 --mtag medsiglip >> results\log_qual.txt 2>&1
%PY% analysis\qualitative_flips.py --dataset skincap --mtag siglipv1-so400m-384 >> results\log_qual.txt 2>&1
%PY% analysis\qualitative_flips.py --dataset treeoflife --mtag bioclip >> results\log_qual.txt 2>&1
%PY% analysis\qualitative_flips.py --dataset fashion200k --mtag metaclip2-ww-huge >> results\log_qual.txt 2>&1
%PY% analysis\qualitative_flips.py --dataset semart --mtag clip-vitl14-laion2b >> results\log_qual.txt 2>&1
%PY% analysis\qualitative_flips.py --dataset scimmir --mtag clip-vitl14-laion2b >> results\log_qual.txt 2>&1
echo qualitative mining complete
pause
