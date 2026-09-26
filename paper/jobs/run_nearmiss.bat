@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM [1] Near-miss mass (descriptive a-priori statistic) for all 58 cells. CPU, minutes.
REM     -> results\nearmiss_apriori.csv
echo [1/2] near-miss statistic, all cells
%PY% analysis\apriori_nearmiss.py --all > results\log_nearmiss.txt 2>&1
REM [2] Rank transport on trained maps (justifies the band): 10 cells, GPU, ~1-2 min each.
REM     -> results\<ds>\rank_transport_<mtag>.json
echo [2/2] rank transport on trained maps
%PY% analysis\rank_transport.py --dataset nwpu >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset nwpu --mtag georsclip-vitb32 >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset rocov2 >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset rocov2 --mtag medsiglip >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset skincap >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset goodnews >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset scimmir >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset treeoflife >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset semart >> results\log_nearmiss.txt 2>&1
%PY% analysis\rank_transport.py --dataset coco >> results\log_nearmiss.txt 2>&1
echo done - see results\log_nearmiss.txt
pause
