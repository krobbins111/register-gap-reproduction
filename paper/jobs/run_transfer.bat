@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
echo [1/2] export the default-encoder maps for the 11 collections (GPU, ~1 min each; skips cells already exported)
%PY% ablations\export_maps.py --cells coco:default facad:default fashion200k:default goodnews:default semart:default scimmir:default skincap:default treeoflife:default rocov2:default rsicd:default nwpu:default > results\log_transfer.txt 2>&1
echo [2/2] cross-collection transfer matrix (CPU)
%PY% analysis\transfer_matrix.py >> results\log_transfer.txt 2>&1
type results\log_transfer.txt | findstr /C:"[transfer]"
pause
