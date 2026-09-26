@echo off
setlocal
cd /d %~dp0
set PY=..\venv\Scripts\python.exe
if not exist results mkdir results
echo Two-arm MLP (both + shared) on the default SigLIP2 encoder, 11 collections, seeds 0 1 2. Compare with symmetric_ablation_default.json.
echo Reports: results\^<ds^>\mlp_ablation_default.json  Logs: results\log_mlp_^<ds^>_default.txt

echo [1/11] skincap default
%PY% ablations\mlp_arms.py --dataset skincap --arms both shared --seeds 0 1 2 > results\log_mlp_skincap_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_skincap_default.txt
echo [2/11] semart default
%PY% ablations\mlp_arms.py --dataset semart --arms both shared --seeds 0 1 2 > results\log_mlp_semart_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_semart_default.txt
echo [3/11] rsicd default
%PY% ablations\mlp_arms.py --dataset rsicd --arms both shared --seeds 0 1 2 > results\log_mlp_rsicd_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_rsicd_default.txt
echo [4/11] rocov2 default
%PY% ablations\mlp_arms.py --dataset rocov2 --arms both shared --seeds 0 1 2 > results\log_mlp_rocov2_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_rocov2_default.txt
echo [5/11] scimmir default
%PY% ablations\mlp_arms.py --dataset scimmir --arms both shared --seeds 0 1 2 > results\log_mlp_scimmir_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_scimmir_default.txt
echo [6/11] nwpu default
%PY% ablations\mlp_arms.py --dataset nwpu --arms both shared --seeds 0 1 2 > results\log_mlp_nwpu_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_nwpu_default.txt
echo [7/11] treeoflife default
%PY% ablations\mlp_arms.py --dataset treeoflife --arms both shared --seeds 0 1 2 > results\log_mlp_treeoflife_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_treeoflife_default.txt
echo [8/11] goodnews default
%PY% ablations\mlp_arms.py --dataset goodnews --arms both shared --seeds 0 1 2 > results\log_mlp_goodnews_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_goodnews_default.txt
echo [9/11] fashion200k default
%PY% ablations\mlp_arms.py --dataset fashion200k --arms both shared --seeds 0 1 2 > results\log_mlp_fashion200k_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_fashion200k_default.txt
echo [10/11] facad default
%PY% ablations\mlp_arms.py --dataset facad --arms both shared --seeds 0 1 2 > results\log_mlp_facad_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_facad_default.txt
echo [11/11] coco default
%PY% ablations\mlp_arms.py --dataset coco --arms both shared --seeds 0 1 2 > results\log_mlp_coco_default.txt 2>&1
findstr /C:"[mlp] both" /C:"[mlp] shared" results\log_mlp_coco_default.txt

echo Done.
pause
