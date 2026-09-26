@echo off
setlocal
cd /d %~dp0
set PY=..\venv\Scripts\python.exe
if not exist results mkdir results
echo Adapter baselines: 14 cells (Tip-Adapter, Tip-Adapter-F, CLIP-Adapter x3 placements).

echo [1/14] skincap default
%PY% ablations\adapter_baselines.py --dataset skincap --seeds 0 1 > results\log_adpt_skincap_default.txt 2>&1
echo [2/14] coco default
%PY% ablations\adapter_baselines.py --dataset coco --seeds 0 1 > results\log_adpt_coco_default.txt 2>&1
echo [3/14] nwpu default
%PY% ablations\adapter_baselines.py --dataset nwpu --seeds 0 1 > results\log_adpt_nwpu_default.txt 2>&1
echo [4/14] rocov2 default
%PY% ablations\adapter_baselines.py --dataset rocov2 --seeds 0 1 > results\log_adpt_rocov2_default.txt 2>&1
echo [5/14] facad default
%PY% ablations\adapter_baselines.py --dataset facad --seeds 0 1 > results\log_adpt_facad_default.txt 2>&1
echo [6/14] rsicd default
%PY% ablations\adapter_baselines.py --dataset rsicd --seeds 0 1 > results\log_adpt_rsicd_default.txt 2>&1
echo [7/14] treeoflife default
%PY% ablations\adapter_baselines.py --dataset treeoflife --seeds 0 1 > results\log_adpt_treeoflife_default.txt 2>&1
echo [8/14] scimmir default
%PY% ablations\adapter_baselines.py --dataset scimmir --seeds 0 1 > results\log_adpt_scimmir_default.txt 2>&1
echo [9/14] fashion200k default
%PY% ablations\adapter_baselines.py --dataset fashion200k --seeds 0 1 > results\log_adpt_fashion200k_default.txt 2>&1
echo [10/14] goodnews default
%PY% ablations\adapter_baselines.py --dataset goodnews --seeds 0 1 > results\log_adpt_goodnews_default.txt 2>&1
echo [11/14] rocov2 clip-vitl14-laion2b
%PY% ablations\adapter_baselines.py --dataset rocov2 --mtag clip-vitl14-laion2b --seeds 0 1 > results\log_adpt_rocov2_clip_vitl14_laion2b.txt 2>&1
echo [12/14] nwpu clip-vitl14-laion2b
%PY% ablations\adapter_baselines.py --dataset nwpu --mtag clip-vitl14-laion2b --seeds 0 1 > results\log_adpt_nwpu_clip_vitl14_laion2b.txt 2>&1
echo [13/14] coco clip-vitl14-laion2b
%PY% ablations\adapter_baselines.py --dataset coco --mtag clip-vitl14-laion2b --seeds 0 1 > results\log_adpt_coco_clip_vitl14_laion2b.txt 2>&1
echo [14/14] skincap clip-vitl14-laion2b
%PY% ablations\adapter_baselines.py --dataset skincap --mtag clip-vitl14-laion2b --seeds 0 1 > results\log_adpt_skincap_clip_vitl14_laion2b.txt 2>&1

echo DONE.
pause