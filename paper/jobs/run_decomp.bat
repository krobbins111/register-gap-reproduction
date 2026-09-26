@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM Both-vs-shared decomposition, full grid (55 cells).
REM Each cell trains both+shared (seed 0) and writes
REM results\<ds>\both_decomposition_<tag>.json

%PY% ablations\decompose_both.py --dataset coco --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED coco/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset coco >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED coco/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset coco --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED coco/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset coco --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED coco/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset facad --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED facad/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset facad >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED facad/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset facad --mtag fashionclip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED facad/fashionclip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset facad --mtag marqo-fashionsiglip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED facad/marqo-fashionsiglip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset facad --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED facad/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset facad --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED facad/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset fashion200k --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED fashion200k/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset fashion200k >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED fashion200k/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset fashion200k --mtag fashionclip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED fashion200k/fashionclip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset fashion200k --mtag marqo-fashionsiglip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED fashion200k/marqo-fashionsiglip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset fashion200k --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED fashion200k/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset fashion200k --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED fashion200k/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset goodnews --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED goodnews/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset goodnews >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED goodnews/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset goodnews --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED goodnews/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset goodnews --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED goodnews/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset nwpu --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED nwpu/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset nwpu >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED nwpu/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset nwpu --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED nwpu/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset nwpu --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED nwpu/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rocov2 --mtag biomedclip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rocov2/biomedclip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rocov2 --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rocov2/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rocov2 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rocov2/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rocov2 --mtag medsiglip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rocov2/medsiglip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rocov2 --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rocov2/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rocov2 --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rocov2/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rsicd --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rsicd/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rsicd >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rsicd/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rsicd --mtag georsclip-vitb32 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rsicd/georsclip-vitb32 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rsicd --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rsicd/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rsicd --mtag remoteclip-vitl14 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rsicd/remoteclip-vitl14 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset rsicd --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED rsicd/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset scimmir --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED scimmir/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset scimmir >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED scimmir/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset scimmir --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED scimmir/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset scimmir --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED scimmir/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset semart --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED semart/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset semart --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED semart/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset semart --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED semart/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset skincap --mtag biomedclip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED skincap/biomedclip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset skincap --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED skincap/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset skincap >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED skincap/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset skincap --mtag medsiglip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED skincap/medsiglip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset skincap --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED skincap/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset skincap --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED skincap/siglipv1-so400m-384 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset treeoflife --mtag bioclip >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED treeoflife/bioclip >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset treeoflife --mtag bioclip2 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED treeoflife/bioclip2 >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset treeoflife --mtag clip-vitl14-laion2b >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED treeoflife/clip-vitl14-laion2b >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset treeoflife >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED treeoflife/default >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset treeoflife --mtag metaclip2-ww-huge >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED treeoflife/metaclip2-ww-huge >> results\log_decomp.txt
%PY% ablations\decompose_both.py --dataset treeoflife --mtag siglipv1-so400m-384 >> results\log_decomp.txt 2>&1
if errorlevel 1 echo FAILED treeoflife/siglipv1-so400m-384 >> results\log_decomp.txt

echo decomposition run complete
pause
