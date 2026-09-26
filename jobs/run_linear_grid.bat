@echo off
setlocal
cd /d %~dp0\..
set PY=..\venv\Scripts\python.exe
if not exist results\logs mkdir results\logs
echo Linear arms (both + shared, seeds 0 1, baselines) on all 58 cells, timed. ~1 min/cell on an RTX 3080; ROCOv2 and the 50k collections take longer.
echo Reports: results\^<collection^>\^<encoder^>\linear_arms.json   Then: %PY% scripts\timing_table.py

echo [1/58] coco x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection coco --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_coco_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_coco_clip-vitl14-laion2b.txt
echo [2/58] coco x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection coco --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_coco_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_coco_metaclip2-ww-huge.txt
echo [3/58] coco x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection coco --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_coco_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_coco_siglip2-so400m-16-384.txt
echo [4/58] coco x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection coco --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_coco_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_coco_siglipv1-so400m-14-384.txt
echo [5/58] facad x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection facad --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_facad_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_facad_clip-vitl14-laion2b.txt
echo [6/58] facad x fashionclip
%PY% -m ltg.maps.linear --collection facad --encoder fashionclip --arms both shared --seeds 0 1 --baselines > results\logs\linear_facad_fashionclip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_facad_fashionclip.txt
echo [7/58] facad x marqo-fashionsiglip
%PY% -m ltg.maps.linear --collection facad --encoder marqo-fashionsiglip --arms both shared --seeds 0 1 --baselines > results\logs\linear_facad_marqo-fashionsiglip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_facad_marqo-fashionsiglip.txt
echo [8/58] facad x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection facad --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_facad_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_facad_metaclip2-ww-huge.txt
echo [9/58] facad x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection facad --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_facad_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_facad_siglip2-so400m-16-384.txt
echo [10/58] facad x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection facad --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_facad_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_facad_siglipv1-so400m-14-384.txt
echo [11/58] fashion200k x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection fashion200k --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_fashion200k_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_fashion200k_clip-vitl14-laion2b.txt
echo [12/58] fashion200k x fashionclip
%PY% -m ltg.maps.linear --collection fashion200k --encoder fashionclip --arms both shared --seeds 0 1 --baselines > results\logs\linear_fashion200k_fashionclip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_fashion200k_fashionclip.txt
echo [13/58] fashion200k x marqo-fashionsiglip
%PY% -m ltg.maps.linear --collection fashion200k --encoder marqo-fashionsiglip --arms both shared --seeds 0 1 --baselines > results\logs\linear_fashion200k_marqo-fashionsiglip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_fashion200k_marqo-fashionsiglip.txt
echo [14/58] fashion200k x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection fashion200k --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_fashion200k_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_fashion200k_metaclip2-ww-huge.txt
echo [15/58] fashion200k x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection fashion200k --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_fashion200k_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_fashion200k_siglip2-so400m-16-384.txt
echo [16/58] fashion200k x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection fashion200k --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_fashion200k_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_fashion200k_siglipv1-so400m-14-384.txt
echo [17/58] goodnews x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection goodnews --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_goodnews_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_goodnews_clip-vitl14-laion2b.txt
echo [18/58] goodnews x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection goodnews --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_goodnews_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_goodnews_metaclip2-ww-huge.txt
echo [19/58] goodnews x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection goodnews --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_goodnews_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_goodnews_siglip2-so400m-16-384.txt
echo [20/58] goodnews x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection goodnews --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_goodnews_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_goodnews_siglipv1-so400m-14-384.txt
echo [21/58] nwpu x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection nwpu --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_nwpu_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_nwpu_clip-vitl14-laion2b.txt
echo [22/58] nwpu x georsclip-vitb32
%PY% -m ltg.maps.linear --collection nwpu --encoder georsclip-vitb32 --arms both shared --seeds 0 1 --baselines > results\logs\linear_nwpu_georsclip-vitb32.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_nwpu_georsclip-vitb32.txt
echo [23/58] nwpu x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection nwpu --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_nwpu_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_nwpu_metaclip2-ww-huge.txt
echo [24/58] nwpu x remoteclip-vitl14
%PY% -m ltg.maps.linear --collection nwpu --encoder remoteclip-vitl14 --arms both shared --seeds 0 1 --baselines > results\logs\linear_nwpu_remoteclip-vitl14.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_nwpu_remoteclip-vitl14.txt
echo [25/58] nwpu x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection nwpu --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_nwpu_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_nwpu_siglip2-so400m-16-384.txt
echo [26/58] nwpu x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection nwpu --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_nwpu_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_nwpu_siglipv1-so400m-14-384.txt
echo [27/58] rocov2 x biomedclip
%PY% -m ltg.maps.linear --collection rocov2 --encoder biomedclip --arms both shared --seeds 0 1 --baselines > results\logs\linear_rocov2_biomedclip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rocov2_biomedclip.txt
echo [28/58] rocov2 x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection rocov2 --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_rocov2_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rocov2_clip-vitl14-laion2b.txt
echo [29/58] rocov2 x medsiglip
%PY% -m ltg.maps.linear --collection rocov2 --encoder medsiglip --arms both shared --seeds 0 1 --baselines > results\logs\linear_rocov2_medsiglip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rocov2_medsiglip.txt
echo [30/58] rocov2 x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection rocov2 --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_rocov2_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rocov2_metaclip2-ww-huge.txt
echo [31/58] rocov2 x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection rocov2 --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_rocov2_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rocov2_siglip2-so400m-16-384.txt
echo [32/58] rocov2 x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection rocov2 --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_rocov2_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rocov2_siglipv1-so400m-14-384.txt
echo [33/58] rsicd x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection rsicd --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_rsicd_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rsicd_clip-vitl14-laion2b.txt
echo [34/58] rsicd x georsclip-vitb32
%PY% -m ltg.maps.linear --collection rsicd --encoder georsclip-vitb32 --arms both shared --seeds 0 1 --baselines > results\logs\linear_rsicd_georsclip-vitb32.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rsicd_georsclip-vitb32.txt
echo [35/58] rsicd x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection rsicd --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_rsicd_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rsicd_metaclip2-ww-huge.txt
echo [36/58] rsicd x remoteclip-vitl14
%PY% -m ltg.maps.linear --collection rsicd --encoder remoteclip-vitl14 --arms both shared --seeds 0 1 --baselines > results\logs\linear_rsicd_remoteclip-vitl14.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rsicd_remoteclip-vitl14.txt
echo [37/58] rsicd x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection rsicd --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_rsicd_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rsicd_siglip2-so400m-16-384.txt
echo [38/58] rsicd x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection rsicd --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_rsicd_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_rsicd_siglipv1-so400m-14-384.txt
echo [39/58] scimmir x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection scimmir --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_scimmir_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_scimmir_clip-vitl14-laion2b.txt
echo [40/58] scimmir x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection scimmir --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_scimmir_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_scimmir_metaclip2-ww-huge.txt
echo [41/58] scimmir x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection scimmir --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_scimmir_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_scimmir_siglip2-so400m-16-384.txt
echo [42/58] scimmir x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection scimmir --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_scimmir_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_scimmir_siglipv1-so400m-14-384.txt
echo [43/58] semart x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection semart --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_semart_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_semart_clip-vitl14-laion2b.txt
echo [44/58] semart x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection semart --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_semart_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_semart_metaclip2-ww-huge.txt
echo [45/58] semart x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection semart --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_semart_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_semart_siglip2-so400m-16-384.txt
echo [46/58] semart x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection semart --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_semart_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_semart_siglipv1-so400m-14-384.txt
echo [47/58] skincap x biomedclip
%PY% -m ltg.maps.linear --collection skincap --encoder biomedclip --arms both shared --seeds 0 1 --baselines > results\logs\linear_skincap_biomedclip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_skincap_biomedclip.txt
echo [48/58] skincap x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection skincap --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_skincap_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_skincap_clip-vitl14-laion2b.txt
echo [49/58] skincap x medsiglip
%PY% -m ltg.maps.linear --collection skincap --encoder medsiglip --arms both shared --seeds 0 1 --baselines > results\logs\linear_skincap_medsiglip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_skincap_medsiglip.txt
echo [50/58] skincap x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection skincap --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_skincap_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_skincap_metaclip2-ww-huge.txt
echo [51/58] skincap x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection skincap --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_skincap_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_skincap_siglip2-so400m-16-384.txt
echo [52/58] skincap x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection skincap --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_skincap_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_skincap_siglipv1-so400m-14-384.txt
echo [53/58] treeoflife x bioclip
%PY% -m ltg.maps.linear --collection treeoflife --encoder bioclip --arms both shared --seeds 0 1 --baselines > results\logs\linear_treeoflife_bioclip.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_treeoflife_bioclip.txt
echo [54/58] treeoflife x bioclip2
%PY% -m ltg.maps.linear --collection treeoflife --encoder bioclip2 --arms both shared --seeds 0 1 --baselines > results\logs\linear_treeoflife_bioclip2.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_treeoflife_bioclip2.txt
echo [55/58] treeoflife x clip-vitl14-laion2b
%PY% -m ltg.maps.linear --collection treeoflife --encoder clip-vitl14-laion2b --arms both shared --seeds 0 1 --baselines > results\logs\linear_treeoflife_clip-vitl14-laion2b.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_treeoflife_clip-vitl14-laion2b.txt
echo [56/58] treeoflife x metaclip2-ww-huge
%PY% -m ltg.maps.linear --collection treeoflife --encoder metaclip2-ww-huge --arms both shared --seeds 0 1 --baselines > results\logs\linear_treeoflife_metaclip2-ww-huge.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_treeoflife_metaclip2-ww-huge.txt
echo [57/58] treeoflife x siglip2-so400m-16-384
%PY% -m ltg.maps.linear --collection treeoflife --encoder siglip2-so400m-16-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_treeoflife_siglip2-so400m-16-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_treeoflife_siglip2-so400m-16-384.txt
echo [58/58] treeoflife x siglipv1-so400m-14-384
%PY% -m ltg.maps.linear --collection treeoflife --encoder siglipv1-so400m-14-384 --arms both shared --seeds 0 1 --baselines > results\logs\linear_treeoflife_siglipv1-so400m-14-384.txt 2>&1
findstr /C:"[sym] both" /C:"[sym] shared" results\logs\linear_treeoflife_siglipv1-so400m-14-384.txt

%PY% scripts\timing_table.py
%PY% scripts\compare_all_arms.py --legacy ..\indexability\out --master ..\iclr2027\results\master_arms.csv
pause
