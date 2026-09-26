@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PY=..\venv\Scripts\python.exe
REM LoRA fine-tuning of CLIP ViT-L/14 (LAION-2B) as the comparator for the maps.
REM Sized for a 10 GB card (bf16, gradient checkpointing, batch 64). Roughly 6-7 h total.
REM Each run also trains the maps on the same pairs, maps on top of LoRA, and scores COCO for forgetting.
%PY% -m pip install -q peft >> results\log_lora.txt 2>&1
echo [1/10] NWPU full
%PY% ablations\lora_finetune.py --dataset nwpu >> results\log_lora.txt 2>&1
echo [2/10] SkinCAP full (30 epochs, small catalog)
%PY% ablations\lora_finetune.py --dataset skincap --epochs 30 >> results\log_lora.txt 2>&1
echo [3/10] TreeOfLife full
%PY% ablations\lora_finetune.py --dataset treeoflife >> results\log_lora.txt 2>&1
echo [4/10] ROCOv2, 20k-pair subsample
%PY% ablations\lora_finetune.py --dataset rocov2 --limit-train 20000 >> results\log_lora.txt 2>&1
echo [5/10] GoodNews, 20k-pair subsample
%PY% ablations\lora_finetune.py --dataset goodnews --limit-train 20000 >> results\log_lora.txt 2>&1
echo [6-10/10] 1,000-pair dose on each (30 epochs)
%PY% ablations\lora_finetune.py --dataset nwpu --limit-train 1000 --epochs 30 >> results\log_lora.txt 2>&1
%PY% ablations\lora_finetune.py --dataset skincap --limit-train 1000 --epochs 30 >> results\log_lora.txt 2>&1
%PY% ablations\lora_finetune.py --dataset treeoflife --limit-train 1000 --epochs 30 >> results\log_lora.txt 2>&1
%PY% ablations\lora_finetune.py --dataset rocov2 --limit-train 1000 --epochs 30 >> results\log_lora.txt 2>&1
%PY% ablations\lora_finetune.py --dataset goodnews --limit-train 1000 --epochs 30 >> results\log_lora.txt 2>&1
type results\log_lora.txt | findstr /C:"[lora] LoRA test" /C:"maps_batch" /C:"COCO" /C:"->"
echo done - see results\log_lora.txt
pause
