"""LoRA, head fine-tuning and the comparisons that need the encoder's weights (registergap-pipeline; the Colab jobs under jobs/registergap/).

These runs need a GPU and the images, so they live as self-contained Colab launcher jobs (the registergap pipeline's queue-daemon pattern, jobs/registergap/
daemon_2026-09-22.py): each job clones nothing from this package, embeds what it needs, trains, and pushes a results JSON per stage.
    colab_job_registergap_scaling_2026-09-22.py     maps vs LoRA (r=16, both towers) vs head-FT at 50/100/300/1000/all pairs, SigLIP2 base
    colab_job_registergap_lora_sets_2026-09-22.py   LoRA trained with the set objective (+ gate) against the linear set maps
    registergap_maps_on_lora_2026-09-22.py          LoRA rank sweep 4/16/64 with two maps on top; COCO forgetting; N50 on the LoRA embeddings
    registergap_biomedclip_2026-09-22.py            BiomedCLIP cells (ROCOv2, SkinCAP) with two maps
Their results are imported into results/<collection>/<encoder>/lora_*.json by scripts/import_registergap.py. The contract for new runs:
write the LoRA encoder's embeddings as a cell `<encoder>_lora_<tag>` (ltg.cache.save_cell) so maps-on-LoRA is just ltg.maps.linear.
    python -m ltg.maps.lora --list   prints the jobs and what each writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

JOBS = Path(__file__).resolve().parents[2] / "jobs" / "registergap"


def main(argv=None):
    for p in sorted(JOBS.glob("*.py")):
        doc = open(p, encoding="utf-8").read().split('"""')[1].strip().splitlines()[0] if '"""' in open(p, encoding="utf-8").read() else ""
        print(f"{p.name:55s} {doc[:110]}")


if __name__ == "__main__":
    main(sys.argv[1:])
