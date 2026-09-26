"""The register ladder on COCO (registergap-pipeline).

STATUS: placeholder — to be ported by registergap-pipeline / the other pipeline. See ANONYMIZATION.md for the rules.

Source to port:   registergap story/ladder/{build_captions,build_recaption_rung,eval_ladder,nickname_dose,transfer_matrix,write_tables,make_figs}.py
Contract:         Rungs are data (data/coco/ladder/rungs.json, committed: 5,000 captions x rungs, gpt-5-mini) so the eval never needs OpenAI; eval_ladder(encoder) -> results/coco/<encoder>/ladder.json; build_rungs() is the only step that needs a key.

Interface (do not change): ltg.cache.load_cell / save_cell / patches_path / phrases_path / result_path; encoder keys from
configs/encoders.json; collection ids from configs/collections.json; results are JSON only, one file per (collection, encoder,
experiment); nothing under the repo may hold embeddings, models or images.
"""
from __future__ import annotations

import sys


def main(argv=None):
    raise SystemExit("ltg/analysis/ladder.py: not ported yet (see ANONYMIZATION.md)")


if __name__ == "__main__":
    main(sys.argv[1:])
