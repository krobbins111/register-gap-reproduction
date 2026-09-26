"""Supplementary ablations for the paper's "design-justifying negatives".

These scripts use flat imports (``from query_projection import QueryProjection``,
``from config import ...``) against the package root one level up. Inserting that
root on sys.path lets each script run unchanged either directly
(``python ablations/register_vs_modality.py``) or as a module
(``python -m ablations.register_vs_modality``).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
