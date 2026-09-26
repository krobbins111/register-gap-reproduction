"""SemArt test-split manifest.

Reads semart_test.csv (tab-delimited, cp1252) and produces an ordered list
of records. The order of this list defines gallery index order for the
whole pipeline — every later artifact (embeddings, captions, indexability
scores) refers to images by their position here.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Record:
    idx: int            # gallery index (position in this manifest)
    image_file: str     # filename inside Images/
    description: str    # curator comment = the real query for this image
    title: str
    author: str
    school: str
    timeframe: str
    painting_type: str


def load_manifest(semart_dir: Path, split: str = "test") -> list[Record]:
    csv_path = semart_dir / f"semart_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Point --semart-dir (or SEMART_DIR env "
            f"var) at the folder containing semart_test.csv and Images/."
        )
    image_dir = semart_dir / "Images"

    records: list[Record] = []
    skipped = 0
    with open(csv_path, encoding="cp1252", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            img = row["IMAGE_FILE"].strip()
            if not (image_dir / img).exists():
                skipped += 1
                continue
            records.append(Record(
                idx=len(records),
                image_file=img,
                description=row.get("DESCRIPTION", "").strip(),
                title=row.get("TITLE", "").strip(),
                author=row.get("AUTHOR", "").strip(),
                school=row.get("SCHOOL", "").strip(),
                timeframe=row.get("TIMEFRAME", "").strip(),
                painting_type=row.get("TYPE", "").strip(),
            ))
    if skipped:
        print(f"[semart_data] WARNING: skipped {skipped} rows with missing "
              f"image files; gallery size = {len(records)}")
    return records


def save_manifest(records: list[Record], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=1)


def read_manifest(path: Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return [Record(**d) for d in json.load(f)]


def image_path(semart_dir: Path, rec: Record) -> Path:
    return semart_dir / "Images" / rec.image_file
