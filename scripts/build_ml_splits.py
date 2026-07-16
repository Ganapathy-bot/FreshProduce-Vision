#!/usr/bin/env python3
"""Rebuild ml_splits/ ImageFolder tree from annotations.csv (hardlink or copy)."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotations" / "annotations.csv"


def main():
    with ANN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        src = ROOT / r["relative_path"]
        dest = ROOT / "ml_splits" / r["split"] / r["class"] / r["image_name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        try:
            dest.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dest)
    print(f"Built ml_splits for {len(rows)} images under {ROOT/'ml_splits'}")


if __name__ == "__main__":
    main()
