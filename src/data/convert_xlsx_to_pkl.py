# src/data/convert_xlsx_to_pkl.py
# -*- coding: utf-8 -*-
"""
Corpus conversion: raw XLSX -> PKL.

Goal
----
Convert every XLSX file in data/raw/ into a PKL with the same schema
(one .pkl per .xlsx, same stem), so the rest of the pipeline reads a single,
fast, consistent format instead of parsing Excel repeatedly.

Usage (example)
---------------
from pathlib import Path
from src.data.convert_xlsx_to_pkl import convert_folder_xlsx_to_pkl

convert_folder_xlsx_to_pkl(
    raw_dir=Path("data/raw"),
    processed_dir=Path("data/01_corpus_pkl"),
)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def convert_folder_xlsx_to_pkl(
    raw_dir: Path,
    processed_dir: Path,
    pattern: str = "*.xlsx",
    engine: str = "openpyxl",
) -> list[Path]:
    """
    Converts every XLSX in raw_dir to a PKL in processed_dir.
    Returns the list of generated PKL files.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(raw_dir.glob(pattern))
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files found in: {raw_dir}")

    generated = []
    for xlsx_path in xlsx_files:
        df = pd.read_excel(xlsx_path, engine=engine)
        pkl_path = processed_dir / f"{xlsx_path.stem}.pkl"
        df.to_pickle(pkl_path)
        generated.append(pkl_path)

    return generated
