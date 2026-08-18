# scripts/all_xlsx_to_pkl.py
"""
Standalone XLSX -> PKL converter (superseded by `python main.py --prepare_corpus`).

Note: BASE_DIR here resolves to scripts/, and the referenced paths
(data/raw/FakeNewsCorpusSpanish, data/processed/FakeNewsCorpusSpanish) don't
match the current repo layout (data/raw/*.xlsx flat, data/01_corpus_pkl/
output) -- this script predates that layout and is kept for reference only.
Use `python main.py --prepare_corpus` for the current corpus-prep step.
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.data.convert_xlsx_to_pkl import convert_folder_xlsx_to_pkl


def main():
    parser = argparse.ArgumentParser(description="FakeNewsStyle Main Entry Point")
    parser.add_argument(
        "--convert_xlsx",
        type=int,
        default=0,
        help="Convert XLSX corpus to PKL (0 = No, 1 = Yes)",
    )
    args = parser.parse_args()

    RAW_DIR = BASE_DIR / "data" / "raw" / "FakeNewsCorpusSpanish"
    PROCESSED_DIR = BASE_DIR / "data" / "processed" / "FakeNewsCorpusSpanish"

    if args.convert_xlsx == 1:
        print("🔄 Converting XLSX corpus to PKL...")
        generated = convert_folder_xlsx_to_pkl(RAW_DIR, PROCESSED_DIR)
        for p in generated:
            print(f"✅ Saved: {p.name}")
        print("✅ Conversion completed.\n")
    else:
        print("⏭️  Skipping XLSX → PKL conversion.")

    print("🚀 FakeNewsStyle main pipeline ready.")


if __name__ == "__main__":
    main()
