# =====================================================
# Imports
# =====================================================
import argparse
import sys
from pathlib import Path

# =====================================================
# Project setup
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data.convert_xlsx_to_pkl import convert_folder_xlsx_to_pkl

# =====================================================
# Argument parser (GLOBAL)
# =====================================================
parser = argparse.ArgumentParser(
    description="FakeNewsStyle Main Entry Point"
)

parser.add_argument(
    "--prepare_corpus",
    type=int,
    default=0,
    help="Prepare corpus from raw to processed (0 = No, 1 = Yes)"
)

args = parser.parse_args()


# =====================================================
# Pipeline control (OUTSIDE main)
# =====================================================
RAW_DIR = BASE_DIR / "data" / "raw" / "FakeNewsCorpusSpanish"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "FakeNewsCorpusSpanish"

if args.prepare_corpus == 1:
    print("Preparing corpus from raw to processed")
    generated = convert_folder_xlsx_to_pkl(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR
    )
    for p in generated:
        print(f"Saved: {p.name}")
    print("Corpus preparation completed")
else:
    print("Corpus preparation skipped")

# =====================================================
# Main
# =====================================================
def main():
    print("FakeNewsStyle main initialized")

# =====================================================
# Entrypoint
# =====================================================
if __name__ == "__main__":
    main()
