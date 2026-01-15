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
from src.experiments.run_experiment import run_train, run_test, run_train_test
from src.text.preprocess_text import preprocess_corpus_splits
from src.features.semantic_extractor import extract_semantic_features_for_splits


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

parser.add_argument(
    "--mode",
    type=str,
    default=None,
    choices=["train", "test", "train_test"],
    help="Execution mode: train | test | train_test",
)

parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to experiment config (json/yaml) used by Run/M3FEND",
)

parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help="Checkpoint path (required for --mode test)",
)

parser.add_argument(
    "--out_dir",
    type=str,
    default="./runs",
    help="Directory to store run artifacts (metrics/ckpt pointers)",
)

parser.add_argument(
    "--preprocess_text",
    type=int,
    default=0,
    help="Preprocess text for XLM-RoBERTa (0 = No, 1 = Yes)",
)

parser.add_argument(
    "--preprocess_input_dir",
    type=str,
    default=None,
    help="Input dir with train/val/test PKLs (defaults to data/processed/FakeNewsCorpusSpanish)",
)

parser.add_argument(
    "--preprocess_output_dir",
    type=str,
    default=None,
    help="Output dir for preprocessed PKLs (defaults to data/processed_by_model/FakeNewsCorpusSpanish)",
)

parser.add_argument(
    "--log_dir",
    type=str,
    default="logs/preprocess",
    help="Directory to store preprocessing logs",
)

parser.add_argument(
    "--extract_semantic",
    type=int,
    default=0,
    help="Extract semantic features using XLM-R (0 = No, 1 = Yes)",
)

parser.add_argument(
    "--semantic_pooling",
    type=str,
    default="mean",
    choices=["mean", "cls", "attention"],
    help="Pooling strategy for semantic extractor",
)

parser.add_argument(
    "--semantic_device",
    type=str,
    default="cpu",
    help="Device for semantic extractor: cpu | cuda",
)

args = parser.parse_args()


# =====================================================
# Pipeline control (OUTSIDE main)
# =====================================================
RAW_DIR = BASE_DIR / "data" / "raw" / "FakeNewsCorpusSpanish"
PROCESSED_DIR = BASE_DIR / "data" / "processed_to_PKL" / "FakeNewsCorpusSpanish"

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
# Text preprocessing control (OUTSIDE main)
# =====================================================
if args.preprocess_text == 1:
    # Defaults if user doesn't pass dirs
    input_dir = Path(args.preprocess_input_dir) if args.preprocess_input_dir else PROCESSED_DIR
    output_dir = (
        Path(args.preprocess_output_dir)
        if args.preprocess_output_dir
        else (BASE_DIR / "data" / "processed_by_model" / "FakeNewsCorpusSpanish")
    )
    log_dir = BASE_DIR / args.log_dir

    print("Preprocessing text for XLM-RoBERTa")
    preprocess_corpus_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        log_dir=log_dir,
    )
    print("Text preprocessing completed")
else:
    print("Text preprocessing skipped")

# =====================================================
# Semantic feature extraction (OUTSIDE main)
# =====================================================
if args.extract_semantic == 1:
    print("Extracting semantic features (XLM-RoBERTa)")

    input_dir = (
        Path(args.preprocess_output_dir)
        if args.preprocess_output_dir
        else BASE_DIR / "data" / "processed_by_model" / "FakeNewsCorpusSpanish"
    )

    output_dir = BASE_DIR / "data" / "features" / "semantic" / "FakeNewsCorpusSpanish"

    extract_semantic_features_for_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        log_dir=Path(args.log_dir) / "semantic",
        pooling=args.semantic_pooling,
        device=args.semantic_device,
        batch_size=8,
        max_len=256,
    )

    print("Semantic feature extraction completed")


# =====================================================
# Main
# =====================================================
def main():
    print("FakeNewsStyle main initialized")

    # If no mode provided, do nothing else
    if args.mode is None:
        print("No --mode provided. Exiting.")
        return

    # Basic validation
    if args.config is None:
        raise ValueError("--config is required when using --mode")

    if args.mode == "test" and not args.ckpt:
        raise ValueError("--ckpt is required when --mode test")

    # Dispatch
    if args.mode == "train":
        best_ckpt = run_train(config_path=args.config, out_dir=args.out_dir)
        print(best_ckpt or "")
        return

    if args.mode == "test":
        metrics = run_test(config_path=args.config, ckpt_path=args.ckpt, out_dir=args.out_dir)
        print("Test completed")
        return

    if args.mode == "train_test":
        metrics = run_train_test(config_path=args.config, out_dir=args.out_dir)
        print("Train+Test completed")
        return

# =====================================================
# Entrypoint
# =====================================================
if __name__ == "__main__":
    main()
