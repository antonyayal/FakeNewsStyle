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
from src.features.emotion_extractor import EmotionExtractor, EmotionExtractorConfig


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

parser.add_argument(
    "--extract_emotion",
    type=int,
    default=0,
    help="Extract emotion features using pysentimiento (0 = No, 1 = Yes)",
)

parser.add_argument(
    "--emotion_use_preprocess_tweet",
    type=int,
    default=0,
    help="Apply pysentimiento preprocess_tweet (0 = No, 1 = Yes)",
)

parser.add_argument(
    "--emotion_input_dir",
    type=str,
    default=None,
    help="Input dir with train/val/test PKLs (defaults to data/processed_by_model/FakeNewsCorpusSpanish if exists else processed_to_PKL)",
)

parser.add_argument(
    "--emotion_text_column",
    type=str,
    default="text",
    help="Column name containing the text field in the PKLs",
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
# Emotion feature extraction (OUTSIDE main)
# =====================================================
if args.extract_emotion == 1:
    print("Extracting emotion features (pysentimiento)")

    import pickle

    # Input: prefer processed_by_model (si existe), si no, processed_to_PKL
    emotion_input_dir = (
        Path(args.emotion_input_dir)
        if args.emotion_input_dir
        else (
            (BASE_DIR / "data" / "processed_by_model" / "FakeNewsCorpusSpanish")
            if (BASE_DIR / "data" / "processed_by_model" / "FakeNewsCorpusSpanish").exists()
            else PROCESSED_DIR
        )
    )

    # Output
    emotion_output_dir = BASE_DIR / "data" / "features" / "emotion" / "FakeNewsCorpusSpanish"
    emotion_output_dir.mkdir(parents=True, exist_ok=True)

    # Config extractor
    extractor = EmotionExtractor(
        EmotionExtractorConfig(
            lang="es",
            use_preprocess_tweet=(args.emotion_use_preprocess_tweet == 1),
            # normalize_signals_by="chars",  # opcional
            # extra_signals=True,            # opcional
        )
    )

    # Tus splits actuales en el repo
    splits = {
        "train": "train.pkl",
        "development": "development.pkl",
        "test": "test.pkl",
    }

    def _load_pkl_any(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _extract_texts_and_ids(obj, text_col: str):
        """
        Soporta:
        - list[dict]
        - dict con key 'data' -> list[dict]
        - pandas.DataFrame (si en algún momento guardaste DF)
        """
        # pandas DataFrame
        if hasattr(obj, "columns") and hasattr(obj, "__getitem__"):
            cols = list(obj.columns)
            if text_col not in cols:
                raise ValueError(f"Column '{text_col}' not found. Available columns: {cols}")
            texts = obj[text_col].astype(str).tolist()

            # ids: preferir 'id' si existe
            ids = None
            if "id" in cols:
                ids = obj["id"].tolist()
            return texts, ids

        # dict wrapper
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            obj = obj["data"]

        # list[dict]
        if isinstance(obj, list):
            if not obj:
                return [], None
            if not isinstance(obj[0], dict):
                raise ValueError("PKL list must contain dict rows.")
            if text_col not in obj[0]:
                raise ValueError(
                    f"Key '{text_col}' not found in PKL rows. "
                    f"Available keys: {list(obj[0].keys())}"
                )
            texts = [str(r.get(text_col, "")) for r in obj]

            # ids: usa 'id' si existe, si no, None (pero filas siguen alineadas por índice)
            ids = [r.get("id") for r in obj] if "id" in obj[0] else None
            return texts, ids

        raise ValueError(f"Unsupported PKL content type: {type(obj)}")

    for split_name, filename in splits.items():
        in_path = emotion_input_dir / filename
        if not in_path.exists():
            print(f"Skipped (missing): {in_path}")
            continue

        obj = _load_pkl_any(in_path)
        texts, ids = _extract_texts_and_ids(obj, args.emotion_text_column)

        out_path = emotion_output_dir / f"{split_name}_emotion.pkl"

        # Guarda un PKL estándar: {"ids", "X", "feature_names", ...}
        extractor.extract_and_save_pkl(
            texts=texts,
            ids=ids,
            output_path=out_path,
            batch_size=32,
            metadata={
                "dataset": "FakeNewsCorpusSpanish",
                "split": split_name,
                "source_pkl": str(in_path),
                "text_column": args.emotion_text_column,
            },
        )

        print(f"Saved emotion features: {out_path.name} | samples={len(texts)}")

    print("Emotion feature extraction completed")
else:
    print("Emotion feature extraction skipped")


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
