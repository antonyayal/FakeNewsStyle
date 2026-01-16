# main.py
# =====================================================
# Imports
# =====================================================
import argparse
import sys
from pathlib import Path
import pickle

# =====================================================
# Project setup
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data.convert_xlsx_to_pkl import convert_folder_xlsx_to_pkl
from src.experiments.run_experiment import run_train, run_test, run_train_test
from src.text.preprocess_text import preprocess_corpus_splits
from src.features.semantic_extractor import extract_semantic_features_for_splits
from src.features.emotion_extractor import extract_emotion_features_for_splits
from src.features.style_extractor import StyleExtractor, StyleExtractorConfig
from src.features.context_extractor import ContextExtractor, ContextExtractorConfig

# Optional torch: for CUDA detection messaging (EmotionExtractor already handles device)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# =====================================================
# Helpers
# =====================================================
def _load_pkl_any(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_texts_and_ids_from_obj(obj, text_col: str, id_col: str | None = None):
    """
    Soporta:
    - pandas.DataFrame
    - dict con key 'data' -> list[dict]
    - list[dict]
    Regresa: (texts, ids)
    """
    # pandas DataFrame
    if hasattr(obj, "columns") and hasattr(obj, "__getitem__"):
        cols = list(obj.columns)
        if text_col not in cols:
            raise ValueError(f"Column '{text_col}' not found. Available columns: {cols}")
        texts = obj[text_col].astype(str).tolist()

        ids = None
        if id_col and id_col in cols:
            ids = obj[id_col].tolist()
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
        ids = None
        if id_col and id_col in obj[0]:
            ids = [r.get(id_col) for r in obj]
        return texts, ids

    raise ValueError(f"Unsupported PKL content type: {type(obj)}")


def _default_input_dir(user_dir: str | None, fallback_a: Path, fallback_b: Path) -> Path:
    """
    user_dir si viene, si no:
    - si existe fallback_a (processed_by_model), úsalo
    - si no, usa fallback_b (processed_to_PKL)
    """
    if user_dir:
        return Path(user_dir)
    return fallback_a if fallback_a.exists() else fallback_b


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_emotion_device(requested: str) -> str:
    """
    Normaliza device para EmotionExtractorConfig:
    - "cuda" si se pidió cuda y hay CUDA disponible
    - si no, "cpu"
    """
    req = (requested or "cpu").lower().strip()
    if req not in {"cpu", "cuda"}:
        req = "cpu"

    if req == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return "cpu"


# =====================================================
# Argument parser (GLOBAL)  -> SOLO se define una vez
# =====================================================
parser = argparse.ArgumentParser(description="FakeNewsStyle Main Entry Point")

# ---- corpus
parser.add_argument(
    "--prepare_corpus",
    type=int,
    default=0,
    help="Prepare corpus from raw to processed (0 = No, 1 = Yes)",
)

# ---- run
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

# ---- preprocess
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
    help="Input dir with train/val/test PKLs (defaults to processed_to_PKL/FakeNewsCorpusSpanish)",
)
parser.add_argument(
    "--preprocess_output_dir",
    type=str,
    default=None,
    help="Output dir for preprocessed PKLs (defaults to processed_by_model/FakeNewsCorpusSpanish)",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="logs/preprocess",
    help="Directory to store preprocessing logs",
)

# ---- semantic
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

# ---- emotion
parser.add_argument(
    "--extract_emotion",
    type=int,
    default=0,
    help="Extract emotion features using pysentimiento (0 = No, 1 = Yes)",
)
parser.add_argument(
    "--emotion_device",
    type=str,
    default="cpu",
    help="Device for emotion extractor: cpu | cuda",
)
parser.add_argument(
    "--emotion_batch_size",
    type=int,
    default=32,
    help="Batch size for emotion extractor",
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
    help="Input dir for emotion (defaults to processed_by_model if exists else processed_to_PKL)",
)
parser.add_argument(
    "--emotion_text_column",
    type=str,
    default="Text",
    help="Text column for emotion extractor (your DF uses: Headline or Text)",
)
parser.add_argument(
    "--emotion_id_column",
    type=str,
    default="Id",
    help="ID column for alignment (your DF uses: Id)",
)

# ---- style
parser.add_argument("--extract_style", type=int, default=0, help="Extract style features using spaCy/textstat/wordfreq (0 = No, 1 = Yes)")
parser.add_argument("--style_input_dir", type=str, default=None, help="Input dir for style (defaults to processed_by_model if exists else processed_to_PKL)")
parser.add_argument("--style_text_column", type=str, default="Text", help="Text column for style extractor (your DF uses: Headline or Text)")
parser.add_argument("--style_id_column", type=str, default="Id", help="ID column for alignment (your DF uses: Id)")
parser.add_argument("--style_spacy_model", type=str, default="es_core_news_sm", help="spaCy model name for Spanish")
parser.add_argument("--style_batch_size", type=int, default=64, help="Batch size for style extractor spaCy pipe")
parser.add_argument("--style_no_readability", type=int, default=0, help="Disable readability features (0 = keep, 1 = disable)")
parser.add_argument("--style_no_formality", type=int, default=0, help="Disable formality features (0 = keep, 1 = disable)")
parser.add_argument("--style_no_oov", type=int, default=0, help="Disable OOV/spelling proxy features (0 = keep, 1 = disable)")
parser.add_argument("--style_no_diversity", type=int, default=0, help="Disable diversity features (0 = keep, 1 = disable)")
parser.add_argument("--style_no_extra_signals", type=int, default=0, help="Disable extra stylometric signals (0 = keep, 1 = disable)")
parser.add_argument("--style_oov_zipf_threshold", type=float, default=1.5, help="Zipf threshold (wordfreq) below which a word is considered rare/OOV-ish")

# ---- context
parser.add_argument("--extract_context", type=int, default=0, help="Extract context features (Source/Domain/Topic/Age) (0 = No, 1 = Yes)")
parser.add_argument("--context_input_dir", type=str, default=None, help="Input dir for context (defaults to processed_by_model if exists else processed_to_PKL)")
parser.add_argument("--context_topic_column", type=str, default="Topic", help="Column name for topic/category in input PKLs")
parser.add_argument("--context_source_column", type=str, default="Source", help="Column name for source/media in input PKLs")
parser.add_argument("--context_link_column", type=str, default="Link", help="Column name for URL link in input PKLs")
parser.add_argument("--context_id_column", type=str, default="Id", help="Column name for row ID to preserve alignment across feature PKLs")
parser.add_argument("--context_author_column", type=str, default=None, help="Optional column name for author (if exists). If not provided, tries heuristic from URL")
parser.add_argument("--context_date_column", type=str, default=None, help="Optional column name for publish date to compute age in days")
parser.add_argument("--context_source_dim", type=int, default=32, help="Hash-embedding dim for Source")
parser.add_argument("--context_domain_dim", type=int, default=32, help="Hash-embedding dim for Domain extracted from Link")
parser.add_argument("--context_topic_dim", type=int, default=16, help="Hash-embedding dim for Topic")
parser.add_argument("--context_author_dim", type=int, default=16, help="Hash-embedding dim for Author (if available)")
parser.add_argument("--context_n_hashes", type=int, default=2, help="Number of hashes per field for hash embeddings")
parser.add_argument("--context_unsigned", type=int, default=0, help="Disable signed hashing (0 = signed, 1 = unsigned)")

# ---- parse ONCE
args = parser.parse_args()


# =====================================================
# Paths
# =====================================================
RAW_DIR = BASE_DIR / "data" / "raw" / "FakeNewsCorpusSpanish"
PROCESSED_DIR = BASE_DIR / "data" / "processed_to_PKL" / "FakeNewsCorpusSpanish"
PROCESSED_BY_MODEL_DIR = BASE_DIR / "data" / "processed_by_model" / "FakeNewsCorpusSpanish"

# Logs (paradigma: como preprocess, pero por módulo)
LOGS_FEATURES_DIR = BASE_DIR / "logs" / "features"
LOGS_SEMANTIC_DIR = _ensure_dir(LOGS_FEATURES_DIR / "semantic")
LOGS_EMOTION_DIR = _ensure_dir(LOGS_FEATURES_DIR / "emotion")
LOGS_STYLE_DIR = _ensure_dir(LOGS_FEATURES_DIR / "style")
LOGS_CONTEXT_DIR = _ensure_dir(LOGS_FEATURES_DIR / "context")


# =====================================================
# Step 1: Prepare corpus (WITH LOG)
# =====================================================
if args.prepare_corpus == 1:
    from datetime import datetime

    log_dir_step1 = _ensure_dir(BASE_DIR / "logs" / "prepare_corpus")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir_step1 / f"prepare_corpus_{timestamp}.log"

    def _log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    print("Preparing corpus from raw to processed")
    _log("PrepareCorpus: START")
    _log(f"raw_dir={RAW_DIR}")
    _log(f"processed_dir={PROCESSED_DIR}")

    try:
        generated = convert_folder_xlsx_to_pkl(
            raw_dir=RAW_DIR,
            processed_dir=PROCESSED_DIR,
        )

        for p in generated:
            print(f"Saved: {p.name}")
            _log(f"saved_file={p.name}")

        _log(f"num_files_generated={len(generated)}")
        _log("PrepareCorpus: END (SUCCESS)")
        print("Corpus preparation completed")

    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}")
        _log("PrepareCorpus: END (FAILED)")
        raise

else:
    print("Corpus preparation skipped")



# =====================================================
# Step 2: Preprocess text for XLM-R
# =====================================================
if args.preprocess_text == 1:
    input_dir = Path(args.preprocess_input_dir) if args.preprocess_input_dir else PROCESSED_DIR
    output_dir = Path(args.preprocess_output_dir) if args.preprocess_output_dir else PROCESSED_BY_MODEL_DIR
    log_dir = BASE_DIR / args.log_dir

    print("Preprocessing text for XLM-RoBERTa")
    preprocess_corpus_splits(input_dir=input_dir, output_dir=output_dir, log_dir=log_dir)
    print("Text preprocessing completed")
else:
    print("Text preprocessing skipped")


# =====================================================
# Step 3: Semantic features
# =====================================================
if args.extract_semantic == 1:
    print("Extracting semantic features (XLM-RoBERTa)")

    input_dir = Path(args.preprocess_output_dir) if args.preprocess_output_dir else PROCESSED_BY_MODEL_DIR
    output_dir = BASE_DIR / "data" / "features" / "semantic" / "FakeNewsCorpusSpanish"

    extract_semantic_features_for_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        log_dir=LOGS_SEMANTIC_DIR,
        pooling=args.semantic_pooling,
        device=args.semantic_device,
        batch_size=8,
        max_len=256,
    )
    print("Semantic feature extraction completed")
else:
    print("Semantic feature extraction skipped")


# =====================================================
# Step 4: Emotion features
# =====================================================
if args.extract_emotion == 1:
    print("Extracting emotion features (pysentimiento)")

    emotion_input_dir = _default_input_dir(
        args.emotion_input_dir,
        PROCESSED_BY_MODEL_DIR,
        PROCESSED_DIR,
    )

    emotion_output_dir = BASE_DIR / "data" / "features" / "emotion" / "FakeNewsCorpusSpanish"
    emotion_output_dir.mkdir(parents=True, exist_ok=True)

    extract_emotion_features_for_splits(
        input_dir=emotion_input_dir,
        output_dir=emotion_output_dir,
        log_dir=LOGS_EMOTION_DIR,
        text_col=args.emotion_text_column,          # ⚠️ usa text_xlmr si vienes de preprocess
        batch_size=int(args.emotion_batch_size),
        device=args.emotion_device,                 # "cuda" | "cpu"
        use_preprocess_tweet=(args.emotion_use_preprocess_tweet == 1),
        normalize_signals_by="chars",
        extra_signals=True,
    )

    print("Emotion feature extraction completed")
else:
    print("Emotion feature extraction skipped")


# =====================================================
# Step 5: Style features
# =====================================================
if args.extract_style == 1:
    print("Extracting style features (spaCy/textstat/wordfreq)")

    style_input_dir = _default_input_dir(args.style_input_dir, PROCESSED_BY_MODEL_DIR, PROCESSED_DIR)
    style_output_dir = BASE_DIR / "data" / "features" / "style" / "FakeNewsCorpusSpanish"
    style_output_dir.mkdir(parents=True, exist_ok=True)

    style_extractor = StyleExtractor(
        StyleExtractorConfig(
            spacy_model=args.style_spacy_model,
            compute_readability=(args.style_no_readability == 0),
            compute_formality=(args.style_no_formality == 0),
            compute_oov=(args.style_no_oov == 0),
            compute_diversity=(args.style_no_diversity == 0),
            extra_signals=(args.style_no_extra_signals == 0),
            oov_zipf_threshold=float(args.style_oov_zipf_threshold),
        )
    )

    splits = {"train": "train.pkl", "val": "val.pkl", "test": "test.pkl"}

    for split_name, filename in splits.items():
        in_path = style_input_dir / filename
        if not in_path.exists():
            print(f"Skipped (missing): {in_path}")
            continue

        obj = _load_pkl_any(in_path)
        texts, ids = _extract_texts_and_ids_from_obj(obj, args.style_text_column, args.style_id_column)

        out_path = style_output_dir / f"{split_name}_style.pkl"

        style_extractor.save_features_pkl(
            texts=texts,
            ids=ids,
            output_path=out_path,
            batch_size=int(args.style_batch_size),
            metadata={
                "dataset": "FakeNewsCorpusSpanish",
                "split": split_name,
                "source_pkl": str(in_path),
                "text_column": args.style_text_column,
                "id_column": args.style_id_column,
            },
            log_dir=LOGS_STYLE_DIR,
            log_name=f"style_{split_name}.log",
        )

        print(f"Saved style features: {out_path.name} | samples={len(texts)}")

    print("Style feature extraction completed")
else:
    print("Style feature extraction skipped")


# =====================================================
# Step 6: Context features
# =====================================================
if args.extract_context == 1:
    print("Extracting context features (Source/Domain/Topic/Age)")

    import pandas as pd

    context_input_dir = _default_input_dir(args.context_input_dir, PROCESSED_BY_MODEL_DIR, PROCESSED_DIR)
    context_output_dir = BASE_DIR / "data" / "features" / "context" / "FakeNewsCorpusSpanish"
    context_output_dir.mkdir(parents=True, exist_ok=True)

    ctx_extractor = ContextExtractor(
        ContextExtractorConfig(
            topic_column=args.context_topic_column,
            source_column=args.context_source_column,
            link_column=args.context_link_column,
            id_column=args.context_id_column,
            author_column=args.context_author_column,
            date_column=args.context_date_column,
            source_dim=int(args.context_source_dim),
            domain_dim=int(args.context_domain_dim),
            topic_dim=int(args.context_topic_dim),
            author_dim=int(args.context_author_dim),
            n_hashes=int(args.context_n_hashes),
            signed=(args.context_unsigned == 0),
        )
    )

    splits = {"train": "train.pkl", "val": "val.pkl", "test": "test.pkl"}

    for split_name, filename in splits.items():
        in_path = context_input_dir / filename
        if not in_path.exists():
            print(f"Skipped (missing): {in_path}")
            continue

        df = pd.read_pickle(in_path)

        required_cols = [args.context_topic_column, args.context_source_column, args.context_link_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {in_path.name}: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        ids = df[args.context_id_column].tolist() if (args.context_id_column and args.context_id_column in df.columns) else None
        rows = df.to_dict(orient="records")

        out_path = context_output_dir / f"{split_name}_context.pkl"

        ctx_extractor.save_features_pkl(
            rows=rows,
            ids=ids,
            output_path=out_path,
            metadata={
                "dataset": "FakeNewsCorpusSpanish",
                "split": split_name,
                "source_pkl": str(in_path),
                "topic_column": args.context_topic_column,
                "source_column": args.context_source_column,
                "link_column": args.context_link_column,
                "id_column": args.context_id_column,
                "author_column": args.context_author_column,
                "date_column": args.context_date_column,
            },
            log_dir=LOGS_CONTEXT_DIR,
            log_name=f"context_{split_name}.log",
        )

        print(f"Saved context features: {out_path.name} | samples={len(rows)}")

    print("Context feature extraction completed")
else:
    print("Context feature extraction skipped")


# =====================================================
# Main (training/testing only)
# =====================================================
def main():
    print("FakeNewsStyle main initialized")

    if args.mode is None:
        print("No --mode provided. Exiting.")
        return

    if args.config is None:
        raise ValueError("--config is required when using --mode")

    if args.mode == "test" and not args.ckpt:
        raise ValueError("--ckpt is required when --mode test")

    if args.mode == "train":
        best_ckpt = run_train(config_path=args.config, out_dir=args.out_dir)
        print(best_ckpt or "")
        return

    if args.mode == "test":
        _ = run_test(config_path=args.config, ckpt_path=args.ckpt, out_dir=args.out_dir)
        print("Test completed")
        return

    if args.mode == "train_test":
        _ = run_train_test(config_path=args.config, out_dir=args.out_dir)
        print("Train+Test completed")
        return


# =====================================================
# Entrypoint
# =====================================================
if __name__ == "__main__":
    main()
