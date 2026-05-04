# main.py
# =====================================================
# Imports
# =====================================================
import argparse
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

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
from src.features.merge_raw_features_for_kan import merge_split
from src.models.train_vae_from_pkl import train_vae_from_paths
from src.models.kan import train_kan_from_pkls

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


# =====================================================
# Helpers
# =====================================================
def _load_pkl_any(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_texts_and_ids_from_obj(obj, text_col: str, id_col: str | None = None):
    if hasattr(obj, "columns") and hasattr(obj, "__getitem__"):
        cols = list(obj.columns)
        if text_col not in cols:
            raise ValueError(f"Column '{text_col}' not found. Available columns: {cols}")
        texts = obj[text_col].astype(str).tolist()

        ids = None
        if id_col and id_col in cols:
            ids = obj[id_col].tolist()
        return texts, ids

    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        obj = obj["data"]

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
    if user_dir:
        return Path(user_dir)
    return fallback_a if fallback_a.exists() else fallback_b


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_emotion_device(requested: str) -> str:
    req = (requested or "cpu").lower().strip()
    if req not in {"cpu", "cuda"}:
        req = "cpu"

    if req == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return "cpu"


# =====================================================
# Argument parser
# =====================================================
parser = argparse.ArgumentParser(description="FakeNewsStyle Main Entry Point")

# ---- corpus
parser.add_argument("--prepare_corpus", type=int, default=0)

# ---- run
parser.add_argument("--mode", type=str, default=None, choices=["train", "test", "train_test"])
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./runs")

# ---- preprocess
parser.add_argument("--preprocess_text", type=int, default=0)
parser.add_argument("--preprocess_input_dir", type=str, default=None)
parser.add_argument("--preprocess_output_dir", type=str, default=None)
parser.add_argument("--log_dir", type=str, default="logs/preprocess")

# ---- semantic
parser.add_argument("--extract_semantic", type=int, default=0)
parser.add_argument("--semantic_pooling", type=str, default="mean", choices=["mean", "cls", "attention"])
parser.add_argument("--semantic_device", type=str, default="cpu")

# ---- emotion
parser.add_argument("--extract_emotion", type=int, default=0)
parser.add_argument("--emotion_device", type=str, default="cpu")
parser.add_argument("--emotion_batch_size", type=int, default=32)
parser.add_argument("--emotion_use_preprocess_tweet", type=int, default=0)
parser.add_argument("--emotion_input_dir", type=str, default=None)
parser.add_argument("--emotion_text_column", type=str, default="Text")
parser.add_argument("--emotion_id_column", type=str, default="Id")

# ---- style
parser.add_argument("--extract_style", type=int, default=0)
parser.add_argument("--style_input_dir", type=str, default=None)
parser.add_argument("--style_text_column", type=str, default="Text")
parser.add_argument("--style_id_column", type=str, default="Id")
parser.add_argument("--style_spacy_model", type=str, default="es_core_news_sm")
parser.add_argument("--style_batch_size", type=int, default=64)
parser.add_argument("--style_no_readability", type=int, default=0)
parser.add_argument("--style_no_formality", type=int, default=0)
parser.add_argument("--style_no_oov", type=int, default=0)
parser.add_argument("--style_no_diversity", type=int, default=0)
parser.add_argument("--style_no_extra_signals", type=int, default=0)
parser.add_argument("--style_oov_zipf_threshold", type=float, default=1.5)

# ---- context
parser.add_argument("--extract_context", type=int, default=0)
parser.add_argument("--context_input_dir", type=str, default=None)
parser.add_argument("--context_topic_column", type=str, default="Topic")
parser.add_argument("--context_source_column", type=str, default="Source")
parser.add_argument("--context_link_column", type=str, default="Link")
parser.add_argument("--context_id_column", type=str, default="Id")
parser.add_argument("--context_author_column", type=str, default=None)
parser.add_argument("--context_date_column", type=str, default=None)
parser.add_argument("--context_source_dim", type=int, default=32)
parser.add_argument("--context_domain_dim", type=int, default=32)
parser.add_argument("--context_topic_dim", type=int, default=16)
parser.add_argument("--context_author_dim", type=int, default=16)
parser.add_argument("--context_n_hashes", type=int, default=2)
parser.add_argument("--context_unsigned", type=int, default=0)

# ---- merge raw features before VAE
parser.add_argument(
    "--merge_raw_features",
    type=int,
    default=0,
    help="Merge semantic/emotion/style/context raw features before VAE",
)
parser.add_argument(
    "--raw_features_merge_output_dir",
    type=str,
    default=None,
    help="Output dir for raw merged feature PKLs",
)

# ---- VAE latent extraction
parser.add_argument("--run_vaes", type=int, default=0)
parser.add_argument("--semantic_latent_dim", type=int, default=128)
parser.add_argument("--emotion_latent_dim", type=int, default=16)
parser.add_argument("--style_latent_dim", type=int, default=16)
parser.add_argument("--context_latent_dim", type=int, default=64)

parser.add_argument("--vae_epochs", type=int, default=100)
parser.add_argument("--vae_batch_size", type=int, default=32)
parser.add_argument("--vae_learning_rate", type=float, default=1e-3)
parser.add_argument("--vae_beta", type=float, default=1.0)
parser.add_argument("--vae_dropout", type=float, default=0.1)
parser.add_argument("--vae_data_output_dir", type=str, default="data/vae_outputs")
parser.add_argument("--vae_model_output_dir", type=str, default="models/vae")

# ---- merge VAE latents
parser.add_argument(
    "--merge_vae_latents",
    type=int,
    default=0,
    help="Merge VAE latent PKLs into one KAN-ready dataset",
)
parser.add_argument(
    "--merge_output_dir",
    type=str,
    default=None,
    help="Output dir for merged latent VAE PKLs",
)

# ---- KAN classifier
parser.add_argument(
    "--train_kan",
    type=int,
    default=0,
    help="Train KAN classifier using merged PKLs",
)
parser.add_argument("--kan_train_pkl", type=str, default=None)
parser.add_argument("--kan_val_pkl", type=str, default=None)
parser.add_argument("--kan_test_pkl", type=str, default=None)
parser.add_argument("--kan_output_dir", type=str, default=None)
parser.add_argument("--kan_feature_key", type=str, default=None)
parser.add_argument("--kan_label_key", type=str, default="label")
parser.add_argument("--kan_hidden_dim", type=int, default=64)
parser.add_argument("--kan_num_basis", type=int, default=16)
parser.add_argument("--kan_dropout", type=float, default=0.2)
parser.add_argument("--kan_epochs", type=int, default=100)
parser.add_argument("--kan_batch_size", type=int, default=32)
parser.add_argument("--kan_lr", type=float, default=1e-3)
parser.add_argument("--kan_weight_decay", type=float, default=1e-4)
parser.add_argument("--kan_patience", type=int, default=15)

args = parser.parse_args()


# =====================================================
# Paths
# =====================================================
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed_to_PKL"
PROCESSED_BY_MODEL_DIR = BASE_DIR / "data" / "processed_by_model"

LOGS_FEATURES_DIR = BASE_DIR / "logs" / "features"
LOGS_SEMANTIC_DIR = _ensure_dir(LOGS_FEATURES_DIR / "semantic")
LOGS_EMOTION_DIR = _ensure_dir(LOGS_FEATURES_DIR / "emotion")
LOGS_STYLE_DIR = _ensure_dir(LOGS_FEATURES_DIR / "style")
LOGS_CONTEXT_DIR = _ensure_dir(LOGS_FEATURES_DIR / "context")

RAW_FEATURES_MERGE_OUTPUT_DIR = (
    Path(args.raw_features_merge_output_dir)
    if args.raw_features_merge_output_dir
    else BASE_DIR / "data" / "features_merged_for_kan"
)

VAE_LATENT_MERGE_OUTPUT_DIR = (
    Path(args.merge_output_dir)
    if args.merge_output_dir
    else BASE_DIR / "data" / "vae_latent_merged"
)


# =====================================================
# Step 1: Prepare corpus
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
        generated = convert_folder_xlsx_to_pkl(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
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
# Step 2: Preprocess text
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
    output_dir = BASE_DIR / "data" / "features" / "semantic"

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

    emotion_input_dir = _default_input_dir(args.emotion_input_dir, PROCESSED_BY_MODEL_DIR, PROCESSED_DIR)
    emotion_output_dir = BASE_DIR / "data" / "features" / "emotion"
    emotion_output_dir.mkdir(parents=True, exist_ok=True)

    extract_emotion_features_for_splits(
        input_dir=emotion_input_dir,
        output_dir=emotion_output_dir,
        log_dir=LOGS_EMOTION_DIR,
        text_col=args.emotion_text_column,
        batch_size=int(args.emotion_batch_size),
        device=args.emotion_device,
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
    style_output_dir = BASE_DIR / "data" / "features" / "style"
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

    context_input_dir = _default_input_dir(args.context_input_dir, PROCESSED_BY_MODEL_DIR, PROCESSED_DIR)
    context_output_dir = BASE_DIR / "data" / "features" / "context"
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

        required_cols = [
            args.context_topic_column,
            args.context_source_column,
            args.context_link_column,
        ]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns in {in_path.name}: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        ids = (
            df[args.context_id_column].tolist()
            if (args.context_id_column and args.context_id_column in df.columns)
            else None
        )

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
# Step 7: Merge raw feature PKLs before VAE
# =====================================================
if args.merge_raw_features == 1:
    print("Merging raw feature PKLs before VAE")

    raw_feature_dirs = {
        "semantic": BASE_DIR / "data" / "features" / "semantic",
        "emotion": BASE_DIR / "data" / "features" / "emotion",
        "style": BASE_DIR / "data" / "features" / "style",
        "context": BASE_DIR / "data" / "features" / "context",
    }

    raw_merge_output_dir = RAW_FEATURES_MERGE_OUTPUT_DIR
    raw_merge_output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        merge_split(
            split=split,
            feature_dirs=raw_feature_dirs,
            output_dir=raw_merge_output_dir,
        )

    print("Raw feature merge completed")
else:
    print("Raw feature merge skipped")


# =====================================================
# Step 8: Train VAEs for latent feature extraction
# =====================================================
if args.run_vaes == 1:
    print("Training VAEs for latent feature extraction")

    vae_configs = {
        "semantic": {
            "latent_dim": int(args.semantic_latent_dim),
            "hidden_dims": [512, 256],
            "feature_columns": None,
            "train_pkl": BASE_DIR / "data" / "features" / "semantic" / "train_semantic.pkl",
            "val_pkl": BASE_DIR / "data" / "features" / "semantic" / "val_semantic.pkl",
            "test_pkl": BASE_DIR / "data" / "features" / "semantic" / "test_semantic.pkl",
        },
        "emotion": {
            "latent_dim": int(args.emotion_latent_dim),
            "hidden_dims": [128, 64],
            "feature_columns": ["emo_probs", "sent_probs", "signals"],
            "train_pkl": BASE_DIR / "data" / "features" / "emotion" / "train_emotion.pkl",
            "val_pkl": BASE_DIR / "data" / "features" / "emotion" / "val_emotion.pkl",
            "test_pkl": BASE_DIR / "data" / "features" / "emotion" / "test_emotion.pkl",
        },
        "style": {
            "latent_dim": int(args.style_latent_dim),
            "hidden_dims": [128, 64],
            "feature_columns": None,
            "train_pkl": BASE_DIR / "data" / "features" / "style" / "train_style.pkl",
            "val_pkl": BASE_DIR / "data" / "features" / "style" / "val_style.pkl",
            "test_pkl": BASE_DIR / "data" / "features" / "style" / "test_style.pkl",
        },
        "context": {
            "latent_dim": int(args.context_latent_dim),
            "hidden_dims": [256, 128],
            "feature_columns": None,
            "train_pkl": BASE_DIR / "data" / "features" / "context" / "train_context.pkl",
            "val_pkl": BASE_DIR / "data" / "features" / "context" / "val_context.pkl",
            "test_pkl": BASE_DIR / "data" / "features" / "context" / "test_context.pkl",
        },
    }

    for feature_name, cfg in vae_configs.items():
        latent_dim = cfg["latent_dim"]

        output_data_dir = (
            BASE_DIR
            / args.vae_data_output_dir
            / feature_name
            / f"latent{latent_dim}"
        )

        output_model_dir = (
            BASE_DIR
            / args.vae_model_output_dir
            / feature_name
            / f"latent{latent_dim}"
        )

        print("=" * 80)
        print(f"Training VAE: {feature_name.upper()}")
        print(f"Latent dim: {latent_dim}")
        print(f"Feature columns: {cfg.get('feature_columns')}")
        print(f"Train PKL: {cfg['train_pkl']}")
        print(f"Data output: {output_data_dir}")
        print(f"Model output: {output_model_dir}")
        print("=" * 80)

        train_vae_from_paths(
            train_pkl=cfg["train_pkl"],
            val_pkl=cfg["val_pkl"],
            test_pkl=cfg["test_pkl"],
            feature_column=cfg.get("feature_columns"),
            label_column=None,
            latent_dim=latent_dim,
            hidden_dims=cfg["hidden_dims"],
            dropout=float(args.vae_dropout),
            beta=float(args.vae_beta),
            epochs=int(args.vae_epochs),
            batch_size=int(args.vae_batch_size),
            learning_rate=float(args.vae_learning_rate),
            output_data_dir=output_data_dir,
            output_model_dir=output_model_dir,
            feature_name=feature_name,
        )

    print("VAE latent feature extraction completed")
else:
    print("VAE latent feature extraction skipped")


# =====================================================
# Step 9: Merge VAE latent outputs for KAN
# =====================================================
if args.merge_vae_latents == 1:
    print("Merging VAE latent PKLs for KAN input")

    latent_dims = {
        "semantic": int(args.semantic_latent_dim),
        "emotion": int(args.emotion_latent_dim),
        "style": int(args.style_latent_dim),
        "context": int(args.context_latent_dim),
    }

    latent_dirs = {
        name: BASE_DIR / "data" / "vae_outputs" / name / f"latent{dim}"
        for name, dim in latent_dims.items()
    }

    merge_output_dir = VAE_LATENT_MERGE_OUTPUT_DIR
    merge_output_dir.mkdir(parents=True, exist_ok=True)

    print("Latent input dirs:")
    for name, path in latent_dirs.items():
        print(f"  {name}: {path}")

    print(f"Merged output dir: {merge_output_dir}")

    splits = ["train", "val", "test"]

    for split in splits:
        dfs = []
        labels = None

        for feature_name, feature_dir in latent_dirs.items():
            pkl_path = feature_dir / f"{split}.pkl"

            if not pkl_path.exists():
                raise FileNotFoundError(f"Missing latent PKL: {pkl_path}")

            df = pd.read_pickle(pkl_path)

            if "label" in df.columns:
                current_labels = df["label"].reset_index(drop=True)

                if labels is None:
                    labels = current_labels
                else:
                    if len(labels) != len(current_labels):
                        raise ValueError(
                            f"Label length mismatch in VAE latent merge for split '{split}', "
                            f"feature '{feature_name}'."
                        )

                df = df.drop(columns=["label"])

            df = df.reset_index(drop=True)

            df.columns = [
                col if str(col).startswith(f"{feature_name}_")
                else f"{feature_name}_{col}"
                for col in df.columns
            ]

            dfs.append(df)

        merged_df = pd.concat(dfs, axis=1)

        if labels is not None:
            merged_df["label"] = labels.values

        out_path = merge_output_dir / f"{split}.pkl"
        merged_df.to_pickle(out_path)

        print(
            f"Saved merged latent {split}: {out_path} | "
            f"samples={len(merged_df)} | dims={merged_df.shape[1]}"
        )

    print("VAE latent merge completed")
else:
    print("VAE latent merge skipped")


# =====================================================
# Step 10: Train KAN classifier
# =====================================================
if args.train_kan == 1:
    print("Training KAN classifier")

    # Default: use VAE latent merge output.
    # For raw-feature baseline, pass --kan_train_pkl/--kan_val_pkl/--kan_test_pkl explicitly.
    kan_train_pkl = (
        Path(args.kan_train_pkl)
        if args.kan_train_pkl
        else VAE_LATENT_MERGE_OUTPUT_DIR / "train.pkl"
    )

    kan_val_pkl = (
        Path(args.kan_val_pkl)
        if args.kan_val_pkl
        else VAE_LATENT_MERGE_OUTPUT_DIR / "val.pkl"
    )

    kan_test_pkl = (
        Path(args.kan_test_pkl)
        if args.kan_test_pkl
        else VAE_LATENT_MERGE_OUTPUT_DIR / "test.pkl"
    )

    kan_output_dir = (
        Path(args.kan_output_dir)
        if args.kan_output_dir
        else BASE_DIR / "data" / "kan_outputs" / "merged"
    )

    print(f"KAN train PKL: {kan_train_pkl}")
    print(f"KAN val PKL:   {kan_val_pkl}")
    print(f"KAN test PKL:  {kan_test_pkl}")
    print(f"KAN output:    {kan_output_dir}")

    train_kan_from_pkls(
        train_pkl=str(kan_train_pkl),
        val_pkl=str(kan_val_pkl),
        test_pkl=str(kan_test_pkl),
        feature_key=args.kan_feature_key,
        label_key=args.kan_label_key,
        hidden_dim=int(args.kan_hidden_dim),
        num_basis=int(args.kan_num_basis),
        dropout=float(args.kan_dropout),
        epochs=int(args.kan_epochs),
        batch_size=int(args.kan_batch_size),
        lr=float(args.kan_lr),
        weight_decay=float(args.kan_weight_decay),
        patience=int(args.kan_patience),
        output_dir=str(kan_output_dir),
    )

    print("KAN training completed")
else:
    print("KAN training skipped")


# =====================================================
# Main training/testing
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
