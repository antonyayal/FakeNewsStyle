"""
Text preprocessing module aligned with XLM-RoBERTa tokenizer.

This module prepares raw Spanish text so it can be safely consumed by:
  FacebookAI/xlm-roberta-large-finetuned-conll02-spanish

The goal is to:
- remove noise that breaks tokenizers
- preserve all semantic and stylistic information
- ensure deterministic, reproducible preprocessing

Outputs (per split PKL):
- text_raw   : Headline + "\n\n" + Text
- text_xlmr  : cleaned text safe for XLM-R tokenizer
- label      : copied from Category
"""

import logging
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from datetime import datetime


# =====================================================
# Model name (single source of truth)
# =====================================================
MODEL_NAME = "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish"


# =====================================================
# Regex patterns (minimal & safe)
# =====================================================
RE_CONTROL_CHARS = re.compile(r"[\u0000-\u001F\u007F-\u009F\u200B\u200C\u200D\uFEFF]")
RE_MULTISPACE = re.compile(r"\s+")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)


# =====================================================
# Logging helpers
# =====================================================
def get_logger(log_dir: Path, name: str = "preprocess_text") -> logging.Logger:
    """
    Create a logger that writes to:
      - console
      - <log_dir>/<name>_YYYY-MM-DD_HH-MM-SS.log

    A NEW log file is created on every execution.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = (log_dir / f"{name}_{timestamp}.log").resolve()

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # ⛔ avoid duplicate root logging

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # File handler (unique per run)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to file: {log_file}")

    return logger



# =====================================================
# Core text preprocessing
# =====================================================
def clean_text_for_xlmr(
    text: str,
    replace_urls: bool = True,
    replace_emails: bool = True,
) -> str:
    """
    Minimal, tokenizer-safe cleaning for XLM-RoBERTa.

    Notes:
    - We intentionally do NOT remove punctuation, accents, or casing.
    - We do NOT apply stopwords, lemmatization, or stemming.
    - Output remains text (string), ready for the model tokenizer.
    """
    if text is None:
        return ""

    text = str(text)

    # Remove invisible / control characters that can break tokenization
    text = RE_CONTROL_CHARS.sub(" ", text)

    # Optional replacements (normalize, do not delete content)
    if replace_urls:
        text = RE_URL.sub("<URL>", text)

    if replace_emails:
        text = RE_EMAIL.sub("<EMAIL>", text)

    # Normalize whitespace
    text = RE_MULTISPACE.sub(" ", text).strip()

    return text


# =====================================================
# PKL-level preprocessing (single split)
# =====================================================
def preprocess_pkl_for_model(
    input_pkl: Path,
    output_pkl: Path,
    headline_col: str = "Headline",
    body_col: str = "Text",
    label_col: str = "Category",
    logger: logging.Logger | None = None,
) -> Path:
    """
    Preprocess a single PKL corpus split.

    Creates:
    - text_raw   : Headline + Text
    - text_xlmr  : cleaned text for XLM-R tokenizer
    - label      : copied from Category

    Original columns are preserved.
    """
    df = pd.read_pickle(input_pkl)

    required = {headline_col, body_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_pkl.name}: {missing}")

    headline = df[headline_col].fillna("").astype(str)
    body = df[body_col].fillna("").astype(str)

    text_raw = (headline + "\n\n" + body).str.strip()
    text_xlmr = text_raw.apply(clean_text_for_xlmr)

    out_df = df.copy()
    out_df["text_raw"] = text_raw
    out_df["text_xlmr"] = text_xlmr
    out_df["label"] = df[label_col]

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_pkl)

    if logger is not None:
        logger.info(f"PKL created: {output_pkl}")

    return output_pkl


# =====================================================
# Split resolver (handles development.pkl as val)
# =====================================================
def _resolve_split_file(input_dir: Path, split: str) -> Path:
    """
    Resolve expected PKL filename for a given split.

    Supported aliases:
      - train: train.pkl
      - val: val.pkl, valid.pkl, validation.pkl, dev.pkl, development.pkl
      - test: test.pkl
    """
    candidates = {
        "train": ["train.pkl"],
        "val": [
            "val.pkl",
            "valid.pkl",
            "validation.pkl",
            "dev.pkl",
            "development.pkl",
        ],
        "test": ["test.pkl"],
    }

    if split not in candidates:
        raise ValueError(f"Unsupported split: {split}")

    for fname in candidates[split]:
        p = input_dir / fname
        if p.exists():
            return p

    # Default expected path (used only for error messages)
    return input_dir / candidates[split][0]


# =====================================================
# Folder-level preprocessing (train/val/test)
# =====================================================
def preprocess_corpus_splits(
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
) -> None:
    """
    Preprocess train/val/test PKLs located in input_dir and write to output_dir.

    Input expected:
      - train.pkl
      - (val split) one of: val.pkl | valid.pkl | validation.pkl | dev.pkl | development.pkl
      - test.pkl

    Output written (standard names):
      - train.pkl
      - val.pkl
      - test.pkl
    """
    logger = get_logger(log_dir=log_dir)

    logger.info("Starting text preprocessing")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")

    train_in = _resolve_split_file(input_dir, "train")
    val_in = _resolve_split_file(input_dir, "val")
    test_in = _resolve_split_file(input_dir, "test")

    if not train_in.exists():
        logger.error(f"Missing input split (train). Expected: train.pkl. Dir: {input_dir}")
        raise FileNotFoundError(train_in)

    if not val_in.exists():
        logger.error(
            "Missing input split (val). Expected one of: val.pkl, valid.pkl, validation.pkl, dev.pkl, development.pkl. "
            f"Dir: {input_dir}"
        )
        raise FileNotFoundError(val_in)

    if not test_in.exists():
        logger.error(f"Missing input split (test). Expected: test.pkl. Dir: {input_dir}")
        raise FileNotFoundError(test_in)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train.pkl"
    val_out = output_dir / "val.pkl"
    test_out = output_dir / "test.pkl"

    logger.info(f"Processing split: train (input={train_in.name})")
    preprocess_pkl_for_model(input_pkl=train_in, output_pkl=train_out, logger=logger)

    logger.info(f"Processing split: val (input={val_in.name})")
    preprocess_pkl_for_model(input_pkl=val_in, output_pkl=val_out, logger=logger)

    logger.info(f"Processing split: test (input={test_in.name})")
    preprocess_pkl_for_model(input_pkl=test_in, output_pkl=test_out, logger=logger)

    logger.info("Text preprocessing completed")


# =====================================================
# Sanity check with tokenizer (optional)
# =====================================================
def sanity_check_tokenization(sample_text: str) -> None:
    """
    Utility function to verify that the text is compatible with the tokenizer.
    Not used in the pipeline, only for debugging.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(sample_text, truncation=True)
    print("Number of tokens:", len(tokens["input_ids"]))
