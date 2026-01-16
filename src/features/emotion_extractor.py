"""
Emotion feature extractor (pysentimiento) + CUDA enablement (pattern aligned with semantic_extractor)

Goal
----
This module generates emotion + sentiment probabilities per sample using
pysentimiento (transformers under the hood) and saves them as new PKLs
(train/val/test). It also adds lightweight "emotional style signals".

Why this file exists
--------------------
- pysentimiento provides strong Spanish emotion and sentiment classifiers.
- We export stable features (probabilities + signals) for downstream models.

Outputs
-------
For each input PKL (train/val/test), creates an output PKL with:
- Id (if exists)
- label (if exists)
- emo_probs : list[float] (sorted by class name)
- sent_probs: list[float] (sorted by class name)
- emo_labels: list[str]
- sent_labels: list[str]
- signals  : list[float]
- signal_names : list[str]
- device : device used ("cuda"|"cpu")
- batch_size : batch size used
- normalize_signals_by : "chars" or "tokens"

Expected input columns
----------------------
- text (string)  OR  text_clean  OR  text_xlmr  (configurable via text_col)
Optional:
- Id, label (or Category)

Usage (example)
---------------
from pathlib import Path
from src.features.emotion_extractor import extract_emotion_features_for_splits

extract_emotion_features_for_splits(
    input_dir=Path("data/processed_by_model/FakeNewsCorpusSpanish"),
    output_dir=Path("data/features/emotion/FakeNewsCorpusSpanish"),
    log_dir=Path("logs/features"),
    text_col="text_xlmr",
    batch_size=32,
    device="cuda"  # or "cpu"
)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import emoji

from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

# Optional torch (for CUDA detection + no_grad/inference_mode)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# =====================================================
# Defaults
# =====================================================

DEFAULT_INTENSIFIERS = {
    "terrible",
    "increíble",
    "impactante",
    "alarmante",
    "horrible",
    "devastador",
    "escandaloso",
    "urgente",
    "brutal",
    "extremo",
    "indignante",
    "horrendo",
    "gravísimo",
    "espantoso",
    "inaudito",
    "bomba",
    "shock",
    "pánico",
    "catástrofe",
    "tragedia",
}


# =====================================================
# Logging (same pattern)
# =====================================================
def get_logger(log_dir: Path, name: str = "emotion_extractor") -> logging.Logger:
    """
    Create a logger that writes to console and to a unique file per run:
      <log_dir>/<name>_YYYY-MM-DD_HH-MM-SS.log
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = (log_dir / f"{name}_{timestamp}.log").resolve()

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to file: {log_file}")
    return logger


def _flush_and_close_logger(logger: logging.Logger) -> None:
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass


# =====================================================
# Config
# =====================================================
@dataclass
class ExtractConfig:
    lang: str = "es"
    batch_size: int = 32
    device: str = "cpu"  # "cuda" if available
    use_preprocess_tweet: bool = False

    # "chars" or "tokens"
    normalize_signals_by: str = "chars"
    extra_signals: bool = True
    safe_numeric: bool = True

    intensifiers: Optional[set] = None


# =====================================================
# Device helpers
# =====================================================
def _resolve_device(requested: str, logger: logging.Logger) -> str:
    """
    Resolve "cuda" vs "cpu" in the same spirit as semantic_extractor.
    """
    req = (requested or "cpu").lower().strip()
    if req not in {"cpu", "cuda"}:
        req = "cpu"

    if req == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        logger.warning("Requested device=cuda but CUDA is not available. Falling back to CPU.")
        return "cpu"

    return "cpu"


def _try_move_analyzer_to_device(analyzer: Any, device: str, logger: logging.Logger) -> None:
    """
    Best-effort: pysentimiento analyzers usually wrap a HF model.
    We try to move analyzer.model to the chosen device if present.
    """
    if device != "cuda":
        return
    if torch is None:
        logger.warning("torch is not available; cannot move model to CUDA. Using CPU.")
        return

    model = getattr(analyzer, "model", None)
    if model is None:
        # Not fatal: wrapper might manage device internally.
        logger.info("Analyzer has no attribute 'model'. Skipping explicit model.to(device).")
        return

    try:
        model.to("cuda")
        model.eval()
        logger.info("Moved analyzer.model to CUDA.")
    except Exception as e:
        logger.warning(f"Could not move analyzer.model to CUDA: {type(e).__name__}: {e}")


# =====================================================
# Signals (CPU-light)
# =====================================================
def _uppercase_ratio(text: str) -> float:
    uppercase = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    return float(uppercase) / max(letters, 1)


def _emoji_ratio(text: str) -> float:
    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    return float(len(emojis)) / max(len(text.split()), 1)


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(1 for c in text if (not c.isalnum() and not c.isspace()))
    return float(punct) / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for c in text if c.isdigit())
    return float(digits) / max(len(text), 1)


def _repeat_punct_ratio(text: str, pattern: str, denom: int) -> float:
    if not text:
        return 0.0
    repeats = len(re.findall(pattern, text))
    return float(repeats) / max(denom, 1)


def _quote_ratio(text: str) -> float:
    if not text:
        return 0.0
    quotes = text.count('"') + text.count("“") + text.count("”") + text.count("'")
    return float(quotes) / max(len(text), 1)


def _signals(
    text: str,
    intensifiers: set,
    normalize_signals_by: str = "chars",
    extra_signals: bool = True,
) -> np.ndarray:
    tokens = re.findall(r"\w+", (text or "").lower())
    num_tokens = max(len(tokens), 1)
    num_chars = max(len(text), 1)

    denom = num_chars if normalize_signals_by == "chars" else num_tokens

    exclam = (text or "").count("!")
    question = (text or "").count("?")

    exclam_ratio = exclam / max(denom, 1)
    question_ratio = question / max(denom, 1)
    upper_ratio = _uppercase_ratio(text or "")
    emo_ratio = _emoji_ratio(text or "")
    intens_ratio = float(sum(1 for t in tokens if t in intensifiers)) / max(len(tokens), 1)

    base = np.array(
        [exclam_ratio, question_ratio, upper_ratio, emo_ratio, intens_ratio],
        dtype=np.float32,
    )

    if not extra_signals:
        return base

    punct = _punct_ratio(text or "")
    digits = _digit_ratio(text or "")
    rep_ex = _repeat_punct_ratio(text or "", r"!{2,}", denom)
    rep_q = _repeat_punct_ratio(text or "", r"\?{2,}", denom)
    elip = _repeat_punct_ratio(text or "", r"\.{3,}", denom)
    quot = _quote_ratio(text or "")

    extra = np.array(
        [
            float(num_chars),
            float(num_tokens),
            punct,
            digits,
            rep_ex,
            rep_q,
            elip,
            quot,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra]).astype(np.float32)


def signal_feature_names(extra_signals: bool = True) -> List[str]:
    names = [
        "sig_exclam_ratio",
        "sig_question_ratio",
        "sig_uppercase_ratio",
        "sig_emoji_ratio",
        "sig_intensifier_ratio",
    ]
    if not extra_signals:
        return names
    names += [
        "sig_len_chars",
        "sig_len_tokens",
        "sig_punct_ratio",
        "sig_digit_ratio",
        "sig_repeat_exclam_ratio",
        "sig_repeat_question_ratio",
        "sig_elipsis_ratio",
        "sig_quote_ratio",
    ]
    return names


def _safe(x: np.ndarray, safe_numeric: bool) -> np.ndarray:
    if not safe_numeric:
        return x
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =====================================================
# Core extraction (batch)
# =====================================================
def _sorted_proba_keys(probas: Dict[str, float]) -> List[str]:
    return sorted(probas.keys())


def _probas_to_vec(probas: Dict[str, float], labels: List[str]) -> np.ndarray:
    return np.array([float(probas.get(k, 0.0)) for k in labels], dtype=np.float32)


def _predict_many(analyzer: Any, texts: List[str], batch_size: int, logger: logging.Logger) -> List[Any]:
    """
    Prefer predict_many if available; else try analyzer.predict(list); else loop.

    Note:
    - The real speedup comes from having the underlying HF model on CUDA
      and using batch inference.
    """
    bs = max(int(batch_size), 1)

    # 1) predict_many
    if hasattr(analyzer, "predict_many"):
        try:
            return analyzer.predict_many(texts, batch_size=bs)  # type: ignore[misc]
        except Exception as e:
            logger.info(f"predict_many not usable: {type(e).__name__}: {e}")

    # 2) analyzer.predict(list)
    try:
        out = analyzer.predict(texts)  # type: ignore[misc]
        if isinstance(out, list):
            return out
    except Exception:
        pass

    # 3) fallback loop
    preds: List[Any] = []
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        for t in chunk:
            preds.append(analyzer.predict(t))
    return preds


def extract_emotion_features(
    df: pd.DataFrame,
    cfg: ExtractConfig,
    logger: logging.Logger,
    text_col: str = "text_xlmr",
) -> Dict[str, Any]:
    """
    Extract emotion + sentiment probabilities and signals for df rows.

    Returns a dict with:
      - emo_labels, sent_labels
      - emo_mat, sent_mat, sig_mat (np.float32)
      - device, batch_size, signal_names
    """
    if text_col not in df.columns:
        raise ValueError(f"Missing required column '{text_col}'. Available: {list(df.columns)}")

    device = _resolve_device(cfg.device, logger)

    intensifiers = cfg.intensifiers or DEFAULT_INTENSIFIERS

    logger.info(f"Loading pysentimiento analyzers: lang={cfg.lang}")
    emo_an = create_analyzer(task="emotion", lang=cfg.lang)
    sent_an = create_analyzer(task="sentiment", lang=cfg.lang)

    # Try to move models to CUDA (best-effort)
    _try_move_analyzer_to_device(emo_an, device, logger)
    _try_move_analyzer_to_device(sent_an, device, logger)

    texts = df[text_col].fillna("").astype(str).tolist()
    if cfg.use_preprocess_tweet:
        texts = [preprocess_tweet(t, lang=cfg.lang) for t in texts]

    logger.info(f"Extracting emotion/sentiment: N={len(texts)} | batch_size={cfg.batch_size} | device={device}")

    # Inference mode (avoid autograd overhead)
    if torch is not None and hasattr(torch, "inference_mode"):
        ctx = torch.inference_mode()  # type: ignore[attr-defined]
    elif torch is not None:
        ctx = torch.no_grad()
    else:
        ctx = None

    if ctx is not None:
        with ctx:
            emo_preds = _predict_many(emo_an, texts, cfg.batch_size, logger)
            sent_preds = _predict_many(sent_an, texts, cfg.batch_size, logger)
    else:
        emo_preds = _predict_many(emo_an, texts, cfg.batch_size, logger)
        sent_preds = _predict_many(sent_an, texts, cfg.batch_size, logger)

    if not emo_preds or not sent_preds:
        raise RuntimeError("No predictions returned by analyzers.")

    emo_labels = _sorted_proba_keys(emo_preds[0].probas)
    sent_labels = _sorted_proba_keys(sent_preds[0].probas)

    emo_mat = np.vstack([_probas_to_vec(p.probas, emo_labels) for p in emo_preds]).astype(np.float32)
    sent_mat = np.vstack([_probas_to_vec(p.probas, sent_labels) for p in sent_preds]).astype(np.float32)

    sig_mat = np.vstack(
        [
            _signals(
                t,
                intensifiers=intensifiers,
                normalize_signals_by=cfg.normalize_signals_by,
                extra_signals=cfg.extra_signals,
            )
            for t in texts
        ]
    ).astype(np.float32)

    emo_mat = _safe(emo_mat, cfg.safe_numeric)
    sent_mat = _safe(sent_mat, cfg.safe_numeric)
    sig_mat = _safe(sig_mat, cfg.safe_numeric)

    return {
        "emo_labels": emo_labels,
        "sent_labels": sent_labels,
        "emo_mat": emo_mat,
        "sent_mat": sent_mat,
        "sig_mat": sig_mat,
        "signal_names": signal_feature_names(cfg.extra_signals),
        "device": device,
        "batch_size": cfg.batch_size,
        "normalize_signals_by": cfg.normalize_signals_by,
        "use_preprocess_tweet": cfg.use_preprocess_tweet,
    }


# =====================================================
# PKL split helpers (same pattern)
# =====================================================
def _resolve_split_file(input_dir: Path, split: str) -> Path:
    candidates = {
        "train": ["train.pkl"],
        "val": ["val.pkl", "development.pkl", "dev.pkl", "valid.pkl", "validation.pkl"],
        "test": ["test.pkl"],
    }
    if split not in candidates:
        raise ValueError(f"Unsupported split: {split}")

    for fname in candidates[split]:
        p = input_dir / fname
        if p.exists():
            return p

    return input_dir / candidates[split][0]


def _save_features_pkl(
    original_df: pd.DataFrame,
    payload: Dict[str, Any],
    output_pkl: Path,
    logger: logging.Logger,
) -> Path:
    """
    Save a new PKL with emotion features.
    """
    emo_mat: np.ndarray = payload["emo_mat"]
    sent_mat: np.ndarray = payload["sent_mat"]
    sig_mat: np.ndarray = payload["sig_mat"]

    out_df = pd.DataFrame()

    if "Id" in original_df.columns:
        out_df["Id"] = original_df["Id"].values

    if "label" in original_df.columns:
        out_df["label"] = original_df["label"].values
    elif "Category" in original_df.columns:
        out_df["label"] = original_df["Category"].values

    # Store lists for pickle friendliness
    out_df["emo_probs"] = [emo_mat[i].tolist() for i in range(emo_mat.shape[0])]
    out_df["sent_probs"] = [sent_mat[i].tolist() for i in range(sent_mat.shape[0])]
    out_df["signals"] = [sig_mat[i].tolist() for i in range(sig_mat.shape[0])]

    # Metadata columns (stable per file)
    out_df["emo_labels"] = [payload["emo_labels"]] * len(out_df)
    out_df["sent_labels"] = [payload["sent_labels"]] * len(out_df)
    out_df["signal_names"] = [payload["signal_names"]] * len(out_df)

    out_df["device"] = payload["device"]
    out_df["batch_size"] = payload["batch_size"]
    out_df["normalize_signals_by"] = payload["normalize_signals_by"]
    out_df["use_preprocess_tweet"] = payload["use_preprocess_tweet"]

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_pkl)
    logger.info(f"Features PKL created: {output_pkl.resolve()}")
    return output_pkl


# =====================================================
# Public API: extract for train/val/test
# =====================================================
def extract_emotion_features_for_splits(
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    text_col: str = "text_xlmr",
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,  # kept for interface parity; not used here
    use_preprocess_tweet: bool = False,
    normalize_signals_by: str = "chars",
    extra_signals: bool = True,
) -> None:
    """
    Extract and save emotion features for train/val/test splits.

    Reads:
      input_dir/train.pkl
      input_dir/val.pkl (or development.pkl)
      input_dir/test.pkl

    Writes:
      output_dir/train_features.pkl
      output_dir/val_features.pkl
      output_dir/test_features.pkl
    """
    logger = get_logger(log_dir=log_dir)

    cfg = ExtractConfig(
        lang="es",
        batch_size=batch_size,
        device=device,
        use_preprocess_tweet=use_preprocess_tweet,
        normalize_signals_by=normalize_signals_by,
        extra_signals=extra_signals,
        safe_numeric=True,
        intensifiers=None,
    )

    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        logger.info("Starting emotion feature extraction")
        logger.info(f"Input dir: {input_dir.resolve()}")
        logger.info(f"Output dir: {output_dir.resolve()}")
        logger.info(f"Text col: {text_col}")
        logger.info(f"Batch size: {cfg.batch_size}")
        logger.info(f"Device (requested): {device}")

        train_pkl = _resolve_split_file(input_dir, "train")
        val_pkl = _resolve_split_file(input_dir, "val")
        test_pkl = _resolve_split_file(input_dir, "test")

        for p in [train_pkl, val_pkl, test_pkl]:
            if not p.exists():
                raise FileNotFoundError(f"Missing split file: {p}")

        # Train
        logger.info(f"Loading split: train ({train_pkl.name})")
        df_train = pd.read_pickle(train_pkl)
        payload_train = extract_emotion_features(df_train, cfg=cfg, logger=logger, text_col=text_col)
        _save_features_pkl(df_train, payload_train, output_dir / "train_features.pkl", logger)

        # Val
        logger.info(f"Loading split: val ({val_pkl.name})")
        df_val = pd.read_pickle(val_pkl)
        payload_val = extract_emotion_features(df_val, cfg=cfg, logger=logger, text_col=text_col)
        _save_features_pkl(df_val, payload_val, output_dir / "val_features.pkl", logger)

        # Test
        logger.info(f"Loading split: test ({test_pkl.name})")
        df_test = pd.read_pickle(test_pkl)
        payload_test = extract_emotion_features(df_test, cfg=cfg, logger=logger, text_col=text_col)
        _save_features_pkl(df_test, payload_test, output_dir / "test_features.pkl", logger)

        logger.info("Emotion feature extraction completed")

    finally:
        _flush_and_close_logger(logger)
