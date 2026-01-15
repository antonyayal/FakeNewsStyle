"""
Semantic feature extractor (XLM-RoBERTa + optional Attention Pooling)

Goal
----
This module generates a dense semantic representation per sample and saves it
as a new PKL with an embedding column.

Why this file exists
--------------------
- XLM-R already uses self-attention internally.
- Here, "attention layer" refers to an *attention pooling* that learns how to
  weight token embeddings to create a single document vector.

Outputs
-------
For each input PKL (train/val/test), creates an output PKL with:
- Id (if exists)
- label (if exists)
- sem_emb : list[float] (document embedding)
- pooling : pooling type used
- model_name : HF model name used
- max_len : tokenizer max length used

Expected input columns
----------------------
- text_xlmr (string)  <-- produced by src/text/preprocess_text.py
Optional:
- Id, label

Usage (example)
---------------
from pathlib import Path
from src.features.semantic_extractor import extract_semantic_features_for_splits

extract_semantic_features_for_splits(
    input_dir=Path("data/processed_by_model/FakeNewsCorpusSpanish"),
    output_dir=Path("data/features/semantic/FakeNewsCorpusSpanish"),
    log_dir=Path("logs/features"),
    pooling="attention",  # "mean" | "cls" | "attention"
    max_len=256,
    batch_size=8,
    device="cuda"  # or "cpu"
)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# =====================================================
# Defaults
# =====================================================
MODEL_NAME_DEFAULT = "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish"


# =====================================================
# Logging
# =====================================================
def get_logger(log_dir: Path, name: str = "semantic_extractor") -> logging.Logger:
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
# Dataset
# =====================================================
class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str = "text_xlmr"):
        if text_col not in df.columns:
            raise ValueError(f"Missing required column '{text_col}'. Available: {list(df.columns)}")
        self.df = df.reset_index(drop=True)
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "idx": idx,
            "text": "" if pd.isna(row[self.text_col]) else str(row[self.text_col]),
        }


@dataclass
class ExtractConfig:
    model_name: str = MODEL_NAME_DEFAULT
    pooling: str = "mean"  # "mean" | "cls" | "attention"
    max_len: int = 256
    batch_size: int = 8
    num_workers: int = 0
    device: str = "cpu"  # "cuda" if available
    fp16: bool = False  # set True only if you know your GPU supports it


# =====================================================
# Attention Pooling (token-level -> document vector)
# =====================================================
class AttentionPooling(nn.Module):
    """
    Self-attentive pooling over token embeddings.

    Inputs
    ------
    last_hidden_state: [B, T, D]
    attention_mask:   [B, T] (1 for real tokens, 0 for padding)

    Output
    ------
    doc_emb: [B, D]
    """

    def __init__(self, hidden_size: int, attn_size: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_size, attn_size)
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # [B, T, A]
        scores = torch.tanh(self.proj(last_hidden_state))
        # [B, T, 1] -> [B, T]
        scores = self.v(scores).squeeze(-1)

        # Mask padding: set scores to very negative where mask == 0
        scores = scores.masked_fill(attention_mask == 0, -1e9)

        # [B, T]
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum: [B, D]
        doc_emb = torch.bmm(weights.unsqueeze(1), last_hidden_state).squeeze(1)
        return doc_emb


# =====================================================
# Pooling helpers
# =====================================================
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over non-padding tokens.
    last_hidden_state: [B, T, D]
    attention_mask:   [B, T]
    returns:          [B, D]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = torch.sum(last_hidden_state * mask, dim=1)             # [B, D]
    denom = torch.clamp(mask.sum(dim=1), min=1e-6)                  # [B, 1]
    return summed / denom


def cls_pooling(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Use first token embedding (<s>) as document embedding.
    returns: [B, D]
    """
    return last_hidden_state[:, 0, :]


# =====================================================
# Core extraction
# =====================================================
@torch.no_grad()
def extract_embeddings(
    df: pd.DataFrame,
    cfg: ExtractConfig,
    logger: logging.Logger,
    text_col: str = "text_xlmr",
) -> np.ndarray:
    """
    Extract a document embedding per row in df. Returns numpy array [N, D].
    """
    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if cfg.device == "cuda" and device.type != "cuda":
        logger.warning("Requested device=cuda but CUDA is not available. Falling back to CPU.")

    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    logger.info(f"Loading encoder model: {cfg.model_name}")
    model = AutoModel.from_pretrained(cfg.model_name)
    model.eval().to(device)

    hidden_size = model.config.hidden_size
    attn_pool: Optional[AttentionPooling] = None
    if cfg.pooling == "attention":
        logger.info("Using AttentionPooling (PyTorch layer).")
        attn_pool = AttentionPooling(hidden_size=hidden_size, attn_size=256).to(device)
        attn_pool.eval()  # NOTE: untrained unless you train end-to-end

    dataset = TextDataset(df=df, text_col=text_col)

    def collate_fn(batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        idxs = [b["idx"] for b in batch]
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_len,
            return_tensors="pt",
        )
        tok["idxs"] = torch.tensor(idxs, dtype=torch.long)
        return tok

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    logger.info(f"Extracting embeddings: N={len(df)} | batch_size={cfg.batch_size} | max_len={cfg.max_len} | pooling={cfg.pooling}")
    out = np.zeros((len(df), hidden_size), dtype=np.float32)

    use_amp = bool(cfg.fp16 and device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast  # type: ignore[attr-defined]

    for step, batch in enumerate(loader, start=1):
        idxs = batch.pop("idxs").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}

        # Encoder forward
        with (torch.cuda.amp.autocast() if use_amp else torch.no_grad()):
            outputs = model(**batch)
            h = outputs.last_hidden_state  # [B, T, D]
            mask = batch["attention_mask"]  # [B, T]

            if cfg.pooling == "mean":
                emb = mean_pooling(h, mask)
            elif cfg.pooling == "cls":
                emb = cls_pooling(h)
            elif cfg.pooling == "attention":
                assert attn_pool is not None
                emb = attn_pool(h, mask)
            else:
                raise ValueError(f"Unknown pooling: {cfg.pooling}. Use: mean | cls | attention")

        emb = emb.detach().cpu().numpy().astype(np.float32)
        out[idxs] = emb

        if step % 20 == 0:
            logger.info(f"Processed batches: {step}")

    logger.info("Embedding extraction completed.")
    return out


# =====================================================
# PKL split helpers
# =====================================================
def _resolve_split_file(input_dir: Path, split: str) -> Path:
    """
    Resolve split paths.
    Supports development.pkl as val if needed.
    """
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
    embeddings: np.ndarray,
    output_pkl: Path,
    cfg: ExtractConfig,
    logger: logging.Logger,
) -> Path:
    """
    Save a new PKL with semantic embeddings.
    """
    out_df = pd.DataFrame()

    # Keep Id if present
    if "Id" in original_df.columns:
        out_df["Id"] = original_df["Id"].values

    # Keep label if present
    if "label" in original_df.columns:
        out_df["label"] = original_df["label"].values
    elif "Category" in original_df.columns:
        out_df["label"] = original_df["Category"].values

    # Store embedding as a python list so it can be pickled easily
    out_df["sem_emb"] = [embeddings[i].tolist() for i in range(embeddings.shape[0])]
    out_df["pooling"] = cfg.pooling
    out_df["model_name"] = cfg.model_name
    out_df["max_len"] = cfg.max_len

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_pickle(output_pkl)
    logger.info(f"Features PKL created: {output_pkl.resolve()}")

    return output_pkl


# =====================================================
# Public API: extract for train/val/test
# =====================================================
def extract_semantic_features_for_splits(
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    pooling: str = "mean",
    model_name: str = MODEL_NAME_DEFAULT,
    max_len: int = 256,
    batch_size: int = 8,
    device: str = "cpu",
    num_workers: int = 0,
) -> None:
    """
    Extract and save semantic embeddings for train/val/test splits.

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
        model_name=model_name,
        pooling=pooling,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        fp16=False,
    )

    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        logger.info("Starting semantic feature extraction")
        logger.info(f"Input dir: {input_dir.resolve()}")
        logger.info(f"Output dir: {output_dir.resolve()}")
        logger.info(f"Model: {cfg.model_name}")
        logger.info(f"Pooling: {cfg.pooling}")

        train_pkl = _resolve_split_file(input_dir, "train")
        val_pkl = _resolve_split_file(input_dir, "val")
        test_pkl = _resolve_split_file(input_dir, "test")

        for p in [train_pkl, val_pkl, test_pkl]:
            if not p.exists():
                raise FileNotFoundError(f"Missing split file: {p}")

        # Train
        logger.info(f"Loading split: train ({train_pkl.name})")
        df_train = pd.read_pickle(train_pkl)
        emb_train = extract_embeddings(df_train, cfg=cfg, logger=logger)
        _save_features_pkl(df_train, emb_train, output_dir / "train_features.pkl", cfg, logger)

        # Val
        logger.info(f"Loading split: val ({val_pkl.name})")
        df_val = pd.read_pickle(val_pkl)
        emb_val = extract_embeddings(df_val, cfg=cfg, logger=logger)
        _save_features_pkl(df_val, emb_val, output_dir / "val_features.pkl", cfg, logger)

        # Test (label may or may not exist)
        logger.info(f"Loading split: test ({test_pkl.name})")
        df_test = pd.read_pickle(test_pkl)
        emb_test = extract_embeddings(df_test, cfg=cfg, logger=logger)
        _save_features_pkl(df_test, emb_test, output_dir / "test_features.pkl", cfg, logger)

        logger.info("Semantic feature extraction completed")

    finally:
        _flush_and_close_logger(logger)
