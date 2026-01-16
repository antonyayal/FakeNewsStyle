# src/features/feature_merger.py
# -*- coding: utf-8 -*-
"""
Feature Merger (sem ⊕ emo ⊕ sty ⊕ ctx)

Goal
----
Merge per-split feature PKLs produced by:
- semantic_extractor  -> pandas.DataFrame with "Id" + "sem_emb" (list[float])
- emotion_extractor   -> can be:
    A) pandas.DataFrame with lists: emo_probs, sent_probs, signals
    B) dict payload with: X (np.ndarray), feature_names (list[str]), ids (list)
- style_extractor     -> dict payload with: X, feature_names, ids
- context_extractor   -> dict payload with: X, feature_names, ids

Key design decisions
--------------------
1) Alignment by Id (never by row order).
2) Missing policy default: fill_zero (+ optional presence flags).
3) Stable feature_names with prefixes: sem_, emo_, sty_, ctx_.
4) Output is a single dict payload per split:
   {
     "ids": [...],
     "X": np.ndarray (N, D_total),
     "feature_names": [...],
     "blocks": {"sem": (a,b), "emo": (b,c), "sty": (c,d), "ctx": (d,e), "flags": (e,f)},
     "labels": optional list,
     "meta": {...}
   }

Usage (example)
---------------
from pathlib import Path
from src.features.feature_merger import merge_features_for_splits

merge_features_for_splits(
    semantic_dir=Path("data/features/semantic/FakeNewsCorpusSpanish"),
    emotion_dir=Path("data/features/emotion/FakeNewsCorpusSpanish"),
    style_dir=Path("data/features/style/FakeNewsCorpusSpanish"),
    context_dir=Path("data/features/context/FakeNewsCorpusSpanish"),
    output_dir=Path("data/features/merged/FakeNewsCorpusSpanish"),
    log_dir=Path("logs/features/merge"),
    missing_policy="fill_zero",  # "fill_zero" | "drop" | "error"
    add_presence_flags=True,
)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =====================================================
# Logging (same pattern as semantic/emotion)
# =====================================================
def get_logger(log_dir: Path, name: str = "feature_merger") -> logging.Logger:
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
class MergerConfig:
    # Filename expectations (your current outputs)
    semantic_files: Dict[str, str] = None  # type: ignore[assignment]
    emotion_files: Dict[str, str] = None  # type: ignore[assignment]
    style_files: Dict[str, str] = None  # type: ignore[assignment]
    context_files: Dict[str, str] = None  # type: ignore[assignment]

    # Merge behavior
    missing_policy: str = "fill_zero"  # "fill_zero" | "drop" | "error"
    add_presence_flags: bool = True

    # Column conventions for DF-based PKLs
    id_col: str = "Id"
    label_col: str = "label"

    # Prefixes (avoid collisions)
    sem_prefix: str = "sem_"
    emo_prefix: str = "emo_"
    sty_prefix: str = "sty_"
    ctx_prefix: str = "ctx_"

    def __post_init__(self) -> None:
        if self.semantic_files is None:
            self.semantic_files = {"train": "train_features.pkl", "val": "val_features.pkl", "test": "test_features.pkl"}
        if self.emotion_files is None:
            # support both conventions: emotion extractor may produce train_features.pkl OR train_emotion.pkl
            self.emotion_files = {
                "train": "train_features.pkl",
                "val": "val_features.pkl",
                "test": "test_features.pkl",
            }
        if self.style_files is None:
            self.style_files = {"train": "train_style.pkl", "val": "val_style.pkl", "test": "test_style.pkl"}
        if self.context_files is None:
            self.context_files = {"train": "train_context.pkl", "val": "val_context.pkl", "test": "test_context.pkl"}


# =====================================================
# IO helpers
# =====================================================
def _read_pkl_any(path: Path) -> Any:
    return pd.read_pickle(path)


def _ensure_2d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x.astype(np.float32, copy=False)


def _to_id_map(ids: List[Any], X: np.ndarray) -> Dict[Any, np.ndarray]:
    X = _ensure_2d_float32(X)
    return {ids[i]: X[i] for i in range(len(ids))}


def _prefix_names(names: List[str], prefix: str) -> List[str]:
    return [f"{prefix}{n}" for n in names]


# =====================================================
# Loaders for each feature type
# =====================================================
def _load_semantic_df(path: Path, cfg: MergerConfig, logger: logging.Logger) -> Tuple[List[Any], np.ndarray, List[str], Optional[List[Any]]]:
    """
    Semantic PKL: DataFrame with Id, label, sem_emb(list[float]).
    Returns: ids, X_sem, feature_names, labels(optional)
    """
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Semantic PKL must be a DataFrame, got {type(df)}: {path}")

    for col in [cfg.id_col, "sem_emb"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' in semantic DF. Columns={list(df.columns)} | file={path}")

    ids = df[cfg.id_col].tolist()
    sem_list = df["sem_emb"].tolist()
    X = np.vstack([np.asarray(v, dtype=np.float32) for v in sem_list]).astype(np.float32)

    d = X.shape[1]
    names = [f"sem_emb_{i}" for i in range(d)]
    names = _prefix_names(names, cfg.sem_prefix)

    labels = None
    if cfg.label_col in df.columns:
        labels = df[cfg.label_col].tolist()

    logger.info(f"Loaded semantic: {path.name} | N={len(ids)} | D={X.shape[1]}")
    return ids, X, names, labels


def _load_emotion(path: Path, cfg: MergerConfig, logger: logging.Logger) -> Tuple[List[Any], np.ndarray, List[str], Optional[List[Any]]]:
    """
    Emotion PKL can be:
    A) DataFrame with Id,label, emo_probs(list), sent_probs(list), signals(list),
       plus label lists (emo_labels/sent_labels/signal_names)
    B) dict with X, feature_names, ids (and optionally labels in meta)
    """
    obj = _read_pkl_any(path)

    # Case B: dict payload
    if isinstance(obj, dict) and "X" in obj and "ids" in obj and "feature_names" in obj:
        ids = list(obj["ids"])
        X = _ensure_2d_float32(obj["X"])
        names = [str(n) for n in obj["feature_names"]]
        # Ensure prefix (if not already)
        if not (len(names) > 0 and names[0].startswith(cfg.emo_prefix)):
            names = _prefix_names(names, cfg.emo_prefix)

        labels = None
        meta = obj.get("meta", {}) if isinstance(obj.get("meta", {}), dict) else {}
        # no strict label support here; you can add it if you store labels later
        logger.info(f"Loaded emotion(dict): {path.name} | N={len(ids)} | D={X.shape[1]}")
        return ids, X, names, labels

    # Case A: DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj
        if cfg.id_col not in df.columns:
            raise ValueError(f"Missing '{cfg.id_col}' in emotion DF. Columns={list(df.columns)} | file={path}")

        ids = df[cfg.id_col].tolist()

        # Build X = emo_probs + sent_probs + signals
        for col in ["emo_probs", "sent_probs", "signals"]:
            if col not in df.columns:
                raise ValueError(f"Missing '{col}' in emotion DF. Columns={list(df.columns)} | file={path}")

        emo = np.vstack([np.asarray(v, dtype=np.float32) for v in df["emo_probs"].tolist()]).astype(np.float32)
        sent = np.vstack([np.asarray(v, dtype=np.float32) for v in df["sent_probs"].tolist()]).astype(np.float32)
        sig = np.vstack([np.asarray(v, dtype=np.float32) for v in df["signals"].tolist()]).astype(np.float32)

        X = np.hstack([emo, sent, sig]).astype(np.float32)

        # Names: prefer stored label lists if present (first row repeats them)
        emo_labels = None
        sent_labels = None
        sig_names = None
        if "emo_labels" in df.columns and len(df) > 0:
            emo_labels = list(df["emo_labels"].iloc[0])
        if "sent_labels" in df.columns and len(df) > 0:
            sent_labels = list(df["sent_labels"].iloc[0])
        if "signal_names" in df.columns and len(df) > 0:
            sig_names = list(df["signal_names"].iloc[0])

        if emo_labels and sent_labels and sig_names:
            names = (
                [f"emo_{x}" for x in emo_labels]
                + [f"sent_{x}" for x in sent_labels]
                + [f"{x}" for x in sig_names]
            )
        else:
            # fallback: generic indexing
            names = [f"emo_feat_{i}" for i in range(X.shape[1])]

        names = _prefix_names(names, cfg.emo_prefix)

        labels = None
        if cfg.label_col in df.columns:
            labels = df[cfg.label_col].tolist()

        logger.info(f"Loaded emotion(DF): {path.name} | N={len(ids)} | D={X.shape[1]}")
        return ids, X, names, labels

    raise ValueError(f"Unsupported emotion PKL type: {type(obj)} | file={path}")


def _load_dict_features(path: Path, prefix: str, logger: logging.Logger) -> Tuple[List[Any], np.ndarray, List[str]]:
    obj = _read_pkl_any(path)
    if not (isinstance(obj, dict) and "X" in obj and "ids" in obj and "feature_names" in obj):
        raise ValueError(f"Expected dict payload with X/ids/feature_names, got {type(obj)} | file={path}")

    ids = list(obj["ids"])
    X = _ensure_2d_float32(obj["X"])
    names = [str(n) for n in obj["feature_names"]]

    # Apply prefix if not already
    if not (len(names) > 0 and names[0].startswith(prefix)):
        names = _prefix_names(names, prefix)

    logger.info(f"Loaded dict features: {path.name} | N={len(ids)} | D={X.shape[1]}")
    return ids, X, names


# =====================================================
# Merge core
# =====================================================
def _merge_by_ids(
    base_ids: List[Any],
    blocks: Dict[str, Tuple[Dict[Any, np.ndarray], int, List[str]]],
    missing_policy: str,
    add_presence_flags: bool,
    logger: logging.Logger,
) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int]], np.ndarray]:
    """
    blocks: name -> (id_to_vec, dim, feature_names)
    Returns:
      X_merged, merged_feature_names, block_slices, flags_mat
    """
    missing_policy = (missing_policy or "fill_zero").lower().strip()
    if missing_policy not in {"fill_zero", "drop", "error"}:
        raise ValueError("missing_policy must be: fill_zero | drop | error")

    # Determine final row ids if drop policy
    kept_ids: List[Any] = []
    if missing_policy == "drop":
        for _id in base_ids:
            ok = all((_id in blocks[b][0]) for b in blocks.keys())
            if ok:
                kept_ids.append(_id)
        ids_used = kept_ids
        logger.info(f"Missing policy=drop | kept={len(ids_used)}/{len(base_ids)}")
    else:
        ids_used = list(base_ids)

    # Total dim
    total_dim = sum(dim for (_, dim, _) in blocks.values())
    flag_dim = len(blocks) if add_presence_flags else 0
    X = np.zeros((len(ids_used), total_dim + flag_dim), dtype=np.float32)
    flags = np.zeros((len(ids_used), len(blocks)), dtype=np.float32) if add_presence_flags else np.zeros((len(ids_used), 0), dtype=np.float32)

    # Feature names + slices
    merged_names: List[str] = []
    slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    block_names = list(blocks.keys())

    for bname in block_names:
        _, dim, fnames = blocks[bname]
        slices[bname] = (cursor, cursor + dim)
        merged_names.extend(fnames)
        cursor += dim

    if add_presence_flags:
        slices["flags"] = (cursor, cursor + flag_dim)
        merged_names.extend([f"has_{bname}" for bname in block_names])

    # Fill rows
    for i, _id in enumerate(ids_used):
        col_cursor = 0
        for bi, bname in enumerate(block_names):
            id_map, dim, _ = blocks[bname]
            v = id_map.get(_id, None)
            if v is None:
                if missing_policy == "error":
                    raise KeyError(f"Missing id={_id} in block '{bname}'")
                # fill_zero: keep zeros
                if add_presence_flags:
                    flags[i, bi] = 0.0
            else:
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                if v.shape[0] != dim:
                    raise ValueError(f"Dim mismatch for id={_id} block={bname}: expected {dim}, got {v.shape[0]}")
                X[i, col_cursor : col_cursor + dim] = v
                if add_presence_flags:
                    flags[i, bi] = 1.0
            col_cursor += dim

        if add_presence_flags:
            X[i, col_cursor : col_cursor + flag_dim] = flags[i]

    return X, merged_names, slices, flags


# =====================================================
# Public API
# =====================================================
def merge_features_for_splits(
    semantic_dir: Path,
    emotion_dir: Path,
    style_dir: Path,
    context_dir: Path,
    output_dir: Path,
    log_dir: Path,
    missing_policy: str = "fill_zero",
    add_presence_flags: bool = True,
    # optional: override filenames
    cfg: Optional[MergerConfig] = None,
    # optional: try alternative emotion filename if your run produced train_emotion.pkl etc.
    emotion_alt_files: Optional[Dict[str, str]] = None,
) -> None:
    """
    Merge features for train/val/test and write <split>_merged.pkl to output_dir.
    """
    logger = get_logger(log_dir=log_dir)

    cfg = cfg or MergerConfig()
    cfg.missing_policy = missing_policy
    cfg.add_presence_flags = add_presence_flags

    # If your emotion extractor produces train_emotion.pkl etc, you can pass this
    # e.g. {"train":"train_emotion.pkl","val":"val_emotion.pkl","test":"test_emotion.pkl"}
    if emotion_alt_files is None:
        emotion_alt_files = {}

    try:
        semantic_dir = Path(semantic_dir)
        emotion_dir = Path(emotion_dir)
        style_dir = Path(style_dir)
        context_dir = Path(context_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting feature merge")
        logger.info(f"semantic_dir={semantic_dir.resolve()}")
        logger.info(f"emotion_dir={emotion_dir.resolve()}")
        logger.info(f"style_dir={style_dir.resolve()}")
        logger.info(f"context_dir={context_dir.resolve()}")
        logger.info(f"output_dir={output_dir.resolve()}")
        logger.info(f"missing_policy={cfg.missing_policy} | add_presence_flags={cfg.add_presence_flags}")

        for split in ["train", "val", "test"]:
            sem_path = semantic_dir / cfg.semantic_files[split]

            # emotion: try primary, then optional alt
            emo_path = emotion_dir / cfg.emotion_files[split]
            if not emo_path.exists() and split in emotion_alt_files:
                emo_path = emotion_dir / emotion_alt_files[split]

            sty_path = style_dir / cfg.style_files[split]
            ctx_path = context_dir / cfg.context_files[split]

            for p in [sem_path, emo_path, sty_path, ctx_path]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing required feature file for split={split}: {p}")

            logger.info(f"--- Split: {split} ---")

            # Load blocks
            ids_sem, X_sem, names_sem, labels_sem = _load_semantic_df(sem_path, cfg, logger)
            ids_emo, X_emo, names_emo, labels_emo = _load_emotion(emo_path, cfg, logger)
            ids_sty, X_sty, names_sty = _load_dict_features(sty_path, cfg.sty_prefix, logger)
            ids_ctx, X_ctx, names_ctx = _load_dict_features(ctx_path, cfg.ctx_prefix, logger)

            # Build id->vec maps
            sem_map = _to_id_map(ids_sem, X_sem)
            emo_map = _to_id_map(ids_emo, X_emo)
            sty_map = _to_id_map(ids_sty, X_sty)
            ctx_map = _to_id_map(ids_ctx, X_ctx)

            blocks = {
                "sem": (sem_map, int(X_sem.shape[1]), names_sem),
                "emo": (emo_map, int(X_emo.shape[1]), names_emo),
                "sty": (sty_map, int(X_sty.shape[1]), names_sty),
                "ctx": (ctx_map, int(X_ctx.shape[1]), names_ctx),
            }

            # Merge by base ids from semantic (recommended)
            X, merged_names, block_slices, flags = _merge_by_ids(
                base_ids=ids_sem,
                blocks=blocks,
                missing_policy=cfg.missing_policy,
                add_presence_flags=cfg.add_presence_flags,
                logger=logger,
            )

            # Resolve labels: prefer semantic, else emotion, else None
            labels = labels_sem if labels_sem is not None else labels_emo

            # If drop policy, ids used might be shorter; rebuild ids_used from flags length
            # When missing_policy != drop, ids_used == ids_sem
            ids_used = ids_sem
            if cfg.missing_policy == "drop":
                # ids that have all blocks present -> flags row all ones
                if cfg.add_presence_flags and flags.size > 0:
                    mask = (flags.sum(axis=1) == flags.shape[1])
                    ids_used = [ids_sem[i] for i in range(len(ids_sem)) if bool(mask[i])]
                else:
                    # fallback: recompute by checking maps
                    ids_used = [i for i in ids_sem if all(i in blocks[b][0] for b in blocks.keys())]

                if labels is not None:
                    # align labels with ids_used
                    label_map = {ids_sem[i]: labels[i] for i in range(len(ids_sem))}
                    labels = [label_map[_id] for _id in ids_used]

            # Build payload
            payload: Dict[str, Any] = {
                "ids": list(ids_used),
                "X": X,
                "feature_names": merged_names,
                "blocks": block_slices,
                "num_samples": int(X.shape[0]),
                "feature_dim": int(X.shape[1]),
                "meta": {
                    "module": "FeatureMerger",
                    "split": split,
                    "missing_policy": cfg.missing_policy,
                    "add_presence_flags": cfg.add_presence_flags,
                    "sources": {
                        "semantic": str(sem_path.resolve()),
                        "emotion": str(emo_path.resolve()),
                        "style": str(sty_path.resolve()),
                        "context": str(ctx_path.resolve()),
                    },
                    "dims": {
                        "sem": int(X_sem.shape[1]),
                        "emo": int(X_emo.shape[1]),
                        "sty": int(X_sty.shape[1]),
                        "ctx": int(X_ctx.shape[1]),
                        "total": int(X.shape[1]),
                    },
                    "created_at": datetime.now().isoformat(),
                },
            }
            if labels is not None:
                payload["labels"] = list(labels)

            out_path = output_dir / f"{split}_merged.pkl"
            with open(out_path, "wb") as f:
                import pickle

                pickle.dump(payload, f)

            logger.info(f"Saved merged: {out_path.name} | N={payload['num_samples']} | D={payload['feature_dim']}")
            logger.info(f"Block slices: {block_slices}")

        logger.info("Feature merge completed")

    finally:
        _flush_and_close_logger(logger)
